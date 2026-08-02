const MODEL_URL = "model/best.onnx";
const INPUT_SIZE = 640;
const CONF_THRES = 0.25;
const IOU_THRES = 0.45;
const MASK_COLOR = [255, 60, 60];

const fileInput = document.getElementById("file-input");
const fileLabel = document.getElementById("file-input-label");
const statusEl = document.getElementById("status");
const canvasWrap = document.getElementById("canvas-wrap");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const resultInfo = document.getElementById("result-info");

let session = null;

function setStatus(text, isError) {
	statusEl.textContent = text;
	statusEl.className = isError ? "error" : "";
}

async function loadModel() {
	// Chi dung WASM backend, ep numThreads=1: GitHub Pages khong gui header COOP/COEP nen
	// SharedArrayBuffer (can cho multi-thread) khong dung duoc - single thread luon chay on dinh.
	ort.env.wasm.wasmPaths = "ort/";
	ort.env.wasm.numThreads = 1;

	fileLabel.classList.add("disabled");
	setStatus("Đang tải model (~13MB, chỉ tải 1 lần, trình duyệt sẽ cache lại)...");
	try {
		session = await ort.InferenceSession.create(MODEL_URL, {
			executionProviders: ["wasm"],
		});
		fileLabel.classList.remove("disabled");
		setStatus("Sẵn sàng. Chọn 1 ảnh để bắt đầu.");
	} catch (err) {
		setStatus(
			"Không tải được model tại " + MODEL_URL + " - kiểm tra file đã đặt đúng chỗ chưa. Lỗi: " + err.message,
			true,
		);
	}
}

// Letterbox: resize giữ nguyên tỷ lệ khung hình vào ô 640x640, phần dư pad màu xám (114,114,114)
// giống hệt tiền xử lý mặc định của YOLOv8 - bắt buộc làm đúng bước này, sai là toạ độ box/mask lệch hết.
function letterbox(img, size) {
	const scale = Math.min(size / img.width, size / img.height);
	const newW = Math.round(img.width * scale);
	const newH = Math.round(img.height * scale);
	const padX = Math.floor((size - newW) / 2);
	const padY = Math.floor((size - newH) / 2);

	const off = document.createElement("canvas");
	off.width = size;
	off.height = size;
	const offCtx = off.getContext("2d");
	offCtx.fillStyle = "rgb(114,114,114)";
	offCtx.fillRect(0, 0, size, size);
	offCtx.drawImage(img, 0, 0, img.width, img.height, padX, padY, newW, newH);

	return { canvas: off, scale, padX, padY };
}

function imageDataToTensor(imgData, size) {
	const { data } = imgData;
	const floatData = new Float32Array(3 * size * size);
	const area = size * size;
	for (let i = 0; i < area; i++) {
		floatData[i] = data[i * 4] / 255; // R plane
		floatData[area + i] = data[i * 4 + 1] / 255; // G plane
		floatData[area * 2 + i] = data[i * 4 + 2] / 255; // B plane
	}
	return new ort.Tensor("float32", floatData, [1, 3, size, size]);
}

function iou(a, b) {
	const x1 = Math.max(a.x1, b.x1);
	const y1 = Math.max(a.y1, b.y1);
	const x2 = Math.min(a.x2, b.x2);
	const y2 = Math.min(a.y2, b.y2);
	const interW = Math.max(0, x2 - x1);
	const interH = Math.max(0, y2 - y1);
	const inter = interW * interH;
	const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
	const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
	return inter / (areaA + areaB - inter + 1e-6);
}

function nms(dets, iouThres) {
	dets.sort((a, b) => b.score - a.score);
	const keep = [];
	const used = new Array(dets.length).fill(false);
	for (let i = 0; i < dets.length; i++) {
		if (used[i]) continue;
		keep.push(dets[i]);
		for (let j = i + 1; j < dets.length; j++) {
			if (used[j]) continue;
			if (iou(dets[i], dets[j]) > iouThres) used[j] = true;
		}
	}
	return keep;
}

function sigmoid(x) {
	return 1 / (1 + Math.exp(-x));
}

// output0: [1, 4 + numClasses + numMaskCoef, numAnchors] (chuan xuat cua YOLOv8-seg ONNX, single class nen
// numClasses=1). proto: [1, 32, protoH, protoW] (mask prototypes, do phan giai 1/4 so voi input 640 -> 160).
function decode(output0, proto, numAnchors, numMaskCoef, protoH, protoW) {
	const dets = [];
	for (let a = 0; a < numAnchors; a++) {
		const score = output0[(4) * numAnchors + a]; // class 0 score row
		if (score < CONF_THRES) continue;
		const cx = output0[0 * numAnchors + a];
		const cy = output0[1 * numAnchors + a];
		const w = output0[2 * numAnchors + a];
		const h = output0[3 * numAnchors + a];
		const maskCoef = new Float32Array(numMaskCoef);
		for (let m = 0; m < numMaskCoef; m++) {
			maskCoef[m] = output0[(5 + m) * numAnchors + a];
		}
		dets.push({
			x1: cx - w / 2,
			y1: cy - h / 2,
			x2: cx + w / 2,
			y2: cy + h / 2,
			score,
			maskCoef,
		});
	}
	return nms(dets, IOU_THRES);
}

// Ghep mask cho 1 detection: mask = sigmoid(maskCoef . proto), crop theo box, resize ve kich thuoc box
// tren canvas hien thi (khong gian anh goc, da bo padding letterbox).
function buildMaskCanvas(det, proto, protoH, protoW, scale, padX, padY, dispW, dispH) {
	const numMaskCoef = det.maskCoef.length;
	const maskFull = new Float32Array(protoH * protoW);
	for (let p = 0; p < protoH * protoW; p++) {
		let sum = 0;
		for (let m = 0; m < numMaskCoef; m++) {
			sum += det.maskCoef[m] * proto[m * protoH * protoW + p];
		}
		maskFull[p] = sigmoid(sum);
	}

	// Box dang o toa do 640-space (co padding). Doi ve toa do goc anh (bo padding + chia scale).
	const ox1 = Math.max(0, (det.x1 - padX) / scale);
	const oy1 = Math.max(0, (det.y1 - padY) / scale);
	const ox2 = Math.min(dispW, (det.x2 - padX) / scale);
	const oy2 = Math.min(dispH, (det.y2 - padY) / scale);
	if (ox2 <= ox1 || oy2 <= oy1) return null;

	// proto dang o 160-space tuong ung 640-space (chua tru padding) - quy doi box ve proto-space de crop.
	const protoScaleX = protoW / INPUT_SIZE;
	const protoScaleY = protoH / INPUT_SIZE;
	const px1 = Math.max(0, Math.floor(det.x1 * protoScaleX));
	const py1 = Math.max(0, Math.floor(det.y1 * protoScaleY));
	const px2 = Math.min(protoW, Math.ceil(det.x2 * protoScaleX));
	const py2 = Math.min(protoH, Math.ceil(det.y2 * protoScaleY));
	const cropW = Math.max(1, px2 - px1);
	const cropH = Math.max(1, py2 - py1);

	const small = document.createElement("canvas");
	small.width = cropW;
	small.height = cropH;
	const sctx = small.getContext("2d");
	const imgData = sctx.createImageData(cropW, cropH);
	for (let y = 0; y < cropH; y++) {
		for (let x = 0; x < cropW; x++) {
			const srcIdx = (py1 + y) * protoW + (px1 + x);
			const v = maskFull[srcIdx] > 0.5 ? 255 : 0;
			const dstIdx = (y * cropW + x) * 4;
			imgData.data[dstIdx] = MASK_COLOR[0];
			imgData.data[dstIdx + 1] = MASK_COLOR[1];
			imgData.data[dstIdx + 2] = MASK_COLOR[2];
			imgData.data[dstIdx + 3] = v > 0 ? 140 : 0;
		}
	}
	sctx.putImageData(imgData, 0, 0);

	return { canvas: small, ox1, oy1, ox2, oy2 };
}

async function runInference(img) {
	setStatus("Đang xử lý ảnh...");
	resultInfo.textContent = "";

	const { canvas: lbCanvas, scale, padX, padY } = letterbox(img, INPUT_SIZE);
	const lbCtx = lbCanvas.getContext("2d");
	const imgData = lbCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
	const inputTensor = imageDataToTensor(imgData, INPUT_SIZE);

	const feeds = {};
	feeds[session.inputNames[0]] = inputTensor;
	const t0 = performance.now();
	const results = await session.run(feeds);
	const elapsedMs = Math.round(performance.now() - t0);

	// Khong gia dinh thu tu output theo ten/index co dinh - chon theo so chieu de tranh sai neu ONNX
	// export ra thu tu khac: du doan la tensor 3D [1,C,A], proto mask la tensor 4D [1,nm,protoH,protoW].
	const outA = results[session.outputNames[0]];
	const outB = results[session.outputNames[1]];
	const out0 = outA.dims.length === 3 ? outA : outB;
	const out1 = outA.dims.length === 4 ? outA : outB;
	if (out0.dims.length !== 3 || out1.dims.length !== 4) {
		throw new Error("Output shape không đúng như kỳ vọng (cần 1 tensor 3D + 1 tensor 4D): " + outA.dims + " / " + outB.dims);
	}
	const [, C, numAnchors] = out0.dims;
	const [, numMaskCoef, protoH, protoW] = out1.dims;
	if (C !== 4 + 1 + numMaskCoef) {
		throw new Error(`Số kênh output (${C}) không khớp công thức 4+nc+nm cho model 1 class (kỳ vọng ${4 + 1 + numMaskCoef}). Model có thể không phải single-class như giả định.`);
	}

	const dets = decode(out0.data, out1.data, numAnchors, numMaskCoef, protoH, protoW);

	// Ve anh goc len canvas hien thi
	canvas.width = img.width;
	canvas.height = img.height;
	ctx.drawImage(img, 0, 0, img.width, img.height);

	for (const det of dets) {
		const built = buildMaskCanvas(det, out1.data, protoH, protoW, scale, padX, padY, img.width, img.height);
		if (!built) continue;
		ctx.drawImage(
			built.canvas,
			0, 0, built.canvas.width, built.canvas.height,
			built.ox1, built.oy1, built.ox2 - built.ox1, built.oy2 - built.oy1,
		);
		ctx.strokeStyle = "rgb(255,60,60)";
		ctx.lineWidth = Math.max(2, img.width / 400);
		ctx.strokeRect(built.ox1, built.oy1, built.ox2 - built.ox1, built.oy2 - built.oy1);
	}

	canvasWrap.style.display = "block";
	if (dets.length === 0) {
		resultInfo.textContent = `Không phát hiện vết nứt nào (ngưỡng tin cậy ${CONF_THRES}). Thời gian xử lý: ${elapsedMs}ms.`;
	} else {
		const scores = dets.map((d) => (d.score * 100).toFixed(0) + "%").join(", ");
		resultInfo.textContent = `Phát hiện ${dets.length} vùng nứt (độ tin cậy: ${scores}). Thời gian xử lý: ${elapsedMs}ms.`;
	}
	setStatus("Xong. Chọn ảnh khác nếu muốn thử tiếp.");
}

fileInput.addEventListener("change", (e) => {
	const file = e.target.files && e.target.files[0];
	if (!file || !session) return;
	const img = new Image();
	img.onload = () => {
		runInference(img).catch((err) => {
			setStatus("Lỗi khi xử lý ảnh: " + err.message, true);
			console.error(err);
		});
	};
	img.onerror = () => setStatus("Không đọc được ảnh này.", true);
	img.src = URL.createObjectURL(file);
});

loadModel();
