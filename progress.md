# Heritage Twin — Nhật ký tiến độ

> File này ghi lại toàn bộ quyết định + trạng thái hiện tại của dự án, để phiên làm việc sau (người hoặc AI)
> đọc vào là resume được ngay, không phải dò lại từ đầu.

## 1. Ý tưởng gốc

Heritage Twin — hệ sinh thái AI Digital Twin cho bảo tồn/giám sát/quảng bá di sản văn hóa Việt Nam (dự án
dự thi HTML, THPT Chuyên Hà Nội - Amsterdam). Ý tưởng đầy đủ gồm 4 luồng: (1) dựng bản sao số 3D từ video
điện thoại bằng 3D Gaussian Splatting, (2) AI thị giác máy tính phát hiện xuống cấp (nứt, nghiêng, rêu mốc),
(3) trợ lý ảo RAG thuyết minh lịch sử, (4) nền tảng cộng đồng đóng góp dữ liệu. Chi tiết đầy đủ nằm trong
`[HTML] HERITAGE TWIN (1).pdf` (297MB, không commit vào git — xem mục 6).

**Quyết định scope cho demo video đầu tiên:** chỉ làm (1) dựng 3D + (2) phát hiện **vết nứt** (bỏ qua
nghiêng/rêu mốc/đa thời điểm vì phức tạp, không cần thiết cho demo). (3) và (4) chưa làm.

## 2. Trạng thái hiện tại (đã xong)

- [x] Gộp 2 notebook cũ (`video_to_colmap` + `train_from_colmap_zip`) thành 1 pipeline liền mạch:
      `heritage_twin_pipeline_colab.ipynb`.
- [x] Toàn bộ dữ liệu trung gian của pipeline 3D lưu vào Google Drive (`MyDrive/HeritageTwin/...`), resume
      được nếu Colab ngắt session.
- [x] Dựng web viewer 3D tự host (`docs/viewer/`), landing page giới thiệu dự án (`docs/index.html`).
- [x] Đẩy lên GitHub, bật GitHub Pages: repo `TieuDaoChanNhan/heritage-twin`, live tại
      `https://tieudaochannhan.github.io/heritage-twin/`.
- [x] Train YOLOv8-seg phát hiện vết nứt xong (150 epoch, dataset DeepCrack), weight tại
      `MyDrive/HeritageTwin/crack_detection/runs/crack_yolov8n_seg/weights/best.pt`. Đã phân tích
      `results.csv`, **tạm chấp nhận model này** để dùng cho demo — xem chẩn đoán chi tiết ở mục 5.
- [x] Test viewer với model 3D pikachu đã train — chạy ổn, upload `.ply` xem được bình thường.
- [ ] **Chưa làm**: nối 2D crack detection vào 3D model (xem mục 5 "Việc tiếp theo").

## 3. Cấu trúc repo

```
heritage_twin_pipeline_colab.ipynb        Pipeline 3D: video -> COLMAP -> train 3D Gaussian Splatting
heritage_twin_crack_detection_colab.ipynb Train YOLOv8-seg phát hiện vết nứt (độc lập, xem mục 5)
heritage_twin_video_to_colmap_colab.ipynb        Notebook cũ, giữ tham khảo, đã gộp vào file pipeline trên
heritage_twin_train_from_colmap_zip_colab (1).ipynb  Notebook cũ, giữ tham khảo, đã gộp vào file pipeline trên

docs/                      Website tự host (GitHub Pages, branch main, folder /docs)
  index.html               Landing page giới thiệu dự án
  viewer/                  Trang xem model 3D (.ply/.splat), vendor từ antimatter15/splat (MIT)
    index.html, main.js    WebGL thuần, không CDN, có nút "Chọn file model" + kéo-thả
    convert.py             Convert .ply -> .splat offline (tuỳ chọn)
    README.md              Hướng dẫn dùng viewer

progress.md                File này
README.md                  Tổng quan repo
.gitignore                 Loại trừ file nặng (pdf/mp4/zip/ply/splat/model output folders)
```

**Không commit vào git** (bị `.gitignore` chặn, nằm local trên máy `D:\Heritage_Twin`):
`[HTML] HERITAGE TWIN (1).pdf`, `pikachu.mp4`, `pikachu_colmap.zip`, `pikachu_gaussian_7000.zip`,
thư mục `pikachu_gaussian_7000/` (giải nén để test viewer).

## 4. Quyết định thiết kế quan trọng (và lý do)

- **Toàn bộ dữ liệu trung gian lưu Google Drive, không dùng `/content`**: Colab tự ngắt session khi idle
  hoặc hết giờ, `/content` mất sạch mỗi session mới. Cả 2 notebook Colab đều mount Drive và ghi trực tiếp
  vào đó.
- **Mọi bước nặng đều idempotent (skip nếu đã xong)** + **train có checkpoint/resume**
  (`--checkpoint_iterations`/`--start_checkpoint` cho 3DGS, `resume=True` cho YOLO): chạy lại notebook sau
  khi bị ngắt session sẽ tự tiếp tục, không train lại từ đầu.
- **COLMAP dùng `sequential_matcher` thay vì `exhaustive_matcher`** (khác `convert.py` gốc): video là chuỗi
  frame liên tục nên không cần so khớp mọi cặp ảnh. Lưu ý: bước `colmap mapper` (bundle adjustment) **luôn
  chạy CPU**, đây là giới hạn của chính COLMAP, đổi matcher không giúp bước này nhanh hơn nhiều.
- **`TRAIN_ITERS` mặc định = 15000** (đã tăng từ 7000 vì 7000 cho kết quả mờ). Lưu ý quan trọng: chất lượng
  ảnh mờ hay không phụ thuộc **chất lượng video gốc** nhiều hơn là số iteration — lần chạy trước, log
  extract frame báo tất cả frame đều dưới ngưỡng nét (blur score cao nhất chỉ 35.4/ngưỡng 60). Nên ưu tiên
  quay video cẩn thận hơn (cầm chắc tay, đủ sáng, khoá exposure/focus) trước khi đổ thêm iteration.
- **nbformat gotcha đã gặp và fix**: khi tạo notebook bằng script, mỗi dòng trong `source` phải giữ `\n`
  cuối dòng (trừ dòng cuối cùng của cell) — thiếu `\n` khiến Colab (parser nghiêm ngặt) dính chữ các dòng
  lại với nhau (markdown vỡ layout, code lỗi cú pháp), dù VS Code vẫn hiển thị bình thường (tự join bằng
  `\n`). Cả 2 notebook hiện tại đã đúng format.
- **GitHub Pages "deploy from branch" chỉ nhận `/(root)` hoặc `/docs`**, không nhận thư mục tuỳ ý — đây là
  lý do thư mục web đặt tên `docs/` thay vì `viewer/` hay `website/`.
- **Toàn bộ text tiếng Việt trong code/web phải có dấu đầy đủ** — từng viết không dấu ở vài chỗ (quen tránh
  lỗi encode console khi debug), bị nhắc và đã sửa lại toàn bộ, có quét lại bằng grep để xác nhận không sót.
- **Viewer (antimatter15/splat) tự động tải file `.splat` xuống máy khi bạn nạp `.ply`** — đây là hành vi
  có chủ đích của thư viện (convert `.ply` nặng thành `.splat` nhẹ để dùng lại lần sau), không phải lỗi.
  Có thể tắt (đổi `save: true` thành `false` trong `selectFile()` ở `main.js`) nếu thấy gây khó chịu lúc
  demo — **chưa tắt, đang giữ nguyên mặc định**.

## 5. Việc tiếp theo — tích hợp crack detection vào mô hình 3D

**Trạng thái hiện tại:** 2D crack detection (YOLOv8-seg) và 3D reconstruction đang là **2 pipeline độc
lập, chưa nối nhau**. YOLO không thể và sẽ không bao giờ nhận `.ply` làm input — đây không phải hướng đi.

**Đánh giá model crack detection hiện tại (từ `results.csv`, train xong đủ 150 epoch, không early-stop dù
`patience=30`):**
- Không có dấu hiệu overfit (val loss vẫn giảm/ổn định cùng train loss suốt quá trình, không tách ra).
- Nhưng **mAP mask đã chững (plateau) từ khoảng epoch 85-110** — `mAP50-95(M)` dao động quanh 0.10-0.14 tới
  hết epoch 150 (kết thúc 0.12), không còn cải thiện rõ rệt dù train thêm. Kết luận: **bị giới hạn bởi
  lượng/độ đa dạng dữ liệu (256 ảnh train), không phải do thiếu epoch** — train thêm epoch từ giờ không
  còn đáng.
- `Precision(M)` = 0.603, `Recall(M)` = 0.364 (epoch cuối) — model thiên về thận trọng, đoán đúng khi báo
  có nứt nhưng **bỏ sót phần lớn vết nứt thật** (chỉ bắt được ~36%).
- **Quyết định**: tạm chấp nhận model này để dùng cho demo, không train thêm/cải tiến ngay bây giờ.
- **Nếu sau này muốn cải thiện** (chưa làm, xếp theo ưu tiên): (1) fine-tune tiếp từ `best.pt` hiện tại
  trên 1 bộ nhỏ ảnh domain thật (đá/gỗ/gạch di sản, tự chụp + gán nhãn tay ~30-50 ảnh bằng Roboflow/
  makesense.ai) — đáng giá nhất vì đánh thẳng vào domain gap đã biết và vào việc recall thấp; (2) đổi
  `yolov8n-seg.pt` → `yolov8s-seg.pt` (rẻ, ít rủi ro, chưa thử); (3) gộp thêm dataset crack công khai khác
  (Crack500, CFD...) — ưu tiên thấp hơn vì vẫn không giải quyết domain gap.
- **Resume behavior đã xác nhận**: chạy lại `heritage_twin_crack_detection_colab.ipynb` lần sau sẽ **tự
  động skip training**, chỉ load `best.pt` có sẵn (logic idempotent trong cell `train01`, giống pattern
  của pipeline 3D) — miễn `RUN_NAME` vẫn giữ nguyên `"crack_yolov8n_seg"`. Muốn train lại thì bật
  `FORCE_RETRAIN = True`.

**Cơ chế đúng để nối 2D vào 3D (đã thống nhất, CHƯA code):**

1. Chạy crack detector trên đúng các frame nằm trong `INPUT_DIR` của pipeline 3D
   (`MyDrive/HeritageTwin/scenes/{SCENE_NAME}/input/frame_XXXX.jpg`) — **bắt buộc phải là đúng các frame
   này**, vì chỉ chúng mới có camera pose đã được COLMAP tính sẵn trong `sparse/0/images.bin`. Ảnh upload
   tuỳ ý (như cell `demo02` hiện tại) không có pose nên không backproject được.
   - Lưu ý: không phải mọi frame trong `input/` đều chắc chắn có pose (COLMAP có thể đăng ký thất bại vài
     frame) — cần lọc theo danh sách thực sự có trong `images.bin`, không giả định tất cả.
2. Với mỗi frame có cả (a) mask vết nứt 2D từ YOLO và (b) camera pose từ COLMAP: chiếu ngược từng tâm
   Gaussian trong `point_cloud.ply` vào ảnh 2D của frame đó (phép chiếu phối cảnh chuẩn, dùng lại
   `scene/colmap_loader.py` đã có sẵn trong repo `gaussian-splatting` đã clone — không cần tự viết parser
   COLMAP binary từ đầu).
3. Điểm 3D nào chiếu trúng vùng mask → đánh dấu là "crack". **Cần thêm bước lọc theo độ sâu** (so với độ
   sâu ước lượng từ sparse point cloud tại vùng đó) để loại các điểm ở phía trước/sau bề mặt thật vô tình
   trùng toạ độ 2D — đây là phần khó nhất, chưa có code, cần viết mới (ước lượng: 1 buổi làm việc tập
   trung, dùng numpy/scipy, không cần thư viện ngoài nào nặng).
4. Output: `point_cloud.ply` (hoặc file phụ liệt kê index điểm bị nứt) mang thông tin vết nứt ngay trong
   không gian 3D, mở trong `docs/viewer/` là thấy tô màu khác trên mô hình.

**Việc phụ chưa làm (nhỏ hơn):**
- Thêm 1 cell vào `heritage_twin_crack_detection_colab.ipynb` để tự đọc frame thẳng từ `INPUT_DIR` của
  pipeline 3D (dùng chung `SCENE_NAME`/`DRIVE_ROOT`) thay vì chỉ upload tay — chuẩn bị cho bước tích hợp ở
  trên. Đã đề xuất, **user chưa xác nhận có muốn làm không**.
- Cân nhắc thêm `USE_EXPOSURE_COMPENSATION` (có ở notebook cũ, bị rơi mất khi gộp) nếu ảnh 3DGS bị loang
  sáng/tối không đều giữa các frame.
- Cân nhắc GLOMAP thay cho COLMAP mapper để giảm thời gian CPU-bound (chưa nghiên cứu kỹ, chưa làm).

## 6. Việc quay video demo

- Đã thống nhất: bắt đầu với **vật thể nhỏ có vết nứt thật hoặc tự tạo** (mảnh gốm vỡ, phù điêu nhỏ, gạch/
  ngói nứt...) thay vì cả công trình — dựng nhanh hơn nhiều, validate được cả pipeline đầu-cuối trước khi
  làm scene lớn.
- **Cần quay lại video cẩn thận hơn lần trước**: cầm chắc tay, đủ sáng, khoá exposure/focus nếu máy cho
  phép, đi chậm đúng 1 vòng quanh vật thể (xem hướng dẫn chi tiết trong `heritage_twin_pipeline_colab.ipynb`
  cell `guide01`). Lần quay trước (video pikachu) bị mờ dưới ngưỡng nét yêu cầu.
- Dataset DeepCrack dùng để train crack model là bê tông/tường hiện đại, **không phải đá/gỗ cổ** — cần tự
  test model trên ảnh thật của vật thể định quay demo (cell `demo02`) trước khi tin tưởng dùng chính thức,
  và không nên PR là "chính xác cho di sản Việt Nam" nếu chưa kiểm chứng.

## 7. Tham khảo

- Nghiên cứu liên quan đã tìm thấy (không phải tiên phong về ý tưởng, nhưng chưa có sản phẩm ứng dụng cho
  di sản Việt Nam cụ thể): xem tin nhắn "có video demo nào trên mạng sẵn không" trong lịch sử hội thoại —
  3 bài gần nhất là "Intelligent Defect Detection of Ancient City Walls" (Jingzhou), "Defect segmentation +
  3D reconstruction using SAM 2 and 3DGS", "3D visualization of damaged statues using Gaussian Splatting".
- Repo: https://github.com/TieuDaoChanNhan/heritage-twin (public)
- Demo web: https://tieudaochannhan.github.io/heritage-twin/
