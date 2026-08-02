# Heritage Twin - Demo phát hiện vết nứt (chạy trên trình duyệt)

Trang này chạy model YOLOv8-seg (đã fine-tune trên dataset DeepCrack) **ngay trên trình duyệt** bằng
[ONNX Runtime Web](https://github.com/microsoft/onnxruntime) (MIT license) — không cần server, không cần
GPU, ảnh không rời khỏi máy người dùng.

## Cấu trúc

```
index.html       Giao diện: chọn ảnh, hiển thị kết quả
main.js          Tiền xử lý ảnh (letterbox), chạy inference, hậu xử lý (NMS + giải mã mask), vẽ overlay
ort/             Vendor từ npm package `onnxruntime-web` (MIT), chỉ dùng WASM backend (không CDN)
model/best.onnx  Model đã convert từ best.pt - CHƯA có sẵn, xem hướng dẫn bên dưới
```

## Lấy file `model/best.onnx`

File này chưa được commit sẵn (cần export từ Colab trước). Các bước:

1. Mở `heritage_twin_crack_detection_colab.ipynb`, chạy tới cell "Export sang ONNX" (sau cell đánh giá test
   set). Cell này tự phát hiện model đã train xong (`weights/best.pt` trên Drive), export sang ONNX, lưu
   lại vào Drive và tự tải file `best.onnx` về máy.
2. Copy file vừa tải vào đúng `docs/crack/model/best.onnx` trong repo.
3. Test local: `cd docs && python -m http.server 8000`, mở `http://localhost:8000/crack/`.

## Vì sao dùng WASM (CPU) thay vì WebGL/WebGPU

`onnxruntime-web` hỗ trợ nhiều backend, nhưng bản vendor ở đây chỉ lấy backend WASM vì:
- Chạy được trên mọi trình duyệt hiện đại, không phụ thuộc driver GPU của máy khách.
- Model `yolov8n-seg` đã rất nhẹ (nano), CPU qua WASM vẫn đủ nhanh cho việc xử lý 1 ảnh tĩnh (thường dưới
  1-2 giây), không cần tối ưu thêm bằng GPU backend.
- **Chạy single-thread có chủ đích** (`ort.env.wasm.numThreads = 1` trong `main.js`): chế độ multi-thread
  của WASM cần header `Cross-Origin-Opener-Policy`/`Cross-Origin-Embedder-Policy` mà GitHub Pages không hỗ
  trợ cấu hình — single-thread tránh phụ thuộc vào điều này, đổi lại chậm hơn multi-thread nhưng vẫn đủ
  dùng cho demo 1 ảnh.

## Giới hạn đã biết

- Model train trên dataset DeepCrack (bê tông/tường hiện đại) — xem lưu ý domain gap trong `progress.md`
  gốc repo.
- Output ONNX giả định model **1 class duy nhất** ("crack") — code trong `main.js` có kiểm tra cứng số
  kênh output khớp công thức `4 + 1 + 32`, báo lỗi rõ ràng nếu model export ra khác định dạng này thay vì
  âm thầm cho kết quả sai.
