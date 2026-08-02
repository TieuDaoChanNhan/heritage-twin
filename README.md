# Heritage Twin

Prototype hệ thống AI Digital Twin cho bảo tồn - giám sát di sản văn hóa Việt Nam: dựng bản sao số 3D từ
video quay bằng điện thoại (3D Gaussian Splatting) và (đang phát triển) phát hiện vết nứt/xuống cấp bằng
Computer Vision.

## Cấu trúc repo

```
heritage_twin_pipeline_colab.ipynb   Pipeline chính: video -> COLMAP -> train 3D Gaussian Splatting
                                       (chạy trên Google Colab, lưu dữ liệu trung gian vào Google Drive
                                       để resume được nếu Colab bị ngắt session)

heritage_twin_video_to_colmap_colab.ipynb        Notebook cũ (giữ lại để tham khảo, đã gộp vào file trên)
heritage_twin_train_from_colmap_zip_colab (1).ipynb  Notebook cũ (giữ lại để tham khảo, đã gộp vào file trên)

heritage_twin_crack_detection_colab.ipynb   Train YOLOv8-seg phát hiện vết nứt (độc lập với pipeline 3D)
                                       (dataset DeepCrack -> convert nhãn -> train -> demo trên ảnh thật)

docs/                                  Website tự host, không cần backend/database
                                       (đặt tên "docs" để GitHub Pages deploy được trực tiếp từ nhánh main)
  index.html                          Trang giới thiệu dự án (landing page)
  viewer/                             Trang xem model 3D Gaussian Splatting riêng
    index.html, main.js               Vendor từ antimatter15/splat (MIT license), WebGL thuần
    convert.py                        Convert .ply -> .splat offline (tuỳ chọn)
    README.md                         Hướng dẫn chạy và nạp model
  crack/                              Demo phát hiện vết nứt, chạy model ngay trên trình duyệt
    index.html, main.js               Tiền xử lý ảnh + inference + hậu xử lý (NMS, giải mã mask)
    ort/                               Vendor từ onnxruntime-web (MIT license), chỉ dùng WASM backend
    model/best.onnx                   Model đã export - KHÔNG có sẵn, xem docs/crack/README.md để lấy
    README.md                         Hướng dẫn lấy model + chạy thử
```

## Quy trình dùng thử

1. Quay video 20-30s quanh vật thể (xem hướng dẫn trong notebook pipeline).
2. Chạy `heritage_twin_pipeline_colab.ipynb` trên Google Colab (GPU T4) -> ra file `point_cloud.ply`.
3. Mở `docs/viewer/index.html` (xem `docs/viewer/README.md`), chọn/kéo-thả file `.ply` vào để xem trực
   tiếp trên trình duyệt. Hoặc dùng link GitHub Pages sau khi bật trong Settings -> Pages -> Branch
   `main` / folder `/docs` (trang chủ là `docs/index.html`, nút "Xem demo mô hình 3D" dẫn sang viewer).
4. Chạy `heritage_twin_crack_detection_colab.ipynb` tới cell export ONNX, tải `best.onnx`, đặt vào
   `docs/crack/model/best.onnx` (xem `docs/crack/README.md`) để bật demo phát hiện vết nứt trên web.

## Lưu ý

File video/model output (`.mp4`, `.zip`, `.ply`, `.splat`) và file thuyết trình PDF không được commit vào
repo này (file nặng, không phù hợp lưu trong git) — xem `.gitignore`.
