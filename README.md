# Heritage Twin

Prototype hệ thống AI Digital Twin cho bảo tồn - giám sát di sản văn hóa Việt Nam: dựng bản sao số 3D từ
video quay bằng điện thoại (3D Gaussian Splatting) và (đang phát triển) phát hiện vết nứt/xuống cấp bằng
Computer Vision.

## Cấu trúc repo

```
heritage_twin_pipeline_colab.ipynb   Pipeline chinh: video -> COLMAP -> train 3D Gaussian Splatting
                                       (chay tren Google Colab, luu du lieu trung gian vao Google Drive
                                       de resume duoc neu Colab bi ngat session)

heritage_twin_video_to_colmap_colab.ipynb        Notebook cu (giu lai de tham khao, da gop vao file tren)
heritage_twin_train_from_colmap_zip_colab (1).ipynb  Notebook cu (giu lai de tham khao, da gop vao file tren)

docs/                                  Web viewer 3D Gaussian Splatting, tu host, khong can backend/database
                                       (dat ten "docs" de GitHub Pages deploy duoc truc tiep tu nhanh main)
  index.html, main.js                 Vendor tu antimatter15/splat (MIT license), WebGL thuan
  convert.py                          Convert .ply -> .splat offline (tuy chon)
  README.md                           Huong dan chay va nap model
```

## Quy trình dùng thử

1. Quay video 20-30s quanh vật thể (xem hướng dẫn trong notebook pipeline).
2. Chạy `heritage_twin_pipeline_colab.ipynb` trên Google Colab (GPU T4) -> ra file `point_cloud.ply`.
3. Mở `docs/index.html` (xem `docs/README.md`), kéo-thả file `.ply` vào để xem trực tiếp trên trình duyệt.
   Hoặc dùng link GitHub Pages sau khi bật trong Settings -> Pages -> Branch `main` / folder `/docs`.

## Lưu ý

File video/model output (`.mp4`, `.zip`, `.ply`, `.splat`) và file thuyết trình PDF không được commit vào
repo này (file nặng, không phù hợp lưu trong git) — xem `.gitignore`.
