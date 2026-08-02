# Heritage Twin - 3D Viewer (tự host, không cần backend/database)

Viewer này là bản vendor từ [antimatter15/splat](https://github.com/antimatter15/splat) (MIT license) —
1 file `index.html` + 1 file `main.js`, WebGL thuần, **không phụ thuộc CDN hay thư viện ngoài**, nên host ở
đâu cũng được (static hosting): GitHub Pages, Cloudflare Pages, Netlify, hoặc chạy local để test.

## Chạy thử local

```bash
cd viewer
python -m http.server 8000
```

Mở `http://localhost:8000` trên trình duyệt.

## Cách nạp model

Có 2 cách:

1. **Kéo-thả trực tiếp file `.ply`** (output gốc của `train.py` trong pipeline Gaussian Splatting, vd
   `point_cloud/iteration_7000/point_cloud.ply`) vào trang web đang mở. Trình duyệt tự convert sang định
   dạng `.splat` ngay phía client, không cần server xử lý gì thêm.
2. **Dùng query param `?url=`** trỏ tới 1 file `.splat` host sẵn (phải cho phép CORS), vd:
   `http://localhost:8000/?url=pikachu.splat`. Cách này phù hợp khi muốn có 1 link cố định để demo/nhúng,
   không cần thao tác kéo-thả mỗi lần.

   Có thể convert `.ply` -> `.splat` offline trước bằng `convert.py` đi kèm (không cần internet lúc demo):
   ```bash
   python convert.py input.ply output.splat
   ```

**Lưu ý:** nếu mở trang mà không kéo-thả file và không có `?url=`, viewer sẽ mặc định tải 1 scene demo
(`train.splat`) từ Hugging Face để test nhanh — cần internet cho trường hợp này. Khi demo thật (offline
hoặc tại sự kiện không chắc có mạng), nên dùng cách 1 hoặc 2 ở trên với file model thật của mình, tránh phụ
thuộc vào scene demo mạng ngoài.

## Bước tiếp theo

- Sau khi Colab train xong và tải `point_cloud.ply` về (qua Google Drive theo pipeline trong
  `heritage_twin_pipeline_colab.ipynb`), thử kéo-thả file đó vào viewer này để kiểm tra end-to-end.
- Khi đã ổn, deploy `index.html` + `main.js` lên 1 static host (GitHub Pages là nhanh nhất, miễn phí) để có
  link demo cố định cho video/ban giám khảo, kèm sẵn 1-2 file `.splat` convert trước bằng `convert.py`.
