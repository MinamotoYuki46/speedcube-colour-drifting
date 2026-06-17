# Speedcube Gradio Local App

Aplikasi Gradio lokal untuk klasifikasi warna speedcube menggunakan **EfficientNetV2-S + MLP**.

Project ini adalah hasil refactor dari notebook `Demo_ML.ipynb` menjadi file Python biasa agar bisa dijalankan secara lokal.

## Struktur Folder

```text
speedcube_gradio_local/
├── app.py
├── requirements.txt
├── README.md
└── models/
    ├── efficientnetv2_mlp_head.pt
    └── efficientnetv2_mlp_scaler.pkl
```

## File Model yang Dibutuhkan

Sebelum menjalankan aplikasi, letakkan file berikut di folder `models/`:

```text
models/efficientnetv2_mlp_head.pt
models/efficientnetv2_mlp_scaler.pkl
```

Catatan: file `.pt` dan `.pkl` tidak otomatis dibuat dari script ini. Keduanya harus berasal dari hasil training/export model.

## Instalasi

Disarankan menggunakan virtual environment.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### CMD

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Menjalankan Aplikasi

Jalankan:

```bash
python app.py
```

Secara default, aplikasi akan berjalan di:

```text
http://127.0.0.1:7860
```

Buka URL tersebut di browser.

## Opsi Tambahan

Menjalankan dengan folder model berbeda:

```bash
python app.py --model-dir path/to/models
```

Menjalankan di host dan port tertentu:

```bash
python app.py --host 0.0.0.0 --port 7860
```

Membuat public link Gradio:

```bash
python app.py --share
```

Mengaktifkan debug:

```bash
python app.py --debug
```

## Mode Prediksi

Aplikasi menyediakan dua mode:

1. **Single Sticker / ROI**  
   Digunakan untuk memprediksi satu gambar stiker speedcube.

2. **Full Face 3x3**  
   Digunakan untuk memprediksi satu sisi speedcube utuh. Sistem akan melakukan deteksi bidang Rubik, perspective correction, membagi gambar menjadi 9 ROI, lalu memprediksi tiap stiker.

## Catatan Penting

Aplikasi ini tetap menggunakan **EfficientNetV2-S pretrained dari torchvision** sebagai feature extractor. Pada run pertama, `torchvision` mungkin perlu mengunduh weight EfficientNetV2-S jika belum ada di cache lokal.

Jika ingin menjalankan sepenuhnya offline, jalankan aplikasi sekali dengan internet terlebih dahulu agar weight EfficientNet tersimpan di cache PyTorch/torchvision.

## Troubleshooting

### Model tidak ditemukan

Jika muncul error seperti:

```text
File model tidak ditemukan
```

Pastikan file model berada di:

```text
models/efficientnetv2_mlp_head.pt
```

### Scaler tidak ditemukan

Jika muncul error seperti:

```text
File scaler tidak ditemukan
```

Pastikan file scaler berada di:

```text
models/efficientnetv2_mlp_scaler.pkl
```

### OpenCV error saat install

Jika `opencv-python` bermasalah pada environment tertentu, upgrade pip terlebih dahulu:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Perbedaan dari Notebook

Versi ini sudah tidak menggunakan:

- `!pip install ...`
- `google.colab.drive`
- path `/content/drive/...`
- `demo.launch(share=True, inline=True)`

Sebagai gantinya, aplikasi menggunakan path lokal dan dapat dijalankan langsung dengan:

```bash
python app.py
```
