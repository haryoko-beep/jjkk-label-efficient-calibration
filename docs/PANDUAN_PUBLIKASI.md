# Panduan publikasi GitHub dan Zenodo

Repositori publik sudah tersedia di https://github.com/haryoko-beep/jjkk-label-efficient-calibration. Release v1.0.0 telah diarsipkan pada https://doi.org/10.5281/zenodo.23880279. Langkah di bawah dipertahankan sebagai panduan pengelolaan versi berikutnya. Nama yang disarankan: `jjkk-label-efficient-calibration`. File kode memakai lisensi MIT; pastikan pilihan ini sesuai kesepakatan para penulis sebelum dipublikasikan.

## 1 Periksa paket

Ekstrak ZIP ke folder baru. Gunakan folder repositori di dalam ZIP, bukan seluruh workspace penelitian `D:/JJKK ITS`. Buka terminal di folder yang berisi README, LICENSE, CITATION.cff, dan scripts.

```powershell
python scripts/verify_release.py
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts/reproduce_main_results.py
```

Pemeriksaan checksum tidak memerlukan paket tambahan. Skrip ringkasan membaca hasil asli; skrip tersebut tidak melakukan training. Panduan menjalankan ulang seluruh eksperimen ada di REPRODUCING.md.

## 2 Buat repositori kosong di GitHub

Login ke akun GitHub Anda. Buka https://github.com/new . Masukkan nama `jjkk-label-efficient-calibration`, pilih Public, dan buat repositori. Jangan centang pembuatan README, gitignore, atau lisensi otomatis karena file tersebut sudah tersedia dalam paket. Petunjuk resmi: https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository .

## 3 Unggah dengan Git

Ganti NAMA_AKUN dengan username GitHub Anda pada URL remote. Jalankan dari folder paket, bukan workspace utama.

```powershell
git init
git branch -M main
git add .
git status --short
git commit -m "Release experiment code and aggregate evidence"
git remote add origin https://github.com/NAMA_AKUN/jjkk-label-efficient-calibration.git
git push -u origin main
```

Jika Git meminta identitas commit, isi nama dan email commit milik Anda melalui `git config user.name` dan `git config user.email`, lalu ulangi commit. Saat autentikasi, ikuti login resmi GitHub/Git Credential Manager; jangan menaruh token atau password dalam file repositori. Pastikan data mentah, isi tmp, dan cookie tidak muncul pada `git status`.

## 4 Periksa repositori publik

Buka URL repositori lewat jendela tanpa login. Pastikan README, skrip, evidence, LICENSE, dan CITATION.cff tersedia. Tambahkan `repository-code` dan `url` berisi URL repositori sebenarnya pada CITATION.cff; jangan tambahkan DOI yang belum terbit. Commit dan push perubahan metadata tersebut. Gunakan `git status` untuk memastikan perubahan yang ditambahkan benar-benar metadata publik.

## 5 Hubungkan Zenodo sebelum membuat release

Login di https://zenodo.org menggunakan akun yang Anda pilih dan hubungkan GitHub. Di menu profil Zenodo pilih GitHub, klik Sync now, temukan repositori, lalu aktifkan integrasinya. Petunjuk resmi: https://help.zenodo.org/docs/github/enable-repository/ .

## 6 Terbitkan release v1.0.0

Di GitHub buka Releases, pilih Draft a new release, buat tag `v1.0.0`, target branch main, judul `Experiment code and aggregate evidence v1.0.0`, lalu terbitkan release. Deskripsi: "Code, fixed protocols, aggregate metrics, and scalar training event summaries supporting the JJKK manuscript. Raw datasets and fitted caches are excluded."

Pastikan Zenodo mengarsipkan release dan periksa nama keenam penulis, judul, tipe software, lisensi, dan isi arsip. Ambil DOI versi release yang benar untuk dicantumkan pada paper. Integrasi harus aktif sebelum release; bila release belum muncul di Zenodo, periksa status integrasi terlebih dahulu. Petunjuk resmi: https://help.zenodo.org/docs/github/archive-software/ .

## 7 Perbarui Data Availability

Gunakan teks di DATA_AVAILABILITY_TEMPLATE.md setelah mengganti kolom URL dan DOI dengan tautan nyata. Perbarui paragraf Data Availability dalam naskah Word utama, sumber Markdown, dan S1. Pernyataan penggunaan AI tetap terpisah. Jangan mengganti hasil eksperimen atau mengklaim bahwa log ringkas adalah log query lengkap. Untuk paper gunakan DOI versi yang membekukan kode, bukan tautan branch yang terus berubah.

Jika hanya GitHub yang sudah tersedia, gunakan URL GitHub dan hilangkan kalimat DOI sementara. DOI dapat ditambahkan setelah arsip Zenodo terbit.
