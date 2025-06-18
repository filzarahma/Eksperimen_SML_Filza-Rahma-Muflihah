# Eksperimen_SML_Filza

## Deskripsi Proyek

Eksperimen ini bertujuan untuk melakukan prediksi penyakit jantung menggunakan dataset [Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction). Proyek ini mencakup tahapan EDA, preprocessing otomatis, dan penyimpanan pipeline serta data hasil transformasi.

## Struktur Folder

- `heart_raw.csv` : Dataset mentah.
- `preprocessing/automate.py` : Script preprocessing otomatis.
- `preprocessing/automate.ipynb` : Notebook eksplorasi dan validasi pipeline preprocessing.
- `preprocessing/Eksperimen_Filza.ipynb` : Notebook EDA dan preprocessing manual.
- `preprocessing/heart_preprocessing/` : Output hasil preprocessing.
- `.github/workflows/preprocessing.yml` : Workflow GitHub Actions untuk otomatisasi preprocessing.

## Dataset

Dataset terdiri dari fitur numerik dan kategorikal:
- **Age**: Usia pasien
- **Sex**: Jenis kelamin (M/F)
- **ChestPainType**: Tipe nyeri dada (ATA, NAP, ASY, TA)
- **RestingBP**: Tekanan darah istirahat
- **Cholesterol**: Kadar kolesterol
- **FastingBS**: Gula darah puasa (0/1)
- **RestingECG**: Hasil EKG (Normal, ST, LVH)
- **MaxHR**: Detak jantung maksimum
- **ExerciseAngina**: Angina saat olahraga (Y/N)
- **Oldpeak**: Depresi ST
- **ST_Slope**: Kemiringan segmen ST (Up, Flat, Down)
- **HeartDisease**: Target (0/1)

## Preprocessing Otomatis

Preprocessing dilakukan dengan script [`preprocessing/automate.py`](preprocessing/automate.py):
- Imputasi missing value (mean untuk numerik, 'missing' untuk kategori)
- Scaling numerik dengan MinMaxScaler
- Encoding kategori dengan OrdinalEncoder
- Split data train/test (80:20)
- Simpan pipeline dan header kolom

Contoh penggunaan:
```sh
python preprocessing/automate.py heart_raw.csv HeartDisease preprocessor_pipeline.joblib preprocessing/heart_preprocessing/header_data.csv
```

## Notebook Eksplorasi

- [`preprocessing/Eksperimen_Filza.ipynb`](preprocessing/Eksperimen_Filza.ipynb): EDA, deteksi outlier, scaling, encoding manual, dan insight preprocessing.
- [`preprocessing/automate.ipynb`](preprocessing/automate.ipynb): Validasi pipeline otomatis, contoh inference dan inverse transform.

## Otomatisasi dengan GitHub Actions

Workflow di `.github/workflows/preprocessing.yml` akan:
- Menjalankan preprocessing otomatis setiap ada perubahan pada folder preprocessing atau dataset mentah.
- Mengunggah hasil preprocessing ke artifact GitHub Actions.

## Output

Hasil preprocessing dapat ditemukan di folder `preprocessing/heart_preprocessing/`:
- `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`
- `preprocessor_pipeline.joblib` (pipeline siap inference)
- `header_data.csv` (header kolom fitur)

---

**Catatan:**  
Seluruh kode dan pipeline dapat digunakan untuk inference data baru dengan memanfaatkan pipeline yang telah disimpan (`preprocessor_pipeline.joblib`) dan mengikuti format yang sama seperti dataset asli.
