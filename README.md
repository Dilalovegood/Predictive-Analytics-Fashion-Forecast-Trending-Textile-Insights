# Laporan Proyek Machine Learning - Dila Aura Futri

## Domain Proyek

Industri fashion bergerak sangat dinamis, didorong oleh perubahan selera konsumen, musim, dan inovasi desain. Bagi pelaku bisnis fashion seperti brand pakaian dan e-commerce, kemampuan untuk mengidentifikasi tren populer dan atribut produk yang menarik menjadi kunci kesuksesan dalam memenangkan pasar. Melalui pendekatan berbasis data, bisnis fashion kini dapat menggunakan machine learning untuk menganalisis atribut seperti Brand, Kategori, Warna, Ukuran, dan Material guna memahami pola penetapan harga dan memprediksi potensi penjualan produk.

**Mengapa Masalah Ini Harus Diselesaikan**

1. Mengidentifikasi Tren Pasar – Tanpa pemahaman terhadap atribut yang diminati pasar, perusahaan berisiko memproduksi produk yang tidak relevan dan kurang laku
2. Optimalisasi Desain Produk dan Harga – Dengan informasi atribut yang mempengaruhi harga, desainer dan manajer produk dapat mengambil keputusan lebih tepat.
3. Prediksi Penjualan Produk Baru – Machine learning memungkinkan prediksi permintaan produk fashion bahkan sebelum diluncurkan ke pasar.
4. Keunggulan Kompetitif – Perusahaan yang mampu menerapkan analitik cerdas dalam desain dan strategi pricing akan unggul dalam persaingan.

**Bagaimana Masalah Ini Dapat Diselesaikan**

1. Eksplorasi dan Praproses Data – Menangani data kategorikal, normalisasi harga, dan deteksi outlier.
2. Feature Engineering – Encoding atribut fashion agar dapat digunakan dalam model regresi.
3. Pembangunan Model Prediktif – Menggunakan algoritma seperti Random Forest, XGBoost, dan Support Vector Machine (SVM) untuk memprediksi harga sebagai proksi dari popularitas atau nilai pasar.
4. Evaluasi dan Interpretasi Model – Menganalisis pengaruh fitur terhadap prediksi harga dan validasi model dengan metrik error.

**REFERENSI**
[AI Tailoring: Evaluating Influence of Image Features on Fashion
Product Popularity](https://arxiv.org/abs/2411.14737)

## Business Understanding

Tujuan Utama adalah menganalisis dataset Fashion Forecast: Trending Textile Insights untuk memberikan wawasan tentang atribut produk yang paling memengaruhi harga, serta membangun model prediktif yang dapat digunakan untuk merancang produk fashion yang lebih laku dan bernilai tinggi.

### Problem Statements

- Bagaimanan distribusi harga terhadap kategori Brand, Category, Color, Size, dan Material
- Bagaimanan frekueansi dari kategori Brand, Category, Color, Size, dan Material
- Apakah terdapat hubungan tertentu antara fitur-fitur tersebut dengan harga (Price)?
- Atribut apa saja yang paling berpengaruh terhadap harga produk fashion?

### Goals

- Menyediakan analisis deskriptif untuk mengungkap pola umum dalam tren fashion.
- Menemukan hubungan antara atribut produk dengan harga.
- Membangun model prediktif yang akurat untuk memproyeksikan harga produk fashion berdasarkan atribut.
- Memprediksi potensi penjualan untuk meningkatkan revenue.

### Solution statements

Berikut beberapa rencana solusi yang dapat diukur dengan metrik evaluasi (MSE):

- Analisis Deskriptif dan Visualisasi
  Menggunakan boxplot dan countplot untuk melihat distribusi harga dan frekuensi tiap atribut (Brand, Category, Color, Size, Material). Tujuannya adalah mengidentifikasi pola harga serta atribut yang sering muncul.

- Transformasi Fitur Kategorikal
  Fitur seperti Brand dan Material diubah ke bentuk numerik melalui Label Encoding dan One-Hot Encoding, agar dapat digunakan dalam model prediksi.

- Penerapan Model Regresi
  Tiga model digunakan: SVM, Random Forest, dan XGBoost. Masing-masing model dipakai untuk memprediksi harga berdasarkan atribut produk, sekaligus menganalisis feature importance untuk mengetahui atribut paling berpengaruh.

- Evaluasi dan Interpretasi Model
  Evaluasi dilakukan dengan Mean Squared Error (MSE) untuk menilai akurasi model. Visualisasi prediksi vs nilai aktual juga digunakan untuk memahami kinerja dan stabilitas model. Model terbaik dipilih berdasarkan keseimbangan akurasi dan interpretabilitas.

## Data Understanding

Dataset ini diunduh dari Kaggle : [Fashion Forecast Trending Textile Insights](https://www.kaggle.com/code/abdmental01/fashion-forecast-trending-textile-insights/notebook).

### Variabel-variabel pada Superstore Giant Sales Dataset dataset adalah sebagai berikut:

Dataset berisi 1.000 produk fashion dari sebuah perusahaan fast fashion Eropa, mencakup:

- Brand : Merek dari produk pakaian, dipilih dari sejumlah merek populer seperti Nike, Adidas, Puma, Under Armour, Reebok, dan New Balance.
- Category : Kategori dari produk pakaian, seperti Kaos (T-shirt), Celana Jeans, Gaun (Dress), Jaket, Sweater, atau Sepatu (Shoes).
- Color : Warna dari produk pakaian, dipilih dari berbagai pilihan warna seperti Merah, Biru, Hijau, Kuning, Hitam, dan Putih.
- Size : Ukuran dari produk pakaian, dinyatakan dalam ukuran standar seperti XS, S, M, L, XL, atau XXL.
- Material : Komposisi bahan dari produk pakaian, termasuk pilihan seperti Katun (Cotton), Poliester (Polyester), Nilon (Nylon), Wol (Wool), Sutra (Silk), dan Denim.
- Price : Harga dari produk pakaian, diukur dalam mata uang lokal.

### Eksplorasi Data Understanding

- Dataset memiliki 1.000 baris dan 6 kolom :
- Terdapat 5 kolom dengan tipe object, yaitu: Brand, Category, Color Size, Material (fitur non-numerik).
- Terdapat 1 kolom numerik dengan tipe data float64 yaitu: Price.
- Tidak ada missing value.
- Perlu dilakukan encoding dan normalisasi harga.
- Distribusi harga cenderung right-skewed.

## Data Preparation

Pada tahap ini, dilakukan dua teknik data preparation untuk mempersiapkan data mentah menjadi bentuk yang optimal.

- <b>Encoding Fitur Kategori</b>
  **Metode** : Label Encoding menggunakan LabelEncoder dari sklearn.preprocessing.
  **Kolom yang diubah** :
  Semua kolom bertipe object diubah menjadi numerik : Brand, Category, Color, Size, Material.
- <b>Train-Test Split</b>
  **Metode** : Pembagian dataset diabgi menjadi 80:20 yaitu 80% data latih dan 20% data uji menggunakan train_test_split.
  **Alasan** : Tujuan dari pemisahan data adalah untuk melatih model pada sebagian data, lalu menguji kinerjanya pada data yang belum pernah dilihat sebelumnya (test set), sehingga kita bisa mengukur kemampuan generalisasi model.

## Modeling

Pada tahap ini dilakukan pembangunan dan evaluasi tiga model regresi untuk memprediksi harga produk fashion berdasarkan fitur seperti Brand, Category, Color, Size, dan Material. Algoritma yang digunakan meliputi:

1. Linear Regression
2. Random Forest Regressor
3. XGBoost Regressor

Untuk ketiga algoritma tersebut, dilakukan proses Bayesian Optimization menggunakan BayesSearchCV untuk mencari kombinasi parameter terbaik berdasarkan evaluasi Mean Squared Error (MSE).

1. <b>Linear Regression</b>
   Merupakan model baseline yang sederhana dan interpretatif. Model ini bekerja dengan mengasumsikan hubungan linear antara fitur dan target (harga).

- Hasil MSE: 2959.45
- Kelebihan: cepat dan mudah diinterpretasi.
- Kelemahan: tidak cocok untuk hubungan non-linear yang kompleks

2. <b>Random Forest Regressor</b>
   Model ensemble berbasis decision tree yang menggunakan teknik bagging. Parameter model dioptimasi menggunakan Bayesian Optimization.

- Parameter terbaik: n_estimators=200, max_depth=28, min_samples_split=7
- Hasil MSE: 2919.98
- Kelebihan: kuat terhadap outlier dan non-linearitas, memberikan feature importance.
- Kelemahan: lebih kompleks dan memerlukan waktu training lebih lama

3. <b>XGBoost Regressor</b>
   Merupakan model boosting yang terkenal efektif untuk data tabular. Hyperparameter juga disesuaikan dengan Bayesian Optimization.

- Parameter terbaik: learning_rate=0.01, n_estimators=200, max_depth=3, subsample=0.5
- Hasil MSE: 2927.29
- Kelebihan: akurat dan mampu menangani data kompleks.
- Kelemahan: sensitif terhadap overfitting jika tuning tidak tepat.

Model Random Forest Regressor dipilih sebagai solusi akhir karena memberikan kombinasi terbaik dari segi akurasi dan generalisasi.

## Evaluation

Evaluasi model dilakukan menggunakan metrik Mean Squared Error (MSE) karena proyek ini merupakan regresi. MSE mengukur rata-rata dari kuadrat selisih antara prediksi dan nilai aktual — semakin kecil nilai MSE, semakin baik performa model.</br>

<p align="center">
  <img src="evaluasi.png" />
</p>

<b>Kesimpulan Evaluasi</b>

1. Model Random Forest memberikan MSE terendah, menunjukkan generalisasi yang baik dan performa paling stabil terhadap data uji.
2. XGBoost mendekati Random Forest, namun sedikit lebih tinggi nilai error-nya.
3. Linear Regression memiliki MSE tertinggi, mengindikasikan bahwa model linear kurang mampu menangkap kompleksitas hubungan antar fitur.

Prediksi dari masing-masing model terhadap satu sampel data:

<p align="center">
  <img src="prediksi.png" />
</p>

Model SVM (dengan pipeline dan scaling) menghasilkan prediksi paling dekat ke nilai aktual dan menunjukkan keseimbangan antara akurasi dan stabilitas.

<b>Kesimpulan Akhir</b>
Model Support Vector Machine (SVM) dipilih sebagai model terbaik secara keseluruhan karena:

1. Memiliki error prediksi paling kecil pada contoh data
2. Tidak menunjukkan overfitting ekstrem seperti model tree-based.
3. Stabil dan cocok untuk deployment pada data fashion yang fluktuatif.

Proses modeling ini membuktikan bahwa pendekatan berbasis machine learning dapat digunakan untuk memahami dan memprediksi tren harga produk fashion secara lebih akurat dan berbasis data.
