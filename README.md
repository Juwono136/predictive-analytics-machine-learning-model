# Laporan Proyek Machine Learning - Juwono

## Domain Proyek

![img-ames-housing-dataset](https://storage.googleapis.com/kaggle-media/competitions/House%20Prices/kaggle_5407_media_housesbanner.png)

Misalkan anda sedang mencari rumah yang ideal untuk tempat tinggal bersama keluarga, pasti anda ingin mencari rumah yang menurut anda sesuai baik itu dari segi harga rumah, kondisi lingkungan atau beberapa faktor lainnya. Pasar perumahan memang merupakan salah satu yang dinamis namun kompleks. Fluktuasi atau perubahan harga rumah dapat dipengaruhi oleh beberapa faktor termasuk lokasi, ukuran rumah, kondisi bangunan, faktor ekonomi makro dan sebagainya. Beberapa faktor tadi terkadang membingungkan dan membutuhkan waktu bagi anda untuk mempertimbangkan dan memilih rumah yang sesuai. Oleh karena itu, dibutuhkan sebuah sistem untuk memprediksi harga rumah yang dapat memberikan estimasi dengan tepat/pantas dan membantu para penjual dan pembeli dalam mengambil keputusan.

Dengan membangun sistem yang bisa memprediksi harga rumah, pembeli maupun penjual akan mendapatkan harga rumah yang akurat serta ekspektasi harga yang realistis sehingga bisa meminimalkan risiko overpricing atau underpricing. Jika anda seorang investor atau pengembang, memiliki perkiraan harga rumah yang akurat dapat membantu anda dalam menentukan apakah suatu properti merupakan investasi yang menguntungkan atau tidak.

Untuk proses pengembangan model machine learning, dataset yang kita gunakan pada proyek ini adalah [Ames Housing Dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview). Dataset ini berisi berbagai atribut rumah yang ada di daerah Ames, Iowa, United States termasuk dengan harga jualnya dan jumlah fitur yang mencangkup setiap aspek rumah hunian di daerah tersebut. Menurut hasil studi dari [Alan Ihre et al](https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1354741&dswid=-7113) dalam jurnalnya berjudul *Predicting House Prices with Machine Learning Methods* dengan menggunakan dataset yang sama, didapat hasil bahwa permasalahan mengenai prediksi harga rumah bisa menggunakan metode regresi seperti misalnya KNN (K-Nearest Neighbor) atau Random Forest. Dengan membangun model prediksi harga rumah berdasarkan dataset dari daerah Ames, diharapkan hasil model dari proyek ini dapat juga digunakan untuk memprediksi harga rumah di daerah lain.

## Business Understanding
### Problem Statements
Sekarang, bayangkan anda adalah pemilik perusahaan atau pengembang properti perumahan yang ingin menentukan harga jual rumah pantas tapi perusahaan tetap ingin mendapat keuntungan sebesar mungkin dari penjualan properti tersebut. Berdasarkan kondisi yang telah diuraikan sebelumnya, perusahaan akan mengembangkan sistem yang mampu memprediksi harga rumah untuk menjawab permasalahan berikut:
- Dari serangkaian fitur yang ada, fitur apa saja yang paling berpengaruh terhadap harga jual rumah?
- Berapa harga pasar jual rumah dengan karakteristik atau fitur tertentu?

### Goals
Untuk menjawab pertanyaan tersebut, perusahaan akan membuat predictive modelling dengan tujuan atau goals sebagai berikut:
- Mengetahui fitur yang paling berkorelasi dengan harga jual rumah.
- Membuat model machine learning yang dapat memprediksi harga jual rumah seakurat mungkin berdasarkan fitur - fitur yang ada.

### Metodologi
Prediksi harga adalah tujuan yang ingin dicapai. Seperti yang kita ketahui, harga merupakan variabel kontinu. Dalam predictive analytics, untuk data yang bersifat variabel kontinu artinya merupakan permasalahan regresi. Oleh karena itu, metodologi pada proyek ini adalah: membangun model regresi dengan harga jual rumah sebagai target.

### Solution Statements
Berdasarkan tujuan atau goals yang sudah dijelaskan sebelumnya, solusi yang diberikan untuk menyelesaikan permasalahan adalah sebagai berikut:
- Membuat sebuah model untuk memprediksi harga jual rumah. Pengembangan model akan menggunakan beberapa algoritma machine learning yaitu K-Nearest Neighbor, Random Forest, dan Boosting Algorithm dengan mengatur hyperparameter tuning dan melakukan improvement pada baseline model. Dari ketiga model ini, akan dipilih satu model yang memiliki nlai kesalahan prediksi terkecil untuk membuat model yang seakurat mungkin.
- Metrik digunakan untuk mengevaluasi seberapa baik model dalam memprediksi harga. Karena kasus kali ini merupakan kasus regresi, beberapa metrik yang digunakan adalah Mean Squared Error (MSE) dan Root Mean Squared Error (RMSE). Secara umum, metrik ini mengukur seberapa jauh hasil prediksi dengan nilai sebenarnya.

## Data Understanding
Data yang digunakan pada proyek ini adalah Ames Housing Dataset yang diunduh dari platform [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data). Dataset ini merupakan dataset competition yang diguanakan untuk latihan di Kaggle dan didalamnya terdapat 1.460 jenis rumah hunian dengan berbagai karakteristik dan harga. Karakteristik terdiri dari 79 jenis fitur yang mencangkup fitur numerik dan non-numerik serta terdapat 1 label yaitu sale price (harga rumah) sebagai variabel target. Fitur numerik seperti misalnya MSSub Class, Lot Frontage, dan Lot Area, sedangkan fitur non-numerik misalnya Street, Alley, dan Lot Shape. 79 Jenis fitur ini adalah fitur yang akan digunakan dalam menentukan pola data, sedangkan harga jual merupakan fitur target. Karena banyaknya jumlah fitur/variabel yang tersedia dalam dataset, maka dibuatkan file bernama data_description.txt untuk melihat informasi lebih lengkap mengenai kategori dan penjelasan dari masing - masing fitur yang tersedia. Berikut adalah struktur folder Ames Housing Dataset yang sudah di donwload:

    ├── house-prices                 <- nama folder utama.
       ├── data_description.txt      <- berisi kategori dan penjelasan lengkap dari masing - masing fitur.
       ├── sample_submission.csv     <- berisi contoh submission kompetisi.
       ├── test.csv                  <- test data untuk melakukan pengujian model.
       └── train.csv                 <- train data untuk melatih model.

### Variabel - variabel pada Ames Housing Dataset adalah sebagai berikut:
- SalePrice : Harga jual rumah dalam dollar. Ini adalah variabel target yang ingin kita prediksi.
- MSSubClass : Tipe atau kelas bangunan, terdiri dari beberapa jenis kelas rumah yang direpresentasikan dalam bentuk angka.
- MSZoning : Klasifikasi zonasi umum dari penjualan properti yang direpresentasikan dalam bentuk huruf seperti A (Agriculture/Pertanian), C (Commercial/Komersial), FV (Pemukiman Desa Terapung), I (Industri), RH (Kepadatan tinggi perumahan), RL (Kepadatan rendah perumahan), RP (Kepadatan rendah perumahan taman), dan RM (Kepadatan menengah perumahan).
- LotFrontage : Panjang linear dari jalan yang terhubung ke properti.
- LotArea : Luas lot (lahan/tanah) dalam kaki persegi.
- Street : Jenis akses jalan ke properti.
- Alley : Jenis akses gang ke properti.
- LotShape : Bentuk umum dari properti.
- LandContour : Kontur tanah dari properti.
- Utilities : Jenis utilitas yang tersedia.
- LotConfig : Konfigurasi lot.
- LandSlope : Kemiringan properti.
- Neighborhood : Lokasi fisik dalam batas kota Ames.
- Condition1 : Proksimitas (kedekatan) dengan properti.
- Condition2 : Proksimitas terhadap berbagai kondisi di dekat properti (jika lebih dari satu).
- BldgType : Tipe hunian.
- HouseStyle : Gaya hunian.
- OverallQual : Penilaian bahan/material dan penyelesaian keseluruhan rumah.
- OverallCond : Penilaian kondisi keseluruhan rumah.
- YearBuilt : Tanggal konstruksi asli.
- YearRemodAdd : Tanggal remodel (sama dengan tanggal konstruksi jika tidak ada renovasi atau penambahan).
- RoofStyle : Jenis atap.
- RoofMatl : Bahan atap.
- Exterior1st : Penutup luar rumah.
- Exterior2nd : Penutup luar rumah (jika lebih dari satu bahan).
- MasVnrType : Jenis veneer batu bata.
- MasVnrArea : Luas veneer batu bata dalam kaki persegi.
- ExterQual : Kualitas material pada eksterior.
- ExterCond : Kondisi saat ini dari material pada eksterior.
- Foundation : Jenis pondasi.
- BsmtQual : Ketinggian basement.
- BsmtCond : Kondisi umum basement.
- BsmtExposure : Dinding tingkat berjalan atau taman.
- BsmtFinType1 : Penilaian area basement yang sudah selesai.
- BsmtFinSF1 : Tipe 1 yang sudah selesai dalam kaki persegi.
- BsmtFinType2 : Penilaian area basement yang sudah selesai (jika ada beberapa tipe).
- BsmtFinSF2 : tipe 2 yang sudah selesai dalam kaki persegi.
- BsmtUnfSF : area yang belum selesai dari area basement dalam kaki persegi.
- TotalBsmtSF : Total area basement dalam kaki persegi.
- Heating : jenis pemanas.
- HeatingQC : Kualitas dan kondisi pemanas.
- CentralAir : terdapat pemanas udara pusat.
- Electrical : Sistem listrik.
- 1stFlrSF : Luas lantai pertama dalam kaki persegi.
- 2ndFlrSF : Luas lantai kedua dalam kaki persegi.
- LowQualFinSF : luas lantai yang sudah selesai berkualitas rendah dalam kaki persegi.
- GrLivArea : Area tinggal di atas tingkat tanah dalam kaki persegi.
- BsmtFullBath : Kamar mandi penuh di basement.
- BsmtHalfBath : Kamar mandi setengah di basement.
- FullBath : Kamar mandi penuh di atas tingkat tanah.
- HalfBath : Kamar mandi setengah di atas tingkat tanah (lantai atas).
- Bedroom : Kamar tidur di atas tingkat tanah (TIDAK termasuk kamar tidur di basement).
- Kitchen : Dapur di atas tingkat tanah.
- KitchenQual : Kualitas dapur.
- TotRmsAbvGrd : Total kamar di atas tingkat tanah (tidak termasuk kamar mandi).
- Functional : Fungsionalitas rumah (Anggap tipikal kecuali ada pengurangan).
- Fireplaces : Jumlah perapian.
- FireplaceQu : Kualitas perapian.
- GarageType : Lokasi garasi.
- GarageYrBlt : Tahun pembuatan garasi.
- GarageFinish : Penyelesaian interior garasi.
- GarageCars : Ukuran garasi dalam kapasitas mobil.
- GarageArea : Ukuran garasi dalam kaki persegi.
- GarageQual : Kualitas garasi.
- GarageCond : Kondisi garasi.
- PavedDrive : Jalan beraspal.
- WoodDeckSF : Luas area dek kayu dalam kaki persegi.
- OpenPorchSF : Luas area teras terbuka dalam kaki persegi.
- EnclosedPorch : Luas area teras tertutup dalam kaki persegi.
- 3SsnPorch : Luas area teras tiga musim dalam kaki persegi.
- ScreenPorch : Luas area teras berlayar dalam kaki persegi.
- PoolArea : Luas area kolam renang dalam kaki persegi.
- PoolQC : Kualitas kolam renang.
- Fence : Kualitas pagar.
- MiscFeature : Fitur tambahan lainnya yang tidak termasuk dalam kategori lain.
- MiscVal : Nilai fitur tambahan dalam dolar.
- MoSold : Bulan Dijual (MM).
- YrSold : Tahun Dijual (YYYY).
- SaleType : Jenis penjualan.
- SaleCondition : Kondisi penjualan.

### Berikut adalah beberapa tahapan untuk memahami data:
- Data Loading
- Exploratory Data Analysis - Deskripsi Variabel
- Exploratory Data Analysis - Identifikasi Missing Value, Outliers dan Menghapus fitur yang tidak diperlukan untuk pelatihan model
- Exploratory Data Analysis - Univariate Analysis
- Exploratory Data Analysis - Multivariate Analysis

### Data Loading
Pada bagian ini, kita akan mencoba membaca dataset secara langsung dari folder dataset yang sudah di download melalui [Ames Housing Dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data). Dataset yang diguanakan adalah train.csv yang berisi dataset untuk proses pelatihan model.
```
  # load the dataset
  dataset = 'house-prices/train.csv'
  house = pd.read_csv(dataset)
  house
```

![data01](https://github.com/Juwono136/predictive-analytics-machine-learning-model/assets/70443393/2f79b0c5-14ca-4e25-99fd-8d8b35c3b8e2)

Output dari kode diatas memberikan informasi sebagai berikut:
- Terdapat 1.460 baris (records atau jumlah pengamatan) dalam dataset.
- Terdapat 81 kolom yaitu: Id, MSSubClass, MSZoning, LotFrontage, dan sebagainya.

### Exploratory Data Analysis - Deskripsi Variabel
Exploratory data analysis atau sering disingkat EDA merupakan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data. Pada proses EDA ini kita akan melakukan deskripsi variabel untuk mengetahui informasi lebih lengkap dan mengecek informasi pada dataset. Berdasarkan informasi dari [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) Terdapat sekitar 80 variabel (termasuk harga rumah) yang dijelaskan lebih lengkap di file data_description.txt. Berikut adalah contoh penjelasan dari salah satu variabel yaitu MSZoning yang berasal dari file data_description.txt:
- MSZoning : Mengidentifikasi klasifikasi zona umum dari penjualan. Terdiri dari:
   - A : Pertanian
   - C : Komersial
   - FV : Perumahan Desa Mengambang
   - I : Industri
   - RH : Kepadatan Tinggi Perumahan
   - RL : Kepadatan Rendah Perumahan
   - RP : Taman Kepadatan Rendah Perumahan
   - RM : Kepadatan Sedang Perumahan

Pertama, kita akan mengecek informasi pada dataset menggunakan fungsi info() berikut.
```
  house.info()
```

Dari output terlihat bahwa:
- Terdapat 43 kolom dengan tipe object. Kolom ini merupakan categorical features (fitur non-numerik)
- Terdapat 3 kolom dengan tipe data float64. Kolom ini merupakan fitur numerik yang merupakan hasil pengukuran secara fisik.
- Terdapat 35 kolom dengan tipe data int64. Kolom ini merupakan fitur numerik yang salah satunya adalah target fitur kita yaitu harga jual rumah.

Terdapat juga beberapa fitur/kolom yang memiliki nilai null/NaN. Berdasarkan hasil analisis, terdapat 19 fitur yang mempunyai nilai NaN/Null yaitu 'LotFrontage', 'Alley', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', dan 'MiscFeature'. Karena nilai ini nantinya akan mengganggu kinerja dari model maka kita akan hapus fitur yang memiliki nilai NaN/Null tersebut.

Setelah di drop fitur yang memiliki nilai NaN/Null, dataset akan menjadi seperti berikut:

![data2](https://github.com/Juwono136/predictive-analytics-machine-learning-model/assets/70443393/cfe83fab-b5d6-45b4-9015-0cbcff5affae)

Sekarang jumlah fitur adalah sebesar 62 fitur setelah menghilangkan fitur yang memiliki nilai NaN/Null. Selanjutnya, karena semua kolom telah memiliki tipe data yang sesuai. kita perlu mengecek deksripsi statistik data menggunakan fitur describe().
```
  house.describe()
```

Fungsi describe() memberikan informasi statistik pada masing - masing kolom, antara lain:
- Count  adalah jumlah sampel pada data.
- Mean adalah nilai rata-rata.
- Std adalah standar deviasi.
- Min yaitu nilai minimum setiap kolom. 
- 25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama. 
- 50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
- 75% adalah kuartil ketiga.
- Max adalah nilai maksimum.

### Exploratory Data Analysis - Menangani Missing Value, Outliers dan Menghapus fitur yang tidak perlu
#### Identifikasi Missing Value
Dari hasil fungsi describe(), nilai minimum untuk beberapa kolom adalah 0. Karena terdapat banyak fitur dalam dataset, sehingga dibeberapa kasus nilai 0 ini menyulitkan bagi kita untuk melihat keseluruhan fitur, maka kita lakukan investigasi untuk mencari informasi semua fitur dengan nilai minimum sama dengan 0 melalui kode dibawah ini.
```
# Melihat informasi semua fitur dengan nilai minimum 0
describe_result = house.describe()
min_values_0 = (describe_result.loc["min"] == 0)
col_with_min_0  = min_values_0[min_values_0].index

print(f"Jumlah kolom dengan nilai minimum 0: {len(col_with_min_0)}\n")
print("Fitur dengan nilai minimum 0:")
print(f"{col_with_min_0}\n")

for min_0 in col_with_min_0:
    index_min_0 = (house[min_0] == 0).sum()
    print(f"Nilai 0 di kolom {min_0} ada: ", index_min_0)
```

Berdasarkan kode diatas, didapat bahwa terdapat 22 fitur yang memiliki nilai minimum 0. Berdasarkan data description di file data_description.txt kita mengetahui bahwa nilai 0 di dataset juga memiliki makna tertentu yang artinya bukan merupakan nilai missing value tapi berupa informasi yang ada di beberapa variabel. Misalnya untuk fitur PoolArea yang paling banyak terdapat nilai 0 sebesar 1.453, nilai 0 disini memiliki arti bahwa di beberapa jenis properti rumah tersebut tidak terdapat kolam renang. Nilai 0 ini juga tidak bisa dianggap sebagai tipe data boolean karena angka 0 disini memiliki arti jumlah (kuantitas) bukan bermakna True/False. Jadi bisa kita simpulkan bahwa tidak terdapat missing value pada dataset tersebut.

#### Menghilangkan outliers
Outliers adalah sampel yang nilainya sangat jauh dari cakupan umum data utama. Ada beberapa teknik outliers yang umum digunakan. Pada proyek ini kita akan menggunakan metode IQR (Inter Quartile Range). IQR menggunakan konsep kuartil untuk menghilangkan outliers, Kuartil dari suatu populasi adalah tiga nilai yang membagi distribusi data menjadi empat sebaran. Seperempat dari data berada di bawah kuartil pertama (Q1), setengah dari data berada di bawah kuartil kedua (Q2), dan tiga perempat dari data berada di kuartil ketiga (Q3). Dengan demikian interquartile range atau IQR = Q3 - Q1.

Hal pertama yang perlu dilakukan adalah membuat batas bawah dan batas atas. Untuk membuat batas bawah, kurangi Q1 dengan 1,5 * IQR. Kemudian, untuk membuat batas atas, tambahkan 1.5 * IQR dengan Q3.
```
Batas bawah = Q1 - 1.5 * IQR
Batas atas = Q3 + 1.5 * IQR
```

Sekarang kita visualisasikan terlebih dahulu dataset dengan boxplot untuk mendeteksi outliers pada beberapa fitur numerik. Misalnya pada fitur 'MSSubClass':

![data3](https://github.com/Juwono136/predictive-analytics-machine-learning-model/assets/70443393/acf1eb63-dd78-4e1d-ba9e-f5e9acd5d76d)

Jika kita perhatikan dan cek akan ada beberapa fitur numerik yang memiliki outliers. Selanjutnya kita akan mengatasi outliers tersebut dengan metode IQR. Kita akan menggunakan metode IQR untuk mengidentifikasi outlier yang berada di luar Q1 dan Q3. Nilai apa pun yang berada di luar batas ini dianggap sebagai outlier.
```
Q1 = house.quantile(0.25)
Q3 = house.quantile(0.75)
IQR = Q3-Q1
house = house[~((house<(Q1-1.5*IQR))|(house>(Q3+1.5*IQR))).any(axis=1)]
 
# Cek ukuran dataset setelah kita drop outliers
house.shape
```

Output yang didapat dari kode diatas adalah (601, 62). Ini berarti dataset kita sudah bersih dan memiliki 601 sampel. Untuk lebih jelasnya, kita bisa cek kembali fitur 'MSSubClass' menggunakan boxplot dan terlihat bahwa fitur tersebut sudah bersih dari outliers.

![data4](https://github.com/Juwono136/predictive-analytics-machine-learning-model/assets/70443393/4767e2ef-c0fc-40c6-ac1e-7b21e0d37388)

#### Menghapus fitur yang tidak diperlukan
Sebelum kita melakukan proses analisis data lebih lanjut. Kita perlu mengecek fitur yang tidak terlalu berpengaruh pada proses pemodelan nantinya. Penghapusan fitur yang tidak diperlukan akan membantu mempercepat proses pelatihan model dan membantu kita lebih memahami data dengan lebih mudah. 
- Pertama, Kita akan menghapus kolom Id karena tidak terlalu berpengaruh pada proses training nantinya.
- Mengecek jumlah fitur yang mempunyai unique value hanya 1 saja. Jika suatu fitur hanya memiliki satu nilai unik, maka fitur tersebut tidak terlalu berpengaruh atau memberikan banyak informasi yang berguna dalam analisis statistik atau pemodelan. Fitur ini sering disebut juga sebagai fitur konstan.

Setelah dibersihkan dari fitur yang tidak perlu, maka dataset kita hanya tersisa 49 fitur/variabel dan akan terlihat seperti berikut:

![data5](https://github.com/Juwono136/predictive-analytics-machine-learning-model/assets/70443393/dbb065fe-b5c7-453a-a87d-e07b8656e336)

### Exploratory Data Analysis - Univariate Analysis
Sebelum kita akan melakukan proses analisis data dengan teknik Univariate EDA. Pertama, kita bagi fitur pada dataset menjadi dua bagian, yaitu numerical features dan categorical features (non numerik). Lakukan analisis pada fitur kategori terlebih dahulu.
```
numerical_features = house.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_features = house.select_dtypes(include=['object']).columns.tolist()
```
#### Categorical Features
Kemudian, Untuk fitur kategori, kita visualisasikan dalam bentuk plot dan data frame untuk menganalisis persentase dari masing - masing fitur. Berikut adalah salah satu visualisasi dari fitur 'KitchenQual':

![data6](https://github.com/Juwono136/predictive-analytics-machine-learning-model/assets/70443393/078b08d3-71d6-403d-ba9f-f890b0d3c516)

Berdasarkan informasi diatas, kita mengetahui bahwa beberapa fitur kategori memiliki persentase yang berbeda. Persentase ini menunjukkan jumlah kategori dari masing - masing fitur atau seberapa sering kategori itu muncul pada fitur tersebut.

#### Numerical Features
Untuk fitur numerik kita bisa menggunakan histogram.
```
house.hist(bins=50, figsize=(20,15))
plt.show()
```

![data7](https://github.com/Juwono136/predictive-analytics-machine-learning-model/assets/70443393/f4d9b8b3-39d2-4209-a85b-250db0ce0e47)

Perhatikan histogram diatas, khususnya histogram untuk variabel "SalePrice" yang merupakan fitur target (label) pada dataset kita. Dari histogram "SalePrice" tersebut terdapat beberapa informasi, antara lain:
- Peningkatan harga jual rumah terdistribusi dengan cukup baik. Hal ini dapat dilihat pada histogram "SalePrice" yang mana sampel cenderung meningkat lalu mengalami penurunan seiring dengan meningkatnya harga jual rumah.
- rentang harga jual rumah cukup tinggi yaitu skala puluhan ribu dollar Amerika hingga sekitar $350000.
- Sebagian besar harga jual rumah bernilai antara $100000 sampai $200000.
- Distribusi harga cenderung cukup normal. Hal ini kemungkinan besar akan berimplikasi pada model.

### Exploratory Data Analysis - Multivariate Analysis
Multivariate EDA menunjukkan hubungan antara dua atau lebih variabel pada data. Multivariate EDA yang menunjukkan hubungan antara dua variabel biasa disebut sebagai bivariate EDA. Selanjutnya, kita akan melakukan analisis data pada fitur kategori dan numerik.

#### Categorical Features
Untuk fitur kategori, kita akan mengamati rata - rata harga jual rumah terhadap fitur kategori. Berikut adalah salah satu visualisasi dari fitur kategori yaitu 'PavedDrive':

![data8](https://github.com/Juwono136/predictive-analytics-machine-learning-model/assets/70443393/02efc494-192d-4a8a-9c95-4b2f14ef0920)

Dengan mengamati rata - rata harga jual rumah relatif terhadap fitur kategori diatas, kita memperoleh insight yaitu bahwa beberapa fitur kategori memiliki pengaruh yang cukup tinggi terhadap harga jual rumah. Misalkan pada fitur 'PavedDrive' (jalan masuk beraspal), harga jual tertinggi dimiliki oleh tipe Y (Paved).

#### Numerical Features
Untuk mengamati hubungan antara fitur numerik, kita akan menggunakan Corellation matrix untuk melihat hubungan korelasi antar fitur. Koefisien korelasi berkisar antara -1 dan +1. Ia mengukur kekuatan hubungan antara dua variabel serta arahnya (positif atau negatif). Mengenai kekuatan hubungan antar variabel, semakin dekat nilainya ke 1 atau -1, korelasinya semakin kuat. Sedangkan, semakin dekat nilainya ke 0, korelasinya semakin lemah. Arah korelasi antara dua variabel bisa bernilai positif (nilai kedua variabel cenderung meningkat bersama-sama) maupun negatif (nilai salah satu variabel cenderung meningkat ketika nilai variabel lainnya menurun).

![data9](https://github.com/Juwono136/predictive-analytics-machine-learning-model/assets/70443393/1a4891d5-efd1-44d0-a392-1d897ec91a21)

Berdasarkan matriks korelasi diatas, jika kita amati ada beberapa fitur yang memiliki skor korelasi yang cukup besar diatas 70% dengan fitur target yaitu "SalePrice", Fitur tersebut adalah "OveralQual", "GLivArea", "GarageCars", dan "GarageArea". Sementara fitur lainnya memiliki korelasi yang kecil. Sehingga, fitur - fitur tersebut dapat di-drop.
```
house.drop(["MSSubClass", "LotArea", "OverallCond", "YearBuilt", "YearRemodAdd", "BsmtFinSF1", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "BsmtFullBath", "FullBath", "HalfBath", "BedroomAbvGr", "TotRmsAbvGrd", "Fireplaces", "WoodDeckSF", "OpenPorchSF", "MoSold", "YrSold"], inplace=True, axis=1)

house.head()
```

Kita juga harus memastikan bawa tidak ada nilai None atau NaN di dalam fitur numerik. 
```
numerical_features = house.select_dtypes(include=['float64', 'int64'])
numerical_features = numerical_features.dropna()
numerical_features
```

Setelah dilakukan proses drop dan menghilangkan nilai None/NaN, dataset kita akan tersisa 29 fitur.

![data10](https://github.com/Juwono136/predictive-analytics-machine-learning-model/assets/70443393/c6002906-3b37-4077-afbe-e8c6ac0d0b39)

## Data Preparation
Pada bagian ini, terdapat empat tahap persiapan data, yaitu:
- Encoding fitur kategori.
- Reduksi dimensi dengan Principal Component Analysis (PCA).
- Pembagian dataset dengan fungsi train_test_split dari library sklearn.
- Standarisasi.

### Encoding Fitur Kategori
Proses encoding fitur kategori menggunakan teknik one-hot-encoding. Teknik ini adalah salah satu metode dalam proses encoding fitur (feature encoding) pada data kategorikal. Tujuannya adalah untuk mengubah variabel kategorikal menjadi representasi biner yang dapat digunakan dalam algoritma pembelajaran mesin. Kita memiliki beberapa variabel kategori. Kita bisa lakukan proses encoding ini dengan fitur get_dummies. Dan menghasilkan dataset sebagai berikut:

![data11](https://github.com/Juwono136/predictive-analytics-machine-learning-model/assets/70443393/a9741226-b2cc-41f1-a12c-7643fe523152)

### Reduksi Dimensi dengan PCA
Teknik reduksi (pengurangan) dimensi adalah prosedur yang mengurangi jumlah fitur dengan tetap mempertahankan informasi pada data. Teknik pengurangan dimensi yang digunakan pada proyek ini adalah PCA. PCA adalah teknik untuk mereduksi dimensi, mengekstraksi fitur, dan mentransformasi data dari “n-dimensional space” ke dalam sistem berkoordinat baru dengan dimensi m, di mana m lebih kecil dari n.

PCA bekerja menggunakan metode aljabar linier. Ia mengasumsikan bahwa sekumpulan data pada arah dengan varians terbesar merupakan yang paling penting (utama). PCA umumnya digunakan ketika variabel dalam data memiliki korelasi yang tinggi. Korelasi tinggi ini menunjukkan data yang berulang atau redundant. Karena hal inilah, teknik PCA digunakan untuk mereduksi variabel asli menjadi sejumlah kecil variabel baru yang tidak berkorelasi linier, disebut komponen utama (PC). Komponen utama ini dapat menangkap sebagian besar varians dalam variabel asli. Sehingga, saat teknik PCA diterapkan pada data, ia hanya akan menggunakan komponen utama dan mengabaikan sisanya.

Proses analisis fitur numerik menggunakan pairplot. Dari hasil pairplot terdapat beberapa fitur yang akan dilakukan proses reduksi. Fitur GrLivArea dan GarageArea memiliki korelasi yang cukup tinggi. Hal ini terjadi karena beberapa fitur tersebut mengandung informasi yang sama yaitu area/luas. Selanjutnya kita aplikasikan class [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) dari library scikit learn dengan kode berikut:
```
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=123)
pca.fit(house[['GrLivArea','GarageArea']])
princ_comp = pca.fit_transform(house[['GrLivArea','GarageArea']])
```

Kode di atas memanggil class PCA() dari library scikit-learn. Paremeter yang kita masukkan ke dalam class adalah n_components dan random_state. Parameter n_components merupakan jumlah komponen atau dimensi, dalam kasus kita jumlahnya ada 2, yaitu 'GrLivArea' dan 'GarageArea'.

Sedangkan, parameter random_state berfungsi untuk mengontrol random number generator yang digunakan. Parameter ini berupa bilangan integer dan nilainya bebas. Pada kasus ini, kita menerapkan random_state = 123. Berapa pun nilai integer yang kita tentukan (selama itu bilangan integer), ia akan memberikan hasil yang sama setiap kali dilakukan pemanggilan fungsi (dalam kasus kita, class PCA).

Setelah class PCA dibuat, kita bisa mengetahui proporsi informasi dari kedua komponen tersebut.
```
pca.explained_variance_ratio_.round(3)

Output:
array([0.873, 0.127])
```

Arti dari output diatas adalah, 87,3% informasi pada kedua fitur 'GrLivArea' dan 'GarageArea' terdapat pada Principal Component (PC) pertama. Sedangkan sisanya, sebesar 1,27% terdapat pada PC kedua. Berdasarkan hasil ini, kita akan mereduksi fitur (dimensi) dan hanya mempertahankan PC (komponen) pertama saja. PC pertama ini akan menjadi fitur dimensi atau ukuran area menggantikan fitur 'GrLivArea' dan 'GarageArea'. Kita beri nama fitur ini 'dimension'.
```
pca = PCA(n_components=1, random_state=123)
pca.fit(house[['GrLivArea','GarageArea']])
house['dimension'] = pca.fit_transform(house.loc[:, ('GrLivArea','GarageArea')]).flatten()
house.drop(['GrLivArea','GarageArea'], axis=1, inplace=True)
```

Setelah dilakukan proses PCA, maka akan terdapat fitur baru bernama 'dimension' yang merupakan pengurangan dimensi dari fitur 'GrLivArea' dan 'GarageArea'.

![data12](https://github.com/Juwono136/predictive-analytics-machine-learning-model/assets/70443393/7a4d8f1d-b7fa-46d6-a2a3-78ad2efaefc6)

### Train-Test-Split
Selanjutnya adalah membagi dataser menjadi data latih (train) dan data uji (test). Proses pembagian dataset menggunakan library sklearn yaitu train-test-split. Proporsi pembagian adalah 80:20. Tidak lupa juga kita akan memisahkan fitur dengan target (label) yaitu SalePrice.
```
from sklearn.model_selection import train_test_split
 
X = house.drop(["SalePrice"],axis =1)
y = house["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
```

### Standarisasi
Algoritma machine learning memiliki performa lebih baik dan konvergen lebih cepat ketika dimodelkan pada data dengan skala relatif sama atau mendekati distribusi normal. Proses scaling dan standarisasi membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. 

Standardisasi adalah teknik transformasi yang paling umum digunakan dalam tahap persiapan pemodelan. Untuk fitur numerik, kita tidak akan melakukan transformasi dengan one-hot-encoding seperti pada fitur kategori. Kita akan menggunakan teknik StandarScaler dari library Scikitlearn, StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi. StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1.

Untuk menghindari kebocoran informasi pada data uji, kita hanya akan menerapkan fitur standarisasi pada data latih. Kemudian, pada tahap evaluasi, kita akan melakukan standarisasi pada data uji.

![data13](https://github.com/Juwono136/predictive-analytics-machine-learning-model/assets/70443393/61f07131-a214-4ca5-985d-ecb083f384f6)

## Modeling
Pada tahap ini, kita akan mengembangkan model machine learning dengan tiga algoritma. Kemudian, kita akan mengevaluasi performa masing-masing algoritma dan menentukan algoritma mana yang memberikan hasil prediksi terbaik. Ketiga algoritma yang akan kita gunakan, antara lain:
- K-Nearest Neighbor
- Random Forest
- Boosting Algorithm

Untuk proses mencari nilai parameter terbaik atau hyperparameter tunning, kita akan menggunakan metode [GridSearch](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) dari Library SkLearn. GridSearch akan menentukan hyperparameter terbaik berdasarkan best score untuk nanti digunakan pada proses pelatihan model machine learning. Dengan menggunakan GridSearch akan menghemat dalam proses analisis dan pencarian parameter untuk di-tune dalam model machine learning.

### K-Nearest Neighbor
[KNN](https://www.ibm.com/topics/knn#:~:text=Next%20steps-,K-Nearest%20Neighbors%20Algorithm,of%20an%20individual%20data%20point.) adalah algoritma yang relatif sederhana dibandingkan dengan algoritma lain. Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan. KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif), itulah mengapa algoritma ini dinamakan K-nearest neighbor (sejumlah k tetangga terdekat). KNN bisa digunakan untuk kasus klasifikasi dan regresi. Walaupun begitu KNN memiliki kelebihan dan kekurangan sebagai berikut:

Kelebihan KNN adalah:
- Sederhana dan Mudah Dipahami: Konsep K-NN relatif mudah untuk dipahami dan diimplementasikan. Ini adalah salah satu algoritma pembelajaran mesin yang paling sederhana.
- Cocok untuk Klasifikasi Non-Linier: K-NN bisa sangat efektif untuk masalah klasifikasi yang tidak memiliki batas keputusan linier yang jelas. Algoritma ini mampu menangani relasi non-linier antara fitur.
- Cocok untuk Dataset dengan Banyak Fitur: K-NN dapat berfungsi dengan baik bahkan pada dataset dengan banyak fitur, asalkan jumlah tetangga (k) dipilih dengan benar.

Kelemahan KNN adalah:
- Sensitif terhadap Pemilihan Jumlah Tetangga (k): K-NN sangat sensitif terhadap nilai k yang dipilih. Jika k terlalu kecil, model akan menjadi sensitif terhadap noise dan outlier. Jika k terlalu besar, model dapat kehilangan kemampuan untuk memahami struktur lokal dari data.
- Membutuhkan Memori Lebih Banyak: Algoritma K-NN memerlukan penyimpanan seluruh dataset latih di memori. Untuk dataset besar, ini dapat menghabiskan banyak memori.
- Komputasi yang Tinggi pada Pengujian: Untuk memprediksi label atau nilai untuk setiap sampel baru, algoritma K-NN harus menghitung jarak dari sampel baru ke semua sampel dalam set pelatihan, yang dapat memakan waktu jika dataset besar.
- Tidak Cocok untuk Data Berkasatria Tinggi (High-Dimensional Data): Ketika jumlah fitur sangat besar, ruang berkasatria menjadi sangat penuh dan mengukur jarak antara tetangga mungkin kehilangan makna.

Sebelum menggunakan ketiga algoritma tersebut, terlebih dahulu kita siapkan data frame untuk analisis data.
```
# Siapkan dataframe untuk analisis model
models = pd.DataFrame(index=['train_mse', 'test_mse'], 
                      columns=['KNN', 'RandomForest', 'Boosting'])
```

Berikut adalah kode untuk membuat model menggunakan algoritma KNN:
```
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)
models.loc['train_mse','knn'] = mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train)
```

Parameter n_neighbor adalah nilai k tetangga yang digunakan dalam model. Pemilihan nilai k sangat penting dan berpengaruh terhadap performa model. Jika kita memilih k yang terlalu rendah, maka akan menghasilkan model yang overfit dan hasil prediksinya memiliki varians tinggi. Jika kita memilih k terlalu tinggi, maka model yang dihasilkan akan underfit dan prediksinya memiliki bias yang tinggi. Namun, kita dapat mencoba beberapa nilai k yang berbeda, misal: nilai dari 1 hingga 20, kemudian membandingkan mana nilai yang paling sesuai untuk model. 

### Random Forest
Random forest merupakan salah satu model machine learning yang termasuk ke dalam kategori ensemble (group) learning, Ia merupakan model prediksi yang terdiri dari beberapa model dan bekerja secara bersama-sama. Ide dibalik model ensemble adalah sekelompok model yang bekerja bersama menyelesaikan masalah. Sehingga, tingkat keberhasilan akan lebih tinggi dibanding model yang bekerja sendirian. Pada model ensemble, setiap model harus membuat prediksi secara independen. Kemudian, prediksi dari setiap model ensemble ini digabungkan untuk membuat prediksi akhir. Disebut random forest karena algoritma ini disusun dari banyak algoritma pohon (decision tree) yang pembagian data dan fiturnya dipilih secara acak.

Kelebihan Random Forest:
- Akurasi Tinggi: Random Forest adalah salah satu algoritma yang memiliki akurasi tinggi dalam masalah klasifikasi dan regresi. Karena menggabungkan banyak pohon keputusan, cenderung mengurangi overfitting.
- Tidak Sensitif terhadap Outlier dan Data Missing: Random Forest bisa menangani data yang tidak seimbang dan fitur yang hilang (missing values) tanpa memerlukan pre-processing yang ekstensif.
- Bisa Mengatasi Data Berkasatria Tinggi (High-Dimensional Data): Algoritma ini bekerja dengan baik pada data yang memiliki banyak fitur.
- Mampu Menangani Variabel Numerik dan Kategorikal: Random Forest dapat menangani baik variabel numerik maupun kategorikal tanpa memerlukan transformasi tambahan.

Kekurangan Random Forest:
- Kesulitan dalam Interpretasi Model: Random Forest adalah model ensemble kompleks, yang bisa sulit untuk diinterpretasi dan menjelaskan mengapa keputusan spesifik dibuat.
- Membutuhkan Memori Lebih Banyak: Karena Random Forest menggabungkan beberapa pohon keputusan, ia memerlukan lebih banyak memori daripada model tunggal.
- Kurang Cepat dalam Proses Prediksi: Proses prediksi dengan Random Forest mungkin lebih lambat daripada model tunggal seperti pohon keputusan karena harus menggabungkan hasil dari beberapa pohon.

Berikut adalah kode untuk membuat model menggunakan algoritma Random Forest:
```
from sklearn.ensemble import RandomForestRegressor

# buat model prediksi
RF = RandomForestRegressor(n_estimators=60, max_depth=32, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)
models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)
```

Berdasarkan kode diatas, kita mengimpor RandomForestRegressor dari library scikit-learn dan mengimpor mean_squared_error sebagai metrik untuk mengevaluasi performa model. Kemudian dibuat juga variabel RF dan memanggil RandomForestRegressor dengan beberapa nilai parameter. Berikut adalah parameter-parameter yang digunakan:
- n_estimator: jumlah trees (pohon) di forest. Di sini kita set n_estimator=60 yang didapat dari GridSearch.
- max_depth: kedalaman atau panjang pohon. Ia merupakan ukuran seberapa banyak pohon dapat membelah (splitting) untuk membagi setiap node ke dalam jumlah pengamatan yang diinginkan.
- random_state: digunakan untuk mengontrol random number generator yang digunakan. 
- n_jobs: jumlah job (pekerjaan) yang digunakan secara paralel. Ia merupakan komponen untuk mengontrol thread atau proses yang berjalan secara paralel. n_jobs=-1 artinya semua proses berjalan secara paralel.

### Boosting Algorithm
Boosting Algorithm adalah metode pembelajaran mesin ensemble yang berusaha meningkatkan kinerja model dengan menggabungkan sejumlah kecil model lemah (biasanya pohon keputusan dangkal atau pengklasifikasi lemah lainnya) menjadi model yang kuat. Secara umum, algoritma boosting bekerja dengan cara memberikan bobot yang berbeda pada setiap sampel dalam dataset sehingga model berfokus pada sampel yang sulit diprediksi oleh model sebelumnya.

Ada beberapa jenis algoritma boosting yang populer, tapi pada proyek ini yang digunakan adalah AdaBoost (Adaptive Boosting). AdaBoost menggunakan model lemah dan menyesuaikan bobot pada setiap sampel, memberikan lebih banyak fokus pada sampel yang salah diklasifikasikan sebelumnya.

Kelebihan algoritma boosting:
- Akurasi Tinggi: Boosting sering menghasilkan model yang memiliki akurasi yang sangat tinggi, karena mampu mengurangi bias dan varians.
- Mampu Menangani Data yang Tidak Seimbang: Boosting dapat menangani masalah klasifikasi dengan dataset yang tidak seimbang dengan baik, karena memberi bobot lebih pada sampel dari kelas yang kurang umum.
- Tidak Sensitif terhadap Data Outlier: Boosting memiliki kekebalan terhadap outlier, karena fokus pada sampel yang sulit diprediksi oleh model sebelumnya.
- Mampu Menangani Variabel Kategorikal dan Numerik: Banyak implementasi boosting dapat menangani baik variabel kategorikal maupun numerik tanpa memerlukan pre-processing tambahan.

Kelemahan algoritma boosting:
- Memerlukan Waktu Komputasi yang Lebih Lama: Training boosting algorithms mungkin memerlukan lebih banyak waktu dan sumber daya komputasi dibandingkan dengan beberapa algoritma pembelajaran mesin lainnya.
- Overfitting Jika Tidak Dikontrol: Ada kemungkinan overfitting jika parameter tidak diatur dengan benar atau jika terlalu banyak pohon digunakan dalam ensemble.
- Rentan terhadap Noise: Boosting bisa sangat sensitif terhadap noise dalam data latih.

Berikut adalah kode untuk membuat model menggunakan algoritma Boosting (AdaBoost):
```
from sklearn.ensemble import AdaBoostRegressor
boosting = AdaBoostRegressor(learning_rate=0.1, random_state=55) 
boosting.fit(X_train, y_train)
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train)
```

Berikut merupakan parameter-parameter yang digunakan pada potongan kode di atas:
- learning_rate: bobot yang diterapkan pada setiap regressor di masing-masing proses iterasi boosting.
- random_state: digunakan untuk mengontrol random number generator yang digunakan.

## Evaluation
Metrik yang digunakan pada proyek ini untuk melakukan evaluasi model adalah MSE atau [Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error) yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. Jika prediksi mendekati nilai sebenarnya, performanya baik. Sedangkan jika tidak, performanya buruk. Secara teknis, selisih antara nilai sebenarnya dan nilai prediksi disebut eror. Maka, semua metrik mengukur seberapa kecil nilai eror tersebut. MSE didefinisikan dalam persamaan berikut:

![data14](https://github.com/Juwono136/predictive-analytics-machine-learning-model/assets/70443393/f111e139-e7cb-49fa-899d-c327f108f7a0)

Keterangan:
- N = jumlah dataset
- yi = nilai sebenarnya
- y_pred = nilai prediksi

Sebelum menghitung nilai MSE, kita perlu melakukan proses scaling fitur numerik pada data uji. Karena sebelumnya , kita baru melakukan proses scaling hanya pada data latih saja. Setelah model dilatih menggunakan 3 jenis algoritma yaitu KNN, Random Forest dan AdaBoost, kita harus melakukan scaling fitur pada data uji. Hal ini harus dilakukan agar skala antara data latih dan data uji sama dan kita bisa melakukan evaluasi.
```
X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])
```

Selanjutnya, kita akan mengevaluasi ketiga model dengan metrik MSE. Saat menghitung nilai MSE pada data train dan test kita akan membaginya dengan 1e6, hal ini bertujuan agar nilai mse tidak terlalu besar skalanya. Sehingga didapat nilai MSE sebagai berikut:

![data15](https://github.com/Juwono136/predictive-analytics-machine-learning-model/assets/70443393/65109eff-8ef9-4a8b-a108-90df3d462469)

Berdasarkan dataframe, terlihat bahwa, model Random Forest (RF) memberikan skor nilai error paling kecil dibandingkan algoritma lain seperti KNN dan AdaBoost. Jadi, Model Random Forest yang akan dipilih sebagai model terbaik untuk memprediksi harga jual rumah. Untuk mengujinya, kita buat prediksi menggunakan beberapa harga dari data test dan didapatkan output atau hasil sebagai berikut:

![data16](https://github.com/Juwono136/predictive-analytics-machine-learning-model/assets/70443393/ad185bb4-20ad-4974-aef2-8f7a719771c2)

Terlihat bahwa prediksi Random Forest (RF) memberikan hasil yang paling mendekati dengan y_true (data test). Dimana nilai y_true adalah 233000 sedangkan nilai prediksi dari Random Forest adalah 234851.5.

--- END ---
