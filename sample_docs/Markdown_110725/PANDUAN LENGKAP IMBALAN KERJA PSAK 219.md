# Pendahuluan

Buku panduan ini hadir sebagai cetakan kedua dengan pembaruan yang lebih mendalam untuk membantu perusahaan memahami dan mengimplementasikan #PSAK-219 secara efektif dan komprehensif, khususnya dalam #Perhitungan-Aktuaria #Imbalan-pasca-kerja. Dengan mempertimbangkan dinamika ekonomi dan perubahan regulasi, panduan ini menawarkan wawasan strategis yang lebih relevan untuk menjawab tantangan terkini.

#PSAK-219 tetap menjadi standar penting dalam pengakuan dan pengukuran kewajiban imbalan kerja. Namun, dalam praktiknya, banyak perusahaan menghadapi hambatan seperti perubahan [[Asumsi]], tekanan audit, dan dampak inflasi. Oleh karena itu, buku ini tidak hanya membahas aspek teknis aktuaria, tetapi juga pendekatan inovatif untuk mitigasi risiko dan strategi perencanaan jangka panjang yang adaptif.

Dalam edisi ini, Anda akan menemukan analisis teknis, studi kasus implementasi, hingga eksplorasi teknologi seperti otomasi dan integrasi data untuk mendukung akurasi perhitungan. Harapannya, panduan ini dapat menjadi referensi utama bagi profesional keuangan dan SDM dalam mengelola kewajiban #imbalan-kerja secara akurat, efisien, dan siap audit.

Dengan berbagai pembaruan dan tambahan materi, kami berharap panduan ini dapat menjadi referensi utama bagi perusahaan, akuntan, auditor, dan profesional HR dalam mengelola kewajiban #Imbalan-pasca-kerja secara lebih akurat dan strategis.

Selamat membaca, semoga panduan ini memberikan manfaat dan nilai tambah bagi perusahaan Anda.

---
# BAB 1 Dasar #Imbalan-pasca-kerja #PSAK-219
## Ilustrasi #Imbalan-pasca-kerja

Untuk memudahkan pemahaman tentang  #Imbalan-pasca-kerja, mari amati situasi ini:

**Janji Rudi**
*Rudi berencana mengajak istri dan anak-anaknya berlibur ke Swiss 5 tahun dari sekarang, yaitu di tahun 2030. Rudi mengestimasi bahwa biaya yang dibutuhkan sebesar Rp100.000.000. Rudi tentunya mulai menabung dari sekarang dan juga menginvestasikannya ke dalam deposito. Dengan target hasil investasi sebesar 5%.*

Sama halnya dengan menghitung cicilan KPR, Rudi perlu menabung Rp17.235.695 per tahun. Jika semuanya berjalan sesuai dengan rencana dan estimasi perhitungan seperti berikut.

| Tahun | Saldo Awal Tahun | Tabungan per Tahun | Hasil Investasi | Saldo Akhir Tahun |
| ----- | ---------------- | ------------------ | --------------- | ----------------- |
| 1     | -                | 17.235.695         | 861.785         | 18.097.480        |
| 2     | 18.097.480       | 17.235.695         | 1.766.659       | 37.099.833        |
| 3     | 37.099.833       | 17.235.695         | 2.716.776       | 57.052.305        |
| 4     | 57.052.305       | 17.235.695         | 3.714.400       | 78.002.400        |
| 5     | 78.002.400       | 17.235.695         | 4.761.905       | **100.000.000**   |
Setiap tahun, Rudi menyisihkan sejumlah tabungan dan ingin mendapat hasil investasi agar dananya berkembang. Jika Rudi tidak menempatkan tabungannya di deposito dan lebih memilih menyimpannya sendiri, maka ia perlu menambahkan dana tabungan tahunan beserta potensi hasil investasinya.

*Lalu, bagaimana jika bunga deposito berubah di tahun ketiga, misalnya menjadi 6%? Beginilah ilustrasi besaran tabungan Rudi per tahunnya dalam kondisi tersebut.*

| Tahun | Bunga  | Saldo Awal Tahun | Tabungan per Tahun | Hasil Investasi | Saldo Akhir Tahun |
| ----- | ------ | ---------------- | ------------------ | --------------- | ----------------- |
| 1     | 5%     | -                | 17.235.695         | 861.785         | 18.097.480        |
| 2     | 5%     | 18.097.480       | 17.235.695         | 1.766.659       | 37.099.833        |
| 3     | **6%** | 37.099.833       | **16.539.216**     | 3.218.343       | 56.857.393        |
| 4     | **6%** | 56.857.393       | **16.539.216**     | 4.403.797       | 77.800.406        |
| 5     | **6%** | 77.800.406       | **16.539.216**     | 5.660.377       | **100.000.000**   |

---

Akibat kenaikan bunga ini, **Rudi untung** karena beban tabungan yang harus disetorkan setiap tahun berkurang. Namun, keadaan pasar bisa saja berbalik. Misalkan, ternyata di tahun keempat dan kelima, target investasi turun hingga ke angka 4%, sehingga beban tabungan kembali naik dan kondisi **Rudi rugi.**

| Tahun | Bunga  | Saldo Awal Tahun | Tabungan per Tahun | Hasil Investasi | Saldo Akhir Tahun |
| ----- | ------ | ---------------- | ------------------ | --------------- | ----------------- |
| 1     | 5%     | -                | 17.235.695         | 861.785         | 18.097.480        |
| 2     | 5%     | 18.097.480       | 17.235.695         | 1.766.659       | 37.099.833        |
| 3     | **6%** | 37.099.833       | **16.539.216**         | 3.218.343       | 56.857.393        |
| 4     | **4%** | 56.857.393       | **18.148.116**         | 3.000.220       | 78.005.729        |
| 5     | **4%** | 78.005.729       | **18.148.116**         | 3.846.154       | **100.000.000**   |

*Selama proses menabung, Rudi bisa saja berada dalam posisi “untung” atau “rugi” ketika mengumpulkan dana liburan.*

Gambaran ini mirip dengan janji perusahaan kepada karyawannya—janji manfaat di masa depan dengan jumlah dan waktu yang pasti, dengan analogi berikut.

### Ilustrasi Dana #Imbalan-pasca-kerja – Komponen Biaya

| Tahun | Bunga | Saldo Awal Tahun | **Tabungan per Tahun (Service Cost)** | **Hasil Investasi (Interest Cost)** | Saldo Akhir Tahun |
|-------|-------|------------------|----------------------------------------|--------------------------------------|--------------------|
| 1     | 5%    | -                | 17.235.695                             | 861.785                              | 18.097.480         |
| 2     | 5%    | 18.097.480       | 17.235.695                             | 1.766.659                            | 37.099.833         |
| 3     | 6%    | 37.099.833       | 16.539.216                             | 3.218.343                            | **56.857.393**     |
| 4     | 4%    | 56.857.393       | 18.148.116                             | 3.000.220                            | **78.005.729**     |
| 5     | 4%    | 78.005.729       | 18.148.116                             | 3.846.154                            | **100.000.000**    |
Penjelasan Komponen:
- **Service Cost** = *Tabungan per tahun*: mencerminkan kewajiban tahunan terhadap pegawai.
- **Interest Cost** = *Hasil Investasi*: estimasi hasil dari asumsi bunga atas saldo awal tahun.
- **Gain/Loss** terjadi saat:
	1. Ada **perubahan asumsi** (misalnya: tingkat bunga berubah dari 5% ke 6%, lalu ke 4%).
	2. Ada **penyesuaian jumlah tabungan** untuk mencapai target akhir.

Namun, berbeda dari janji Rudi yang **tidak memiliki standar** atau peraturan akuntansi dan **tidak ada asumsi** tentang tabungan tadi. Sementara, manfaat karyawan dari perusahaan memiliki peraturan atau standar nasional maupun internasional yang harus dipatuhi.

Perusahaan juga menggunakan #Perhitungan-Aktuaria yang sangat mempertimbangkan berbagai faktor, seperti asumsi dan metode aktuaria. Dimulai dari pemahaman dasar, proses #Perhitungan-Aktuaria hingga penyajian di #laporan-keuangan, akan dibahas dalam buku panduan ini secara menyeluruh.

---
### Pengertian #Imbalan-pasca-kerja

#Imbalan-pasca-kerja adalah sejumlah manfaat yang disediakan oleh perusahaan kepada karyawan setelah mereka menyelesaikan masa kerjanya. Ini adalah bentuk apresiasi atas kontribusi yang telah diberikan selama bertahun-tahun. Imbalan ini tidak hanya penting bagi karyawan seperti Rudi, tetapi juga bagi perusahaan dalam menjaga reputasi dan komitmennya terhadap kesejahteraan karyawan.

#Imbalan-pasca-kerja mencakup berbagai bentuk manfaat seperti:
##### Uang #Pesangon 
Sejumlah uang yang diberikan saat pengakhiran hubungan kerja, baik karena pensiun / pemutusan hubungan kerja.
#### Uang Pisah  
Uang penghargaan kontribusi kepada karyawan yang mengundurkan diri secara sukarela.
#### Uang Penghargaan Masa Kerja 
Imbalan yang diberikan kepada karyawan dengan masa kerja tertentu di perusahaan.
#### Uang Penggantian Hak 
Kompensasi pengganti hak-hak yang tidak diambil oleh karyawan selama masa kerja, seperti cuti yang belum diambil.
#### Uang Kompensasi / Uang Duka 
Uang bantuan sebagai ganti rugi atau imbalan atas kondisi tertentu / diberikan kepada ahli waris karyawan yang meninggal dunia.

#### Kategori Utama #Imbalan-pasca-kerja

Pengakuan, pengukuran, dan pengungkapan dalam #Imbalan-pasca-kerja terbagi menjadi dua kategori utama:
##### 1. Iuran Pasti (Defined Contribution Plan)
- Perusahaan hanya wajib menyetor sejumlah iuran tertentu ke #Dana-Pensiun.
- Setelah iuran disetor, risiko dan tanggung jawab pindah ke karyawan.

**Pengakuan dan Pengukuran:**
	1. Diakui sebagai **biaya/liabilitas**, saat karyawan memberikan jasanya.
	2. Jika kelebihan dibayar sebelum jatuh tempo, diakui sebagai Aset (*dibayar di muka*).
**Pengungkapan:**
	3. Jumlah iuran yang dibayar wajib diungkapkan dalam #laporan-keuangan.

##### 2. Imbalan Pasti (Defined Benefit Plan)
- Perusahaan bertanggung jawab atas jumlah Manfaat pensiun [[Asumsi#Program-Manfaat]] tertentu yang akan diterima karyawan, terlepas dari hasil investasi #Dana-Pensiun.

**Pengakuan dan Pengukuran:**
	1. Biaya jasa dicatat di #Laporan-Laba-Rugi.
	2. Biaya neto atas liabilitas/aset imbalan pasti juga dicatat di #Laporan-Laba-Rugi.
	3. Selisih pengukuran (*remeasurement*) masuk ke penghasilan komprehensif lain (**OCI**).
**Pengungkapan:**
	4. Perusahaan diwajibkan mengungkapkan:
		- Karakteristik dan risiko program.
		- Jumlah manfaat yang timbul dalam #laporan-keuangan.
		- Bagaimana dampak program terhadap arus kas masa depan.

---
## Standar #PSAK-24

#PSAK-24 merupakan **Standar Akuntansi Keuangan (SAK)** yang memberikan pedoman kepada perusahaan dalam mencatat, mengukur, dan melaporkan imbalan kerja yang diberikan kepada karyawan. Standar ini mencakup imbalan yang disediakan langsung oleh perusahaan maupun melalui pihak ketiga.

1. **Diperbarui pada 27 Agustus 2014 oleh Dewan Standar Akuntansi Keuangan (DSAK IAI)** agar selaras dengan praktik akuntansi global.
2. **Berlaku untuk semua jenis** #imbalan-kerja , kecuali imbalan berbasis saham (PSAK 53).
3. **Sudah diterapkan oleh berbagai entitas** (perusahaan publik, asuransi, perbankan, BUMN, dan #Dana-Pensiun), khususnya yang terdaftar atau akan mendaftar di pasar modal.

Standar ini mengatur pencatatan beban, dengan mewajibkan perusahaan mengakui liabilitas atas imbalan kerja ketika karyawan telah memberikan jasa dan berhak atas imbalan di masa depan.
Tidak hanya memengaruhi pembukuan keuangan, tetapi juga memengaruhi manajemen sumber daya manusia, kepatuhan hukum, dan stabilitas keuangan perusahaan.
#### Dampak #PSAK-24 terhadap Fungsi-Fungsi Perusahaan

##### Human Resources
1. Efektivitas manajemen #imbalan-kerja 
2. Perencanaan keuangan jangka panjang
3. Retensi dan kepuasan bagi karyawan
##### Legal
1. Kepatuhan regulasi pe#laporan-keuangan
2. Konsistensi dan akurasi audit keuangan
3. Perlindungan hak kontrak karyawan
##### Finance & Accounting
1. Prinsip akuntansi *accrual basis*
2. Pengungkapan kewajiban akrual
3. Stabilitas laba dan arus kas perusahaan

Selama bertahun-tahun, #PSAK-24 telah menjadi standar yang mengatur pengakuan, pengukuran, dan pelaporan imbalan kerja di Indonesia. Namun, dalam dunia yang terus berkembang dan terintegrasi secara global, standar akuntansi yang berlaku di Indonesia pun harus terus beradaptasi agar tetap relevan dan akurat.

---
## Transisi Nomenklatur ke #PSAK-219

Langkah ini mencerminkan komitmen tentang transparansi dan konsistensi dalam pe#laporan-keuangan, khususnya dalam hal imbalan kerja.
### Landasan Hukum Perubahan

Dengan perkembangan *International Financial Reporting Standards* (IFRS) dan perubahan dalam dunia bisnis global, nomenklatur #PSAK-24 telah **diubah menjadi #PSAK-219**, berlaku **efektif sejak 1 Januari 2025**. Perubahan ini bertujuan memberikan panduan yang lebih komprehensif sesuai kemajuan ilmu akuntansi dan praktik akuntansi global.

Jadi, perusahaan diharapkan dapat menyesuaikan diri dengan panduan #PSAK-219 terkini untuk mengatasi keterbatasan yang ada dalam #PSAK-24, khususnya dalam penyesuaian [[Asumsi]] aktuaria  dan pengukuran kewajiban dalam penyusunan laporan dan audit keuangan.

Di Indonesia, #PSAK-219 (Imbalan Kerja) tidak berdiri sendiri, melainkan terhubung dengan berbagai standar akuntansi lain dalam kerangka pe#laporan-keuangan. Beberapa di antaranya adalah:
#### PSAK 1 (Penyajian #laporan-keuangan)
Dikaitkan ke #PSAK-219, kewajiban imbalan kerja diakui sebagai bagian dari liabilitas, dan harus diungkapkan secara jelas dalam #laporan-keuangan.
#### PSAK 2 (Laporan Arus Kas)
#PSAK-219 memengaruhi laporan arus kas, khususnya terkait pengeluaran dana untuk pembayaran imbalan kerja. Ini harus diklasifikasikan sesuai dengan ketentuan PSAK 2.
#### PSAK 5 (Segmen Operasi)
Dalam PSAK 5, imbalan kerja yang dihitung berdasarkan #PSAK-219 mungkin perlu disegmentasi dalam #laporan-keuangan untuk mencerminkan kinerja segmen yang berbeda, memungkinkan pemangku kepentingan memahami dampaknya dalam konteks operasional.
#### PSAK 53 (Akuntansi Imbalan Kerja dengan Pembayaran Saham)
Mengatur pengakuan, pengukuran, dan pengungkapan imbalan kerja dalam bentuk pembayaran saham dan saham opsional. Meskipun jenis imbalannya berbeda, keduanya saling terkait dalam hal pengakuan total kewajiban imbalan kerja di #laporan-keuangan.

---
## Dasar Hukum #PSAK-219

Beberapa peraturan dan undang-undang yang menjadi dasar hukum penerapan #PSAK-219:
#### Undang-Undang Ketenagakerjaan Nomor 13 Tahun 2003 /  #UUK 
Mengatur hak-hak pekerja, termasuk *#Pesangon* dan jaminan pensiun, dengan pengakuan dan pengukuran imbalan kerja #PSAK-219.
#### Peraturan Pemerintah No. 78 Tahun 2015 
Mengenai pengupahan, penghitungan dan pembayaran imbalan kerja secara adil yang mendukung penerapan #PSAK-219.
#### Undang-Undang Ketenagakerjaan No. 11 Tahun 2020  
Menekankan penyesuaian perhitungan kewajiban aktuaria untuk mencegah *overcost* dan *overtaxation*, sesuai standar akuntansi.
#### Peraturan Pemerintah No. 34 Tahun 2021 
Menerangkan bahwa perusahaan dapat mengurangi beban kewajiban jangka pendeknya karena TKA (Tenaga Kerja Asing) tidak berhak mendapat kompensasi.
#### Peraturan Pemerintah No. 35 Tahun 2021  
Peraturan terkait kompensasi pekerja yang terkena PHK, perjanjian kerja waktu tertentu, dan lainnya, sejalan dengan standar ketenagakerjaan yang berlaku.
#### Undang-Undang Ketenagakerjaan No. 6 Tahun 2023  / #UUCK
Memastikan pemenuhan kewajiban aktuaria perusahaan terhadap karyawan, **dengan penekanan pada pengakuan liabilitas imbalan kerja**.

## Ruang Lingkup #PSAK-219

Jenis-jenis imbalan kerja yang dicakup oleh #PSAK-219 meliputi:
### Imbalan Kerja Jangka Pendek
Contohnya: gaji, upah, bonus, dan tunjangan lain yang dibayarkan dalam waktu kurang dari 12 bulan setelah akhir periode kerja.

### #Imbalan-pasca-kerja
Manfaat pensiun [[Asumsi#Program-Manfaat]] yang diberikan setelah karyawan menyelesaikan masa kerjanya. #PSAK-219 meminta perusahaan mengakui dan mengukur kewajiban ini dengan nilai yang mencerminkan besaran sebenarnya.
### Imbalan Jangka Panjang Lainnya
Meliputi hak cuti jangka panjang, jaminan kesehatan pasca pensiun, dan manfaat serupa yang dibayarkan lebih dari 12 bulan setelah periode kerja berakhir.
### #Pesangon Pemutusan Kerja
Mengatur kewajiban dan pengukuran #Pesangon saat karyawan berhenti bekerja, baik pemutusan secara sukarela maupun tidak sukarela.

---
Sekarang kita paham bagaimana ruang lingkup standar ini menuntut perusahaan menggunakan metode aktuaria dan perhitungan yang kompleks dalam penyusunan #laporan-keuangan.

Namun, tidak semua entitas memiliki skala dan karakteristik yang sama. Banyak perusahaan kecil dan menengah memiliki kebutuhan dan kapasitas pelaporan yang berbeda. Untuk itulah, SAK ETAP hadir sebagai alternatif standar akuntansi yang lebih sederhana.
## Standar #SAK-EP ( #SAK-ETAP)

#SAK-EP merupakan singkatan dari *Standar Akuntansi Keuangan Entitas Tanpa Akuntabilitas Publik*, yang diterbitkan oleh **Ikatan Akuntan Indonesia (IAI)**.

1. Sama halnya dengan #PSAK-219, mulai 1 Januari 2025, SAK ETAP digantikan oleh #SAK-EP.
2. Ditujukan bagi entitas privat yang tidak memiliki kewajiban akuntabilitas publik serta tidak menerbitkan #laporan-keuangan kepada publik.

#SAK-EP memberikan alternatif bagi entitas yang menginginkan pe#laporan-keuangan yang lebih komprehensif dan terstruktur dibandingkan SAK ETAP. Karena #SAK-EP disusun berbasis *IFRS for SMEs*, sehingga mempertimbangkan kebutuhan entitas privat untuk menyusun #laporan-keuangan yang andal, namun tidak serumit PSAK berbasis IFRS seperti #PSAK-219.
#### Entitas yang menggunakan #SAK-EP umumnya:
- Tidak memiliki kewajiban menyampaikan #laporan-keuangan kepada publik,
- Tidak terdaftar di bursa efek,
- Skala usaha relatif kecil atau menengah,
- Tidak memiliki kepentingan publik yang signifikan.

**Contoh entitas pengguna** #SAK-EP:
- Koperasi simpan pinjam,
- Yayasan pendidikan,
- Perusahaan keluarga skala kecil-menengah,
- CV atau firma.

Dalam #SAK-EP, kewajiban imbalan kerja tidak dihitung secara aktuaria, melainkan dicatat saat manfaat tersebut menjadi kewajiban hukum yang pasti. Tidak ada kewajiban menghitung nilai sekarang dari manfaat masa depan seperti pada #PSAK-219. Namun, banyak entitas tetap membuat estimasi internal atas kewajiban ini, terutama jika ingin memiliki proyeksi beban keuangan jangka panjang. 

Terkait pe#laporan-keuangan terdapat penambahan dampak berupa komponen #Other-Comprehensive-Income ( #OCI).

Penambahan tersebut memberikan ruang untuk mengakui perubahan nilai kewajiban imbalan kerja secara tidak langsung. Hal ini membantu entitas kecil dan menengah menyajikan #laporan-keuangan yang lebih transparan dan mencerminkan fluktuasi kewajiban jangka panjang tanpa membebani laba rugi secara langsung.

---
## Regulasi #PSAK-219 terhadap IFRS 19

#PSAK-219 diadopsi dengan mengacu pada **IFRS 19 (Employee Benefits)**, yang merupakan standar internasional mengenai pelaporan imbalan kerja. Hal ini bertujuan untuk memastikan bahwa #laporan-keuangan perusahaan di Indonesia:
- Konsisten dengan praktik internasional,
- Meningkatkan transparansi,
- Meningkatkan komparabilitas pada tingkat global.
### Keuntungan Penyesuaian #PSAK-219 terhadap IFRS 19

1. **Pengukuran Liabilitas Manfaat Lebih Akurat dan Relevan**
	Menggunakan asumsi yang lebih *realistis* dan diperbarui secara *dinamis* membantu #laporan-keuangan mencerminkan **kondisi pasar yang aktual**.
	
2. Pengungkapan yang Lebih Terperinci
	Menurut lebih banyak pengungkapan, misalnya rincian **program manfaat karyawan** #laporan-keuangan menjadi lebih **transparan** bagi investor dan pemangku kepentingan.
	
3. Manajemen Risiko yang Lebih Komprehensif
	Perusahaan juga harus mengenali dan **mengelola risiko** – seperti fluktuasi pasar / risiko kredit. Strategi investasi dan perencanaan keuangan bisa disesuaikan secara **lebih akurat**.
### Pengawasan dan Regulasi: Apa yang harus diperhatikan?

1. **Tanggung Jawab Hukum & Risiko Non-Kepatuhan**:  
	Setiap entitas yang menyusun #laporan-keuangan wajib menerapkan #PSAK-219. Ketidakpatuhan dapat memicu sanksi otoritas dan risiko litigasi.
2. **Pengawasan Regulator**:  
	 OJK dan Bapepam-LK mengawasi pelaksanaan #PSAK-219 untuk memastikan #laporan-keuangan perusahaan mencerminkan kondisi keuangan sebenarnya.
3. **Kewajiban Dokumentasi**:  
	Perusahaan harus menyimpan dokumen perhitungan kewajiban imbalan kerja ([[Asumsi]] aktuaria , metode, data karyawan) sebagai bukti kepatuhan dan untuk keperluan audit.

--- 
## Penerapan #IFRIC AD 

IFRS *Interpretation Committee,* sebelumnya dikenal sebagai #IFRIC (International Financial Reporting Interpretation Committee),** adalah sebuah lembaga yang bertugas menginterpretasikan standar akuntansi IFRS agar penerapannya konsisten di seluruh dunia. Komite ini bekerja sama dengan *International Accounting Standards Board* (IASB) untuk menjawab pertanyaan terkait standar akuntansi dan isu teknis terkait standar IFRS.

DSAK IAI menyimpulkan bahwa skema pensiun di Indonesia, yang mengikuti Undang‐Undang Ketenagakerjaan atau Undang‐Undang Cipta Kerja (UUK/UUCK), memiliki pola fakta mirip dengan yang dibahas IFRIC AD sehingga dianggap relevan satu sama lain.
### Perbandingan IFRIC AD vs UUCK 

| **IFRIC AD**                                                                                                                                                       | **UUCK**                                                                                                                                                               |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Karyawan berhak atas Manfaat pensiun [[Asumsi#Program-Manfaat]] hanya jika mencapai **#Usia-Pensiun 62 tahun** dan masih bekerja di perusahaan pada saat itu                                   | Karyawan berhak atas Manfaat pensiun [[Asumsi#Program-Manfaat]] hanya jika mencapai **#Usia-Pensiun 56 tahun** dan masih bekerja di perusahaan pada saat itu                                       |
| Manfaat pensiun [[Asumsi#Program-Manfaat]] dihitung berdasarkan **1 bulan gaji terakhir untuk setiap tahun masa kerja** sebelum #Usia-Pensiun, namun dibatasi maksimal **16 tahun** masa kerja | Manfaat pensiun [[Asumsi#Program-Manfaat]] didapat dari **penjumlahan dua komponen** — misalnya *uang #Pesangon dan uang penghargaan masa kerja*, masing-masing memiliki batas tahun kerja berbeda |
Manfaat pensiun [[Asumsi#Program-Manfaat]] dihitung hanya dengan menggunakan jumlah tahun kerja berturut-turut tepat sebelum #Usia-Pensiun
### Perbandingan Sebelum dan Sesudah Penerapan IFRIC AD

**IFRIC AD** telah menjelaskan lebih rinci tentang paragraf **70, 71, 72, dan 74** dari #PSAK-219 yang berbunyi:

> 70. *Metode #Projected-Unit-Credit (sering kali disebut sebagai metode imbalan yang diakru yang diperhitungkan secara pro rata sesuai jasa atau sebagai metode imbalan dibagi tahun jasa)* menganggap setiap periode jasa akan menghasilkan satu unit tambahan imbalan dan mengukur setiap unit secara terpisah untuk menghasilkan kewajiban final.

> 71. *Entitas mendiskontokan semua kewajiban #Imbalan-pasca-kerja, walaupun sebagian kewajiban jatuh tempo dalam jangka waktu 12 (dua belas bulan) bulan setelah periode pelaporan.*

> 72. *Dalam menentukan nilai kini kewajiban imbalan pasti dan #Biaya-Jasa-Kini yang terkait dan #Biaya-Jasa-Lalu (jika dapat diterapkan) entitas mengalokasikan imbalan sepanjang periode jasa dengan menggunakan formula imbalan yang dimiliki program. Namun, jika jasa pekerja di tahun-tahun akhir meningkat secara material dibandingkan dengan tahun-tahun sebelumnya, maka entitas mengalokasikan imbalan tersebut dengan dasar metode garis lurus, sejak:
> 	(a) saat jasa pekerja pertama kali menghasilkan imbalan dalam program (baik imbalan tersebut bergantung pada jasa selanjutnya atau tidak); sampai dengan*
> 	*(b) saat jasa pekerja selanjutnya tidak menghasilkan imbalan yang material dalam program, selain dari #Kenaikan-Gaji berikutnya.*

> 74. *Dalam program imbalan pasti jasa pekerja akan menimbulkan kewajiban, walaupun imbalan itu bergantung pada status bekerjanya di masa depan (dengan kata lain tidak vested). Jasa pekerja sebelum tanggal vesting menimbulkan kewajiban konstruktif karena, pada setiap akhir periode pelaporan yang berurutan, jumlah jasa di masa depan yang harus diberikan pekerja sebelum pekerja berhak atas imbalan tersebut menjadi berkurang. Dalam mengukur kewajiban imbalan pasti, entitas memperhitungkan kemungkinan bahwa beberapa pekerja tidak akan memenuhi ketentuan vesting. Sama halnya, walaupun #Imbalan-pasca-kerja tertentu, sebagai contoh jaminan kesehatan pasca kerja, terutang hanya jika peristiwa tertentu terjadi pada saat pekerja tidak lagi bekerja, namun kewajiban muncul pada saat pekerja memberikan jasa yang menimbulkan hak atas imbalan jika peristiwa tertentu tersebut terjadi. Kemungkinan bahwa peristiwa tertentu akan terjadi berpengaruh terhadap pengukuran kewajiban, namun tidak menentukan apakah kewajiban tersebut ada.

| **Sebelum IFRIC AD**                                                                                                                                                                                   | **Sesudah IFRIC AD**                                                                                                                                                                                                                |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Kewajiban pasca kerja dianggap timbul **sejak karyawan mulai bekerja**                                                                                                                                 | Kewajiban pasca kerja baru timbul ketika sisa masa kerja karyawan, yaitu 24 tahun sebelum #Usia-Pensiun nya                                                                                                                         |
| *Contoh*: Jika perusahaan A menetapkan #Usia-Pensiun 56 tahun dan Rudi mulai bekerja pada usia 25, perusahaan langsung mulai menghitung dan mencatat kewajiban untuk Rudi **sejak ia berusia 25 tahun** | *Contoh*: Jika perusahaan A menetapkan #Usia-Pensiun 56 tahun dan Rudi mulai bekerja di perusahaan A ketika umur 25 tahun, perusahaan baru mulai menghitung dan mencatat kewajiban untuk Rudi **ketika usianya 32 tahun** (56 – 24) |

---
## Kebutuhan #Valuasi-Aktuaria

Ruang lingkup #PSAK-219 yang memerlukan perhitungan #Valuasi-Aktuaria adalah:
###  #Imbalan-pasca-kerja
Yang dihitung antara lain:
	- #Pesangon untuk karyawan pensiun
	- #Pesangon untuk karyawan meninggal dunia
	- #Pesangon untuk karyawan sakit berkepanjangan / cacat
	- #Pesangon untuk karyawan mengundurkan diri

Referensi: [[Perhitungan Tetap]]
### Imbalan Jangka Panjang Lainnya
Atau Other Long-Term Employee Benefits ( #OLTEB), bila perusahaan menjanjikan manfaat penghargaan di luar UU maka perusahaan perlu melakukan perhitungan #Valuasi-Aktuaria.

**Contoh Imbalan Jangka Panjang Lainnya adalah:**
- **Manfaat Cuti Besar** ( #CBS) / Penghargaan Masa Kerja yang dapat diuangkan
- **Manfaat Penghargaan Emas**

>  *Jadi, #Valuasi-Aktuaria adalah proses perhitungan dan analisis yang menentukan besarnya kewajiban #Imbalan-pasca-kerja perusahaan, dengan mempertimbangkan faktor usia, masa kerja, gaji, dan harapan hidup karyawan, untuk memastikan perusahaan memiliki dana yang cukup guna memenuhi kewajiban tersebut saat jatuh tempo.

### Proses #Valuasi-Aktuaria 

untuk imbalan kerja yang dilakukan secara menyeluruh, antara lain:
#### 1. Pengumpulan Data Karyawan dan Perusahaan
***Data Karyawan*** – Ini termasuk informasi dasar, seperti usia karyawan, gaji, dan masa kerja.
- Nomor Induk Pegawai (NIP) / NIK  
- Nama Karyawan *(Opsional)*  
- Tanggal Lahir Karyawan  
- Jenis Kelamin  
- Status Karyawan *(Tetap / Kontrak)*  
- Tanggal Masuk Kerja  
- Gaji / Upah  
- Tanggal Henti Kerja  
- Gaji / Upah Saat Henti Kerja  
- Departemen / Unit Kerja tempat karyawan bekerja  
- Besarnya Pembayaran *(Periode Tertentu)*  
- #Usia-Pensiun (Tahun)  
- Jenis Karyawan  
- #DPLK (Iuran dari Perusahaan) (Opsional) – Iuran / Bulan  
- #DPLK (Iuran dari Perusahaan) (Opsional) – Saldo Akhir
- #DPLK (Iuran dari Karyawan) (Opsional) - Iuran / Bulan
- #DPLK (Iuran dari Karyawan) (Opsional) - Saldo Akhir
- Tunjangan Tetap (Opsional)
- Tunjangan Tidak Tetap (Opsional)

***Informasi Perusahaan*** adalah informasi dasar mengenai perusahaan, seperti berikut:
- Nama dan Alamat Perusahaan  
- Jenis Industri Perusahaan  
- Manfaat Pasca Kerja yang diberikan  
- Program #Dana-Pensiun *(selain BPJS)*  
- Standar Akuntansi yang dipakai ( #PSAK-219 / #SAK-EP)  
- Periode / Valuasi yang akan dihitung  
- Jumlah Karyawan Tetap yang akan dihitung  
- Jumlah Karyawan Kontrak yang akan dihitung  
- Nama KAP (Auditor) yang dipakai  
- Riwayat perhitungan oleh #Konsultan-Aktuaria  
- PIC penanggung jawab riwayat perhitungan  
- Penanggung #Pajak-Manfaat pensiun (Perusahaan / Karyawan)  
- Besaran #Kenaikan-Gaji Terakhir  
- Rata-rata #Kenaikan-Gaji 5 tahun terakhir  
- Total realisasi pembayaran #Pesangon  
- BOD yang dihitung  
- PIC untuk pengiriman laporan aktuaria dan kertas kerja  

> **Catatan**: Perlu adanya Tambahan Informasi Perusahaan apabila mengikuti Program #Dana-Pensiun.
#### 2. Penggunaan Model Matematika
Setelah data dikumpulkan, berikutnya adalah menggunakan model matematika untuk mengestimasi beban dan liabilitas imbalan kerja secara relevan. 

Diagram alur #Perhitungan-Aktuaria sebagai berikut.
1. **Input Utama**  
	- Asumsi **Demografis**  
	- Asumsi **Keuangan**
2. **Proyeksi Kewajiban**  
	Menggunakan #Projected-Unit-Credit (#PUC) —  tujuannya menghitung kewajiban #Imbalan-pasca-kerja berdasarkan penilaian proyeksi masa depan.
3. **Hasil Akhir**  
	**Nilai kini kewajiban** (*present value*)

Kuncinya, **asumsi yang digunakan sangat penting** karena akan memengaruhi hasil proyeksi. Misalnya, jika diasumsikan bahwa gaji akan naik lebih cepat, maka kewajiban perusahaan akan lebih besar. Asumsi juga harus **realistis** untuk menghasilkan proyeksi yang akurat.

#### 3. #Analisis-Risiko
Tujuannya untuk mengetahui potensi lonjakan kewajiban, sehingga perusahaan dapat menyiapkan strategi cadangan, investasi, atau kebijakan manajemen risiko lain demi menjaga stabilitas #laporan-keuangan. Berikut langkah-langkahnya:
##### Identifikasi Sumber Risiko
Terdapat dua jenis risiko yang terlibat, yaitu risiko pasar dan risiko demografi, misalnya:
- **Perubahan** #Tingkat-Diskonto akibat fluktuasi suku bunga pasar.  
- #Kenaikan-Gaji yang lebih tinggi atau lebih rendah dari perkiraan.  
- **Tingkat keluar karyawan (withdrawal rate)** maupun **mortalita** yang berbeda dari asumsi.

Referensi: [[Asumsi]]
##### Metode Analisis
Cara yang dilakukan untuk #Analisis-Risiko, antara lain:
- **Uji sensitivitas** ( #Analisis-Sensitivitas):  
  Mengukur seberapa besar perubahan kewajiban jika asumsi kunci berubah (misalnya ±1% tingkat bunga).
- **Stress testing**:  
  Menerapkan skenario ekstrem (misalnya resesi, inflasi tinggi) untuk memproyeksikan dampaknya pada kewajiban.
- **#Maturity-Analysis** (Analisis jatuh tempo):  
  Menunjukkan kapan kewajiban akan jatuh tempo atau dibayarkan—apakah lebih banyak dalam 5 tahun ke depan, 10 tahun ke depan, atau lebih dari 15 tahun?
##### Apa itu Uji Sensitivitas  dan #Maturity-Analysis?
Agar mudah dipahami, begini perbedaan antara keduanya.

| **Uji Sensitivitas**                                                           | **#Maturity-Analysis**                                                                            |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| **“Apa dampaknya jika #Tingkat-Diskonto turun 1%?”**                           | **“Berapa tahun lagi mayoritas Manfaat pensiun [[Asumsi#Program-Manfaat]] akan dibayarkan?”**    |
| Mengukur dampak perubahan perubahan asumsi (seperti diskonto, gaji, mortalita) | Memberikan gambaran tentang profil distribusi jatuh tempo kewajiban (kapan manfaat akan dibayar) |
| Hasil berupa perubahan nilai kewajiban (misalnya, naik / turun 5–10%)          | Informasi seperti rata-rata durasi kewajiban dan waktu pembayaran manfaat terbanyak              |

---
## Tantangan dalam Audit Keuangan

Tantangan utama yang sering dihadapi pengguna dan perusahaan adalah validasi [[Asumsi]] aktuaria, keakuratan data yang digunakan, dan kepatuhan terhadap standar akuntansi yang berlaku, seperti #PSAK-219. Perusahaan harus mengambil pendekatan yang sistematis dan kolaboratif.

1. **Transparansi Perhitungan**
	Semua asumsi dan metode harus **jelas dan terdokumentasi**  
	Contoh: #Tingkat-Diskonto, #Kenaikan-Gaji, metode #PUC
2. **Kolaborasi Efektif**
	Aktuaris, manajemen, dan auditor perlu bekerja sama agar hasil audit:
	- Akurat  
	- Minim konflik interpretasi
3. **Dokumentasi Memadai**
	- Data karyawan harus lengkap dan **up-to-date**  
	- Simulasi sensitivitas dan penyesuaian data perlu **didokumentasikan**
4. **Validasi Auditor**
	Auditor akan:
	- Verifikasi metode dan hasil  
	- Melakukan **pengujian ulang** untuk memastikan laporan valid dan akurat
5. **Mitigasi Risiko**
	- Proses harus rutin **direvisi** agar sesuai standar  
	- **Otomatisasi** & **pelatihan** bantu kurangi risiko kesalahan manual

Kesinambungan kelima poin tersebut akan menunjang kelancaran proses **audit keuangan secara baik dan benar**, serta menjaga **kepatuhan pada regulasi**.

---
# BAB 2 #Teknis dalam #Perhitungan-Aktuaria

Setelah memahami berbagai tantangan dalam audit kewajiban imbalan kerja—seperti kebutuhan transparansi, keakuratan data, dan kolaborasi antar tim—muncul kebutuhan akan pihak yang mampu menjembatani aspek teknis dan regulasi secara profesional. Di sinilah peran #Konsultan-Aktuaria menjadi sangat penting.

## Peran #Konsultan-Aktuaria / KKA

#Konsultan-Aktuaria membantu perusahaan dalam menghitung dan mengelola kewajiban jangka panjang seperti #Pesangon dan pensiun. Kantor #Konsultan-Aktuaria (KKA) tentu wajib berizin resmi, dan tugasnya tidak hanya menyusun angka, tapi juga memberikan edukasi, menjelaskan metode perhitungan, dan memastikan hasil sesuai standar akuntansi yang berlaku, seperti #PSAK-219.

1. Pengumpulan & Analisis Data  
2. #Perhitungan-Aktuaria  
3. Komunikasi & Audit Support

#Imbalan-pasca-kerja merupakan kewajiban yang kompleks dan berdampak besar pada #laporan-keuangan. Jadi, #Konsultan-Aktuaria memastikan semua asumsi dihitung dengan akurat dan terdokumentasi dengan baik.

Kolaborasi yang erat antara perusahaan, konsultan, dan auditor adalah kunci keberlanjutan program #Imbalan-pasca-kerja. Tidak hanya itu, #Konsultan-Aktuaria juga berperan dalam membantu perusahaan merencanakan dan mengevaluasi program #Dana-Pensiun. Program ini, yang dapat dikelola secara internal maupun melalui lembaga keuangan, berpengaruh langsung pada kewajiban yang harus dicatat dalam #laporan-keuangan.

## Program #Dana-Pensiun

Dalam UU No. 11 Tahun 1992, #Dana-Pensiun adalah program pasca kerja yang dilakukan dengan cara mengumpulkan dana secara terpisah dari kekayaan perusahaan (pendiri).
Artinya, dana ini tidak boleh menjadi bagian dari cadangan perusahaan untuk keperluan pembayaran imbalan kerja, melainkan dikelola secara mandiri. UU tersebut juga membedakan jenis #Dana-Pensiun, yang dapat menjadi pengurang Kewajiban #Imbalan-pasca-kerja, antara lain:
1. #Dana-Pensiun Lembaga Keuangan (#DPLK)  
2. #Dana-Pensiun Pemberi Kerja (DPPK)  
3. #Dana-Pensiun Lembaga Asuransi
---
##### Perbandingan Karakteristik Skema #Dana-Pensiun

| Karakteristik       | **#DPLK**                                               | **DPPK**                                                                            | **DP - Lembaga Asuransi**                                       |
| ------------------- | ------------------------------------------------------ | ----------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| **Pengelola**       | Dikelola Lembaga Keuangan                              | Dikelola perusahaan tempat karyawan bekerja                                         | Dikelola Perusahaan Asuransi                                    |
| **Sumber Iuran**    | Dari karyawan dan/atau pemberi kerja                   | Dari pemberi kerja, bisa ditambah kontribusi karyawan                               | Dari premi yang dibayarkan pemberi kerja atau karyawan          |
| **Manfaat**         | Berdasarkan akumulasi iuran dan hasil investasi        | Berdasarkan perjanjian kerja atau kebijakan perusahaan                              | Berdasarkan polis asuransi yang dibeli                          |
| **Risiko**          | Ditanggung oleh peserta (karyawan)                     | Ditanggung oleh pemberi kerja                                                       | Ditanggung oleh perusahaan asuransi                             |
| **Fleksibilitas**   | Cukup fleksibel, bisa diikuti oleh berbagai perusahaan | Bergantung pada kebijakan perusahaan, biasanya hanya untuk karyawan tetap           | Disesuaikan dengan polis asuransi yang disepakati               |
| **Penggunaan Dana** | Setelah pensiun, sesuai ketentuan yang disepakati      | Umumnya dana hanya bisa digunakan setelah pensiun atau putus hubungan kerja         | Disesuaikan dengan ketentuan polis asuransi                     |
| **Keuntungan**      | Keuntungan bergantung pada hasil investasi             | Terkait dengan kesejahteraan karyawan, biasanya punya komitmen kuat dari perusahaan | Kepastian manfaat dengan risiko yang lebih rendah bagi karyawan |
Sekarang kita akan mempelajari ilustrasi bagaimana Manfaat pensiun [[Asumsi#Program-Manfaat]] ini berhubungan dengan kewajiban #Imbalan-pasca-kerja, seperti yang diatur dalam UUK No. 13 Tahun 2003 sebagai berikut.
- Jika Manfaat pensiun perusahaan **lebih besar** dari Manfaat pensiun menurut #UUK, maka kewajiban perusahaan **hanya sebesar Iuran Pemberi Kerja.**
$$
(2 \times 40)G \times 75\%  > 30G \
$$
$$\mathbf{60G > 30G}$$
- Jika Manfaat pensiun perusahaan **lebih kecil** dari Manfaat pensiun menurut #UUK, maka kewajiban perusahaan adalah **selisih dari Manfaat pensiun [[Asumsi#Program-Manfaat]] UUK dan Manfaat pensiun [[Asumsi#Program-Manfaat]] perusahaan.**
$$
(1 \times 15)G \times 75\% < 25G \
$$
$$\mathbf{11.25G < 25G}$$
$$\mathbf{jadi, 25G - 11.25G}$$
Kewajiban perusahaan = $$ \mathbf{13.75G}$$*Catatan: G=Gaji*

Dalam laporan aktuaria dan audit #PSAK-219, selisih inilah yang harus dicatat sebagai **liabilitas imbalan kerja,** meskipun perusahaan sudah menyelenggarakan program pensiun.

---

Seperti yang telah dibahas tentang kebutuhan #Valuasi-Aktuaria sebelumnya, terdapat beberapa contoh perhitungan sederhana dalam program #Dana-Pensiun untuk #Pesangon dan imbalan jangka panjang lainnya.

## #Perhitungan-Aktuaria
#### Contoh Perhitungan #Pesangon

Mengenai perhitungan #Pesangon, beberapa contoh kasusnya antara lain:
##### 1. Perusahaan tidak mengikuti program #DPLK atau Asuransi pensiun lainnya

*Perusahaan menjanjikan pembayaran #Pesangon kepada karyawan A pada saat berhenti bekerja di #Usia-Pensiun normal sebesar Rp200.000.000,- dengan masa kerja total hingga mencapai #Usia-Pensiun adalah 20 tahun. 

Maka berdasarkan metode #PUC (asumsi-asumsi diabaikan) adalah sebagai berikut.
Unit menurut Periode Jasa: Rp200.000.000 / 20 = Rp10.000.000,- sehingga pengakuan pada Laba Rugi dan Neraca adalah:

| Tahun | Beban Tahun Berjalan | Kewajiban pada Akhir Periode |
| ----- | -------------------- | ---------------------------- |
| 1     | 10.000.000           | **10.000.000**               |
| 2     | 10.000.000           | **20.000.000**               |
| 3     | 10.000.000           | **30.000.000**               |
| ...   | ...                  | ...                          |
| 20    | 10.000.000           | **200.000.000**              |
##### 2. Perusahaan mengikuti program #DPLK atau Asuransi pensiun lainnya (manfaat #Dana-Pensiun lebih kecil dari Manfaat pensiun [[Asumsi#Program-Manfaat]] karyawan)

*Perusahaan menjanjikan pembayaran #Pesangon kepada karyawan A pada saat berhenti bekerja di #Usia-Pensiun normal sebesar Rp200.000.000,- dengan masa kerja total hingga mencapai #Usia-Pensiun adalah 20 tahun.*

 > *Diketahui Perusahaan mengikuti sebuah program #DPLK dengan total iuran pertahunnya sebesar 7.000.000 (asumsi investasi diabaikan)*

Maka berdasarkan metode #PUC (asumsi-asumsi diabaikan) adalah sebagai berikut.

Unit menurut Periode Jasa: Rp200.000.000 / 20 = Rp10.000.000,- sehingga pengakuan pada Laba Rugi dan Neraca adalah:

| Tahun | Beban Tahun Berjalan | Saldo #DPLK Akhir Periode | Kewajiban Akhir Periode |
|-------|-----------------------|---------------------------|--------------------------|
| 1     | 10.000.000            | 7.000.000                 | **3.000.000**            |
| 2     | 10.000.000            | 14.000.000                | **6.000.000**            |
| 3     | 10.000.000            | 21.000.000                | **9.000.000**            |
| ...   | ...                   | ...                       | ...                      |
| 20    | 10.000.000            | 140.000.000               | **60.000.000**           |
##### 3. Perusahaan mengikuti program #DPLK atau Asuransi pensiun lainnya (manfaat #Dana-Pensiun lebih besar dari Manfaat pensiun [[Asumsi#Program-Manfaat]] karyawan)

*Perusahaan menjanjikan pembayaran #Pesangon kepada karyawan A pada saat berhenti bekerja di #Usia-Pensiun normal sebesar Rp200.000.000,- dengan masa kerja total hingga mencapai #Usia-Pensiun adalah 20 tahun.*

>*Diketahui Perusahaan mengikuti sebuah program #DPLK dengan total iuran pertahunnya sebesar 11.000.000 (asumsi investasi diabaikan)*

Maka berdasarkan metode #PUC (asumsi-asumsi diabaikan) adalah sebagai berikut.

Unit menurut Periode Jasa: Rp200.000.000 / 20 = Rp10.000.000,- sehingga pengakuan pada Laba Rugi dan Neraca adalah:

| Tahun | Beban Tahun Berjalan | Saldo #DPLK Akhir Periode | Kewajiban Akhir Periode |
|-------|-----------------------|---------------------------|--------------------------|
| 1     | 10.000.000            | **11.000.000**            | **0**                    |
| 2     | 10.000.000            | **22.000.000**            | **0**                    |
| 3     | 10.000.000            | **33.000.000**            | **0**                    |
| ...   | ...                   | ...                       | ...                      |
| 20    | 10.000.000            | **220.000.000**           | **0**                    |

Bisa disimpulkan:
	- Contoh 2 menunjukkan **kewajiban perusahaan tetap ada,** karena dana #DPLK belum mencukupi.
	- Contoh 3 menunjukkan **kewajiban perusahaan = 0,** karena manfaat dari #DPLK sudah menutupi seluruh #Pesangon.

Selain #Pesangon, beralih ke imbalan jangka panjang lainnya bagi karyawan, dimana terdapat Cuti Besar (CBS) yang merupakan hak karyawan untuk mengambil cuti panjang setelah masa kerja tertentu dan wajib dihitung secara aktuaria.

#### Contoh Perhitungan Imbalan Jangka Panjang Lainnya

Perhitungannya didasarkan pada [[Asumsi]], seperti **tingkat pengambilan cuti, gaji saat cuti,** dan #Tingkat-Diskonto, yang dapat memengaruhi arus kas perusahaan. Berikut adalah contoh perhitungan imbalan jangka panjang terkait CBS tersebut.

> *Perusahaan menjanjikan pembayaran uang cuti besar kepada setiap karyawannya bagi karyawan yang telah mencapai masa kerja 10 tahun sebesar Rp10.000.000,-.*

Maka berdasarkan metode #PUC (asumsi-asumsi diabaikan) adalah sebagai berikut:

Unit menurut Periode Jasa: Rp10.000.000 / 10 = Rp1.000.000,- sehingga pengakuan pada Laba Rugi dan Neraca adalah:

| Tahun | Beban Tahun Berjalan | Kewajiban pada Akhir Periode |
|-------|-----------------------|-------------------------------|
| 1     | 1.000.000             | 1.000.000                     |
| 2     | 1.000.000             | 2.000.000                     |
| 3     | 1.000.000             | 3.000.000                     |
| ...   | ...                   | ...                           |
| 10    | 1.000.000             | 10.000.000                    |

---
## Formula & Pajak #Manfaat-Karyawan 

### Manfaat Karyawan Tetap / PKWTT

Berdasarkan #UUK No. 13 Tahun 2003 dan #UUCK,  rumusan manfaat #Pesangon dan **uang penghargaan masa kerja** adalah sebagai berikut

| Service (YoS)** / Masa Kerja | Severance Pay / Uang #Pesangon* | Service Pay / Uang Penghargaan Masa Kerja* |
| ---------------------------- | ------------------------------ | ------------------------------------------ |
| YoS < 1                      | 1                              | 0                                          |
| 1 ≤ YoS < 2                  | 2                              | 0                                          |
| 2 ≤ YoS < 3                  | 3                              | 0                                          |
| 3 ≤ YoS < 4                  | 4                              | 2                                          |
| 4 ≤ YoS < 5                  | 5                              | 2                                          |
| 5 ≤ YoS < 6                  | 6                              | 2                                          |
| 6 ≤ YoS < 7                  | 7                              | 3                                          |
| 7 ≤ YoS < 8                  | 8                              | 4                                          |
| 8 ≤ YoS < 9                  | 9                              | 4                                          |
| 9 ≤ YoS < 10                 | 9                              | 5                                          |
| 10 ≤ YoS < 12                | 9                              | 6                                          |
| 12 ≤ YoS < 15                | 9                              | 7                                          |
| 15 ≤ YoS < 18                | 9                              | 8                                          |
| 18 ≤ YoS < 21                | 9                              | 9                                          |
| 21 ≤ YoS < 24                | 9                              | 9                                          |
| YoS ≥ 24                     | 9                              | 10                                         |

> * *Severance Pay dan Service Pay dihitung dalam kelipatan gaji (multiple of wages).*  
> ** YoS = Years of Service / Masa Kerja.
##### Formula Manfaat #UUK No. 13 Tahun 2003

| **Benefit Value**    | **Rumus Besar Manfaat / Benefit Formula**                     | **Jenis Manfaat**         |
| -------------------- | ------------------------------------------------------------- | ------------------------- |
| Normal Retirement    | (2 × Severance Pay + 1 × Service Pay) × 115%                  | Pensiun Normal            |
| Death Benefit        | (2 × Severance Pay + 1 × Service Pay) × 115%                  | Pekerja Meninggal Dunia   |
| Disability / Illness | (2 × Severance Pay + 2 × Service Pay) × 115%                  | Sakit Berkepanjangan      |
| Voluntary Resign     | (1 × Severance Pay + 1 × Service Pay) × 115% + Separation Pay | Pekerja Mengundurkan Diri |

---
##### Formula Manfaat UU Cipta Kerja ( #UUCK)

| **Benefit Value**    | **Rumus Besar Manfaat / Benefit Formula**                   | **Jenis Manfaat**         |
| -------------------- | ----------------------------------------------------------- | ------------------------- |
| Normal Retirement    | (1,75 × Severance Pay + 1 × Service Pay) + Compensation Pay | Pensiun Normal            |
| Death Benefit        | (2 × Severance Pay + 1 × Service Pay) + Compensation Pay    | Pekerja Meninggal Dunia   |
| Disability / Illness | (2 × Severance Pay + 2 × Service Pay) + Compensation Pay    | Sakit Berkepanjangan      |
| Voluntary Resign     | Compensation Pay                                            | Pekerja Mengundurkan Diri |
### Manfaat Karyawan Kontrak / PKWT

Berdasarkan **PP No. 35 Tahun 2015**, besaran uang kompensasi bagi karyawan dengan Perjanjian Kerja Waktu Tertentu (PKWT) dihitung berdasarkan masa kerja, dengan ketentuan:

| Masa Kerja PKWT               | Kompensasi                                               |
| ----------------------------- | -------------------------------------------------------- |
| PKWT 12 bulan (terus menerus) | Kompensasi 1 bulan upah                                  |
| PKWT < 12 bulan               | Pro-rata, sesuai rumus: (masa kerja / 12) × 1 bulan upah |
| PKWT > 12 bulan               | Pro-rata, sesuai rumus: (masa kerja / 12) × 1 bulan upah |
Penentuan besaran kompensasi ini memastikan bahwa karyawan yang bekerja di bawah PKWT menerima hak kompensasi yang adil sesuai dengan lamanya mereka bekerja.
#### Contoh #Perhitungan-Aktuaria Kompensasi Karyawan Kontrak

> *Perusahaan menjanjikan pembayaran kompensasi kepada karyawan A pada saat berakhirnya kontrak, dengan gaji saat ini sebesar **Rp12.000.000,-**.  Maka berdasarkan metode #PUC (asumsi-asumsi diabaikan), perhitungannya adalah sebagai berikut:*

Unit menurut Periode Jasa: Rp12.000.000 / 12 = Rp1.000.000,- sehingga pengakuan pada Laba Rugi dan Neraca adalah:

| Tahun | Beban Tahun Berjalan | Kewajiban pada Akhir Periode |
|-------|-----------------------|-------------------------------|
| 1     | 1.000.000             | 1.000.000                     |
| 2     | 1.000.000             | 2.000.000                     |
| 3     | 1.000.000             | 3.000.000                     |
| ...   | ...                   | ...                           |
| 20    | 1.000.000             | 12.000.000                    |

---
### Hubungan #Pajak-Manfaat pensiun dengan #Imbalan-pasca-kerja

Referensi: [[Asumsi#Pajak]] , [[Asumsi#Program-Manfaat]]

| Lapisan Tarif | Penghasilan Kena Pajak            | Tarif Pajak |
|---------------|---------------------------------|-------------|
| I             | < 60.000.000                    | 5%          |
| II            | 60.000.000 - 250.000.000        | 15%         |
| III           | 250.000.000 - 500.000.000       | 25%         |
| IV            | 500.000.000 - 5.000.000.000     | 30%         |
| V             | > 5.000.000.000                 | 35%         |
- **Jika pajak ditanggung perusahaan,** maka beban perusahaan lebih besar, karena:  
  $\text{Total Kewajiban}= Manfaat + PPh$  
  Ini sering disebut sebagai *gross-up*.
- **Jika pajak ditanggung karyawan,** maka perusahaan hanya membayar jumlah manfaat yang dijanjikan, dan pajak dipotong dari penerimaan karyawan (*net benefit*).

#### Contoh sederhana penerapan #Pajak-Manfaat pensiun 

*Perusahaan menjanjikan pembayaran* #Pesangon *kepada karyawan A pada saat berhenti bekerja di* #Usia-Pensiun normal sebesar *Rp200.000.000,  dengan masa kerja total hingga mencapai #Usia-Pensiun adalah 20 tahun.*

> Diketahui #Pajak-Manfaat pensiun ditanggung oleh perusahaan dengan tarif flat 5%. Maka, perusahaan harus menanggung **5% dari setiap unit**.
> Misalnya, 5% × Rp10.000.000 = Rp500.000 per tahun, dan seterusnya.

Unit menurut Periode Jasa: Rp200.000.000 / 20 = Rp10.000.000,- sehingga pengakuan pada Laba Rugi dan Neraca adalah:

| Tahun | Beban Tahun Berjalan | Kewajiban Sebelum Pajak | Kewajiban Setelah Pajak   |
|-------|-----------------------|-------------------------|---------------------------|
| 1     | 10.000.000            | 10.000.000              | **10.500.000**            |
| 2     | 10.000.000            | 20.000.000              | **21.000.000**            |
| 3     | 10.000.000            | 30.000.000              | **31.500.000**            |
| ...   | ...                   | ...                     | ...                       |
| 20    | 10.000.000            | 200.000.000             | **210.000.000**           |

---

Setelah diketahui berapa besar Manfaat pensiun [[Asumsi#Program-Manfaat]] yang seharusnya diterima karyawan sesuai ketentuan perundang-undangan, pertanyaan berikutnya adalah:  
**"Kapan dan bagaimana perusahaan harus mengakui beban atau kewajiban tersebut dalam #laporan-keuangannya?"**

Di sinilah metode aktuaria dibutuhkan. Metode yang umum digunakan dan diakui dalam standar akuntansi #PSAK-219 adalah #Projected-Unit-Credit ( #PUC).

## Metode #Projected-Unit-Credit 

Atau metode #PUC, adalah cara penting untuk menghitung kewajiban imbalan kerja, terutama dalam aturan #PSAK-219 yang sejalan dengan interpretasi IFRIC. Metode ini mengharuskan perusahaan untuk mengalokasikan imbalan kerja ke dua periode:

- Periode berjalan untuk menentukan #Biaya-Jasa-Kini ( #CSC)**.  
- Periode berjalan dan periode-periode lalu untuk menentukan **Nilai Kini Kewajiban** ( #PVDBO).

### Diagram Perspektif Garis Karier Karyawan

#### Total Masa Kerja Karyawan
Terdiri dari:
1. **Masa kerja telah dijalani** atau #Masa-Kerja-Lalu: dari #Usia-Mulai-Masa-Kerja hingga #Usia-Saat-Valuasi, mencakup:
	- #PVDBO **(Kewajiban Saat Ini)**
		Mengukur **nilai kini** atas imbalan yang telah diakumulasi hingga **saat ini**.
	- #CSC **(Biaya Tahun Ini)**
		→ Merupakan bagian dari #PVDBO yang dibebankan pada #Laporan-Laba-Rugi tahun berjalan.
		→ Digunakan untuk mencatat #Biaya-Jasa-Kini (current service cost).
2. **Future Service Cost** atau #Sisa-Masa-Kerja 
	- **Proyeksi masa kerja**: dari #Usia-Saat-Valuasi hingga #Usia-Pensiun normal.
	- Manfaat yang diproyeksikan untuk diberikan di masa mendatang, yang akan menjadi bagian dari biaya tahun-tahun berikutnya.

Jadi adanya nilai #PVFB / Present Value of Future Benefits selama Total Masa Kerja Karyawan, dengan analogi:
$$PVFB=PVDBO+ \text {Future Service Cost}$$

Di mana:
- #PVFB: Present Value of Future Benefit (nilai kini dari seluruh Manfaat pensiun yang akan dibayarkan).
- #PVDBO: Nilai Kini Kewajiban hingga saat ini.
- **Future Service Cost**: Proyeksi beban jasa di masa mendatang.

Referensi: [[Perhitungan Tetap]]

---
### Terminologi Aktuaria #imbalan-kerja 

#### Manfaat pensiun yang Dijanjikan
Terdiri dari:
1. [Manfaat yang sudah menjadi Hak]  **(Vested Benefit)**  → masa kerja yang berlalu
	- Dihitung sebagai #PVDBO 
	- Menghasilkan #CSC ( #Biaya-Jasa-Kini) , yang diakui di **#Laporan-Laba-Rugi**
		Tambahan Biaya Lainnya:
		→  #Biaya-Bunga ( #Interest-Cost)
		→  #Biaya-Jasa-Lalu (Past Service Cost)
		→ Hasil Investasi / Return on Plan Assets (jika ada)
	- #Laporan-Laba-Rugi + #OCI:
		→ Beban Imbalan Kerja ( #CSC + #Biaya-Bunga )
		→ #OCI: Keuntungan / Kerugian Aktuaria
2. [Manfaat yang masih Diproyeksikan]  
	→ Masa kerja yang akan datang  
	→ (belum diakui, masuk **future service cost**)

#### Glosarium Terminologi

##### #PVDBO  
Kewajiban imbalan kerja yang sudah menjadi tanggungan perusahaan hari ini untuk Manfaat pensiun [[Asumsi#Program-Manfaat]] di masa depan yang telah diperoleh karyawan dari masa kerja yang telah dilalui.

##### #CSC ( #Biaya-Jasa-Kini)  
Bagian #PVDBO yang berasal dari tahun berjalan untuk pensiun yang akan datang, dihitung untuk karyawan aktif. ***Dicatat di #Laporan-Laba-Rugi.***

##### #Biaya-Bunga  
Atau disebut #interest-cost , adalah kenaikan kewajiban karena waktu berjalan yang dihitung dari nilai kewajiban yang didiskontokan, dihitung dengan rumus tingkat diskonto x #PVDBO.

##### #Biaya-Jasa-Lalu ( #BJS)  
Tambahan kewajiban aktuaria yang berasal dari manfaat masa lalu yang belum dihitung sebelumnya, misalnya akibat perubahan manfaat retroaktif.

##### Vested Benefit  
Bagian dari Manfaat pensiun yang sudah menjadi hak karyawan dan tidak dapat dibatalkan. Vested berlaku meskipun karyawan mengundurkan diri sebelum pensiun minimal tertentu.

##### Return on Plan Assets 
Hasil dari investasi aset #Dana-Pensiun yang dikelola perusahaan. Nilainya dibandingkan dengan [[Asumsi]] aktuaria. Jika lebih tinggi dari asumsi → keuntungan.

##### Keuntungan / Kerugian Aktuaria 
Selisih antara [[Asumsi]] aktuaria  & realisasi aktual.  
Angka disajikan di **#laporan-keuangan** dalam bagian #OCI – #Other-Comprehensive-Income.

###### Beban diakui di #Laporan-Laba-Rugi & #OCI  
Laporan akuntansi yang mencatat:  
- Beban imbalan kerja ( #CSC, #Biaya-Bunga)  
- #Biaya-Jasa-Lalu (jika ada)  
- Keuntungan / Kerugian Aktuaria  → Disajikan di #OCI  

---
#### Rumus Metode #Projected-Unit-Credit / #PUC

Metode #PUC adalah metode *accrued benefit cost* dengan menggunakan asumsi #Kenaikan-Gaji. Misalkan rumus Manfaat pensiun [[Asumsi#Program-Manfaat]] dari suatu program imbalan pensiun adalah faktor penghargaan dikalikan dengan masa kerja dan upah, maka besar Manfaat pensiun pada #Usia-Pensiun $r$ tahun adalah sebagai berikut:
$$
B_r = F_{(r-e)} \cdot \left( (1 + s)^{r - x} \, S_x \right)
$$
Di mana:

- $x$ = #Usia-Saat-Valuasi  
- $e$ = #Usia-Mulai-Masa-Kerja  
- $r$ = #Usia-Pensiun  
- $(r - e)$ = masa kerja peserta dihitung sejak mulai bekerja di usia $e$ hingga #Usia-Pensiun normal $r$ maksimal 24 tahun  
- $B_r$ = Manfaat pensiun pada #Usia-Pensiun normal $r$ 
- $F$ = faktor penghargaan (berdasarkan Undang-Undang atau #Peraturan-Perusahaan)  
- $(1 + s)^{r - x} S_x$= upah peserta sebelum pensiun dikaitkan dengan kenaikan tingkat gaji  

##### 1. Nilai Kini (Present Value Future Benefit — #PVFB)
adalah Nilai Kini atas Manfaat pensiun [[Asumsi#Program-Manfaat]] dari suatu program imbalan pensiun dengan menggunakan #Tingkat-Diskonto dan memperhitungkan peluang hidup karyawan tersebut dapat mencapai #Usia-Pensiun $r$ tahun.
$$
{}^{r}\text{(PVFB)}_x = B_r \cdot v^{r - x} \cdot r_{-x}p_x^{(\tau)}
$$
Di mana:
- $x$ = #Usia-Saat-Valuasi  
- $e$ = #Usia-Mulai-Masa-Kerja  
- $r$ = #Usia-Pensiun  
- $(r - e)$ = masa kerja peserta yang dihitung sejak peserta mulai bekerja di usia $e$ hingga mencapai #Usia-Pensiun normal $r$ tahun, dengan maksimal masa kerja peserta adalah 24 tahun.  
- $(x - e)$ = masa kerja lalu peserta yang dihitung sejak peserta mulai bekerja di usia $e$ tahun hingga mencapai #Usia-Saat-Valuasi $x$ tahun  
- ${}^r\text{(PVFB)}_x$ = nilai sekarang dari Manfaat pensiun normal di usia $x$ tahun  
- $B_r$ = Manfaat pensiun pada #Usia-Pensiun normal $r$ tahun  
- $v^{r-x}$ = faktor diskonto selama $(r - x)$ tahun  
- $r_{-x}p_x^{(\tau)}$ = peluang kehidupan total di usia $x$ tahun hingga #Usia-Pensiun $r$ tahun  

---

##### 2. Present Value of Defined Benefit Obligation — #PVDBO
adalah nilai kini pembayaran masa depan yang diperlukan untuk menyelesaikan kewajiban atas jasa pekerja periode berjalan dan periode-periode lalu.
$$PVDBO_x = \frac{^r (PVFB)_x}{(r - e)} \cdot (x - e)$$
Di mana:
- $x$ = #Usia-Saat-Valuasi  
- $e$ = #Usia-Mulai-Masa-Kerja  
- $r$ = #Usia-Pensiun
- **$(r - e)$** = masa kerja peserta yang dihitung sejak peserta mulai bekerja di usia $e$ tahun hingga mencapai #Usia-Pensiun normal **r** tahun, dengan maksimal masa kerja peserta adalah 24 tahun.
- **$(x - e)$** = masa kerja lalu peserta yang dihitung sejak peserta mulai bekerja di usia $e$ tahun hingga mencapai #Usia-Saat-Valuasi $x$ tahun.
- **$^r(PVFB)_x$** = nilai sekarang dari Manfaat pensiun [[Asumsi#Program-Manfaat]] normal di usia $x$ tahun.

##### 3. Current Service Cost — #CSC
adalah adalah kenaikan nilai kini kewajiban imbalan pasti atas jasa pekerja dalam periode berjalan.
$$CSC_x = \frac{^r(PVFB)_x}{(r - e)}$$
Di mana:
- $x$ = #Usia-Saat-Valuasi  
- $e$ = #Usia-Mulai-Masa-Kerja  
- $r$ = #Usia-Pensiun
- $(r - e)$ = masa kerja peserta yang dihitung sejak peserta mulai bekerja di usia $e$ tahun hingga mencapai #Usia-Pensiun normal $r$ tahun, dengan maksimal masa kerja peserta adalah 24 tahun.
- $^r(PVFB)_x$ = nilai sekarang dari Manfaat pensiun [[Asumsi#Program-Manfaat]] normal di usia $x$ tahun.

#### Penerapan Metode #Projected-Unit-Credit

Diketahui :
- Manfaat pensiun [[Asumsi#Program-Manfaat]]: 1,75 x masa kerja x gaji saat pensiun.
- #Usia-Mulai-Masa-Kerja: 35 tahun.
- #Usia-Saat-Valuasi (x): 40 tahun.
- #Usia-Pensiun (r): 55 tahun.
- Gaji saat valuasi: Rp10.000.000,-
- #Tingkat-Diskonto (v): 10%
- Tingkat #Kenaikan-Gaji (s): 5%
- Asumsi lainnya diabaikan

Simak langkah perhitungannya satu per satu sebagai gambaran kecil perhitungan imbalan per satu karyawan.
##### 1. Hitung Manfaat di Masa Pensiun (Future Benefit / A)

Rumus manfaat:
$$
A = 1,75 \times \text{Masa Kerja} \times \text{Gaji pada saat pensiun} \times (1 + \text{Asumsi Kenaikan Gaji})^{\text{Sisa Masa Kerja}}
$$
$$
A = 1,75 \times (r - e) \times gaji \times (1 + s)^{(r-e)}
$$
$$
A = 1,75 \times 20 \times 10.000.000 \times (1,05)^{15} \approx Rp \, 728.625.000
$$
##### 2. Hitung Nilai Sekarang dari Manfaat ( #PVFB / B)
Yaitu Manfaat pensiun yang didiskontokan ke usia saat ini:

$$
B = \frac{A}{(1 + \text{Tingkat-Diskonto})^{\text{Sisa Masa Kerja}}}
$$
$$
B = \frac{A}{(1+v)^{(r-e)}} = \frac{728.625.000}{(1,10)^{15}} = \frac{728.625.000}{4,17725} \approx Rp \, 174.370.000
$$

##### 3. Hitung Nilai Kini Kewajiban ( #PVDBO / C)
Proporsikan #PVFB berdasarkan masa kerja saat ini:

$$
C = B \times \frac{\text{Masa Kerja saat ini}}{\text{Total Masa Kerja hingga Pensiun}}
$$
$$
C = B \times \frac{(x-e)}{(r-e)} = 174.370.000 \times \frac{5}{20} = Rp \, 43.592.500
$$

##### 4. Hitung #Biaya-Jasa-Kini ( #CSC / D)
#CSC adalah bagian #PVFB yang dialokasikan hanya untuk tahun berjalan:
$$
D = B \times \frac{1}{\text{Total Masa Kerja hingga Pensiun}}
$$
$$
D = B \times \frac{1}{(r-e)} = 174.370.000 \times \frac{1}{20} = Rp \, 8.718.500
$$

Perhitungan metode #PUC di atas berlaku per individu, tetapi dalam praktiknya dilakukan untuk seluruh karyawan satu per satu, lalu **diakumulasi agar menghasilkan total kewajiban dan beban perusahaan secara keseluruhan.** Referensi: [[Perhitungan Tetap]]

Untuk perusahaan dengan jumlah karyawan yang besar, #aktuaris bisa menggunakan software aktuaria berbasis sistem #Kalkulator atau pemodelan berbasis spreadsheet untuk solusi perhitungan yang inovatif.

Kembali ke komponen perhitungan, dimana salah satu yang penting adalah [[Asumsi]] **aktuaria itu sendiri**. Asumsi ini menjadi fondasi utama, karena meskipun data karyawan bersifat faktual, perhitungan tetap membutuhkan **proyeksi ke masa depan**— seperti berapa lama karyawan akan bekerja, berapa besar gaji mereka akan naik, hingga peluang karyawan meninggal dunia atau mengundurkan diri sebelum pensiun.

---
## [[Asumsi]] Aktuaria 

Referensi: [[Asumsi]]

Dalam perhitungan kewajiban imbalan kerja menurut #PSAK-219, pemilihan [[Asumsi]] aktuaria  sangat penting. Asumsi ini terbagi dua:
- **Asumsi Demografi** (berkaitan dengan kondisi karyawan)
- **Asumsi Keuangan** (berkaitan dengan faktor ekonomi)

Keduanya harus sesuai kondisi pasar untuk menghasilkan proyeksi yang masuk akal.
### Asumsi Demografi
1.  **Tingkat Kematian** / Mortalita, referensi: [[Asumsi#Asumsi-Kematian]]
    Menunjukkan kemungkinan karyawan meninggal dunia. Ini penting, terutama untuk program pensiun seumur hidup. Biasanya menggunakan data seperti TMI IV (Tabel Mortalita Indonesia 2019).
2. **Tingkat Cacat** / Sakit Berkepanjangan, referensi:  [[Asumsi#Asumsi-Kecacatan]]
    Menggambarkan kemungkinan karyawan tidak bisa bekerja lagi karena sakit atau cacat. Ini memengaruhi hak atas imbalan. Biasanya ditetapkan antara **5% – 10%** dari asumsi kematian.
3. **Tingkat Pengunduran Diri** (Withdrawal Rate), referensi: [[Asumsi#Asumsi-Resign]]
    Mengukur berapa banyak karyawan yang kemungkinan keluar sebelum pensiun. Digunakan untuk menghitung siapa saja yang berpotensi menerima imbalan.
4. #Usia-Pensiun (Retirement Age), referensi: [[Asumsi#Usia-Pensiun]]
    Merupakan usia rata-rata karyawan akan pensiun, sesuai aturan perusahaan. Ini akan menentukan kapan hak imbalan mulai dibayarkan.
### Asumsi Keuangan
5. #Tingkat-Diskonto **(Discount Rate)**, referensi: [[Asumsi#Tingkat-Diskonto]]
    Suku bunga yang digunakan untuk menghitung nilai kini kewajiban yang akan dibayar di masa depan. Biasanya mengacu ke data suku bunga obligasi pemerintah.
6.  **Asumsi** #Kenaikan-Gaji , referensi: [[Asumsi#Tingkat-Kenaikan-Gaji]]
    Menyesuaikan proyeksi #Kenaikan-Gaji karyawan, bisa berdasarkan: rata-rata kenaikan historis, inflasi nasional, atau kombinasi keduanya.
7. **Tingkat Return On Investment (ROI)**
    Untuk program yang memiliki aset dana (seperti #DPLK), ROI menunjukkan berapa besar hasil investasi yang bisa digunakan untuk membayar kewajiban di masa depan.
8. **Hasil Investasi Aset Program (Assets Return)**
    Asumsi penting untuk menilai seberapa besar aset program imbalan kerja bisa berkembang.
### Dampak Perubahan [[Asumsi]] aktuaria 

| **Asumsi**             | **Jika Naik maka**                                       | **Jika Turun maka** |
| ---------------------- | -------------------------------------------------------- | ------------------- |
| #Tingkat-Diskonto       | Kewajiban **turun**                                      | Kewajiban **naik**  |
| #Kenaikan-Gaji          | Kewajiban **naik**                                       | Kewajiban **turun** |
| Umur Hidup (Mortalita) | Kewajiban **naik** *(karena manfaat dibayar lebih lama)* | Kewajiban **turun** |
| Withdrawal (Resign)    | Kewajiban **turun** *(karena lebih banyak yang keluar)*  | Kewajiban **naik**  |
| #Usia-Pensiun           | Kewajiban **turun** *(karena dibayar lebih lambat)*      | Kewajiban **naik**  |

Dari [[Asumsi]] aktuaria  tersebut, langkah selanjutnya perusahaan dapat mengakui dan mengukur kewajiban imbalan kerja tersebut melalui #Perhitungan-Aktuaria dan disajikan dalam #laporan-keuangan.

---
### Penilaian Kewajiban #imbalan-kerja 

Pengakuan dan pengukuran ini merupakan proses penting karena menentukan **berapa besar beban yang harus dicatat saat ini**, seperti pensiun atau tunjangan lainnya agar dicatat secara akurat dan sesuai dengan standar akuntansi yang berlaku ( #PSAK-219).
#### Kapan Liabilitas Harus Diakui?
**Liabilitas** diakui ketika perusahaan memiliki kewajiban yang muncul dari kejadian di masa lalu dan kemungkinan besar akan mengakibatkan pengeluaran sumber daya ekonomi perusahaan di masa depan. 

Agar kewajiban #imbalan-kerja dapat diakui, tiga syarat ini **harus terpenuhi**:
 1. **Nilai Diestimasi secara Andal**
	Jika peluang pengeluaran lebih besar dari 50%, perusahaan sudah wajib mengakuinya. Ini sesuai prinsip *probable outflow*.
2.  **Ada Kewajiban Saat Ini**
	Perusahaan memiliki kewajiban kepada karyawan akibat peristiwa masa lalu, misalnya karena perjanjian kerja atau #Peraturan-Perusahaan.
3. Ada Potensi Pengeluaran**
	Nilai kewajiban dapat dihitung secara rasional berdasarkan data dan asumsi yang wajar.
#### Bagaimana Liabilitas Diukur?
Pengukuran liabilitas menggunakan metode standar yaitu #Projected-Unit-Credit ( #PUC)**, seperti yang sudah dibahas sebelumnya, untuk membagi manfaat karyawan ke setiap tahun kerja secara proporsional.

---
## Penyajian #laporan-keuangan 

Penyajian nilai #imbalan-kerja dalam #laporan-keuangan membantu pengguna laporan, seperti manajemen, auditor, atau investor, untuk memahami beban dan kewajiban yang timbul akibat program #Imbalan-pasca-kerja. Sesuai dengan #PSAK-219, komponen imbalan kerja disajikan pada dua bagian utama #laporan-keuangan:
#### #Laporan-Laba-Rugi (Profit or Loss)
Berikut adalah komponen imbalan kerja yang langsung memengaruhi laba bersih perusahaan di #laporan-keuangan:

| **Komponen**              | **Penjelasan**                                                                         |
| ------------------------- | -------------------------------------------------------------------------------------- |
| #Biaya-Jasa-Kini ( #CSC)  | Biaya tahun berjalan atas manfaat yang diperoleh karyawan selama masa kerja aktif.     |
| #Biaya-Jasa-Lalu          | Tambahan beban akibat perubahan manfaat yang berdampak ke masa kerja sebelumnya.       |
| Biaya Bunga               | Penyesuaian atas kewajiban karena waktu berlalu (diskonto).                            |
| Return on Plan Assets     | Penghasilan dari aset program pensiun, mengurangi beban (jika ada).                    |
| Kurtailmen / Penyelesaian | Pengurangan kewajiban akibat pemutusan hubungan kerja massal atau penghentian program. |
*Catatan:
Untuk entitas yang menggunakan SAK ETAP, komponen yang biasanya masuk OCI tetap diakui langsung di Laba Rugi karena tidak ada kolom OCI dalam model #laporan-keuangannya.

#### #Other-Comprehensive-Income ( #OCI)

#OCI digunakan untuk menampung perubahan jangka panjang yang tidak memengaruhi laba rugi tahun berjalan, tapi tetap memengaruhi posisi ekuitas perusahaan. Adapun komponen penyusun #OCI, antara lain:
- Keuntungan / kerugian aktuaria pada **kewajiban**
- Keuntungan / kerugian aktuaria pada **Aktiva Program**
- Keuntungan / Kerugian aktuaria pada **Penyesuaian Aset Program**

Nilai #OCI umumnya muncul karena:

| **Sebab**                          | **Penjelasan**                                                                                                  |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Keuntungan / Kerugian Aktuaria** | #Pengalaman-Penyesuaian, berupa selisih antara [[Asumsi]] (mortalita, withdrawal, gaji) dengan realisasi aktual. |
| **Selisih Hasil Investasi**        | Perbedaan antara hasil investasi aktual dan estimasi dari aset program pensiun.                                 |
| **Perbedaan Data**                 | Adanya koreksi perbedaan data karyawan dan [[Asumsi]] aktuaria .                                                |
*Contoh:
Jika hasil investasi aktual lebih rendah dari asumsi, maka selisih negatif tersebut dicatat di OCI, bukan langsung menambah beban tahun berjalan.*

---
#### Faktor yang memengaruhi Nilai Kewajiban & #OCI

| **Komponen**                       | **Penjelasan**                                                                          |
| ---------------------------------- | --------------------------------------------------------------------------------------- |
| **Jumlah Karyawan Bertambah**      | #PVDBO dan #CSC meningkat karena bertambahnya peserta yang dihitung.                    |
| **#Kenaikan-Gaji Aktual > Asumsi** | Nilai Manfaat pensiun naik → #PVDBO dan #CSC ikut naik → memicu kenaikan #OCI.          |
| **Mendekati #Usia-Pensiun**        | Penyesuaian atas kewajiban karena waktu berlalu (diskonto).                             |
| **Return on Plan Assets**          | Penghasilan dari aset program pensiun, mengurangi beban dan mengurangi #OCI (jika ada). |
| **Kurtailmen / Penyelesaian**      | Pengurangan kewajiban akibat pemutusan hubungan kerja massal atau penghentian program.  |
| **Perubahan [[Asumsi]] aktuaria ** | Perubahan #Tingkat-Diskonto, gaji, mortalita, dan lainnya → memengaruhi nilai #OCI.     |
| **Perbedaan Data Karyawan**        | Adanya koreksi dari data karyawan perusahaan itu sendiri                                |
*Contoh:
Jika tahun ini jumlah peserta naik dari 499 → 600 orang dan gaji naik tajam dari Rp499 juta → Rp600 juta, maka kenaikan kewajiban bisa lebih besar dari estimasi biasa.* 

#### #Analisis-Sensitivitas

Analisis ini bertujuan untuk membantu perusahaan melihat seberapa besar dampak perubahan kecil dalam asumsi terhadap hasil perhitungan.

| **Komponen**                          | Nilai                                     |
| ------------------------------------- | ----------------------------------------- |
| **#Tingkat-Diskonto naik 1%**          | Selisih = #PVDBO (naik 1%) – #PVDBO aktual  |
| **#Tingkat-Diskonto turun 1%**         | Selisih = #PVDBO (turun 1%) – #PVDBO aktual |
| **Salary increase rate naik 1%**      | Selisih = #CSC (naik 1%) – #CSC aktual      |
| **Salary increase rate turun 1%<br>** | Selisih = #CSC (turun 1%) – #CSC aktual     |
*Contoh:
Jika #PVDBO aktual = Rp10 M, dan naik 1% menjadi Rp10,1 M → sensitivitas = Rp100 juta.*

---
## Proses #Valuasi-Aktuaria & #Laporan-Aktuaria 

Setelah memahami bagaimana komponen #imbalan-kerja disajikan dalam #laporan-keuangan—baik dalam laba rugi maupun penghasilan komprehensif lain ( #OCI)—pertanyaan berikutnya yang sering muncul adalah: ***bagaimana nilai-nilai tersebut diperoleh?***

Untuk menjawabnya, kita perlu melihat lebih dekat proses di balik layar, yaitu bagaimana #Perhitungan-Aktuaria dilakukan dan bagaimana laporan tersebut disusun.

Sebagai contoh kasus sederhana, ini salah satu penerapan #Perhitungan-Aktuaria dari salah satu perusahaan di Indonesia.

>*PT ABC adalah perusahaan manufaktur sepatu ingin melakukan #Valuasi-Aktuaria di tahun 2024, terdiri dari karyawan tetap dan kontrak, dan tidak mengikuti program #DPLK.*

### Proses #Valuasi-Aktuaria 

Referensi: [[PANDUAN LENGKAP IMBALAN KERJA PSAK 219#Proses Valuasi-Aktuaria]]
#### 1. Menyiapkan Data yang Dibutuhkan

##### **Data Karyawan**

| Permanent Employees              | **31 Desember 2024** | **31 Desember 2023** | **Keterangan (Bahasa Indonesia)**                 |
| -------------------------------- | -------------------- | -------------------- | ------------------------------------------------- |
| Number of Participants           | 26.897               | 17.839               | Jumlah Peserta (orang)                            |
| Average Age of Employee (Years)  | 26,57                | 26,91                | Rata-rata Usia (Tahun) untuk Karyawan Tetap       |
| Average Years of Service (Years) | 2,55                 | 3,42                 | Rata-rata Masa Kerja (Tahun) untuk Karyawan Tetap |

| Contract Employees               | **31 Desember 2024** | **31 Desember 2023** | **Keterangan (Bahasa Indonesia)**                   |
| -------------------------------- | -------------------- | -------------------- | --------------------------------------------------- |
| Number of Participants           | 11                   | -                    | Jumlah Peserta (orang)                              |
| Average Age of Employee (Years)  | 56,82                | -                    | Rata-rata Usia (Tahun) untuk Karyawan Kontrak       |
| Average Years of Service (Years) | 0,47                 | -                    | Rata-rata Masa Kerja (Tahun) untuk Karyawan Kontrak |
##### **Data Keuangan**

|                                                          | **31 Desember 2024** | **31 Desember 2023** |                                                                  |
| -------------------------------------------------------- | -------------------- | -------------------- | ---------------------------------------------------------------- |
| Total Monthly Wages for Permanent and Contract Employees | 69.714.825.714       | 44.352.264.000       | Jumlah Gaji Sebulan untuk Karyawan Tetap dan Karyawan Kontrak    |
| Benefits Payment in The Period                           | (1.617.024.017)      | (2.268.565.000)      | Realisasi Pembayaran Manfaat Dalam Tahun Berjalan                |
| Benefit Paid in The Period for Asset Program             | -                    | -                    | Realisasi Pembayaran Manfaat Dalam Tahun Berjalan (Akun Program) |
| Company Contribution Paid in Period                      | -                    | -                    | Iuran Perusahaan Dalam Tahun Berjalan                            |
| Saldo #DPLK (Employer Portion)                            | -                    | -                    | Saldo #DPLK Porsi Perusahaan                                      |
Pada data keuangan, dapat diambil informasi bahwa:
- Jumlah karyawan (ditambah karyawan kontrak baru yang tercatat) dan Total Gaji [[Perhitungan Tetap#Total-Gaji]] bulanan naik signifikan, menunjukkan **potensi kenaikan kewajiban** ( #CSC & #PVDBO).
- Rata-rata usia stabil, menunjukkan banyak karyawan baru yang mungkin belum fully vested.
- #Realisasi-Manfaat menurun kemungkinan karena **berkurangnya kasus resign/pensiun/meninggal tahun berjalan.**
- Tidak ada aset program / iuran perusahaan, artinya seluruh kewajiban akan **dibebankan penuh ke perusahaan** (tanpa offset aset).

##### Hal yang perlu Dikonfirmasi:
- Apakah #Kenaikan-Gaji terjadi secara merata atau karena ada kategori jabatan tertentu?
- Apakah karyawan kontrak menerima manfaat jangka panjang? *Jika ya, harus dimasukkan ke perhitungan.*
- Apakah terdapat perubahan struktur manfaat *(misalnya: perubahan aturan pensiun atau #Pesangon)?*
- Konfirmasi bahwa tidak ada aset #Dana-Pensiun internal atau pihak ketiga yang perlu diperhitungkan.
#### 2. Menetapkan [[Asumsi]] aktuaria 

|                                                       | **31 Desember 2024**                                                                                                         | **31 Desember 2023**                                                                                           |                                                           |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| Discount Rate Beginning Period                        | 7,00%                                                                                                                        | 7,25%                                                                                                          | #Tingkat-Diskonto Awal Tahun                              |
| Discount Rate Ending Period                           | 7,13%                                                                                                                        | 7,00%                                                                                                          | #Tingkat-Diskonto Akhir Tahun                             |
| Expected Rate of Return on Plan Assets at BoP (years) | -                                                                                                                            | -                                                                                                              | Tingkat Harapan Investasi atas Aktiva Program (per tahun) |
| Future Salary Increases (per annum)                   | 8,00%                                                                                                                        | 8,00%                                                                                                          | Tingkat #Kenaikan-Gaji (per Tahun)                        |
| Mortality Table                                       | TMI IV                                                                                                                       | TMI IV                                                                                                         | Tabel Mortalita                                           |
| Disability Rate                                       | 10,00% dari TMI IV                                                                                                           | 10,00% dari TMI IV                                                                                             | Tingkat Cacat                                             |
| Withdrawal Rate                                       | 20–29  =  6.00%  <br>30–34  =  3.00%  <br>35–39  =  1.80%  <br>40–50  =  1.20%  <br>51–52  =  0.60%  <br> >52   =  0.00%<br> | 20–29  =  6.00%<br>30–34  =  3.00%<br>35–39  =  1.80%<br>40–50  =  1.20%<br>51–52  =  0.60%<br> >52   =  0.00% | Tingkat Pengunduran Diri                                  |
| Cost Method                                           | #PUC (IFRIC)                                                                                                                 | #PUC (IFRIC)                                                                                                   | Metode #Perhitungan-Aktuaria                              |
| Normal Retirement Age (Years old)                     | 55                                                                                                                           | 55                                                                                                             | #Usia-Pensiun Normal (Tahun)                              |

- Naiknya #Tingkat-Diskonto akhir adalah faktor yang bisa menyebabkan #PVDBO sedikit menurun, meskipun data karyawan dan gaji tetap stabil.
- Penyesuaian withdrawal rate, semakin banyak karyawan yang **diperkirakan bertahan → #PVDBO naik.**
- #Kenaikan-Gaji tidak berubah → jika realisasi ternyata **lebih tinggi,** bisa menimbulkan **kerugian aktuaria** di tahun berikutnya.
##### Hal yang perlu Dikonfirmasi:
1. Validasi bahwa seluruh asumsi telah dikaji ulang terhadap:
    - Data aktual tahun sebelumnya
    - Kebijakan SDM dan tren internal perusahaan
    - Kondisi pasar (yield curve, inflasi, #Kenaikan-Gaji sektoral)
2. Dokumentasikan alasan perubahan asumsi, khususnya jika diskonto turun atau withdrawal rate disesuaikan, untuk keperluan audit dan keterlacakan hasil.
3. Koordinasi dengan HR atau manajemen SDM, jika terdapat potensi perubahan #Usia-Pensiun, pengurangan tenaga kerja, atau kebijakan baru yang bisa memengaruhi kewajiban masa depan.
#### 3. Perhitungan Beban Kewajiban

**Manfaat yang Diterima**
   - Uang #Pesangon, Uang Penghargaan Masa Kerja dan Uang Penggantian Hak sesuai #Peraturan-Perusahaan dan UU Ketenagakerjaan No. 6 Tahun 2023;
   #Usia-Pensiun normal adalah 55 tahun.
**Formula Manfaat** 
   - Uang #Pesangon dan Uang Penghargaan Masa Kerja sesuai #Peraturan-Perusahaan dan UU Ketenagakerjaan No. 6 Tahun 2023:

Referensi: [[Asumsi#Program-Manfaat]]

| Service (YoS)** / Masa Kerja | Severance Pay / Uang #Pesangon* | Service Pay / Uang Penghargaan Masa Kerja* |
| ---------------------------- | ------------------------------ | ------------------------------------------ |
| YoS < 1                      | 1                              | 0                                          |
| 1 ≤ YoS < 2                  | 2                              | 0                                          |
| 2 ≤ YoS < 3                  | 3                              | 0                                          |
| 3 ≤ YoS < 4                  | 4                              | 2                                          |
| 4 ≤ YoS < 5                  | 5                              | 2                                          |
| 5 ≤ YoS < 6                  | 6                              | 2                                          |
| 6 ≤ YoS < 7                  | 7                              | 3                                          |
| 7 ≤ YoS < 8                  | 8                              | 4                                          |
| 8 ≤ YoS < 9                  | 9                              | 4                                          |
| 9 ≤ YoS < 12                 | 9                              | 5                                          |
| 12 ≤ YoS < 15                | 9                              | 6                                          |
| 15 ≤ YoS < 18                | 9                              | 7                                          |
| 18 ≤ YoS < 21                | 9                              | 8                                          |
| 21 ≤ YoS < 24                | 9                              | 9                                          |
| YoS ≥ 24                     | 9                              | 10                                         |

> *Catatan:*  
> * Severance Pay dan Service Pay dinyatakan dalam kelipatan gaji (Multiple of Wages).
> ** YoS = Years of Service / Masa Kerja.

| **Benefit Value**     | **Rumus Besar Manfaat / Benefit Formula**                    | **Jenis Manfaat**               |
|------------------------|---------------------------------------------------------------|----------------------------------|
| Normal Retirement      | (2 × Severance Pay + 1 × Service Pay)                         | Pensiun Normal                   |
| Death Benefit          | (1,75 × Severance Pay + 1 × Service Pay)                      | Pekerja Meninggal Dunia          |
| Disability / Illness   | (2 × Severance Pay + 2 × Service Pay)                         | Sakit Berkepanjangan             |
| Voluntary Resign       | Severance Pay + Service Pay                                   | Pekerja Mengundurkan Diri        |

   - Uang Penggantian Hak sesuai #Peraturan-Perusahaan dan UU Ketenagakerjaan No. 6 Tahun 2023:
	Uang Penggantian Hak yang diterima meliputi:
    1. Cuti tahunan yang belum diambil dan belum gugur;
	2. Penggantian perumahan serta pengobatan dan perawatan ditetapkan 15% (lima belas perseratus) dari uang #Pesangon dan/atau uang penghargaan masa kerja bagi yang memenuhi syarat;
	3. Hal-hal lain yang ditetapkan dalam perjanjian kerja, #Peraturan-Perusahaan, atau Perjanjian Kerja Bersama.

---
### Penyajian #Laporan-Aktuaria 
#### Tabel 1
Ikhtisar Data dan [[Asumsi]] aktuaria Perhitungan

| **No.** | **Explanation**                                    | **31 Desember 2024** | **31 Desember 2023** | **Uraian**                                          |
| ------- | -------------------------------------------------- | -------------------- | -------------------- | --------------------------------------------------- |
| 1.      | Basic Data and Assumptions                         |                      |                      | Data dan Asumsi                                     |
| 2.      | Number of Employee                                 | 26,908               | 17,839               | Jumlah Karyawan (orang)                             |
| 3.      | Monthly Wages for Permanent and Contract EEs       | 69,714,825,714       | 44,352,264,000       | Jumlah Gaji Sebulan – Karyawan Tetap dan Kontrak    |
| 4.      | Average Wages                                      | 2,590,169            | 2,486,231            | Rata-rata Gaji                                      |
| 5.      | Average Age of Permanent and Contract EEs          | 26.58                | 26.91                | Rata-rata Usia (Tahun) – Karyawan Tetap dan Kontrak |
| 6.      | Average YoS – Permanent and Contract EEs           | 2.55                 | 3.42                 | Rata-rata Masa Kerja (Tahun)                        |
| 7.      | Average Future Service                             | 29.00                | 28.00                | Rata-rata Sisa Masa Kerja                           |
| 8.      | Average Expected Remaining Working Years           | 27.00                | 26.00                | Rata-rata Sisa Masa Kerja Diperkirakan (Tahun)      |
| 9.      | Discount Rate Beginning Period                     | 7.00%                | 7.25%                | #Tingkat-Diskonto Awal Tahun                         |
| 10.     | Discount Rate Ending Period                        | 7.13%                | 7.00%                | #Tingkat-Diskonto Akhir Tahun                        |
| 11.     | Expected Return on Plan Assets                     | -                    | -                    | Tingkat Harapan Investasi atas Program              |
| 12.     | Future Salary Increases (per annum)                | 8.00%                | 8.00%                | Tingkat #Kenaikan-Gaji Tahunan                       |
| 13.     | Current Service Cost                               | 24,367,806,194       | 18,224,815,000       | #Biaya-Jasa-Kini – tahun berjalan                    |
| 14.     | Total Benefit Paid in Year                         | (1,617,024,017)      | (2,268,565,000)      | Imbalan yang dibayarkan                             |
| 15.     | Company Contribution Paid in Year                  | -                    | -                    | Iuran yang dibayar perusahaan                       |
| 16.     | Present Value of Obligation at BoP                 | 72,176,560,000       | 54,148,129,000       | Nilai kini kewajiban – awal tahun                   |
| 17.     | Present Value of Obligation at EoP                 | 86,691,067,766       | 72,175,960,000       | Nilai kini kewajiban – akhir tahun                  |
| 18.     | Past Service Cost – Non Vested at BoP              | -                    | -                    | #Biaya-Jasa-Lalu – non-vested                        |
| 19.     | Past Service Cost – Vested at BoP                  | (13,369,423,448)     | (4,732,020,000)      | #Biaya-Jasa-Lalu – vested                            |
| 20.     | Fair Value of Plan Asset Program – Start of Period | -                    | -                    | Nilai wajar aktiva program – awal tahun             |
| 21.     | Fair Value of Plan Asset Program – End of Period   | -                    | -                    | Nilai wajar aktiva program – akhir tahun            |

|     | **Explanation**                   | **31 Desember 2024**                                                                                           | **31 Desember 2023**                                                                                           | **Uraian**                  |
| --- | --------------------------------- | -------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | --------------------------- |
| 22. | Mortality Table                   | TMI IV                                                                                                         | TMI III                                                                                                        | Tabel Mortalita             |
| 23. | Disability Rate                   | 10,00% dari TMI IV                                                                                             | 10,00% dari TMI III                                                                                            | Tingkat Cacat               |
| 24. | Withdrawal Rate                   | 20–29  =  6.00%<br>30–34  =  3.00%<br>35–39  =  1.80%<br>40–50  =  1.20%<br>51–52  =  0.60%<br> >52   =  0.00% | 20–29  =  6.00%<br>30–34  =  3.00%<br>35–39  =  1.80%<br>40–50  =  1.20%<br>51–52  =  0.60%<br> >52   =  0.00% | Tingkat Pengunduran Diri    |
| 25. | Actuarial Calculation Method **)  | #PUC (IFRIC)                                                                                                    | #PUC (IFRIC)                                                                                                    | Metode #Perhitungan-Aktuaria |
| 26. | Normal Retirement Age (Years old) | 55                                                                                                             | 55                                                                                                             | #Usia-Pensiun Normal (Tahun) |
Berdasarkan data dan asumsi yang digunakan, berikut hasil #Perhitungan-Aktuaria mencakup karyawan tetap dan kontrak per 31 Desember 2024.
- **Nilai Kini Kewajiban** ( #PVDBO) sebesar Rp86,691,667,786,-
- #Biaya-Jasa-Kini ( #CSC) sebesar Rp24,367,080,194,-
- #Biaya-Jasa-Lalu sebesar (Rp13,349,423,448,-)
##### 1. Mengapa #PVDBO tahun 2024 meningkat?
- Bertambahnya masa kerja** karyawan
- **Bertambahnya jumlah** karyawan
- Adanya **perubahan pada Asumsi Mortalita** yang digunakan
##### 2. Mengapa #Biaya-Jasa-Lalu ( #BJS) muncul?
- Adanya **Perubahan** #Usia-Pensiun normal → jika #Usia-Pensiun yang ditetapkan dalam peraturan baru berbeda dari sebelumnya.
- Adanya **Perubahan** #Peraturan-Perusahaan / PKB / Manfaat yang digunakan
- Adanya **Perubahan Status** → status dari kontrak ke tetap, atau perubahan jabatan sehingga eligible atas manfaat tambahan.
- Adanya **Mutasi Karyawan Masuk**
- Adanya **Mutasi Keluar, dengan Realisasi yang dibayarkan > Kewajiban**

Setelah memahami ikhtisar Tabel 1, langkah berikutnya adalah melihat bagaimana kewajiban perusahaan berkembang sepanjang tahun berjalan, termasuk meneliti faktor-faktor yang menambah atau mengurangi kewajiban, serta apakah terjadi **selisih (gain/loss) antara perkiraan dengan kenyataan.**

---
#### Tabel 2
Perhitungan Keuntungan / Kerugian Aktuaria Perhitungan

|        | **Explanation**                                 | **31 Desember 2024** | **31 Desember 2023** | **Uraian**                                             |
| ------ | ----------------------------------------------- | -------------------- | -------------------- | ------------------------------------------------------ |
| **1.** | **Actual Present Value of Obligation at BoP**   | **72,176,560,000**   | **54,148,329,000**   | **Nilai Kini Kewajiban pada Awal Periode**             |
| 2.     | Past Service Cost - Non Vested                  | -                    | -                    | #Biaya-Jasa-Lalu - Non Vested                           |
| 3.     | Past Service Cost - Vested                      | (13,349,423,448)     | (4,763,220,000)      | #Biaya-Jasa-Lalu - Vested                               |
| 4.     | Interest Cost                                   | 5,052,359,200        | 3,843,519,000        | Biaya Bunga                                            |
| 5.     | Current Service Cost                            | 24,367,080,194       | 17,842,815,000       | #Biaya-Jasa-Kini                                        |
| 6.     | Benefit Payments                                | (1,617,024,071)      | (2,268,565,000)      | Pembayaran Manfaat                                     |
| 7.     | Changes in Benefit Plans                        | 81,388,415           | -                    | Perubahan Program Manfaat                              |
| 8.     | Curtailment-Settlement                          | (2,490,539,640)      | -                    | Kurtailmen-Penyelesaian                                |
| **9.** | **Present Value of Obligation at EoP**          | **84,057,623,820**   | **68,802,878,000**   | **Nilai Kini Kewajiban pada Akhir Periode - Program**  |
| 10.    | Actuarial (Gain) or Loss on Obligation          | 2,552,655,552        | 3,373,682,000        | (Keuntungan)/Kerugian Aktuaria pada Kewajiban          |
| 11.    | Change in Financial Assumption                  | (2,650,857,598)      | 2,224,431,000        | Perubahan Asumsi Keuangan                              |
| 12.    | Change in Demographic Assumption                | 81,388,415           | -                    | Perubahan Asumsi Demografi                             |
| 13.    | Experience Adjustment                           | 5,203,513,150        | 1,149,251,000        | #Pengalaman-Penyesuaian                                 |
| 14.    | Present Value of Obligation at EoP - Actual     | 86,610,279,371       | 72,176,560,000       | Nilai Kini Kewajiban pada Akhir Periode - Aktual       |
| 15.    | Actuarial (Gain) or Loss on Obligation - Actual | 2,552,655,552        | 3,373,682,000        | (Keuntungan)/Kerugian Aktuaria pada Kewajiban          |
| 16.    | Actuarial (Gain)/Loss on Benefit Payment        | -                    | -                    | (Keuntungan)/Kerugian Aktuaria pada Pembayaran Manfaat |
| 17.    | Actuarial (Gain)/Loss on Plan Assets            | -                    | -                    | (Keuntungan)/Kerugian Aktuaria pada Nilai Wajar Aktiva |
| **18.**    | **Total Actuarial (Gain)/Loss**                     | **2,552,655,552**        | **3,373,682,000**        | **Total (Keuntungan)/Kerugian Aktuaria Tahun Berjalan**    |

Dari tabel di atas, bisa didapat informasi bahwa:
1. #PVDBO **Akhir Perkiraan** (jika semua [[Asumsi]] sesuai rencana), didapat dari penjumlahan semua komponen di atasnya (nomor 1 s.d. 8)
2. **Perubahan asumsi keuangan,** terjadi keuntungan karena adanya kenaikan #Tingkat-Diskonto menjadi 7.13% (sebelumnya 7%), dan aktual dari #Kenaikan-Gaji lebih kecil dapada yang diasumsikan, yaitu 8%.
3. **Penyesuaian asumsi demografi**, terjadi kerugian karena perubahan dari TMI III ke TMI IV (dari Tabel 1) adanya peningkatan rate mortalita.
4. #Pengalaman-Penyesuaian , disebabkan oleh selain dari penyebab perubahan asumsi keuangan dan demografi, seperti:
	- Perbedaan metode pengakuan usia
	- Adanya perubahan #Usia-Pensiun atau manfaat
	- Adanya mutasi karyawan, dengan realisasi < kewajiban

Lalu, muncul kewajiban yang benar-benar dihitung di akhir tahun berdasarkan data dan asumsi terbaru sebagai #PVDBO Akhir Aktual **yang lebih besar dari Perkiraan,** dimana selisih ini menunjukkan adanya **kerugian aktuaria.**

Akibatnya, perusahaan perlu mencatat tambahan beban, agar #laporan-keuangan mencerminkan kewajiban secara lebih realistis. Biasanya, angka ini masuk ke #laporan-keuangan melalui bagian Tabel 3 #Other-Comprehensive-Income ( #OCI).

---
#### Tabel 3
Pendapatan Komprehensif Lainnya Perhitungan ( #OCI)

|       | **EXPLANATION**                                  | **31 Desember 2024** | **31 Desember 2023** | **URAIAN**                                        |
| ----- | ------------------------------------------------ | -------------------- | -------------------- | ------------------------------------------------- |
| **1** | **#Other-Comprehensive-Income at BoP**            | **(31,372,832,000)** | **(34,746,514,000)** | **Pendapatan Komprehensif lainnya awal periode**  |
| 2     | Actuarial (Gain)/Loss at on Period - Obligation  | 2,552,655,552        | 3,373,682,000        | (Keuntungan)/Kerugian aktuaria – kewajiban        |
| 3     | Actuarial (Gain)/Loss at on Period - Plan Assets | -                    | -                    | (Keuntungan)/Kerugian aktuaria – Aset Program     |
| **4** | **Total Actuarial (Gain)/Loss at on Period**     | **2,552,655,552**    | **3,373,682,000**    | **Total (Keuntungan)/kerugian aktuaria**          |
| **5** | **#Other-Comprehensive-Income at EoP**            | **(28,820,176,448)** | **(31,372,832,000)** | **Pendapatan Komprehensif lainnya akhir periode** |
- Tidak ada aset program, sehingga #OCI hanya mencerminkan perubahan dari sisi **kewajiban** saja.
- Kerugian aktuarial pada tahun berjalan sebesar Rp2,55 M sehingga nilai akhir dari #OCI sekitar (Rp28 M), artinya perusahaan masih memperoleh keuntungan.
- Perusahaan tetap harus mengungkapkan nilai #OCI ini dalam #laporan-keuangan, meskipun **tidak langsung berdampak ke laba rugi, karena tetap memengaruhi ekuitas** perusahaan.

Ketika #OCI akhir lebih kecil dari #OCI awal (masih negatif tapi mengecil), itu menyatakan adanya ***dampak positif***. Artinya: deviasi kewajiban makin kecil, prediksi makin akurat, dan beban kewajiban jangka panjang lebih terkendali.

Setelah melihat bagaimana keuntungan / kerugian aktuaria dicatat, sekarang saatnya memahami bagaimana seluruh kewajiban #imbalan-kerja ini diakui dan disajikan dalam **#laporan-keuangan perusahaan, khususnya di neraca berdasarkan Tabel 4.**

---
#### Tabel 4
Posisi Pendanaan & Pengakuan Kewajiban / (Kekayaan) dalam Neraca Perhitungan

|       | **EXPLANATION**                                        | **31 Desember 2024** | **31 Desember 2023** | **URAIAN**                                         |
| ----- | ------------------------------------------------------ | -------------------- | -------------------- | -------------------------------------------------- |
| **1** | **FUNDED STATUS**                                      |                      |                      | **STATUS PENDANAAN**                               |
| **2** | **Assets and Obligation**                              |                      |                      | **Kekayaan dan Kewajiban**                         |
| 3     | Present Value of Obligation at EOP                     | 86,691,667,786       | 72,176,560,000       | Nilai Kini Kewajiban (Present Value of Obligation) |
| 4     | Fair Value of Plan Assets                              | -                    | -                    | Nilai Wajar Aset Program                           |
| **5** | **Funded Status**                                      | **86,691,667,786**   | **72,176,560,000**   | **Posisi Pendanaan**                               |
| 6     | Unrecognized Past Service Cost - Non Vested            | -                    | -                    | #Biaya-Jasa-Lalu yang Belum Diakui - Non Vested     |
| 7     | Unrecognized Actuarial (Gains)/Losses                  | -                    | -                    | Keuntungan/(Kerugian) Aktuarial yang Belum Diakui  |
| **8**     | **Liability/(Assets) Recognized in The Balance Sheet** | **86,691,667,786**   | **72,176,560,000**   | **Kewajiban/(Kekayaan) yang Diakui dalam Neraca**  |

|        | **EXPLANATION**                                              | **31 Desember 2024** | **31 Desember 2023** | **URAIAN**                                                  |
| ------ | ------------------------------------------------------------ | -------------------- | -------------------- | ----------------------------------------------------------- |
|        | **Reconciliation of Liability/(Asset) in The Balance Sheet** |                      |                      | **Perubahan Kewajiban/(Kekayaan) yang Diakui dalam Neraca** |
| 9      | Liability/(Asset) at BoP                                     | **72,176,560,000**       | **54,148,329,000**       | Kewajiban/(Kekayaan) pada Awal Periode                      |
| 10     | Expense/(Income)                                             | 13,570,476,306       | 16,923,114,000       | Beban/(Pendapatan)                                          |
| 11     | Benefit Payment - Actual                                     | (1,167,047,213)      | (2,086,505,000)      | Realisasi Pembayaran Manfaat                                |
| 12     | Company Contributions                                        | -                    | -                    | Iuran Perusahaan                                            |
| 13     | #Other-Comprehensive-Income                                   | 2,552,655,552        | 3,373,682,000        | Pendapatan Komprehensif Lainnya                             |
| **14** | **Liability/(Assets) at EoP**                                | **86,691,667,786**   | **72,176,560,000**   | **Kewajiban/(Kekayaan) pada Akhir Periode**                 |
- Seluruh kewajiban sebesar Rp86,69 M **telah diakui sepenuhnya dalam neraca.**
- Tidak ada aset program atau komponen yang ditunda pengakuannya → laporan sudah bersih dan mencerminkan realitas.
- **Nilai akhir kewajiban** sebesar Rp86,69 M terjadi karena:
    1. Adanya **penambahan beban** pada tahun berjalan sebesar Rp13,5 M yang menyebabkan kewajiban bertambah
    2. Adanya realisasi **pembayaran manfaat** sebesar Rp1,6 M yang mengurangi kewajiban
    3. Adanya **kerugian pada OCI** tahun berjalan sebesar Rp2 M yang menyebabkan kewajiban bertambah

Terdapat nilai Beban/(Pendapatan) yang akan dijelaskan #Laporan-Laba-Rugi (Profit & Loss) sesuai Tabel 5.

---
#### Tabel 5
Pengakuan Beban / (Pendapatan) yang diakui dalam Laba Rugi

|       | **EXPLANATION**                                                   | **31 Desember 2024** | **31 Desember 2023** | **URAIAN**                                                 |
| ----- | ----------------------------------------------------------------- | -------------------- | -------------------- | ---------------------------------------------------------- |
| 1     | **COST COMPONENTS**                                               |                      |                      | **KOMPONEN BEBAN**                                         |
| 2     | Current Service Cost                                              | 24,367,080,194       | 17,842,815,000       | #Biaya-Jasa-Kini                                            |
| 3     | Interest Cost                                                     | 5,052,359,200        | 3,843,519,000        | Biaya Bunga                                                |
| 4     | Expected Return on Plan Assets                                    | -                    | -                    | Harapan dari Hasil Investasi                               |
| **5** | **Immediate Recognition of Past Service Cost - Vested**           | **(13,949,423,448)** | **(4,763,220,000)**  | **Pengakuan Segera dari #Biaya-Jasa-Lalu yang Vested**      |
| 6     | Curtailment Effect / Settlement                                   | (2,490,539,640)      | -                    | Dampak Kurtailmen / Penyelesaian                           |
| **7** | **Expense/(Income) should be Recognized In The Income Statement** | **13,579,476,306**   | **16,923,114,000**   | **Beban/(Pendapatan) yang Diakui dalam Laporan Laba/Rugi** |

Nilai Beban/(Pendapatan) muncul sebagai **total seluruh beban imbalan kerja** yang harus dicatat tahun ini, hasil perhitungan dari seluruh komponen ( #CSC, interest, #BJS, #Kurtailmen ).

Setelah membahas seluruh proses #Valuasi-Aktuaria, maka langkah berikutnya adalah melengkapi laporan dengan analisis tambahan, seperti #Analisis-Sensitivitas, #Pengalaman-Penyesuaian, dan #Maturity-Analysis .

Apa tujuannya?
- Mengantisipasi dampak dari perubahan asumsi utama di masa depan,
- Serta mempersiapkan strategi finansial yang berbasis data dan proyeksi aktual.

---
#### Uji Sensitivitas ( #Analisis-Sensitivitas)
#Analisis-Sensitivitas adalah semacam **“simulasi stres”** untuk kewajiban perusahaan. Perusahan bisa tahu: *"Kalau suku bunga turun 1%, seberapa besar tambahan beban kewajiban yang harus saya siapkan?"*

**Dampak perubahan 1%** tingkat bunga diskonto dan #Kenaikan-Gaji

**Permanent and Contract Employees    |      Karyawan Tetap dan Kontrak**

|                     | 31 Desember 2024     | **1% Increase** | **1% Decrease** | **31 Desember 2024** | **Keterangan**         |
| ------------------- | -------------------- | --------------- | --------------- | -------------------- | ---------------------- |
| **Discount Rate**   |                      | 8.13%           | 6.13%           |                      | Tingkat bunga Diskonto |
|                     | #PVDBO                | 77,423,945,640  | 97,592,966,835  | Nilai Kini Kewajiban |                        |
|                     | Current Service Cost | 21,478,902,667  | 27,806,646,653  | #Biaya-Jasa-Kini      |                        |
| **Salary Increase** |                      | 9.00%           | 7.00%           |                      | #Kenaikan-Gaji          |
|                     | #PVDBO                | 97,275,821,353  | 77,509,439,620  | Nilai Kini Kewajiban |                        |
|                     | Current Service Cost | 27,705,683,158  | 21,505,772,319  | #Biaya-Jasa-Kini      |                        |

Juga, membantu **pengambilan keputusan jangka panjang.** Misalnya, untuk menentukan apakah perlu:
- Meningkatkan pencadangan,
- Meninjau ulang kebijakan #Kenaikan-Gaji,
- Atau mempertimbangkan program pensiun yang lebih berkelanjutan.

---
#### #Pengalaman-Penyesuaian
Experience adjustment adalah **“cermin”** yang memperlihatkan seberapa akurat asumsi perusahaan dan bisa memengaruhi #OCI apabila terjadi perubahan data perhitungan.

**Permanent and Contract Employees    |      Karyawan Tetap dan Kontrak**

|                             | **31 Desember 2024** | **31 Desember 2023** |                            |
| --------------------------- | -------------------- | -------------------- | -------------------------- |
| Present Value of Obligation | 86,691,667,786       | 72,176,560,000       | Nilai kini kewajiban       |
| Fair Value of Plan Assets   | -                    | -                    | Nilai wajar aktiva program |
| Funded Status               | 86,691,667,786       | 72,176,560,000       | Posisi Pendanaan           |
| Experience Adjustment       | 5,203,513,150        | 1,149,251,000        | #Pengalaman-Penyesuaian     |

Penyesuaian cukup besar (Rp5,2 M), menandakan perlu **evaluasi ulang terhadap asumsi atau data yang digunakan.**

#### Analisis Jatuh Tempo ( #Maturity-Analysis)
Analisis ini merupakan proyeksi waktu pencairan kewajiban perusahaan ke depan. Ini mencakup pensiun dini, resign, atau kematian mendadak. Harus disiapkan agar **tidak mengganggu arus kas.**

Permanent and Contract Employees    |      Karyawan Tetap dan Kontrak

| #Maturity-Analysis (Rp 000) | **31 Desember 2024** | **31 Desember 2023** |                       |
| -------------------------- | -------------------- | -------------------- | --------------------- |
| Less than 1 year           | 609,256,655          | -                    | Dibawah 1 tahun       |
| Between 1 and 2 years      | 2,429,840,606        | -                    | Antara 1 dan 2 tahun  |
| Between 2 and 3 years      | 1,417,719,537        | -                    | Antara 2 dan 3 tahun  |
| Between 3 and 5 years      | 5,337,665,383        | -                    | Antara 3 dan 5 tahun  |
| Between 5 and 10 years     | 14,001,401,767       | -                    | Antara 5 dan 10 tahun |
| Beyond 10 years            | 205,086,796,188      | -                    | Diatas 10 tahun       |
| **Total**                  | **228,882,680,136**  | -                    | **Jumlah**            |

#Maturity-Analysis berguna untuk:
- Proyeksi arus kas jangka panjang,
- Penyusunan anggaran HR tahunan,
- Menentukan perlunya ikut #DPLK atau membentuk cadangan khusus.

Setelah memahami detail #Perhitungan-Aktuaria, mulai dari penyiapan data, penetapan asumsi, hingga pembentukan #Laporan-Aktuaria kewajiban dan beban #imbalan-kerja, kini saatnya kita melihat gambaran yang lebih luas: bagaimana **implementasi** #PSAK-219 dilakukan secara menyeluruh di perusahaan, dan bagaimana **#Teknologi-Aktuaria** berperan penting dalam mendukung proses ini agar lebih efisien, akurat, dan dapat diaudit dengan baik.

Mari kita bahas lebih lanjut dalam bab berikutnya tentang **Implementasi** #PSAK-219 **& #Teknologi-Aktuaria.**

---
# BAB 3 Implementasi #PSAK-219 #Teknologi-Aktuaria

## Implementasi #PSAK-219 di Perusahaan

Implementasi #PSAK-219 tentang imbalan kerja merupakan langkah penting yang wajib dilakukan perusahaan setiap tahun untuk memastikan #laporan-keuangan mencerminkan kewajiban aktual terhadap karyawan. Proses ini tidak hanya soal hitung-menghitung oleh #aktuaris, tetapi juga menyangkut tata kelola, transparansi, dan kepatuhan terhadap standar akuntansi.
#### Kolaborasi Tim Internal
1. **Human Resources / HR Department** : Menyediakan data karyawan seperti tanggal lahir, masa kerja, dan status kontrak.
2. **Finance**  : Mencatat hasil valuasi ke #laporan-keuangan.
3. **Aktuaris**  : Menghitung kewajiban berdasarkan data & asumsi.
4. **Manajemen**  : Menyetujui asumsi yang digunakan dan kebijakan manfaat.
5. **KAP**  : Verifikasi pencatatan kewajiban.

#### Isu yang Sering Membingungkan Perusahaan
1. **“Karyawan kontrak dihitung gak sih?”**  
   Ya, jika durasi kontraknya >1 tahun dan ada manfaat yang dijanjikan.
2. **“Jika belum pernah valuasi, harus mulai dari mana?”**  
   Mulai dari mengumpulkan data dan konsultasikan ke aktuaris.
3. **“Apakah setiap tahun harus hitung ulang?”**  
   Ya, karena data karyawan dan asumsi bisa berubah setiap tahun.
4. **“Kalau gak punya program pensiun?”**  
   Tetap wajib hitung, jika ada kewajiban #Pesangon atau PHK.

Setelah memahami bagaimana #PSAK-219 diimplementasikan di perusahaan dan tantangan yang kerap dihadapi, pertanyaan berikutnya adalah: bagaimana proses #Perhitungan-Aktuaria ini dapat dilakukan secara efisien dan akurat, terutama ketika data yang dihadapi sangat kompleks dan jumlah karyawan bisa mencapai ribuan?

Di sinilah peran teknologi menjadi semakin penting. Dengan bantuan sistem cerdas dan analisis berbasis data, teknologi dapat mendukung **proses aktuaria menjadi lebih cepat, akurat, dan mudah ditelusuri.**

---
## Teknologi #Perhitungan-Aktuaria

Teknologi, terutama kecerdasan buatan dan sistem digital lainnya, kini memainkan peran penting dalam proses percepatan #Perhitungan-Aktuaria. Penggunaan teknologi tidak hanya mempercepat pekerjaan, tapi juga meningkatkan akurasi, efisiensi, dan transparansi hasil perhitungan imbalan kerja.
> 💭 *Saya butuh #Perhitungan-Aktuaria untuk lebih dari 1.000 karyawan di perusahaan saya, PT XYZ, seluruh Indonesia.*

> 💻 *Selama bertahun-tahun, perhitungannya secara manual hanya dengan bantuan spreadsheet dan sistem payroll terpisah.*
#### Masalah yang Dapat Terjadi?
1. **Keterlambatan Pe#laporan-keuangan**
	Proses perhitungan membutuhkan waktu hampir 2 bulan. Hal ini menyebabkan #laporan-keuangan tahunan tidak selesai tepat waktu untuk audit #KAP.
2. **Ketidaksesuaian Angka Antara Tahun**
	Terjadi lonjakan liabilitas yang tidak terduga karena kesalahan input asumsi diskonto dan pengabaian faktor mutasi karyawan. Auditor menemukan bahwa UPH tidak dihitung secara lengkap pada tahun sebelumnya
3. **Tidak Ada Audit Trail yang Jelas**
	Karena prosesnya dilakukan manual, tidak ada log sistem untuk melihat perubahan asumsi atau data. Auditor akan mempertanyakan dan mempersyaratkan sistemisasi proses ke depan.
#### Solusinya?
PT XYZ memutuskan bekerja sama dengan platform #Teknologi-Aktuaria yang dapat melakukan:

| **Fungsi**                                                     | **Penjelasan**                                                                                     |
| -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| #Analisis-Risiko dan Manfaat pensiun [[Asumsi#Program-Manfaat]] | Sistem dapat mengolah data besar dan menganalisis tren risiko pensiun.                             |
| Prediksi & Simulasi Masa Depan                                 | Algoritma kecerdasan buatan bisa membuat proyeksi kewajiban berdasarkan skenario ekonomi tertentu. |
| Audit Trail dan Regulasi                                       | Sistem digital menyimpan catatan histori perhitungan untuk kebutuhan audit.                        |
| Pengelolaan Jangka Panjang                                     | Memastikan keberlanjutan analisis dari tahun ke tahun, tanpa ketergantungan personel.              |
Teknologi **tidak menggantikan peran** #aktuaris, tapi meningkatkan efisiensi dan kualitas hasil kerja mereka. Aktuaris tetap memiliki peran penting dalam menentukan asumsi, menafsirkan hasil, dan memberikan saran strategis kepada manajemen.

---
## #Kalkulator Manfaat #imbalan-kerja 

Dalam dunia yang semakin digital, perusahaan tidak hanya membutuhkan laporan yang akurat, tetapi juga alat bantu yang cepat dan mudah digunakan untuk melakukan simulasi dan penghitungan manfaat #imbalan-kerja. Di sinilah **#Kalkulator manfaat imbalan kerja** memainkan peran penting.
#### Apa itu #Kalkulator Manfaat?
#Kalkulator manfaat imbalan kerja adalah alat bantu berbasis sistem (baik web maupun software) yang digunakan untuk:
- Menghitung estimasi manfaat #Pesangon, penghargaan masa kerja, dan penggantian hak sesuai aturan perusahaan atau #UUCK.
- Mensimulasikan kewajiban imbalan kerja berdasarkan data karyawan dan asumsi tertentu.
- Membantu HR dan manajemen memahami nilai manfaat #Manfaat yang akan dibayarkan ketika karyawan pensiun, mengundurkan diri, atau mengalami pemutusan hubungan kerja.

Referensi: [[Perhitungan Tetap#Manfaat]]

#Kalkulator ini dilengkapi teknologi terintegrasi mencakup:
- **Dashboard dan visualisasi data:** yang memungkinkan penyajian hasil perhitungan ke dalam bentuk grafik dan tabel, untuk melihat tren dan pola dalam data.
- **Predictive modelling:** untuk menganalisis data historis berdasarkan algoritma dan memproyeksikan manfaat karyawan di masa depan.
- **Cloud computing and storage:** memastikan bahwa data diakses dan dikerjakan secara real-time dan aman, dimanapun mereka berada.
#### Manfaat dan Fungsi #Kalkulator Manfaat
Beberapa manfaat pentingnya antara lain:
1. Simulasi cepat dan praktis untuk perencanaan internal dan komparatif antar karyawan berdasrakan regulasi acuan.
2. Mengurangi kesalahan estimasi manual
3. Membantu user memahami komponen manfaat seperti pensiun, meninggal dunia, dan cacat tetap, dilengkapi kejelasan soal besarannya.

Dengan memasukkan komponen utama perhitungan, meliputi data karyawan dan data perusahaan dapat diperoleh hasil perhitungan per individu karyawan perusahaan secara cepat, akurat, dan komprehensif.

---
## Integrasi #API vs Proses #Valuasi-Aktuaria 

#API (*Application Programming Interface*) memungkinkan sistem aktuaria terhubung langsung dengan sumber data internal perusahaan seperti:
- **Payroll System**  
- **HRIS (Human Resource Information System)**  
- **Sistem Keuangan / ERP**  
- **DPLK, BPJS atau vendor asuransi lainnya**
#### Skema #API Sistem Aktuaria
- **Payroll System**  
- **Human Resource Information System**  
- **Enterprise Resource Planning**  
  → Terhubung ke **API**  
  → Lalu terintegrasi dengan #Valuasi-Aktuaria **Web-based System**
#### Manfaat Integrasi API
- **Otomasi & Efisiensi Proses**  
	Menghilangkan proses input manual dan *excel tracking* yang rawan error.
- **Peningkatan Akurasi Data**  
	Mendapatkan data real-time dan valid, langsung dari sumber aslinya.
- **Analisis Lebih Cepat & Dinamis**  
	Membantu dalam skenario *what-if analysis* untuk pengambilan keputusan.
- **Kesiapan Audit & Pelaporan** 
	Cocok untuk perusahaan terbuka atau yang diaudit oleh #KAP besar.

Jadi, integrasi #API menjadi kunci modernisasi proses aktuaria yang selama ini bergantung pada input manual dan lembar kerja yang rumit.

Beralih ke #Valuasi-Aktuaria, Bab 4 akan membahas tentang **penyajian** #Laporan-Aktuaria serta analisis faktor fluktuasi** perubahan dari nilai komponen #Perhitungan-Aktuaria yang krusial yang sering ditanyakan manajemen perusahaan.

---
# BAB 4 Penyajian #Laporan-Aktuaria dan #Analisis-Aktuaria

#Perhitungan-Aktuaria memiliki peran penting dalam #laporan-keuangan, khususnya untuk menunjukkan kewajiban #imbalan-kerja karyawan. Angka-angka ini **berdampak langsung pada posisi keuangan dan laba rugi** perusahaan, sehingga perlu dipahami tidak hanya oleh tim keuangan, tetapi juga oleh manajemen dan auditor.

Pemahaman yang baik terhadap penyajian ini akan membantu manajemen menjelaskan angka-angka kepada auditor maupun pemangku kepentingan lain, serta menjamin bahwa #laporan-keuangan mencerminkan kondisi yang sebenarnya. 

## Penyajian Aktuaria #laporan-keuangan 

Hasil #Perhitungan-Aktuaria biasanya dituangkan dalam dua laporan utama perusahaan: **Laporan Posisi Keuangan** dan **#Laporan-Laba-Rugi Komprehensif.** Setiap laporan memuat elemen-elemen yang berasal dari perhitungan kewajiban imbalan kerja dan, jika ada, aset program.
### Laporan Posisi Keuangan (Neraca)

Pada bagian ini, kewajiban #imbalan-kerja disajikan sebagai "**Liabilitas Imbalan Kerja Jangka Panjang**". Nilai yang ditampilkan merupakan total kewajiban perusahaan atas manfaat yang akan dibayarkan kepada karyawan di masa depan, yang dihitung dengan metode aktuaria.

Contoh Penyajian:
(sumber: PT HABCO TRANS MARITIMA Tbk 2024)

| LIABILITAS                                                             | 30 Juni 2024        | 31 Desember 2023    |
| ---------------------------------------------------------------------- | ------------------- | ------------------- |
| **LIABILITAS JANGKA PENDEK**                                           |                     |                     |
| Utang usaha - Pihak ketiga                                             | 33.662.442.387      | 43.996.129.180      |
| Utang usaha - Pihak berelasi                                           | 77.346.225          | 129.856.993         |
| Utang pajak                                                            | 3.271.515.868       | 6.300.206.133       |
| Beban masih harus dibayar                                              | 7.223.977.686       | 510.131.124         |
| Pinjaman bank jangka panjang yang jatuh tempo dalam satu tahun         | 29.925.000.000      | 29.925.000.000      |
| **Jumlah Liabilitas Jangka Pendek**                                    | **74.160.282.166**  | **80.861.323.430**  |
|                                                                        |                     |                     |
| **LIABILITAS JANGKA PANJANG**                                          |                     |                     |
| Pinjaman bank jangka panjang setelah dikurangi bagian yang jatuh tempo | 79.806.249.994      | 94.768.750.000      |
| ==Liabilitas imbalan kerja karyawan==                                  | ==1.532.712.360==   | ==1.115.048.555==   |
| **Jumlah Liabilitas Jangka Panjang**                                   | **81.338.962.354**  | **95.883.798.555**  |
|                                                                        |                     |                     |
| **JUMLAH LIABILITAS**                                                  | **155.499.244.520** | **176.745.121.985** |

Angka ini menunjukkan nilai kini dari total kewajiban #imbalan-kerja pada tanggal valuasi 30 Juni 2024. Dibandingkan dengan posisi 31 Desember 2023 sebesar Rp1.115.048.555, terdapat kenaikan liabilitas sebesar Rp 417.663.805,-.

---
### #Laporan-Laba-Rugi Komprehensif

Beban / pendapatan dari #Perhitungan-Aktuaria disajikan dalam dua bagian:
- **Laba Rugi (*Profit or Loss*)**, sebagai dari **beban usaha** (khususnya dalam beban umum dan administrasi bagian “Imbalan kerja karyawan”).
- **Penghasilan Komprehensif Lain (OCI)**, yang umumnya **pengukuran kembali (*remeasurement*)** atas kewajiban karena perubahan asumsi atau deviasi antara proyeksi dan realisasi.
#### 1. Laba Rugi - Beban Usaha
Dalam laporan untuk periode enam bulan yang berakhir pada 30 Juni 2024, beban imbalan kerja karyawan disajikan sebagai bagian dari **Beban Umum dan Administrasi**, dengan rincian:
> **Imbalan kerja karyawan (Catatan 17)**  Rp484.577.404,-

Angka ini merupakan komponen dari total beban usaha sebesar:
> **Total Beban Usaha**  Rp22.983.369.313,-

Contoh Penyajian:
(sumber: PT HABCO TRANS MARITIMA Tbk 2024)

**BEBAN UMUM DAN ADMINISTRASI**

| Akun                                                            | 2024                | 2023 (Tidak Diaudit/Unaudited) |
| --------------------------------------------------------------- | ------------------- | ------------------------------ |
| Gaji dan tunjangan (Salary and allowance)                       | 19.376.175.024      | 8.837.895.532                  |
| Perjamuan (Entertainment)                                       | 648.214.631         | 462.681.788                    |
| Perjalanan dinas (Business travel)                              | 500.116.416         | 466.767.691                    |
| ==**Imbalan kerja karyawan (Catatan 17) (Employee benefits)**== | ==**484.577.404**== | ==**555.213.460**==            |
| Tenaga ahli (Professional fees)                                 | 397.017.244         | 255.045.949                    |
| Sewa kantor (Office rent)                                       | 276.388.890         | 271.944.444                    |
| Perbaikan dan perawatan (Repairs and maintenance)               | 170.445.752         | 470.512.949                    |
| Penyusutan (Catatan 10) (Depreciation)                          | 115.687.254         | 106.928.850                    |
| Amortisasi (Catatan 11) (Amortization)                          | 98.437.000          | 98.437.000                     |
| Perlengkapan kantor (Office supplies)                           | 67.514.115          | 140.796.185                    |
| Donasi (Donation)                                               | -                   | 180.000.000                    |
| Perijinan (Permit)                                              | 36.683.563          | 11.000.000                     |
| Lainnya (Others)                                                | 752.651.523         | 1.412.011.128                  |
| **Jumlah (Total)**                                              | **22.983.369.313**  | **12.959.989.527**             |

#### 2. Penghasilan Komprehensif Lain ( #OCI)

Selain itu, terdapat pencatatan atas **pengukuran kembali atas** #Imbalan-pasca-kerja yang langsung masuk ke penghasilan komprehensif lain:
> **Penghasilan Komprehensif Lain:**
> 1. Pengukuran kembali atas #Imbalan-pasca-kerja  Rp32.623.599,-

Angka ini disajikan pada bagian akhir #Laporan-Laba-Rugi komprehensif dan menambah total laba komprehensif perusahaan:
> **2. Jumlah Laba Komprehensif Tahun Berjalan**  
> Rp22.983.369.313,-

Contoh Penyajian:
(sumber: PT HABCO TRANS MARITIMA Tbk 2024)

| Uraian                                                                                  | 2024                 | 2023 (Tidak Diaudit/Unaudited) |
| --------------------------------------------------------------------------------------- | -------------------- | ------------------------------ |
| **PENDAPATAN (REVENUES)**                                                               | 367.125.987.209      | 254.340.065.148                |
| **BEBAN POKOK PENDAPATAN (COST OF REVENUES)**                                           | (254.578.558.899)    | (147.190.888.036)              |
| **LABA KOTOR (GROSS PROFIT)**                                                           | 112.547.428.310      | 107.149.177.112                |
|                                                                                         |                      |                                |
| ==**BEBAN USAHA (OPERATING EXPENSES)**==                                                |                      |                                |
| ==Beban umum dan administrasi (General and administrative expenses)==                   | ==(22.983.369.313)== | ==(12.959.989.527)==           |
|                                                                                         |                      |                                |
| **LABA USAHA (OPERATING PROFIT)**                                                       | 89.564.058.997       | 94.189.187.585                 |
| Penghasilan keuangan (Finance income)                                                   | 3.301.682.237        | 521.405.174                    |
| Laba (rugi) selisih kurs - Bersih (Gain (loss) on foreign exchange)                     | (247.367.803)        | 2.917.303.853                  |
| Beban keuangan (Finance costs)                                                          | (4.446.354.163)      | (4.288.333.332)                |
| Beban lain-lain (Other expense)                                                         | -                    | (762.621.644)                  |
|                                                                                         |                      |                                |
| **LABA SEBELUM PAJAK FINAL DAN PAJAK PENGHASILAN (PROFIT BEFORE FINAL AND INCOME TAX)** | 88.172.019.268       | 92.576.941.636                 |
| PAJAK FINAL (FINAL TAX)                                                                 | (4.405.511.796)      | (3.052.080.746)                |
| **LABA SEBELUM BEBAN PAJAK PENGHASILAN (PROFIT BEFORE INCOME TAX EXPENSE)**             | 83.766.507.472       | 89.524.860.890                 |
| BEBAN PAJAK PENGHASILAN (INCOME TAX EXPENSES)                                           | -                    | -                              |
| **LABA BERSIH TAHUN BERJALAN (NET PROFIT FOR THE YEAR)**                                | **83.766.507.472**   | **89.524.860.890**             |
|                                                                                         |                      |                                |
| ==**PENGHASILAN KOMPREHENSIF LAIN (OTHER COMPREHENSIVE INCOME)**==                          |                      |                                |
| ==Pos yang tidak akan direklasifikasi ke laba rugi:==                                       |                      |                                |
| ==Pengukuran kembali atas imbalan pasca kerja (Remeasurement of employee benefits)==        | ==32.623.599==           | ==-==                              |
|                                                                                         |                      |                                |
| ==**JUMLAH LABA KOMPREHENSIF TAHUN BERJALAN (TOTAL COMPREHENSIVE INCOME FOR THE YEAR)**==   | ==**83.799.131.071**==   | ==**89.524.860.890**==             |
|                                                                                         |                      |                                |
| **LABA PER SAHAM DASAR (BASIC EARNINGS PER SHARE)**                                     | 11,97                | 12,79                          |
Dengan penyajian seperti ini, pembaca #laporan-keuangan dapat melihat:
- Berapa besar beban #imbalan-kerja yang mempengaruhi laba rugi operasional.
- Berapa besar kewajiban yang masih harus dibayarkan perusahaan kepada karyawan.
- Bagaimana perubahan asumsi atau hasil aktual (remeasurement) berdampak ke #OCI, bukan ke laba rugi.

Setelah memahami bagaimana hasil #Perhitungan-Aktuaria ditampilkan dalam #laporan-keuangan, pertanyaan berikutnya yang sering muncul dari manajemen dan auditor adalah:

> ***"Apa yang menyebabkan nilai kewajiban naik atau turun secara signifikan dari tahun ke tahun?"***

Perubahan dalam angka aktuaria tidak terjadi secara acak. Di balik setiap kenaikan atau penurunan kewajiban, **berbagai faktor teknis** yang bisa dianalisis dan dijelaskan. Mari kita pahami tentang faktor fluktuasi komponen-komponen utama aktuaria yang mempengaruhi #laporan-keuangan.

---
## Analisis #Komponen-Aktuaria

Nilai #Komponen-Aktuaria dapat berubah secara signifikan setiap tahun atau periode pelaporan. Agar #laporan-keuangan tetap akuntabel dan bisa dijelaskan kepada para pemangku kepentingan, penting untuk memahami penyebab perubahan tersebut dari berbagai kondisi aktuaria dan finansial perusahaan dari berbagai industri.

### Nilai Kini Kewajiban / #PVDBO
Atau *Present Value of Defined Benefit Obligation* – #PVDBO, merupakan gambaran total kewajiban perusahaan yang dihitung dengan memperhitungkan asumsi diskonto, #Kenaikan-Gaji, kematian, dan faktor demografi lainnya, seperti pada contoh perhitungan berikut:

| Reconciliation of liability and Asset       | **31 Desember 2024**   | **31 Desember 2023**  | Rekonsiliasi kewajiban dan Kekayaan             |
| ------------------------------------------- | ---------------------- | --------------------- | ----------------------------------------------- |
| Liability/(Assets) at Beginning of Period   | 8,440,942,097          | 12,807,667,528        | Kewajiban/(Kekayaan) pada Awal Periode          |
| Expense/(Income)                            | 5,207,196,396          | -2,070,226,743        | Beban/(Pendapatan)                              |
| Benefit Payment – Actual                    | (1,582,291,770)        | (1,385,421,237)       | Realisasi Pembayaran Manfaat                    |
| Company Contributions                       | -                      | -                     | Iuran Perusahaan                                |
| #Other-Comprehensive-Income                 | 7,135,040,019          | (7,253,503,551)       | Pendapatan Komprehensif Lainnya                 |
| ==**Liability/(Assets) at End of Period**== | ==**19,396,459,736**== | ==**8,440,942,097**== | ==**Kewajiban/(Kekayaan) pada Akhir Periode**== |

Perbedaan nilai kini kewajiban sebesar Rp10.955.517.639,-. Ini kenaikan besar sekali (lebih dari 2x lipat). Setelah ditelusuri, adapun ikhtisar data dan keuntungan / kerugian aktuarianya pada tabel berikut.

***Summary Data and Actuarial Assumption (Valuation as of 31 December 2024)***
**Ikhtisar Data dan [[Asumsi]] aktuaria  (Perhitungan per 31 Desember 2024)**

|     | Explanation                                                   | 31 Desember 2024 | 31 Desember 2023 | Uraian                                                    |
| --- | ------------------------------------------------------------- | ---------------- | ---------------- | --------------------------------------------------------- |
| 1   | **Basic Data and Assumptions**                                |                  |                  | **Data dan Asumsi**                                       |
| 2   | ==Number of Employee==                                            | ==**241**==      | ==**178**==      | ==Jumlah Karyawan (orang)==                                   |
| 3   | Monthly Wages for Permanent and Contract EEs                  | 3,816,852,767    | 2,439,485,140    | Jumlah Gaji Sebulan – Karyawan Tetap dan Kontrak          |
| 4   | Average Monthly Wages for Permanent and Contract EEs          | 15,837,563       | 13,704,973       | Rata-rata Gaji Sebulan                                    |
| 5   | Average Age of Permanent and Contract EEs (years)             | 34.88            | 35.13            | Rata-rata Usia (Tahun) – Karyawan Tetap dan Kontrak       |
| 6   | Average Years of Service (Years) – Permanent and Contract EEs | 4.19             | 4.81             | Rata-rata Masa Kerja (Tahun) – Karyawan Tetap dan Kontrak |
| 7   | Average Future Service (years)                                | 18.19            | 19.57            | Rata-rata Masa Kerja Yang Akan Datang (Tahun)             |
| 8   | Expected Remaining Working Lives (years)                      | 14.62            | 15.32            | Rata-rata Sisa Masa Kerja yang Diperkirakan (Tahun)       |
| 9   | Discount Rate Beginning Period                                | 6.86%            | 7.40%            | #Tingkat-Diskonto Awal Tahun                              |
| 10  | ==Discount Rate Ending Period==                               | ==**7.13%**==    | ==**6.86%**==    | ==#Tingkat-Diskonto Akhir Tahun==                         |

***(Gain) and Loss Calculations – Valuation as of 31 December 2024***
**Perhitungan Keuntungan / Kerugian Aktuarial – Perhitungan per 31 Desember 2024**

| No  | EXPLANATION                                       | 31 Desember 2024    | 31 Desember 2023    | URAIAN                                                        |
| --- | ------------------------------------------------- | ------------------- | ------------------- | ------------------------------------------------------------- |
| 1   | **Actual Present Value of Obligation at BoP**     | **8,440,942,097**   | **12,807,667,528**  | **Nilai Kini Kewajiban pada Awal Periode**                    |
| 2   | Past Service Cost - Non Vested                    | -                   | -                   | Biaya Jasa Lalu - Non Vested                                  |
| 3   | Past Service Cost - Vested                        | -                   | 129,316,432         | Biaya Jasa Lalu - Vested                                      |
| 4   | ==Interest Cost==                                 | ==380,307,373==     | ==815,784,343==     | ==#Biaya-Bunga==                                              |
| 5   | ==Current Service Cost==                          | ==3,793,592,485==   | ==2,227,764,821==   | ==#Biaya-Jasa-Kini==                                          |
| 6   | ==Benefit Payments==                              | ==(1,582,291,776)== | ==(1,383,451,237)== | ==Pembayaran Manfaat==                                        |
| 7   | Changes in Benefit Plans                          | -                   | -                   | Perubahan Program Manfaat                                     |
| 8   | Curtailment-Settlement                            | -                   | -                   | Kurtailmen-Penyelesaian                                       |
| 9   | **Present Value of Obligation at EoP - Expected** | **11,032,550,179**  | **14,597,079,576**  | **Nilai Kini Kewajiban pada Akhir Periode – Perkiraan**       |
| 10  | Actuarial (Gain) or Loss on Obligation            | 8,363,909,557       | (6,156,137,479)     | (Keuntungan)/Kerugian Aktuarial pada Kewajiban                |
| 11  | ==Change in Financial Assumption==                | ==(535,719,893)==   | ==302,303,182==     | ==Perubahan asumsi keuangan==                                 |
| 12  | Change in Demography Assumption                    | -                   | -                   | Perubahan asumsi demografi                                   |
| 13  | ==Experience Adjustment==                         | ==8,899,629,450==   | ==(6,458,440,659)== | ==Pengalaman penyesuaian==                                    |
| 14  | **Present Value of Obligation at EoP – Actual**   | **19,396,459,736**  | **8,440,942,097**   | **Nilai Kini Kewajiban pada Akhir Periode – Aktual**          |
| 15  | Actuarial (Gain) or Loss on Obligation            | 8,363,909,557       | (6,156,137,479)     | (Keuntungan)/Kerugian Aktuarial pada Kewajiban                |
| 16  | Actuarial (Gain)/Loss on Benefit Payment          | -                   | -                   | (Keuntungan)/Kerugian Aktuarial pada Pembayaran Manfaat       |
| 17  | Actuarial (Gain)/Loss on Plan Assets              | -                   | -                   | (Keuntungan)/Kerugian Aktuarial pada Nilai Wajar Aset Rencana |
| 18  | **Total Actuarial (Gain)/Loss for Period**        | **8,363,909,557**   | **(6,156,137,479)** | **Total (Keuntungan)/Kerugian Aktuarial Tahun Berjalan**      |

| **Faktor Penyebab**         | **Nilai (Rp)**    | **Pengaruh**                   |
| --------------------------- | ----------------- | ------------------------------ |
| #Biaya-Jasa-Kini            | 3.793.592.485     | ↑ Kenaikan kewajiban           |
| #Biaya-Bunga                | 380.307.373       | ↑ Kenaikan kewajiban           |
| Pembayaran Manfaat          | (1.582.291.776)   | ↓ Pengurangan kewajiban        |
| Perubahan Asumsi Keuangan   | (535.719.893)     | ↓ Pengurangan kewajiban        |
| **#Pengalaman-Penyesuaian** | **8.899.629.450** | **↑ Kenaikan kewajiban utama** |

#Pengalaman-Penyesuaian sebesar Rp8.899.629.450 menunjukkan adanya perubahan pada selain perubahan asumsi keuangan dan demografi, seperti:
- terdapat karyawan dengan masa kerja lebih panjang (sudah mencapai #Usia-Pensiun tetapi masih dipekerjakan)
- terdapat **perubahan komposisi karyawan** terlampir dalam perhitungan, sebagai berikut:
	- 2023 : tetap (76 orang), kontrak (102 orang)
	- 2024 : tetap (71 orang), kontrak (170 orang)
- adanya **penambahan manfaat** yang diperhitungkan pada tahun berjalan (tahun valuasi)
- adanya kemungkinan “under reserve” pada tahun sebelumnya

Manajemen perlu memahami bahwa bukan hanya perubahan asumsi #Tingkat-Diskonto yang berpengaruh, tapi realisasi aktual kondisi karyawan yang menyebabkan *experience adjustment* besar.

---
### #Biaya-Jasa-Lalu

Diringkas #BJS *(Past Service Cost)*, adalah biaya tambahan yang muncul ketika perusahaan mengubah program imbalan kerja sehingga hak manfaat karyawan atas masa kerja sebelumnya ikut berubah, baik menjadi lebih besar maupun lebih kecil, seperti pada contoh perhitungan berikut:

**Manfaat pensiun**

| **Cost Components**                                     | **31 Desember 2024** | **31 Desember 2023** | **Komponen Biaya **                                        |
| ------------------------------------------------------- | -------------------- | -------------------- | ---------------------------------------------------------- |
| Current Service Cost                                    | 4,974,876,505        | 4,022,249,000        | #Biaya-Jasa-Kini                                           |
| Interest Cost                                           | 1,972,887,944        | 1,665,561,000        | #Biaya-Bunga                                               |
| Expected Return on Plan Assets                          | -                    | -                    | Harapan dari Hasil Investasi                               |
| ==Immediate Recognition of Past Service Cost – Vested==     | ==59,919,081==           | ==3,179,615,000==        | ==Pengakuan Segera dari #Biaya-Jasa-Lalu yang Vested==         |
| Curtailment Effect / Settlement                         | -                    | -                    | Dampak Kurtailmen / Penyelesaian                           |
| **Expense/(Income) Recognized in the Income Statement** | **7,007,683,530**    | **8,867,425,000**    | **Beban/(Pendapatan) yang Diakui dalam Laporan Laba/Rugi** |

Manfaat #OLTEB

| **Cost Components**                                 | **31 Desember 2024** | **31 Desember 2023** | **Komponen Biaya **                                |
| --------------------------------------------------- | -------------------- | -------------------- | -------------------------------------------------- |
| Current Service Cost                                | 523,219,701          | 1,485,788,000        | #Biaya-Jasa-Kini                                   |
| Interest Cost                                       | 619,304,628          | 594,472,000          | Biaya Bunga                                        |
| Expected Return on Plan Assets                      | -                    | -                    | Harapan dari Hasil Investasi                       |
| ==Immediate Recognition of Past Service Cost – Vested== | ==(6,458,697,633)==      | ==(1,430,907,000)==      | ==Pengakuan Segera dari #Biaya-Jasa-Lalu yang Vested== |
| Curtailment Effect / Settlement                     | -                    | -                    | Dampak Kurtailmen / Penyelesaian                   |
| **Expense/(Income) in Income Statement**            | **(5,316,173,304)**  | **649,353,000**      | **Beban/(Pendapatan) dalam Laporan Laba/Rugi**     |

Fluktuasi #Biaya-Jasa-Lalu ( #BJS) umumnya terjadi karena:
- Adanya **perubahan program manfaat**: misalnya menaikkan formula perhitungan #Pesangon atau pensiun.
- Adanya **mutasi karyawan**
- Adanya **perubahan** #Usia-Pensiun

#### Perlakuan Akuntansi #Biaya-Jasa-Lalu :
- Dicatat langsung sebagai beban dalam #Laporan-Laba-Rugi saat hak tersebut diberikan (vested).
- Tidak masuk ke #OCI, karena dampaknya berasal dari keputusan manajemen, **bukan karena perubahan asumsi atau pengalaman aktuaria.**

>- Tidak tergantung apakah manfaat itu dibayarkan sekarang atau nanti.
>- Pengakuannya bisa menjadi beban (jika manfaat bertambah) atau justru pendapatan (jika manfaat dikurangi)

---
Untuk memahaminya, berikut pemahaman pengakuan #Biaya-Jasa-Lalu suatu perusahaan dengan adanya imbalan jangka panjang lainnya.
#### Manfaat pensiun

***Summary Data and Actuarial Assumption*** ***Valuation as of 31 December 2024***
**Ikhtisar Data dan [[Asumsi]] aktuaria Perhitungan per 31 Desember 2024**

|     | **Explanation**                                        | **31 Desember 2024** | **31 Desember 2023** | **Uraian**                                          |
| --: | ------------------------------------------------------ | -------------------- | -------------------- | --------------------------------------------------- |
|   1 | ==Number of Employee==                                 | ==1701==             | ==1781==             | ==Jumlah Karyawan (orang)==                         |
|   2 | Monthly Wages for Permanent and Contract EEs           | 7,576,331,271        | 7,392,890,709        | Jumlah Gaji Sebulan – Karyawan Tetap dan Kontrak    |
|   3 | Average Monthly Wages                                  | 4,454,045            | 4,151,794            | Rata-rata Gaji Bulanan                              |
|   4 | Average Age of Permanent and Contract EEs              | 46.75                | 46.06                | Rata-rata Usia (Tahun) – Karyawan Tetap dan Kontrak |
|   5 | ==Average Years of Service (Years)==                   | ==6.47==             | ==5.97==             | ==Rata-rata Masa Kerja (Tahun)==                    |
|   6 | Average Future Service (Years)                         | 20.29                | 20.28                | Rata-rata Sisa Masa Kerja                           |
|   7 | Average Expected Remaining Working Lives (years)       | 16.05                | 16.05                | Rata-rata Sisa Masa Kerja Diperkirakan              |
|   8 | Discount Rate Beginning Period                         | 6.80%                | 7.35%                | #Tingkat-Diskonto Awal Tahun                        |
|   9 | Discount Rate Ending Period                            | 7.13%                | 6.80%                | #Tingkat-Diskonto Akhir Tahun                       |
|  10 | Expected Return on Plan Assets                         | -                    | -                    | Harapan atas Hasil Investasi Aktiva Program         |
|  11 | Future Salary Increases (per annum)                    | 6.00%                | 6.00%                | Tingkat #Kenaikan-Gaji (per Tahun)                  |
|  12 | Current Service Cost                                   | 523,219,701          | 1,485,788,000        | #Biaya-Jasa-Kini                                    |
|  13 | Interest Cost                                          | 619,304,628          | 594,472,000          | Biaya Bunga                                         |
|  14 | Total Benefit Paid in Year                             | (1,582,291,770)      | (1,385,421,237)      | Imbalan yang dibayarkan                             |
|  15 | Company Contribution Paid in Year                      | -                    | -                    | Iuran yang dibayarkan oleh perusahaan               |
|  16 | Present Value of Obligation at BoP                     | 20,885,030,088       | 18,250,475,000       | Nilai kini kewajiban awal tahun                     |
|  17 | Present Value of Obligation at EoP                     | 20,036,115,586       | 20,885,030,088       | Nilai kini kewajiban akhir tahun                    |
|  18 | Past Service Cost – Non Vested at BoP                  | -                    | -                    | #Biaya-Jasa-Lalu – non-vested                       |
|  19 | ==Past Service Cost – Vested at BoP==                  | ==59,919,081==       | ==3,179,615,000==    | ==#Biaya-Jasa-Lalu – vested==                       |
|  20 | Fair Value of Plan Asset Program – Beginning of Period | -                    | -                    | Nilai wajar aktiva program awal periode             |
|  21 | Fair Value of Plan Asset Program – End of Period       | -                    | -                    | Nilai wajar aktiva program akhir periode            |

| **Komponen**            | **Catatan**         |
|-------------------------|---------------------|
| Jumlah karyawan total   | ↓ -80 orang         |
| Rata-rata masa kerja    | ↑ meningkat         |
| #BJS vested              | ↓ menurun           |
Penjelasan #BJS **positif** sebagai penambah beban:
1. Bisa muncul karena **perubahan kebijakan minor,** misalnya:
    - Revisi masa kerja dihitung ulang
    - Penyesuaian bagi karyawan yang sebelumnya belum diakui manfaatnya (misalnya: kontrak jadi tetap)
2. Nilainya kecil (±0,85% dari total beban), jadi kemungkinan bukan perubahan besar, **hanya pengakuan tambahan atas hak karyawan tertentu.**

##### Manfaat #OLTEB

***Summary Data and Actuarial Assumption Valuation as of 31 December 2024***
**Ikhtisar Data dan [[Asumsi]] aktuaria Perhitungan per 31 Desember 2024**

|     | **EXPLANATION**                                        | **31 Desember 2024** | **31 Desember 2023** | **URAIAN**                                        |
| --: | ------------------------------------------------------ | -------------------: | -------------------: | ------------------------------------------------- |
|   1 | **Basic Data and Assumptions**                         |                      |                      | **Data dan Asumsi**                               |
|   2 | ==Number of Employee==                                 |        ==**1,697**== |        ==**1,781**== | ==Jumlah Karyawan (orang)==                       |
|   3 | Monthly Wages for Contract EEs                         |        7,357,531,271 |        7,392,980,709 | Jumlah Gaji Sebulan – Karyawan Kontrak                |
|   4 | Average Monthly Wages for Contract EEs                 |            4,335,611 |            4,151,028 | Rata-rata Gaji Bulanan                            |
|   5 | Average Age of Contract EEs (Years)                    |                36.71 |                36.09 | Rata-rata Usia (Tahun) – Karyawan Kontrak             |
|   6 | ==Average Years of Service (Years) – Contract EEs==    |         ==**6.48**== |         ==**5.97**== | ==Rata-rata masa kerja (Tahun) – Karyawan Kontrak==   |
|   7 | Average Future Service (Years)                         |                20.29 |                20.29 | Rata-rata Masa Kerja yang Akan Datang (Tahun)     |
|   8 | Average Expected Remaining Working Lives (Years)       |                17.01 |                12.09 | Rata-rata Sisa Masa Kerja yang Diharapkan (Tahun) |
|   9 | Discount Rate Beginning Period                         |                6.80% |                7.35% | #Tingkat-Diskonto Awal Tahun                      |
|  10 | Discount Rate Ending Period                            |                7.13% |                6.80% | #Tingkat-Diskonto Akhir Tahun                     |
|  11 | Expected Rate of Return on Plan Assets                 |                    - |                    - | Tingkat Harapan Investasi atas Aktiva Program     |
|  12 | Future Salary Increases (per annum)                    |                6.00% |                6.00% | Tingkat #Kenaikan-Gaji (per Tahun)                |
|  13 | Current Service Cost                                   |          523,219,701 |        1,485,788,000 | #Biaya-Jasa-Kini                                  |
|  14 | Total Benefit Paid in Year                             |      (1,582,291,770) |      (1,385,421,237) | Imbalan yang dibayarkan                           |
|  15 | Company Contribution Paid in Year                      |                    - |                    - | Iuran yang dibayarkan                             |
|  16 | Present Value of Obligation at BoP                     |        9,107,421,001 |        8,458,015,000 | Nilai kini kewajiban awal periode                 |
|  17 | Present Value of Obligation at EoP                     |        9,917,296,001 |        9,107,421,001 | Nilai kini kewajiban akhir periode                |
|  18 | Past Service Cost – Non Vested at BoP                  |                    - |                    - | #Biaya-Jasa-Lalu – non-vested                     |
|  19 | ==Past Service Cost – Vested at BoP==                      |  ==**(8,793,231,900)**== |                    ==-== | ==#Biaya-Jasa-Lalu – vested==                         |
|  20 | Fair Value of Plan Asset Program – Beginning of Period |                    - |                    - | Nilai wajar aktiva program awal periode           |
|  21 | Fair Value of Plan Asset Program – End of Period       |                    - |                    - | Nilai wajar aktiva program akhir periode          |

| **Komponen**                       | **Catatan**                      |
|-----------------------------------|----------------------------------|
| Jumlah karyawan kontrak           | ↓ -84 orang                      |
| Rata-rata masa kerja kontrak      | ↑ meningkat                      |
| #BJS vested                        | ↓ perubahan besar                |
| **#BJS yang diakui di laba rugi**  | **! pendapatan meningkat**       |
Penjelasan #BJS negatif diakui sebagai **pendapatan (pengurang kewajiban):**
- Meskipun rata-rata masa kerja meningkat, penurunan jumlah karyawan berarti banyak yang keluar ***sebelum mencapai milestone*** (10–30 tahun).
- Karena program #OLTEB berbasis milestone, jika karyawan keluar ***sebelum masa kerja tertentu, maka hak atas apresiasi hangus,*** dan cadangan dihapuskan.
- Ini menghasilkan #BJS negatif yang besar dan langsung diakui.

> Fluktuasi #Biaya-Jasa-Lalu mencerminkan **penyesuaian hak karyawan dan efisiensi program.** Bagi manajemen, hal ini penting untuk mengevaluasi kebijakan manfaat dan menjaga keberlanjutan beban keuangan jangka panjang.

---
### #Realisasi-Manfaat

Merupakan pembayaran aktual kepada karyawan selama periode berjalan, yang biasanya akan **mengurangi kewajiban aktuaria jika telah dicadangkan.** Jumlah ini mencerminkan manfaat yang telah dibayar langsung, dan dapat memengaruhi kewajiban yang tersisa pada akhir periode.

Total pembayaran manfaat tersebut mencakup dua jenis kondisi karyawan:

| **PHK**                                                   | **Non-PHK**                                                                                |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| #Pesangon sesuai UU / PP                                  | Imbalan pensiun normal                                                                     |
| Uang penghargaan masa kerja                               | Uang manfaat meninggal                                                                     |
| Uang penggantian hak                                      | Apresiasi masa kerja saat pensiun                                                          |
| Komponen #Imbalan-pasca-kerja yang masih berlaku saat PHK | Klaim atas manfaat jangka panjang (misalnya OTLEB) / Program #DPLK dibayarkan saat pensiun |
Fluktuasi #Realisasi-Manfaat umumnya disebabkan oleh:
1. **Ketepatan dan kelengkapan data realisasi,** terutama jika realisasi tersebut dikirimkan oleh perusahaan dalam bentuk rekap Excel terpisah;
2. **Adanya perubahan regulasi internal perusahaan atau UU Ketenagakerjaan** yang mengubah komponen manfaat yang wajib dibayarkan.

Salah satu catatan penting dalam analisis ini adalah **apakah data realisasi sudah mencakup PHK atau tidak.** Ketika realisasi tidak disertakan (exclude) #PHK, maka manfaat yang seharusnya diakui bisa tidak tercermin sepenuhnya dalam hasil perhitungan, dan dapat menimbulkan ***selisih signifikan*** pada rekonsiliasi antara manfaat aktual dan proyeksi kewajiban, contohnya seperti pada tabel berikut.

| Reconciliation of Liability and Asset     | **31 Desember 2024** | **31 Desember 2023** | Rekonsiliasi Kewajiban dan Kekayaan         |
| ----------------------------------------- | -------------------: | -------------------: | ------------------------------------------- |
| Liability/(Assets) at Beginning of Period |        8,440,942,097 |       12,807,667,528 | Kewajiban/(Kekayaan) pada Awal Periode      |
| Expense/(Income)                          |        5,402,769,396 |        4,270,229,741 | Beban/(Pendapatan)                          |
| ==Benefit Payment – Actual==                  |      ==(1,582,291,776)== |      ==(1,383,451,237)== | ==Realisasi Pembayaran Manfaat==                |
| Company Contributions                     |                    – |                    – | Iuran Perusahaan                            |
| #Other-Comprehensive-Income               |        7,135,040,019 |      (7,253,503,935) | Pendapatan Komprehensif Lainnya             |
| **Liability/(Assets) at End of Period**   |   **19,396,459,736** |    **8,440,942,097** | **Kewajiban/(Kekayaan) pada Akhir Periode** |
Untuk memahaminya, berikut pemahaman bagaimana #Realisasi-Manfaat tercakup dalam perhitungan #Imbalan-pasca-kerja.
1. Total benefit paid (realisasi): Rp1.582.291.776 (tahun 2024)
2. Adanya catatan menyatakan:
	 **“Realisasi include PHK sesuai file yang diterima”**
	 **“Tidak ada perubahan data karyawan”**
3. Tidak ada perubahan jumlah asumsi atau skema manfaat.
4. Jumlah karyawan naik dari 178 (2023) → 241 (2024), artinya tidak terjadi penurunan massal.

|     | **EXPLANATION**                   |                                                                                         **31 Desember 2024** |                                                                                     **31 Desember 2023** | **URAIAN**                   |
| --: | --------------------------------- | -----------------------------------------------------------------------------------------------------------: | -------------------------------------------------------------------------------------------------------: | ---------------------------- |
|  22 | Mortality Table                   |                                                                                                       TMI IV |                                                                                                   TMI IV | Tabel Mortalita              |
|  23 | Disability Rate                   |                                                                                            5.00% dari TMI IV |                                                                                        5.00% dari TMI IV | Tingkat Cacat                |
|  24 | Withdrawal Rate                   | ≤ 16 = 0.00%<br>17 – 39 = 5.00%<br>40 – 44 = 3.00%<br>45 – 49 = 2.00%<br>50 – 54 = 1.00%<br>≥ 55 = 0.00%<br> | ≤ 16 = 0.00%<br>17 – 39 = 5.00%<br>40 – 44 = 3.00%<br>45 – 49 = 2.00%<br>50 – 54 = 1.00%<br>≥ 55 = 0.00% | Tingkat Pengunduran Diri     |
|  25 | Actuarial Calculation Method      |                                                                                                  #PUC (IFRIC) |                                                                                              #PUC (IFRIC) | Metode #Perhitungan-Aktuaria  |
|  26 | Normal Retirement Age (Years old) |                                                                                                           55 |                                                                                                       55 | #Usia-Pensiun Normal (Tahun) |
#### Penilaian Terhadap #Realisasi-Manfaat
Meskipun terdapat #PHK, realiasi manfaat PHK tetap dimasukkan dalam komponen benefit paid. Ini bisa diterima secara aktuaria jika:
1. PHK sudah menjadi bagian dari asumsi model (misalnya asumsi withdrawal/exit rate).
2. Jumlah PHK tidak bersifat luar biasa (non-routine) atau bukan karena restrukturisasi besar-besaran.
3. Nilai pembayaran masih relatif konsisten dengan tren tahun sebelumnya, yaitu:
    - 2023: Rp1,38 M
    - 2024: Rp1,58 M (+15%), masih wajar dibanding kenaikan karyawan + kinerja

***Gain and Loss Calculations  Valuation as of 31 Desember 2024***
**Perhitungan (Keuntungan)/Kerugian Aktuarial per 31 Desember 2024**

| No  | EXPLANATION                                       | 31 Desember 2024    | 31 Desember 2023    | URAIAN                                                          |
| --- | ------------------------------------------------- | ------------------- | ------------------- | --------------------------------------------------------------- |
| 1   | **Actual Present Value of Obligation at BoP**     | **8,440,942,097**   | **12,807,667,528**  | **Nilai Kini Kewajiban pada Awal Periode**                      |
| 2   | Past Service Cost - Non Vested                    | -                   | -                   | Biaya Jasa Lalu - Non Vested                                    |
| 3   | Past Service Cost - Vested                        | -                   | 129,316,532         | Biaya Jasa Lalu – Vested                                        |
| 4   | ==Interest Cost==                                 | ==380,307,373==     | ==815,784,343==     | ==#Biaya-Bunga==                                                |
| 5   | ==Current Service Cost==                          | ==3,793,592,485==   | ==2,227,762,410==   | ==#Biaya-Jasa-Kini==                                            |
| 6   | ==Benefit Payments==                              | ==(1,582,291,776)== | ==(1,383,451,237)== | ==Pembayaran Manfaat==                                          |
| 7   | Changes in Benefit Plans                          | -                   | -                   | Perubahan Program Manfaat                                       |
| 8   | Curtailment-Settlement                            | -                   | -                   | Kurtailmen-Penyelesaian                                         |
| 9   | **Present Value of Obligation at EoP – Expected** | **11,032,550,179**  | **14,597,079,576**  | **Nilai Kini Kewajiban pada Akhir Periode – Perkiraan**         |
| 10  | Actuarial (Gain) or Loss on Obligation            | 8,363,909,557       | (6,156,137,479)     | (Keuntungan)/Kerugian Aktuarial pada Kewajiban                  |
| 11  | ==Change in Financial Assumption==                | ==(535,719,893)==   | ==302,303,180==     | ==Perubahan asumsi keuangan==                                   |
| 12  | Change in Demography Assumption                   | -                   | -                   | Perubahan asumsi demografi                                      |
| 13  | ==Experience Adjustment==                             | ==8,899,629,450==       | ==(6,458,440,659)==     | ==#Pengalaman-Penyesuaian==                                         |
| 14  | **Present Value of Obligation at EoP – Actual**   | **19,396,459,736**  | **8,440,942,097**   | **Nilai Kini Kewajiban pada Akhir Periode – Aktual**            |
| 15  | Actuarial (Gain) or Loss on Obligation            | 8,363,909,557       | (6,156,137,479)     | (Keuntungan)/Kerugian Aktuarial pada Kewajiban                  |
| 16  | Actuarial (Gain)/Loss on Benefit Payment          | -                   | -                   | (Keuntungan)/Kerugian Aktuarial pada Pembayaran Manfaat         |
| 17  | Actuarial (Gain)/Loss on Plan Assets              | -                   | -                   | (Keuntungan)/Kerugian Aktuarial pada Nilai Wajar Aktiva Program |
| 18  | **Total Actuarial (Gain)/Loss for Period**        | **8,363,909,557**   | **(6,156,137,479)** | **Total (Keuntungan)/Kerugian Aktuarial Tahun Berjalan**        |
***Catatan:***
Ketika perusahaan membayar manfaat aktual (realisasi) kepada karyawan, sering kali jumlah yang dibayar **tidak sama dengan** nilai kewajiban yang dihitung secara aktuaria. Selisih ini bisa terjadi karena perubahan manfaat, koreksi data, atau asumsi.

| **Realisasi < Liabilitas**                                        | **Realisasi > Liabilitas**                                                |
|-------------------------------------------------------------------|---------------------------------------------------------------------------|
| Dianggap sebagai keuntungan aktuaria (*gain*)                     | Biasanya mencerminkan tambahan manfaat yang belum sempat dicatat         |
| Selisihnya masuk ke **#OCI** (*#Other-Comprehensive-Income*)        | Selisihnya masuk sebagai **#Biaya-Jasa-Lalu** (*Past Service Cost*)       |
Dalam praktik audit dan aktuaria, berikut adalah beberapa kondisi yang sering muncul:

| **Situasi**                                     | **Perlakuan**                                                                                                                             |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| #PHK rutin                                      | Dimasukkan sebagai *benefit paid* (dibayarkan sebagai bagian dari estimasi), selaras dengan asumsi → **VALID**                            |
| #PHK massal atau restrukturisasi                | Harus dicatat sebagai **#Biaya-Jasa-Lalu** atau **perubahan program**, bukan pengurang kewajiban → **tidak valid** sebagai *benefit paid* |
| Tidak ada data PHK, tapi realisasi besar        | Auditor biasanya minta klarifikasi                                                                                                        |
| Ada perbedaan realisasi dan [[Asumsi]] aktuaria | Masuk sebagai *actuarial loss/gain*, **bukan** *benefit paid*                                                                             |
>Dalam kasus ini, karena ***tidak ada data perubahan karyawan dan PHK sudah diterima dalam rekap,*** maka asumsi bahwa pembayaran PHK masuk dalam skema dihitung valid.

Jadi, #Realisasi-Manfaat sebesar Rp1,58 M tahun 2024, termasuk PHK, telah dihitung sebagai pengurang kewajiban secara wajar karena:
- Sudah masuk dalam file resmi,
- Tidak ada perubahan data karyawan,
- Konsisten dengan asumsi dan tren tahun sebelumnya.

Selesai dengan pemahaman analisis fluktuasi #komponen-aktuaria, bab selanjutnya berisi tentang **kumpulan pertanyaan yang sering menjadi isu aktuaria, dilengkapi jawaban komprehensif** dalam #Perhitungan-Aktuaria #Imbalan-pasca-kerja.

---
# BAB 5 Frequently Asked Questions #FAQ

## #FAQ Legal

List ini merupakan frequently asked questions terkait **Legal / Hukum Perhitungan Aktuaria KKA Nirmala**

Note: 
- 'kami' pada format Heading 4 merujuk pada point of view user atau klien (penanya)
- 'kami' pada format paragraph merujuk pada point of view KKA Nirmala (penjawab)

---
#### 1. Apa itu aktuaria? Mengapa #Perhitungan-Aktuaria tersebut dibutuhkan?

Aktuaria adalah ilmu yang menggabungkan matematika, statistik, dan keuangan untuk mengelola risiko, terutama dalam #imbalan-kerja, asuransi, dan investasi jangka panjang. #Perhitungan-Aktuaria digunakan untuk mengestimasi dan mencatat kewajiban perusahaan terkait manfaat karyawan, **seperti #Pesangon dan pensiun, sesuai standar akuntansi seperti #PSAK-219.**

Hasil #Perhitungan-Aktuaria tidak wajib dilaporkan langsung ke OJK atau Dinas Ketenagakerjaan, kecuali bagi perusahaan di bawah regulasi tertentu seperti asuransi dan dana pensiun. Namun, bagi perusahaan yang diawasi OJK, **#Laporan-Aktuaria sering menjadi bagian dari audit keuangan tahunan.**

Untuk memastikan kepatuhan terhadap aturan ketenagakerjaan, perusahaan juga perlu merujuk pada **PP No. 35 Tahun 2021, yang mengatur kompensasi bagi karyawan PKWT dan PKWTT.** Meskipun #Laporan-Aktuaria tidak wajib diserahkan ke regulator, hasilnya sangat penting untuk audit, pengambilan keputusan keuangan, dan perencanaan dana #imbalan-kerja.

#### 2. Siapa yang berhak atau boleh sesuai peraturan untuk menyusun dan membuat #Laporan-Aktuaria?

Pihak yang berhak membuat #Laporan-Aktuaria, antara lain:

1. #Aktuaris **Bersertifikasi dari PAI dan Berizin dari OJK**

	Hanya #aktuaris yang memiliki gelar FSAI (Fellow of the Society of Actuaries of Indonesia) dari Persatuan Aktuaris Indonesia (PAI) yang berhak menyusun dan menandatangani #Laporan-Aktuaria.
	Untuk **sektor keuangan** seperti asuransi, dana pensiun, dan perbankan, penyusunan #Laporan-Aktuaria harus dilakukan oleh #aktuaris yang memiliki izin praktik dari Otoritas Jasa Keuangan (OJK) sesuai regulasi yang berlaku.

2. Kantor #Konsultan-Aktuaria **(KKA) yang Terdaftar**

	Perusahaan yang membutuhkan #Laporan-Aktuaria, terutama untuk kepentingan audit #PSAK-219 atau perhitungan kewajiban #imbalan-kerja, harus menggunakan jasa Kantor Konsultan Aktuaria (KKA) yang memiliki #aktuaris bersertifikasi dan terdaftar.

3. **Aktuaris Internal di Perusahaan Asuransi &** #Dana-Pensiun

	Perusahaan asuransi dan dana pensiun dapat menyusun #Laporan-Aktuaria melalui #aktuaris internal yang memiliki izin praktik resmi dan bertanggung jawab atas perhitungan kewajiban aktuaria serta manajemen risiko.

#### 3. Bagaimana jika perusahaan belum memiliki #Peraturan-Perusahaan?

Perusahaan yang belum memiliki #Peraturan-Perusahaan dapat mengisi formulir yang akan kami sediakan.

#### 4. Bagaimana jika realisasi yang dibayarkan berbeda dengan yang telah dicadangkan?

Realisasi dapat berpedoman pada undang-undang yang berlaku atau PP jika ada. Tidak masalah jika besaran yang dibayarkan berbeda dari yang dicadangkan karena akan terkoreksi pada laporan berikutnya.

#### 5. Apa perbedaan #PSAK-219 dan #SAK-ETAP?

#PSAK-219 digunakan oleh perusahaan yang wajib melaporkan #laporan-keuangannya ke publik, seperti perusahaan terbuka atau entitas yang diawasi oleh regulator. Sementara itu, #SAK-ETAP digunakan oleh perusahaan kecil hingga menengah yang tidak diwajibkan menyampaikan #laporan-keuangan ke publik.

Dalam #SAK-ETAP, tidak ada perhitungan OCI. Artinya, dalam #SAK-ETAP, keuntungan atau kerugian akibat perubahan asumsi aktuaria langsung dicatat dalam #Laporan-Laba-Rugi tahun berjalan, berbeda dengan #PSAK-219 yang mencatatnya di OCI untuk menghindari fluktuasi besar dalam laba rugi.

#### 6. Apa perbedaan #UUK dan #UUCK?

1. #UUK **(Undang-Undang Ketenagakerjaan No. 13 Tahun 2003)** :

	Mengatur hak-hak karyawan secara lebih ketat, termasuk kompensasi #Pesangon, uang penghargaan masa kerja (UPMK), dan uang penggantian hak (UPH) bagi karyawan yang mengalami #PHK atau pensiun.

2. #UUCK **(Undang-Undang Cipta Kerja No. 11 Tahun 2020 dan aturan turunannya, PP No. 35 Tahun 2021)** :

	- Mengurangi nilai #Pesangon maksimum bagi karyawan yang terkena #PHK dari 32 bulan gaji (versi #UUK) menjadi 25 bulan gaji.
	- Menyederhanakan skema kompensasi bagi karyawan kontrak (PKWT).
	- Menambahkan Jaminan Kehilangan Pekerjaan (JKP) sebagai bagian dari manfaat yang diberikan kepada pekerja yang terkena #PHK.

**Hingga saat ini, kedua UU tersebut masih menjadi landasan pokok** dalam perhitungan #imbalan-kerja sesuai #PSAK-219. Perbedaan utama terletak pada besaran manfaat yang diberikan.

Manfaat yang diberikan menurut #UUCK lebih kecil daripada #UUK.

- Menurut #UUCK adalah (1,75 x #Pesangon + 1 x Uang Penghargaan Masa Kerja)
- Menurut #UUK adalah (2 x #Pesangon + 1 x Uang Penghargaan Masa Kerja).

#### 7. Apa itu #IFRIC? Apa bedanya dengan IAS?

#IFRIC AD adalah aturan yang memberikan panduan lebih rinci dalam mengalokasikan #Imbalan-pasca-kerja, seperti pensiun. Misalnya, cadangan pensiun **mulai dihitung 24 tahun sebelum usia pensiun.**

Penerapan **IFRIC AD** mengurangi nilai kewajiban #imbalan-kerja dibandingkan sebelumnya, yang berdampak pada #laporan-keuangan perusahaan. Keuntungan dari perubahan ini langsung diakui dalam beban #Imbalan-pasca-kerja tahun berjalan. Selain itu, #IFRIC 14 menjelaskan batas pengakuan aset dari surplus dana pensiun di IAS 19. **IAS 19** mengatur metode perhitungan liabilitas #imbalan-kerja, asumsi aktuaria, dan lainnya.

Intinya:

- IAS 19 = Standar utama untuk mengatur akuntansi #imbalan-kerja.
- #IFRIC = Penafsiran tambahan untuk menjawab pertanyaan atau situasi yang tidak dijelaskan secara rinci di IAS / IFRS.

#### 8. Apa perbedaan #SAK-EP dan #SAK-ETAP?

**SAK-EP (Standar Akuntansi Keuangan Entitas Privat)** :

- Berlaku mulai 1 Januari 2025 sebagai pengganti #SAK-ETAP.
- Disusun berdasarkan IFRS for SMEs versi 2015, namun sudah diadopsi penuh dan diterjemahkan oleh IAI.
- Terdapat #Other-Comprehensive-Income (OCI).
- Lebih komprehensif dan sistematis, mencakup penilaian kewajiban manfaat pasca kerja dengan metode aktuaria #Projected-Unit-Credit.
- Memberikan opsi yang lebih fleksibel untuk entitas privat dalam menyusun #laporan-keuangan secara sederhana namun sesuai standar internasional.

**SAK-ETAP (Entitas Tanpa Akuntabilitas Publik)** :

- Diterbitkan pada tahun 2009, disusun untuk entitas skala kecil-menengah yang tidak memiliki akuntabilitas publik dan tidak menerbitkan #laporan-keuangan untuk umum.
- Tidak ada perhitungan OCI. Artinya, dalam #SAK-ETAP, keuntungan / kerugian akibat perubahan asumsi aktuaria langsung dicatat dalam #Laporan-Laba-Rugi tahun berjalan.
- Pendekatan lebih sederhana, contohnya:
    - #Imbalan-pasca-kerja dihitung tanpa metode aktuaria (cukup dengan estimasi sederhana).
    - Instrumen keuangan tidak perlu dinilai dengan nilai wajar secara kompleks.
- Akan digantikan sepenuhnya oleh #SAK-EP mulai 2025.

#### 9. Apakah boleh pakai #SAK-ETAP di tahun 2023 dan mulai pakai #SAK-EP di 2024?

#SAK-ETAP masih boleh digunakan hingga akhir tahun 2024. #SAK-EP secara efektif berlaku untuk periode tahun buku yang **dimulai 1 Januari 2025**. Namun, early adoption (penerapan dini) #SAK-EP sudah diperbolehkan sejak beberapa tahun terakhir.

• Jadi jika entitas sudah menggunakan #SAK-EP di 2024, itu sah.
• Tapi jika masih pakai #SAK-ETAP di 2023 dan 2024, juga masih diperbolehkan.

#### 10. Jika perusahaan memilikib #Peraturan-Perusahaan, apakah mengacu kepada PP atau UU?

Umumnya, dapat dilihat dari apakah PP tersebut mengacu pada #UUCK atau #UUK. Jika tidak tersirat, perbandingan manfaat-manfaatnya dapat dilakukan. Jika manfaat dari **PP lebih kecil daripada UU, maka perhitungan akan mengikuti UU.**

#### 11. Apakah Direksi berhak atas #Pesangon?

Berdasarkan Pasal 15 **UU No. 40 Tahun 2007**, direksi berbeda dengan karyawan. Direksi umumnya memiliki **kontrak sendiri**, sehingga umumnya tidak diikutsertakan dalam perhitungan #Imbalan-pasca-kerja.

#### 12. Apakah Direksi bertanggung jawab terhadap pengelolaan #imbalan-kerja?

Ya, sesuai dengan Pasal 15 UU No. 40 Tahun 2007, Direksi adalah organ Perseroan yang bertanggung jawab penuh atas pengurusan Perseroan, termasuk dalam hal pengelolaan #imbalan-kerja. Ini berarti Direksi harus **memastikan bahwa perusahaan memenuhi semua kewajiban hukum terkait #imbalan-kerja,** baik untuk pekerja lokal maupun TKA.

#### 13. Apa manfaat yang diberikan berbeda, antara karyawan di #PHK karena meninggal dunia, sakit berkepanjangan, mengundurkan diri?

Semua karyawan yang **pensiun normal** berhak atas:
- #Pesangon
- Uang penghargaan masa kerja
- Uang pisah (jika ada)
- Uang penggantian hak (jika ada)

Bila di #PHK dengan sebab tertentu, berhak atas manfaat yang sama dengan tambahan, antara lain:
- Yang karena meninggal dunia: bantuan **uang duka** (jika ada).
- Yang karena sakit berkepanjangan: bantuan **uang cacat** (jika ada).
- Yang karena mengundurkan diri: **uang pisah/apresiasi** (jika ada).

Semua jenis #PHK dicatat dalam #Perhitungan-Aktuaria untuk memastikan akurasi dalam #laporan-keuangan dan mencerminkan kewajiban yang sudah terjadi maupun proyeksi di masa depan.

Bonus akibat #PHK tidak dihitung sebagai kewajiban aktuaria jangka panjang ( #PVDBO). Sebaliknya, bonus ini dicatat sebagai **biaya operasional tahun berjalan atau termasuk dalam curtailment**, yaitu pengurangan kewajiban karena penghentian hubungan kerja.

#### 14. Bagaimana pengaturan terkait TKA mempengaruhi #Perhitungan-Aktuaria?

Tenaga Kerja Asing (TKA) yang dipekerjakan pada jabatan tertentu dan periode tertentu dapat memengaruhi #Perhitungan-Aktuaria, karena aktuaria harus mempertimbangkan durasi kontrak kerja dan jabatan yang dipegang untuk menentukan kewajiban #imbalan-kerja.

Berdasarkan #PSAK-24 paragraf 1 dan 2, #Imbalan-pasca-kerja dapat diberikan kepada karyawan yang memiliki hubungan kerja formal dengan entitas pelapor, melalui program resmi maupun tidak resmi, tanpa membedakan kewarganegaraan. Oleh karena itu, TKA dapat termasuk dalam ruang lingkup PSAK 24 jika memenuhi kriteria berikut:

- Memiliki **kontrak kerja yang sah** dengan entitas pelapor.
- **Termasuk dalam kebijakan perusahaan** yang memberikan #Imbalan-pasca-kerja, baik dalam bentuk program pensiun, #Pesangon, atau bentuk imbalan lainnya.
- **Tidak dikecualikan secara eksplisit** dalam perjanjian kerja atau #Peraturan-Perusahaan.

#### 15. Apa pemberian uang kompensasi juga berlaku bagi TKA dengan PKWT?

Menurut Pasal 15 PP No. 35 Tahun 2021, pemberian uang kompensasi **tidak berlaku bagi TKA yang dipekerjakan dengan Perjanjian Kerja Waktu Tertentu (PKWT)**. Hal ini sering membingungkan, karena mungkin adanya asumsi semua pekerja berhak atas kompensasi. Tetapi, jika **dalam #Peraturan-Perusahaan** dinyatakan TKA tidak dikecualikan secara tegas dalam perjanjian kerja, maka perusahaan tetap wajib menghitung kewajiban #Imbalan-pasca-kerja untuk karyawan tidak tetap, sesuai ketentuan #PSAK-24 (sekarang #PSAK-219).

#### 16. Kapan #auditor harus melakukan komunikasi tertulis kepada pihak yang bertanggung jawab atas tata kelola tentang temuan signifikan selama audit?

#auditor harus menyampaikan komunikasi tertulis **sesegera mungkin setelah menemukan temuan signifikan, terutama jika berpotensi berdampak material** pada #laporan-keuangan atau pengendalian internal. Komunikasi ini biasanya dilakukan sebelum laporan audit final diterbitkan, dalam bentuk management letter atau laporan temuan audit. Sesuai dengan ISA 260 dan SA 315, #auditor wajib menginformasikan kelemahan pengendalian, penyimpangan standar akuntansi, atau risiko kecurangan agar perusahaan dapat segera mengambil tindakan perbaikan.

#### 17. Apa fungsi dari aktuaria pada perusahaan sekuritas?

#aktuaris membantu perusahaan sekuritas dalam mengelola risiko keuangan dan menilai nilai wajar investasi, seperti obligasi dan opsi. Mereka memakai model matematika dan data statistik untuk memprediksi keuntungan dan risiko.

Kalau perusahaan punya program pensiun atau tunjangan jangka panjang, #aktuaris juga menghitung kewajiban tersebut agar #laporan-keuangan sesuai standar (#PSAK-219) dan lebih akurat.

Selain itu, #aktuaris juga meng#Analisis-Risiko investasi dan melakukan simulasi skenario (stress testing) agar perusahaan siap menghadapi perubahan pasar. Dengan pendekatan berbasis data, #aktuaris membantu menyusun strategi supaya perusahaan tetap aman dan stabil di tengah gejolak pasar.

#### 18. Perhitungan #PSAK-24 nya menggunakan regulasi apa, UU Ketenagakerjaan atau Omnibus Law?

Dalam praktiknya, perhitungan PSAK 24 (sekarang #PSAK-219) merujuk pada standar akuntansi keuangan juga harus selaras dengan regulasi ketenagakerjaan yang berlaku. Regulasi yang menjadi acuan saat ini antara lain:

- **UU No. 13 Tahun 2003** atau #UUK / UU Ketenagakerjaan (sebelum revisi oleh Omnibus Law), yang mengatur hak-hak karyawan atas #Pesangon, penghargaan masa kerja, dan penggantian hak.
- **UU No. 11 Tahun 2020** atau UU Cipta Kerja / #UUCK (Omnibus Law) dan peraturan turunannya seperti PP No. 35 Tahun 2021, yang memperbarui skema perhitungan #Pesangon dan memberikan opsi bagi perusahaan untuk menggunakan skema asuransi #DPLK.
- Ketentuan dari perusahaan sendiri, seperti #Peraturan-Perusahaan, PKB (Perjanjian Kerja Bersama), atau kebijakan internal lainnya.

Jika suatu perusahaan sudah mengikuti Omnibus Law, perhitungan #PSAK-219 akan mempertimbangkan skema #Pesangon yang lebih rendah dibanding aturan lama. Namun, jika masih mengikuti aturan lama ( #UUK), maka kewajiban perusahaan terhadap karyawan bisa lebih besar.

#### 19. Apa perbedaan antara #Imbalan-pasca-kerja dalam valuasi aktuaria (sesuai #PSAK-24) dan dalam BPJS Ketenagakerjaan?

**BPJS** (Badan Penyelenggara Jaminan Sosial) dan #Imbalan-pasca-kerja yang diatur dalam PSAK 24 adalah dua hal yang berbeda dalam konteks #imbalan-kerja karyawan. BPJS adalah program jaminan sosial yang ditawarkan pemerintah untuk jaminan kesehatan, jaminan pensiun, dan jaminan sosial lainnya.

Sementara itu #Imbalan-pasca-kerja adalah segala bentuk imbalan yang diberikan kepada karyawan setelah mereka tidak lagi bekerja di perusahaan, termasuk pensiun, tunjangan kesehatan pensiunan, dan asuransi jiwa pasca kerja.

**BPJS**:
- Program jaminan sosial yang dikelola pemerintah.
- Memberikan jaminan kesehatan, jaminan pensiun, jaminan kecelakaan kerja, dan jaminan kematian.
- Iuran BPJS menjadi kewajiban perusahaan dan pekerja.
- Perusahaan dapat mencatat iuran BPJS sebagai bagian dari beban.

#Imbalan-pasca-kerja **(sesuai PSAK 24)**:
- Segala bentuk imbalan yang diberikan kepada karyawan setelah masa kerjanya berakhir.
- Dapat berupa tunjangan pensiun, tunjangan kesehatan pensiunan, asuransi jiwa, dll.
- Perusahaan memiliki kewajiban untuk memberikan imbalan ini.
- Kewajiban #Imbalan-pasca-kerja diukur oleh nilai kini dari imbalan yang akan dibayarkan di masa depan.
- PSAK 24 mengatur pengakuan, pengukuran, penyajian, dan pengungkapan #Imbalan-pasca-kerja.

#Imbalan-pasca-kerja yang dihitung dalam #PSAK-24 ( #PSAK-219) adalah imbalan selain BPJS yang nantinya akan dibayarkan kepada karyawan ketika karyawan tersebut di #PHK yang diatur dalam UU Ketenagakerjaan karena sebab berikut:
	- pensiun normal
	- meninggal dunia
	- resign
	- sakit berkepanjangan


| **Kriteria**  | **BPJS**                                 | **#Imbalan-pasca-kerja (PSAK 24)**         |
| ------------- | ---------------------------------------- | ----------------------------------------- |
| Penyelenggara | Pemerintah                               | Perusahaan                                |
| Jaminan       | Kesehatan, pensiun, kecelakaan, kematian | Pensiun, tunjangan kesehatan, dan lainnya |
| Kewajiban     | Perusahaan dan pekerja                   | Perusahaan                                |
| Pengaturan    | Undang-undang                            | PSAK 24                                   |
#### 20. Apakah #imbalan-kerja atau kompensasi PKWT wajib diberikan kepada karyawan?

Ya, berdasarkan regulasi ketenagakerjaan di Indonesia, karyawan dengan **Perjanjian Kerja Waktu Tertentu (PKWT) berhak menerima kompensasi** setelah masa kerja mereka berakhir. Hal ini diatur dalam **PP No. 35 Tahun 2021**, yang merupakan turunan dari #UUCK (Omnibus Law). Besaran kompensasi yang diberikan bergantung pada masa kerja karyawan, yaitu 1 bulan upah untuk setiap 12 bulan kerja secara terus-menerus, dan dihitung secara proporsional jika kurang dari 1 tahun.

#### 21. Jika RUPS belum menentukan besaran #Imbalan-pasca-kerja Direksi, bagaimana perhitungannya dilakukan?

Jika belum ada keputusan RUPS, perhitungan mengacu pada **UU Cipta Kerja (UUCK) atau perjanjian tertulis lain yang berlaku**. Dokumen pendukung seperti surat perjanjian kerja atau keputusan internal dapat digunakan sebagai dasar perhitungan.

Dokumen yang diperlukan:
- Keputusan RUPS (jika tersedia).
- Surat perjanjian kerja atau dokumen lain yang memuat janji #Imbalan-pasca-kerja.
- #Laporan-Aktuaria sebelumnya atau asumsi perhitungan yang relevan.

#### 22. Bagaimana jika 2025 masih ada perusahaan yang menggunakan #PSAK-24 (bukan #PSAK-219), apa dampaknya pada perusahaan tersebut?

1. **Regulasi yang Digunakan**
	#PSAK-24 masih merujuk pada aturan lama, sedangkan #PSAK-219 sudah mengikuti UU Cipta Kerja. Jadi, memakai #PSAK-219 membantu perusahaan mematuhi peraturan yang berlaku saat ini.

2. **Pengakuan Liabilitas dan Peran #DPLK**
	Dalam #PSAK-24, saldo #DPLK (program pensiun) tidak mengurangi kewajiban secara langsung. Di #PSAK-219, #DPLK bisa mengurangi kewajiban, sehingga hasil perhitungannya lebih mencerminkan kondisi sebenarnya.

3. **Penyajian dalam #laporan-keuangan**
	#PSAK-219 membuat #laporan-keuangan jadi lebih transparan, termasuk bagian yang masuk ke pendapatan komprehensif lainnya (OCI). Hal ini membantu perusahaan mengambil keputusan dengan data yang lebih akurat.

4. **Persoalan Audit**
	Mulai 2025, penggunaan #PSAK-24 dapat berisiko menimbulkan perbedaan dalam audit karena ketidaksesuaian dengan standar terbaru. Beralih ke #PSAK-219 akan membuat perusahaan lebih siap, lebih dipercaya investor, dan lebih sesuai dengan kondisi keuangan nyata.

#### 23. Mengapa penting untuk memverifikasi pengakuan utang dalam laporan #imbalan-kerja?

Verifikasi dari #KAP (Kantor Akuntan Publik) penting karena bertujuan memastikan bahwa pengakuan kewajiban #imbalan-kerja di #laporan-keuangan perusahaan **sudah sesuai regulasi dan akurat.**

Misalkan, perusahaan X mencatat:
	Debit: Beban #imbalan-kerja – Rp2,872 M
	Kredit: Utang #imbalan-kerja – Rp2,872 M

#KAP sebagai #auditor akan mengecek apakah angka ini benar-benar mencerminkan kewajiban perusahaan kepada karyawan, dan apakah ada aset terkait (seperti program pensiun) yang juga harus dicatat. Hasil verifikasi #KAP akan **menjamin bahwa #laporan-keuangan menjadi lebih transparan** bagi #auditor dan pemangku kepentingan.

#### 24. Bagaimana penentuan usia pensiun jika tidak ada #Peraturan-Perusahaan (PP)?

Jika tidak ada PP, penentuan usia pensiun biasanya didasarkan pada kebijakan internal perusahaan, peraturan pemerintah, atau UU yang berlaku.

#### 25. Bagaimana ketentuan usia pensiun normal sesuai dengan peraturan pemerintah yang berlaku?

Usia pensiun ditentukan berdasarkan Pasal 15 ayat (1), (2), dan (3) PP No. 45 Tahun 2015 sebagai berikut:
- Untuk pertama kali, usia pensiun ditetapkan 56 tahun.

Mulai 1 Januari 2019, usia pensiun menjadi 57 tahun.
- Usia pensiun akan **bertambah 1 tahun setiap 3 tahun berikutnya** sampai mencapai 65 tahun.
- Misalnya, 2019-2021: 57 tahun, 2022-2024: 58 tahun, dan seterusnya.

#### 26. Bagaimana jika sudah pernah menghitung secara internal / menggunakan tenaga ahli (bukan KKA)?

Langkah-langkah yang dapat dilakukan:

1. **Periksa kertas kerja perhitungan sebelumnya**
	Pastikan dokumen perhitungan terdokumentasi dengan baik, termasuk metode yang digunakan, [[Asumsi]] aktuaria (tingkat diskonto, #Kenaikan-Gaji, Mortalita), dan data yang dijadikan dasar perhitungan.
2. **Cek kesesuaiannya dengan standar yang berlaku**
	Pastikan bahwa perhitungan menggunakan metode #Projected-Unit-Credit (#PUC) dan mengikuti standar akuntansi berlaku seperti #PSAK-219 agar dapat diterima dalam #laporan-keuangan yang diaudit.
3. **Konfirmasi dengan** #auditor atau #aktuaris Profesional
	Jika hasil perhitungan sudah digunakan dalam #laporan-keuangan yang telah diaudit, maka data tersebut dapat dijadikan saldo awal untuk perhitungan selanjutnya. Jika belum, sebaiknya dilakukan review dan validasi untuk memastikan keakuratannya.
4. **Lakukan penyesuaian jika diperlukan**
	Jika terdapat perbedaan metode atau asumsi, perusahaan bisa mempertimbangkan untuk melakukan penyesuaian dengan bantuan Kantor Konsultan Aktuaria (KKA) agar hasil perhitungan sesuai dengan standar audit dan tidak menimbulkan potensi koreksi dalam #laporan-keuangan.

#### 27. Apa standar perusahaan harus membuat perhitungan post employment benefit?

Perhitungan #Imbalan-pasca-kerja (post employment benefit) wajib dilakukan oleh perusahaan yang menyusun #laporan-keuangan sesuai standar akuntansi seperti #PSAK-219, terutama bagi **perusahaan tercatat di bursa efek, diaudit, atau memiliki akuntabilitas publik.**

Namun, meskipun perusahaan skala kecil tidak wajib mengikuti standar akuntansi tertentu, mereka **tetap harus membayar** #imbalan-kerja sesuai #UUK No. 13 Tahun 2003 atau PP No. 35 Tahun 2021 (Omnibus Law). Oleh karena itu, perhitungan #Imbalan-pasca-kerja tetap penting dalam manajemen keuangan perusahaan.

---
## #FAQ Teknis dan Perhitungan

#### 1. Data apa saja yang dibutuhkan untuk perhitungan?

Kami memiliki 2 #file-template data untuk dilengkapi. Pertama, berupa file #template-excel yang berisikan data pribadi karyawan seperti **tanggal lahir, tanggal masuk kerja, gaji, dan lainnya.** Kedua, berupa file word yang berisikan informasi perusahaan, seperti **nama perusahaan, alamat perusahaan, rata-rata** #Kenaikan-Gaji, penanggung pajak oleh karyawan / perusahaan dan lainnya.**

Selain 2 template tersebut, kami membutuhkan data pendukung lainnya seperti **#Peraturan-Perusahaan, NPWP perusahaan, lembar persetujuan penawaran yang telah ditandatangani,** #Laporan-Aktuaria tahun lalu (jika ada) serta saldo & **iuran** #DPLK (jika ada).

Hal-hal yang perlu dipastikan meliputi:
- Pastikan jumlah karyawan yang dihitung mencakup karyawan aktif dan tidak aktif, dilengkapi status tetap dan kontrak (jika ada).
- Pastikan untuk tidak memasukkan anggota Board of Directors (BOD) dan Tenaga Kerja Asing (TKA) (jika ada) dalam perhitungan.

#### 2. Bagaimana memahami pengakuan beban / pendapatan yang memisahkan #Biaya-Jasa-Kini dan #Biaya-Jasa-Lalu dalam pencatatan #imbalan-kerja?

Pengakuan beban / pendapatan dalam pencatatan #imbalan-kerja, penting dipisahkan antara:
1. **Debit** #imbalan-kerja sebagai total beban yang timbul selama tahun berjalan dan harus diakui di #Laporan-Laba-Rugi.
    - #Biaya-Jasa-Kini (Current Service Cost) → hak karyawan yang bekerja aktif selama tahun valuasi.
    - #Biaya-Jasa-Lalu (Past Service Cost) → koreksi atas manfaat masa lalu, yang sifatnya mengurangi kewajiban perusahaan secara signifikan (karena ada penghapusan atau penyesuaian manfaat vested). Jika dampaknya besar, bisa menciptakan lonjakan beban atau justru pendapatan (jika kewajiban dikurangi).
2.  **Kredit** #imbalan-kerja sebagai jumlah kewajiban (utang) yang harus dibayar oleh perusahaan kepada karyawan di masa mendatang.
**Diketahui**:
Pada awal tahun 2024, kewajiban perusahaan sebesar Rp9,1 M. Setelah beban tahun berjalan dikurangi sebesar Rp5,3 M → maka kewajiban akhir tahun 2024 turun menjadi Rp3,79 M.

Ini menunjukkan bahwa sebagian besar kewajiban perusahaan sudah dikoreksi atau direvisi, baik karena **pengurangan manfaat, penghapusan program, atau perubahan data karyawan.** Jadi, pengakuan finalnya adalah:
- Debit: Beban #imbalan-kerja ( #CSC + #interest-cost) = Rp1,14 M
- Debit: Koreksi #Biaya-Jasa-Lalu ( #BJS) = Rp -6,45 M
- Kredit: Penurunan Bersih Kewajiban = Rp -5,31 M

Karena nilai koreksi #Biaya-Jasa-Lalu sangat besar dan bersifat mengurangi kewajiban, maka perusahaan mencatatnya sebagai **pendapatan** dalam #Laporan-Laba-Rugi.

Dengan memisahkan kedua biaya tersebut, #Laporan-Aktuaria memberikan transparansi penuh tentang:
- **berapa biaya yang benar-benar timbul tahun berjalan,**
- berapa yang muncul karena kebijakan baru,
- dan apakah perubahan program memberikan **beban atau justru “penghematan”** di #laporan-keuangan.

#### 3. Bagaimana cara memastikan saldo #Imbalan-pasca-kerja pada #laporan-keuangan sebelumnya?

Pastikan untuk memeriksa apakah pada #laporan-keuangan sebelumnya sudah ada saldo **beban** #Imbalan-pasca-kerja yang dicatat. Jika ini adalah pertama kali perusahaan menghitung, tetapi sebelumnya sudah ada perhitungan internal, dibutuhkan data detail perhitungannya, setidaknya saldo yang digunakan pada #laporan-keuangan sebelumnya.

#### 4. Bagaimana cara memastikan saldo #Imbalan-pasca-kerja pada #laporan-keuangan sebelumnya?

- Umumnya dicatat sebagai **Utang** jika #imbalan-kerja masih berupa kewajiban **yang belum dibayarkan**, seperti #Biaya-Jasa-Kini dan #Biaya-Jasa-Lalu.
- Dalam kondisi tertentu dicatat sebagai **Aset** jika perusahaan **memiliki dana pensiun** atau **aset investasi yang dikelola secara khusus** untuk membayar manfaat tersebut di masa depan.

Dengan bantuan Kantor Akuntan Publik (KAP), pencatatan ini dilakukan sesuai standar akuntansi seperti #PSAK-219 untuk memastikan transparansi #laporan-keuangan.

#### 5. Periode #perhitungan-IPK dilakukan per tahun, per bulan, atau per kuartal?

#perhitungan-IPK **biasanya dilakukan per tahun**, mengikuti periode pelaporan audit, karena merupakan salah satu komponen yang diperlukan auditor dalam #laporan-keuangan. Namun, bisa juga dihitung per kuartal **tergantung kebutuhan perusahaan dan auditor** untuk mendapatkan nilai estimasi #imbalan-kerja yang **lebih proporsional secara berkala.**

#### 6. Apa perbedaan perhitungan untuk karyawan tetap dan kontrak ?

- Karyawan tetap (PKWTT): Perhitungannya mencakup beberapa komponen, seperti cadangan untuk pensiun, meninggal dunia, cacat, dan resign, yang formulanya mengikuti peraturan perundang-undangan yang berlaku.
- Karyawan kontrak (PKWT): Perhitungannya didasarkan pada masa kerja (proporsional), yaitu masa kerja (bulan) / 12 x gaji.

Catatan: Setelah #IFRIC diterapkan, nilai manfaat PKWTT berusia < 34 tahun umumnya lebih tinggi dibandingkan PKWT. Ini karena perhitungan PKWT hanya mencakup manfaat kematian, cacat, dan pengunduran diri, **tanpa manfaat pensiun**.

#perhitungan-IPK untuk PKWTT maupun PKWT terutama dalam alokasi kewajiban perusahaan di masa mendatang, namun terdapat perbedaan pandangan untuk karyawan PKWT.
- Jika ada karyawan dengan masa kontrak karyawan > 1 tahun, perusahaan harus mencadangkan **kompensasi saat karyawan tersebut saat keluar nanti.**
- Jika tidak ada, biasanya masuk dalam imbalan jangka pendek.
Namun, keputusan ini bergantung pada kebijakan masing-masing KAP, karena setiap KAP dapat memiliki perlakuan yang berbeda.

#### 7. Bagaimana dengan status karyawan harian?

Karyawan harian tidak masuk ke dalam perhitungan.

#### 8. Bagaimana ketentuan gaji yang digunakan dalam #perhitungan-IPK?

Gaji yang digunakan untuk #perhitungan-IPK adalah gaji bruto untuk setiap karyawan.

#### 9. Bagaimana jika karyawan kontrak berubah setiap proyek?

Untuk kondisi ini, perhitungan akan difokuskan pada data karyawan tetap.

#### 10. Komponen inti data karyawan untuk memulai perhitungan apa saja?

Identitas karyawan (NIK/Nama), tanggal lahir, tanggal mulai kerja, gaji pokok dan tunjangan (jika ada), dan status karyawan.

#### 11. Apa saja komponen utama penyusun beban pada laporan IPK?

Terdiri dari #Biaya-Jasa-Kini, #Biaya-Jasa-Lalu dan #Biaya-Bunga. Bisa ditambahkan juga komponen tambahan lainnya seperti: pengaruh #Kurtailmen atau Penyelesaian, jika ada penghentian program pensiun, dan (Dikurangi) Hasil Investasi Aset Program, bila perusahaan punya aset dana pensiun.

#### 12. Metode apa yang digunakan dalam perhitungan?

#perhitungan-IPK menggunakan metode #Projected-Unit-Credit (#PUC) sesuai dengan yang dipersyaratkan dalam #PSAK-219. Metode ini juga sesuai dengan IFRS/IAS 19 (Employee Benefits), sebagai standar #Perhitungan-Aktuaria internasional.

#### 13. Bagaimana cara menghitung nilai estimated future working dalam #PSAK-219?

Estimated future working dapat dihitung berdasarkan #Sisa-Masa-Kerja untuk setiap karyawan. Untuk karyawan tetap dihitung hingga karyawan tersebut mencapai #Usia-Pensiun nya, sedangkan untuk karyawan kontrak dihitung hingga berakhirnya kontrak karyawan tersebut (diasumsikan satu tahun).

#### 14. Apa yang harus dilakukan jika klien sudah pernah melakukan perhitungan tetapi tidak memiliki data rincian terkait perhitungan kewajiban / #PVDBO?

Jika klien tidak memiliki data rincian terkait perhitungan, maka dibutuhkan data saldo #Imbalan-pasca-kerja dan saldo #OCI yang tercatat pada #laporan-keuangan yang telah diaudit sebelumnya sebagai saldo awal. Selain itu, dibutuhkan juga terkait data karyawan yang digunakan untuk perhitungan sebelumnya untuk melihat asumsi yang relevan yang akan digunakan pada perhitungan saat ini.

#### 15. Apakah karyawan yang sudah pensiun tetapi masih bekerja sebagai komisaris dan menerima gaji harus tetap dimasukkan dalam perhitungan #PSAK-219?

Tidak. Karyawan yang telah pensiun dan diangkat sebagai komisaris tidak lagi masuk dalam #Perhitungan-Aktuaria #PSAK-219 karena hubungan mereka dengan perusahaan berubah menjadi hubungan profesional, bukan lagi karyawan.

Namun, jika perusahaan memiliki kebijakan memberikan #Imbalan-pasca-kerja kepada komisaris, kewajiban tersebut tetap harus dihitung, tetapi secara terpisah dari kewajiban aktuaria karyawan. Sesuai Pasal 15 UU No. 40 Tahun 2007, komisaris dan direksi bukan bagian dari karyawan, sehingga tidak berhak atas #imbalan-kerja sesuai #PSAK-219, kecuali ada kebijakan perusahaan yang secara khusus mengatur tunjangan mereka.

#### 16. Bagaimana mutasi karyawan akan berdampak pada nilai #Biaya-Jasa-Lalu dan #OCI suatu perusahaan?

- Untuk karyawan keluar, jika kewajiban > realisasi, selisihnya akan diakui pada #Pengalaman-Penyesuaian (salah satu komponen dari #OCI). Namun, jika kewajiban < realisasi, selisihnya akan diakui pada #Biaya-Jasa-Lalu.
- Untuk karyawan masuk, akan diakui pada #Biaya-Jasa-Lalu sebesar #PVDBO - #CSC.

#### 17. Apa yang dimaksud dengan #Biaya-Jasa-Lalu dan #Biaya-Jasa-Kini? Bagaimana cara perhitungannya menurut #PSAK-219 untuk karyawan tetap?

1. #Biaya-Jasa-Lalu (Past Service Cost) adalah biaya yang muncul ketika perusahaan baru pertama kali mencatat kewajiban #Imbalan-pasca-kerja / IPK secara menyeluruh atau terjadi perubahan besar, seperti perubahan kebijakan cuti besar, restruktur kompensasi, mutasi karyawan, perubahan pada metode / asumsi, atau revisi program pensiun.

	Berikut adalah ketentuan mengenai #Biaya-Jasa-Lalu untuk penerapan pertama kali:
	- Jika masa kerja < 1 tahun: #Biaya-Jasa-Lalu = 0
	- Jika masa kerja ≥ 1 tahun: #Biaya-Jasa-Lalu = #PVDBO – #CSC

	#Biaya-Jasa-Lalu biasanya langsung diakui dalam #Laporan-Laba-Rugi jika tidak vested atau diamortisasi jika vested.

2. #Biaya-Jasa-Kini (Current Service Cost) adalah biaya atas manfaat pensiun yang "ditabung" perusahaan untuk karyawan selama periode berjalan. Dihitung berdasarkan asumsi aktuaria ( #Kenaikan-Gaji, usia, diskonto, dan lainnya). Faktor yang bisa meningkatkan #CSC, seperti:
	- Tambahan pajak dari manfaat
	- Penurunan suku bunga (discount rate)
	- #Kenaikan-Gaji aktual lebih tinggi dari asumsi
	- Perubahan manfaat (misalnya tunjangan pensiun baru)

Rumus sederhana #Biaya-Jasa-Kini / #CSC
$$
CSC = \frac{PVFB}{MK}
$$
	dimana:
	PVFB = present value of Future Benefit atau nilai sekarang dari estimasi manfaat masa depan MK = total masa kerja hingga karyawan mencapai #Usia-Pensiun normal

#### 18. Mengapa nilai #OCI hanya mencakup karyawan tetap, dan bagaimana perlakuan keuntungan / kerugian aktuarial dalam #laporan-keuangan?

Nilai #OCI hanya mencakup karyawan tetap karena merekalah yang umumnya menerima manfaat jangka panjang seperti pensiun atau #Imbalan-pasca-kerja, yang menjadi sumber timbulnya keuntungan atau kerugian aktuaria.

#### 19. Apakah karyawan kontrak / PKWT berdampak kepada #OCI ?

Pada dasarnya perhitungan karyawan kontrak tidak diatur jelas dalam standar praktik #imbalan-kerja. Sebagian KKA tidak memakai #OCI karena menganggap itu bagian dari #OLTEB. Kelebihan atau kekurangan kewajiban masuk di #Laporan-Laba-Rugi.

Tapi kami, KKA Nirmala disini memakai nilai #OCI karena menilai kompensasi karyawan kontrak masuk kategori #Imbalan-pasca-kerja. Perhitungannya dengan metode #PUC dengan nilai manfaat sebesar proporsi masa kerja x gaji yang menggunakan asumsi diskonto dan probabilitas. Perbedaan asumsi juga akan menimbulkan #OCI.

Umumnya, perlakuannya disesuaikan dengan ketentuan kontrak setiap perusahaan.
- Jika setelah kontrak berakhir **kompensasi dibayarkan**, maka **tidak terdapat pengakuan** #OCI. Maka langsung dibebankan sebagai beban pada tahun berjalan.
- ⁠Jika setelah kontrak berakhir **kompensasi tidak dibayarkan**, maka akan **terdapat pengakuan dalam** #OCI.

#### 20. Mengapa karyawan yang di atas #Usia-Pensiun masih masuk dalam perhitungan?

Selama karyawan belum menerima #Pesangon atas #Usia-Pensiun nya dan/ memang masih bekerja, perusahaan masih memiliki kewajiban untuk mencadangkan #Pesangon untuk karyawan tersebut.

#### 21. Bila memiliki #DPLK, data terkait apa saja yang dibutuhkan untuk perhitungan?

Kami membutuhkan data berupa saldo akhir #DPLK, iuran #DPLK dan ROI rate dari #DPLK.

#### 22. Apa yang perlu diperhatikan jika #perhitungan-IPK termasuk imbalan jangka panjang lainnya ( #OLTEB) dan #DPLK?

1. Pastikan aturan mengenai imbalan jangka panjang lainnya / #OLTEB, seperti Cuti Besar ( #CBS) dan Penghargaan Emas, **sudah ditetapkan dalam kebijakan perusahaan.**
2. Jika karyawan terdaftar dalam #DPLK, kewajiban pensiun sebagian dapat ditanggung oleh #DPLK, sehingga perusahaan hanya perlu mencatat kewajiban tambahan yang belum ditanggung oleh dana pensiun tersebut.
3. Imbalan Jangka Panjang Lainnya yang belum digunakan, juga harus diperhitungkan dalam kewajiban perusahaan. Karyawan yang memiliki **#CBS akan memiliki kewajiban tambahan**, sedangkan yang **non-#CBS mungkin memiliki kewajiban yang lebih kecil**.
4. Jika perusahaan sudah pernah melakukan perhitungan valuasi sebelumnya, pastikan bahwa perhitungan terbaru mencakup seluruh komponen ini agar nilai kewajiban yang diakui dalam #laporan-keuangan tetap sesuai dengan standar #PSAK-219.

#### 23. Bagaimana cara perusahaan menghitung kewajiban untuk karyawan yang tidak memiliki hak cuti besar dan tidak terdaftar dalam #DPLK?

Perusahaan hanya perlu menghitung kewajiban berdasarkan masa kerja, gaji yang berlaku, dan asumsi aktuaria terkait, **tanpa adanya tambahan** dari hak cuti besar atau kontribusi #DPLK.

#### 24. Apa yang harus diperhatikan dari kewajiban untuk karyawan non-#DPLK?

Karyawan non-#DPLK tidak memiliki dana pensiun tambahan yang didanai oleh pihak ketiga, sehingga seluruh kewajiban #imbalan-kerja harus ditanggung oleh perusahaan. Ini berarti bahwa **perusahaan perlu mencadangkan seluruh biaya pensiun** karyawan tersebut dalam #laporan-keuangan mereka.

#### 25. Apa perbedaan dalam perhitungan kewajiban antara karyawan yang memiliki #CBS dan non-#CBS?

Untuk karyawan yang memiliki #CBS, kewajiban #imbalan-kerja mencakup biaya tambahan yang terkait dengan hak cuti besar yang belum digunakan. Sebaliknya, karyawan non-#CBS tidak memiliki komponen tambahan ini, sehingga nilai kewajibannya mungkin lebih rendah. Jika seorang karyawan **memutuskan untuk tidak menggunakan hak #CBS, kewajiban tersebut tetap dicatat** hingga hak tersebut kadaluarsa atau dibayarkan oleh perusahaan.

#### 26. Apakah perusahaan dapat mengubah kebijakan #CBS atau #DPLK di tengah masa kerja karyawan, dan bagaimana ini mempengaruhi perhitungan kewajiban?

Bisa saja, namun perubahan ini harus dilakukan sesuai dengan peraturan ketenagakerjaan yang berlaku dan harus disosialisasikan kepada karyawan. Setiap perubahan dalam kebijakan ini dapat mempengaruhi perhitungan kewajiban #imbalan-kerja dan perlu diperhitungkan dalam #laporan-keuangan sesuai dengan ketentuan yang berlaku.

#### 27. Bagaimana ketentuan mengenai realisasi pembayaran karyawan kontrak?

Untuk karyawan kontrak, perlu dipastikan apakah realisasi pembayaran kompensasi dilakukan **setiap akhir kontrak**—baik bagi karyawan yang kontraknya diperpanjang maupun yang tidak—atau **hanya saat karyawan keluar** dari perusahaan.

#### 28. Bagaimana cara menentukan [[Asumsi#Tingkat-Kenaikan-Gaji]] dalam #perhitungan-IPK?

[[Asumsi#Tingkat-Kenaikan-Gaji]] umumnya ditentukan berdasarkan:
- Rata-rata #Kenaikan-Gaji historis perusahaan dalam beberapa tahun terakhir.
- Kenaikan UMK di wilayah operasional perusahaan.
- Tingkat inflasi dalam beberapa tahun terakhir.
- Rencana #Kenaikan-Gaji ke depan, berdasarkan kebijakan internal atau surat resmi dari manajemen perusahaan (management letter).
- Asumsi tahun sebelumnya, bila masih relevan dan belum berubah signifikan.

#### 29. Apakah terdapat perbedaan perhitungan yang digunakan jika dibandingkan dengan KKA lain?

Setiap KKA, dalam melakukan #perhitungan-IPK dihitung menggunakan metode yang sama, sesuai dengan yang diprasyaratkan dalam #PSAK-219, namun umumnya terdapat perbedaan dalam pendekatan menghitung usia yang akan digunakan dalam menghitung kewajiban.

#### 30. Mengapa hasil perhitungan tahun valuasi lebih tinggi dibandingkan tahun sebelumnya?

Beberapa faktor yang dapat menyebabkan kenaikan nilai kewajiban:
- Bertambahnya usia setiap karyawan
- Bertambahkan masa kerja setiap karyawan
- Penurunan asumsi #Usia-Pensiun normal
- Perbedaan pengakuan usia pada tanggal valuasi (contoh, penggunaan usia tepat dibandingkan dengan pembulatan usia).
- Penurunan #Tingkat-Diskonto 
- #Kenaikan-Gaji aktual yang lebih besar daripada asumsi
- Penambahan komponen pajak dalam perhitungan
- Perubahan manfaat IPK yang digunakan (penambahan manfaat)

#### 31. Mengapa ada perbedaan antara salary increase rate yang digunakan dalam laporan dan aktual #Kenaikan-Gaji?

Perbedaan tersebut bisa terjadi karena dalam laporan merupakan asumsi jangka panjang, sedangkan #Kenaikan-Gaji aktual dapat dipengaruhi oleh faktor-faktor eksternal dan kebijakan perusahaan dalam jangka pendek. Dalam perhitungan, rate yang digunakan untuk memproyeksikan kewajiban di masa depan dengan mempertimbangkan:
1. Tren historis – Pola #Kenaikan-Gaji rata-rata dalam beberapa tahun terakhir.
2. Inflasi & kondisi ekonomi – Perkiraan pertumbuhan ekonomi yang mempengaruhi upah.
3. Historis karyawan – #Kenaikan-Gaji berdasarkan tingkat jabatan, performa, dan masa kerja.
4. Kebijakan perusahaan – Rencana kompensasi jangka panjang.

Oleh karena itu, jika rate mengalami kenaikan / penurunan aktual dari asumsi dalam #Laporan-Aktuaria, maka dapat terjadi deviasi yang akan dikoreksi dalam penyesuaian aktuaria berikutnya.

#### 32. Apakah nilai kewajiban dapat diperkecil?

Ya, bisa diperkecil dengan beberapa strategi, tetapi harus tetap sesuai dengan prinsip aktuaria dan standar akuntansi. Berikut adalah beberapa faktor yang dapat mengurangi nilai kewajiban:

1. Mengubah [[Asumsi]] Aktuaria
	- Meningkatkan #Tingkat-Diskonto
	- Menurunkan [[Asumsi#Tingkat-Kenaikan-Gaji]]
	- Menyesuaikan asumsi mortalita & withdrawal

2. Menyesuaikan Kebijakan Manfaat
	- Beralih dari manfaat pasti ke iuran pasti
	- Mengurangi atau membatasi manfaat (contohnya, membatasi #Pesangon atau manfaat kesehatan pasca kerja)
	- Menerapkan batas maksimal manfaat

3. Menggunakan Curtailment atau Settlement
	- Curtailment → Mengurangi skema manfaat (misalnya akibat #PHK massal).
	- Settlement → Menyelesaikan kewajiban lebih awal, misalnya dengan pembayaran lump sum atau membeli polis asuransi pensiun.

Namun, perubahan ini harus dipertimbangkan secara hati-hati agar tetap sesuai dengan regulasi ketenagakerjaan dan tidak berdampak negatif pada kesejahteraan karyawan.

#### 33. Apa itu Curtailment dalam perhitungan #PSAK-219? Dan apa dampaknya pada #laporan-keuangan Perusahaan?

#Kurtailmen (curtailment) terjadi ketika perusahaan melakukan tindakan yang secara signifikan mengurangi jumlah karyawan yang berhak atas #Imbalan-pasca-kerja. Nilai ini berdampak pada estimasi nilai kewajiban / #PVDBO yang mengakibatkan keuntungan atau kerugian yang harus segera diakui dalam #Laporan-Laba-Rugi.

#Kurtailmen dapat terjadi karena beberapa faktor, seperti **#PHK massal** akibat efisiensi atau restrukturisasi perusahaan.
**Penyelesaian** terjadi apabila adanya perubahan skema manfaat pasti / kebijakan perusahaan, sehingga hak karyawan terhadap **manfaat tersebut dihentikan, atau nilai kewajiban sudah dibayar penuh.**

Jika perusahaan melakukan #PHK, maka realisasi pembayaran karyawan akan dikategorikan sebagai curtailment penyelesaian, selama tidak berkaitan dengan:
- Pensiun
- Meninggal dunia
- Sakit berkepanjangan / cacat
- Mengundurkan diri (resign)

Pembayaran di luar empat manfaat tersebut, wajib dicatat sebagai curtailment penyelesaian.

#### 34. Apakah ada keuntungan bagi perusahaan dalam menghitung #CBS atau mendaftarkan karyawan ke dalam #DPLK terkait perhitungan kewajiban?

Menghitung #CBS dapat meningkatkan kewajiban jangka pendek terkait hak cuti yang belum digunakan, tetapi juga dapat membantu dalam retensi karyawan. Selain itu, mendaftarkan karyawan ke #DPLK dapat mengurangi kewajiban perusahaan karena sebagian dari beban pensiun ditanggung oleh pihak ketiga, yang bisa mengurangi beban finansial perusahaan dalam jangka panjang.

#### 35. Bagaimana perhitungan kewajiban apabila karyawan terdaftar dalam #DPLK dan pencatatannya dalam #laporan-keuangan?

Nilai saldo #DPLK setiap karyawan dapat digunakan pengurang kewajiban, sehingga yang dicatat dalam laporan IPK hanya sisa kewajiban setelah dikurangi saldo #DPLK yang tersedia.

Namun, imbal hasil dari #DPLK tidak secara langsung mengurangi #CSC, karena #CSC dihitung berdasarkan masa kerja dan [[Asumsi#Tingkat-Kenaikan-Gaji]]. Sebaliknya, imbal hasil dari #DPLK lebih memengaruhi perubahan kewajiban aktuaria dan #Biaya-Bunga dalam #laporan-keuangan.

Dalam #Laporan-Laba-Rugi, apabila perusahaan sudah membayar iuran ke #DPLK, pengakuan beban dicatat sebagai berikut:
- Jika liabilitas > #DPLK, selisihnya masih harus dicatat sebagai kewajiban di neraca.
- Jika liabilitas ≤ #DPLK, maka kewajiban di neraca bisa dianggap nol, karena sudah ter-cover oleh dana yang terkumpul di #DPLK.

#### 36. Apakah ada perubahan dalam perhitungan kewajiban jika karyawan menggunakan sebagian hak Imbalan Jangka Panjang Lainnya (OLTEB) sebelum pensiun?

Jika karyawan mengambil sebagian hak itu sebelum pensiun (misalnya cuti besar dibayar atau penghargaan masa kerja), maka:
- **Kewajiban perusahaan berkurang**, karena sebagian manfaat sudah dibayar.
- Namun, **sisa hak yang belum digunakan tetap harus dihitung** dalam kewajiban aktuarial.
- Penyesuaian dilakukan dengan **mengurangi proyeksi manfaat** berdasarkan hak yang sudah diambil.
- **Sisa kewajiban tetap didiskontokan** menggunakan asumsi yang berlaku (seperti #Tingkat-Diskonto, #Tingkat-Kenaikan-Gaji, dan mortalita).

#### 37. Mengapa nilai #Biaya-Jasa-Lalu bisa muncul dari perbedaan #Usia-Pensiun dan mutasi karyawan?

Perbedaan #Usia-Pensiun dan mutasi karyawan dapat berdampak langsung pada perhitungan kewajiban aktuaria karena mengubah periode kerja yang diperhitungkan dan jumlah karyawan yang berhak atas manfaat.

Jika #Usia-Pensiun **diperpanjang, nilai kini kewajiban meningkat** karena periode kerja lebih lama, sehingga manfaat yang harus dibayar bertambah. Kenaikan ini diakui sebagai #Biaya-Jasa-Lalu. Sebaliknya, jika terjadi **mutasi atau pengurangan karyawan yang memenuhi syarat manfaat, kewajiban bisa berkurang,** mengarah pada pengakuan keuntungan (gain) dari perubahan skema manfaat. #Biaya-Jasa-Lalu muncul karena **perubahan estimasi #PVDBO.**

#### 38. Jika terdapat karyawan yang keluar di tahun valuasi, namun realisasinya baru akan dibayarkan di tahun berikutnya, bagaimana #perhitungan-IPK nya?

Jika realisasi dibayarkan di tahun selanjutnya, nilai realisasi akan masuk dalam perhitungan sesuai dengan kapan realisasi tersebut dibayarkan. Kondisi tersebut tetap dicatat dalam laporan tahun valuasi, meskipun pembayaran manfaat baru dilakukan di tahun berikutnya.

Dalam #Perhitungan-Aktuaria, estimasi pembayaran tersebut **tetap diperhitungkan** sebagai bagian dari Nilai Kini Kewajiban / #PVDBO hingga manfaat benar-benar dibayarkan. Sehingga, kewajiban tetap diakui pada periode saat karyawan keluar, dan **penyesuaian dilakukan di laporan tahun berikutnya** saat realisasi pembayaran terjadi.

#### 39. Apa yang dimaksud dengan Nilai Kini Kewajiban ( #PVDBO) perusahaan per tanggal valuasi?

Nilai Kini Kewajiban ( #PVDBO) adalah estimasi atau proyeksi dari nilai total kewajiban yang harus dicadangkan oleh perusahaan pada tanggal perhitungan (tanggal valuasi) untuk memenuhi seluruh manfaat karyawan di masa mendatang. Perhitungan #PVDBO mempertimbangkan faktor diskonto, tingkat #Kenaikan-Gaji, harapan hidup, dan probabilitas karyawan tetap bekerja hingga menerima manfaat.

Komponen #PVDBO mencakup:
- Manfaat Pensiun – Nilai kini dari seluruh kewajiban pensiun yang harus disediakan oleh perusahaan.
- Manfaat Kematian – Nilai kini dari manfaat yang akan diberikan kepada ahli waris jika karyawan meninggal sebelum pensiun.
- Manfaat Cacat / Sakit Berkepanjangan – Nilai kini dari kewajiban perusahaan jika karyawan mengalami kecacatan / sakit berkepanjangan.
- Manfaat Pengunduran Diri – Nilai kini dari manfaat yang akan diberikan kepada karyawan yang berhenti sebelum pensiun.

**Catatan**: Mengingat ini adalah estimasi, jadi jumlah aktual yang diterima oleh karyawan di masa mendatang mungkin berbeda tergantung pada berbagai faktor, termasuk kinerja investasi, perubahan dalam ketentuan pensiun, dan lainnya. Estimasi ini membantu perusahaan dalam memastikan bahwa ada kecukupan dana untuk memenuhi kewajiban di masa depan.

#### 40. Bagaimana cara menghitung nilai kewajiban suatu perusahaan?

Nilai kewajiban akhir suatu perusahaan diperoleh dari:
- Nilai kewajiban awal (jika ada) (+)
- Beban tahun berjalan (+)
- Realisasi tahun berjalan (-)
- Nilai #OCI tahun berjalan (+)
- Iuran #DPLK porsi perusahaan tahun berjalan (jika ada) (-)

#### 41. Apa #Tingkat-Diskonto nya menggunakan single rate atau multiple rate?

Untuk karyawan tetap, penetapan diskonto menggunakan multiple rate, dimana besarannya akan berbeda untuk setiap karyawan yang ditentukan berdasarkan rata-rata sisa masa kerja yang diprakirakan untuk setiap karyawan, sedangkan untuk penyajiannya dalam laporan menggunakan average rate-nya.

Sedangkan untuk karyawan kontrak, menggunakan rata-rata diskonto yang diperoleh dari interpolasi sisa masa kerja setiap karyawan.

#### 42. Mengapa ada #Perhitungan-Aktuaria untuk laporan di tahun berjalan, baru bisa dilakukan di awal tahun selanjutnya?

#Perhitungan-Aktuaria mengandalkan #Tingkat-Diskonto yang biasanya didasarkan pada yield obligasi pemerintah per tanggal valuasi (misalnya 31 Desember 2024). Namun, data resmi #Tingkat-Diskonto ini baru tersedia di awal tahun berikutnya (Januari 2025). Untuk kebutuhan internal, perusahaan dapat menggunakan estimasi #Tingkat-Diskonto sementara (misalnya per 30 November), tetapi hasil akhir tetap harus diperbarui setelah data resmi tersedia.

Kami mengacu kepada spot rate yang dikeluarkan oleh pemerintah melalui PHEI (PT Penilai Harga Efek Indonesia) setiap bulannya. Hal ini dilakukan agar kewajiban #imbalan-kerja mencerminkan nilai wajar sesuai #PSAK-219.

#### 43. Apa risiko jika #Perhitungan-Aktuaria dilakukan menggunakan estimasi #Tingkat-Diskonto sebelum data resmi dirilis?

Risiko utamanya adalah perbedaan antara hasil estimasi dan hasil akhir setelah #Tingkat-Diskonto resmi dirilis. Jika selisihnya signifikan, #laporan-keuangan bisa dianggap tidak mencerminkan kewajiban sebenarnya, berpotensi mendapat catatan audit dan menurunkan kepercayaan pemangku kepentingan. Akibatnya, perusahaan harus merevisi laporan dan menghadapi keterlambatan pelaporan ke regulator dan pemegang saham.

#### 44. Berapa suku bunga diskonto yang digunakan dalam #Perhitungan-Aktuaria untuk tahun valuasi?

Suku bunga diskonto bervariasi tergantung pada kebijakan perusahaan dan regulasi yang berlaku, karena tidak ada standar tunggal yang berlaku di tiap tahun valuasi untuk semua entitas. Perusahaan lain mungkin menggunakan asumsi suku bunga yang berbeda berdasarkan kondisi pasar, kebijakan internal, dan panduan dari otoritas terkait.

#### 45. Bagaimana jika terdapat realisasi pembayaran untuk karyawan yang sebelumnya belum masuk pada #perhitungan-IPK?

Pembayaran tersebut tetap diakui dalam perhitungan tahun berjalan. Namun, karena karyawan tersebut tidak termasuk dalam #perhitungan-IPK sebelumnya, maka kewajiban aktuaria perlu diperbarui karena ada peningkatan liabilitas yang tidak terduga.

- Jika pembayaran berasal dari perubahan skema manfaat: Diakui sebagai #Biaya-Jasa-Lalu karena ada hak tambahan yang diberikan kepada karyawan yang sebelumnya tidak dihitung.
- Jika pembayaran terjadi karena **kesalahan estimasi atau data yang tidak lengkap**: Dapat dicatat sebagai **penyesuaian saldo awal** kewajiban aktuaria, yang mencerminkan koreksi atas perhitungan sebelumnya.

#### 46. Apakah data dan #Kenaikan-Gaji karyawan kontrak yang diangkat menjadi karyawan tetap tetap perlu dihitung?

Ya, data ini perlu dicatat dan dihitung. Informasi #Kenaikan-Gaji untuk karyawan yang diangkat menjadi karyawan tetap sangat penting untuk memperkirakan kewajiban #Imbalan-pasca-kerja di masa depan. Perubahan status dari kontrak ke tetap juga dapat memengaruhi asumsi aktuaria terkait #Kenaikan-Gaji rata-rata dan durasi masa kerja.

#### 47. Apakah dasar gaji yang digunakan untuk menghitung #imbalan-kerja mencakup gaji pokok, tunjangan tetap, dan PPh?

Dasar gaji yang digunakan dalam perhitungan #imbalan-kerja bergantung pada kebijakan perusahaan dan regulasi ketenagakerjaan yang berlaku. Secara umum, dalam #Perhitungan-Aktuaria sesuai #PSAK-219, yang digunakan sebagai dasar adalah Gaji Pokok + Tunjangan Tetap.

- Gaji Pokok → Komponen utama yang diperhitungkan dalam manfaat pasca kerja.
- Tunjangan Tetap → Tunjangan yang diberikan secara rutin, nilainya tetap dan tidak bergantung pada kehadiran atau kinerja, seperti tunjangan jabatan, tunjangan keluarga, atau tunjangan fungsional.
- PPh 21 (Pajak Penghasilan Pasal 21) → Biasanya tidak dimasukkan dalam perhitungan #imbalan-kerja, karena bukan bagian dari manfaat yang diterima karyawan, melainkan kewajiban pajak yang ditanggung perusahaan atau karyawan.

#### 48. Bagaimana menangani karyawan dengan struktur gaji yang berbeda, seperti kondisi tanpa tunjangan tetap?

Apabila ada kondisi, misalnya:
- Karyawan A: Hanya memiliki gaji pokok → #Perhitungan-Aktuaria berdasarkan gaji pokok.
- Karyawan B: Memiliki gaji pokok + tunjangan tetap (misalnya tunjangan transportasi) → perhitungan mencakup keduanya.

Pendekatan ini memastikan kewajiban #imbalan-kerja sesuai dengan kebijakan perusahaan dan menghindari kesalahan perhitungan.

#### 49. Apakah tunjangan variabel juga perlu dihitung jika menjadi bagian dari total gaji pada beberapa karyawan?

Tunjangan variabel, seperti bonus tahunan atau insentif berbasis kinerja, umumnya tidak dihitung dalam #Perhitungan-Aktuaria #imbalan-kerja karena sifatnya tidak tetap dan tidak bisa diproyeksikan secara konsisten. Namun, jika perusahaan secara jelas menyatakan bahwa tunjangan variabel ini termasuk dalam perhitungan #Imbalan-pasca-kerja, maka harus dimasukkan dalam total gaji.

#### 50. Apakah tingkat dan rata-rata #Kenaikan-Gaji dihitung dari Gaji Pokok saja atau Total Gaji (gaji pokok + tunjangan tetap)?

Diketahui:
Seorang karyawan memiliki Gaji Pokok sebesar Rp10 juta dan Tunjangan Tetap Rp2 juta. Total gaji menjadi Rp12 juta. Setelah satu tahun, Gaji Pokok naik menjadi Rp11 juta, dan Tunjangan Tetap tetap Rp2 juta. Total gaji baru menjadi Rp13 juta.

Pertanyaan:
	Berapa tingkat #Kenaikan-Gaji berdasarkan Gaji Pokok?
	Berapa tingkat #Kenaikan-Gaji berdasarkan Total Gaji?

Jawaban:
- Tingkat #Kenaikan-Gaji Pokok:
	Tingkat Kenaikan = (Gaji Pokok Baru − Gaji Pokok Lama) / Gaji Pokok Lama × 100%
	Tingkat Kenaikan = (11.000.000 − 10.000.000) / 10.000.000 × 100%
	Tingkat Kenaikan = 10%

- Tingkat Kenaikan Total Gaji:
	Tingkat Kenaikan = (Total Gaji Baru − Total Gaji Lama) / Total Gaji Lama × 100%
	Tingkat Kenaikan = (13.000.000 − 12.000.000) / 12.000.000 × 100%
	Tingkat Kenaikan = 8,33%

Total gaji lebih relevan karena #Imbalan-pasca-kerja seperti #Pesangon atau manfaat pensiun sering dihitung berdasarkan **gaji terakhir yang mencakup gaji pokok dan tunjangan tetap**. Jika hanya menggunakan gaji pokok, perhitungan bisa menjadi undervalued.

Referensi: [[Perhitungan Tetap#Total-Gaji]]
#### 51. Apakah data karyawan yang mengundurkan diri tetap harus dicatat, meskipun mereka tidak dihitung dalam jumlah total karyawan untuk perhitungan?

Ya, data mereka tetap perlu dicatat dalam #file-template excel. Hal ini penting untuk mendokumentasikan movement (perubahan status) karyawan, seperti pengunduran diri, sehingga #aktuaris dapat melacak perubahan populasi karyawan dari waktu ke waktu. Data ini juga relevan untuk perhitungan terkait kewajiban settlement atau curtailment, apabila ada kompensasi yang diberikan pada saat pengunduran diri.

#### 52. Jika jumlah karyawan bertambah tetapi liabilitas dari #Perhitungan-Aktuaria lebih kecil dibandingkan tahun sebelumnya, bagaimana cara memastikan Uang Penghargaan Hak (UPH) sudah dihitung dengan benar?

Dalam perhitungan manfaat UU Cipta Kerja (UUCK), jika jumlah karyawan bertambah tetapi liabilitas lebih kecil dibanding tahun sebelumnya, perlu dipastikan bahwa Uang Penghargaan Hak (UPH) sudah dimasukkan dalam **formula 15% x** ( #Pesangon **+ Penghargaan Masa Kerja).** Seringkali, #Laporan-Aktuaria sebelumnya hanya menyebutkan total perhitungan tanpa secara eksplisit menuliskan UPH secara terpisah. Oleh karena itu, penting untuk memeriksa kembali perumusan manfaat guna memastikan UPH sudah diperhitungkan dengan benar dalam kewajiban aktuaria.

#### 53. Bagaimana menangani data gaji karyawan dalam perhitungan #imbalan-kerja, terutama dengan adanya #Kenaikan-Gaji di bulan tertentu sesuai PP No. 5 Tahun 2024?

Dalam #Perhitungan-Aktuaria, yang digunakan adalah gaji posisi valuasi (misalnya, per Desember 2024) karena mencerminkan kondisi #laporan-keuangan. Jika ada #Kenaikan-Gaji di bulan tertentu tahun 2025, maka sesuai PP No. 5 Tahun 2024, data tersebut **tidak langsung digunakan dalam perhitungan kewajiban, tetapi tetap relevan** untuk memvalidasi [[Asumsi#Tingkat-Kenaikan-Gaji]] di masa depan. Meskipun gaji Desember menjadi dasar utama, data #Kenaikan-Gaji Februari bisa diberikan sebagai tambahan untuk analisis lebih lanjut, seperti mengevaluasi dampaknya terhadap kewajiban atau menyesuaikan asumsi aktuaria.

#### 54. Bagaimana penjelasannya jika jumlah karyawan berkurang dan #Kenaikan-Gaji tidak signifikan, namun alokasi biaya justru meningkat cukup besar?

1. Karyawan tetap: Dalam #Perhitungan-Aktuaria, #Kenaikan-Gaji menjadi salah satu faktor utama yang meningkatkan nilai kewajiban, karena estimasi manfaat di masa depan bergantung pada besaran gaji terakhir karyawan. Maka dari itu, beban #imbalan-kerja tetap bisa meningkat walau jumlah karyawannya berkurang.
2. Karyawan kontrak: Terdapat perbedaan antara asumsi awal dan realisasi, baik dari sisi #Kenaikan-Gaji maupun komposisi data karyawan. Beberapa kemungkinan penyebab lonjakan biaya di antaranya:
    - Penambahan jumlah karyawan kontrak.
    - Bertambahnya masa kerja rata-rata yang berpengaruh pada nilai manfaat.
    - Perubahan [[Asumsi]] atau parameter aktuaria seperti #Tingkat-Diskonto atau tingkat keluar-masuk karyawan.
    - Penggunaan data aktual menggantikan estimasi yang sebelumnya lebih rendah.

**Rekomendasi**: Melakukan **pengecekan kembali terhadap data gaji tahun sebelumnya**, apakah saat itu sudah mencakup seluruh komponen tetap yang relevan. Jika sebelumnya hanya menggunakan gaji pokok sebagai dasar perhitungan, maka hasil perhitungan tahun tersebut kemungkinan terlalu rendah. Ketika datanya diperbaiki di tahun berikutnya menggunakan total gaji, maka kenaikan hasil perhitungan adalah hal yang wajar.

Referensi: [[Asumsi#Tingkat-Kenaikan-Gaji]]

---
## #FAQ Inovasi Sistem

#### 1. Apa itu #Kalkulator Manfaat Karyawan?

#Kalkulator Manfaat Karyawan merupakan alat digital yang digunakan untuk **menghitung dan memproyeksikan berbagai jenis** #Manfaat-Karyawan beserta kewajiban perusahaan. Alat ini memberikan estimasi nilai manfaat pensiun, santunan kematian, serta manfaat jika karyawan mengalami sakit berkepanjangan atau cacat tetap.

Meskipun hasil perhitungan bersifat estimasi, perusahaan tetap memperoleh acuan nilai kewajiban berdasarkan perhitungan aktuaria yang detail, yang dihitung menggunakan data lengkap dari perusahaan. Nilai yang dihasilkan kemudian **diolah dan disajikan langsung dalam bentuk buku** #laporan-aktuaria.

Pengguna hanya perlu memasukkan data dasar, seperti nama dan tanggal lahir, lalu kalkulator akan memprosesnya dengan rumus aktuaria. Untuk melihat contoh #laporan-aktuaria yang lebih rinci, kami menyediakan #draft-laporan sesuai #PSAK-24, yang dapat diakses melalui tautan berikut: [https://apps.valuasiaktuaria.com/contoh_buku_psak24](https://apps.valuasiaktuaria.com/contoh_buku_psak24).

#### 2. Apakah data yang saya masukkan aman dan terlindungi?

Tentu saja. Keamanan dan privasi data pelanggan adalah prioritas utama kami. Berikut adalah langkah-langkah yang kami ambil antara lain:
- **Enkripsi**: Seluruh data dimasukkan ke dalam sistem kami dilindungi dengan enkripsi tingkat tinggi. Ini berarti bahwa informasi perusahaan disandikan dengan cara yang hanya dapat dibaca oleh sistem kami, menjadikannya sulit untuk diakses oleh pihak ketiga yang tidak sah.
- **Penghapusan Data**: Kami memiliki kebijakan untuk menghapus data secara berkala. Hal ini memastikan bahwa informasi sensitif Bapak / Ibu tidak disimpan lebih lama dari yang diperlukan dan mengurangi risiko kebocoran data.
- **Penggunaan Data Terbatas**: Meskipun kami menyimpan beberapa informasi seperti nama, nomor telepon, dan alamat email PIC untuk kebutuhan komunikasi di masa mendatang, kami tidak akan menggunakan data tersebut untuk tujuan lain tanpa persetujuan perusahaan. Data ini hanya akan digunakan untuk komunikasi relevan.
- **Keamanan Infrastruktur**: Infrastruktur IT kami dilindungi oleh protokol keamanan canggih dan firewall terbaru untuk melindungi terhadap ancaman eksternal, seperti serangan hacker atau malware.

Jika pihak perusahaan memiliki pertanyaan atau kekhawatiran lebih lanjut mengenai keamanan data, jangan ragu untuk menghubungi kami.

#### 3. Informasi apa saja yang perlu dimasukkan ke dalam #kalkulator?

Untuk menggunakan #kalkulator manfaat karyawan, perlu memasukkan beberapa informasi dasar karyawan dan perusahaan, meliputi:
- **Nama Karyawan**
- **Besaran Pendapatan/Gaji Bulanan**
- **Tanggal Lahir**
- **Tanggal Mulai Kerja**
- **Usia Pensiun**
- **Tanggal Valuasi**
- **Nama Perusahaan**
- **Alamat Email** untuk diskusi lebih lanjut
- **Nomor Telepon** untuk diskusi lebih lanjut

#### 4. Dapatkah saya menggunakan #kalkulator ini untuk memprediksi kebutuhan #Manfaat-Karyawan di masa mendatang?

Tentu, #kalkulator manfaat karyawan ini dirancang untuk memberikan gambaran awal mengenai estimasi #Manfaat-Karyawan pada saat tertentu. Namun, ada beberapa hal yang perlu dipertimbangkan:

- **Estimasi Awal**: Sebagai alat berbasis kecerdasan buatan, kalkulator ini mampu menghasilkan estimasi berdasarkan data yang dimasukkan. Estimasi ini memberikan gambaran umum tentang potensi kewajiban #Manfaat-Karyawan perusahaan terkait.
- **Keunikan Setiap Perusahaan**: Setiap perusahaan memiliki karakteristik dan kebutuhan yang masing-masing. Faktor-faktor seperti demografi karyawan, struktur gaji, kebijakan pensiun, dan lainnya mempengaruhi kalkulasi #Manfaat-Karyawan. Oleh karena itu, meskipun kalkulator ini dapat memberikan gambaran awal, disarankan untuk berdiskusi lebih lanjut dengan kami untuk mendapatkan analisis yang lebih dalam dan sesuai dengan kebutuhan perusahaan Anda.
- **Diskusi Mendalam**: Untuk mendapatkan gambaran yang lebih akurat tentang kewajiban #Manfaat-Karyawan di masa mendatang, kami menyarankan Bapak/Ibu untuk berdiskusi langsung dengan tim kami. Hal ini untuk dapat memahami lebih dalam tentang kebutuhan, harapan, dan kondisi perusahaan Anda sehingga dapat memberikan solusi dan perhitungan yang tepat.

#### 5. Bagaimana saya bisa memahami hasil perhitungan yang diberikan?

Dalam konteks #Manfaat-Karyawan, memang memerlukan pemahaman hasil perhitungan aktuaria yang mendalam. Langkah-langkah untuk memudahkan pemahaman hasil perhitungan dari #kalkulator:

1. **Memahami Dasar Perhitungan**
	- Estimasi #Manfaat-Karyawan: Ini adalah proyeksi nilai yang akan diterima karyawan saat manfaat jatuh tempo, seperti pensiun, meninggal dunia, atau sakit berkepanjangan, berdasarkan data karyawan dan peraturan yang berlaku di perusahaan terkait.
	- #Nilai-Kini-Kewajiban Perusahaan: Ini adalah proyeksi cadangan yang harus dicadangkan perusahaan untuk memenuhi kewajiban masa depan terhadap karyawan, membantu dalam perencanaan keuangan dan memastikan kesiapan dana.

2. **Memahami #Peraturan-Perusahaan dan Peraturan Pemerintah yang berlaku**
	Setiap perusahaan memiliki peraturan dan ketentuan sendiri terkait #Manfaat-Karyawan. Juga, peraturan pemerintah yang berlaku membantu memahami dasar dan #Asumsi dalam perhitungan manfaat.

3. Melakukan konsultasi dengan Tim Ahli KKA
	Jika memiliki pertanyaan, bisa segera dikonsultasikan langsung dengan tim ahli KKA. kami dapat memberikan penjelasan lebih rinci dan memandu perusahaan pada setiap aspek perhitungan.

Dengan langkah-langkah di atas, pihak perusahaan akan lebih siap untuk memahami dan memanfaatkan hasil perhitungan yang diberikan oleh kalkulator #Manfaat-Karyawan untuk kepentingan perusahaan terkait.

---








