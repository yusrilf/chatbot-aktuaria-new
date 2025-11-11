---
keywords:
  - PSAK 219
  - imbalan pasca kerja
  - PSAK 24
  - transisi
  - IFRIC
  - defined benefit
  - defined contribution
  - UUK
  - UUCK
  - aktuaria
  - sains aktuaria
  - matematis aktuaria
  - valuasi aktuaria
difficulty: basic
estimated_reading: 90 minutes
target_audience:
  - management
  - hr
  - finance
  - legal
  - actuaries
document_type: foundation
last_updated: 2025-07-28
version: "2025.1"
related_regulations:
  - PSAK_219
  - UU_13_2003
  - UU_11_2020
  - PP_35_2021
  - IFRS_19

# SCOPE CONTROL - AKTUARIA FOCUS  
domain: aktuaria
scope: sains_aktuaria_imbalan_kerja
exclude_domains:
  - asuransi_umum
  - perbankan_komersial
  - insurance_reserves
  - claim_accounting
  - underwriting
context_boundary: strict

# GLOBAL ANTI-HALLUCINATION CONTROLS
semantic_focus:
  - perhitungan aktuaria
  - valuasi aktuaria
  - metode aktuaria
  - standar aktuaria
  - imbalan pasca kerja
  - PSAK 219 implementation
  - defined benefit calculation
  - projected unit credit method
  - sains aktuaria
  - matematis aktuaria
  - IFRIC interpretation
  - employee benefits accounting

semantic_exclude:
  - outstanding claims reserve
  - insurance underwriting
  - claim settlement
  - premium calculation
  - general insurance
  - life insurance products
  - bancassurance
  - RBNS
  - IBNR
  - insurance marketing
  - insurance sales
  - banking products
  - investment products

fallback_response: "Informasi tidak tersedia dalam konteks sains aktuaria dan imbalan kerja. Silakan ajukan pertanyaan terkait PSAK 219, valuasi aktuaria, atau imbalan kerja karyawan."
---

---
# BAB 1: Dasar Imbalan Pasca Kerja PSAK 219

**📖 Bagian dari:** [PANDUAN LENGKAP IMBALAN KERJA PSAK 219](README.md)

> **SCOPE DELIMITER**: Dokumen ini KHUSUS membahas aspek aktuaria imbalan kerja dan employment benefits. 

---

## 🎯 **Tujuan Pembelajaran**

Setelah membaca bab ini, pembaca akan mampu:

- [ ] Memahami konsep dasar imbalan pasca kerja melalui ilustrasi praktis
- [ ] Membedakan antara defined contribution dan defined benefit plan
- [ ] Menjelaskan transisi dari PSAK 24 ke PSAK 219 dan dampaknya
- [ ] Mengidentifikasi kebutuhan valuasi aktuaria dalam perusahaan
- [ ] Memahami regulasi dan compliance requirements yang berlaku

---

## ⚡ **Quick Reference**

### 🔤 **Key Concepts:**
- **Imbalan Pasca Kerja:** Manfaat yang diberikan kepada karyawan setelah masa kerja berakhir (pesangon, pensiun, penghargaan masa kerja)
- **PSAK 219:** Standar akuntansi Indonesia untuk imbalan kerja, efektif 1 Januari 2025 (menggantikan PSAK 24)
- **PUC Method:** Projected Unit Credit - metode aktuaria untuk perhitungan kewajiban imbalan kerja
- **IFRIC Implementation:** Pengakuan kewajiban dimulai 24 tahun sebelum usia pensiun normal

### 📅 **Critical Dates:**
- **1 Jan 2025:** PSAK 219 effective date (replaces PSAK 24)
- **31 Des 2024:** Last reporting period under PSAK 24
- **24 tahun:** IFRIC rule - kewajiban mulai diakui 24 tahun sebelum usia pensiun
- **56-65 tahun:** Usia pensiun normal sesuai PP No. 45 Tahun 2015 (bertahap)

### ⚖️ **Key Regulations:**
- **UU No. 13/2003 (UUK):** Framework dasar imbalan kerja
- **UU No. 11/2020 (UUCK):** Cipta Kerja - revised benefit calculation
- **PP No. 35/2021:** Implementasi UUCK untuk kompensasi PKWT
- **PSAK 219:** Accounting standard effective 2025

---

## 📋 **Outline Bab**

- [Ilustrasi Imbalan Pasca Kerja](#ilustrasi-imbalan-pasca-kerja)
- [Pengertian Imbalan Pasca Kerja](#pengertian-imbalan-pasca-kerja)
- [Standar PSAK 24](#standar-psak-24)
- [Transisi Nomenklatur ke PSAK 219](#transisi-nomenklatur-ke-psak-219)
- [Ruang Lingkup PSAK 219](#ruang-lingkup-psak-219)
- [Standar SAK EP (SAK ETAP)](#standar-sak-ep-sak-etap)
- [Regulasi PSAK 219 terhadap IFRS 19](#regulasi-psak-219-terhadap-ifrs-19)
- [Penerapan IFRIC AD](#penerapan-ifric-ad)
- [Kebutuhan Valuasi Aktuaria](#kebutuhan-valuasi-aktuaria)
- [Tantangan dalam Audit Keuangan](#tantangan-dalam-audit-keuangan)

---

## #ilustrasi-imbalan-pasca-kerja

### **Janji Rudi - Memahami Konsep Dasar**

Untuk memudahkan pemahaman tentang **imbalan pasca kerja**, mari amati situasi ini:

> **💡 Studi Kasus Ilustratif:** Rudi berencana mengajak istri dan anak-anaknya berlibur ke Swiss 5 tahun dari sekarang, yaitu di tahun 2030. Rudi mengestimasi bahwa biaya yang dibutuhkan sebesar Rp100.000.000,- dengan masa kerja total hingga mencapai Usia Pensiun adalah 20 tahun.

Sama halnya dengan menghitung cicilan KPR, Rudi perlu menabung Rp17.235.695 per tahun. Jika semuanya berjalan sesuai dengan rencana dan estimasi perhitungan seperti berikut:

| Tahun | Saldo Awal Tahun | Tabungan per Tahun | Hasil Investasi | Saldo Akhir Tahun |
|:------|:-----------------|:-------------------|:----------------|:------------------|
| 1     | -                | 17.235.695        | 861.785         | 18.097.480        |
| 2     | 18.097.480       | 17.235.695        | 1.766.659       | 37.099.833        |
| 3     | 37.099.833       | 17.235.695        | 2.716.776       | 57.052.305        |
| 4     | 57.052.305       | 17.235.695        | 3.714.400       | 78.002.400        |
| 5     | 78.002.400       | 17.235.695        | 4.761.905       | **100.000.000**   |

Setiap tahun, Rudi menyisihkan sejumlah tabungan dan ingin mendapat hasil investasi agar dananya berkembang. Jika Rudi tidak menempatkan tabungannya di deposito dan lebih memilih menyimpannya sendiri, maka ia perlu menambahkan dana tabungan tahunan beserta potensi hasil investasinya.

### **Dampak Perubahan Asumsi**

**Skenario 1: Kenaikan Bunga di Tahun Ketiga**  
Bagaimana jika bunga deposito berubah di tahun ketiga, misalnya menjadi 6%? Beginilah ilustrasi besaran tabungan Rudi per tahunnya dalam kondisi tersebut:

| Tahun | Bunga | Saldo Awal Tahun | Tabungan per Tahun | Hasil Investasi | Saldo Akhir Tahun |
|:------|:------|:-----------------|:-------------------|:----------------|:------------------|
| 1     | 5%    | -                | 17.235.695        | 861.785         | 18.097.480        |
| 2     | 5%    | 18.097.480       | 17.235.695        | 1.766.659       | 37.099.833        |
| 3     | **6%**| 37.099.833       | **16.539.216**    | 3.218.343       | 56.857.393        |
| 4     | **6%**| 56.857.393       | **16.539.216**    | 4.403.797       | 77.800.406        |
| 5     | **6%**| 77.800.406       | **16.539.216**    | 5.660.377       | **100.000.000**   |

Akibat kenaikan bunga ini, **Rudi untung** karena beban tabungan yang harus disetorkan setiap tahun berkurang.

**Skenario 2: Fluktuasi Bunga**  
Namun, keadaan pasar bisa saja berbalik. Misalkan, ternyata di tahun keempat dan kelima, target investasi turun hingga ke angka 4%, sehingga beban tabungan kembali naik dan kondisi **Rudi rugi**:

| Tahun | Bunga | Saldo Awal Tahun | Tabungan per Tahun | Hasil Investasi | Saldo Akhir Tahun |
|:------|:------|:-----------------|:-------------------|:----------------|:------------------|
| 1     | 5%    | -                | 17.235.695        | 861.785         | 18.097.480        |
| 2     | 5%    | 18.097.480       | 17.235.695        | 1.766.659       | 37.099.833        |
| 3     | **6%**| 37.099.833       | **16.539.216**    | 3.218.343       | 56.857.393        |
| 4     | **4%**| 56.857.393       | **18.148.116**    | 3.000.220       | 78.005.729        |
| 5     | **4%**| 78.005.729       | **18.148.116**    | 3.846.154       | **100.000.000**   |

> **📊 Insight:** Selama proses menabung, Rudi bisa saja berada dalam posisi "untung" atau "rugi" ketika mengumpulkan dana liburan tergantung pada fluktuasi tingkat bunga.

### **Ilustrasi Dana Imbalan Pasca Kerja – Komponen Biaya**

Gambaran ini mirip dengan janji perusahaan kepada karyawannya—janji manfaat di masa depan dengan jumlah dan waktu yang pasti, dengan analogi berikut:

| Tahun | Bunga | Saldo Awal Tahun | **Tabungan per Tahun (Service Cost)** | **Hasil Investasi (Interest Cost)** | Saldo Akhir Tahun |
|:------|:------|:-----------------|:--------------------------------------|:------------------------------------|:------------------|
| 1     | 5%    | -                | 17.235.695                           | 861.785                             | 18.097.480        |
| 2     | 5%    | 18.097.480       | 17.235.695                           | 1.766.659                          | 37.099.833        |
| 3     | 6%    | 37.099.833       | 16.539.216                           | 3.218.343                          | **56.857.393**    |
| 4     | 4%    | 56.857.393       | 18.148.116                           | 3.000.220                          | **78.005.729**    |
| 5     | 4%    | 78.005.729       | 18.148.116                           | 3.846.154                          | **100.000.000**   |

**Penjelasan Komponen:**
- **Service Cost** = _Tabungan per tahun_: mencerminkan kewajiban tahunan terhadap pegawai
- **Interest Cost** = _Hasil Investasi_: estimasi hasil dari asumsi bunga atas saldo awal tahun
- **Gain/Loss** terjadi saat:
    1. Ada **perubahan asumsi** (misalnya: tingkat bunga berubah dari 5% ke 6%, lalu ke 4%)
    2. Ada **penyesuaian jumlah tabungan** untuk mencapai target akhir

> **⚠️ Perbedaan Kunci:** Berbeda dari janji Rudi yang **tidak memiliki standar** atau peraturan akuntansi dan **tidak ada asumsi** tentang tabungan tadi. Sementara, manfaat karyawan dari perusahaan memiliki peraturan atau standar nasional maupun internasional yang harus dipatuhi.

Perusahaan juga menggunakan perhitungan aktuaria yang sangat mempertimbangkan berbagai faktor, seperti asumsi dan metode aktuaria. Dimulai dari pemahaman dasar, proses perhitungan aktuaria hingga penyajian di laporan keuangan, akan dibahas dalam buku panduan ini secara menyeluruh.

---

## #pengertian-imbalan-pasca-kerja

**Imbalan pasca kerja** adalah sejumlah manfaat yang disediakan oleh perusahaan kepada karyawan setelah mereka menyelesaikan masa kerjanya. Ini adalah bentuk apresiasi atas kontribusi yang telah diberikan selama bertahun-tahun. Imbalan ini tidak hanya penting bagi karyawan seperti Rudi, tetapi juga bagi perusahaan dalam menjaga reputasi dan komitmennya terhadap kesejahteraan karyawan.

### **Jenis-jenis Imbalan Pasca Kerja**

Imbalan pasca kerja mencakup berbagai bentuk manfaat seperti:

#### **Uang Pesangon**
Sejumlah uang yang diberikan saat pengakhiran hubungan kerja, baik karena pensiun / pemutusan hubungan kerja.

#### **Uang Pisah**
Uang penghargaan kontribusi kepada karyawan yang mengundurkan diri secara sukarela.

#### **Uang Penghargaan Masa Kerja**
Imbalan yang diberikan kepada karyawan dengan masa kerja tertentu di perusahaan.

#### **Uang Penggantian Hak**
Kompensasi pengganti hak-hak yang tidak diambil oleh karyawan selama masa kerja, seperti cuti yang belum diambil.

#### **Uang Kompensasi / Uang Duka**
Uang bantuan sebagai ganti rugi atau imbalan atas kondisi tertentu / diberikan kepada ahli waris karyawan yang meninggal dunia.

### **Kategori Utama Imbalan Pasca Kerja**

Pengakuan, pengukuran, dan pengungkapan dalam imbalan pasca kerja terbagi menjadi dua kategori utama:

#### **1. Iuran Pasti (Defined Contribution Plan)**

- Perusahaan hanya wajib menyetor sejumlah iuran tertentu ke dana pensiun
- Setelah iuran disetor, risiko dan tanggung jawab pindah ke karyawan

**Pengakuan dan Pengukuran:**
1. Diakui sebagai **biaya/liabilitas**, saat karyawan memberikan jasanya
2. Jika kelebihan dibayar sebelum jatuh tempo, diakui sebagai Aset (_dibayar di muka_)

**Pengungkapan:**
3. Jumlah iuran yang dibayar wajib diungkapkan dalam laporan keuangan

#### **2. Imbalan Pasti (Defined Benefit Plan)**

- Perusahaan bertanggung jawab atas jumlah manfaat pensiun tertentu yang akan diterima karyawan, terlepas dari hasil investasi dana pensiun

**Pengakuan dan Pengukuran:**
1. Biaya jasa dicatat di Laporan Laba Rugi
2. Biaya neto atas liabilitas/aset imbalan pasti juga dicatat di Laporan Laba Rugi
3. Selisih pengukuran (_remeasurement_) masuk ke pendapatan komprehensif lainnya (**OCI**)

**Pengungkapan:**
4. Perusahaan diwajibkan mengungkapkan:
   - Karakteristik dan risiko program
   - Jumlah manfaat yang timbul dalam laporan keuangan
   - Bagaimana dampak program terhadap arus kas masa depan

---

## #standar-psak-24

PSAK 24 merupakan **Standar Akuntansi Keuangan (SAK)** yang memberikan pedoman kepada perusahaan dalam mencatat, mengukur, dan melaporkan imbalan kerja yang diberikan kepada karyawan. Standar ini mencakup imbalan yang disediakan langsung oleh perusahaan maupun melalui pihak ketiga.

### **Karakteristik PSAK 24**

1. **Diperbarui pada 27 Agustus 2014 oleh Dewan Standar Akuntansi Keuangan (DSAK IAI)** agar selaras dengan praktik akuntansi global
2. **Berlaku untuk semua jenis imbalan kerja**, kecuali imbalan berbasis saham (PSAK 53)
3. **Sudah diterapkan oleh berbagai entitas** (perusahaan publik, asuransi, perbankan, BUMN, dan dana pensiun), khususnya yang terdaftar atau akan mendaftar di pasar modal

Standar ini mengatur pencatatan beban, dengan mewajibkan perusahaan mengakui liabilitas atas imbalan kerja ketika karyawan telah memberikan jasa dan berhak atas imbalan di masa depan. Tidak hanya memengaruhi pembukuan keuangan, tetapi juga memengaruhi manajemen sumber daya manusia, kepatuhan hukum, dan stabilitas keuangan perusahaan.

### **Dampak PSAK 24 terhadap Fungsi-Fungsi Perusahaan**

#### **Human Resources**
1. Efektivitas manajemen imbalan kerja
2. Perencanaan keuangan jangka panjang
3. Retensi dan kepuasan bagi karyawan

#### **Legal**
1. Kepatuhan regulasi pelaporan keuangan
2. Konsistensi dan akurasi audit keuangan
3. Perlindungan hak kontrak karyawan

#### **Finance & Accounting**
1. Prinsip akuntansi _accrual basis_
2. Pengungkapan kewajiban akrual
3. Stabilitas laba dan arus kas perusahaan

Selama bertahun-tahun, PSAK 24 telah menjadi standar yang mengatur pengakuan, pengukuran, dan pelaporan imbalan kerja di Indonesia. Namun, dalam dunia yang terus berkembang dan terintegrasi secara global, standar akuntansi yang berlaku di Indonesia pun harus terus beradaptasi agar tetap relevan dan akurat.

---

## #transisi-nomenklatur-ke-psak-219

Langkah ini mencerminkan komitmen tentang transparansi dan konsistensi dalam pelaporan keuangan, khususnya dalam hal imbalan kerja.

### **Landasan Hukum Perubahan**

Dengan perkembangan _International Financial Reporting Standards_ (IFRS) dan perubahan dalam dunia bisnis global, nomenklatur PSAK 24 telah **diubah menjadi PSAK 219**, berlaku **efektif sejak 1 Januari 2025**. Perubahan ini bertujuan memberikan panduan yang lebih komprehensif sesuai kemajuan ilmu akuntansi dan praktik akuntansi global.

Jadi, perusahaan diharapkan dapat menyesuaikan diri dengan panduan PSAK 219 terkini untuk mengatasi keterbatasan yang ada dalam PSAK 24, khususnya dalam penyesuaian asumsi aktuaria dan pengukuran kewajiban dalam penyusunan laporan dan audit keuangan.

### **Integrasi dengan Standar Akuntansi Lain**

Di Indonesia, PSAK 219 (Imbalan Kerja) tidak berdiri sendiri, melainkan terhubung dengan berbagai standar akuntansi lain dalam kerangka pelaporan keuangan. Beberapa di antaranya adalah:

#### **PSAK 1 (Penyajian Laporan Keuangan)**
Dikaitkan ke PSAK 219, kewajiban imbalan kerja diakui sebagai bagian dari liabilitas, dan harus diungkapkan secara jelas dalam laporan keuangan.

#### **PSAK 2 (Laporan Arus Kas)**
PSAK 219 memengaruhi laporan arus kas, khususnya terkait pengeluaran dana untuk pembayaran imbalan kerja. Ini harus diklasifikasikan sesuai dengan ketentuan PSAK 2.

#### **PSAK 5 (Segmen Operasi)**
Dalam PSAK 5, imbalan kerja yang dihitung berdasarkan PSAK 219 mungkin perlu disegmentasi dalam laporan keuangan untuk mencerminkan kinerja segmen yang berbeda, memungkinkan pemangku kepentingan memahami dampaknya dalam konteks operasional.

#### **PSAK 53 (Akuntansi Imbalan Kerja dengan Pembayaran Saham)**
Mengatur pengakuan, pengukuran, dan pengungkapan imbalan kerja dalam bentuk pembayaran saham dan saham opsional. Meskipun jenis imbalannya berbeda, keduanya saling terkait dalam hal pengakuan total kewajiban imbalan kerja di laporan keuangan.

### **Dasar Hukum PSAK 219**

Beberapa peraturan dan undang-undang yang menjadi dasar hukum penerapan PSAK 219:

#### **Undang-Undang Ketenagakerjaan Nomor 13 Tahun 2003 (UUK)**
Mengatur hak-hak pekerja, termasuk pesangon dan jaminan pensiun, dengan pengakuan dan pengukuran imbalan kerja PSAK 219.

#### **Peraturan Pemerintah No. 78 Tahun 2015**
Mengenai pengupahan, penghitungan dan pembayaran imbalan kerja secara adil yang mendukung penerapan PSAK 219.

#### **Undang-Undang Ketenagakerjaan No. 11 Tahun 2020**
Menekankan penyesuaian perhitungan kewajiban aktuaria untuk mencegah _overcost_ dan _overtaxation_, sesuai standar akuntansi.

#### **Peraturan Pemerintah No. 34 Tahun 2021**
Menerangkan bahwa perusahaan dapat mengurangi beban kewajiban jangka pendeknya karena TKA (Tenaga Kerja Asing) tidak berhak mendapat kompensasi.

#### **Peraturan Pemerintah No. 35 Tahun 2021**
Peraturan terkait kompensasi pekerja yang terkena PHK, perjanjian kerja waktu tertentu, dan lainnya, sejalan dengan standar ketenagakerjaan yang berlaku.

#### **Undang-Undang Ketenagakerjaan No. 6 Tahun 2023 (UUCK)**
Memastikan pemenuhan kewajiban aktuaria perusahaan terhadap karyawan, **dengan penekanan pada pengakuan liabilitas imbalan kerja**.

---

## #ruang-lingkup-psak-219

Jenis-jenis imbalan kerja yang dicakup oleh PSAK 219 meliputi:

#### **Imbalan Kerja Jangka Pendek**
Contohnya: gaji, upah, bonus, dan tunjangan lain yang dibayarkan dalam waktu kurang dari 12 bulan setelah akhir periode kerja.

#### **Imbalan Pasca Kerja**
Manfaat pensiun yang diberikan setelah karyawan menyelesaikan masa kerjanya. PSAK 219 meminta perusahaan mengakui dan mengukur kewajiban ini dengan nilai yang mencerminkan besaran sebenarnya.

#### **Imbalan Jangka Panjang Lainnya**
Meliputi hak cuti jangka panjang, jaminan kesehatan pasca pensiun, dan manfaat serupa yang dibayarkan lebih dari 12 bulan setelah periode kerja berakhir.

#### **Pesangon Pemutusan Kerja**
Mengatur kewajiban dan pengukuran pesangon saat karyawan berhenti bekerja, baik pemutusan secara sukarela maupun tidak sukarela.

---

## #standar-sak-ep-sak-etap

Sekarang kita paham bagaimana ruang lingkup standar ini menuntut perusahaan menggunakan metode aktuaria dan perhitungan yang kompleks dalam penyusunan laporan keuangan.

Namun, tidak semua entitas memiliki skala dan karakteristik yang sama. Banyak perusahaan kecil dan menengah memiliki kebutuhan dan kapasitas pelaporan yang berbeda. Untuk itulah, SAK ETAP hadir sebagai alternatif standar akuntansi yang lebih sederhana.

**SAK EP** merupakan singkatan dari _Standar Akuntansi Keuangan Entitas Tanpa Akuntabilitas Publik_, yang diterbitkan oleh **Ikatan Akuntan Indonesia (IAI)**.

1. Sama halnya dengan PSAK 219, mulai 1 Januari 2025, SAK ETAP digantikan oleh SAK EP
2. Ditujukan bagi entitas privat yang tidak memiliki kewajiban akuntabilitas publik serta tidak menerbitkan laporan keuangan kepada publik

SAK EP memberikan alternatif bagi entitas yang menginginkan pelaporan keuangan yang lebih komprehensif dan terstruktur dibandingkan SAK ETAP. Karena SAK EP disusun berbasis _IFRS for SMEs_, sehingga mempertimbangkan kebutuhan entitas privat untuk menyusun laporan keuangan yang andal, namun tidak serumit PSAK berbasis IFRS seperti PSAK 219.

#### **Entitas yang menggunakan SAK EP umumnya:**

- Tidak memiliki kewajiban menyampaikan laporan keuangan kepada publik
- Tidak terdaftar di bursa efek
- Skala usaha relatif kecil atau menengah
- Tidak memiliki kepentingan publik yang signifikan

**Contoh entitas pengguna SAK EP:**
- Koperasi simpan pinjam
- Yayasan pendidikan
- Perusahaan keluarga skala kecil-menengah
- CV atau firma

Dalam SAK EP, kewajiban imbalan kerja tidak dihitung secara aktuaria, melainkan dicatat saat manfaat tersebut menjadi kewajiban hukum yang pasti. Tidak ada kewajiban menghitung nilai sekarang dari manfaat masa depan seperti pada PSAK 219. Namun, banyak entitas tetap membuat estimasi internal atas kewajiban ini, terutama jika ingin memiliki proyeksi beban keuangan jangka panjang.

Terkait pelaporan keuangan terdapat penambahan dampak berupa komponen Other Comprehensive Income (OCI).

Penambahan tersebut memberikan ruang untuk mengakui perubahan nilai kewajiban imbalan kerja secara tidak langsung. Hal ini membantu entitas kecil dan menengah menyajikan laporan keuangan yang lebih transparan dan mencerminkan fluktuasi kewajiban jangka panjang tanpa membebani laba rugi secara langsung.

---

## #regulasi-psak-219-terhadap-ifrs-19

PSAK 219 diadopsi dengan mengacu pada **IFRS 19 (Employee Benefits)**, yang merupakan standar internasional mengenai pelaporan imbalan kerja. Hal ini bertujuan untuk memastikan bahwa laporan keuangan perusahaan di Indonesia:

- Konsisten dengan praktik internasional
- Meningkatkan transparansi
- Meningkatkan komparabilitas pada tingkat global

### **Keuntungan Penyesuaian PSAK 219 terhadap IFRS 19**

#### **1. Pengukuran Liabilitas Manfaat Lebih Akurat dan Relevan**
Menggunakan asumsi yang lebih _realistis_ dan diperbarui secara _dinamis_ membantu laporan keuangan mencerminkan **kondisi pasar yang aktual**.

#### **2. Pengungkapan yang Lebih Terperinci**
Menurut lebih banyak pengungkapan, misalnya rincian **program manfaat karyawan** laporan keuangan menjadi lebih **transparan** bagi investor dan pemangku kepentingan.

#### **3. Manajemen Risiko yang Lebih Komprehensif**
Perusahaan juga harus mengenali dan **mengelola risiko** – seperti fluktuasi pasar / risiko kredit. Strategi investasi dan perencanaan keuangan bisa disesuaikan secara **lebih akurat**.

### **Pengawasan dan Regulasi: Apa yang harus diperhatikan?**

#### **1. Tanggung Jawab Hukum & Risiko Non-Kepatuhan**
Setiap entitas yang menyusun laporan keuangan wajib menerapkan PSAK 219. Ketidakpatuhan dapat memicu sanksi otoritas dan risiko litigasi.

#### **2. Pengawasan Regulator**
Direktorat Pembinaan dan Pengawasan Profesi Keuangan Kementerian Keuangan mengawasi pelaksanaan PSAK 219 untuk memastikan laporan keuangan perusahaan mencerminkan kondisi keuangan sebenarnya.

#### **3. Kewajiban Dokumentasi**
Perusahaan harus menyimpan dokumen perhitungan kewajiban imbalan kerja (asumsi aktuaria, metode, data karyawan) sebagai bukti kepatuhan dan untuk keperluan audit.

---

## #penerapan-ifric-ad

IFRS _Interpretation Committee,_ sebelumnya dikenal sebagai IFRIC (International Financial Reporting Interpretation Committee), adalah sebuah lembaga yang bertugas menginterpretasikan standar akuntansi IFRS agar penerapannya konsisten di seluruh dunia. Komite ini bekerja sama dengan _International Accounting Standards Board_ (IASB) untuk menjawab pertanyaan terkait standar akuntansi dan isu teknis terkait standar IFRS.

DSAK IAI menyimpulkan bahwa skema pensiun di Indonesia, yang mengikuti Undang‐Undang Ketenagakerjaan atau Undang‐Undang Cipta Kerja (UUK/UUCK), memiliki pola fakta mirip dengan yang dibahas IFRIC AD sehingga dianggap relevan satu sama lain.

### **Perbandingan IFRIC AD vs UUCK**

| **IFRIC AD** | **UUCK** |
|:-------------|:---------|
| Karyawan berhak atas manfaat pensiun hanya jika mencapai **usia pensiun 62 tahun** dan masih bekerja di perusahaan pada saat itu | Karyawan berhak atas manfaat pensiun hanya jika mencapai **usia pensiun 56 tahun** dan masih bekerja di perusahaan pada saat itu |
| Manfaat pensiun dihitung berdasarkan **1 bulan gaji terakhir untuk setiap tahun masa kerja** sebelum usia pensiun, namun dibatasi maksimal **16 tahun** masa kerja | Manfaat pensiun didapat dari **penjumlahan dua komponen** — misalnya _uang pesangon dan uang penghargaan masa kerja_, masing-masing memiliki batas tahun kerja berbeda |
| Manfaat pensiun dihitung hanya dengan menggunakan jumlah tahun kerja berturut-turut tepat sebelum usia pensiun | |

### **Perbandingan Sebelum dan Sesudah Penerapan IFRIC AD**

**IFRIC AD** telah menjelaskan lebih rinci tentang paragraf **70, 71, 72, dan 74** dari PSAK 219 yang berbunyi:

> **Paragraf 70:** _Metode Projected Unit Credit (sering kali disebut sebagai metode imbalan yang diakru yang diperhitungkan secara pro rata sesuai jasa atau sebagai metode imbalan dibagi tahun jasa) menganggap setiap periode jasa akan menghasilkan satu unit tambahan imbalan dan mengukur setiap unit secara terpisah untuk menghasilkan kewajiban final._

> **Paragraf 71:** _Entitas mendiskontokan semua kewajiban imbalan pasca kerja, walaupun sebagian kewajiban jatuh tempo dalam jangka waktu 12 (dua belas bulan) bulan setelah periode pelaporan._

> **Paragraf 72:** _Dalam menentukan nilai kini kewajiban imbalan pasti dan biaya jasa kini yang terkait dan biaya jasa lalu (jika dapat diterapkan) entitas mengalokasikan imbalan sepanjang periode jasa dengan menggunakan formula imbalan yang dimiliki program. Namun, jika jasa pekerja di tahun-tahun akhir meningkat secara material dibandingkan dengan tahun-tahun sebelumnya, maka entitas mengalokasikan imbalan tersebut dengan dasar metode garis lurus, sejak: (a) saat jasa pekerja pertama kali menghasilkan imbalan dalam program (baik imbalan tersebut bergantung pada jasa selanjutnya atau tidak); sampai dengan (b) saat jasa pekerja selanjutnya tidak menghasilkan imbalan yang material dalam program, selain dari kenaikan gaji berikutnya._

> **Paragraf 74:** _Dalam program imbalan pasti jasa pekerja akan menimbulkan kewajiban, walaupun imbalan itu bergantung pada status bekerjanya di masa depan (dengan kata lain tidak vested). Jasa pekerja sebelum tanggal vesting menimbulkan kewajiban konstruktif karena, pada setiap akhir periode pelaporan yang berurutan, jumlah jasa di masa depan yang harus diberikan pekerja sebelum pekerja berhak atas imbalan tersebut menjadi berkurang. Dalam mengukur kewajiban imbalan pasti, entitas memperhitungkan kemungkinan bahwa beberapa pekerja tidak akan memenuhi ketentuan vesting. Sama halnya, walaupun imbalan pasca kerja tertentu, sebagai contoh jaminan kesehatan pasca kerja, terutang hanya jika peristiwa tertentu terjadi pada saat pekerja tidak lagi bekerja, namun kewajiban muncul pada saat pekerja memberikan jasa yang menimbulkan hak atas imbalan jika peristiwa tertentu tersebut terjadi. Kemungkinan bahwa peristiwa tertentu akan terjadi berpengaruh terhadap pengukuran kewajiban, namun tidak menentukan apakah kewajiban tersebut ada._

| **Sebelum IFRIC AD** | **Sesudah IFRIC AD** |
|:---------------------|:---------------------|
| Kewajiban pasca kerja dianggap timbul **sejak karyawan mulai bekerja** | Kewajiban pasca kerja baru timbul ketika sisa masa kerja karyawan, yaitu 24 tahun sebelum usia pensiunnya |
| _Contoh_: Jika perusahaan A menetapkan usia pensiun 56 tahun dan Rudi mulai bekerja pada usia 25, perusahaan langsung mulai menghitung dan mencatat kewajiban pensiun untuk Rudi **sejak ia berusia 25 tahun** | _Contoh_: Jika perusahaan A menetapkan usia pensiun 56 tahun dan Rudi mulai bekerja di perusahaan A ketika umur 25 tahun, perusahaan baru mulai menghitung dan mencatat kewajiban pensiun untuk Rudi **ketika usianya 32 tahun** (56 – 24) |

### **📊 Formula Impact Analisis**

**PVFB = PVDBO + Future Service Cost**

**Dalam bahasa sederhana:**
- **PVFB (Present Value Future Benefit):** Total nilai kini semua manfaat yang akan dibayar di masa depan
- **PVDBO (Present Value Defined Benefit Obligation):** Nilai kini kewajiban berdasarkan masa kerja yang sudah dilalui
- **Future Service Cost:** Proyeksi biaya untuk masa kerja yang tersisa

**Impact IFRIC Implementation:**
- **Before IFRIC:** Pengakuan dimulai sejak hari pertama kerja
- **After IFRIC:** Pengakuan dimulai 24 tahun sebelum usia pensiun
- **Result:** Reduced liability recognition untuk karyawan muda, delayed cost recognition

---

## #kebutuhan-valuasi-aktuaria

Ruang lingkup PSAK 219 yang memerlukan perhitungan valuasi aktuaria adalah:

### **Imbalan Pasca Kerja**

Yang dihitung antara lain:
- Pesangon untuk karyawan pensiun
- Pesangon untuk karyawan meninggal dunia
- Pesangon untuk karyawan sakit berkepanjangan / cacat
- Pesangon untuk karyawan mengundurkan diri

### **Imbalan Jangka Panjang Lainnya**

Atau Other Long-Term Employee Benefits (OLTEB), bila perusahaan menjanjikan manfaat penghargaan di luar UU maka perusahaan perlu melakukan perhitungan valuasi aktuaria.

**Contoh Imbalan Jangka Panjang Lainnya adalah:**
- **Manfaat Cuti Besar (CBS)** / Penghargaan Masa Kerja yang dapat diuangkan
- **Manfaat Penghargaan Emas**

> **📋 Definisi:** Valuasi aktuaria adalah proses perhitungan dan analisis yang menentukan besarnya kewajiban imbalan pasca kerja perusahaan, dengan mempertimbangkan faktor usia, masa kerja, gaji, dan harapan hidup karyawan, untuk memastikan perusahaan memiliki dana yang cukup guna memenuhi kewajiban tersebut saat jatuh tempo.

### **Proses Valuasi Aktuaria**

untuk imbalan kerja yang dilakukan secara menyeluruh, antara lain:

#### **1. Pengumpulan Data Karyawan dan Perusahaan**

**Data Karyawan** – Ini termasuk informasi dasar, seperti usia karyawan, gaji, dan masa kerja.

| Data Requirement | Keterangan | Status |
|:------------------|:-----------|:-------|
| Nomor Induk Pegawai (NIP) / NIK | Identitas unik karyawan | Wajib |
| Nama Karyawan | Identitas karyawan | Opsional |
| Tanggal Lahir Karyawan | Untuk perhitungan usia | Wajib |
| Jenis Kelamin | Untuk tabel mortalitas | Wajib |
| Status Karyawan | Tetap / Kontrak | Wajib |
| Tanggal Masuk Kerja | Untuk perhitungan masa kerja | Wajib |
| Gaji / Upah | Dasar perhitungan manfaat | Wajib |
| Tanggal Henti Kerja | Untuk karyawan yang keluar | Kondisional |
| Gaji / Upah Saat Henti Kerja | Untuk perhitungan benefit | Kondisional |
| Departemen / Unit Kerja | Untuk segmentasi analisis | Opsional |
| Besarnya Pembayaran | Realisasi pembayaran benefit | Kondisional |
| Usia Pensiun (Tahun) | Usia pensiun normal | Wajib |
| Jenis Karyawan | Klasifikasi karyawan | Wajib |
| DPLK (Iuran dari Perusahaan) - Iuran/Bulan | Program dana pensiun | Opsional |
| DPLK (Iuran dari Perusahaan) - Saldo Akhir | Akumulasi dana pensiun | Opsional |
| DPLK (Iuran dari Karyawan) - Iuran/Bulan | Kontribusi karyawan | Opsional |
| DPLK (Iuran dari Karyawan) - Saldo Akhir | Akumulasi kontribusi | Opsional |
| Tunjangan Tetap | Komponen gaji tetap | Opsional |
| Tunjangan Tidak Tetap | Komponen gaji variabel | Opsional |

**Informasi Perusahaan** adalah informasi dasar mengenai perusahaan, seperti berikut:

| Informasi Requirement | Keterangan | Status |
|:----------------------|:-----------|:-------|
| Nama dan Alamat Perusahaan | Identitas perusahaan | Wajib |
| Jenis Industri Perusahaan | Untuk benchmarking | Wajib |
| Manfaat Pasca Kerja yang diberikan | Scope perhitungan | Wajib |
| Program Dana Pensiun (selain BPJS) | Skema pensiun tambahan | Kondisional |
| Standar Akuntansi yang dipakai | PSAK 219 / SAK EP | Wajib |
| Periode / Valuasi yang akan dihitung | Tanggal valuasi | Wajib |
| Jumlah Karyawan Tetap yang akan dihitung | Scope perhitungan | Wajib |
| Jumlah Karyawan Kontrak yang akan dihitung | Scope perhitungan | Wajib |
| Nama KAP (Auditor) yang dipakai | Koordinasi audit | Opsional |
| Riwayat perhitungan oleh Konsultan Aktuaria | Data historis | Opsional |
| PIC penanggung jawab riwayat perhitungan | Koordinasi internal | Wajib |
| Penanggung Pajak Manfaat pensiun | Perusahaan / Karyawan | Wajib |
| Besaran Kenaikan Gaji Terakhir | Data historis | Wajib |
| Rata-rata Kenaikan Gaji 5 tahun terakhir | Asumsi kenaikan gaji | Wajib |
| Total realisasi pembayaran Pesangon | Data historis | Wajib |
| BOD yang dihitung | Scope perhitungan | Kondisional |
| PIC untuk pengiriman laporan aktuaria | Koordinasi pelaporan | Wajib |

> **Catatan:** Perlu adanya Tambahan Informasi Perusahaan apabila mengikuti Program Dana Pensiun.

#### **2. Penggunaan Model Matematika**

Setelah data dikumpulkan, berikutnya adalah menggunakan model matematika untuk mengestimasi beban dan liabilitas imbalan kerja secara relevan.

**Diagram alur Perhitungan Aktuaria sebagai berikut:**

1. **Input Utama**
   - Asumsi **Demografis**
   - Asumsi **Keuangan**
2. **Proyeksi Kewajiban** Menggunakan Projected Unit Credit (PUC) — tujuannya menghitung kewajiban imbalan pasca kerja berdasarkan penilaian proyeksi masa depan.
3. **Hasil Akhir** **Nilai kini kewajiban** (_present value_)

Kuncinya, **asumsi yang digunakan sangat penting** karena akan memengaruhi hasil proyeksi. Misalnya, jika diasumsikan bahwa gaji akan naik lebih cepat, maka kewajiban perusahaan akan lebih besar. Asumsi juga harus **realistis** untuk menghasilkan proyeksi yang akurat.

#### **3. Analisis Risiko**

Tujuannya untuk mengetahui potensi lonjakan kewajiban, sehingga perusahaan dapat menyiapkan strategi cadangan, investasi, atau kebijakan manajemen risiko lain demi menjaga stabilitas laporan keuangan. Berikut langkah-langkahnya:

##### **Identifikasi Sumber Risiko**

Terdapat dua jenis risiko yang terlibat, yaitu risiko pasar dan risiko demografi, misalnya:

| Jenis Risiko | Sumber Risiko | Dampak Potensial |
|:-------------|:--------------|:-----------------|
| **Risiko Pasar** | Perubahan tingkat diskonto akibat fluktuasi suku bunga pasar | Volatilitas nilai kewajiban |
| **Risiko Ekonomi** | Kenaikan gaji yang lebih tinggi atau lebih rendah dari perkiraan | Perubahan beban future benefit |
| **Risiko Demografi** | Tingkat keluar karyawan (withdrawal rate) yang berbeda dari asumsi | Perubahan populasi yang diasuransikan |
| **Risiko Longevity** | Umur hidup (mortalitas) yang berbeda dari tabel standar | Durasi pembayaran benefit |

##### **Metode Analisis**

Cara yang dilakukan untuk analisis risiko, antara lain:

**Uji sensitivitas (Analisis Sensitivitas):** Mengukur seberapa besar perubahan kewajiban jika asumsi kunci berubah (misalnya ±1% tingkat bunga).

**Stress testing:** Menerapkan skenario ekstrem (misalnya resesi, inflasi tinggi) untuk memproyeksikan dampaknya pada kewajiban.

**Maturity Analysis (Analisis jatuh tempo):** Menunjukkan kapan kewajiban akan jatuh tempo atau dibayarkan—apakah lebih banyak dalam 5 tahun ke depan, 10 tahun ke depan, atau lebih dari 15 tahun?

##### **Apa itu Uji Sensitivitas dan Maturity Analysis?**

Agar mudah dipahami, begini perbedaan antara keduanya:

| **Uji Sensitivitas** | **Maturity Analysis** |
|:---------------------|:----------------------|
| **"Apa dampaknya jika tingkat diskonto turun 1%?"** | **"Berapa tahun lagi mayoritas manfaat pensiun akan dibayarkan?"** |
| Mengukur dampak perubahan perubahan asumsi (seperti diskonto, gaji, mortalita) | Memberikan gambaran tentang profil distribusi jatuh tempo kewajiban (kapan manfaat akan dibayar) |
| Hasil berupa perubahan nilai kewajiban (misalnya, naik / turun 5–10%) | Informasi seperti rata-rata durasi kewajiban dan waktu pembayaran manfaat terbanyak |

---

## #tantangan-dalam-audit-keuangan

Tantangan utama yang sering dihadapi pengguna dan perusahaan adalah validasi asumsi aktuaria, keakuratan data yang digunakan, dan kepatuhan terhadap standar akuntansi yang berlaku, seperti PSAK 219. Perusahaan harus mengambil pendekatan yang sistematis dan kolaboratif.

#### **Framework Mengatasi Tantangan Audit**

| Komponen | Deskripsi | Target Outcome |
|:---------|:----------|:---------------|
| **Transparansi Perhitungan** | Semua asumsi dan metode harus **jelas dan terdokumentasi**<br/>Contoh: Tingkat Diskonto, Kenaikan Gaji, metode PUC | Audit trail yang lengkap |
| **Kolaborasi Efektif** | Aktuaris, manajemen, dan auditor perlu bekerja sama agar hasil audit:<br/>- Akurat<br/>- Minim konflik interpretasi | Proses audit yang smooth |
| **Dokumentasi Memadai** | - Data karyawan harus lengkap dan **up-to-date**<br/>- Simulasi sensitivitas dan penyesuaian data perlu **didokumentasikan** | Compliance yang terjaga |
| **Validasi Auditor** | Auditor akan:<br/>- Verifikasi metode dan hasil<br/>- Melakukan **pengujian ulang** untuk memastikan laporan valid dan akurat | Validasi independen |
| **Mitigasi Risiko** | - Proses harus rutin **direvisi** agar sesuai standar<br/>- **Otomatisasi** & **pelatihan** bantu kurangi risiko kesalahan manual | Risk reduction |

Kesinambungan kelima poin tersebut akan menunjang kelancaran proses **audit keuangan secara baik dan benar**, serta menjaga **kepatuhan pada regulasi**.

---

## 📝 **Ringkasan Bab 1**

### **Key Takeaways**

- **Konsep Dasar:** Imbalan pasca kerja adalah kewajiban perusahaan yang harus dihitung secara aktuaria menggunakan berbagai asumsi dan proyeksi masa depan, seperti yang diilustrasikan melalui analogi "Janji Rudi"
- **Standar Akuntansi:** Transisi dari PSAK 24 ke PSAK 219 memberikan panduan yang lebih komprehensif dan selaras dengan standar internasional IFRS 19
- **Regulatory Framework:** Multiple regulations (UUK, UUCK, berbagai PP) menjadi dasar hukum yang harus dipatuhi perusahaan dalam implementasi
- **IFRIC Implementation:** Penerapan IFRIC AD mengubah timing pengakuan kewajiban dari sejak mulai bekerja menjadi 24 tahun sebelum usia pensiun
- **Valuasi Requirement:** Perusahaan wajib melakukan valuasi aktuaria untuk imbalan pasca kerja dan imbalan jangka panjang lainnya

### **Regulatory Timeline dan Compliance**

| Periode | Regulasi | Key Changes |
|:--------|:---------|:------------|
| **2003** | UU No. 13/2003 (UUK) | Framework dasar imbalan kerja |
| **2014** | PSAK 24 Update | Alignment dengan praktik global |
| **2020** | UU No. 11/2020 (UUCK) | Perubahan formula benefit calculation |
| **2025** | PSAK 219 Effective | Nomenklatur baru dengan enhanced guidance |

---

## 🔄 **Langkah Selanjutnya**

### **Persiapan Bab 2**

Bab selanjutnya akan membahas aspek teknis perhitungan aktuaria secara detail, termasuk:

- **Metodologi Projected Unit Credit:** Mathematical framework dan implementation
- **Asumsi Aktuaria Detail:** Economic dan demographic assumptions setting
- **Contoh Perhitungan Step-by-step:** Practical calculation examples
- **Integrasi dengan Program Dana Pensiun:** DPLK dan pension fund considerations

---

**Navigasi:** [📋 Daftar Isi](README.md) | [Bab 2: Teknis Perhitungan Aktuaria](02_teknis_perhitungan.md) ➡️

---

> **💡 Catatan:** Untuk pertanyaan lebih lanjut tentang materi bab ini, silakan merujuk ke [FAQ Legal](05b_faq_legal.md) di Bab 5, atau konsultasikan dengan konsultan aktuaria yang berpengalaman untuk guidance yang lebih spesifik sesuai kondisi perusahaan Anda.