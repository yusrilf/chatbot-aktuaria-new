---
keywords:
  - projected unit credit
  - PUC
  - PVDBO
  - CSC
  - PVFB
  - metode aktuaria
  - current service cost
  - interest cost
  - sains aktuaria
  - matematis aktuaria
  - perhitungan aktuaria
  - CBS
  - Cuti Besar
difficulty:
  - advanced
estimated_reading: 35 minutes
target_audience:
  - finance
  - actuaries
  - technical_staff
document_type: technical_guide
last_updated: 2025-07-28
version: "2025.1"
related_regulations:
  - PSAK_219
  - IFRS_19
domain: aktuaria
scope: sains_aktuaria_metode_perhitungan
exclude_domains:
  - asuransi_umum
  - perbankan_komersial
  - insurance_reserves
context_boundary: strict
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
  - PUC method
  - PVDBO calculation
  - current service cost
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
fallback_response: Informasi tidak tersedia dalam konteks sains aktuaria dan imbalan kerja. Silakan ajukan pertanyaan terkait PSAK 219, valuasi aktuaria, atau imbalan kerja karyawan.
---
---
# Metode Projected Unit Credit (PUC)

**📖 Bagian dari:** [PANDUAN LENGKAP IMBALAN KERJA PSAK 219](README.md)

---
## 🎯 **Tujuan Pembelajaran**

Setelah membaca bab ini, pembaca akan mampu:

- [ ]  Menguasai metode Projected Unit Credit (PUC) untuk perhitungan kewajiban imbalan kerja
- [ ]  Memahami formula manfaat karyawan berdasarkan UUK dan UUCK
- [ ]  Menghitung PVDBO, CSC, dan komponen aktuaria lainnya
- [ ]  Melakukan perhitungan step-by-step untuk kasus praktis

---
## ⚡ **Quick Reference**

### 🔤 **Key Formulas:**
- **PVDBO:** Present Value of Defined Benefit Obligation = Nilai kini kewajiban berdasarkan masa kerja lalu
- **CSC:** Current Service Cost = Biaya jasa periode berjalan  
- **PUC Method:** Projected Unit Credit = Metode standar untuk perhitungan aktuaria PSAK 219
- **PVFB:** Present Value of Future Benefits = Total nilai kini semua manfaat masa depan

### 📊 **Essential Calculations:**
- **Unit per Tahun:** PVFB ÷ Total masa kerja hingga pensiun
- **PVDBO:** PVFB × (Masa kerja lalu ÷ Total masa kerja)  
- **CSC:** PVFB ÷ Total masa kerja hingga pensiun
- **Interest Cost:** PVDBO awal tahun × Tingkat diskonto

### 🏦 **Dana Pensiun Types:**
- **DPLK:** Dana Pensiun Lembaga Keuangan (bank/asuransi)
- **DPPK:** Dana Pensiun Pemberi Kerja (internal perusahaan)
- **Asuransi:** Dana Pensiun Lembaga Asuransi (polis khusus)

### ⚖️ **Key Regulations:**
- **UUK Formula:** (2×Severance + 1×Service) × 115% untuk pensiun normal
- **UUCK Formula:** (1,75×Severance + 1×Service) + Compensation untuk pensiun normal
- **PKWT Compensation:** Pro-rata 1 bulan upah per 12 bulan kerja

---

Setelah memahami program Dana Pensiun, pertanyaan berikutnya adalah: **"Kapan dan bagaimana perusahaan harus mengakui beban atau kewajiban tersebut dalam laporan keuangannya?"** Di sinilah metode aktuaria dibutuhkan.

## **Formula & Pajak Manfaat Karyawan**

### **Manfaat Karyawan Tetap / PKWTT**

Berdasarkan UUK No. 13 Tahun 2003 dan UUCK, rumusan manfaat Pesangon dan **uang penghargaan masa kerja** adalah sebagai berikut:

#### **Benefit Calculation Table**

|Service (YoS)** / Masa Kerja|Severance Pay / Uang Pesangon*|Service Pay / Uang Penghargaan Masa Kerja*|
|:--|:--|:--|
|YoS < 1|1|0|
|1 ≤ YoS < 2|2|0|
|2 ≤ YoS < 3|3|0|
|3 ≤ YoS < 4|4|2|
|4 ≤ YoS < 5|5|2|
|5 ≤ YoS < 6|6|2|
|6 ≤ YoS < 7|7|3|
|7 ≤ YoS < 8|8|4|
|8 ≤ YoS < 9|9|4|
|9 ≤ YoS < 10|9|5|
|10 ≤ YoS < 12|9|6|
|12 ≤ YoS < 15|9|7|
|15 ≤ YoS < 18|9|8|
|18 ≤ YoS < 21|9|9|
|21 ≤ YoS < 24|9|9|
|YoS ≥ 24|9|10|

> *Severance Pay dan Service Pay dihitung dalam kelipatan gaji (multiple of wages).  
> **YoS = Years of Service / Masa Kerja.

#### **Formula Manfaat UUK No. 13 Tahun 2003**

|**Benefit Value**|**Rumus Besar Manfaat / Benefit Formula**|**Jenis Manfaat**|
|:--|:--|:--|
|Normal Retirement|(2 × Severance Pay + 1 × Service Pay) × 115%|Pensiun Normal|
|Death Benefit|(2 × Severance Pay + 1 × Service Pay) × 115%|Pekerja Meninggal Dunia|
|Disability / Illness|(2 × Severance Pay + 2 × Service Pay) × 115%|Sakit Berkepanjangan|
|Voluntary Resign|(1 × Severance Pay + 1 × Service Pay) × 115% + Separation Pay|Pekerja Mengundurkan Diri|

#### **Formula Manfaat UU Cipta Kerja (UUCK)**

|**Benefit Value**|**Rumus Besar Manfaat / Benefit Formula**|**Jenis Manfaat**|
|:--|:--|:--|
|Normal Retirement|(1,75 × Severance Pay + 1 × Service Pay) + Compensation Pay|Pensiun Normal|
|Death Benefit|(2 × Severance Pay + 1 × Service Pay) + Compensation Pay|Pekerja Meninggal Dunia|
|Disability / Illness|(2 × Severance Pay + 2 × Service Pay) + Compensation Pay|Sakit Berkepanjangan|
|Voluntary Resign|Compensation Pay|Pekerja Mengundurkan Diri|

### **Manfaat Karyawan Kontrak / PKWT**

Berdasarkan **PP No. 35 Tahun 2015**, besaran uang kompensasi bagi karyawan dengan Perjanjian Kerja Waktu Tertentu (PKWT) dihitung berdasarkan masa kerja, dengan ketentuan:

#### **PKWT Compensation Table**

|Masa Kerja PKWT|Kompensasi|
|:--|:--|
|PKWT 12 bulan (terus menerus)|Kompensasi 1 bulan upah|
|PKWT < 12 bulan|Pro-rata, sesuai rumus: (masa kerja / 12) × 1 bulan upah|
|PKWT > 12 bulan|Pro-rata, sesuai rumus: (masa kerja / 12) × 1 bulan upah|

### **Contoh Perhitungan Aktuaria Kompensasi Karyawan Kontrak**

**Case:** Perusahaan menjanjikan pembayaran kompensasi kepada karyawan A pada saat berakhirnya kontrak, dengan gaji saat ini sebesar **Rp12.000.000,-**

**Calculation Steps:**

1. **Unit per Period:** Rp12.000.000 ÷ 12 = Rp1.000.000,-
2. **Monthly Accrual:** Based on PUC method (assumptions ignored for simplicity)

**Results:**

|Bulan|Beban Bulanan|Kewajiban pada Akhir Periode|
|:--|:--|:--|
|1|1.000.000|1.000.000|
|2|1.000.000|2.000.000|
|3|1.000.000|3.000.000|
|...|...|...|
|12|1.000.000|12.000.000|

---

Setelah diketahui berapa besar Manfaat pensiun yang seharusnya diterima karyawan sesuai ketentuan perundang-undangan, pertanyaan berikutnya adalah:  
**"Kapan dan bagaimana perusahaan harus mengakui beban atau kewajiban tersebut dalam laporan keuangannya?"**

Di sinilah metode aktuaria dibutuhkan. Metode yang umum digunakan dan diakui dalam standar akuntansi PSAK 219 adalah Projected Unit Credit (PUC).

## #metode-projected-unit-credit

Metode PUC adalah cara penting untuk menghitung kewajiban imbalan kerja, terutama dalam aturan PSAK 219 yang sejalan dengan interpretasi IFRIC. Metode ini mengharuskan perusahaan untuk mengalokasikan imbalan kerja ke dua periode:

- **Periode berjalan** untuk menentukan Biaya Jasa Kini (CSC)
- **Periode berjalan dan periode-periode lalu** untuk menentukan **Nilai Kini Kewajiban** (PVDBO)

### **Employee Career Perspective**

#### **Total Masa Kerja Karyawan**

Terdiri dari:

**1. Masa Kerja Telah Dijalani (Past Service)**

- **PVDBO (Kewajiban Saat Ini):** Mengukur **nilai kini** atas imbalan yang telah diakumulasi hingga **saat ini**
- **CSC (Biaya Tahun Ini):** Merupakan bagian dari PVDBO yang dibebankan pada Laporan Laba Rugi tahun berjalan untuk mencatat Biaya Jasa Kini (current service cost)

**2. Future Service Cost (Sisa Masa Kerja)**

- **Proyeksi masa kerja:** dari usia saat valuasi hingga Usia Pensiun normal
- **Manfaat yang diproyeksikan** untuk diberikan di masa mendatang, yang akan menjadi bagian dari biaya tahun-tahun berikutnya

#### **Total Future Benefits Relationship**

Jadi adanya nilai PVFB / Present Value of Future Benefits selama Total Masa Kerja Karyawan, dengan analogi:

**PVFB = PVDBO + Future Service Cost**

**Dalam bahasa sederhana:**

- **PVFB:** Present Value of Future Benefit (nilai kini dari seluruh Manfaat pensiun yang akan dibayarkan)
- **PVDBO:** Nilai Kini Kewajiban hingga saat ini
- **Future Service Cost:** Proyeksi beban jasa di masa mendatang

## **Terminologi Aktuaria Imbalan Kerja**

### **Komponen Utama**

#### **PVDBO**

Kewajiban imbalan kerja yang sudah menjadi tanggungan perusahaan hari ini untuk Manfaat pensiun di masa depan yang telah diperoleh karyawan dari masa kerja yang telah dilalui.

#### **Biaya Jasa Kini (CSC)**

Atau disebut Current Service Cost, adalah bagian PVDBO yang berasal dari tahun berjalan untuk pensiun yang akan datang, dihitung untuk karyawan aktif. **_Dicatat di Laporan Laba Rugi._**

#### **Biaya Bunga**

Atau disebut Interest Cost, adalah kenaikan kewajiban karena waktu berjalan yang dihitung dari nilai kewajiban yang didiskontokan, dihitung dengan rumus tingkat diskonto × PVDBO.

#### **Biaya Jasa Lalu (BJS)**

Atau Past Service Cost, ialah tambahan kewajiban aktuaria yang berasal dari manfaat masa lalu yang belum dihitung sebelumnya, misalnya akibat perubahan manfaat retroaktif.

#### **Vested Benefit**

Bagian dari Manfaat pensiun yang sudah menjadi hak karyawan dan tidak dapat dibatalkan. Vested berlaku meskipun karyawan mengundurkan diri sebelum pensiun minimal tertentu.

### **Financial Statement Recognition**

**Beban diakui di Laporan Laba Rugi & OCI**

Laporan akuntansi yang mencatat:

- **Laporan Laba Rugi:** Beban imbalan kerja (CSC, Biaya Bunga), Biaya Jasa Lalu (jika ada)
- **OCI:** Keuntungan/Kerugian Aktuaria

## **Rumus Metode Projected Unit Credit**

Metode PUC adalah metode _accrued benefit cost_ dengan menggunakan asumsi kenaikan gaji. Misalkan rumus Manfaat pensiun dari suatu program imbalan pensiun adalah faktor penghargaan dikalikan dengan masa kerja dan upah, maka besar Manfaat pensiun pada Usia Pensiun adalah sebagai berikut:

### **Basic Benefit Formula**

```
B_r = F_(r-e) × (1 + s)^(r - x) × S_x
```

**Variable Definitions:**

- x = usia saat valuasi
- e = usia mulai masa kerja
- r = Usia Pensiun
- (r - e) = masa kerja peserta dihitung sejak mulai bekerja di usia e hingga Usia Pensiun normal r maksimal 24 tahun
- B_r = Manfaat pensiun pada Usia Pensiun normal r
- F = faktor penghargaan (berdasarkan Undang-Undang atau Peraturan Perusahaan)
- (1 + s)^(r - x) S_x = upah peserta sebelum pensiun dikaitkan dengan kenaikan tingkat gaji

### **1. Present Value Future Benefit (PVFB)**

adalah Nilai Kini atas Manfaat pensiun dari suatu program imbalan pensiun dengan menggunakan Tingkat Diskonto dan memperhitungkan peluang hidup karyawan tersebut dapat mencapai Usia Pensiun r tahun.

```
PVFB_x = B_r × v^(r - x) × p_x^(r-x)
```

**Variable Definitions:**

- x = usia saat valuasi
- r = Usia Pensiun
- B_r = Manfaat pensiun pada Usia Pensiun normal r tahun
- v^(r-x) = faktor diskonto selama (r - x) tahun
- p_x^(r-x) = peluang kehidupan total di usia x tahun hingga Usia Pensiun r tahun

**Dalam bahasa sederhana:** PVFB adalah total nilai kini dari semua manfaat yang akan dibayar di masa depan, dengan mempertimbangkan faktor diskonto dan probabilitas karyawan mencapai usia pensiun.

### **2. Present Value of Defined Benefit Obligation (PVDBO)**

adalah nilai kini pembayaran masa depan yang diperlukan untuk menyelesaikan kewajiban atas jasa pekerja periode berjalan dan periode-periode lalu.

```
PVDBO_x = PVFB_x × (x - e) / (r - e)
```

**Variable Definitions:**

- x = usia saat valuasi
- e = usia mulai masa kerja
- r = Usia Pensiun
- (r - e) = total masa kerja hingga pensiun
- (x - e) = masa kerja yang telah dilalui

**Dalam bahasa sederhana:** PVDBO adalah bagian dari PVFB yang sudah "earned" berdasarkan masa kerja yang telah dilalui dibandingkan dengan total masa kerja hingga pensiun.

### **3. Current Service Cost (CSC)**

adalah kenaikan nilai kini kewajiban imbalan pasti atas jasa pekerja dalam periode berjalan.

```
CSC_x = PVFB_x / (r - e)
```

**Variable Definitions:**

- x = usia saat valuasi
- e = usia mulai masa kerja
- r = Usia Pensiun
- (r - e) = total masa kerja hingga pensiun

**Dalam bahasa sederhana:** CSC adalah porsi PVFB yang dialokasikan untuk tahun berjalan, representing the "earning" of benefits for current year of service.

## **Penerapan Metode Projected Unit Credit**

### **Practical Example**

**Given Data:**

- **Manfaat pensiun:** 1,75 × masa kerja × gaji saat pensiun
- **Usia mulai masa kerja:** 35 tahun
- **Usia saat valuasi (x):** 40 tahun
- **Usia Pensiun (r):** 55 tahun
- **Gaji saat valuasi:** Rp10.000.000,-
- **Tingkat Diskonto (v):** 10%
- **Tingkat Kenaikan Gaji (s):** 5%
- **Asumsi lainnya diabaikan**

### **Step-by-Step Calculation**

#### **Step 1: Hitung Manfaat di Masa Pensiun (Future Benefit)**

**Formula:**

```
A = 1,75 × Masa Kerja × Gaji pada saat pensiun × (1 + Asumsi Kenaikan Gaji)^Sisa Masa Kerja
```

**Calculation:**

```
A = 1,75 × 20 × 10.000.000 × (1,05)^15
A = 1,75 × 20 × 10.000.000 × 2,078928
A ≈ Rp 728.625.000
```

#### **Step 2: Hitung Nilai Sekarang dari Manfaat (PVFB)**

**Formula:**

```
PVFB = A / (1 + Tingkat Diskonto)^Sisa Masa Kerja
```

**Calculation:**

```
PVFB = 728.625.000 / (1,10)^15
PVFB = 728.625.000 / 4,17725 
PVFB ≈ Rp 174.370.000
```

#### **Step 3: Hitung Nilai Kini Kewajiban (PVDBO)**

**Formula:**

```
PVDBO = PVFB × (Masa Kerja saat ini / Total Masa Kerja hingga Pensiun)
```

**Calculation:**

```
PVDBO = 174.370.000 × (5 / 20)
PVDBO = 174.370.000 × 0,25 
PVDBO = Rp 43.592.500
```

#### **Step 4: Hitung Biaya Jasa Kini (CSC)**

**Formula:**

```
CSC = PVFB × (1 / Total Masa Kerja hingga Pensiun)
```

**Calculation:**

```
CSC = 174.370.000 × (1 / 20)
CSC = 174.370.000 × 0,05 
CSC = Rp 8.718.500
```

### **Implementation Notes**

Perhitungan metode PUC di atas berlaku per individu, tetapi dalam praktiknya dilakukan untuk seluruh karyawan satu per satu, lalu **diakumulasi agar menghasilkan total kewajiban dan beban perusahaan secara keseluruhan.**

Untuk perusahaan dengan jumlah karyawan yang besar, aktuaris bisa menggunakan software aktuaria berbasis sistem atau pemodelan berbasis spreadsheet untuk solusi perhitungan yang efisien.

## **Contoh Perhitungan Imbalan Jangka Panjang Lainnya**

### **Cuti Besar (CBS)**

**Case:** Perusahaan menjanjikan pembayaran uang cuti besar kepada setiap karyawannya bagi karyawan yang telah mencapai masa kerja 10 tahun sebesar Rp10.000.000,-

**Calculation Steps:**

1. **Unit per Period:** Rp10.000.000 ÷ 10 = Rp1.000.000,-
2. **Annual Accrual:** Based on PUC method

**Results:**

|Tahun|Beban Tahun Berjalan|Kewajiban pada Akhir Periode|
|:--|:--|:--|
|1|1.000.000|1.000.000|
|2|1.000.000|2.000.000|
|3|1.000.000|3.000.000|
|...|...|...|
|10|1.000.000|10.000.000|

## **Hubungan Pajak Manfaat Pensiun**

### **Tax Rate Structure**

|Lapisan Tarif|Penghasilan Kena Pajak|Tarif Pajak|
|:--|:--|:--|
|I|≤ 50.000.000|0%|
|II|50.000.000 - 100.000.000|5%|
|III|100.000.000 - 500.000.000|15%|
|IV|> 500.000.000|25%|

### **Tax Treatment Options**

**Option 1: Company Bears Tax (Gross-up)**

- **Total Kewajiban = Manfaat + PPh**
- Company pays both benefit dan tax portion

**Option 2: Employee Bears Tax (Net)**

- **Company pays benefit amount only**
- Tax deducted from employee receipt

### **Contoh Penerapan Pajak Manfaat Pensiun**

**Case:** Manfaat pensiun Rp200.000.000 dengan masa kerja 20 tahun, pajak ditanggung perusahaan dengan tarif flat 5%.

**Calculation Steps:**

1. **Unit per Period:** Rp200.000.000 ÷ 20 = Rp10.000.000,-
2. **Tax per Unit:** 5% × Rp10.000.000 = Rp500.000
3. **Total Annual Obligation:** Rp10.000.000 + Rp500.000 = Rp10.500.000

**Results:**

|Tahun|Beban Tahun Berjalan|Kewajiban Sebelum Pajak|Kewajiban Setelah Pajak|
|:--|:--|:--|:--|
|1|10.000.000|10.000.000|**10.500.000**|
|2|10.000.000|20.000.000|**21.000.000**|
|3|10.000.000|30.000.000|**31.500.000**|
|...|...|...|...|
|20|10.000.000|200.000.000|**210.000.000**|

---

## 📝 **Ringkasan**

### **Key Takeaways**

- **Metode PUC:** Standar PSAK 219 untuk mengalokasikan manfaat secara proporsional
- **Three Key Components:** PVFB, PVDBO, dan CSC saling berkaitan
- **Individual Calculation:** Dilakukan per karyawan kemudian diagregasi
- **Tax Considerations:** Pajak dapat menambah kewajiban jika ditanggung perusahaan

### **Formula Summary**

| **Component**      | **Formula**                                 | **Purpose**             |
| :----------------- | :------------------------------------------ | :---------------------- |
| **PVDBO**          | PVFB × (Masa kerja lalu ÷ Total masa kerja) | Kewajiban saat ini      |
| **CSC**            | PVFB ÷ Total masa kerja hingga pensiun      | Beban tahun berjalan    |
| **Interest Cost**  | PVDBO awal tahun × Tingkat diskonto         | Biaya bunga             |

### **Next Steps**

Setelah memahami metode PUC, langkah berikutnya adalah mempelajari asumsi aktuaria yang digunakan dalam perhitungan untuk memastikan hasil yang akurat dan dapat dipertanggungjawabkan.

---

**Navigasi:** ⬅️ [Dana Pensiun](02b_dana_pensiun.md) | [📋 Daftar Isi](README.md) | [Asumsi Aktuaria](02d_asumsi_aktuaria.md) ➡️