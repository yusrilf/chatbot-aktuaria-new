---
keywords:
  - proses valuasi
  - laporan aktuaria
  - tabel aktuaria
  - PVDBO
  - CSC
  - gain loss
  - experience adjustment
  - sains aktuaria
  - matematis aktuaria
  - perhitungan aktuaria
  - BJS
  - biaya jasa lalu
difficulty:
  - advanced
estimated_reading: 30 minutes
target_audience:
  - finance
  - actuaries
  - management
  - technical_staff
document_type: technical_guide
last_updated: 2025-07-28
version: "2025.1"
domain: aktuaria
scope: sains_aktuaria_tabel_perhitungan
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
  - tabel aktuaria
  - PVDBO calculation
  - experience adjustment
  - gain loss analysis
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
# Tabel Perhitungan Aktuaria

**📖 Bagian dari:** [PANDUAN LENGKAP IMBALAN KERJA PSAK 219](README.md) dan melanjutkan dari [Data Preparation](02f_proses_valuasi.md)

---
## 🎯 **Tujuan Pembelajaran**

Setelah membaca bab ini, pembaca akan mampu:

- [ ]  Menyusun laporan aktuaria dengan 5 tabel utama
- [ ]  Menganalisis hasil perhitungan dan variance analysis

---
## ⚡**Quick Reference**

### 📊 **5 Tabel Utama Laporan Aktuaria:**

1. **Tabel 1:** Ikhtisar Data dan Asumsi
2. **Tabel 2:** Perhitungan Gain/Loss Aktuarial
3. **Tabel 3:** Other Comprehensive Income (OCI)
4. **Tabel 4:** Posisi Pendanaan & Neraca
5. **Tabel 5:** Beban Laporan Laba Rugi

### 🔍 **Advanced Analysis Tools:**

- **Analisis Sensitivitas:** Dampak perubahan asumsi kunci ±1%
- **Pengalaman Penyesuaian:** Varians aktual vs asumsi
- **Analisis Jatuh Tempo:** Proyeksi arus kas per periode

### 📈 **Key Results Framework:**

- **PVDBO:** Nilai kini kewajiban
- **CSC:** Biaya jasa kini
- **Keuntungan/Kerugian Aktuaria:** Dampak OCI


---
## **Penyajian Laporan Aktuaria**

> 🔗 **Roadmap:** Setelah memahami data dan asumsi, berikut 5 tabel utama yang membentuk laporan aktuaria lengkap: [Tabel 1](#tabel-1-data-asumsi) → [Tabel 2](#tabel-2-gain-loss) → [Tabel 3](#tabel-3-oci) → [Tabel 4](#tabel-4-balance-sheet) → [Tabel 5](#tabel-5-pnl-expense)

### #tabel-1-data-asumsi

#### **Tabel 1 - Ikhtisar Data dan Asumsi Aktuaria Perhitungan**

| **No.** | **Explanation**                                    | **31 Desember 2024** | **31 Desember 2023** | **Uraian**                                          |
| ------- | -------------------------------------------------- | -------------------- | -------------------- | --------------------------------------------------- |
| **1.**  | **Basic Data and Assumptions**                     |                      |                      | **Data dan Asumsi**                                 |
| **2.**  | Number of Employee                                 | 26,908               | 17,839               | Jumlah Karyawan (orang)                             |
| **3.**  | Monthly Wages for Permanent and Contract EEs       | Rp69,71 miliar       | Rp44,35 miliar       | Jumlah Gaji Sebulan – Karyawan Tetap dan Kontrak    |
| **4.**  | Average Wages                                      | Rp2,59 juta          | Rp2,49 juta          | Rata-rata Gaji                                      |
| **5.**  | Average Age of Permanent and Contract EEs          | 26.58                | 26.91                | Rata-rata Usia (Tahun) – Karyawan Tetap dan Kontrak |
| **6.**  | Average YoS – Permanent and Contract EEs           | 2.55                 | 3.42                 | Rata-rata Masa Kerja (Tahun)                        |
| **7.**  | Average Future Service                             | 29.00                | 28.00                | Rata-rata Sisa Masa Kerja                           |
| **8.**  | Average Expected Remaining Working Years           | 27.00                | 26.00                | Rata-rata Sisa Masa Kerja Diperkirakan (Tahun)      |
| **9.**  | Discount Rate Beginning Period                     | 7.00%                | 7.25%                | Tingkat Diskonto Awal Tahun                         |
| **10.** | Discount Rate Ending Period                        | 7.13%                | 7.00%                | Tingkat Diskonto Akhir Tahun                        |
| **11.** | Expected Return on Plan Assets                     | -                    | -                    | Tingkat Harapan Investasi atas Program              |
| **12.** | Future Salary Increases (per annum)                | 8.00%                | 8.00%                | Tingkat Kenaikan Gaji Tahunan                       |
| **13.** | **Current Service Cost**                           | **Rp24,37 miliar**   | **Rp18,22 miliar**   | **Biaya Jasa Kini – tahun berjalan**                |
| **14.** | Total Benefit Paid in Year                         | (Rp1,62 miliar)      | (Rp2,27 miliar)      | Imbalan yang dibayarkan                             |
| **15.** | Company Contribution Paid in Year                  | -                    | -                    | Iuran yang dibayar perusahaan                       |
| **16.** | **Present Value of Obligation at BoP**             | **Rp72,18 miliar**   | **Rp54,15 miliar**   | **Nilai kini kewajiban – awal tahun**               |
| **17.** | **Present Value of Obligation at EoP**             | **Rp86,69 miliar**   | **Rp72,18 miliar**   | **Nilai kini kewajiban – akhir tahun**              |
| **18.** | Past Service Cost – Non Vested at BoP              | -                    | -                    | Biaya Jasa Lalu – non-vested                        |
| **19.** | Past Service Cost – Vested at BoP                  | (Rp13,37 miliar)     | (Rp4,73 miliar)      | Biaya Jasa Lalu – vested                            |
| **20.** | Fair Value of Plan Asset Program – Start of Period | -                    | -                    | Nilai wajar aktiva program – awal tahun             |
| **21.** | Fair Value of Plan Asset Program – End of Period   | -                    | -                    | Nilai wajar aktiva program – akhir tahun            |
**Continuation - Assumptions Details:**

| **No.** | **Explanation** | **31 Desember 2024** | **31 Desember 2023** | **Uraian** |
|:--------|:----------------|:---------------------|:---------------------|:-----------|
| **22.** | Mortality Table | TMI IV | TMI III | Tabel Mortalitas |
| **23.** | Disability Rate | 10,00% dari TMI IV | 10,00% dari TMI III | Tingkat Cacat |
| **24.** | Withdrawal Rate | 20–29 = 6.00%<br>30–34 = 3.00%<br>35–39 = 1.80%<br>40–50 = 1.20%<br>51–52 = 0.60%<br>>52 = 0.00% | 20–29 = 6.00%<br>30–34 = 3.00%<br>35–39 = 1.80%<br>40–50 = 1.20%<br>51–52 = 0.60%<br>>52 = 0.00% | Tingkat Pengunduran Diri |
| **25.** | Actuarial Calculation Method | PUC (IFRIC) | PUC (IFRIC) | Metode Perhitungan Aktuaria |
| **26.** | Normal Retirement Age (Years old) | 55 | 55 | Usia Pensiun Normal (Tahun) |

#### **Analisis Hasil Kunci**

> 📊 **KEY RESULTS:** PVDBO naik 20,1% menjadi Rp86,69 miliar, CSC naik 33,7% menjadi Rp24,37 miliar - indikasi pertumbuhan bisnis yang sehat namun perlu monitoring kewajiban.

Berdasarkan data dan asumsi yang digunakan, berikut hasil Perhitungan Aktuaria mencakup karyawan tetap dan kontrak per 31 Desember 2024:

**Hasil Utama:**

- **Nilai Kini Kewajiban (PVDBO):** Rp86,69 miliar
- **Biaya Jasa Kini (CSC):** Rp24,36 miliar
- **Biaya Jasa Lalu (BJS):** (Rp13,34 miliar)

#### **❓ Frequently Asked Questions - Tabel 1**

**Q: Mengapa PVDBO PT ABC naik 20,1%?** A: Kombinasi 3 faktor: ✅ Pertumbuhan karyawan 50,8% ✅ Kenaikan gaji 57,2% ✅ Perubahan asumsi mortalitas TMI III→IV

**Q: Apakah CSC naik 33,7% normal?** A: Ya, sesuai dengan pertumbuhan karyawan. CSC per karyawan sebenarnya relatif stabil (Rp906rb vs Rp1,02jt).

**Q: Apa arti Biaya Jasa Lalu negatif?** A: Ada perubahan kebijakan yang menguntungkan perusahaan (contoh: pengurangan manfaat atau perubahan vesting period). (Referensi: T103, T165)

> 🔗 **Lihat Juga:** [Tabel 2 - Analisis Varians](#tabel-2-gain-loss) | [Business Impact Analysis](#data-analysis-insights)

#### **Pertanyaan Analisis dan Jawaban**

**1. Mengapa PVDBO tahun 2024 meningkat?**

- **Pertumbuhan Karyawan:** Bertambahnya masa kerja karyawan
- **Kenaikan Populasi:** Bertambahnya jumlah karyawan
- **Perubahan Asumsi:** Adanya **perubahan pada Asumsi Mortalitas** yang digunakan

**2. Mengapa Biaya Jasa Lalu (BJS) muncul?**

- **Perubahan Usia Pensiun:** Adanya **Perubahan** Usia Pensiun normal
- **Perubahan Kebijakan:** Adanya **Perubahan Peraturan Perusahaan/PKB/Manfaat** yang digunakan
- **Perubahan Status:** Adanya **Perubahan Status** → status dari kontrak ke tetap, atau perubahan jabatan
- **Pergerakan Karyawan:** Adanya **Mutasi Karyawan Masuk**
- **Dampak Pemutusan:** Adanya **Mutasi Keluar, dengan Realisasi yang dibayarkan > Kewajiban**
- **Perubahan Metodologi:** Adanya **Perubahan Metode Perhitungan** jika ada update dari regulasi

Setelah memahami ikhtisar Tabel 1, langkah berikutnya adalah melihat bagaimana kewajiban perusahaan berkembang sepanjang tahun berjalan, termasuk meneliti faktor-faktor yang menambah atau mengurangi kewajiban, serta apakah terjadi **selisih (keuntungan/kerugian) antara perkiraan dengan kenyataan.**

### #tabel-2-gain-loss

#### **Tabel 2 - Perhitungan Keuntungan/Kerugian Aktuarial Tahun Berjalan**

| **No.** | **Explanation**                                 | **31 Desember 2024** | **31 Desember 2023** | **Uraian**                                             |
| ------- | ----------------------------------------------- | -------------------- | -------------------- | ------------------------------------------------------ |
| **1.**  | **Actual Present Value of Obligation at BoP**   | **Rp72,18 miliar**   | **Rp54,15 miliar**   | **Nilai Kini Kewajiban pada Awal Periode**             |
| **2.**  | Past Service Cost - Non Vested                  | -                    | -                    | Biaya Jasa Lalu - Non Vested                           |
| **3.**  | Past Service Cost - Vested                      | (Rp13,35 miliar)     | (Rp4,76 miliar)      | Biaya Jasa Lalu - Vested                               |
| **4.**  | **Interest Cost**                               | **Rp5,05 miliar**    | **Rp3,84 miliar**    | **Biaya Bunga**                                        |
| **5.**  | **Current Service Cost**                        | **Rp24,37 miliar**   | **Rp17,84 miliar**   | **Biaya Jasa Kini**                                    |
| **6.**  | **Benefit Payments**                            | **(Rp1,62 miliar)**  | **(Rp2,27 miliar)**  | **Pembayaran Manfaat**                                 |
| **7.**  | Changes in Benefit Plans                        | Rp81,39 juta         | -                    | Perubahan Program Manfaat                              |
| **8.**  | Curtailment-Settlement                          | (Rp2,49 miliar)      | -                    | Kurtailmen-Penyelesaian                                |
| **9.**  | **Present Value of Obligation at EoP**          | **Rp84,06 miliar**   | **Rp68,80 miliar**   | **Nilai Kini Kewajiban pada Akhir Periode - Program**  |
| **10.** | **Actuarial (Gain) or Loss on Obligation**      | **Rp2,55 miliar**    | **Rp3,37 miliar**    | **(Keuntungan)/Kerugian Aktuaria pada Kewajiban**      |
| **11.** | **Change in Financial Assumption**              | **(Rp2,65 miliar)**  | **Rp2,22 miliar**    | **Perubahan Asumsi Keuangan**                          |
| **12.** | Change in Demographic Assumption                | Rp81,39 juta         | -                    | Perubahan Asumsi Demografi                             |
| **13.** | **Experience Adjustment**                       | **Rp5,20 miliar**    | **Rp1,15 miliar**    | **Pengalaman Penyesuaian**                             |
| **14.** | **Present Value of Obligation at EoP - Actual** | **Rp86,61 miliar**   | **Rp72,18 miliar**   | **Nilai Kini Kewajiban pada Akhir Periode - Aktual**   |
| **15.** | Actuarial (Gain) or Loss on Obligation - Actual | Rp2,55 miliar        | Rp3,37 miliar        | (Keuntungan)/Kerugian Aktuaria pada Kewajiban          |
| **16.** | Actuarial (Gain)/Loss on Benefit Payment        | -                    | -                    | (Keuntungan)/Kerugian Aktuaria pada Pembayaran Manfaat |
| **17.** | Actuarial (Gain)/Loss on Plan Assets            | -                    | -                    | (Keuntungan)/Kerugian Aktuaria pada Nilai Wajar Aktiva |
| **18.** | **Total Actuarial (Gain)/Loss**                 | **Rp2,55 miliar**    | **Rp3,37 miliar**    | **Total (Keuntungan)/Kerugian Aktuaria Tahun**         |

#### **Analisis Varians**

> 🔍 **VARIANCE ANALYSIS:** Experience adjustment naik 352% dari Rp1,15M ke Rp5,20M - indikasi perlunya review asumsi atau ada perubahan data material.

Dari tabel di atas, bisa didapat informasi bahwa:

**Temuan Kunci:**

1. **PVDBO Ekspektasi vs Aktual:** PVDBO **Akhir Perkiraan** (jika semua Asumsi sesuai rencana), didapat dari penjumlahan semua komponen di atasnya (nomor 1 s.d. 8)
2. **Dampak Asumsi Keuangan:** Terjadi keuntungan karena adanya kenaikan tingkat diskonto menjadi 7.13% (sebelumnya 7%), dan aktual dari kenaikan gaji lebih kecil dari pada yang diasumsikan, yaitu 8%
3. **Dampak Asumsi Demografis:** Terjadi kerugian karena perubahan dari TMI III ke TMI IV (dari Tabel 1) adanya peningkatan rate mortalitas
4. **Pengalaman Penyesuaian:** Disebabkan oleh selain dari penyebab perubahan asumsi keuangan dan demografi, seperti:
    - Realisasi kenaikan gaji karyawan lebih tinggi/lebih rendah dari asumsi tingkat kenaikan gaji
    - Ada perubahan data tanggal lahir/tanggal masuk kerja karena kesalahan informasi
    - Jumlah karyawan yang berhenti, meninggal, dan kecacatan lebih banyak/lebih sedikit dari asumsi keluar dan tabel mortalitas

> ⚠️ **WARNING:** Experience adjustment sebesar Rp5,20 miliar (352% kenaikan) mengindikasikan perlunya investigasi mendalam terhadap akurasi data atau asumsi yang digunakan.

#### **❓ Frequently Asked Questions - Tabel 2**

**Q: Apa penyebab experience adjustment naik 352%?** A: Kemungkinan: ✅ Data salary increase actual >8% ✅ Perubahan data demografis ✅ Turnover actual berbeda dari asumsi ✅ Kesalahan input data (Referensi: [T062, T063](05c_faq_teknis.md#metodologi-dan-asumsi-aktuaria))

**Q: Mengapa perubahan asumsi keuangan menguntungkan?** A: Discount rate naik dari 7% ke 7,13% → kewajiban turun karena nilai present value lebih kecil.

**Q: Kapan perlu concern dengan actuarial gain/loss?** A: Jika >5% dari PVDBO atau experience adjustment >10% dari CSC, perlu investigasi asumsi.

> 🔗 **Lihat Juga:** [Tabel 3 - Dampak ke OCI](#tabel-3-oci) | [Analisis Sensitivitas](#uji-sensitivitas-sensitivity-analysis)

**Penilaian Dampak:** Lalu, muncul kewajiban yang benar-benar dihitung di akhir tahun berdasarkan data dan asumsi terbaru sebagai PVDBO Akhir Aktual **yang lebih besar dari Perkiraan,** dimana selisih ini menunjukkan adanya **kerugian aktuaria.**

Akibatnya, perusahaan perlu mencatat tambahan beban, agar laporan keuangan mencerminkan kewajiban secara lebih realistis. Biasanya, angka ini masuk ke laporan keuangan melalui bagian Tabel 3 Pendapatan Komprehensif Lainnya (OCI).

### #tabel-3-oci

#### **Tabel 3 - Pendapatan Komprehensif Lainnya Perhitungan (OCI)**

| **No.** | **EXPLANATION**                                     | **31 Desember 2024** | **31 Desember 2023** | **URAIAN**                                        |
| ------- | --------------------------------------------------- | -------------------- | -------------------- | ------------------------------------------------- |
| **1**   | **Other Comprehensive Income at BoP**               | **(Rp31,37 miliar)** | **(Rp34,75 miliar)** | **Pendapatan Komprehensif lainnya awal periode**  |
| **2**   | **Actuarial (Gain)/Loss at on Period - Obligation** | **Rp2,55 miliar**    | **Rp3,37 miliar**    | **(Keuntungan)/Kerugian aktuaria – kewajiban**    |
| **3**   | Actuarial (Gain)/Loss at on Period - Plan Assets    | -                    | -                    | (Keuntungan)/Kerugian aktuaria – Aset Program     |
| **4**   | **Total Actuarial (Gain)/Loss at on Period**        | **Rp2,55 miliar**    | **Rp3,37 miliar**    | **Total (Keuntungan)/kerugian aktuaria**          |
| **5**   | **Other Comprehensive Income at EoP**               | **(Rp28,82 miliar)** | **(Rp31,37 miliar)** | **Pendapatan Komprehensif lainnya akhir periode** |
#### **Analisis OCI**

> 💰 **EQUITY IMPACT:** OCI membaik dari (Rp31,37M) ke (Rp28,82M), menunjukkan deviasi kewajiban vs prediksi mengecil - tren positif untuk pengendalian risiko.

**Wawasan Kunci:**

- **Tidak Ada Aset Program:** Tidak ada aset program, sehingga OCI hanya mencerminkan perubahan dari sisi **kewajiban** saja
- **Dampak Kerugian Aktuaria:** Kerugian aktuarial pada tahun berjalan sebesar Rp2,55 M sehingga nilai akhir dari OCI sekitar (Rp28 M), artinya perusahaan masih memperoleh keuntungan
- **Dampak Laporan Keuangan:** Perusahaan tetap harus mengungkapkan nilai OCI ini dalam laporan keuangan, meskipun **tidak langsung berdampak ke laba rugi, karena tetap memengaruhi ekuitas** perusahaan

#### **❓ Frequently Asked Questions - Tabel 3**

**Q: Mengapa OCI masih negatif tapi membaik?** A: OCI (Rp28,82M) vs (Rp31,37M) → angka negatif mengecil = improvement. Accumulated gains masih lebih besar dari accumulated losses.

**Q: Kapan OCI akan berdampak ke P&L?** A: Tidak pernah. OCI khusus actuarial gains/losses tidak akan di-recycle ke P&L sesuai PSAK 24. (Referensi: L03)

**Q: Strategi mengelola volatilitas OCI?** 
A: ✅ Konsistensi asumsi ✅ Regular assumption review ✅ Consider asset investment ✅ Hedge interest rate risk

> 🔗 **Lihat Juga:** [Tabel 4 - Balance Sheet Impact](#tabel-4-balance-sheet) | [Variance Analysis](#analisis-varians)

Setelah melihat bagaimana keuntungan/kerugian aktuaria dicatat, sekarang saatnya memahami bagaimana seluruh kewajiban imbalan kerja ini diakui dan disajikan dalam **laporan keuangan perusahaan, khususnya di neraca berdasarkan Tabel 4.**

### #tabel-4-balance-sheet

#### **Tabel 4 - Posisi Pendanaan & Pengakuan Kewajiban/(Kekayaan) dalam Neraca Perhitungan**

|**No.**|**EXPLANATION**|**31 Desember 2024**|**31 Desember 2023**|**URAIAN**|
|---|---|---|---|---|
|**1**|**FUNDED STATUS**|||**STATUS PENDANAAN**|
|**2**|**Assets and Obligation**|||**Kekayaan dan Kewajiban**|
|**3**|**Present Value of Obligation at EOP**|**Rp86,69 miliar**|**Rp72,18 miliar**|**Nilai Kini Kewajiban (Present Value of Obligation)**|
|**4**|Fair Value of Plan Assets|-|-|Nilai Wajar Aset Program|
|**5**|**Funded Status**|**Rp86,69 miliar**|**Rp72,18 miliar**|**Posisi Pendanaan**|
|**6**|Unrecognized Past Service Cost - Non Vested|-|-|Biaya Jasa Lalu yang Belum Diakui - Non Vested|
|**7**|Unrecognized Actuarial (Gains)/Losses|-|-|Keuntungan/(Kerugian) Aktuarial yang Belum Diakui|
|**8**|**Liability/(Assets) Recognized in The Balance Sheet**|**Rp86,69 miliar**|**Rp72,18 miliar**|**Kewajiban/(Kekayaan) yang Diakui dalam Neraca**|

**Reconciliation of Balance Sheet Movement:**

| **No.** | **EXPLANATION**                                              | **31 Desember 2024** | **31 Desember 2023** | **URAIAN**                                                  |
| ------- | ------------------------------------------------------------ | -------------------- | -------------------- | ----------------------------------------------------------- |
|         | **Reconciliation of Liability/(Asset) in The Balance Sheet** |                      |                      | **Perubahan Kewajiban/(Kekayaan) yang Diakui dalam Neraca** |
| **9**   | **Liability/(Asset) at BoP**                                 | **Rp72,18 miliar**   | **Rp54,15 miliar**   | **Kewajiban/(Kekayaan) pada Awal Periode**                  |
| **10**  | **Expense/(Income)**                                         | **Rp13,57 miliar**   | **Rp16,92 miliar**   | **Beban/(Pendapatan)**                                      |
| **11**  | **Benefit Payment - Actual**                                 | **(Rp1,17 miliar)**  | **(Rp2,09 miliar)**  | **Realisasi Pembayaran Manfaat**                            |
| **12**  | Company Contributions                                        | -                    | -                    | Iuran Perusahaan                                            |
| **13**  | **Other Comprehensive Income**                               | **Rp2,55 miliar**    | **Rp3,37 miliar**    | **Pendapatan Komprehensif Lainnya**                         |
| **14**  | **Liability/(Assets) at EoP**                                | **Rp86,69 miliar**   | **Rp72,18 miliar**   | **Kewajiban/(Kekayaan) pada Akhir Perio**                   |

#### #analisis-varians

**Analisis Neraca**

>  🔴 **FUNDING GAP:** Kewajiban Rp86,69 miliar tanpa aset program berarti 100% unfunded liability. Pertimbangkan strategi pendanaan jangka panjang.

**Temuan Kunci:**

- **Pengakuan Penuh:** Seluruh kewajiban sebesar Rp86,69 M **telah diakui sepenuhnya dalam neraca**
- **Tidak Ada Aset Program:** Tidak ada aset program atau komponen yang ditunda pengakuannya → laporan sudah bersih dan mencerminkan realitas
- **Analisis Pergerakan:** **Nilai akhir kewajiban** sebesar Rp86,69 M terjadi karena:
    1. **Dampak Beban:** Adanya **penambahan beban** pada tahun berjalan sebesar Rp13,5 M yang menyebabkan kewajiban bertambah
    2. **Dampak Pembayaran:** Adanya realisasi **pembayaran manfaat** sebesar Rp1,6 M yang mengurangi kewajiban
    3. **Dampak OCI:** Adanya **kerugian pada OCI** tahun berjalan sebesar Rp2 M yang menyebabkan kewajiban bertambah

#### **❓ Frequently Asked Questions - Tabel 4**

**Q: Mengapa tidak ada plan assets di PT ABC?** A: Perusahaan belum membentuk dana pensiun atau DPLK, sehingga 100% kewajiban ditanggung perusahaan (unfunded scheme). (Referensi: T153, T155)

**Q: Apa strategi untuk mengurangi funded status negatif?** A: ✅ Bentuk DPLK ✅ Invest in pension fund ✅ Optimize benefit structure ✅ Regular contribution strategy

**Q: Reconciliation balance sheet movement menunjukkan apa?** A: Movement analysis: Awal Rp72,18M + Expense Rp13,57M - Payment Rp1,17M + OCI Rp2,55M = Akhir Rp86,69M

> 🔗 **Lihat Juga:** [Tabel 5 - P&L Impact](#tabel-5-pnl-expense) | [Cash Flow Planning](#analisis-jatuh-tempo)

Terdapat nilai Beban/(Pendapatan) yang akan dijelaskan Laporan Laba Rugi (Profit & Loss) sesuai Tabel 5.

###  #tabel-5-pnl-expense

#### **Tabel 5 - Pengakuan Beban/(Pendapatan) yang diakui dalam Laba Rugi**
| **No.** | **EXPLANATION**                                                   | **31 Desember 2024** | **31 Desember 2023** | **URAIAN**                                                 |
| ------- | ----------------------------------------------------------------- | -------------------- | -------------------- | ---------------------------------------------------------- |
| **1**   | **COST COMPONENTS**                                               |                      |                      | **KOMPONEN BEBAN**                                         |
| **2**   | **Current Service Cost**                                          | **Rp24,37 miliar**   | **Rp17,84 miliar**   | **Biaya Jasa Kini**                                        |
| **3**   | **Interest Cost**                                                 | **Rp5,05 miliar**    | **Rp3,84 miliar**    | **Biaya Bunga**                                            |
| **4**   | Expected Return on Plan Assets                                    | -                    | -                    | Harapan dari Hasil Investasi                               |
| **5**   | **Immediate Recognition of Past Service Cost - Vested**           | **(Rp13,95 miliar)** | **(Rp4,76 miliar)**  | **Pengakuan Segera dari Biaya Jasa Lalu yang Vested**      |
| **6**   | **Curtailment Effect / Settlement**                               | **(Rp2,49 miliar)**  | -                    | **Dampak Kurtailmen / Penyelesaian**                       |
| **7**   | **Expense/(Income) should be Recognized In The Income Statement** | **Rp13,58 miliar**   | **Rp16,92 miliar**   | **Beban/(Pendapatan) yang Diakui dalam Laporan Laba/Rugi** |
#### **Analisis Laporan Laba Rugi**

> 📉 **P&L IMPACT:** Total beban turun 19,7% dari Rp16,92M ke Rp13,58M meski CSC naik, karena adanya past service cost (biaya jasa lalu / BJS) credit dan curtailment effect.

**Rincian Komponen:** Nilai Beban/(Pendapatan) muncul sebagai **total seluruh beban imbalan kerja** yang harus dicatat tahun ini, hasil perhitungan dari seluruh komponen (CSC, bunga, BJS, Kurtailmen).

**Komponen Kunci:**

- **Biaya Inti:** CSC + Biaya Bunga = Beban operasional berkelanjutan
- **Penyesuaian:** Biaya Jasa Lalu + Kurtailmen = Penyesuaian satu kali
- **Dampak Bersih:** Total beban untuk pengakuan P&L

#### **❓ Frequently Asked Questions - Tabel 5**

**Q: Mengapa P&L expense turun padahal CSC naik?** A: BJS credit (Rp13,95M) dan curtailment effect (Rp2,49M) lebih besar dari kenaikan CSC dan interest cost. (Referensi: T156)

**Q: Apa itu curtailment effect?** A: Pengurangan kewajiban karena early retirement, restructuring, atau pengurangan karyawan yang mengurangi future service.

**Q: Apakah trend ini sustainable?** A: BJS dan curtailment adalah one-time items. Core expense (CSC + Interest) naik signifikan dan lebih representatif untuk proyeksi.

> 🔗 **Lihat Juga:** [Advanced Analysis Tools](#alat-analisis-lanjutan) | [Cash Flow Impact](#analisis-jatuh-tempo)

Setelah membahas seluruh proses Valuasi Aktuaria, maka langkah berikutnya adalah melengkapi laporan dengan analisis tambahan, seperti analisis sensitivitas, pengalaman penyesuaian, dan analisis jatuh tempo.

## #alat-analisis-lanjutan

Berlanjut ke [Analisis Lanjutan](02f2_valuasi_analisis_lanjutan.md)

---

## 📝 **Ringkasan**

### **Key Takeaways**

- **Proses End-to-End:** Dari pengumpulan data hingga pelaporan akhir memerlukan pendekatan sistematis
- **Kerangka Lima Tabel:** Struktur pelaporan komprehensif untuk kebutuhan stakeholder
- **Analitik Lanjutan:** Sensitivitas, pengalaman, dan analisis jatuh tempo memberikan wawasan bisnis
- **Jaminan Kualitas:** Proses validasi yang kuat memastikan hasil yang dapat diandalkan

### **Kerangka Perhitungan Aktuaria**

|**Tahap**|**Komponen Utama**|**Output**|
|:--|:--|:--|
|**Pengumpulan Data**|Data karyawan & informasi perusahaan|Dataset valid dan lengkap|
|**Penetapan Asumsi**|Asumsi demografis & keuangan|Parameter perhitungan|
|**Perhitungan PUC**|PVFB, PVDBO, CSC|Nilai kewajiban & beban|
|**Analisis Komponen**|Keuntungan/kerugian, OCI, biaya jasa lalu|Komponen laporan keuangan|
|**Penyajian Laporan**|5 tabel utama + analisis pendukung|Laporan aktuaria lengkap|
### **Komponen Teknis Utama**

1. **PVDBO:** Nilai kini kewajiban berdasarkan masa kerja yang telah dilalui
2. **CSC:** Biaya jasa periode berjalan yang diakui dalam laba rugi
3. **Biaya Bunga:** Biaya bunga atas kewajiban yang outstanding
4. **Asumsi Demografis:** Mortalitas, pengunduran diri, tingkat cacat
5. **Asumsi Keuangan:** Tingkat diskonto, kenaikan gaji, ROI

### **Implementasi Praktis**

- **Gunakan data berkualitas:** Gunakan data karyawan yang lengkap dan akurat
- **Tetapkan asumsi realistis:** Tetapkan asumsi berdasarkan kondisi pasar dan data historis
- **Lakukan analisis sensitivitas:** Lakukan analisis sensitivitas untuk memahami volatilitas
- **Pelihara dokumentasi:** Dokumentasikan semua metodologi dan asumsi untuk audit
- **Tinjau secara berkala:** Tinjau dan perbarui asumsi secara berkala

---
### **Next Steps

Setelah memahami penyajian tabel perhitungan aktuaria, langkah berikutnya adalah melakukan analisis lanjutan, seperti dampak perubahan asumsi dan berapa jumlah pencadangan saat jangka jatuh tempo tertentu.

---

**Navigasi:** ⬅️ [Data Preparation](02f_valuasi_data.md) | [📋 Daftar Isi](README.md) | [Analisis Lanjutan](02f2_valuasi_analisis_lanjutan.md) ➡️

---

> **💡 Catatan:** Untuk pertanyaan teknis lebih detail tentang proses valuasi, silakan merujuk ke [FAQ Teknis dan Perhitungan](05c_faq_teknis.md) atau hubungi konsultan aktuaria yang berpengalaman.