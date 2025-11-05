---
keywords:
  - proses valuasi
  - laporan aktuaria
  - data karyawan
  - asumsi aktuaria
  - sains aktuaria
  - matematis aktuaria
  - valuasi aktuaria
  - employee benefits
  - imbalan kerja
difficulty: advanced
estimated_reading: 30 minutes
target_audience:
  - finance
  - actuaries
  - management
  - technical_staff
document_type: process_guide
last_updated: 2025-07-28
version: "2025.1"

# SCOPE CONTROL - AKTUARIA FOCUS
domain: aktuaria
scope: sains_aktuaria_proses_valuasi
exclude_domains:
  - asuransi_umum
  - perbankan_komersial
  - insurance_reserves
  - claim_valuation
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
  - proses valuasi aktuaria
  - data karyawan
  - laporan aktuaria

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
# Proses Valuasi Aktuaria

**📖 Bagian dari:** [PANDUAN LENGKAP IMBALAN KERJA PSAK 219](README.md)

---
## 🎯 **Tujuan Pembelajaran**

Setelah membaca bab ini, pembaca akan mampu:

- [ ]  Menyiapkan seluruh data karyawan dan data keuangan perusahaan
- [ ]  Menganalisis asumsi berdasarkan kondisi aktual dan histori perhitungan sebelumnya
- [ ]  Melakukan proses valuasi aktuaria secara end-to-end

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

## #proses-valuasi-aktuaria-laporan-aktuaria 

Setelah memahami bagaimana komponen imbalan kerja disajikan dalam laporan keuangan—baik dalam laba rugi maupun pendapatan komprehensif lainnya (OCI)—pertanyaan berikutnya yang sering muncul adalah: _**bagaimana nilai-nilai tersebut diperoleh?**_

Untuk menjawabnya, kita perlu melihat lebih dekat proses di balik layar, yaitu bagaimana Perhitungan Aktuaria dilakukan dan bagaimana laporan tersebut disusun.

### **Studi Kasus: PT ABC Manufacturing**

Sebagai contoh kasus sederhana, ini salah satu penerapan Perhitungan Aktuaria dari salah satu perusahaan di Indonesia.

> **Profil Perusahaan:** PT ABC adalah perusahaan manufaktur sepatu ingin melakukan Valuasi Aktuaria di tahun 2024, terdiri dari karyawan tetap dan kontrak, dan tidak mengikuti program DPLK.

## **Proses Valuasi Aktuaria**

> 🔗 **Roadmap:** Setelah memahami data dan asumsi, berikut 5 tabel utama yang membentuk laporan aktuaria lengkap: [Persiapan Data dan Asumsi](02f_proses_valuasi.md) → [Penyajian Tabel Aktuaria](02f1_valuasi_tabel_utama.md) → [Analisis Lanjutan](02f2_valuasi_analisis_lanjutan.md)

### **Langkah 1: Menyiapkan Data yang Dibutuhkan**

> ⚠️ **CRITICAL:** Kualitas data menentukan akurasi hasil valuasi. Data yang tidak akurat dapat menyebabkan under/overstatement kewajiban hingga 10-15%.

#### **Ikhtisar Data Karyawan**

**Karyawan Tetap:**

| Metric                               | **31 Desember 2024** | **31 Desember 2023** | **Keterangan**                                    |
| :----------------------------------- | :------------------- | :------------------- | :------------------------------------------------ |
| **Number of Participants**           | 26.897               | 17.839               | Jumlah Peserta (orang)                            |
| **Average Age of Employee (Years)**  | 26,57                | 26,91                | Rata-rata Usia (Tahun) untuk Karyawan Tetap       |
| **Average Years of Service (Years)** | 2,55                 | 3,42                 | Rata-rata Masa Kerja (Tahun) untuk Karyawan Tetap |

**Karyawan Kontrak:**

| Metric | **31 Desember 2024** | **31 Desember 2023** | **Keterangan** |
|:-------|:---------------------|:---------------------|:---------------|
| **Number of Participants** | 11 | - | Jumlah Peserta (orang) |
| **Average Age of Employee (Years)** | 56,82 | - | Rata-rata Usia (Tahun) untuk Karyawan Kontrak |
| **Average Years of Service (Years)** | 0,47 | - | Rata-rata Masa Kerja (Tahun) untuk Karyawan Kontrak |

#### **Ikhtisar Data Keuangan**
| Metric                             | **31 Desember 2024** | **31 Desember 2023** | **Keterangan**                                       |
| ---------------------------------- | -------------------- | -------------------- | ---------------------------------------------------- |
| **Total Monthly Wages**            | Rp69,71 miliar       | Rp44,35 miliar       | Jumlah Gaji Sebulan untuk Karyawan Tetap dan Kontrak |
| **Benefits Payment in The Period** | (Rp1,62 miliar)      | (Rp2,27 miliar)      | Realisasi Pembayaran Manfaat Dalam Tahun Berjalan    |
| **Benefit Paid - Asset Program**   | -                    | -                    | Realisasi Pembayaran Manfaat (Akun Program)          |
| **Company Contribution Paid**      | -                    | -                    | Iuran Perusahaan Dalam Tahun Berjalan                |
| **Saldo DPLK (Employer Portion)**  | -                    | -                    | Saldo DPLK Porsi Perusahaan                          |
#### #data-analysis-insights

##### **Data Analysis Insights**

> 📈 **BUSINESS IMPACT:** Pertumbuhan karyawan 50,8% dan kenaikan gaji 57,2% mengindikasikan potensi kenaikan kewajiban yang material.

Pada data keuangan, dapat diambil informasi bahwa:

**Indikator Positif:**

- **Pertumbuhan Karyawan:** Jumlah karyawan naik signifikan (+50%), menunjukkan **potensi kenaikan kewajiban** (CSC & PVDBO)
- **Demografis Stabil:** Rata-rata usia stabil, menunjukkan banyak karyawan baru yang mungkin belum fully vested

**Faktor Risiko:**

- **Tidak Ada Aset Program:** Tidak ada aset program/iuran perusahaan, artinya seluruh kewajiban akan **dibebankan penuh ke perusahaan**
- **Penurunan Pembayaran Manfaat:** Realisasi manfaat menurun kemungkinan karena **berkurangnya kasus resign/pensiun/meninggal tahun berjalan**

#### **❓ Frequently Asked Questions - Data Preparation**

**Q: Mengapa pertumbuhan karyawan 50% bisa berdampak material ke kewajiban?** A: Setiap karyawan baru menambah CSC dan PVDBO. Dengan 9.058 karyawan baru, estimasi tambahan kewajiban bisa mencapai Rp15-20 miliar.

**Q: Apakah karyawan kontrak perlu dimasukkan dalam valuasi?** A: Ya, jika mereka berhak atas manfaat imbalan kerja sesuai regulasi atau kebijakan perusahaan. (Referensi: [T053](05c_faq_teknis.md#metodologi-dan-asumsi-aktuaria) , [L47](05b_faq_legal.md#aspek-hukum-ketenagakerjaan)

**Q: Indikator apa yang menunjukkan data siap untuk valuasi?** A: ✅ Data lengkap tanpa missing value ✅ Validasi silang dengan payroll ✅ Konfirmasi perubahan kebijakan HR ✅ Backup data tahun sebelumnya

> 🔗 **Lihat Juga:**  [Tabel 1 - Data dan Asumsi](#tabel-1-data-asumsi)

#### #daftar-periksa-validasi-data

##### **Daftar Periksa Validasi Data**

**Konfirmasi yang Diperlukan:**

- [ ] Apakah kenaikan gaji terjadi secara merata atau karena ada kategori jabatan tertentu?
- [ ] Apakah karyawan kontrak menerima manfaat jangka panjang? _Jika ya, harus dimasukkan ke perhitungan._
- [ ] Apakah terdapat perubahan struktur manfaat _(misalnya: perubahan aturan pensiun atau Pesangon)?_
- [ ] Konfirmasi bahwa tidak ada aset Dana Pensiun internal atau pihak ketiga yang perlu diperhitungkan

### **Langkah 2: Menetapkan Asumsi Aktuaria**

> 💡 **TIP:** Gunakan benchmarking terhadap industri sejenis untuk validasi asumsi. Asumsi yang tidak realistis dapat menyebabkan volatilitas OCI yang tinggi.

#### **Tabel Asumsi Aktuaria**

| Assumption                                 | **31 Desember 2024**                                                                             | **31 Desember 2023**                                                                             | **Keterangan**                                |
| :----------------------------------------- | :----------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------- | :-------------------------------------------- |
| **Discount Rate Beginning Period**         | 7,00%                                                                                            | 7,25%                                                                                            | Tingkat Diskonto Awal Tahun                   |
| **Discount Rate Ending Period**            | 7,13%                                                                                            | 7,00%                                                                                            | Tingkat Diskonto Akhir Tahun                  |
| **Expected Rate of Return on Plan Assets** | -                                                                                                | -                                                                                                | Tingkat Harapan Investasi atas Aktiva Program |
| **Future Salary Increases (per annum)**    | 8,00%                                                                                            | 8,00%                                                                                            | Tingkat Kenaikan Gaji (per Tahun)             |
| **Mortality Table**                        | TMI IV                                                                                           | TMI III                                                                                          | Tabel Mortalitas                              |
| **Disability Rate**                        | 10,00% dari TMI IV                                                                               | 10,00% dari TMI III                                                                              | Tingkat Cacat                                 |
| **Withdrawal Rate**                        | 20–29 = 6.00%<br>30–34 = 3.00%<br>35–39 = 1.80%<br>40–50 = 1.20%<br>51–52 = 0.60%<br>>52 = 0.00% | 20–29 = 6.00%<br>30–34 = 3.00%<br>35–39 = 1.80%<br>40–50 = 1.20%<br>51–52 = 0.60%<br>>52 = 0.00% | Tingkat Pengunduran Diri                      |
| **Cost Method**                            | PUC (IFRIC)                                                                                      | PUC (IFRIC)                                                                                      | Metode Perhitungan Aktuaria                   |
| **Normal Retirement Age**                  | 55                                                                                               | 55                                                                                               | Usia Pensiun Normal (Tahun)                   |

#### #analisis-asumsi

##### **Analisis Asumsi**

> ⚠️ **PENTING:** Perubahan tabel mortalitas dari TMI III ke TMI IV memiliki dampak material. Pastikan justifikasi perubahan didokumentasikan untuk audit.

**Perubahan Kunci dan Dampak:**

- **Kenaikan Tingkat Diskonto:** Naiknya Tingkat Diskonto akhir adalah faktor yang bisa menyebabkan PVDBO sedikit menurun, meskipun data karyawan dan gaji tetap stabil
- **Stabilitas Tingkat Pengunduran Diri:** Penyesuaian tingkat pengunduran diri, semakin banyak karyawan yang **diperkirakan bertahan → PVDBO naik**
- **Konsistensi Kenaikan Gaji:** Kenaikan gaji tidak berubah → jika realisasi ternyata **lebih tinggi,** bisa menimbulkan **kerugian aktuaria** di tahun berikutnya

#### **❓ Frequently Asked Questions - Asumsi Aktuaria**

**Q: Kapan perlu mengubah asumsi aktuaria?** A: ✅ Annual review wajib ✅ Perubahan kondisi pasar material ✅ Experience adjustment >5% ✅ Perubahan kebijakan perusahaan

**Q: Dampak perubahan TMI III ke TMI IV?** A: TMI IV memiliki mortalitas yang lebih rendah (umur lebih panjang) → kewajiban naik karena benefit dibayar lebih lama. (Referensi: [T056](05c_faq_teknis.md#metodologi-dan-asumsi-aktuaria))

**Q: Bagaimana menentukan discount rate yang tepat?** 
A: Gunakan yield kurva obligasi pemerintah/korporat grade A dengan durasi sesuai kewajiban (biasanya 10-15 tahun). (Referensi: [T058](05c_faq_teknis.md#metodologi-dan-asumsi-aktuaria))

> 🔗 **Lihat Juga:** [Analisis Sensitivitas](#uji-sensitivitas-sensitivity-analysis) | [Experience Adjustment](#experience-adjustment)
> 
#### **Persyaratan Validasi Asumsi**

**Validasi yang Diperlukan:**

1. **Analisis Benchmark:** Seluruh asumsi telah dikaji ulang terhadap:
    
    - Data aktual tahun sebelumnya
    - Kebijakan SDM dan tren internal perusahaan
    - Kondisi pasar (kurva yield, inflasi, kenaikan gaji sektoral)
2. **Dokumentasi:** Dokumentasikan alasan perubahan asumsi, khususnya jika diskonto turun atau tingkat pengunduran diri disesuaikan, untuk keperluan audit dan ketertelusuran hasil
    
3. **Koordinasi Manajemen:** Koordinasi dengan HR atau manajemen SDM, jika terdapat potensi perubahan usia pensiun, pengurangan tenaga kerja, atau kebijakan baru yang bisa memengaruhi kewajiban masa depan
    

### **Langkah 3: Perhitungan Beban Kewajiban**

#### **Struktur Manfaat**

**Manfaat yang Diterima:**

- **Uang Pesangon, Uang Penghargaan Masa Kerja dan Uang Penggantian Hak** sesuai Peraturan Perusahaan dan UU Ketenagakerjaan No. 6 Tahun 2023
- **Usia Pensiun normal:** 55 tahun

#### **Formula Manfaat UUCK**

**Tabel Formula Manfaat:**


| Service Years | Severance (B) | Service (C) | Pension Factor | Death Factor | Disability Factor | Withdrawal Factor |
| ------------- | ------------- | ----------- | -------------- | ------------ | ----------------- | ----------------- |
| 0             | 1             | 0           | 1.75           | 2.00         | 2.00              | 0.00              |
| 1             | 2             | 0           | 3.50           | 4.00         | 4.00              | 0.00              |
| 2             | 3             | 0           | 5.25           | 6.00         | 6.00              | 0.00              |
| 3             | 4             | 2           | 9.00           | 10.00        | 10.00             | 0.00              |
| 4             | 5             | 2           | 10.75          | 12.00        | 12.00             | 0.00              |
| 5             | 6             | 2           | 12.50          | 14.00        | 14.00             | 0.00              |
| 6             | 7             | 3           | 15.25          | 17.00        | 17.00             | 0.00              |
| 7             | 8             | 3           | 17.00          | 19.00        | 19.00             | 0.00              |
| 8             | 9             | 3           | 18.75          | 21.00        | 21.00             | 0.00              |
| 9             | 9             | 4           | 19.75          | 22.00        | 22.00             | 0.00              |
| 10            | 9             | 4           | 19.75          | 22.00        | 22.00             | 0.00              |
| 11            | 9             | 4           | 19.75          | 22.00        | 22.00             | 0.00              |
| 12            | 9             | 5           | 20.75          | 23.00        | 23.00             | 0.00              |
| 13            | 9             | 5           | 20.75          | 23.00        | 23.00             | 0.00              |
| 14            | 9             | 5           | 20.75          | 23.00        | 23.00             | 0.00              |
| 15            | 9             | 6           | 21.75          | 24.00        | 24.00             | 0.00              |
| 16            | 9             | 6           | 21.75          | 24.00        | 24.00             | 0.00              |
| 17            | 9             | 6           | 21.75          | 24.00        | 24.00             | 0.00              |
| 18            | 9             | 7           | 22.75          | 25.00        | 25.00             | 0.00              |
| 19            | 9             | 7           | 22.75          | 25.00        | 25.00             | 0.00              |
| 20            | 9             | 7           | 22.75          | 25.00        | 25.00             | 0.00              |
| 21            | 9             | 8           | 23.75          | 26.00        | 26.00             | 0.00              |
| 22            | 9             | 8           | 23.75          | 26.00        | 26.00             | 0.00              |
| 23            | 9             | 8           | 23.75          | 26.00        | 26.00             | 0.00              |
| 24            | 9             | 10          | 25.75          | 28.00        | 28.00             | 0.00              |
| 25            | 9             | 10          | 25.75          | 28.00        | 28.00             | 0.00              |
| 26            | 9             | 10          | 25.75          | 28.00        | 28.00             | 0.00              |
| 27            | 9             | 10          | 25.75          | 28.00        | 28.00             | 0.00              |
| 28            | 9             | 10          | 25.75          | 28.00        | 28.00             | 0.00              |
| 29            | 9             | 10          | 25.75          | 28.00        | 28.00             | 0.00              |
| 30            | 9             | 10          | 25.75          | 28.00        | 28.00             | 0.00              |
| 31            | 9             | 10          | 25.75          | 28.00        | 28.00             | 0.00              |
| 32            | 9             | 10          | 25.75          | 28.00        | 28.00             | 0.00              |
| 33            | 9             | 10          | 25.75          | 28.00        | 28.00             | 0.00              |
| 34            | 9             | 10          | 25.75          | 28.00        | 28.00             | 0.00              |
| 35            | 9             | 10          | 25.75          | 28.00        | 28.00             | 0.00              |
| 36            | 9             | 10          | 25.75          | 28.00        | 28.00             | 0.00              |
| 37            | 9             | 10          | 25.75          | 28.00        | 28.00             | 0.00              |
| 38            | 9             | 10          | 25.75          | 28.00        | 28.00             | 0.00              |
| 39            | 9             | 10          | 25.75          | 28.00        | 28.00             | 0.00              |
| 40            | 9             | 10          | 25.75          | 28.00        | 28.00             | 0.00              |

**Komponen Uang Penggantian:**

- **Cuti tahunan yang belum diambil** dan belum gugur
- **Penggantian perumahan** serta pengobatan dan perawatan ditetapkan 15% dari uang Pesangon dan/atau uang penghargaan masa kerja bagi yang memenuhi syarat
- **Hal-hal lain** yang ditetapkan dalam perjanjian kerja, Peraturan Perusahaan, atau Perjanjian Kerja Bersama

## **Penyajian Laporan Aktuaria**

Berlanjut ke [Penyajian Tabel Aktuaria](02f1_valuasi_tabel_utama.md)

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

Setelah menyiapkan data yang telah terverifikasi valid dan lengkap , langkah berikutnya adalah mempelajari proses valuasi aktuaria secara end-to-end dan penyusunan laporan aktuaria yang komprehensif.

---

**Navigasi:** ⬅️ [Laporan Keuangan](02e_laporan_keuangan.md) | [📋 Daftar Isi](README.md) | [Tabel Perhitungan Aktuaria](02f1_valuasi_tabel_utama.md) ➡️

---

> **💡 Catatan:** Untuk pertanyaan teknis lebih detail tentang proses valuasi, silakan merujuk ke [FAQ Teknis dan Perhitungan](05c_faq_teknis.md) atau hubungi konsultan aktuaria yang berpengalaman.