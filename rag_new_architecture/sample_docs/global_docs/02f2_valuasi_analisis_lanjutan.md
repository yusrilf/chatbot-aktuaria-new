---
keywords:
  - proses valuasi
  - laporan aktuaria
  - maturity analysis
  - sensitivity analysis
  - experience adjustment
  - pengalaman penyesuaian
  - analisis jatuh tempo
  - sains aktuaria
  - matematis aktuaria
difficulty: advanced
estimated_reading: 20 minutes
target_audience:
  - finance
  - actuaries
  - management
  - risk_management
document_type: analytical_guide
last_updated: 2025-07-28
version: "2025.1"

# SCOPE CONTROL - AKTUARIA FOCUS
domain: aktuaria
scope: sains_aktuaria_analisis_lanjutan
exclude_domains:
  - asuransi_umum
  - perbankan_komersial
  - insurance_analytics
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
  - sensitivity analysis
  - maturity analysis
  - experience adjustment
  - analisis lanjutan aktuaria

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
# Alat Analisis Lanjutan Aktuaria

**📖 Bagian dari:** [PANDUAN LENGKAP IMBALAN KERJA PSAK 219](README.md) dan melanjutkan dari [Tabel Perhitungan Aktuaria](02f1_valuasi_tabel_utama.md)

---
## 🎯 **Tujuan Pembelajaran**

Setelah membaca bab ini, pembaca akan mampu:

- [ ]  Melakukan analisis lanjutan (sensitivitas, jatuh tempo, pengalaman penyesuaian)

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

## #alat-analisis-lanjutan

### #uji-sensitivitas-sensitivity-analysis
#### **Uji Sensitivitas (Sensitivity Analysis)**

Analisis Sensitivitas adalah semacam **"simulasi stres"** untuk kewajiban perusahaan. Perusahan bisa tahu: _"Kalau suku bunga turun 1%, seberapa besar tambahan beban kewajiban yang harus saya siapkan?"_

#### **Hasil Analisis Sensitivitas**

**Dampak perubahan 1% tingkat bunga diskonto dan kenaikan gaji**

**Karyawan Tetap dan Kontrak**

|                     | **31 Desember 2024**     | **1% Increase**    | **1% Decrease**    | **Keterangan**             |
| ------------------- | ------------------------ | ------------------ | ------------------ | -------------------------- |
| **Discount Rate**   |                          | **8.13%**          | **6.13%**          | **Tingkat bunga Diskonto** |
|                     | **PVDBO**                | **Rp77,42 miliar** | **Rp97,59 miliar** | **Nilai Kini Kewajiban**   |
|                     | **Current Service Cost** | **Rp21,48 miliar** | **Rp27,81 miliar** | **Biaya Jasa Kini**        |
| **Salary Increase** |                          | **9.00%**          | **7.00%**          | **Kenaikan Gaji**          |
|                     | **PVDBO**                | **Rp97,28 miliar** | **Rp77,51 miliar** | **Nilai Kini Kewajiban**   |
|                     | **Current Service Cost** | **Rp27,71 miliar** | **Rp21,51 miliar** | **Biaya Jasa Kini**        |

> ⚠️ **SENSITIVITY ALERT:** Penurunan discount rate 1% > dapat meningkatkan PVDBO hingga Rp20,17 miliar (26% increase) - risiko signifikan jika suku bunga turun.

#### **❓ Frequently Asked Questions - Sensitivity Analysis**

**Q: Variabel mana yang paling sensitive?** A: Salary increase rate > Discount rate. Kenaikan salary 1% → impact Rp10,59M, penurunan discount 1% → impact Rp10,90M. (Referensi: [T054, T057](05c_faq_teknis.md#metodologi-dan-asumsi-aktuaria))

**Q: Bagaimana menggunakan hasil sensitivity untuk risk management?** A: ✅ Set budget contingency ✅ Monitor market rates ✅ Consider hedging strategies ✅ Stress test business plan

**Q: Frekuensi review sensitivity analysis?** A: Annual wajib, quarterly untuk monitoring, immediate jika ada perubahan market/business conditions material.

> 🔗 **Lihat Juga:** [Risk Management Strategy](#dampak-bisnis) | [Experience Adjustment Analysis](#experience-adjustment)

### #dampak-bisnis
#### **Dampak Bisnis**

Juga, membantu **pengambilan keputusan jangka panjang.** Misalnya, untuk menentukan apakah perlu:

- **Meningkatkan pencadangan**
- **Meninjau ulang kebijakan kenaikan gaji**
- **Mempertimbangkan program pensiun yang lebih berkelanjutan**

### #experience-adjustment
#### **Experience Adjustment**

Pengalaman penyesuaian adalah **"cermin"** yang memperlihatkan seberapa akurat asumsi perusahaan dan bisa memengaruhi OCI apabila terjadi perubahan data perhitungan.

Misalkan, perhitungan tanggal valuasi 31 Desember 2024. Dengan data 2024 dihitung menggunakan asumsi yang digunakan tahun sebelumnya.

#### **Hasil Pengalaman Penyesuaian**

**Karyawan Tetap dan Kontrak**

|                                 | **31 Desember 2024** | **31 Desember 2023** |                                |
| :------------------------------ | :------------------- | :------------------- | :----------------------------- |
| **Present Value of Obligation** | **Rp86,69 miliar**   | **Rp72,18 miliar**   | **Nilai kini kewajiban**       |
| **Fair Value of Plan Assets**   | **-**                | **-**                | **Nilai wajar aktiva program** |
| **Funded Status**               | **Rp86,69 miliar**   | **Rp72,18 miliar**   | **Posisi Pendanaan**           |
| **Experience Adjustment**       | **Rp5,20 miliar**    | **Rp1,15 miliar**    | **Pengalaman Penyesuaian**     |

> 🔍 **ASSUMPTION REVIEW:** Experience adjustment naik 352% mengindikasikan perlunya evaluasi komprehensif terhadap asumsi demografis dan keuangan.

#### **❓ Frequently Asked Questions - Experience Adjustment**

**Q: Threshold berapa experience adjustment dianggap material?** A: >5% dari PVDBO atau >10% dari CSC. PT ABC: Rp5,20M/Rp86,69M = 6% → material, perlu investigasi.

**Q: Langkah investigasi experience adjustment tinggi?** A: ✅ Review data quality ✅ Compare actual vs assumed rates ✅ Validate demographic changes ✅ Check calculation methodology

**Q: Dampak ke stakeholder?** A: Auditor akan question, management perlu explain, investor concern dengan predictability.

> 🔗 **Lihat Juga:** [Assumption Setting](#analisis-asumsi) | [Data Quality](#daftar-periksa-validasi-data)

**Analisis:** Penyesuaian cukup besar (Rp5,2 M), menandakan perlu **evaluasi ulang terhadap asumsi atau data yang digunakan.**

### #analisis-jatuh-tempo 

#### **Analisis Jatuh Tempo (Maturity Analysis)**

Analisis ini merupakan proyeksi waktu pencairan kewajiban perusahaan ke depan. Ini mencakup pensiun dini, resign, atau kematian mendadak. Harus disiapkan agar **tidak mengganggu arus kas.**

#### **Hasil Analisis Jatuh Tempo**

**Karyawan Tetap dan Kontrak**

| **Maturity Analysis (Rp)** | **31 Desember 2024** | **31 Desember 2023** | Keterangan                |
| -------------------------- | -------------------- | -------------------- | ------------------------- |
| **Less than 1 year**       | **Rp609,26 juta**    | **-**                | **Dibawah 1 tahun**       |
| **Between 1 and 2 years**  | **Rp2,43 miliar**    | **-**                | **Antara 1 dan 2 tahun**  |
| **Between 2 and 3 years**  | **Rp1,42 miliar**    | **-**                | **Antara 2 dan 3 tahun**  |
| **Between 3 and 5 years**  | **Rp5,34 miliar**    | **-**                | **Antara 3 dan 5 tahun**  |
| **Between 5 and 10 years** | **Rp14,00 miliar**   | **-**                | **Antara 5 dan 10 tahun** |
| **Beyond 10 years**        | **Rp205,09 miliar**  | **-**                | **Diatas 10 tahun**       |
| **Total**                  | **Rp228,88 miliar**  | **-**                | **Jumlah**                |

> 📊 **CASH FLOW INSIGHT:** 89,6% kewajiban jatuh tempo >10 tahun memberikan waktu untuk strategi pendanaan jangka panjang.

### **❓ Frequently Asked Questions - Maturity Analysis**

**Q: Mengapa total maturity analysis Rp228,88M lebih besar dari PVDBO Rp86,69M?** A: Maturity analysis menggunakan undiscounted cash flows, sedangkan PVDBO sudah di-discount ke present value. (Referensi: T110)

**Q: Bagaimana menggunakan info ini untuk cash flow planning?** A: ✅ Short-term liquidity: siapkan Rp4,4M untuk 1-3 tahun ✅ Medium-term: Rp19,3M untuk 3-10 tahun ✅ Long-term funding strategy untuk >10 tahun

**Q: Risiko concentration pada periode tertentu?** A: 89,6% >10 tahun relatif aman, tapi monitor approaching retirement cohorts untuk avoid liquidity shock.

> 🔗 **Lihat Juga:** [Funding Strategy](#analisis-neraca) | [Long-term Planning](#implementasi-praktis)

#### **Aplikasi Strategis**

Analisis Jatuh Tempo berguna untuk:

- **Proyeksi arus kas jangka panjang**
- **Penyusunan anggaran HR tahunan**
- **Menentukan perlunya ikut DPLK atau membentuk cadangan khusus**

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

Setelah memahami proses analisis lanjutan aktuaria, langkah berikutnya adalah mempelajari proses implementasi sistem teknologi yang mempermudah proses perhitungan aktuaria imbalan kerja karyawan dalam sekejap.

---

**Navigasi:** ⬅️ [Tabel Perhitungan Aktuaria](02f1_valuasi_tabel_utama.md) | [📋 Daftar Isi](README.md) | [Implementasi & Teknologi](03_implementasi_teknologi.md) ➡️

---

> **💡 Catatan:** Untuk pertanyaan teknis lebih detail tentang proses valuasi, silakan merujuk ke [FAQ Teknis dan Perhitungan](05c_faq_teknis.md) atau hubungi konsultan aktuaria yang berpengalaman.