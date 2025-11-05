---
keywords:
  - laporan keuangan
  - PSAK 219
  - laporan laba rugi
  - OCI
  - other comprehensive income
  - pendapatan komprehensif lainnya
  - neraca
  - balance sheet
  - financial statement
  - aktuaria
  - sains aktuaria
difficulty: intermediate
estimated_reading: 25 minutes
target_audience:
  - finance
  - accounting
  - auditors
  - actuaries
document_type: technical_guide
last_updated: 2025-07-28
version: "2025.1"
related_regulations:
  - PSAK_219
  - IFRS_19

# SCOPE CONTROL - AKTUARIA FOCUS
domain: aktuaria
scope: sains_aktuaria_pelaporan
exclude_domains:
  - asuransi_umum
  - perbankan_komersial
  - insurance_accounting
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
  - laporan keuangan aktuaria
  - penyajian aktuaria
  - OCI aktuaria

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
# Penyajian Laporan Keuangan

**📖 Bagian dari:** [PANDUAN LENGKAP IMBALAN KERJA PSAK 219](README.md)

---
## 🎯 **Tujuan Pembelajaran**

Setelah membaca bab ini, pembaca akan mampu:

- [ ]  Melakukan penyajian hasil perhitungan aktuaria dalam laporan keuangan
- [ ]  Memahami perbedaan pengakuan di Laporan Laba Rugi vs OCI
- [ ]  Menganalisis faktor-faktor yang memengaruhi perubahan kewajiban
- [ ]  Melakukan analisis sensitivitas untuk manajemen risiko

---
## ⚡ **Quick Reference**

### 📊 **Komponen Laba Rugi:**
- **CSC:** Biaya Jasa Kini (current service cost)
- **Biaya Bunga:** Biaya bunga atas kewajiban
- **Biaya Jasa Lalu:** Biaya jasa lalu (satu kali)
- **Hasil Aset:** Penghasilan dari aset program

### 📈 **Komponen OCI:**
- **Keuntungan/Kerugian Aktuaria:** Perubahan asumsi dan pengalaman
- **Keuntungan/Kerugian Aset:** Selisih return investasi vs asumsi
- **Dampak Batas Atas Aset:** Dampak batas atas aset

### 🏦 **Neraca:**
- **Kewajiban Bersih:** PVDBO - Aset Program
- **Lancar vs Tidak Lancar:** Klasifikasi berdasarkan jatuh tempo

---

# #penyajian-laporan-keuangan 

Penyajian nilai imbalan kerja dalam laporan keuangan membantu pengguna laporan, seperti manajemen, auditor, atau investor, untuk memahami beban dan kewajiban yang timbul akibat program Imbalan Pasca Kerja. Sesuai dengan PSAK 219, komponen imbalan kerja disajikan pada dua bagian utama laporan keuangan.

## **Laporan Laba Rugi (Profit or Loss)**

Berikut adalah komponen imbalan kerja yang langsung memengaruhi laba bersih perusahaan di laporan keuangan:

### **Komponen Laba Rugi**

| **Komponen**                  | **Penjelasan**                                                                        | **Dampak pada Laba Rugi**     |
| ----------------------------- | ------------------------------------------------------------------------------------- | ----------------------------- |
| **Biaya Jasa Kini (CSC)**     | Biaya tahun berjalan atas manfaat yang diperoleh karyawan selama masa kerja aktif     | **Menambah Beban**            |
| **Biaya Jasa Lalu**           | Tambahan beban akibat perubahan manfaat yang berdampak ke masa kerja sebelumnya       | **Menambah/Mengurangi Beban** |
| **Biaya Bunga**               | Penyesuaian atas kewajiban karena waktu berlalu (diskonto)                            | **Menambah Beban**            |
| **Hasil Aset Program**        | Penghasilan dari aset program pensiun, mengurangi beban (jika ada)                    | **Mengurangi Beban**          |
| **Kurtailmen / Penyelesaian** | Pengurangan kewajiban akibat pemutusan hubungan kerja massal atau penghentian program | **Mengurangi Beban**          |
### **Formula Beban Laba Rugi**

```
Total Beban Imbalan Karyawan = CSC + Biaya Bunga + Biaya Jasa Lalu - Ekspektasi Hasil Aset + Kurtailmen/Penyelesaian
```

**Contoh Perhitungan:**

```
CSC                     : Rp 24.367.080.194
Biaya Bunga             : Rp  5.052.359.200  
Biaya Jasa Lalu         : (Rp 13.949.423.448)
Ekspektasi Hasil        : Rp              0
Kurtailmen/Penyelesaian : (Rp  2.490.539.640)
─────────────────────────────────────────
Total Beban Laba Rugi   : Rp 12.979.476.306
```

### **Prinsip Pengakuan**

**Pengakuan Langsung di Laba Rugi:**

- **Biaya jasa kini:** Selalu diakui langsung
- **Biaya jasa lalu:** Diakui langsung ketika vested
- **Biaya bunga:** Berdasarkan tingkat diskonto × PVDBO awal
- **Kurtailmen/penyelesaian:** Ketika transaksi terjadi

**Alternatif untuk SAK ETAP:** _Catatan: Untuk entitas yang menggunakan SAK ETAP, ada pilihan untuk mengakui keuntungan/kerugian aktuaria di Laba Rugi atau di Ekuitas/OCI._

## **Other Comprehensive Income (OCI)**

OCI digunakan untuk menampung perubahan jangka panjang yang tidak memengaruhi laba rugi tahun berjalan, tapi tetap memengaruhi posisi ekuitas perusahaan.

### **Komponen OCI**

**Komponen Utama:**

- **Keuntungan/kerugian aktuaria pada kewajiban**
- **Keuntungan/kerugian aktuaria pada Aset Program**
- **Keuntungan/Kerugian aktuaria pada Dampak Batas Atas Aset**

### **Sumber Umum OCI**

|**Sebab**|**Penjelasan**|**Contoh**|
|---|---|---|
|**Keuntungan/Kerugian Aktuaria**|1. Pengalaman penyesuaian (dampak dari perbedaan antara asumsi aktuarial awal dengan apa yang secara aktual terjadi)<br>2. Perubahan asumsi keuangan dan demografi (tingkat diskonto, tingkat gaji, mortalitas, pengunduran diri)|Tingkat diskonto berubah dari 7% ke 6,5% → Kenaikan kewajiban → Kerugian di OCI|
|**Selisih Hasil Investasi**|Perbedaan antara hasil investasi aktual dan estimasi dari aset program pensiun|Ekspektasi return 8%, aktual return 6% → Kerugian di OCI|

### **Contoh Perhitungan OCI**

**Studi Kasus: PT ABC Manufacturing**

|**Komponen**|**Jumlah (Rp)**|**Penjelasan**|
|---|---|---|
|**Saldo Awal OCI**|(31.372.832.000)|Akumulasi keuntungan/kerugian dari tahun sebelumnya|
|**Kerugian Aktuaria Tahun Ini**|2.552.655.552|Pengalaman tidak menguntungkan dan perubahan asumsi|
|**Keuntungan/Kerugian Aset**|0|Tidak ada aset program|
|**Total OCI Tahun Ini**|2.552.655.552|Kerugian aktuaria bersih untuk tahun ini|
|**Saldo Akhir OCI**|(28.820.176.448)|Posisi membaik (kurang negatif)|

**Interpretasi:**

- **OCI Negatif** menunjukkan akumulasi keuntungan aktuaria (menguntungkan)
- **Pergerakan Positif** (kurang negatif) menunjukkan posisi pendanaan membaik
- **Tidak Ada Aset Program** berarti OCI hanya mencerminkan perubahan dari sisi kewajiban

## **Faktor yang Memengaruhi Nilai Kewajiban & OCI**

### **Penggerak Kewajiban**

| **Komponen**                      | **Penjelasan**                                                                        | **Arah**          |
| --------------------------------- | ------------------------------------------------------------------------------------- | ----------------- |
| **Jumlah Karyawan Bertambah**     | PVDBO dan CSC meningkat karena bertambahnya peserta yang dihitung                     | ↑ **Naik**        |
| **Kenaikan Gaji Aktual > Asumsi** | Nilai manfaat pensiun naik → PVDBO dan CSC ikut naik → memicu kenaikan OCI            | ↑ **Naik**        |
| **Mendekati Usia Pensiun**        | Penyesuaian atas kewajiban karena waktu berlalu (diskonto)                            | ↑ **Naik**        |
| **Hasil Aset Program**            | Penghasilan dari aset program pensiun, mengurangi beban dan mengurangi OCI (jika ada) | ↓ **Turun**       |
| **Kurtailmen/Penyelesaian**       | Pengurangan kewajiban akibat pemutusan hubungan kerja massal atau penghentian program | ↓ **Turun**       |
| **Perubahan Asumsi Aktuaria**     | Perubahan tingkat diskonto, gaji, mortalitas, dan lainnya → memengaruhi nilai OCI     | ↑↓ **Bervariasi** |
| **Perbedaan Data Karyawan**       | Adanya koreksi dari data karyawan perusahaan itu sendiri                              | ↑↓ **Bervariasi** |

### **Contoh Praktis**

**Skenario 1: Pertumbuhan Karyawan**

```
Karyawan: 499 → 600 orang (+20%)
Rata-rata Gaji: Rp 5M → Rp 6M (+20%)
Ekspektasi Dampak: Kenaikan PVDBO ~40-50%
```

**Skenario 2: Perubahan Asumsi**

```
Tingkat Diskonto: 7,0% → 6,5% (-0,5%)
Kenaikan Gaji: 8,0% → 9,0% (+1,0%)
Dampak Kombinasi: Kenaikan PVDBO ~18-25%
```

## **Analisis Sensitivitas**

Analisis ini bertujuan untuk membantu perusahaan melihat seberapa besar dampak perubahan kecil dalam asumsi terhadap hasil perhitungan.

### **Uji Sensitivitas Standar**

| **Komponen**                       | **Skenario Uji**                          | **Dampak Tipikal**             |
| ---------------------------------- | ----------------------------------------- | ------------------------------ |
| **Tingkat diskonto naik 1%**       | Selisih = PVDBO (naik 1%) – PVDBO aktual  | **-8% hingga -12%** kewajiban  |
| **Tingkat diskonto turun 1%**      | Selisih = PVDBO (turun 1%) – PVDBO aktual | **+10% hingga +15%** kewajiban |
| **Tingkat kenaikan gaji naik 1%**  | Selisih = CSC (naik 1%) – CSC aktual      | **+12% hingga +18%** biaya     |
| **Tingkat kenaikan gaji turun 1%** | Selisih = CSC (turun 1%) – CSC aktual     | **-10% hingga -14%** biaya     |

### **Contoh Analisis Sensitivitas**

**Baseline:** PVDBO = Rp 86,69 miliar

|**Perubahan Asumsi**|**PVDBO Baru (Rp miliar)**|**Dampak (Rp miliar)**|**Persentase**|
|---|---|---|---|
|**Tingkat Diskonto +1%**|77,42|-9,27|**-10,7%**|
|**Tingkat Diskonto -1%**|97,59|+10,90|**+12,6%**|
|**Kenaikan Gaji +1%**|97,28|+10,59|**+12,2%**|
|**Kenaikan Gaji -1%**|77,51|-9,18|**-10,6%**|

**Implikasi Bisnis:**

- **Sensitivitas tinggi** terhadap asumsi keuangan
- **Manajemen risiko** diperlukan untuk asumsi yang volatil
- **Strategi hedging** mungkin tepat
- **Pemantauan berkala** sangat penting


## 📝 **Ringkasan**

### **Key Takeaways**

- **Pengakuan Dua Bagian:** Laba rugi untuk biaya terkait layanan, OCI untuk volatilitas aktuaria
- **Dampak Langsung:** CSC, bunga, dan biaya jasa lalu langsung ke laba rugi
- **Buffer Ekuitas:** OCI menyediakan buffer volatilitas untuk perubahan asumsi
- **Alat Manajemen:** Analisis sensitivitas penting untuk manajemen risiko

### **Financial Statement Impact Summary**

| **Laporan**   | **Komponen**                 | **Fokus Manajemen**                    |
| ------------- | ---------------------------- | -------------------------------------- |
| **Laba Rugi** | CSC, Bunga, Jasa Lalu        | Penganggaran tahunan dan kontrol biaya |
| **OCI**       | Keuntungan/kerugian aktuaria | Pemantauan tren jangka panjang         |
| **Neraca**    | Posisi kewajiban bersih      | Alokasi modal dan pendanaan            |
| **Arus Kas**  | Pembayaran imbalan           | Perencanaan likuiditas                 |

### **Critical Success Factors**

- **Metodologi yang jelas** untuk penetapan asumsi
- **Dokumentasi yang kuat** untuk pertahanan audit
- **Pemantauan berkala** metrik kunci dan sensitivitas
- **Komunikasi efektif** dengan stakeholder

### **Next Steps

Setelah memahami penyajian laporan keuangan, langkah berikutnya adalah mempelajari proses valuasi aktuaria secara end-to-end dan penyusunan laporan aktuaria yang komprehensif.

---

**Navigasi:** ⬅️ [Asumsi Aktuaria](02d_asumsi_aktuaria.md) | [📋 Daftar Isi](README.md) | [Proses Valuasi](02f_proses_valuasi.md) ➡️