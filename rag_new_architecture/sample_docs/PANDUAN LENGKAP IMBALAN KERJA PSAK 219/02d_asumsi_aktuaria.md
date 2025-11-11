---
keywords:
  - asumsi aktuaria
  - tingkat diskonto
  - kenaikan gaji
  - mortalitas
  - withdrawal rate
  - TMI
  - discount rate
  - salary increase
  - sains aktuaria
  - matematis aktuaria
difficulty: intermediate
estimated_reading: 20 minutes
target_audience:
  - finance
  - actuaries
  - consultants
document_type: technical_guide
last_updated: 2025-07-28
version: "2025.1"

# SCOPE CONTROL - AKTUARIA FOCUS
domain: aktuaria
scope: sains_aktuaria_asumsi
exclude_domains:
  - asuransi_umum
  - perbankan_komersial
  - insurance_underwriting
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
  - asumsi aktuaria
  - tingkat diskonto
  - kenaikan gaji
  - mortalitas

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
# Asumsi Aktuaria

**📖 Bagian dari:** [PANDUAN LENGKAP IMBALAN KERJA PSAK 219](README.md)

---
## 🎯 **Tujuan Pembelajaran**

Setelah membaca bab ini, pembaca akan mampu:

- [ ]  Memahami dan menerapkan asumsi aktuaria yang tepat dalam perhitungan
- [ ]  Membedakan asumsi demografis dan keuangan serta dampaknya
- [ ]  Melakukan analisis sensitivitas untuk mengukur dampak perubahan asumsi
- [ ]  Menetapkan asumsi berdasarkan data historis dan kondisi pasar

---
## ⚡**Quick Reference**

### 👥 **Asumsi Demografis:**

- **Mortalitas:** TMI IV (Tabel Mortalitas Indonesia 2019)
- **Disability:** 5-10% dari tingkat mortalitas
- **Withdrawal:** Bervariasi berdasarkan usia (2-8% annually)

### 💰 **Asumsi Keuangan:**

- **Tingkat Diskonto:** Berdasarkan obligasi pemerintah (PHEI)
- **Kenaikan Gaji:** Historis + inflasi + kebijakan perusahaan
- **Tingkat Investasi:** 6-12% tergantung portofolio investasi

### 📊 **Dampak Sensitivitas:**

- **Tingkat Diskonto ±1%:** Dampak kewajiban ±10-15%
- **Kenaikan Gaji ±1%:** Dampak kewajiban ±12-18%
- **Perubahan Mortalitas:** Dampak kewajiban ±2-5%

---

Dalam perhitungan kewajiban imbalan kerja menurut PSAK 219, pemilihan asumsi aktuaria sangat penting. Meskipun data karyawan bersifat faktual, perhitungan tetap membutuhkan **proyeksi ke masa depan** seperti berapa lama karyawan akan bekerja, berapa besar gaji mereka akan naik, hingga peluang karyawan meninggal dunia atau mengundurkan diri sebelum pensiun.

# #asumsi-aktuaria

Asumsi aktuaria terbagi dua kategori utama:

- **Asumsi Demografi** (berkaitan dengan kondisi karyawan)
- **Asumsi Keuangan** (berkaitan dengan faktor ekonomi)

Keduanya harus sesuai kondisi pasar untuk menghasilkan proyeksi yang masuk akal dan dapat dipertanggungjawabkan kepada auditor dan stakeholder.

## **Asumsi Demografi**

### **1. Tingkat Kematian / Mortalitas**

**Definition:** Menunjukkan kemungkinan karyawan meninggal dunia. Ini penting, terutama untuk program pensiun seumur hidup dan perhitungan probabilitas pencapaian usia pensiun.

**Common Standards:**

- **TMI IV (2019):** Tabel Mortalitas Indonesia terbaru
- **TMI III (2011):** Versi sebelumnya, masih digunakan beberapa perusahaan
- **International Tables:** WHO atau other country tables untuk benchmarking

**Analisis Dampak:**

- **Tingkat mortalitas tinggi → Kewajiban rendah** (periode pembayaran lebih pendek)
- **Tingkat mortalitas rendah → Kewajiban tinggi** (periode pembayaran lebih panjang)

**Best Practice:**

```
Penetapan Tingkat Mortalitas:
- Gunakan tabel TMI terbaru yang tersedia
- Pertimbangkan pengalaman perusahaan jika data memadai
- Bandingkan dengan rekan industri
- Dokumentasikan alasan pemilihan
```

### **2. Tingkat Cacat / Sakit Berkepanjangan**

**Definition:** Menggambarkan kemungkinan karyawan tidak bisa bekerja lagi karena sakit atau cacat. Ini memengaruhi hak atas imbalan disability benefits.

**Common Practice:**

- **Standard Range:** 5% – 10% dari asumsi mortalitas
- **Industry Adjustment:** Manufacturing vs. office work
- **Age-based Scaling:** Higher rates for older employees

**Analisis Dampak:**

- **Tingkat cacat tinggi → Kewajiban tinggi** (pembayaran manfaat tambahan)
- **Tingkat cacat rendah → Kewajiban rendah** (klaim cacat lebih sedikit)

**Example Setting:**

```
Kelompok Usia    | Tingkat Cacat
20-30 tahun      | 5% dari mortalitas
31-40 tahun      | 7% dari mortalitas  
41-50 tahun      | 10% dari mortalitas
51+ tahun        | 12% dari mortalitas
```

### **3. Tingkat Pengunduran Diri (Turnover / Withdrawal Rate)**

**Definition:** Mengukur berapa banyak karyawan yang kemungkinan keluar sebelum pensiun. Digunakan untuk menghitung siapa saja yang berpotensi menerima imbalan.

**Pola Umum Berdasarkan Usia:**

- **Young employees (20-30):** 5-8% annual withdrawal
- **Mid-career (30-45):** 2-4% annual withdrawal
- **Senior (45-55):** 1-2% annual withdrawal
- **Pre-retirement (55+):** 0-1% annual withdrawal

**Industry Variations:**

- **Tech/Startup:** Higher withdrawal rates
- **Government/State-owned:** Lower withdrawal rates
- **Manufacturing:** Moderate withdrawal rates

**Analisis Dampak:**

- **Tingkat pengunduran diri tinggi → Kewajiban rendah** (lebih sedikit karyawan mencapai pensiun)
- **Tingkat pengunduran diri rendah → Kewajiban tinggi** (lebih banyak karyawan vested)

**Data-Driven Approach:**

```
Analisis Historis (3-5 tahun):
1. Hitung turnover aktual per kelompok usia
2. Sesuaikan dengan siklus ekonomi
3. Pertimbangkan tren industri
4. Terapkan smoothing untuk volatilitas
```

## **Asumsi Keuangan**

### **4. Tingkat Diskonto (Discount Rate)**

**Definition:** Suku bunga yang digunakan untuk menghitung nilai kini kewajiban yang akan dibayar di masa depan. Merupakan asumsi paling sensitif dalam perhitungan aktuaria.

**Reference Sources:**

- **Obligasi Pemerintah:** Yield obligasi pemerintah Indonesia
- **Obligasi Korporasi:** Yield obligasi korporasi berkualitas tinggi
- **Data PHEI:** PT Penilai Harga Efek Indonesia tingkat spot bulanan
- **Bank Indonesia:** Suku bunga acuan dan yield curve 

**Common Range:** 6.5% - 8.5% annually (tergantung kondisi ekonomi)

**Analisis Dampak:**

- **Tingkat diskonto tinggi → Kewajiban rendah** (diskonto nilai sekarang lebih tinggi)
- **Tingkat diskonto rendah → Kewajiban tinggi** (diskonto nilai sekarang lebih rendah)

**Market-Based Setting:**

```
Metodologi Tingkat Diskonto:
1. Peroleh yield obligasi pemerintah 10-15 tahun
2. Tambahkan spread kredit jika menggunakan obligasi korporasi
3. Pertimbangkan pencocokan durasi dengan kewajiban
4. Terapkan pertimbangan profesional untuk kewajaran
```

### **5. Asumsi Kenaikan Gaji**

**Definition:** Menyesuaikan proyeksi kenaikan gaji karyawan untuk menghitung manfaat pensiun masa depan yang akan dibayarkan.

**Faktor yang Dipertimbangkan:**

- **Data Historis Perusahaan:** Rata-rata kenaikan 3-5 tahun terakhir
- **Upah Minimum Regional:** Kenaikan UMK/UMP di wilayah operasional
- **Tingkat Inflasi:** Tingkat inflasi nasional dan proyeksi
- **Kebijakan Perusahaan:** Rencana kenaikan berdasarkan surat manajemen
- **Benchmark Industri:** Kenaikan gaji sektor sejenis

**Common Range:** 5% - 12% annually

**Analisis Dampak:**

- **Kenaikan gaji tinggi → Kewajiban tinggi** (manfaat masa depan lebih besar)
- **Kenaikan gaji rendah → Kewajiban rendah** (manfaat masa depan lebih kecil)

**Structured Approach:**

```
Komponen Pertumbuhan Gaji:
- Inflasi dasar: 3-4%
- Pertumbuhan upah riil: 1-3%  
- Kenaikan promosi/merit: 1-2%
- Penyesuaian khusus perusahaan: 0-2%
Total: 5-11% per tahun
```

### **6. Tingkat Return On Investment (ROI)**

**Definition:** Untuk program yang memiliki aset dana (seperti DPLK), ROI menunjukkan berapa besar hasil investasi yang bisa digunakan untuk membayar kewajiban di masa depan.

**Common Range:** 6-12% annually depending on investment portfolio

**Pertimbangan Portofolio:**

- **Konservatif (Dominan obligasi):** 6-8% ekspektasi return
- **Seimbang (Campuran):** 8-10% ekspektasi return
- **Agresif (Dominan saham):** 10-12% ekspektasi return

**Analisis Dampak:**

- **ROI tinggi → Kewajiban rendah** (pertumbuhan aset lebih baik mengimbangi kewajiban)
- **ROI rendah → Kewajiban tinggi** (pertumbuhan aset tidak memadai)

### **7. Hasil Investasi Aset Program (Assets Return)**

**Definition:** Asumsi penting untuk menilai seberapa besar aset program imbalan kerja bisa berkembang dan mengurangi kewajiban perusahaan.

**Considerations:**

- **Strategi Investasi:** Alokasi aset dari dana pensiun
- **Kondisi Pasar:** Outlook ekonomi dan volatilitas pasar
- **Toleransi Risiko:** Nafsu risiko manajemen dana
- **Kinerja Benchmark:** Perbandingan dengan indeks pasar sejenis

**Professional Management:**

```
Metodologi Return Aset:
1. Analisis kinerja dana historis
2. Pertimbangkan kondisi pasar saat ini
3. Terapkan penyesuaian risiko
4. Benchmark dengan dana sejenis
5. Dokumentasikan asumsi dengan jelas
```

## **Dampak Perubahan Asumsi Aktuaria**

### **Sensitivity Analysis Table**

|**Asumsi**|**Jika Naik maka**|**Jika Turun maka**|**Sensitivity Level**|
|:--|:--|:--|:--|
|**Tingkat Diskonto**|Kewajiban **turun**|Kewajiban **naik**|**High**|
|**Kenaikan Gaji**|Kewajiban **naik**|Kewajiban **turun**|**High**|
|**Umur Hidup (Mortalitas)**|Kewajiban **naik** _(karena manfaat dibayar lebih lama)_|Kewajiban **turun**|**Medium**|
|**Withdrawal (Resign)**|Kewajiban **turun** _(karena lebih banyak yang keluar)_|Kewajiban **naik**|**Medium**|
|**Usia Pensiun**|Kewajiban **turun** _(karena dibayar lebih lambat)_|Kewajiban **naik**|**Low**|


---

## 📝 **Ringkasan**

### **Key Takeaways**

- **Two Categories:** Asumsi demografis (employee behavior) dan keuangan (economic factors)
- **High Sensitivity:** Discount rate dan salary growth paling berdampak pada hasil
- **Data-Driven:** Asumsi harus berdasarkan data historis dan kondisi pasar terkini
- **Regular Review:** Annual review dengan dokumentasi yang comprehensive

### **Critical Success Factors**

|**Faktor**|**Praktik Terbaik**|**Dampak**|
|---|---|---|
|**Kualitas Data**|Gunakan data historis 3-5 tahun|Asumsi lebih akurat|
|**Keselarasan Pasar**|Benchmark dengan suku bunga saat ini|Asumsi dapat dipertahankan dalam audit|
|**Dokumentasi**|Memorandum asumsi komprehensif|Proses audit lancar|
|**Analisis Sensitivitas**|Uji perubahan ±1%|Kesadaran risiko|

### **Kesalahan Umum yang Harus Dihindari**

- **Menggunakan asumsi usang** dari tahun sebelumnya tanpa tinjauan
- **Mengabaikan pengalaman khusus perusahaan** demi asumsi generik
- **Dokumentasi buruk** dari alasan asumsi
- **Analisis sensitivitas tidak memadai** untuk asumsi kunci

### **Next Steps**

Setelah memahami asumsi aktuaria, langkah berikutnya adalah mempelajari bagaimana hasil perhitungan disajikan dalam laporan keuangan sesuai dengan PSAK 219.

---

**Navigasi:** ⬅️ [Metode PUC](02c_metode_puc.md) | [📋 Daftar Isi](README.md) | [Laporan Keuangan](02e_laporan_keuangan.md) ➡️