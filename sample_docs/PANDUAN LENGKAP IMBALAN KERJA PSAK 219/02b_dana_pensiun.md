---
keywords:
  - dana pensiun
  - DPLK
  - DPPK
  - asuransi pensiun
  - UUK
  - UUCK
  - program pensiun
  - funding status
  - aktuaria
  - sains aktuaria
  - imbalan kerja
difficulty: intermediate
estimated_reading: 25 minutes
target_audience:
  - finance
  - management
  - hr
  - actuaries
document_type: technical_guide
last_updated: 2025-07-28
version: "2025.1"
related_regulations:
  - UU_11_1992
  - PSAK_219
  - UUK
  - UUCK

# SCOPE CONTROL - AKTUARIA FOCUS
domain: aktuaria
scope: sains_aktuaria_dana_pensiun
exclude_domains:
  - asuransi_umum
  - perbankan_komersial
  - insurance_products
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
  - dana pensiun
  - program pensiun
  - DPLK
  - DPPK

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
# Program Dana Pensiun

**📖 Bagian dari:** [PANDUAN LENGKAP IMBALAN KERJA PSAK 219](README.md)

---
## 🎯 **Tujuan Pembelajaran**

Setelah membaca bab ini, pembaca akan mampu:

- [ ]  Memahami berbagai jenis program Dana Pensiun dan karakteristiknya
- [ ]  Menjelaskan dampak program Dana Pensiun terhadap kewajiban PSAK 219
- [ ]  Menghitung selisih antara manfaat Dana Pensiun dengan kewajiban UUK/UUCK
- [ ]  Melakukan analisis funding status untuk berbagai skenario

---
## ⚡ **Quick Reference**

### 🏦 **Jenis Dana Pensiun:**

- **DPLK:** Dana Pensiun Lembaga Keuangan (bank/asuransi)
- **DPPK:** Dana Pensiun Pemberi Kerja (internal perusahaan)
- **Asuransi:** Dana Pensiun Lembaga Asuransi (polis khusus)

### 📊 **Impact Formula:**

- **Dana Pensiun > UUK:** Kewajiban = 0 (fully funded)
- **Dana Pensiun < UUK:** Kewajiban = UUK - Coverage
- **Tanpa Dana Pensiun:** Kewajiban = Full UUK obligation

### ⚖️ **Legal Framework:**

- **UU No. 11/1992:** Dasar regulasi Dana Pensiun
- **UUK No. 13/2003:** Formula manfaat pesangon
- **UUCK:** Formula manfaat terbaru

---

## #program-dana-pensiun

### **Legal Framework**

Dalam UU No. 11 Tahun 1992, Dana Pensiun adalah program pasca kerja yang dilakukan dengan cara mengumpulkan dana secara terpisah dari kekayaan perusahaan (pendiri). Artinya, dana ini tidak boleh menjadi bagian dari cadangan perusahaan untuk keperluan pembayaran imbalan kerja, melainkan dikelola secara mandiri. UU tersebut juga membedakan jenis Dana Pensiun, yang dapat menjadi pengurang Kewajiban Imbalan Pasca Kerja, antara lain:

1. **Dana Pensiun Lembaga Keuangan (DPLK)**
2. **Dana Pensiun Pemberi Kerja (DPPK)**  
3. **Dana Pensiun Lembaga Asuransi**

### **Perbandingan Karakteristik Skema Dana Pensiun**

| Karakteristik | **DPLK** | **DPPK** | **DP - Lembaga Asuransi** |
|:--------------|:---------|:----------|:--------------------------|
| **Pengelola** | Dikelola Lembaga Keuangan | Dikelola perusahaan tempat karyawan bekerja | Dikelola Perusahaan Asuransi |
| **Sumber Iuran** | Dari karyawan dan/atau pemberi kerja | Dari pemberi kerja, bisa ditambah kontribusi karyawan | Dari premi yang dibayarkan pemberi kerja atau karyawan |
| **Manfaat** | Berdasarkan akumulasi iuran dan hasil investasi | Berdasarkan perjanjian kerja atau kebijakan perusahaan | Berdasarkan polis asuransi yang dibeli |
| **Risiko** | Ditanggung oleh peserta (karyawan) | Ditanggung oleh pemberi kerja | Ditanggung oleh perusahaan asuransi |
| **Fleksibilitas** | Cukup fleksibel, bisa diikuti oleh berbagai perusahaan | Bergantung pada kebijakan perusahaan, biasanya hanya untuk karyawan tetap | Disesuaikan dengan polis asuransi yang disepakati |
| **Penggunaan Dana** | Setelah pensiun, sesuai ketentuan yang disepakati | Umumnya dana hanya bisa digunakan setelah pensiun atau putus hubungan kerja | Disesuaikan dengan ketentuan polis asuransi |
| **Keuntungan** | Keuntungan bergantung pada hasil investasi | Terkait dengan kesejahteraan karyawan, biasanya punya komitmen kuat dari perusahaan | Kepastian manfaat dengan risiko yang lebih rendah bagi karyawan |

### **Impact on PSAK 219 Liability**

Sekarang kita akan mempelajari ilustrasi bagaimana Manfaat pensiun ini berhubungan dengan kewajiban Imbalan Pasca Kerja, seperti yang diatur dalam UUK No. 13 Tahun 2003 sebagai berikut.

#### **Scenario 1: Dana Pensiun > Manfaat UUK**
Jika Manfaat pensiun perusahaan **lebih besar** dari Manfaat pensiun menurut UUK, maka kewajiban perusahaan **hanya sebesar Iuran Pemberi Kerja.**

**Contoh Perhitungan:**
```
(2 × 40)G × 75% > 30G  
60G > 30G ✓
```
**Result:** Kewajiban perusahaan = 0 (fully covered by pension fund)

#### **Scenario 2: Dana Pensiun < Manfaat UUK**  
Jika Manfaat pensiun perusahaan **lebih kecil** dari Manfaat pensiun menurut UUK, maka kewajiban perusahaan adalah **selisih dari Manfaat pensiun UUK dan Manfaat pensiun perusahaan.**

**Contoh Perhitungan:**
```
(1 × 15)G × 75% < 25G  
11.25G < 25G
Selisih: 25G - 11.25G = 13.75G
```
**Result:** Kewajiban perusahaan = 13.75G

_Catatan: G = Gaji_

Dalam laporan aktuaria dan audit PSAK 219, selisih inilah yang harus dicatat sebagai **liabilitas imbalan kerja,** meskipun perusahaan sudah menyelenggarakan program pensiun.

---

Seperti yang telah dibahas tentang kebutuhan Valuasi Aktuaria sebelumnya, terdapat beberapa contoh perhitungan sederhana dalam program Dana Pensiun untuk Pesangon dan imbalan jangka panjang lainnya.

## #perhitungan-aktuaria

### **Contoh Perhitungan Pesangon**

Mengenai perhitungan Pesangon, beberapa contoh kasusnya antara lain:

#### **Scenario 1: Tanpa Program Dana Pensiun**

**Case:** Perusahaan menjanjikan pembayaran Pesangon kepada karyawan A pada saat berhenti bekerja di Usia Pensiun normal sebesar Rp200.000.000,- dengan masa kerja total hingga mencapai Usia Pensiun adalah 20 tahun.

**Calculation Steps:**
1. **Unit per Period:** Rp200.000.000 ÷ 20 = Rp10.000.000,-
2. **Annual Recognition:** Based on PUC method (assumptions ignored for simplicity)

**Results:**

| Tahun | Beban Tahun Berjalan | Kewajiban pada Akhir Periode |
|:------|:---------------------|:------------------------------|
| 1 | 10.000.000 | **10.000.000** |
| 2 | 10.000.000 | **20.000.000** |
| 3 | 10.000.000 | **30.000.000** |
| ... | ... | ... |
| 20 | 10.000.000 | **200.000.000** |

#### **Scenario 2: Dengan Dana Pensiun (Insufficient Coverage)**

**Case:** Perusahaan menjanjikan pembayaran Pesangon kepada karyawan A pada saat berhenti bekerja di Usia Pensiun normal sebesar Rp200.000.000,- dengan masa kerja total hingga mencapai Usia Pensiun adalah 20 tahun.

> **Additional Info:** Perusahaan mengikuti sebuah program DPLK dengan total iuran pertahunnya sebesar 7.000.000 (asumsi investasi diabaikan)

**Calculation Steps:**
1. **Unit per Period:** Rp200.000.000 ÷ 20 = Rp10.000.000,-  
2. **Net Liability:** Total obligation minus DPLK accumulation

**Results:**

| Tahun | Beban Tahun Berjalan | Saldo DPLK Akhir Periode | Kewajiban Akhir Periode |
|:------|:---------------------|:--------------------------|:-------------------------|
| 1 | 10.000.000 | 7.000.000 | **3.000.000** |
| 2 | 10.000.000 | 14.000.000 | **6.000.000** |
| 3 | 10.000.000 | 21.000.000 | **9.000.000** |
| ... | ... | ... | ... |
| 20 | 10.000.000 | 140.000.000 | **60.000.000** |

#### **Scenario 3: Dengan Dana Pensiun (Excess Coverage)**

**Case:** Perusahaan menjanjikan pembayaran Pesangon kepada karyawan A pada saat berhenti bekerja di Usia Pensiun normal sebesar Rp200.000.000,- dengan masa kerja total hingga mencapai Usia Pensiun adalah 20 tahun.

> **Additional Info:** Perusahaan mengikuti sebuah program DPLK dengan total iuran pertahunnya sebesar 11.000.000 (asumsi investasi diabaikan)

**Calculation Steps:**
1. **Unit per Period:** Rp200.000.000 ÷ 20 = Rp10.000.000,-
2. **Coverage Assessment:** DPLK accumulation vs. total obligation

**Results:**

| Tahun | Beban Tahun Berjalan | Saldo DPLK Akhir Periode | Kewajiban Akhir Periode |
|:------|:---------------------|:--------------------------|:-------------------------|
| 1 | 10.000.000 | **11.000.000** | **0** |
| 2 | 10.000.000 | **22.000.000** | **0** |
| 3 | 10.000.000 | **33.000.000** | **0** |
| ... | ... | ... | ... |
| 20 | 10.000.000 | **220.000.000** | **0** |

### **Key Insights:**
- **Scenario 2:** Kewajiban perusahaan tetap ada karena dana DPLK belum mencukupi
- **Scenario 3:** Kewajiban perusahaan = 0 karena manfaat dari DPLK sudah menutupi seluruh Pesangon

Selain Pesangon, beralih ke imbalan jangka panjang lainnya bagi karyawan, dimana terdapat Cuti Besar (CBS) yang merupakan hak karyawan untuk mengambil cuti panjang setelah masa kerja tertentu dan wajib dihitung secara aktuaria.

### **Contoh Perhitungan Imbalan Jangka Panjang Lainnya**

Perhitungannya didasarkan pada Asumsi, seperti **tingkat pengambilan cuti, gaji saat cuti,** dan Tingkat Diskonto, yang dapat memengaruhi arus kas perusahaan. Berikut adalah contoh perhitungan imbalan jangka panjang terkait CBS tersebut.

#### **Long-term Benefit Calculation Example**

**Case:** Perusahaan menjanjikan pembayaran uang cuti besar kepada setiap karyawannya bagi karyawan yang telah mencapai masa kerja 10 tahun sebesar Rp10.000.000,-

**Calculation Steps:**
1. **Unit per Period:** Rp10.000.000 ÷ 10 = Rp1.000.000,-
2. **Annual Accrual:** Based on PUC method (assumptions ignored for simplicity)

**Results:**

| Tahun | Beban Tahun Berjalan | Kewajiban pada Akhir Periode |
|:------|:---------------------|:------------------------------|
| 1 | 1.000.000 | 1.000.000 |
| 2 | 1.000.000 | 2.000.000 |
| 3 | 1.000.000 | 3.000.000 |
| ... | ... | ... |

---
## 📝 **Ringkasan**

### **Key Takeaways**

- **Legal Separation:** Dana Pensiun must be legally separated from company assets
- **Risk Allocation:** Different pension types allocate risks differently
- **PSAK 219 Impact:** Pension plans can significantly reduce PSAK 219 liability
- **Strategic Choice:** Selection depends on company size, risk appetite, dan financial capacity

### **Practical Framework**

|**Company Profile**|**Recommended Type**|**Key Benefits**|
|---|---|---|
|**SME**|DPLK|Low administrative burden|
|**Large Corp**|DPPK atau combination|Full control dan optimization|
|**Risk-Averse**|Asuransi|Predictable costs|

### **Next Steps**

Setelah memahami program Dana Pensiun, langkah berikutnya adalah mempelajari metode perhitungan aktuaria yang digunakan untuk mengukur kewajiban imbalan kerja.

---

**Navigasi:** ⬅️ [Konsultan Aktuaria](02a_konsultan_aktuaria.md) | [📋 Daftar Isi](README.md) | [Metode PUC](02c_metode_puc.md) ➡️