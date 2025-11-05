---
keywords:
  - teknis perhitungan
  - aktuaria
  - projected unit credit
  - PUC
  - PVDBO
  - asumsi aktuaria
  - valuasi aktuaria
  - konsultan aktuaria
  - current service cost
  - past service cost
  - sains aktuaria
  - matematis aktuaria
difficulty: intermediate
estimated_reading: 120 minutes
target_audience:
  - actuaries
  - finance
  - consultants
  - technical_staff
document_type: technical_guide
last_updated: 2025-07-28
version: "2025.1"
related_regulations:
  - PSAK_219
  - IFRS_19

# SCOPE CONTROL - AKTUARIA FOCUS
domain: aktuaria
scope: sains_aktuaria_perhitungan
exclude_domains:
  - asuransi_umum
  - perbankan_komersial
  - insurance_reserves
  - claim_accounting
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
  - teknis perhitungan aktuaria
  - konsultan aktuaria

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
# BAB 2: Teknis dalam Perhitungan Aktuaria

**📖 Bagian dari:** [PANDUAN LENGKAP IMBALAN KERJA PSAK 219](README.md)

> **SCOPE DELIMITER**: Dokumen ini KHUSUS membahas aspek aktuaria imbalan kerja dan employment benefits. 

---

## 🎯 **Tujuan Pembelajaran**

Setelah membaca bab ini, pembaca akan mampu:

- [ ] Memahami peran dan tanggung jawab Konsultan Aktuaria dalam implementasi PSAK 219
- [ ] Menjelaskan berbagai jenis program Dana Pensiun dan karakteristiknya
- [ ] Menguasai metode Projected Unit Credit (PUC) untuk perhitungan kewajiban imbalan kerja
- [ ] Memahami dan menerapkan asumsi aktuaria yang tepat dalam perhitungan
- [ ] Melakukan penyajian hasil perhitungan aktuaria dalam laporan keuangan

---

## ⚡ **Quick Reference**
### 🔤 **Key Terms & Formulas**

**PVDBO (Present Value of Defined Benefit Obligation):**
- **Definition:** Nilai kini kewajiban berdasarkan masa kerja lalu
- **Formula:** PVDBO = [PVFB ÷ Total masa kerja] × Masa kerja lalu

**CSC (Current Service Cost):**
- **Definition:** Biaya jasa periode berjalan  
- **Formula:** CSC = PVFB ÷ Total masa kerja hingga pensiun

**PUC Method (Projected Unit Credit):**
- **Definition:** Metode standar untuk perhitungan aktuaria PSAK 219
- **Purpose:** Mengalokasikan manfaat secara proporsional sepanjang masa kerja

**PVFB (Present Value of Future Benefits):**
- **Definition:** Total nilai kini semua manfaat masa depan
- **Relationship:** PVFB = PVDBO + Future Service Cost

### 📊 **Essential Calculations**

| **Component** | **Formula** | **Purpose** |
|:--------------|:------------|:------------|
| **Unit per Tahun** | PVFB ÷ Total masa kerja hingga pensiun | Alokasi tahunan manfaat |
| **PVDBO** | PVFB × (Masa kerja lalu ÷ Total masa kerja) | Kewajiban saat ini |
| **CSC** | PVFB ÷ Total masa kerja hingga pensiun | Beban tahun berjalan |
| **Interest Cost** | PVDBO awal tahun × Tingkat diskonto | Biaya bunga |

### 🏦 **Types of Pension Funds**

| **Type** | **Manager** | **Risk Bearer** | **Flexibility** |
|:---------|:------------|:----------------|:----------------|
| **DPLK** | Lembaga Keuangan | Peserta | High |
| **DPPK** | Perusahaan Internal | Pemberi Kerja | Medium |
| **Asuransi** | Perusahaan Asuransi | Asuransi | Low |

### ⚖️ **Key Regulation Formulas**

**UUK No. 13/2003:**
- **Normal Retirement:** (2×Severance + 1×Service) × 115%
- **Death Benefit:** (2×Severance + 1×Service) × 115%
- **Voluntary Resign:** (1×Severance + 1×Service) × 115% + Separation Pay

**UU Cipta Kerja (UUCK):**
- **Normal Retirement:** (1.75×Severance + 1×Service) + Compensation
- **Death Benefit:** (2×Severance + 1×Service) + Compensation
- **Voluntary Resign:** Compensation Pay only

**PKWT Compensation:**
- **Formula:** (masa kerja ÷ 12 bulan) × 1 bulan upah
- **Minimum:** Pro-rata berdasarkan masa kerja aktual

---

## 📋 **Chapter Overview**

**Section Topics:**
- [Peran Konsultan Aktuaria](02a_konsultan_aktuaria.md#peran-konsultan-aktuaria) - [Detail: 02a_konsultan_aktuaria.md](02a_konsultan_aktuaria.md)
- [Program Dana Pensiun](02b_program-dana-pensiun#program-dana-pensiun) - [Detail: 02b_dana_pensiun.md](02b_dana_pensiun.md)
- [Metode Projected Unit Credit](02_metode_puc.md#metode-projected-unit-credit) - [Detail: 02c_metode_puc.md](02c_metode_puc.md)
- [Asumsi Aktuaria](02d_asumsi_aktuaria.md#asumsi-aktuaria) - [Detail: 02d_asumsi_aktuaria.md](02d_asumsi_aktuaria.md)
- [Penyajian Laporan Keuangan](02e_laporan-keuangan.md#penyajian-laporan-keuangan) - [Detail: 02e_laporan_keuangan.md](02e_laporan_keuangan.md)
- [Proses Valuasi Aktuaria](#proses-valuasi-aktuaria) - [Detail: 02f_proses_valuasi.md](02f_proses_valuasi.md)

---

## 📋 **Key Concept: Konsultan Aktuaria (KKA)**

### Apa itu Konsultan Aktuaria?

**Definition:** Konsultan aktuaria membantu perusahaan menghitung dan mengelola kewajiban jangka panjang seperti pesangon dan pensiun.

**Legal Requirements:**
- **KKA License:** Wajib berizin resmi dan terdaftar di AKKAI
- **Professional Certification:** Anggota Asosiasi Kantor Konsultan Aktuaria Indonesia
- **Compliance:** Memastikan hasil sesuai PSAK 219

### Lingkup Tanggung Jawab KKA

| **Area** | **Responsibilities** | **Deliverables** |
|:---------|:---------------------|:-----------------|
| **Data Management** | Validasi data karyawan, verifikasi kebijakan | Dataset lengkap dan akurat |
| **Calculation** | Implementasi metode PUC, setting asumsi | Hasil perhitungan PVDBO dan CSC |
| **Communication** | Edukasi stakeholder, audit support | Laporan aktuaria dan dokumentasi |

**👉 [Baca detail lengkap: Peran Konsultan Aktuaria](02a_konsultan_aktuaria.md)**

---

## 📋 **Key Concept: Program Dana Pensiun**

### Jenis Program Dana Pensiun

**Legal Framework:** UU No. 11 Tahun 1992 - Dana terpisah dari kekayaan perusahaan.

| **Type** | **DPLK** | **DPPK** | **Asuransi** |
|:---------|:---------|:----------|:-------------|
| **Pengelola** | Bank/Lembaga Keuangan | Internal Perusahaan | Perusahaan Asuransi |
| **Risk Bearer** | Peserta | Pemberi Kerja | Asuransi |
| **Funding Source** | Karyawan ± Perusahaan | Perusahaan | Premi |

### Impact on PSAK 219 Liability

**Scenario Analysis:**

1. **Dana Pensiun > Manfaat UUK** → Kewajiban perusahaan = 0
2. **Dana Pensiun < Manfaat UUK** → Kewajiban = Selisih
3. **No Pension Fund** → Kewajiban = Full benefit amount

**👉 [Baca detail lengkap: Program Dana Pensiun](02b_dana_pensiun.md)**

---

## 📋 **Key Concept: Metode Projected Unit Credit (PUC)**

### Apa itu Metode PUC?

**Definition:** Metode aktuaria standar untuk mengalokasikan imbalan kerja secara proporsional sepanjang masa kerja karyawan.

**Core Purpose:** Membagi manfaat ke periode berjalan dan periode masa lalu untuk menentukan CSC dan PVDBO.

### Formula Relationship

**Total Future Benefits:**
```
PVFB = Total nilai kini semua manfaat masa depan
```

**Present Value of Obligation:**
```
PVDBO = PVFB × (Masa kerja lalu ÷ Total masa kerja)
```

**Current Service Cost:**
```
CSC = PVFB ÷ Total masa kerja hingga pensiun
```

### Key Components

| **Component** | **Definition** | **Financial Statement Impact** |
|:--------------|:---------------|:------------------------------|
| **PVDBO** | Kewajiban berdasarkan masa kerja lalu | Balance Sheet (Liability) |
| **CSC** | Biaya jasa periode berjalan | P&L (Expense) |
| **Interest Cost** | Time value impact | P&L (Expense) |
| **Past Service Cost** | Prior period adjustments | P&L (Expense/Income) |

**👉 [Baca detail lengkap: Metode PUC](02c_metode_puc.md)**

---

## 📋 **Key Concept: Asumsi Aktuaria**

### Jenis Asumsi Aktuaria

**Demographic Assumptions:**
- **Mortality Rate:** Probabilitas kematian (TMI IV standard)
- **Disability Rate:** Probabilitas cacat (5-10% dari mortality)
- **Withdrawal Rate:** Probabilitas resign (varies by age)

**Financial Assumptions:**
- **Discount Rate:** Tingkat diskonto (bond yield reference)
- **Salary Growth:** Proyeksi kenaikan gaji (historical + inflation)
- **ROI:** Return on investment (6-12% range)

### Dampak Sensitivitas

| **Assumption** | **If Increases** | **If Decreases** | **Sensitivity** |
|:---------------|:-----------------|:-----------------|:----------------|
| **Discount Rate** | Liability ↓ | Liability ↑ | **High** |
| **Salary Growth** | Liability ↑ | Liability ↓ | **High** |
| **Mortality** | Liability ↑ | Liability ↓ | **Medium** |
| **Withdrawal** | Liability ↓ | Liability ↑ | **Medium** |

**👉 [Baca detail lengkap: Asumsi Aktuaria](02d_asumsi_aktuaria.md)**

---

## 📋 **Key Concept: Penyajian Laporan Keuangan**

### Komponen Laporan Laba Rugi

| **Component**                  | **Impact**       | **Nature** |
| :----------------------------- | :--------------- | :--------- |
| **Current Service Cost (CSC)** | Increase Expense | Recurring  |
| **Interest Cost**              | Increase Expense | Recurring  |
| **Past Service Cost**          | Variable         | One-time   |
| **Return on Plan Assets**      | Decrease Expense | Variable   |

### Other Comprehensive Income (OCI)

**OCI Components:**
- Actuarial gains/losses on obligation
- Actuarial gains/losses on plan assets
- Asset ceiling effect

**Common OCI Sources:**
- Changes in discount rate
- Changes in salary assumptions
- Experience adjustments
- Investment return differences

**👉 [Baca detail lengkap: Penyajian Laporan Keuangan](02e_laporan_keuangan.md)**

---

## 📋 **Key Concept: Proses Valuasi Aktuaria**

### Standard Valuation Process

**Step 1: Data Preparation**
- Employee database validation
- Benefit policy confirmation
- Historical data analysis

**Step 2: Assumption Setting**
- Benchmark analysis
- Market condition review
- Management input incorporation

**Step 3: Calculation & Reporting**
- PUC method application
- Result analysis and validation
- Report preparation

### Standard Report Tables

| **Table** | **Content** | **Purpose** |
|:----------|:------------|:------------|
| **Table 1** | Data summary & assumptions | Overview |
| **Table 2** | Actuarial gain/loss analysis | Variance explanation |
| **Table 3** | OCI movements | Equity impact |
| **Table 4** | Balance sheet presentation | Financial position |
| **Table 5** | P&L recognition | Income statement |

**👉 [Baca detail lengkap: Proses Valuasi](02f_proses_valuasi.md)**

---

## 🔍 **Frequently Asked Questions**

### Kapan Harus Menggunakan Konsultan Aktuaria?
- Perusahaan dengan >50 karyawan
- Benefit program yang kompleks
- Audit requirement compliance
- Annual valuation needs

### Bagaimana Menghitung PVDBO?
1. Calculate PVFB (total future benefits)
2. Determine service ratio (past/total service)
3. Apply formula: PVDBO = PVFB × service ratio

### Apa Perbedaan DPLK dan DPPK?
- **DPLK:** Dikelola bank, risk pada peserta
- **DPPK:** Dikelola perusahaan, risk pada employer

### Mengapa Asumsi Aktuaria Penting?
- Menentukan akurasi perhitungan
- Mempengaruhi volatilitas hasil
- Impact pada financial statement
- Audit compliance requirement

---

## 📝 **Chapter Summary**

### Key Takeaways

1. **KKA Role:** Essential for accurate calculation and audit compliance
2. **Pension Programs:** Three types with different risk allocations
3. **PUC Method:** Standard approach for benefit allocation over service period
4. **Assumptions:** Critical for calculation accuracy and result stability
5. **Financial Reporting:** Structured presentation in P&L and OCI

### Implementation Framework

| **Phase** | **Key Activities** | **Deliverables** |
|:----------|:-------------------|:-----------------|
| **Preparation** | Data collection, assumption setting | Validated dataset |
| **Calculation** | PUC method application | PVDBO, CSC values |
| **Analysis** | Variance analysis, sensitivity testing | Actuarial report |
| **Presentation** | Financial statement preparation | Audit-ready numbers |

### Success Factors

- **Quality Data:** Complete and accurate employee information
- **Realistic Assumptions:** Market-based and historically validated
- **Regular Reviews:** Annual assumption updates
- **Documentation:** Comprehensive audit trail
- **Stakeholder Communication:** Clear explanation of methodology

---

## 🔄 **Navigation & Next Steps**

**Related Documents:**
- [📖 Bab 1: Dasar PSAK 219](01_dasar_psak219.md) - Foundation concepts
- [📖 Bab 3: Implementasi](03_implementasi_teknologi.md) - Practical implementation
- [❓ FAQ Teknis](05c_faq_teknis.md) - Troubleshooting guide

**Section Details:**
- [02a: Konsultan Aktuaria](02a_konsultan_aktuaris.md)
- [02b: Dana Pensiun](02b_dana_pensiun.md)  
- [02c: Metode PUC](02c_metode_puc.md)
- [02d: Asumsi Aktuaria](02d_asumsi_aktuaria.md)
- [02e: Laporan Keuangan](02e_laporan_keuangan.md)
- [02f: Proses Valuasi](02f_proses_valuasi.md)

---

**Navigasi:** ⬅️ [Bab 1: Dasar Imbalan Pasca Kerja](01_dasar_psak219.md) | [📋 Daftar Isi](README.md) | [Bab 3: Implementasi & Teknologi](03_implementasi_teknologi.md) ➡️

---

> **💡 Note:** File ini telah dioptimasi untuk RAG learning system dengan formula yang disederhanakan, struktur yang konsisten, dan cross-reference yang jelas. Setiap section memiliki detail file terpisah untuk chunking yang optimal.