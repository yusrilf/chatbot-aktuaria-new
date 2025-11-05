---
title: "Step 1: Employee Data Foundation"
description: Age, service, and salary calculations for actuarial valuation
keywords:
  - age calculation
  - service calculation
  - salary projection
  - IFRIC method
difficulty:
  - basic
estimated_reading: 15 minutes
target_audience:
  - actuaries
  - finance_team
  - HR_professionals
document_type: calculation_step
step_number: 1
---

# 📊 Step 1: Employee Data Foundation

## 🎯 **Overview**

Langkah pertama dalam valuasi aktuaria adalah menentukan fondasi data karyawan yang akurat. Step ini mencakup perhitungan usia, masa kerja, dan proyeksi gaji yang menjadi basis semua perhitungan selanjutnya.

## 📋 **Input Data Requirements**

```json
{
  "required_fields": {
    "tanggal_lahir": "YYYY-MM-DD",
    "tanggal_masuk_kerja": "YYYY-MM-DD", 
    "tanggal_valuasi": "YYYY-MM-DD",
    "gaji_pokok": "number (IDR)",
    "usia_pensiun": "number (years)",
    "gender": "M|F"
  },
  "optional_fields": {
    "tunjangan_tetap": "number (IDR, default: 0)",
    "tingkat_kenaikan_gaji": "number (decimal, default: 0.08)"
  }
}
```

## 🧮 **Age Calculations**

### **Step 1.1: Usia Saat Valuasi**

```formula
💫 Formula: Usia Saat Valuasi
usia_saat_valuasi = (tanggal_valuasi - tanggal_lahir) / 12
```

**Komponen:**

- usia_saat_valuasi: Usia karyawan pada tanggal valuasi (tahun, 2 desimal)
- tanggal_valuasi: Tanggal penilaian (format: YYYY-MM-DD)
- tanggal_lahir: Tanggal lahir karyawan (format: YYYY-MM-DD)

**Kondisi Validasi:**

- Jika hasil ≥ usia_pensiun: gunakan usia_pensiun
- Jika hasil < 17: flag error "Below minimum working age"
- Jika hasil > 70: flag warning "Above typical working age"

**Contoh Perhitungan:**

```json
{
  "input": {
    "tanggal_lahir": "1972-11-09",
    "tanggal_valuasi": "2026-01-01"
  },
  "calculation": {
    "months_difference": 636.96,
    "unrounded_usia_saat_valuasi": 53.08,
    "usia_saat_valuasi": 53
  }
}
```

### **Step 1.2: Usia Mulai Masa Kerja**

```formula
💫 Formula: Usia Mulai Masa Kerja
usia_mulai_masa_kerja = (tanggal_masuk_kerja - tanggal_lahir) / 12
```

**Komponen:**

- usia_mulai_masa_kerja: Usia saat pertama kali bekerja (tahun, 2 desimal)
- tanggal_masuk_kerja: Tanggal mulai kerja pertama (format: YYYY-MM-DD)
- tanggal_lahir: Tanggal lahir karyawan (format: YYYY-MM-DD)

**Validasi Bisnis:**

- Minimum age: 17 tahun (legal working age)
- Maximum age: 65 tahun (unusual but possible)
- Must be < usia_saat_valuasi

**Contoh Perhitungan:**

```json
{
  "input": {
    "tanggal_lahir": "1972-11-09",
    "tanggal_masuk_kerja": "1995-07-06"
  },
  "calculation": {
    "months_difference": 270.96,
    "unrounded_usia_mulai_masa_kerja": 22.58,
    "usia_mulai_masa_kerja": 22
  }
}
```

## 🕐 **Service Period Calculations**

### **Step 1.3: Masa Kerja Lalu - Metode Standard**

```formula
💫 Formula: Masa Kerja Lalu
masa_kerja_lalu = (tanggal_valuasi - tanggal_masuk_kerja) / 12
```

**Komponen:**

- masa_kerja_lalu: Total masa kerja hingga tanggal valuasi (tahun, 2 desimal)
- Dihitung sejak karyawan mulai kerja pertama kali

**Validasi:**

- Must be ≥ 0
- Must be ≤ 50 years (reasonable working career)
- Must be consistent with age calculations

**Contoh Perhitungan:**

```json
{
  "input": {
    "tanggal_masuk_kerja": "1995-07-06",
    "tanggal_valuasi": "2026-01-01"
  },
  "calculation": {
    "months_difference": 365.04,
    "unrounded_masa_kerja_lalu": 30.42,
    "masa_kerja_lalu": 30
  }
}
```

### **Step 1.4: Masa Kerja Lalu - Metode IFRIC**

Metode IFRIC menggunakan konsep usia minimum untuk menentukan masa kerja yang diakui.

```formula
💫 Formula: Min Usia IFRIC
min_usia = usia_pensiun - 24
```

**Kondisi Penerapan:**

**Kondisi 1:** Jika min_usia ≤ usia_mulai_masa_kerja

```formula
💫 Formula: IFRIC Kondisi 1
masa_kerja_lalu_ifric = (tanggal_valuasi - tanggal_masuk_kerja) / 12
```

**Kondisi 2:** Jika usia_mulai_masa_kerja < min_usia ≤ usia_saat_valuasi

```formula
💫 Formula: IFRIC Kondisi 2
masa_kerja_lalu_ifric = usia_saat_valuasi - min_usia
```

**Contoh IFRIC:**

```json
{
  "input": {
    "usia_pensiun": 55,
    "usia_mulai_kerja": 22.58,
    "usia_saat_valuasi": 53
  },
  "calculation": {
    "min_usia": 31,
    "condition": "usia_mulai_kerja < min_usia <= usia_saat_valuasi",
    "unrounded_masa_kerja_lalu_ifric": 22.08,
    "masa_kerja_lalu_ifric": 22
  }
}
```

### **Step 1.5: Future Service**

```formula
💫 Formula: Future Service
future_service = usia_pensiun - usia_saat_valuasi
```

**Komponen:**

- future_service: Sisa masa kerja hingga pensiun (tahun, 2 desimal)
- Minimum value: 0 (jika sudah pensiun)

**Business Rules:**

- If future_service ≤ 0: employee already retired
- If future_service > 40: unusual, flag for review
- Used for discount rate determination in Step 4

**Contoh Perhitungan:**

```json
{
  "input": {
    "usia_pensiun": 55,
    "usia_saat_valuasi": 53
  },
  "calculation": {
    "future_service": 1.92,
    "status": "near_retirement"
  }
}
```

### **Step 1.6: Total Masa Kerja**

```formula
💫 Formula: Total Masa Kerja
total_masa_kerja = usia_pensiun - usia_mulai_masa_kerja
```

**Usage:** Denominator for PVDBO ratio calculations in Step 4

**Contoh Perhitungan:**

```json
{
  "input": {
    "usia_pensiun": 55,
    "usia_mulai_kerja": 22.58
  },
  "calculation": {
    "total_masa_kerja": 32.34,
    "career_stage": "long_career"
  }
}
```

## 💰 **Salary Calculations**

### **Step 1.7: Gaji Saat Valuasi**

```formula
💫 Formula: Total Gaji
total_gaji_saat_valuasi = gaji_pokok + tunjangan_tetap
```

**Validasi:**

- Must be ≥ regional minimum wage
- Reasonable compared to position/industry
- Consistent with company salary structure

**Contoh Perhitungan:**

```json
{
  "input": {
    "gaji_pokok": 16000000,
    "tunjangan_tetap": 3356000
  },
  "calculation": {
    "total_gaji_saat_valuasi": 19356000,
    "validation": "above_minimum_wage"
  }
}
```

### **Step 1.8: Proyeksi Gaji Masa Depan**

```formula
💫 Formula: Proyeksi Gaji
total_gaji_t = total_gaji_t_before × (1 + tingkat_kenaikan_gaji)
```

**For Pension Calculation:**

```formula
💫 Formula: Salary at Retirement
total_gaji_pensiun = total_gaji_before_usia_pensiun
```

**Contoh Proyeksi:**

```json
{
  "input": {
    "total_gaji_saat_valuasi": 19356000,
    "tingkat_kenaikan_gaji": 0.09
  },
  "calculation": {
    "growth_factor": 1.09
    "total_gaji_pensiun": 21098040,
    "projected_increase": 1742040
  }
}
```

## ✅ **Data Validation Rules**

### **Critical Validation Checks**

```python
# Age logic validation
if usia_saat_valuasi >= usia_pensiun:
    status = "Already retired"
    exclude_from_calculation = True

# Service period validation  
if masa_kerja_lalu < 0:
    raise ValueError("Invalid service period")

# Salary validation
if gaji_pokok < 3_000_000:  # Below minimum wage
    flag_for_review = True

# Consistency checks
if usia_mulai_masa_kerja >= usia_saat_valuasi:
    raise ValueError("Start age cannot be >= current age")
```

### **Business Rules Validation**

- **Minimum Working Age**: 17 tahun
- **Maximum Retirement Age**: 70 tahun
- **Maximum Career Length**: 50 tahun
- **Salary Reasonableness**: Above regional minimum wage
- **Date Consistency**: All dates must be logically ordered

## 🔄 **Next Steps Integration**

Setelah menyelesaikan Step 1, data employee foundation akan digunakan dalam:

- **[Step 2: Multiple Decrement](step02_multiple_decrement.md)** - Menggunakan usia_saat_valuasi untuk mortality/withdrawal lookup
- **[Step 3: Benefit Calculation](step03_benefit_calculation.md)** - Menggunakan masa_kerja_lalu untuk benefit factor lookup
- **[Step 4: Present Value](step04_pvfb_pvdbo.md)** - Menggunakan future_service untuk discount rate, total_masa_kerja untuk PVDBO ratio

## 🎯 **Complete Example Output**

```json
{
  "step1_employee_foundation": {
    "input_data": {
      "tanggal_lahir": "1972-11-09",
      "tanggal_masuk_kerja": "1995-07-06",
      "tanggal_valuasi": "2026-01-01",
      "usia_pensiun": 55,
      "gaji_pokok": 16000000,
      "tunjangan_tetap": 3356000,
      "tingkat_kenaikan_gaji": 0.09
    },
    "calculated_results": {
      "usia_saat_valuasi": 53,
      "usia_mulai_masa_kerja": 22.58,
      "masa_kerja_lalu": 30.42,
      "masa_kerja_lalu_ifric": 22.08,
      "future_service": 1.92,
      "total_masa_kerja": 32.34,
      "total_gaji_saat_valuasi": 19356000,
      "total_gaji_pensiun": 21098040
    },
    "validation_status": {
      "age_logic": "passed",
      "service_consistency": "passed", 
      "salary_reasonableness": "passed",
      "ready_for_step2": true
    }
  }
}
```

---

📎 **Navigation:**

- ⬅️ [Master Guide](INDEX.md)
- ➡️ [Step 2: Multiple Decrement](step02_multiple_decrement.md)
- 📊 [Reference Tables](assumptions_reference.md) | [Troubleshooting](troubleshooting_guide.md)