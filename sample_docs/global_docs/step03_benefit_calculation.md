---
title: "Step 3: Benefit Calculation"
description: "Calculate benefit amounts for pension, death, disability, and withdrawal scenarios"
keywords: [benefit calculation, UUK13, UUCK, company policy, benefit factors, tax calculation]
difficulty: intermediate
estimated_reading: 25 minutes
target_audience: [actuaries, finance_team, HR_professionals]
document_type: calculation_step
step_number: 3
---

# 💰 Step 3: Benefit Calculation

## 🎯 **Overview**

Menghitung besar manfaat yang akan dibayarkan untuk setiap skenario keluar kerja (pensiun, meninggal, cacat, resign). Perhitungan mengacu pada program manfaat yang berlaku dan mempertimbangkan aspek perpajakan.

## 📋 **Benefit Program Framework**

| Program                 | Symbol  | Description          | Coverage           | Status    |
| ----------------------- | ------- | -------------------- | ------------------ | --------- |
| **UUK13**               | Legacy  | UU No. 13/2003       | Existing employees | ⭐⭐ Medium |
| **UUCK**                | Current | UU Cipta Kerja       | New employees      | ⭐⭐⭐ High  |
| **PP / Company Policy** | Custom  | Peraturan Perusahaan | All employees      | ⭐⭐⭐ High  |

**Benefit Factor Tables Quick Access:**

- 📊 **[UUK13 Benefit Factors](benefit_factors_uuk13.md)** - Legacy regulation factors
- 📊 **[UUCK Benefit Factors](benefit_factors_uuck.md)** - Current regulation factors
- 📊 **[Company Policy Factors](benefit_factors_pp.md)** - Customized benefit structure

## 🏭 **Benefit Factor Determination**

**Kondisi Umum:** Menggunakan `masa_kerja_lalu` dari Step 1 untuk semua benefit factor lookup

### **Step 3.1: Program Selection Logic**

```formula
💫 Formula: Program Selection
program_type = determine_program(tanggal_masuk_kerja, company_policy)
```

**Selection Rules:**

- **UUK13**: Employees hired before UUCK effective date (grandfathered)
- **UUCK**: Employees hired after UUCK effective date
- **PP**: Company may choose higher benefits than regulation minimum

### **Step 3.2: Benefit Factor Lookup**


```formula
💫 Formula: Pension Factor Lookup
pension_factor = lookup_table[program_type][masa_kerja_lalu_ifric]

💫 Formula: Other Benefits Factor Lookup  
death_factor = lookup_table[program_type][masa_kerja_lalu]
disability_factor = lookup_table[program_type][masa_kerja_lalu]
withdrawal_factor = lookup_table[program_type][masa_kerja_lalu]
```

**Komponen:**

- program_type: UUK13 | UUCK | PP
- masa_kerja_lalu_ifric (rounded to integer years) 
- masa_kerja_lalu (rounded to integer years) 
- benefit_type: pension | death | disability | withdrawal

**Service Years by Benefit Type:**
**Conceptual Rationale:**

| Benefit Type   | Service Years Used    | Rationale                                              |
| -------------- | --------------------- | ------------------------------------------------------ |
| **Pension**    | masa_kerja_lalu_ifric | Projected benefit approach; IFRIC minimum economic age |
| **Death**      | masa_kerja_lalu       | Immediate lump-sum; full service recognition           |
| **Disability** | masa_kerja_lalu       | Immediate lump-sum; full service recognition           |
| **Withdrawal** | masa_kerja_lalu       | Compensation for total actual service                  |

**Contoh Table Lookup:**

```json
{
  "step1_service_results": {
    "masa_kerja_lalu": 30.42,
    "masa_kerja_lalu_ifric": 22.08
  },
  "step3_table_lookup": {
    "pension_factor_lookup": {
      "table": "benefit_factors_pp",
      "service_years": 22.08,
      "rounded_years": 22,
      "lookup_key": "pension_factor[22]",
      "result_factor": 25.75
    },
    "death_factor_lookup": {
      "table": "benefit_factors_pp", 
      "service_years": 30.42,
      "rounded_years": 30,
      "lookup_key": "death_factor[30]",
      "result_factor": 28.0
    },
    "disability_factor_lookup": {
      "table": "benefit_factors_pp",
      "service_years": 30.42, 
      "rounded_years": 30,
      "lookup_key": "disability_factor[30]",
      "result_factor": 28.0
    },
    "withdrawal_factor_lookup": {
      "table": "benefit_factors_pp",
      "service_years": 30.42,
      "rounded_years": 30, 
      "lookup_key": "withdrawal_factor[30]",
      "result_factor": 3.0
    }
  }
}
```

**Kondisi Aplikasi:**

- Untuk usia_saat_valuasi ≥ usia_pensiun: benefit_factor = 0
- Untuk usia_saat_valuasi < usia_pensiun: gunakan table lookup

**Quick Reference - Common Factor Ranges:**

**UUK13 Program:**

|Service Years|Pension Factor|Death Factor|Disability Factor|Withdrawal Factor|
|---|---|---|---|---|
|5-10 years|16.1 - 25.3|16.1 - 25.3|18.4 - 29.9|1.2 - 1.95|
|15-25 years|27.6 - 32.2|27.6 - 32.2|34.5 - 43.7|2.25 - 2.85|
|30+ years|32.2|32.2|43.7|2.85|

**UUCK Program:**

| Service Years | Pension Factor | Death Factor | Disability Factor | Withdrawal Factor |
| ------------- | -------------- | ------------ | ----------------- | ----------------- |
| 5-10 years    | 12.5 - 19.75   | 14.0 - 22.0  | 14.0 - 22.0       | 0.0               |
| 15-25 years   | 21.75 - 25.75  | 24.0 - 28.0  | 24.0 - 28.0       | 0.0               |
| 30+ years     | 25.75          | 28.0         | 28.0              | 0.0               |

**Company Policy Program:**

| Service Years | Pension Factor | Death Factor | Disability Factor | Withdrawal Factor |
| ------------- | -------------- | ------------ | ----------------- | ----------------- |
| 5-10 years    | 12.5 - 19.75   | 14.0 - 22.0  | 14.0 - 22.0       | 0.5 - 1.5         |
| 15-25 years   | 21.75 - 25.75  | 24.0 - 28.0  | 24.0 - 28.0       | 1.5 - 3.00        |
| 30+ years     | 25.75          | 28.0         | 28.0              | 3.00              |

## 💰 **Salary Base Calculations**

**Kondisi Umum:** Menggunakan hasil proyeksi gaji dari Step 1

### **Step 3.3: Current Salary Foundation**

```formula
💫 Formula: Total Gaji Saat Valuasi
total_gaji_saat_valuasi = gaji_pokok + tunjangan_tetap
```

**From Step 1 Results:**

```json
{
  "step1_salary_results": {
    "gaji_pokok": 16000000,
    "tunjangan_tetap": 3356000,
    "total_gaji_saat_valuasi": 19356000
  }
}
```

### **Step 3.4: Salary Base by Benefit Type**

**Pension Benefit - Final Salary:**

```formula
💫 Formula: Pension Salary Base
pension_salary_base = total_gaji_pensiun  // from Step 1
```

**Current Salary:**

```formula
💫 Formula: Current Benefits Salary Base
benefit_salary_base = total_gaji_saat_valuasi  // from Step 1
```

**Projected Salary t:**

```formula
💫 Formula: Current Benefits Salary Base
benefit_salary_base_t = total_gaji_t  // from Step 1
```


**Contoh Current Salary Base Determination:**

```json
{
  "step1_salary_results": {
    "gaji_pokok": 16000000,
    "tunjangan_tetap": 3356000,
    "total_gaji_saat_valuasi": 19356000,
    "usia_saat_valuasi": 53,
    "usia_pensiun": 55,
    "benefit_salary_base": 19356000
  },
  "step3_salary_bases": {
    "pension_salary_base": 19356000,
    "death_salary_base": 19356000,
    "disability_salary_base": 19356000,
    "withdrawal_salary_base": 19356000,
    "salary_base_logic": "current_salary_usia_saat_valuasi"
  }
}
```

**Contoh Projected Salary Base Determination:**

```json
{
  "step1_salary_results_54": {
    "total_gaji_t_before": 19356000,
    "usia_saat_valuasi": 54,
    "usia_pensiun": 55,
    "tingkat_kenaikan_gaji": 0.09,
    "calculation": "19,356,000 × (1 + 0.09)",
    "benefit_salary_base": 21098040
  },
  "step3_salary_bases_55": {
    "pension_salary_base": 21098040,
    "death_salary_base": 21098040,
    "disability_salary_base": 21098040,
    "withdrawal_salary_base": 21098040,
    "salary_base_logic": "current_salary_54"
  },
  "step1_salary_results_55": {
    "total_gaji_t_before": 21098040,
    "usia_saat_valuasi": 55,
    "usia_pensiun": 55,
    "benefit_salary_base": 21098040
  },
  "step3_salary_bases_55": {
    "pension_salary_base": 21098040,
    "death_salary_base": 21098040,
    "disability_salary_base": 21098040,
    "withdrawal_salary_base": 21098040,
    "salary_base_logic": "current_salary_usia_pensiun"
  }
}
```

## 🧮 **Net Benefit Calculations**

### **Step 3.5: Pension Benefit**

```formula
💫 Formula: Pension Benefit Net
pension_benefit_net = pension_salary_base × pension_factor
```

**Kondisi Aplikasi:**

- Untuk usia_saat_valuasi < usia_pensiun: pension_benefit = 0 (belum waktunya pensiun)
- Untuk usia_saat_valuasi = usia_pensiun: gunakan formula

**Contoh Perhitungan Current Pension Benefit:**

```json
{
  "pension_calculation": {
    "employee_profile": {
      "unrounded_usia_saat_valuasi": 53.08,
      "usia_saat_valuasi": 53,
      "usia_pensiun": 55,
      "status": "aktif_belum_pensiun",
      "calculation_applicable": false
    },
    "factor_lookup": {
      "table_used": "benefit_factors_pp",
      "service_years_source": "masa_kerja_lalu_ifric",
      "service_years_value": 22.08,
      "lookup_years": 22,
      "lookup_key": "pension_factor[22]"
    },
    "pension_salary_base": 19356000,
    "pension_factor": 23.75,
    "pension_factor_validation": "benefit_factor_pp table, row 22 years",
    "calculation": "0 (belum mencapai usia pensiun)",
    "pension_benefit_net": 0,
    "note": "Pension benefit calculation deferred until age 55"
  }
}
```

**Contoh Perhitungan Projected Pension Benefit:**

```json
{
  "pension_calculation_54": {
    "employee_profile": {
      "unrounded_usia_saat_valuasi": 54.08,
      "usia_saat_valuasi": 54,
      "usia_pensiun": 55,
      "status": "aktif_belum_pensiun",
      "calculation_applicable": false
    },
    "factor_lookup": {
      "table_used": "benefit_factors_pp",
      "service_years_source": "masa_kerja_lalu_ifric",
      "service_years_value": 23.08,
      "lookup_years": 23,
      "lookup_key": "pension_factor[23]"
    },
    "pension_salary_base": 21098040,
    "pension_factor": 23.75,
    "pension_factor_validation": "From benefit_factor_pp table, row 23 years",
    "calculation": "0 (belum mencapai usia pensiun)",
    "pension_benefit_net": 0,
    "note": "Pension benefit calculation deferred until age 55"
  },
  "pension_calculation_55": {
    "employee_profile": {
      "usia_saat_valuasi": 55,
      "usia_pensiun": 55,
      "status": "usia_pensiun",
      "calculation_applicable": true
    },
    "factor_lookup": {
      "table_used": "benefit_factors_pp",
      "service_years_source": "masa_kerja_lalu_ifric",
      "service_years_value": 24,
      "lookup_years": 24,
      "lookup_key": "pension_factor[24]"
    },
    "pension_salary_base": 21098040,
    "pension_factor": 25.75,
    "pension_factor_validation": "From benefit_factor_pp table, row 24 years",
    "calculation": "21,098,040 × 25.75",
    "pension_benefit_net": 543274530,
    "note": "Pension benefit fully applicable at retirement age"
  }
}
```
### **Step 3.6: Death Benefit**

```formula
💫 Formula: Death Benefit Net
death_benefit_net = benefit_salary_base × death_factor + uang_duka
```

**Komponen:**

- benefit_salary_base: Gaji saat valuasi (current salary)
- death_factor: Multiplier dari benefit table
- uang_duka: Additional death benefit (company-specific)

**Kondisi Aplikasi:**

- Untuk usia_saat_valuasi ≥ usia_pensiun: death_benefit = 0
- Untuk usia_saat_valuasi < usia_pensiun: gunakan formula

**Contoh Perhitungan Current Death Benefit:**

```json
{
  "death_calculation": {
    "employee_profile": {
      "unrounded_usia_saat_valuasi": 53.08,
      "usia_saat_valuasi": 53,
      "usia_pensiun": 55,
      "status": "aktif_belum_pensiun",
      "calculation_applicable": true
    },
    "factor_lookup": {
      "table_used": "benefit_factor_pp",
      "service_years_source": "masa_kerja_lalu",
      "service_years_value": 30.42,
      "lookup_years": 30,
      "lookup_key": "death_factor[30]"
    },
    "death_salary_base": 19356000,
    "death_factor": 28.0,
    "death_factor_validation": "From benefit_factor_pp table, row 30 years",
    "uang_duka": 0,
    "calculation": "(19,356,000 × 28.0) + 0",
    "death_benefit_net": 541968000
  }
}
```

**Contoh Perhitungan Projected Death Benefit:**

```json
{
  "death_calculation_54": {
    "employee_profile": {
      "unrounded_usia_saat_valuasi": 54.08,
      "usia_saat_valuasi": 54,
      "usia_pensiun": 55,
      "status": "aktif_belum_pensiun",
      "calculation_applicable": true
    },
    "factor_lookup": {
      "table_used": "benefit_factor_pp",
      "service_years_source": "masa_kerja_lalu",
      "service_years_value": 31.42,
      "lookup_years": 31,
      "lookup_key": "death_factor[31]"
    },
    "death_salary_base": 21098040,
    "death_factor": 28.0,
    "death_factor_validation": "From benefit_factor_pp table, row 31 years",
    "uang_duka": 0,
    "calculation": "(21,098,040 × 28.0) + 0",
    "death_benefit_net": 541968000
  },
  "death_calculation_55": {
    "employee_profile": {
      "usia_saat_valuasi": 55,
      "usia_pensiun": 55,
      "status": "usia_pensiun",
      "calculation_applicable": false
    },
    "factor_lookup": {
      "table_used": "benefit_factor_pp",
      "service_years_source": "masa_kerja_lalu",
      "service_years_value": 32.33,
      "lookup_years": 32,
      "lookup_key": "death_factor[32]"
    },
    "death_salary_base": 21098040,
    "death_factor": 0,
    "death_factor_validation": "No death benefit at retirement age",
    "uang_duka": 0,
    "calculation": "0 (sudah mencapai usia pensiun)",
    "death_benefit_net": 0,
    "note": "Death benefit not applicable at retirement age"
  }
}
```

### **Step 3.7: Disability Benefit**

```formula
💫 Formula: Disability Benefit Net
disability_benefit_net = benefit_salary_base × disability_factor
```

**Business Rule Validation:**

```
Disability benefits should be < death benefits
```

**Contoh Perhitungan Current Disability Benefit:**

```json
{
  "disability_calculation": {
    "employee_profile": {
      "unrounded_usia_saat_valuasi": 53.08,
      "usia_saat_valuasi": 53,
      "usia_pensiun": 55,
      "status": "aktif_belum_pensiun",
      "calculation_applicable": true
    },
    "factor_lookup": {
      "table_used": "benefit_factor_pp",
      "service_years_source": "masa_kerja_lalu",
      "service_years_value": 30.42,
      "lookup_years": 30,
      "lookup_key": "disability_factor[30]"
    },
    "disability_salary_base": 19356000,
    "disability_factor": 28.0,
    "disability_factor_validation": "From benefit_factor_pp table, row 30 years",
    "calculation": "19,356,000 × 28.0",
    "disability_benefit_net": 541968000
  }
}
```

**Contoh Perhitungan Projected Disability Benefit:**

```json
{
  "disability_calculation_54": {
    "employee_profile": {
      "unrounded_usia_saat_valuasi": 54.08,
      "usia_saat_valuasi": 54,
      "usia_pensiun": 55,
      "status": "aktif_belum_pensiun",
      "calculation_applicable": true
    },
    "factor_lookup": {
      "table_used": "benefit_factor_pp",
      "service_years_source": "masa_kerja_lalu",
      "service_years_value": 31.42,
      "lookup_years": 31,
      "lookup_key": "disability_factor[31]"
    },
    "disability_salary_base": 21098040,
    "disability_factor": 28.0,
    "disability_factor_validation": "From benefit_factor_pp table, row 31 years",
    "uang_duka": 0,
    "calculation": "21,098,040 × 28.0",
    "disability_benefit_net": 541968000
  },
  "disability_calculation_55": {
    "employee_profile": {
      "usia_saat_valuasi": 55,
      "usia_pensiun": 55,
      "status": "usia_pensiun",
      "calculation_applicable": false
    },
    "factor_lookup": {
      "table_used": "benefit_factor_pp",
      "service_years_source": "masa_kerja_lalu",
      "service_years_value": 32.33,
      "lookup_years": 32,
      "lookup_key": "disability_factor[32]"
    },
    "disability_salary_base": 21098040,
    "disability_factor": 0,
    "disability_factor_validation": "No disability benefit at retirement age",
    "calculation": "0 (sudah mencapai usia pensiun)",
    "disability_benefit_net": 0,
    "note": "Disability benefit not applicable at retirement age"
  }
}
```

### **Step 3.8: Withdrawal Benefit**

```formula
💫 Formula: Withdrawal Benefit Net
withdrawal_benefit_net = benefit_salary_base × withdrawal_factor + uang_pisah
```

**Komponen:**

- withdrawal_factor: Service-based multiplier
- uang_pisah: Separation allowance (flat amount or salary multiple)

**UUCK Special Rule:**

```
withdrawal_factor = 0.0  // No withdrawal benefit under UUCK
withdrawal_benefit = compensation_pay_only  // Limited to compensation payment
```

**Contoh Perhitungan Current Withdrawal Benefit (Company Policy):**

```json
{
  "withdrawal_calculation": {
    "employee_profile": {
      "unrounded_usia_saat_valuasi": 53.08,
      "usia_saat_valuasi": 53,
      "usia_pensiun": 55, 
      "status": "aktif_belum_pensiun",
      "calculation_applicable": true
    },
    "factor_lookup": {
      "table_used": "benefit_factor_pp",
      "service_years_source": "masa_kerja_lalu", 
      "service_years_value": 30.42,
      "lookup_years": 30,
      "lookup_key": "withdrawal_factor[30]"
    },
    "withdrawal_salary_base": 19356000,
    "withdrawal_factor": 3.0,
    "withdrawal_factor_source": "company_policy",
    "uang_pisah": 0,
    "calculation": "(21,098,040 × 3.0) + 0",
    "withdrawal_benefit_net": 58068000,
    "note": "Company policy provides withdrawal benefits (UUCK = 0)"
  }
}
```

**Contoh Perhitungan Projected Withdrawal Benefit (Company Policy):**

```json
{
  "withdrawal_calculation_54": {
    "employee_profile": {
      "unrounded_usia_saat_valuasi": 54.08,
      "usia_saat_valuasi": 54,
      "usia_pensiun": 55, 
      "status": "aktif_belum_pensiun",
      "calculation_applicable": true
    },
    "factor_lookup": {
      "table_used": "benefit_factor_pp",
      "service_years_source": "masa_kerja_lalu", 
      "service_years_value": 31.42,
      "lookup_years": 31,
      "lookup_key": "withdrawal_factor[31]"
    },
    "withdrawal_salary_base": 21098040,
    "withdrawal_factor": 3.0,
    "withdrawal_factor_source": "company_policy",
    "uang_pisah": 0,
    "calculation": "(19,356,000 × 3.0) + 0",
    "withdrawal_benefit_net": 63294120,
    "note": "Company policy provides withdrawal benefits (UUCK = 0)"
  },
  "withdrawal_calculation_55": {
    "employee_profile": {
      "usia_saat_valuasi": 55,
      "usia_pensiun": 55, 
      "status": "aktif_belum_pensiun",
      "calculation_applicable": true
    },
    "factor_lookup": {
      "table_used": "benefit_factor_pp",
      "service_years_source": "masa_kerja_lalu", 
      "service_years_value": 32.33,
      "lookup_years": 32,
      "lookup_key": "withdrawal_factor[32]"
    },
    "withdrawal_salary_base": 21098040,
    "withdrawal_factor": 3.0,
    "withdrawal_factor_source": "company_policy",
    "uang_pisah": 0,
    "calculation": "(21,098,040 × 3.0) + 0",
    "withdrawal_benefit_net": 63294120,
    "note": "Company policy provides withdrawal benefits (UUCK = 0)"
  }
}
```

## 🏢 **Tax Calculations**

### **Step 3.9: Progressive Tax Structure**

📊 **[Complete Tax Rate Table](tax_progressive_rates.md)**

**Quick Reference - Tax Brackets:**

| Benefit Amount (IDR)      | Tax Rate | Batas Bawah |
| ------------------------- | -------- | ----------- |
| 0 - 50,000,000            | 0%       | 0           |
| 50,000,000 - 100,000,000  | 5%       | 50,000,000  |
| 100,000,000 - 500,000,000 | 15%      | 100,000,000 |
| > 500,000,000             | 25%      | 500,000,000 |

### **Step 3.10: Tax Calculation Process**

```formula
💫 Formula: Progressive Tax Calculation
total_tax = sum(tax_per_bracket)

For each bracket:
taxable_in_bracket = min(benefit_amount - bracket_floor, bracket_ceiling - bracket_floor)
tax_per_bracket = max(0, taxable_in_bracket × bracket_rate)
```

### **Step 3.11: Gross Benefit Determination**

**Tax Treatment Options:**

**Option 1: Tax Borne by Employee (Standard)**

```formula
💫 Formula: Gross Benefit (Employee Tax)
gross_benefit = net_benefit
```

**Option 2: Tax Borne by Company (Gross-up)**

```formula
💫 Formula: Gross Benefit (Company Tax)
gross_benefit = net_benefit + total_tax
company_tax_expense = total_tax
```

**Example Tax Calculation Projected Pension Benefit:**

```json
{
  "pension_benefit_tax_calculation": {
    "usia_pensiun": 55,
    "net_benefit": 543274530,
    "progressive_tax_breakdown": [
      {
        "bracket": 1,
        "range": "0 - 50,000,000",
        "taxable_amount": 50000000,
        "rate": 0.00,
        "tax": 0,
        "calculation": "50,000,000 × 0% = 0"
      },
      {
        "bracket": 2,
        "range": "50,000,000 - 100,000,000",
        "taxable_amount": 50000000,
        "rate": 0.05,
        "tax": 2500000,
        "calculation": "50,000,000 × 5% = 2,500,000"
      },
      {
        "bracket": 3,
        "range": "100,000,000 - 500,000,000",
        "taxable_amount": 400000000,
        "rate": 0.15,
        "tax": 60000000,
        "calculation": "400,000,000 × 15% = 60,000,000"
      },
      {
        "bracket": 4,
        "range": "> 500,000,000",
        "taxable_amount": 43274530,
        "rate": 0.25,
        "tax": 10818632.5,
        "calculation": "43,274,530 × 25% = 10,818,632.5"
      }
    ],
    "summary": {
      "total_tax": 73318633,
      "effective_tax_rate": 0.1350,
      "gross_benefit": 616593163
    }
  }
}
```

**Example Tax Calculation Current Death Benefit:**

```json
{
  "death_benefit_tax_calculation": {
    "usia_saat_valuasi": 53,
    "death_benefit": 541968000,
    "progressive_tax_breakdown": [
      {
        "bracket": 1,
        "range": "0 - 50,000,000",
        "taxable_amount": 50000000,
        "rate": 0.00,
        "tax": 0,
        "calculation": "50,000,000 × 0% = 0"
      },
      {
        "bracket": 2,
        "range": "50,000,000 - 100,000,000",
        "taxable_amount": 50000000,
        "rate": 0.05,
        "tax": 2500000,
        "calculation": "50,000,000 × 5% = 2,500,000"
      },
      {
        "bracket": 3,
        "range": "100,000,000 - 500,000,000",
        "taxable_amount": 400000000,
        "rate": 0.15,
        "tax": 60000000,
        "calculation": "400,000,000 × 15% = 60,000,000"
      },
      {
        "bracket": 4,
        "range": "> 500,000,000",
        "taxable_amount": 41968000,
        "rate": 0.25,
        "tax": 10492000,
        "calculation": "41,968,000 × 25% = 10,492,000"
      }
    ],
    "summary": {
      "total_tax": 72992000,
      "effective_tax_rate": 0.1347,
      "gross_benefit": 614960000
    }
  }
}
```

**Example Tax Calculation Projected Death Benefit:**

```json
{
  "death_benefit_tax_calculation": {
    "usia_saat_valuasi_projected": 54,
    "death_benefit": 590745120,
    "progressive_tax_breakdown": [
      {
        "bracket": 1,
        "range": "0 - 50,000,000",
        "taxable_amount": 50000000,
        "rate": 0.00,
        "tax": 0,
        "calculation": "50,000,000 × 0% = 0"
      },
      {
        "bracket": 2,
        "range": "50,000,000 - 100,000,000",
        "taxable_amount": 50000000,
        "rate": 0.05,
        "tax": 2500000,
        "calculation": "50,000,000 × 5% = 2,500,000"
      },
      {
        "bracket": 3,
        "range": "100,000,000 - 500,000,000",
        "taxable_amount": 400000000,
        "rate": 0.15,
        "tax": 60000000,
        "calculation": "400,000,000 × 15% = 60,000,000"
      },
      {
        "bracket": 4,
        "range": "> 500,000,000",
        "taxable_amount": 90745120,
        "rate": 0.25,
        "tax": 22686280,
        "calculation": "90,745,120 × 25% = 22,686,280"
      }
    ],
    "summary": {
      "total_tax": 85186280,
      "effective_tax_rate": 0.1442,
      "gross_benefit": 675931400
    }
  }
}
```

**Example Tax Calculation Current Disability Benefit:**

```json
{
  "disability_benefit_tax_calculation": {
    "usia_saat_valuasi": 53,
    "disability_benefit": 541968000,
    "progressive_tax_breakdown": [
      {
        "bracket": 1,
        "range": "0 - 50,000,000",
        "taxable_amount": 50000000,
        "rate": 0.00,
        "tax": 0,
        "calculation": "50,000,000 × 0% = 0"
      },
      {
        "bracket": 2,
        "range": "50,000,000 - 100,000,000",
        "taxable_amount": 50000000,
        "rate": 0.05,
        "tax": 2500000,
        "calculation": "50,000,000 × 5% = 2,500,000"
      },
      {
        "bracket": 3,
        "range": "100,000,000 - 500,000,000",
        "taxable_amount": 400000000,
        "rate": 0.15,
        "tax": 60000000,
        "calculation": "400,000,000 × 15% = 60,000,000"
      },
      {
        "bracket": 4,
        "range": "> 500,000,000",
        "taxable_amount": 41968000,
        "rate": 0.25,
        "tax": 10492000,
        "calculation": "41,968,000 × 25% = 10,492,000"
      }
    ],
    "summary": {
      "total_tax": 72992000,
      "effective_tax_rate": 0.1347,
      "gross_benefit": 614960000
    }
  }
}
```

**Example Tax Calculation Projected Disability Benefit:**

```json
{
  "disability_benefit_tax_calculation": {
    "usia_saat_valuasi_projected": 54,
    "disability_benefit": 590745120,
    "progressive_tax_breakdown": [
      {
        "bracket": 1,
        "range": "0 - 50,000,000",
        "taxable_amount": 50000000,
        "rate": 0.00,
        "tax": 0,
        "calculation": "50,000,000 × 0% = 0"
      },
      {
        "bracket": 2,
        "range": "50,000,000 - 100,000,000",
        "taxable_amount": 50000000,
        "rate": 0.05,
        "tax": 2500000,
        "calculation": "50,000,000 × 5% = 2,500,000"
      },
      {
        "bracket": 3,
        "range": "100,000,000 - 500,000,000",
        "taxable_amount": 400000000,
        "rate": 0.15,
        "tax": 60000000,
        "calculation": "400,000,000 × 15% = 60,000,000"
      },
      {
        "bracket": 4,
        "range": "> 500,000,000",
        "taxable_amount": 90745120,
        "rate": 0.25,
        "tax": 22686280,
        "calculation": "90,745,120 × 25% = 22,686,280"
      }
    ],
    "summary": {
      "total_tax": 85186280,
      "effective_tax_rate": 0.1442,
      "gross_benefit": 675931400
    }
  }
}
```

**Example Tax Calculation Current Withdrawal Benefit:**

```json
{
  "disability_benefit_tax_calculation": {
    "usia_saat_valuasi": 53,
    "withdrawal_benefit": 58068000,
    "progressive_tax_breakdown": [
      {
        "bracket": 1,
        "range": "0 - 50,000,000",
        "taxable_amount": 50000000,
        "rate": 0.00,
        "tax": 0,
        "calculation": "50,000,000 × 0% = 0"
      },
      {
        "bracket": 2,
        "range": "50,000,000 - 100,000,000",
        "taxable_amount": 8068000,
        "rate": 0.05,
        "tax": 403400,
        "calculation": "8,068,000 × 5% = 403,400"
      },
      {
        "bracket": 3,
        "range": "100,000,000 - 500,000,000",
        "taxable_amount": 0,
        "rate": 0.15,
        "tax": 0,
        "calculation": "0 × 15% = 0"
      },
      {
        "bracket": 4,
        "range": "> 500,000,000",
        "taxable_amount": 0,
        "rate": 0.25,
        "tax": 0,
        "calculation": "0 × 25% = 0"
      }
    ],
    "summary": {
      "total_tax": 403400,
      "effective_tax_rate": 0.0069,
      "gross_benefit": 58471400
    }
  }
}
```

**Example Tax Calculation Projected Withdrawal Benefit:**

```json
{
  "withdrawal_benefit_tax_calculation": {
    "usia_saat_valuasi_projected": 54,
    "withdrawal_benefit": 63294120,
    "progressive_tax_breakdown": [
      {
        "bracket": 1,
        "range": "0 - 50,000,000",
        "taxable_amount": 50000000,
        "rate": 0.00,
        "tax": 0,
        "calculation": "50,000,000 × 0% = 0"
      },
      {
        "bracket": 2,
        "range": "50,000,000 - 100,000,000",
        "taxable_amount": 13294120,
        "rate": 0.05,
        "tax": 664706,
        "calculation": "13,294,120 × 5% = 664,706"
      },
      {
        "bracket": 3,
        "range": "100,000,000 - 500,000,000",
        "taxable_amount": 0,
        "rate": 0.15,
        "tax": 0,
        "calculation": "0 × 15% = 0"
      },
      {
        "bracket": 4,
        "range": "> 500,000,000",
        "taxable_amount": 0,
        "rate": 0.25,
        "tax": 0,
        "calculation": "0 × 25% = 0"
      }
    ],
    "summary": {
      "total_tax": 664706,
      "effective_tax_rate": 0.0105,
      "gross_benefit": 63958826
    }
  }
}
```


## ✅ **Validation & Quality Assurance**

### **Step 3.12: Single Table Consistency Validation**

python

```python
def validate_single_table_consistency(step3_calculations):
    """Validate consistent table usage and lookup keys"""
    
    # All benefits should use the same base table
    expected_table = "benefit_factors_pp"
    
    all_calculations = [
        step3_calculations["pension_calculation"],
        step3_calculations["death_calculation"], 
        step3_calculations["disability_calculation"],
        step3_calculations["withdrawal_calculation"]
    ]
    
    for calc in all_calculations:
        if calc["factor_lookup"]["table_used"] != expected_table:
            raise ValidationError(f"All factors should use {expected_table} table")
    
    # Immediate benefits should have consistent service years
    immediate_benefits = ["death", "disability", "withdrawal"]
    expected_years = step3_calculations["death_calculation"]["factor_lookup"]["lookup_years"]
    
    for benefit in immediate_benefits:
        lookup = step3_calculations[f"{benefit}_calculation"]["factor_lookup"]
        if lookup["lookup_years"] != expected_years:
            raise ValidationError(f"All immediate benefits should use same lookup years: {expected_years}")
```

### **Step 3.13: Factor Value Reasonableness Validation**

python

```python
def validate_factor_reasonableness(step3_calculations):
    """Validate factor values are reasonable for given service years"""
    
    pension_calc = step3_calculations["pension_calculation"]["factor_lookup"]
    death_calc = step3_calculations["death_calculation"]["factor_lookup"] 
    
    pension_years = pension_calc["lookup_years"]
    death_years = death_calc["lookup_years"]
    
    pension_factor = step3_calculations["pension_calculation"]["pension_factor"]
    death_factor = step3_calculations["death_calculation"]["death_factor"]
    
    # For long-service employees with IFRIC adjustment
    if death_years > pension_years:
        # Death factor might be higher due to longer service recognition
        if death_factor < pension_factor:
            flag_for_review("Unusual: Death factor lower than pension factor despite longer service recognition")
    
    # Company Policy specific validations
    if pension_years >= 20 and pension_factor < 25.0:
        flag_for_review("Low pension factor for long service employee")
    
    if death_years >= 25 and death_factor < 25.0:
        flag_for_review("Low death factor for long service employee")
    
    # Disability should equal death in Company Policy
    disability_factor = step3_calculations["disability_calculation"]["disability_factor"]
    if abs(disability_factor - death_factor) > 0.01:
        raise ValidationError("Company Policy: Disability factor should equal death factor")
```

### **Step 3.14: Age-Based Benefit Logic Validation**

python

```python
def validate_age_based_benefit_logic(step3_calculations):
    """Validate benefit calculations based on employee age status"""
    
    employee_profile = step3_calculations["pension_calculation"]["employee_profile"]
    usia_saat_valuasi = employee_profile["usia_saat_valuasi"]
    usia_pensiun = employee_profile["usia_pensiun"]
    
    pension_benefit = step3_calculations["pension_calculation"]["pension_benefit_net"]
    death_benefit = step3_calculations["death_calculation"]["death_benefit_net"]
    disability_benefit = step3_calculations["disability_calculation"]["disability_benefit_net"]
    withdrawal_benefit = step3_calculations["withdrawal_calculation"]["withdrawal_benefit_net"]
    
    if usia_saat_valuasi >= usia_pensiun:
        # Already retired - only pension applicable
        if death_benefit > 0:
            raise ValidationError("Death benefit should be 0 for retirees")
        if disability_benefit > 0:
            raise ValidationError("Disability benefit should be 0 for retirees") 
        if withdrawal_benefit > 0:
            raise ValidationError("Withdrawal benefit should be 0 for retirees")
    else:
        # Still active - pension not applicable yet
        if pension_benefit > 0:
            raise ValidationError("Pension benefit should be 0 for active employees")
        
        # Immediate benefits should be > 0 for active employees
        if death_benefit <= 0:
            flag_for_review("Zero death benefit for active employee - verify factor lookup")
        if disability_benefit <= 0:
            flag_for_review("Zero disability benefit for active employee - verify factor lookup")
```

### **Step 3.15: Lookup Key Format Validation**

python

```python
def validate_lookup_key_format(step3_calculations):
    """Validate lookup key format consistency"""
    
    expected_patterns = {
        "pension": r"pension_factor\[\d+\]",
        "death": r"death_factor\[\d+\]", 
        "disability": r"disability_factor\[\d+\]",
        "withdrawal": r"withdrawal_factor\[\d+\]"
    }
    
    import re
    
    for benefit_type, pattern in expected_patterns.items():
        calc = step3_calculations[f"{benefit_type}_calculation"]
        lookup_key = calc["factor_lookup"]["lookup_key"]
        
        if not re.match(pattern, lookup_key):
            raise ValidationError(f"Invalid lookup key format for {benefit_type}: {lookup_key}")
        
        # Extract years from lookup key and validate
        years_from_key = int(lookup_key.split('[')[1].split(']')[0])
        years_from_calc = calc["factor_lookup"]["lookup_years"]
        
        if years_from_key != years_from_calc:
            raise ValidationError(f"Lookup key years mismatch for {benefit_type}: {years_from_key} vs {years_from_calc}")
```

### **Step 3.17: Cross-Step Integration Validation**

python

```python
def validate_step_integration(step1_results, step3_calculations):
    """Ensure consistency with Step 1 calculations"""
    
    # Validate salary base consistency
    expected_pension_base = step1_results["total_gaji_pensiun"]
    actual_pension_base = step3_calculations["pension_calculation"]["pension_salary_base"]
    
    if abs(actual_pension_base - expected_pension_base) > 1000:
        raise ValidationError("Pension salary base inconsistent with Step 1")
    
    # Validate service years consistency
    step1_masa_kerja_lalu = step1_results["masa_kerja_lalu"]
    step1_masa_kerja_ifric = step1_results["masa_kerja_lalu_ifric"]
    
    # Check pension uses IFRIC years
    pension_years = step3_calculations["pension_calculation"]["factor_lookup"]["service_years_value"]
    if abs(pension_years - step1_masa_kerja_ifric) > 0.1:
        raise ValidationError("Pension service years inconsistent with Step 1 IFRIC calculation")
    
    # Check immediate benefits use full service years
    death_years = step3_calculations["death_calculation"]["factor_lookup"]["service_years_value"]
    if abs(death_years - step1_masa_kerja_lalu) > 0.1:
        raise ValidationError("Death benefit service years inconsistent with Step 1 calculation")
```

## 🔄 **Integration with Next Steps**

Hasil benefit calculations akan digunakan dalam:

- **[Step 4: Present Value Calculations](step04_pvfb_pvdbo.md)** - Net benefit amounts untuk PVFB calculations
- **[Step 5: Sensitivity Analysis](step05_sensitivity_analysis.md)** - Benefit sensitivities untuk impact analysis

**Output Requirements for Step 4:**

```json
{
  "required_outputs": {
    "for_step4": [
      "pension_benefit_gross",
      "death_benefit_gross", 
      "disability_benefit_gross",
      "withdrawal_benefit_gross"
    ],
    "supporting_data": [
      "usia_saat_valuasi", 
      "usia_pensiun",
      "benefit_factors_used",
      "salary_bases_applied",
      "tax_calculations",
      "program_type_selected"
    ]
  }
}
```

## 🎯 **Expected Final Output**

```json
{
  "step3_benefit_calculation": {
    "employee_profile": {
      "usia_saat_valuasi": 53.08,
      "usia_pensiun": 55,
      "masa_kerja_lalu": 30.42,
      "masa_kerja_lalu_ifric": 22.08,
      "program_type": "Company_Policy",
      "status": "aktif_belum_pensiun"
    },
    "table_lookup_structure": {
      "pension_table": "benefit_factors_pp",
      "lookup_keys": {
        "pension": "pension_factor[22]",
        "death": "death_factor[30]",
        "disability": "disability_factor[30]",
        "withdrawal": "withdrawal_factor[30]"
      },
      "service_years_mapping": {
        "pension_uses": "masa_kerja_lalu_ifric (22.08 → 22)",
        "others_use": "masa_kerja_lalu (30.42 → 30)"
      }
    },
    "salary_bases": 19356000
    },
    "benefit_factors": {
      "pension_factor": 23.75,
      "death_factor": 28.0,
      "disability_factor": 28.0,
      "withdrawal_factor": 3.0,
      "factor_validation": "All factors verified against respective lookup tables"
    },
    "net_benefits": {
      "pension": 0,
      "death": 541968000,
      "disability": 541968000,
      "withdrawal": 58068000,
      "calculation_notes": {
        "pension": "Not applicable - age < retirement age",
        "death": "death_factor[30] × total_gaji_saat_valuasi",
        "disability": "disability_factor[30] × total_gaji_saat_valuasi", 
        "withdrawal": "withdrawal_factor[30] × total_gaji_saat_valuasi"
      }
    },
    "tax_calculations": {
      "pension_tax": 0,
      "death_tax": 72992000,
      "disability_tax": 72992000,
      "withdrawal_tax": 403400
    },
    "gross_benefits": {
      "pension": 0,
      "death": 614960000,
      "disability": 614960000,
      "withdrawal": 58471400
    },
    "validation_status": {
      "table_structure_validation": "passed",
      "cross_table_consistency": "passed", 
      "service_years_mapping": "passed",
      "factor_lookup_verification": "passed",
      "ready_for_step4": true
    }
  }
}
```

---

📎 **Navigation:**

- ⬅️ [Step 2: Multiple Decrement](step02_multiple_decrement.md)
- ➡️ [Step 4: Present Value](step04_pvfb_pvdbo.md)
- 📊 [Benefit Tables](assumptions_reference.md#program-manfaat) | [Tax Rates](tax_progressive_rates.md)