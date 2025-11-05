---
title: "Step 2: Multiple Decrement Analysis"
description: Probability calculations for mortality, disability, withdrawal, and pension
keywords:
  - multiple decrement
  - mortality rate
  - disability rate
  - withdrawal rate
  - TMI IV 2019
difficulty:
  - intermediate
estimated_reading: 20 minutes
target_audience:
  - actuaries
  - finance_team
document_type: calculation_step
step_number: 2
rag_optimization: []
---
# 📊 Step 2: Multiple Decrement Analysis

## 🎯 **Overview**

Multiple decrement analysis menghitung probabilitas berbagai cara seorang karyawan dapat keluar dari skema imbalan kerja sebelum pensiun normal. Step ini adalah foundation untuk menghitung expected present value dari future benefits.

## 📋 **Decrement Types Framework**

|Decrement|Symbol|Description|Impact on Benefits|Age Condition|
|---|---|---|---|---|
|**Withdrawal**|qx_w|Voluntary/involuntary termination|Triggers withdrawal benefit|15 ≤ x ≤ retirement-1|
|**Mortality**|qx_d|Death probability|Triggers death benefit|All working ages|
|**Disability**|qx_i|Permanent disability|Triggers disability benefit|All working ages|
|**Retirement**|qx_r|Normal retirement|Triggers pension benefit|x = retirement age|

## 📊 **Probabilitas Dasar (Base Rates)**

**Kondisi Umum:** `x` = `usia_saat_valuasi` tiap karyawan untuk semua multiple decrement rate

### **Step 2.1: qx_w - Withdrawal Rate Dasar**

```formula
💫 Formula: Withdrawal Rate Dasar
withdrawal_rate_dasar = withdrawal_assumptions_table[age]  // berdasarkan x = age
```

**Kondisi Aplikasi:**

- Untuk 0 < `age` ≤ 14 dan `age = usia_pensiun`: `Withdrawal_Rate_Dasar` = 0
- Untuk 15 ≤ `age` ≤ `usia_pensiun - 1`: ambil dari tabel withdrawal assumptions

**Quick Reference - Withdrawal Rates:**

| Age | Withdrawal Rate | Business Context       | Usage Frequency |
| --- | --------------- | ---------------------- | --------------- |
| 25  | 6.0%            | Job mobility peak      | ⭐⭐⭐ High        |
| 35  | 1.8%            | Career stability       | ⭐⭐⭐ High        |
| 45  | 1.2%            | Pre-senior phase       | ⭐⭐⭐ High        |
| 55  | 0.0%            | Retirement preparation | ⭐⭐ Medium       |

**For Complete Withdrawal Data:** 📊 **[Withdrawal Assumptions](withdrawal_assumptions.md)** - Company-specific age-based rates

### **Step 2.2: qx_d - Mortality Rate Dasar**

```formula
💫 Formula: Mortality Rate Dasar Selection
mortality_rate_dasar = TMI_IV_Male(age)    // untuk karyawan pria
mortality_rate_dasar = TMI_IV_Female(age)  // untuk karyawan wanita
```

**Pemilihan Tabel berdasarkan Gender:**

- **Male**: TMI IV 2019 Male untuk karyawan pria
- **Female**: TMI IV 2019 Female untuk karyawan wanita
- **Aggregate**: Weighted average untuk portfolio calculation

**Quick Reference - TMI IV Rates:**

| Age (x) | Male Rate | Female Rate | Business Context    |
| ------- | --------- | ----------- | ------------------- |
| 25      | 0.00052   | 0.00038     | Young professionals |
| 35      | 0.00107   | 0.00080     | Mid-career          |
| 45      | 0.00302   | 0.00187     | Management level    |
| 55      | 0.00789   | 0.00483     | Pre-retirement      |

**For Complete TMI IV Data:** 📊 **[TMI IV Mortality Table](tmi_iv_mortality.md)** - 111 ages, gender-specific rates

### **Step 2.3: qx_i - Disability Rate Dasar**

```formula
💫 Formula: Disability Rate Dasar
disability_rate_dasar = disability_multiplier × mortality_rate_dasar
```

**Where:**

- disability_rate: 5% hingga 10% dari mortality_rate(conservative: 10%, optimistic: 5%)

**Contoh Perhitungan (menggunakan 10%):**

```json
{
  "disability_calculations": {
    "age_25": {
      "mortality_rate": 0.00052,
      "disability_rate": 0.000052,
      "calculation": "0.1 × 0.00052"
    },
    "age_35": {
      "mortality_rate": 0.00107,
      "disability_rate": 0.000107,
      "calculation": "0.1 × 0.00107"
    },
    "age_45": {
      "mortality_rate": 0.00302,
      "disability_rate": 0.000302,
      "calculation": "0.1 × 0.00302"
    },
    "age_55": {
      "mortality_rate": 0.00789,
      "disability_rate": 0.000789,
      "calculation": "0.1 × 0.00789"
    }
  }
}
```

### **Step 2.4: qx_r - Pension Rate Dasar**

```formula
💫 Formula: Pension Rate Dasar
pension_rate_dasar = 0  // untuk x < usia_pensiun
pension_rate_dasar = 1  // untuk x = usia_pensiun
```

**Business Logic:**

```python
def calculate_pension_rate_dasar(usia_saat_valuasi, usia_pensiun):
    if usia_saat_valuasi < usia_pensiun:
        return 0.0  # Not yet eligible for retirement
    elif usia_saat_valuasi == usia_pensiun:
        return 1.0  # Mandatory retirement at retirement age
    else:
        return 0.0  # Already retired (exclude from calculation)
```


## 📊 **Probabilitas Terkoreksi (Corrected Rates)**

**Kondisi Umum:** `x = usia_saat_valuasi` masing-masing karyawan untuk semua decrement rate

### **Step 2.5: Mortality Rate Terkoreksi**

```formula
💫 Formula: Mortality Rate Terkoreksi
mortality_rate = (life_probability - pension_rate_dasar) × mortality_rate_dasar
```

**Kondisi Aplikasi:**

- Untuk `usia_saat_valuasi` ≥ `usia_pensiun`: `mortality_rate` = 0
- Untuk `usia_saat_valuasi` < `usia_pensiun`: menggunakan rumus di atas

### **Step 2.6: Withdrawal Rate Terkoreksi**

```formula
💫 Formula: Withdrawal Rate Terkoreksi
withdrawal_rate = (life_probability - pension_rate_dasar) × withdrawal_rate_dasar
```

**Kondisi Aplikasi:**

- Untuk `usia_saat_valuasi` ≥ `usia_pensiun`: `withdrawal_rate` = 0
- Untuk `usia_saat_valuasi` < `usia_pensiun`: menggunakan rumus di atas

### **Step 2.7: Disability Rate Terkoreksi**

```formula
💫 Formula: Disability Rate Terkoreksi
disability_rate = (life_probability - pension_rate_dasar) × disability_rate_dasar
```

**Kondisi Aplikasi:**

- Untuk `usia_saat_valuasi` ≥ `usia_pensiun`: `disability_rate` = 0
- Untuk `usia_saat_valuasi` < `usia_pensiun`: menggunakan rumus di atas

### **Step 2.8: Pension Rate Terkoreksi**

```formula
💫 Formula: Pension Rate Terkoreksi
pension_rate = pension_rate_dasar
```

**For retirement age:**

```formula
💫 Formula: Pension Rate at Retirement
pension_rate = 1.0  // when x = usia_pensiun
```

## 🔄 **Life Probability Calculation**

### **Step 2.9: Life Probability Foundation**


**Initial Condition:**

```formula
💫 Formula: Starting Life Probability
life_probability_at_valuation = 1.0  // Starting point at current age
```

```formula
💫 Formula: Life Probability Calculation
life_probability_t = life_probability_previous_year - (total_decrement_rate_previous_year)
```

**Where:**

```formula
💫 Formula: Total Decrement Rate
total_decrement_rate = mortality_rate + disability_rate + withdrawal_rate + pension_rate
```

## 📊 **Complete Calculation Example**

### **Employee Profile:**

```json
{
  "employee_profile": {
    "usia_saat_valuasi": 53,
    "usia_pensiun": 55,
    "gender": "male",
    "company_withdrawal_table": "standard_corporate"
  }
}
```

### **Step-by-Step Calculation:**

#### **Step 2.A: Base Rate Lookup**

```json
{
  "base_rate_lookup": {
    "age_53": {
      "withdrawal_rate_dasar": 0.000,
      "mortality_rate_dasar": 0.00667,
      "disability_rate_dasar": 0.000667,
      "pension_rate_dasar": 0.0
    },
    "age_54": {
      "withdrawal_rate_dasar": 0.000,
      "mortality_rate_dasar": 0.00727,
      "disability_rate_dasar": 0.000727,
      "pension_rate_dasar": 0.0
    },
    "age_55": {
      "withdrawal_rate_dasar": 0.000,
      "mortality_rate_dasar": 0.00789,
      "disability_rate_dasar": 0.000789,
      "pension_rate_dasar": 1.0
    }
  }
}
```

#### **Step 2.B: Life Probability Calculation**

```json
{
  "life_probability_progression": {
    "age_53": {
      "life_probability": 1.000000,
      "total_decrements": 0.0,
      "status": "valuation_starting_point"
    },
    "age_54": {
      "life_probability": 0.995567,
      "total_decrements": 0.004433,
      "calculation": "1.0 - (0.004030 + 0.000403 + 0.000000 + 0.0)"
    },
    "age_55": {
      "life_probability": 0.990727,
      "total_decrements": 0.00484,
      "calculation": "0.995567 - (0.004400 + 0.000440 + 0.000000 + 0.0)"
    }
  }
}
```

#### **Step 2.C: Corrected Rates Calculation**

```json
{
  "corrected_rates": {
    "age_53": {
      "life_probability": 1.000000,
      "pension_rate_dasar": 0.0,
      "correction_factor": 1.0,
      "mortality_rate": 0.004030,
      "disability_rate": 0.000403,
      "withdrawal_rate": 0.000000,
      "pension_rate": 0.000000
    },
    "age_54": {
      "life_probability": 0.995567,
      "pension_rate_dasar": 0.0,
      "correction_factor": 0.995567,
      "mortality_rate": 0.004400,
      "disability_rate": 0.000440,
      "withdrawal_rate": 0.000000,
      "pension_rate": 0.000000
    },
    "age_55": {
      "life_probability": 0.990727,
      "pension_rate_dasar": 1.0,
      "correction_factor": -0.021227,
      "mortality_rate": 0.000000,
      "disability_rate": 0.000000,
      "withdrawal_rate": 0.000000,
      "pension_rate": 1.000000
    }
  }
}
```

## ✅ **Enhanced Validation Rules**

### **Probability Consistency Validation**

```python
def validate_probability_consistency(rates_by_age):
    """Ensure corrected rates maintain logical relationships"""
    
    for age, rates in rates_by_age.items():
        # Check individual rate bounds
        for rate_name, rate_value in rates.items():
            if rate_value < 0:
                raise ValueError(f"Age {age}: {rate_name} cannot be negative: {rate_value}")
            
            if rate_value > 1:
                raise ValueError(f"Age {age}: {rate_name} cannot exceed 1.0: {rate_value}")
        
        # Check total probability
        total_prob = (rates["mortality_rate"] + rates["disability_rate"] + 
                     rates["withdrawal_rate"] + rates["pension_rate"])
        
        if total_prob > 1.001:  # Allow small rounding errors
            raise ValueError(f"Age {age}: Total probabilities exceed 1.0: {total_prob}")
```

### **Business Logic Validation**

```python
def validate_business_logic(age, rates, retirement_age):
    """Check business reasonableness of corrected rates"""
    
    # At/after retirement age checks
    if age >= retirement_age:
        if rates["withdrawal_rate"] > 0:
            raise ValueError(f"Age {age}: No withdrawal allowed at/after retirement")
        
        if rates["mortality_rate"] > 0 or rates["disability_rate"] > 0:
            raise ValueError(f"Age {age}: No mortality/disability at/after retirement in this model")
    
    # Before retirement age checks
    if age < retirement_age:
        if rates["pension_rate"] > 0:
            raise ValueError(f"Age {age}: No pension before retirement age {retirement_age}")
        
        # Reasonableness checks
        if age > 50 and rates["withdrawal_rate"] > 0.05:
            flag_for_review(f"High withdrawal rate {rates['withdrawal_rate']} near retirement")
```

### **Cross-Age Consistency**

```python
def validate_cross_age_consistency(results_by_age):
    """Check consistency across different ages"""
    
    ages = sorted(results_by_age.keys())
    
    for i in range(1, len(ages)):
        current_age = ages[i]
        previous_age = ages[i-1]
        
        current_life_prob = results_by_age[current_age]["life_probability"]
        previous_life_prob = results_by_age[previous_age]["life_probability"]
        
        # Life probability should generally decrease with age
        if current_life_prob > previous_life_prob and current_age < retirement_age:
            flag_inconsistency(f"Life probability increased from age {previous_age} to {current_age}")
```

## 🔄 **Integration with Next Steps**

Hasil multiple decrement analysis akan digunakan dalam:

- **[Step 3: Benefit Calculation](step03_benefit_calculation.md)** - Corrected rates untuk weight expected benefits berdasarkan probabilitas keluar kerja
- **[Step 4: Present Value](step04_pvfb_pvdbo.md)** - Life probabilities untuk discount expected cash flows
- **[Step 5: Sensitivity Analysis](step05_sensitivity_analysis.md)** - Base assumptions untuk test sensitivitas

**Output Requirements for Next Steps:**

```json
{
  "required_outputs": {
    "for_step3": ["mortality_rate", "disability_rate", "withdrawal_rate", "pension_rate"],
    "for_step4": ["life_probability", "corrected_rates_by_age"],
    "for_step5": ["base_assumptions", "methodology_parameters"]
  }
}
```

## 🎯 **Expected Final Output**

```json
{
  "multiple_decrement_results": {
    "summary": {
      "usia_saat_valuasi": 53,
      "usia_pensiun": 55,
      "years_to_retirement": 2.0,
      "methodology": "corrected_probabilities_competing_risks"
    },
    "base_assumptions": {
      "mortality_source": "TMI_IV_Male",
      "disability_multiplier": 0.10,
      "withdrawal_policy": "company_withdrawal_assumptions",
      "validation_status": "passed"
    },
    "age_53": {
      "life_probability": 1.000000,
      "mortality_rate": 0.004030,
      "disability_rate": 0.000403,
      "withdrawal_rate": 0.000000,
      "pension_rate": 0.000000
    },
    "age_54": {
      "life_probability": 0.995567,
      "mortality_rate": 0.004400,
      "disability_rate": 0.000440,
      "withdrawal_rate": 0.000000,
      "pension_rate": 0.000000
    },
    "age_55": {
      "life_probability": 0.990727,
      "mortality_rate": 0.000000,
      "disability_rate": 0.000000,
      "withdrawal_rate": 0.000000,
      "pension_rate": 1.000000
    },
    "validation_summary": {
      "probability_consistency": "passed",
      "business_logic": "reasonable",
      "cross_age_checks": "passed",
      "ready_for_step3": true
    }
  }
}
```

---

📎 **Navigation:**

- ⬅️ [Step 1: Employee Data](step01_employee_data.md)
- ➡️ [Step 3: Benefit Calculation](step03_benefit_calculation.md)
- 📊 [TMI IV Table](tmi_iv_mortality.md) | [Withdrawal Table](withdrawal_assumptions.md)