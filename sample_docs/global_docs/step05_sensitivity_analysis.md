---
title: "Step 5: Sensitivity Analysis & Reporting"
description: "Sensitivity analysis, maturity profile, and comprehensive reporting for PSAK 219"
keywords: [sensitivity analysis, maturity profile, actuarial gains, PSAK 219 reporting]
difficulty: advanced
estimated_reading: 25 minutes
target_audience: [actuaries, finance_team, management]
document_type: calculation_step
step_number: 5
---

# 📊 Step 5: Sensitivity Analysis & Reporting

## 🎯 **Overview**

Final step dalam valuasi aktuaria yang menganalisis sensitivitas hasil terhadap perubahan key assumptions, menghitung maturity profile kewajiban, dan menyiapkan comprehensive reporting sesuai PSAK 219 requirements.

_See Also: [Analisis Lanjutan](02f2_valuasi_analisis_lanjutan.md)

## 📈 **Sensitivity Analysis Framework**

| Analysis Type     | Symbol           | Description                   | Usage     | Impact Measurement         |
| ----------------- | ---------------- | ----------------------------- | --------- | -------------------------- |
| **Discount Rate** | Rate Sensitivity | Interest rate risk assessment | ⭐⭐⭐ High  | PVDBO percentage change    |
| **Salary Growth** | Inflation Risk   | Wage inflation impact         | ⭐⭐⭐ High  | Service Cost impact        |

**Key Integration Points:**

- 📈 **[Step 4: Present Value Calculations](step04_pvfb_pvdbo.md)** - Base PVDBO and CSC results
- 📊 **[IGSYC Yield Curve](yield_curve_igsyc.md)** - Discount rate sensitivity scenarios

## 📊 **Primary Sensitivity Testing**

**Kondisi Umum:** Menggunakan hasil `total_pvdbo` dan `total_csc` dari Step 4 sebagai base case untuk semua sensitivity analysis

### **Step 5.1: Discount Rate Sensitivity**

```formula
💫 Formula: Discount Rate Impact
sensitivity_discount = pvdbo_new - pvdbo_base
percentage_impact = (sensitivity_discount / pvdbo_base) × 100%
```

**Test Scenarios:**

- **Up Impact**: discount_rate_up = base_rate + 0.01 (+100 basis points)
- **Down Impact:** discount_rate_down = base_rate - 0.01 (-100 basis points)

**Expected Relationship:**

```
Discount Rate ↑ → PVDBO ↓ (typically 8-15% change for 1% rate change)
Discount Rate ↓ → PVDBO ↑ (typically 10-18% change for 1% rate change)
```

**Kondisi Aplikasi:**

- Untuk all employees: apply uniform rate change
- Recalculate all discount factors from Step 4
- Maintain other assumptions constant

**Contoh Perhitungan:**

```json
{
  "discount_rate_sensitivity": {
    "base_case": {
      "discount_rate": 0.0609,
      "total_pvdbo": 165014208,
      "csc_pension": 7213687
    },
    "up_impact": {
      "new_rate": 0.0709,
      "new_pvdbo": 160688701,
      "sensitivity_pvdbo": -4325507,
      "percentage_impact_pvdbo": -2.62,
      "new_csc_pension": 7019078,
      "sensitivity_csc": -194609,
      "percentage_impact_csc": -2.70
    },
    "down_impact": {
      "new_rate": 0.0509,
      "new_pvdbo": 169418597,
      "sensitivity_pvdbo": 4404389,
      "percentage_impact_pvdbo": 2.67,
      "new_csc_pension": 7411845,
      "sensitivity_csc": 198158,
      "percentage_impact_csc": 2.75
    }
  }
}
```

### **Step 5.2: Salary Increase Sensitivity**

```formula
💫 Formula: Salary Increase Impact
sensitivity_salary = csc_new - csc_base
percentage_impact = (sensitivity_salary / csc_base) × 100%
```

**Test Scenarios:**

- **Higher Inflation**: salary_increase_up = base_rate + 0.01 (+100 basis points)
- **Lower Inflation**: salary_increase_down = base_rate - 0.01 (-100 basis points)

**Expected Relationship:**

```
Salary Increase ↑ → Pension Benefits ↑ → Service Cost ↑ → Future PVDBO ↑
Salary Increase ↓ → Pension Benefits ↓ → Service Cost ↓ → Future PVDBO ↓
```

**Kondisi Aplikasi:**

- Untuk pension benefits only: salary increase affects projected final salary
- Other benefits use current salary (tidak terpengaruh)
- Recalculate dari Step 1 salary projections

**Contoh Perhitungan:**

```json
{
  "salary_increase_sensitivity": {
    "base_case": {
      "salary_increase_rate": 0.09,
      "total_pvdbo": 165014208,
      "csc_pension": 7213687
    },
    "higher_salary_increase": {
      "new_rate": 0.10,
      "new_pvdbo": 170115217,
      "sensitivity_pvdbo": 5101009,
      "percentage_impact_pvdbo": 3.09,
      "new_csc_pension": 7443321,
      "csc_sensitivity": 229634,
      "percentage_impact_csc": 3.18
    },
    "lower_salary_increase": {
      "new_rate": 0.08,
      "new_pvdbo": 159913200,
      "sensitivity_pvdbo": -5101008,
      "percentage_impact_pvdbo": -3.09,
      "new_csc_pension": 6984053,
      "csc_sensitivity": -229634,
      "percentage_impact_csc": -3.18
    }
  }
}
```


## 📅 **Maturity Profile Analysis**

**Kondisi Umum:** Menggunakan employee demographics dan benefit projections dari previous steps

### **Step 5.3: Undiscounted Cash Flow Projection**

```formula
💫 Formula: Undiscounted Maturity Profile
mp_undiscounted_sum = sum(mp_undiscounted_all_employees)

For portfolio aggregation:
mp_undiscounted_all_employees = sum(mp_undiscounted_i) for i = 1 to n employees

For each employee i:
mp_undiscounted_i = projected_pension_benefit_gross_i[usia_pensiun_i]
```

**Komponen:**

- **mp_undiscounted_sum**: Total portfolio undiscounted maturity profile value
- **mp_undiscounted_all_employees**: Array of undiscounted cash flows from all employees
- **mp_undiscounted_i**: Individual employee undiscounted cash flow at retirement
- **projected_pension_benefit_gross_i**: Gross pension benefit for employee i at retirement age
- **usia_pensiun_i**: Retirement age for employee i
- **n**: Total number of employees in the portfolio

### **Step 5.4: Discounted Cash Flow Projection**

```formula
💫 Formula: Discounted Maturity Profile
mp_discounted_sum = sum(mp_discounted_all_employees)

For portfolio aggregation:
mp_discounted_all_employees = sum(mp_discounted_i) for i = 1 to n employees

For each employee i:
mp_discounted_i = projected_pension_benefit_gross_i[usia_pensiun_i] × discount_factor_i[usia_pensiun_i]
```

**Komponen:**

- **mp_discounted_sum**: Total portfolio discounted maturity profile value
- **mp_discounted_all_employees**: Array of discounted cash flows from all employees
- **mp_discounted_i**: Individual employee discounted cash flow at retirement (present value)
- **projected_pension_benefit_gross_i**: Gross pension benefit for employee i at retirement age
- **discount_factor_i[usia_pensiun_i]**: Present value discount factor from valuation date to employee i's retirement age
- **usia_pensiun_i**: Retirement age for employee i
- **n**: Total number of employees in the portfolio

**Business Logic Note:**

- Undiscounted values represent nominal future cash flows
- Discounted values represent present value of future cash flows
- Portfolio aggregation sums individual employee projections
- Each employee may have different retirement ages and benefit amounts

### **Step 5.5: Cash Flow t Macaulay for Benefits**

#### **5.5.1 Macaulay Pension Benefit**

```formula
💫 Formula: Macaulay Pension Benefit
macaulay_pension_benefit_t = pension_benefit_gross_t x life_probability_t
```

**Kondisi Aplikasi:**

- Untuk usia_saat_valuasi < usia_pensiun: macaulay_pension_benefit = 0
- Untuk usia_saat_valuasi = usia_pensiun: gunakan formula

**Contoh Perhitungan Cash Flow Macaulay for Pension Benefit:**

```json
{
  "cash_flow_macaulay_pension_analysis": {
    "employee_profile": {
      "employee_id": "B0652",
      "usia_saat_valuasi": 53.08,
      "usia_pensiun": 55
    },
    "pension_cash_flow_projections": {
	  "age_53": {
		"status": "aktif_belum_pensiun",
	    "calculation_applicable": false,
	    "pension_benefit_gross": 0,
	    "life_probability": 1.000000,
	    "macaulay_pension_benefit": 0
	    },
	  "age_54": {
	    "status": "aktif_belum_pensiun",
	    "calculation_applicable": false,
	    "pension_benefit_gross": 0,
	    "life_probability": 0.995567,
	    "macaulay_pension_benefit": 0,
	    },
	  "age_55": {
	    "status": "usia_pensiun",
	    "calculation_applicable": true,
	    "pension_benefit_gross": 616593163,
	    "life_probability_retirement": 0.990727,
	    "macaulay_pension_benefit": 610875219,
	    }
    },
    "portfolio_contribution": {
      "calculations": "0 + 0 + 610,875,219 = 610,875,219",
      "cash_flow_sum": 610875219,
      "note": "Significant due to near-retirement status"
    },
    "validation": {
      "formula_check_age_55": "616,593,163 × 0.990727 = 610,875,219 ✓",
      "time_to_retirement": 1.92,
      "duration_logic": "Payment occurs at retirement (1.92 years from valuation)", 
      "economic_interpretation": "Single payment at retirement with 1.92-year duration"
    }
  }
}
```

#### **5.5.2 Macaulay Death Benefit**

```formula
💫 Formula: Macaulay Death Benefit
macaulay_death_benefit_t = death_benefit_gross_t x mortality_rate_t
```

**Kondisi Aplikasi:**

- Untuk usia_saat_valuasi ≥ usia_pensiun: macaulay_death_benefit = 0
- Untuk usia_saat_valuasi < usia_pensiun: gunakan formula

**Contoh Perhitungan Cash Flow Macaulay for Death Benefit:**

```json
{
  "cash_flow_macaulay_death_analysis": {
    "employee_profile": {
      "employee_id": "B0652",
      "usia_saat_valuasi": 53.08,
      "usia_pensiun": 55
    },
    "death_cash_flow_projections": {
	  "age_53": {
		"status": "aktif_belum_pensiun",
	    "calculation_applicable": true,
	    "death_benefit_gross": 614960000,
	    "mortality_rate": 0.004030,
	    "macaulay_death_benefit": 2478289
	    },
	  "age_54": {
	    "status": "aktif_belum_pensiun",
	    "calculation_applicable": true,
	    "death_benefit_gross": 675931400,
	    "mortality_rate": 0.004400,
	    "macaulay_death_benefit": 2974373
	    },
	  "age_55": {
	    "status": "usia_pensiun",
	    "calculation_applicable": false,
	    "death_benefit_gross": 0,
	    "mortality_rate": 0,
	    "macaulay_death_benefit": 0
	    }
    },
    "portfolio_contribution": {
      "calculations": "2,478,289 + 2,974,373 + 0 = 5,452,661",
      "cash_flow_sum": 5452661,
      "note": "Significant due to near-retirement status"
    },
    "validation": {
      "formula_check": "Age 53: 614,960,000 × 0.004030 = 2,478,289 ✓ | Age 54: 675,931,400 × 0.004400 = 2,974,373 ✓ | Age 55: 0 × 0 = 0 ✓",
      "time_to_retirement": 1.92,
      "duration_logic": "Death benefit cash flows concentrated in ages 53-54 (active years), zero at retirement age 55", 
      "economic_interpretation": "Expected death benefit payments weighted by mortality probability create portfolio cash flow for Macaulay duration calculation"
    }
  }
}
```

#### **5.5.3 Macaulay Disability Benefit**

```formula
💫 Formula: Macaulay Disability Benefit
macaulay_disability_benefit_t = disability_benefit_gross_t x disability_rate_t
```

**Kondisi Aplikasi:**

- Untuk usia_saat_valuasi ≥ usia_pensiun: macaulay_disability_benefit = 0
- Untuk usia_saat_valuasi < usia_pensiun: gunakan formula

**Contoh Perhitungan Cash Flow Macaulay for Disability Benefit:**

```json
{
  "cash_flow_macaulay_disability_analysis": {
    "employee_profile": {
      "employee_id": "B0652",
      "usia_saat_valuasi": 53.08,
      "usia_pensiun": 55
    },
    "disability_cash_flow_projections": {
	  "age_53": {
		"status": "aktif_belum_pensiun",
	    "calculation_applicable": true,
	    "disability_benefit_gross": 614960000,
	    "disability_rate": 0.000403,
	    "macaulay_disability_benefit": 247829
	    },
	  "age_54": {
	    "status": "aktif_belum_pensiun",
	    "calculation_applicable": true,
	    "disability_benefit_gross": 675931400,
	    "disability_rate": 0.000440,
	    "macaulay_disability_benefit": 297437
	    },
	  "age_55": {
	    "status": "usia_pensiun",
	    "calculation_applicable": false,
	    "disability_benefit_gross": 0,
	    "disability_rate": 0,
	    "macaulay_disability_benefit": 0
	    }
    },
    "portfolio_contribution": {
      "calculations": "247,829 + 297,437 + 0 = 545,266",
      "cash_flow_sum": 545266,
      "note": "Significant due to near-retirement status"
    },
    "validation": {
      "formula_check": "Age 53: 614,960,000 × 0.000403 = 247,829 ✓ | Age 54: 675,931,400 × 0.000440 = 297,437 ✓ | Age 55: 0 × 0 = 0 ✓",
      "time_to_retirement": 1.92,
      "duration_logic": "Disability benefit cash flows concentrated in ages 53-54 (active years), zero at retirement age 55", 
      "economic_interpretation": "Expected disability benefit payments weighted by disability probability create portfolio cash flow for Macaulay duration calculation"
    }
  }
}
```

#### **5.5.4 Macaulay Withdrawal Benefit**

```formula
💫 Formula: Macaulay Withdrawal Benefit
macaulay_withdrawal_benefit_t = withdrawal_benefit_gross_t x withdrawal_rate_t
```

**Kondisi Aplikasi:**

- Untuk usia_saat_valuasi ≥ usia_pensiun: macaulay_withdrawal_benefit = 0
- Untuk usia_saat_valuasi < usia_pensiun: gunakan formula

**Contoh Perhitungan Cash Flow Macaulay for Withdrawal Benefit:**

```json
{
  "cash_flow_macaulay_withdrawal_analysis": {
    "employee_profile": {
      "employee_id": "B0652",
      "usia_saat_valuasi": 53.08,
      "usia_pensiun": 55
    },
    "withdrawal_cash_flow_projections": {
	  "age_53": {
		"status": "aktif_belum_pensiun",
	    "calculation_applicable": true,
	    "withdrawal_benefit_gross": 58471400,
	    "withdrawal_rate": 0.000000,
	    "macaulay_withdrawal_benefit": 0
	    },
	  "age_54": {
	    "status": "aktif_belum_pensiun",
	    "calculation_applicable": true,
	    "withdrawal_benefit_gross": 63958826,
	    "withdrawal_rate": 0.000000,
	    "macaulay_withdrawal_benefit": 0
	    },
	  "age_55": {
	    "status": "usia_pensiun",
	    "calculation_applicable": false,
	    "withdrawal_benefit_gross": 0,
	    "withdrawal_rate": 0,
	    "macaulay_withdrawal_benefit": 0
	    }
    },
    "portfolio_contribution": {
      "calculations": "0 + 0 + 0 = 0",
      "cash_flow_sum": 0,
      "note": "Significant due to near-retirement status"
    },
    "validation": {
      "formula_check": "Age 53: 58,471,400 × 0.000000 = 0 ✓ | Age 54: 63,958,826 × 0.000000 = 0 ✓ | Age 55: 0 × 0 = 0 ✓",
      "time_to_retirement": 1.92,
      "duration_logic": "Withdrawal benefit cash flows concentrated in ages 53-54 (active years), zero at retirement age 55", 
      "economic_interpretation": "Expected withdrawal benefit payments weighted by withdrawal probability create portfolio cash flow for Macaulay duration calculation"
    }
  }
}
```

#### **5.5.5 Summary Cash Flow t for Future Service**

**Cash Flow Period 0 For Employee i**

```formula
💫 Formula: Cash Flow Current Age as Period 0 (Employee i)
cash_flow_i[period_0] = macaulay_pension_benefit_i[usia_saat_valuasi] + 
                          macaulay_death_benefit_i[usia_saat_valuasi] + 
                          macaulay_disability_benefit_i[usia_saat_valuasi] + 
                          macaulay_withdrawal_benefit_i[usia_saat_valuasi]
```

**Contoh Perhitungan:**

```json
{
  "cash_flow_current_service": {
    "employee_profile": {
      "employee_id": "B0652",
      "usia_saat_valuasi": 53.08,
      "usia_pensiun": 55,
      "future_service": 1.92
    },
    "cash_flow_projections": {
      "period_0": {
        "status": "aktif_belum_pensiun",
        "calculation_applicable": true,
        "macaulay_pension_benefit_53": 0,
        "macaulay_death_benefit_53": 2478289,
        "macaulay_disability_benefit_53": 247829,
        "macaulay_withdrawal_benefit_53": 0,
        "calculations": "0 + 2,478,289 + 247,829 + 0 = 2,726,118",
        "cash_flow_period_0": 2726118
      }
    },
    "portfolio_contribution": {
      "cash_flow_period_0": 2726118,
      "percentage_of_total_portfolio": "0.44%",
      "note": "Immediate benefits dominate at current age"
    },
    "validation": {
      "formula_check": "0 + 2,478,289 + 247,829 + 0 = 2,726,118 ✓",
      "time_to_retirement": 1.92,
      "economic_interpretation": "Cash flow of immediate benefit risks at current age"
    }
  }
}
```

**Cash Flow Period t For Employee i**

```formula
💫 Formula: Portfolio PV Cash Flow by Period t
cash_flow_i[period_t] = sum(employee_cash_flow_i) for all employees where round(future_service_i) = period_t

Where for each employee i:
employee_cash_flow_i = macaulay_pension_benefit_sum_i + 
                      sum(macaulay_death_benefit_age_i) + 
                      sum(macaulay_disability_benefit_age_i) + 
                      sum(macaulay_withdrawal_benefit_age_i)

```

**Contoh Perhitungan:**

```json
{
  "cash_flow_future_service": {
    "employee_profile": {
      "employee_id": "B0652",
      "usia_saat_valuasi": 53.08,
      "usia_pensiun": 55,
      "future_service": 1.92,
      "period_allocation_logic": "round(1.92) = 2, allocated to period_1"
    },
    "cash_flow_projections": {
      "period_0": {
        "status": "usia_saat_valuasi",
        "calculation_applicable": true,
        "cash_flow_period_0": 2726118
      },
      "period_1": {
        "projected_age": 54.08,
        "status": "allocated_retirement_period",
        "calculation_applicable": true,
        "macaulay_pension_benefit_sum": 610875219,
        "macaulay_death_benefit_54": 2974373,
        "macaulay_disability_benefit_54": 297437,
        "macaulay_withdrawal_benefit_54": 0,
        "calculations": "610,875,219 + 2,974,373 + 297,437 + 0 = 614,147,029",
        "cash_flow_period_1": 614147029
      },
      "period_2": {
        "projected_age": 55,
        "status": "usia_pensiun",
        "calculation_applicable": false,
        "cash_flow_period_2": 0
      }
    },
    "portfolio_contribution": {
      "cash_flow_period_1": 614147029,
      "percentage_of_total_portfolio": "99.56%",
      "note": "Pension benefit dominates allocated period"
    },
    "validation": {
      "formula_check": "610,875,219 + 2,974,373 + 297,437 + 0 = 614,147,029 ✓",
      "time_to_retirement": 1.92,
      "economic_interpretation": "Pension benefit allocated to period 1 based on future service rounding, plus projected immediate benefits"
    }
  }
}
```

_Cash Flow for Future Service disajikan lebih lengkap dalam [Macaulay Duration](macaulay_duration.md)_

## 🔍 **Variance Analysis & Reconciliation**

**Kondisi Umum:** Comparing current year results dengan prior year actual untuk understanding changes secara menyeluruh (akumulasi perhitungan all employees dalam perusahaan)

### **Step 5.6: Year-over-Year PVDBO Reconciliation**

```formula
💫 Formula: PVDBO Reconciliation
pvdbo_current = pvdbo_prior + service_cost + interest_cost ± actuarial_gains_losses - benefit_payments
```

**Komponen:**

- service_cost: Current year service increment dari Step 4
- interest_cost: PVDBO_prior × discount_rate_prior
- actuarial_gains_losses: Impact of assumption changes + experience variances
- benefit_payments: Actual payments made during the year

### **Step 5.7: Actuarial Gain/Loss Analysis**

```formula
💫 Formula: Actuarial Gain/Loss Breakdown
total_actuarial_gl = assumption_changes + experience_variances

assumption_changes = sum(impact_of_each_assumption_change)
experience_variances = expected_outcome - actual_outcome
```

**Common Sources of Actuarial Gains/Losses:**

| Source                              | Typical Impact | Gain/Loss Direction       | Business Driver       |
| ----------------------------------- | -------------- | ------------------------- | --------------------- |
| Discount rate increase              | -8% to -15%    | Actuarial gain            | Market interest rates |
| Salary increase higher than assumed | +5% to +12%    | Actuarial loss            | Inflation pressure    |
| Higher than expected mortality      | Mixed          | Depends on benefit design | Demographic changes   |
| Lower withdrawal rates              | +2% to +8%     | Actuarial loss            | Employee retention    |

**Contoh Variance Analysis Reporting:**

```json
{
  "variance_analysis": {
    "pvdbo_prior": 12199268076,
    "current_year_components": {
      "csc_current": 1211143745,
      "net_interest_cost": 427366148,
      "past_service_cost_vested": 59563158,
      "benefit_payments": -923898450,
      "actuarial_gains_losses": -2951489309
    },
    "pvdbo_current": 9422713104,
    "total_change": -8152006201,
    "percentage_change": -66.82
  }
}
```

**Referensi Actuarial Reporting Table:**

| No.    | EXPLANATION                                                                     | 30 Juni 2025      | 31 Desember 2024  | URAIAN                                                                |
| ------ | ------------------------------------------------------------------------------- | ----------------- | ----------------- | --------------------------------------------------------------------- |
| **1**  | **Basic Data and Assumptions**                                                  |                   |                   | **Data dan Asumsi**                                                   |
| 2      | Number of Employee                                                              | 199               | 197               | Jumlah Karyawan (orang)                                               |
| 3      | Monthly Wages for Permanent EEs                                                 | 1,679,691,000     | 1,433,211,000     | Jumlah Gaji Sebulan - Kywn Tetap                                      |
| 4      | Average Monthly Wages for Permanent EEs                                         | 8,440,658         | 7,275,183         | Rata-rata Gaji sebulan                                                |
| 5      | Average Age of Permanent EEs (Years)                                            | 39.53             | 39.98             | Rata-rata Usia (Tahun) - Kywn Tetap                                   |
| 6      | Average of Years of Service (Years) – Permanent EEs                             | 11.37             | 13.88             | Rata-rata masa kerja (Tahun) - Kywn Tetap                             |
| 7      | Average Future Service (years)                                                  | 9.22              | 10.31             | Rata-rata Masa Kerja Yang Akan Datang (Tahun)                         |
| 8      | Weighted Average Duration of the Defined Benefit Obligation (Macaulay Duration) | 8.72              | 8.67              | Rata-rata Tertimbang dari Kewajiban Imbalan Pasti (Macaulay Duration) |
| 9      | Discount Rate Beginning Period                                                  | 7.01%             | 6.2% s.d. 7.1%    | Tingkat Diskonto Awal Tahun                                           |
| 10     | Discount Rate Ending Period                                                     | 6.65%             | 7.01%             | Tingkat Diskonto Akhir Tahun                                          |
| 11     | Expected Rate of Return on Plan Assets at BoP (years)                           | 0.00%             | 0.00%             | Tingkat Harapan Investasi atas Aktiva Program (per tahun)             |
| 12     | Future Salary Increases (per annum)                                             | 9.00%             | 9.00%             | Tingkat Kenaikan Gaji (Per Tahun)                                     |
|        |                                                                                 |                   |                   |                                                                       |
| **13** | **Current Service Cost**                                                        | **1,211,143,745** | **1,460,586,150** | **Biaya jasa kini**                                                   |
| 14     | Total Benefit Paid in Year                                                      | (923,898,450)     | (2,490,646,917)   | Imbalan yang dibayarkan                                               |
| 15     | Company Contribution Paid in Year                                               | -                 | -                 | Iuran yang dibayarkan                                                 |
| 16     | Present Value of Obligation at BoP                                              | 12,199,268,076    | 12,560,522,672    | Nilai kini kewajiban awal periode                                     |
| 17     | Present Value of Obligation at EoP                                              | 9,422,713,104     | 12,199,268,076    | Nilai kini kewajiban akhir periode                                    |
| 18     | Past Service Cost - Non Vested                                                  | -                 | -                 | Biaya jasa lalu - non-vested                                          |
| 19     | Past Service Cost - Vested                                                      | 59,563,158        | -                 | Biaya jasa lalu - vested                                              |
| 20     | Fair Value of Plan Asset Program Beginning of Period                            | -                 | -                 | Nilai wajar aktiva program awal periode                               |
| 21     | Fair Value of Plan Asset Program End of Period                                  | -                 | -                 | Nilai wajar aktiva program akhir periode                              |
| 22     | Mortality Table                                                                 | TMI IV            | TMI IV            | Tabel Mortalita                                                       |
| 23     | Disability Rate                                                                 | 10% dari TMI IV   | 10% dari TMI IV   | Tingkat Cacat                                                         |
|        |                                                                                 |                   |                   |                                                                       |
| **24** | **Withdrawal Rate**                                                             |                   |                   | **Tingkat Pengunduran Diri**                                          |
|        | <= 19                                                                           | 0.0%              | 0.0%              |                                                                       |
|        | 20 - 29                                                                         | 0.6%              | 0.6%              |                                                                       |
|        | 30 - 34                                                                         | 0.3%              | 0.3%              |                                                                       |
|        | 35 - 39                                                                         | 0.2%              | 0.2%              |                                                                       |
|        | 40 - 50                                                                         | 0.1%              | 0.1%              |                                                                       |
|        | 51 - 52                                                                         | 0.1%              | 0.1%              |                                                                       |
|        | > 52                                                                            | 0.0%              | 0.0%              |                                                                       |
|        |                                                                                 |                   |                   |                                                                       |
| 25     | Actuarial Calculation Method **)                                                | PUC (IFRIC)       | PUC (IFRIC)       | Metode Perhitungan Aktuaria                                           |
| 26     | Normal Retirement Age (Years old)                                               | 55                | 55                | Usia Pensiun Normal (Tahun)                                           |

**Referensi Actuarial Gain/Losses Table:**

| No. | EXPLANATION                                      | 30 Juni 2025    | 31 Desember 2024 | URAIAN                                          |
| --- | ------------------------------------------------ | --------------- | ---------------- | ----------------------------------------------- |
| 1   | Other Comprehensive Income at BoP                | (202,519,153)   | 160,662,505      | Pendapatan Komprehensif lainnya awal periode    |
| 2   | Actuarial (Gain)/Loss at on Period - Obligation  | (2,951,489,309) | (363,181,658)    | Keuntungan/(kerugian) aktuaria - Kewajiban      |
| 3   | Actuarial (Gain)/Loss at on Period - Plan Assets | -               | -                | (Keuntungan)/kerugian aktuaria - Aktiva Program |
| 4   | Total Actuarial (Gain)/Loss at on Period         | (2,951,489,309) | (363,181,658)    | Total (Keuntungan)/kerugian aktuaria            |
| 5   | Other Comprehensive Income at EoP                | (3,154,008,462) | (202,519,153)    | Pendapatan Komprehensif lainnya akhir periode   |

_Referensi: [Tabel Utama](02f1_valuasi_tabel_utama.md)_

## 📊 **Comprehensive PSAK 219 Reporting**

### **Step 5.8: Executive Summary Preparation**

Key metrics untuk management dan board reporting:

```json
{
  "executive_summary": {
    "valuation_overview": {
      "total_employee_count": 199,
      "active_employees_calculated": 199,
      "tanggal_valuasi": "2025-06-30",
      "methodology": "Projected Unit Credit (IFRIC)"
    },
    "key_financial_results": {
      "total_pvdbo": 9422713104,
      "csc_current": 1211143745,
      "service_cost_as_percentage_of_payroll": 7.2
    },
    "demographic_profile": {
      "average_age": 39.53,
      "average_service_years": 11.37,
      "average_future_service": 9.22,
      "weighted_average_duration": 8.72
    }
  }
}
```

### **Step 5.9: PSAK 219 Disclosure Requirements**

```json
{
  "psak_219_disclosures": {
    "balance_sheet_recognition": {
      "pvdbo_current": 9422713104,
      "plan_assets": 0,
      "net_defined_benefit_liability": 9422713104,
      "asset_ceiling_effect": 0
    },
    "income_statement_recognition": {
      "csc_current": 1211143745,
      "net_interest_on_liability": 427366148,
      "past_service_cost_vested": 59563158,
      "immediate_recognition_past_service": -55426287,
      "curtailment_settlement": -484250818,
      "total_expense": 1098832788
    },
    "other_comprehensive_income": {
      "actuarial_gains_losses": -2951489309,
      "return_on_plan_assets": 0,
      "changes_in_asset_ceiling": 0,
      "total_oci": -2951489309
    },
    "reconciliation_pvdbo": {
      "pvdbo_prior": 12199268076,
      "csc_current": 1211143745,
      "interest_cost": 427366148,
      "past_service_cost_vested": 59563158,
      "benefit_payments": -923898450,
      "changes_in_benefit_plans": -114989445,
      "curtailment_settlement": -484250818,
      "actuarial_gains_losses": -2951489309,
      "pvdbo_current": 9422713104
    },
    "key_assumptions": {
      "discount_rate_beginning": 0.0701,
      "discount_rate_ending": 0.0665,
      "salary_increase_rate": 0.09,
      "mortality_table": "TMI IV",
      "disability_rate": "10% dari TMI IV",
      "usia_pensiun": 55,
      "macaulay_duration": 8.72
    },
    "sensitivity_disclosures": {
      "discount_rate_sensitivity": "±1% change impacts PVDBO by ±8.9%",
      "salary_increase_sensitivity": "±1% change impacts Service Cost by ±6.2%"
    }
  }
}
```

## ✅ **Quality Assurance & Validation**

### **Step 5.10: Red Flag Analysis**

```python
def identify_calculation_red_flags(results, employee_profile):
    """Identify potential calculation issues and unusual results"""
    
    warnings = []
    
    # PVDBO per employee reasonableness
    pvdbo_per_employee = results["total_pvdbo"] / employee_data["count"]
    if pvdbo_per_employee > 1_000_000:  # >1M per employee
        warnings.append(f"Very high PVDBO per employee: {pvdbo_per_employee:,.0f}")
    
    # Service cost as % of payroll
    service_cost_ratio = results["total_csc"] / employee_data["total_payroll"]
    if service_cost_ratio > 0.20:  # >20% of payroll
        warnings.append(f"High service cost ratio: {service_cost_ratio:.1%}")
    
    # Sensitivity analysis reasonableness
    discount_sensitivity = abs(results["sensitivity"]["discount_rate_impact"])
    if discount_sensitivity > 20:  # >20% for 1% rate change
        warnings.append(f"Unusual discount rate sensitivity: {discount_sensitivity:.1f}%")
    
    return warnings
```

### **Step 5.11: Final Validation Checklist**

```python
def final_validation_checklist(all_results):
    """Comprehensive final validation before reporting"""
    
    validation_results = {
        "mathematical_relationships": validate_math_consistency(all_results),
        "sensitivity_reasonableness": validate_sensitivity_ranges(all_results),
        "psak_219_completeness": validate_disclosure_completeness(all_results),
        "prior_period_consistency": validate_methodology_consistency(all_results),
        "business_logic": validate_business_reasonableness(all_results)
    }
    
    # Check for any failing validations
    failed_validations = [k for k, v in validation_results.items() if not v["passed"]]
    
    if failed_validations:
        raise ValidationError(f"Validation failures: {failed_validations}")
    
    return validation_results
```

### **Step 5.12: Documentation & Audit Trail**

**Required Documentation for Audit:**

1. **Methodology Documentation**
    
    - Calculation approach and assumptions
    - Data sources and validation procedures
    - System controls and quality checks
2. **Assumption Setting Documentation**
    
    - Rationale for each key assumption
    - Historical analysis supporting assumption selection
    - Sensitivity analysis results
3. **Results Documentation**
    
    - Complete calculation results by employee
    - Sensitivity analysis detailed results
    - Variance analysis and explanations
4. **Review and Approval Documentation**
    
    - Internal actuarial review sign-off
    - Management review and approval
    - External auditor review notes

## 🔄 **Integration & Final Output**

### **Step 5.13: Comprehensive Results Package**

Final deliverable integrating all previous steps:

```json
{
  "step5_comprehensive_results": {
    "valuation_summary": {
      "tanggal_valuasi": "2025-06-30",
      "total_employee_count": 199,
      "methodology": "Projected Unit Credit (IFRIC) with individual discount rates",
      "compliance_standard": "PSAK 219 (IAS 19 equivalent)"
    },
    "step1_employee_foundation_aggregates": {
      "average_usia_saat_valuasi": 39.53,
      "average_masa_kerja_lalu": 11.37,
      "average_future_service": 9.22,
      "total_monthly_salary": 1679691000,
      "average_monthly_salary": 8440658,
      "tingkat_kenaikan_gaji": 0.09,
      "usia_pensiun": 55
    },
    "step2_multiple_decrement_assumptions": {
      "mortality_table": "TMI IV",
      "disability_rate": "10% dari TMI IV",
      "withdrawal_rates_by_age": {
        "age_20_29": 0.006,
        "age_30_34": 0.003,
        "age_35_39": 0.002,
        "age_40_50": 0.001,
        "age_51_52": 0.001,
        "age_above_52": 0.000
      },
      "methodology": "corrected_probabilities_competing_risks"
    },
    "step3_benefit_calculation_program": {
      "benefit_program": "Company Policy (PP) enhanced benefits",
      "benefit_factors_table": "benefit_factors_pp",
      "pension_factor_basis": "masa_kerja_lalu_ifric",
      "immediate_benefits_basis": "masa_kerja_lalu",
      "tax_treatment": "progressive_tax_structure"
    },
    "step4_present_value_results": {
      "total_pvdbo": 9422713104,
      "total_csc": 1211143745,
      "discount_rate_method": "IGSYC individual future service matching",
      "weighted_average_duration": 8.72,
      "discount_rate_beginning": 0.0701,
      "discount_rate_ending": 0.0665,
      "service_ratio_methodology": "IFRIC for pension, full service for immediate benefits"
    },
    "step5_sensitivity_analysis": {
      "discount_rate_1_percent": {
        "up_impact": {
          "percentage_impact_pvdbo": -2.62,
          "percentage_impact_csc": -2.70
        },
        "down_impact": {
          "percentage_impact_pvdbo": 2.67,
          "percentage_impact_csc": 2.75
        }
      },
      "salary_increase_1_percent": {
        "higher_inflation": {
          "percentage_impact_pvdbo": 3.09,
          "percentage_impact_csc": 3.18
        },
        "lower_inflation": {
          "percentage_impact_pvdbo": -3.09,
          "percentage_impact_csc": -3.18
        }
      }
    },
    "financial_statement_impact": {
      "balance_sheet_recognition": {
        "net_defined_benefit_liability": 9422713104,
        "plan_assets": 0
      },
      "income_statement_recognition": {
        "csc_current": 1211143745,
        "net_interest_cost": 427366148,
        "past_service_cost_vested": 59563158,
        "immediate_recognition_past_service": -55426287,
        "curtailment_settlement": -484250818,
        "total_expense": 1098832788
      },
      "other_comprehensive_income": {
        "actuarial_gains_losses": -2951489309,
        "total_oci": -2951489309
      }
    },
    "pvdbo_reconciliation": {
      "pvdbo_prior": 12199268076,
      "csc_current": 1211143745,
      "interest_cost": 427366148,
      "benefit_payments": -923898450,
      "changes_in_benefit_plans": -114989445,
      "actuarial_gains_losses": -2951489309,
      "pvdbo_current": 9422713104,
      "total_change": -2776554972,
      "percentage_change": -22.75
    },
    "risk_assessment": {
      "primary_actuarial_risks": [
        {
          "risk_factor": "discount_rate_sensitivity",
          "impact_level": "Medium-High",
          "sensitivity_measure": "±2.6% PVDBO per 1% rate change"
        },
        {
          "risk_factor": "salary_increase_sensitivity",
          "impact_level": "Medium",
          "sensitivity_measure": "±3.1% PVDBO per 1% salary growth change"
        }
      ],
      "maturity_profile": {
        "mp_undiscounted_approach": "nominal_future_cash_flows",
        "mp_discounted_approach": "present_value_cash_flows",
        "concentration_analysis": "weighted by individual retirement ages"
      }
    },
    "validation_status": {
      "step1_employee_data": "passed",
      "step2_multiple_decrement": "passed",
      "step3_benefit_calculation": "passed",
      "step4_present_value": "passed",
      "step5_sensitivity_analysis": "passed",
      "psak_219_compliance": "passed",
      "audit_readiness": "confirmed"
    },
    "next_steps": {
      "next_tanggal_valuasi": "2026-06-30",
      "monitoring_requirements": [
        "Quarterly IGSYC yield curve updates",
        "Annual employee data refresh",
        "Semi-annual assumption reviews"
      ],
      "reporting_schedule": "Annual financial statements per PSAK 219"
    }
  }
}
```

---

📎 **Navigation:**

- ⬅️ [Step 4: Present Value Calculations](step04_pvfb_pvdbo.md)
- 🏠 [Master Guide](INDEX.md)
- 📊 [Reference Tables](assumptions_reference.md) | [Troubleshooting](troubleshooting_guide.md)

**🎉 Calculation Complete!** - Comprehensive actuarial valuation results ready for PSAK 219 financial reporting, management review, and audit procedures.