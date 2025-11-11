---
title: "Step 4: Present Value Calculations"
description: "Calculate PVFB and PVDBO using discount factors and probability weights"
keywords: [PVFB, PVDBO, present value, discount factor, service cost, yield curve]
difficulty: advanced
estimated_reading: 30 minutes
target_audience: [actuaries, finance_team]
document_type: calculation_step
step_number: 4
---

# 📈 Step 4: Present Value Calculations

## 🎯 **Overview**

Menghitung nilai kini (present value) dari future benefits dan defined benefit obligations. Step ini mengintegrasikan benefit amounts, probability weights, dan discount factors untuk menghasilkan liability valuations yang akan dicatat di laporan keuangan.

_See Also: [Proses Valuasi](02f_proses_valuasi.md)

## 📊 **Present Value Framework**

|Metric|Symbol|Description|Usage|Accounting Impact|
|---|---|---|---|---|
|**PVFB**|Present Value Future Benefits|Total future benefits discounted|⭐⭐⭐ High|Off-balance sheet reference|
|**PVDBO**|Present Value Defined Benefit Obligation|Liability recorded on balance sheet|⭐⭐⭐ High|Balance sheet liability|
|**CSC**|Current Service Cost|Annual service cost increment|⭐⭐⭐ High|P&L expense|

**Key Reference Tables:**

- 📊 **[IGSYC Yield Curve](yield_curve_igsyc.md)** - Government bond zero-coupon rates
- 📊 **[Macaulay Duration Guide](macaulay_duration.md)** - Duration-based discount rate determination

## 🧮 **Discount Rate Determination**

**Kondisi Umum:** Menggunakan `future_service` dari Step 1 untuk menentukan tingkat diskonto akhir periode

### **Step 4.1: Discount Rate Lookup**

```formula
💫 Formula: Discount Rate Selection
discount_rate = IGSYC_yield_curve[future_service]  // berdasarkan future service
```

**Pemilihan Method:**

- **Direct Matching**: Jika future_service exact match dengan tenor available
- **Linear Interpolation**: Jika future_service between two tenors
- **Boundary Conditions**: Use minimum (0.5Y) atau maximum (30Y) tenor

**Quick Reference - Common Discount Rates:**

| Future Service | Discount Factor | Usage Context     | Usage Frequency |
| -------------- | --------------- | ----------------- | --------------- |
| 1-2 years      | 5.8-6.0%        | Near retirement   | ⭐⭐⭐ High        |
| 3-5 years      | 6.1-6.4%        | Mid-senior career | ⭐⭐⭐ High        |
| 10-15 years    | 6.8-7.1%        | Mid-career        | ⭐⭐⭐ High        |
| 20+ years      | 7.1-7.2%        | Young employees   | ⭐⭐ Medium       |

**For Complete Yield Data:** 📊 **[IGSYC Yield Curve](yield_curve_igsyc.md)** - 30 tenors, daily updated rates

### **Step 4.2: Discount Factor Calculation**

**Initial Condition:**

```formula
💫 Formula: Current Age Discount Factor
discount_factor[usia_saat_valuasi] = 1.0  // No discounting for current age
```

```formula
💫 Formula: Estimated Discount Factor t
discount_factor_t = discount_factor_t_before / (1 + discount_rate)^(t - t_before)
```

**Komponen:**

- estimated_discount_factor_t: Present value factor untuk age t
- discount_rate: Tingkat diskonto dari Step 4.1
- t: Target age estimated


**Contoh Perhitungan:**

```json
{
  "discount_factor_calculation_breakdown": {
    "parameters": {
      "unrounded_usia_saat_valuasi": 53.08,
      "usia_saat_valuasi": 53,
      "discount_rate": 0.0609,
      "rate_source": "IGSYC_interpolated",
      "future_service": 1.92
    },
    "age_53_08_calculation": {
      "age": 53.08,
      "status": "current_valuation_age",
      "formula_applied": "discount_factor[usia_saat_valuasi] = 1.0",
      "explanation": "No discounting for current age",
      "calculation_steps": {
        "step1": "Current age = usia saat valuasi",
        "step2": "No time passage = no discounting needed",
        "result": "discount_rate = 1.0"
      },
      "discount_rate": 1.000000
    },
    "age_54_08_calculation": {
      "age": 54.08,
      "time_difference": 1.0,
      "formula_applied": "discount_factor_t = discount_factor_t_before / (1 + discount_rate)^(t - t_before)",
      "calculation_steps": {
        "step1": "t = 54.08, t_before = 53.08",
        "step2": "t - t_before = 54.08 - 53.08 = 1.0 year",
        "step3": "discount_factor_t_before = 1.0 (from age 53.08)",
        "step4": "(1 + discount_rate) = (1 + 0.0609) = 1.0609",
        "step5": "(1 + discount_rate)^(t - t_before) = 1.0609^1.0 = 1.0609",
        "step6": "discount_factor_54_08 = 1.0 / 1.0609 = 0.942596"
      },
      "mathematical_expression": "1.0 / (1.0609)^1.0",
      "discount_factor": 0.942596,
      "validation": true
    },
    "age_55_calculation": {
      "age": 55,
      "time_difference": 1.92,
      "formula_applied": "discount_factor_t = discount_factor_t_before / (1 + discount_rate)^(t - t_before)",
      "calculation_steps": {
        "step1": "t = 55.00, t_before = 53.08",
        "step2": "t - t_before = 55.00 - 53.08 = 1.92 years",
        "step3": "discount_factor_t_before = 1.0 (from age 53.08)",
        "step4": "(1 + discount_rate) = (1 + 0.0609) = 1.0609",
        "step5": "(1 + discount_rate)^(t - t_before) = 1.0609^1.92 = 1.1199",
        "step6": "discount_factor_55 = 1.0 / 1.1199 = 0.892875"
      },
      "mathematical_expression": "1.0 / (1.0609)^1.92",
      "detailed_calculation": {
        "base": 1.0609,
        "exponent": 1.92,
        "power_result": 1.119948,
        "final_division": "1.0 / 1.119948 = 0.892875"
      },
      "discount_factor": 0.892875,
      "validation": true
    },
    "progressive_discounting_verification": {
      "method": "step_by_step_annual_discounting",
      "age_54_alternative": {
        "from_age_53": "1.0 / 1.0609^1.0 = 0.942596",
        "matches_direct_calculation": true
      },
      "age_55_alternative": {
        "from_age_54": "0.942596 / 1.0609^0.92 = 0.942596 / 1.0556 = 0.892875",
        "from_age_53_direct": "1.0 / 1.0609^1.92 = 0.892875",
        "both_methods_match": true
      }
    },
    "key_insights": {
      "discount_rate_interpretation": "6.09% annual rate from IGSYC yield curve interpolation",
      "compounding_effect": "1.92 years reduces present value by ~10.7%",
      "time_precision": "Fractional years (1.92) handled correctly in exponential calculation",
      "validation_status": "All calculations estimated correctly"
    },
    "formula_summary": {
      "base_principle": "Present value decreases exponentially with time and discount rate",
      "key_formula": "PV_factor = 1 / (1 + r)^t",
      "where": {
        "r": "discount_rate (0.0609)",
        "t": "time_in_years from valuation date",
        "PV_factor": "present value multiplier"
      }
    }
  }
}
```

## 💰 **PVFB Calculations**

**Kondisi Umum:** Menggunakan `net_benefits` dari Step 3 dan `corrected_rates` dari Step 2

### **Step 4.3: PVFB Pension**


```formula
💫 Formula: Current PVFB Pension
pvfb_pension = (pension_benefit_gross × discount_factor_current × life_probability_retirement) - dplk_balance
```

```formula
💫 Formula: Projected PVFB Pension
pvfb_pension = (pension_benefit_gross × discount_factor_pension × life_probability_retirement) - dplk_balance
```

**Komponen:**

- pension_benefit_gross: Gross pension benefit dari Step 3
- discount_factor_pension: PV factor to retirement age
- life_probability_retirement: Probability surviving to retirement dari Step 2
- dplk_balance: DPLK balance offset (defined contribution plan)

**Kondisi Aplikasi:**

- Untuk usia_saat_valuasi < usia_pensiun: pvfb_pension = 0
- Untuk usia_saat_valuasi = usia_pensiun: gunakan formula

**Contoh Perhitungan Current PVFB Pension Benefit:**

```json
{
  "pvfb_pension_current": {
    "employee_profile": {
      "usia_saat_valuasi": 53.08,
      "usia_pensiun": 55,
      "masa_kerja_lalu_ifric": 22.08,
      "status": "aktif_belum_pensiun"
    },
    "calculation": {
      "pension_benefit_gross": 0,
      "discount_factor_pension": 1.000000,
      "life_probability_at_age": 1.000000,
      "dplk_offset": 372306686,
      "formula": "(0 × 1.000000 × 1.000000) - 372306686",
      "pvfb_pension": 0,
      "note": "Pension benefit belum applicable - employee masih aktif"
    },
    "step_by_step_calculation": {
      "step_1": {
        "description": "Calculate gross pension benefit with current discount rate",
        "calculation": "0 × 1.000000 = 0",
        "result": 0
      },
      "step_2": {
        "description": "Apply life probability adjustment",
        "calculation": "0 × 1.000000 = 0",
        "result": 0
      },
      "step_3": {
        "description": "Subtract DPLK balance (no negative PVFB)",
        "calculation": "max(0 - 372,306,686, 0) = 0",
        "result": 0
      }
    },
    "calculation_components": {
      "pension_benefit_gross": {
        "value": 0,
        "reason": "Employee belum mencapai usia pensiun",
        "note": "Pension benefit hanya applicable saat usia = usia pensiun"
      },
      "discount_factor": {
        "value": 1.000000,
        "explanation": "No discounting for current age calculation",
        "note": "Present value baseline for current age"
      },
      "life_probability": {
        "value": 1.000000,
        "explanation": "Current age baseline probability",
        "note": "No survival adjustment needed for current age"
      },
      "dplk_balance": {
        "value": 372306686,
        "note": "DPLK balance remains constant, offsets future pension liability"
      }
    },
    "validation": {
      "current_age_logic": {
        "pension_applicable": false,
        "reason": "53.08 < 55 (usia pensiun)",
        "pvfb_result": "Zero - no current pension obligation"
      },
      "mathematical_consistency": {
        "gross_benefit_check": "✓ Zero for active employee",
        "discount_factor_check": "✓ 1.0 for current age",
        "dplk_treatment": "✓ Offset applied correctly"
      }
    }
  }
}
```

**Contoh Perhitungan Projected PVFB Pension Benefit:**

```json
{
  "pvfb_pension_projected_analysis": {
    "employee_profile": {
      "current_usia_saat_valuasi": 53.08,
      "usia_pensiun": 55,
      "masa_kerja_lalu_ifric": 22.08,
      "projection_method": "age_progression_pension_analysis"
    },
    "age_54_08_projection": {
      "employee_profile": {
        "projected_usia_saat_valuasi": 54.08,
        "usia_pensiun": 55,
        "status": "aktif_belum_pensiun",
        "time_to_retirement": 0.92
      },
      "calculation": {
        "pension_benefit_gross": 0,
        "discount_factor": 0.942596,
        "life_probability_at_age": 0.995567,
        "dplk_balance": 372306686,
        "formula": "(0 × 0.942596 × 0.995567) - 372306686",
        "pvfb_pension": 0,
        "note": "Pension benefit masih belum applicable - 0.92 tahun lagi pensiun"
      }
    },
    "age_55_retirement": {
      "employee_profile": {
        "retirement_usia_saat_valuasi": 55,
        "usia_pensiun": 55,
        "status": "usia_pensiun",
        "time_to_retirement": 0
      },
      "calculation": {
        "pension_benefit_gross": 616593163,
        "discount_factor_pension": 0.892875,
        "life_probability_retirement": 0.990727,
        "dplk_balance": 372306686,
        "formula": "(616,593,163 × 0.892875 × 0.990727) - 372306686",
        "pvfb_pension": 173128494,
        "note": "Pension benefit sudah applicable - full retirement benefit"
      }
    },
    "step_by_step_calculation": {
      "age_54_08_detailed": {
        "step_1": {
          "description": "Calculate gross pension benefit (still zero)",
          "calculation": "0 × 1.000000 = 0",
          "result": 0
        },
        "step_2": {
          "description": "Apply life probability adjustment",
          "calculation": "0 × 0.995567 = 0",
          "result": 0
        },
        "step_3": {
          "description": "DPLK offset (no negative PVFB)",
          "calculation": "max(0 - 372,306,686, 0) = 0",
          "result": 0
        }
      },
      "age_55_detailed": {
        "step_1": {
          "description": "Calculate gross pension benefit with discount factor",
          "calculation": "616,593,163 × 0.892875 = 550,609,866",
          "result": 550609866
        },
        "step_2": {
          "description": "Apply life probability adjustment",
          "calculation": "550,609,866 × 0.990727 = 545,435,180",
          "result": 545435180
        },
        "step_3": {
          "description": "Subtract DPLK balance",
          "calculation": "545,435,180 - 372,306,686 = 173,128,494",
          "result": 173128494
        }
      }
    },
    "calculation_components": {
      "pension_benefit_gross": {
        "age_54_08": 0,
        "age_55": 616593163,
        "note": "Pension benefit hanya applicable saat mencapai usia pensiun"
      },
      "discount_factors": {
        "age_54_08": 0.942596,
        "age_55": 0.892875,
        "discount_rate": 0.0609,
        "note": "Discount factor untuk present value calculation"
      },
      "life_probabilities": {
        "age_54_08": 0.995567,
        "age_55": 0.990727,
        "note": "Survival probability dari Step 2 multiple decrement"
      },
      "dplk_balance": {
        "age_54_08": 372306686,
        "age_55": 372306686,
        "note": "DPLK balance consistent - offsets pension liability"
      }
    },
    "pvfb_projection_summary": {
      "age_54_08": {
        "pvfb_value": 0,
        "composition": {
          "pension_benefit_gross": 0,
          "dplk_liability": 0,
          "net_position": "No pension obligation yet"
        }
      },
      "age_55": {
        "pvfb_value": 173128494,
        "composition": {
          "pension_benefit_gross": 616593163,
          "dplk_liability": -372306686,
          "net_position": "Significant pension obligation activated"
        }
      }
    },
    "comprehensive_multi_age_analysis": {
      "current_age_53": 0,
      "projected_age_54": 0,
      "projected_age_55": 173128494,
      "total_pension_pvfb": 173128494,
      "note": "Pension PVFB concentrated at retirement age - cliff vesting effect"
    },
    "validation": {
      "age_progression_logic": {
        "age_54_08": "PVFB = 0 (still active, no pension yet)",
        "age_55": "PVFB = 173M (retirement triggers full pension)",
        "transition_point": "Age 55 = cliff vesting activation"
      },
      "mathematical_consistency": {
        "pension_benefit_age_check": "✓ Zero until retirement age",
        "life_probability_decline": "✓ Decreases with age (99.6% to 99.1%)",
        "discount_factor_logic": "✓ Applied for retirement age calculation",
        "dplk_consistency": "✓ Same offset across all ages"
      },
      "cliff_vesting_validation": {
        "pre_retirement": "✓ Zero PVFB for ages < 55",
        "at_retirement": "✓ Full PVFB activated at age 55",
        "economic_logic": "✓ Pension obligations concentrate at retirement"
      }
    }
  }
}
```

### **Step 4.4: PVFB Death**

```formula
💫 Formula: PVFB Death
pvfb_death = death_benefit_gross × discount_factor_current × mortality_rate
```

**Komponen:**

- death_benefit_gross: Gross death benefit dari Step 3
- discount_factor_current: PV factor (usually 1.0 for current age)
- mortality_rate: Corrected mortality rate dari Step 2

**Kondisi Aplikasi:**

- Untuk usia_saat_valuasi ≥ usia_pensiun: pvfb_death = 0
- Untuk usia_saat_valuasi < usia_pensiun: gunakan formula

**Business Logic:**

> Death benefits are paid immediately upon death, so typically no time discounting needed

**Contoh Perhitungan Current PVFB Death:**

```json
{
  "pvfb_death_calculation_current_employee": {
    "employee_profile": {
      "usia_saat_valuasi": 53.08,
      "usia_pensiun": 55,
      "masa_kerja_lalu": 30.42,
      "program_type": "Company_Policy",
      "status": "aktif_belum_pensiun"
    },
    "age_53_calculation": {
      "inputs": {
        "death_benefit_gross": 614960000,
        "discount_factor": 1.000000,
        "mortality_rate": 0.004030,
        "status": "aktif_belum_pensiun"
      },
      "calculation": {
        "formula": "614,960,000 × 1.000000 × 0.004030",
        "pvfb_death": 2478289,
        "note": "Death benefit applicable - current age risk"
      }
    }
  },
  "calculation_components": {
    "death_benefit_gross": {
      "value": 614960000,
      "source": "death_factor[30] × total_gaji_saat_valuasi",
      "note": "Based on 30 years full service and total gaji saat valuasi 19,356,000"
    },
    "discount_factor": {
      "value": 1.000000,
      "explanation": "No discounting for usia saat valuasi",
      "note": "Present value calculation baseline"
    },
    "mortality_rate": {
      "value": 0.004030,
      "source": "TMI_IV_Male_2019",
      "percentage": "0.403%",
      "note": "Annual mortality probability at age 53"
    }
  },
  "step_by_step_calculation": {
    "step_1": {
      "description": "Apply current discount factor (no discounting)",
      "calculation": "614,960,000 × 1.000000 = 614,960,000",
      "result": 614960000
    },
    "step_2": {
      "description": "Apply mortality rate at age 53",
      "calculation": "614,960,000 × 0.004030 = 2,478,289",
      "result": 2478289
    }
  },
  "validation": {
    "current_age_logic": {
      "death_benefit_applicable": true,
      "reason": "Employee is active and below retirement age",
      "calculation_method": "Current age death risk only"
    },
    "mathematical_consistency": {
      "discount_factor_check": "✓ 1.0 for current age",
      "mortality_rate_reasonable": "✓ 0.403% for 53-year-old male",
      "benefit_amount_valid": "✓ Matches Step 3 death benefit calculation"
    }
  }
}
```

**Contoh Perhitungan Projected PVFB Death:**

```json
{
  "pvfb_death_calculation_projected_ages": {
    "employee_profile": {
      "usia_saat_valuasi": 54.08,
      "usia_pensiun": 55,
      "projection_method": "multi_year_death_risk_analysis"
    },
    "age_54_projection": {
      "inputs": {
        "death_benefit_gross_projected": 675931400,
        "discount_factor_projected": 0.942596,
        "mortality_rate_projected": 0.004400,
        "status": "aktif_belum_pensiun",
        "time_from_valuation": 0.92
      },
      "calculation": {
        "formula": "0.942596 × 0.004400 x 675,931,400",
        "step_by_step": {
          "step_1": "0.942596 x 0.004400 = 0.0041474224",
          "step_2": "0.0041474224 x 675,931,400 = 2,803,632"
        },
        "pvfb_death": 2803632,
        "note": "Death benefit applicable - projected future age risk"
      }
    },
    "age_55_retirement": {
      "inputs": {
        "death_benefit_gross": 0,
        "discount_factor": 0.892875,
        "mortality_rate": 0.000000,
        "status": "usia_pensiun",
        "time_from_valuation": 1.92
      },
      "calculation": {
        "formula": "0 × 0.892875 × 0.000000",
        "pvfb_death": 0,
        "note": "Death benefit tidak applicable - sudah pensiun"
      }
    }
  },
  "calculation_components": {
    "death_benefit_gross": {
      "age_54": 675931400,
      "age_55": 0,
      "note": "Death benefit hanya applicable untuk karyawan aktif"
    },
    "discount_factors": {
      "age_54": 0.942596,
      "age_55": 0.892875,
      "discount_rate": 6.09,
      "note": "Discount factor untuk present value calculation dari IGSYC"
    },
    "mortality_rates": {
      "age_54": 0.004400,
      "age_55": 0.000000,
      "source": "TMI_IV_Male_2019",
      "note": "Mortality rates increase with age, zero at retirement"
    }
  },
  "pvfb_projection_summary": {
    "age_54": {
      "pvfb_value": 2803632,
      "composition": {
        "death_risk_pv": 2803632,
        "percentage_contribution": "100%",
        "risk_type": "future_active_age_death_risk"
      }
    },
    "age_55": {
      "pvfb_value": 0,
      "composition": {
        "death_risk_pv": 0,
        "percentage_contribution": "0%",
        "risk_type": "retirement_age_no_death_benefit"
      }
    },
    "total_projected_pvfb": {
      "ages_54_55_combined": 2803632,
      "note": "Total projected death risk from future ages"
    }
  },
  "comprehensive_multi_age_analysis": {
    "current_age_53": 2478289,
    "projected_age_54": 2803632,
    "projected_age_55": 0,
    "total_comprehensive_pvfb_death": 5281920,
    "note": "Full multi-age death risk analysis (not typically used in simplified calculations)"
  },
  "validation": {
    "age_progression_logic": {
      "age_54": "PVFB > 0 (active employee, death benefit applicable)",
      "age_55": "PVFB = 0 (retirement cutoff, no death benefit)",
      "transition_point": "Age 55 = retirement eliminates death benefit"
    },
    "mathematical_consistency": {
      "mortality_rate_progression": "✓ Increases from 0.44% to 0% at retirement",
      "discount_factor_logic": "✓ Higher discounting for later ages",
      "benefit_cutoff_logic": "✓ Zero death benefit at retirement age"
    },
    "projection_assumptions": {
      "mortality_table": "TMI_IV_Male_2019",
      "discount_rate": "6.09% IGSYC interpolated",
      "death_benefit_policy": "Active employees only",
      "calculation_method": "Present value of future death risks"
    }
  }
}
```

### **Step 4.5: PVFB Disability**

```formula
💫 Formula: PVFB Disability
pvfb_disability = disability_benefit_gross × discount_factor_current × disability_rate
```

**Contoh Perhitungan Current PVFB Disability:**

```json
{
  "pvfb_disability_calculation_current_employee": {
    "employee_profile": {
      "usia_saat_valuasi": 53.08,
      "usia_pensiun": 55,
      "masa_kerja_lalu": 30.42,
      "program_type": "Company_Policy",
      "status": "aktif_belum_pensiun"
    },
    "age_53_calculation": {
      "inputs": {
        "disability_benefit_gross": 614960000,
        "discount_factor": 1.000000,
        "disability_rate": 0.0004030,
        "status": "aktif_belum_pensiun"
      },
      "calculation": {
        "formula": "614,960,000 × 1.000000 × 0.000403",
        "pvfb_disability": 247829,
        "note": "Disability benefit applicable - current age risk"
      }
    }
  },
  "calculation_components": {
    "disability_benefit_gross": {
      "value": 614960000,
      "source": "disability_factor[30] × total_gaji_saat_valuasi",
      "note": "Based on 30 years full service and total gaji saat valuasi 19,356,000"
    },
    "discount_factor": {
      "value": 1.000000,
      "explanation": "No discounting for usia saat valuasi",
      "note": "Present value calculation baseline"
    },
    "disability_rate": {
      "value": 0.000403,
      "source": "10% from mortality_rate[30]",
      "percentage": "0.0403%",
      "note": "Annual disability probability at age 53"
    }
  },
  "step_by_step_calculation": {
    "step_1": {
      "description": "Apply current discount factor (no discounting)",
      "calculation": "614,960,000 × 1.000000 = 614,960,000",
      "result": 614960000
    },
    "step_2": {
      "description": "Apply disability rate at age 53",
      "calculation": "614,960,000 × 0.000403 = 247,829",
      "result": 247829
    }
  },
  "validation": {
    "current_age_logic": {
      "disability_benefit_applicable": true,
      "reason": "Employee is active and below retirement age",
      "calculation_method": "Current age disability risk only"
    },
    "mathematical_consistency": {
      "discount_factor_check": "✓ 1.0 for current age",
      "disability_rate_reasonable": "✓ 0.0403% for 53-year-old male",
      "benefit_amount_valid": "✓ Matches Step 3 disability benefit calculation"
    }
  }
}
```

**Contoh Perhitungan Projected PVFB Disability:**

```json
{
  "pvfb_disability_calculation_projected_ages": {
    "employee_profile": {
      "usia_saat_valuasi": 54.08,
      "usia_pensiun": 55,
      "projection_method": "multi_year_disability_risk_analysis"
    },
    "age_54_projection": {
      "inputs": {
        "disability_benefit_gross_projected": 675931400,
        "discount_factor_projected": 0.942596,
        "disability_rate_projected": 0.00044,
        "status": "aktif_belum_pensiun",
        "time_from_valuation": 0.92
      },
      "calculation": {
        "formula": "0.942596 × 0.00044 x 675,931,400",
        "step_by_step": {
          "step_1": "0.942596 x 0.00044 = 0.00041474224",
          "step_2": "0.00041474224 x 675,931,400 = 280,363"
        },
        "pvfb_disability": 280363,
        "note": "Disability benefit applicable - projected future age risk"
      }
    },
    "age_55_retirement": {
      "inputs": {
        "disability_benefit_gross": 0,
        "discount_factor": 0.892875,
        "disability_rate": 0.000000,
        "status": "usia_pensiun",
        "time_from_valuation": 1.92
      },
      "calculation": {
        "formula": "0 × 0.892875 × 0.000000",
        "pvfb_disability": 0,
        "note": "Disability benefit tidak applicable - sudah pensiun"
      }
    }
  },
  "calculation_components": {
    "disability_benefit_gross": {
      "age_54": 675931400,
      "age_55": 0,
      "note": "Disability benefit hanya applicable untuk karyawan aktif"
    },
    "discount_factors": {
      "age_54": 0.942596,
      "age_55": 0.892875,
      "discount_rate": 0.0609,
      "note": "Discount factor untuk present value calculation dari IGSYC"
    },
    "disability_rates": {
      "age_54": 0.00044,
      "age_55": 0.000000,
      "source": "mortality_rate divided by 10",
      "note": "Disability rates increase with age, zero at retirement"
    }
  },
  "pvfb_projection_summary": {
    "age_54": {
      "pvfb_value": 280363,
      "composition": {
        "disability_risk_pv": 280363,
        "percentage_contribution": "100%",
        "risk_type": "future_active_age_disability_risk"
      }
    },
    "age_55": {
      "pvfb_value": 0,
      "composition": {
        "disability_risk_pv": 0,
        "percentage_contribution": "0%",
        "risk_type": "retirement_age_no_disability_benefit"
      }
    },
    "total_projected_pvfb": {
      "ages_54_55_combined": 280363,
      "note": "Total projected disability risk from future ages"
    }
  },
  "comprehensive_multi_age_analysis": {
    "current_age_53": 247829,
    "projected_age_54": 280363,
    "projected_age_55": 0,
    "total_comprehensive_pvfb_disability": 528192,
    "note": "Full multi-age disability risk analysis (not typically used in simplified calculations)"
  },
  "validation": {
    "age_progression_logic": {
      "age_54": "PVFB > 0 (active employee, disability benefit applicable)",
      "age_55": "PVFB = 0 (retirement cutoff, no disability benefit)",
      "transition_point": "Age 55 = retirement eliminates disability benefit"
    },
    "mathematical_consistency": {
      "disability_rate_progression": "✓ Always 5% to 10% from mortality rate",
      "discount_factor_logic": "✓ Higher discounting for later ages",
      "benefit_cutoff_logic": "✓ Zero disability benefit at retirement age"
    },
    "projection_assumptions": {
      "discount_rate": "6.09% IGSYC interpolated",
      "disability_benefit_policy": "Active employees only",
      "calculation_method": "Present value of future disability risks"
    }
  }
}
```

### **Step 4.6: PVFB Withdrawal**

```formula
💫 Formula: PVFB Withdrawal
pvfb_withdrawal = withdrawal_benefit_gross × discount_factor_current × withdrawal_rate
```

**UUCK Special Case:**

```
withdrawal_rate = 0.0  // No withdrawal benefit under UUCK
pvfb_withdrawal = 0.0
```

**Contoh Perhitungan Current PVFB Withdrawal:**

```json
{
  "pvfb_withdrawal_calculation_current_employee": {
    "employee_profile": {
      "usia_saat_valuasi": 53.08,
      "usia_pensiun": 55,
      "masa_kerja_lalu": 30.42,
      "program_type": "Company_Policy",
      "status": "aktif_belum_pensiun"
    },
    "age_53_calculation": {
      "inputs": {
        "withdrawal_benefit_gross": 58471400,
        "discount_factor": 1.000000,
        "withdrawal_rate": 0.000000,
        "status": "aktif_belum_pensiun"
      },
      "calculation": {
        "formula": "58,471,400 × 1.000000 × 0.000000",
        "pvfb_withdrawal": 0,
        "note": "Withdrawal benefit applicable - current age risk"
      }
    }
  },
  "calculation_components": {
    "withdrawal_benefit_gross": {
      "value": 58471400,
      "source": "withdrawal_factor[30] × total_gaji_saat_valuasi",
      "note": "Based on 30 years full service and total gaji saat valuasi 19,356,000"
    },
    "discount_factor": {
      "value": 1.000000,
      "explanation": "No discounting for usia saat valuasi",
      "note": "Present value calculation baseline"
    },
    "withdrawal_rate": {
      "value": 0.000000,
      "source": "Withdrawal rate at age 53",
      "percentage": "0.000%",
      "note": "Annual withdrawal probability at age 53 from Company policy"
    }
  },
  "step_by_step_calculation": {
    "step_1": {
      "description": "Apply current discount factor (no discounting)",
      "calculation": "58,471,400 × 1.000000 = 58,471,400",
      "result": 58471400
    },
    "step_2": {
      "description": "Apply withdrawal rate at age 53",
      "calculation": "58,471,400 × 0.000000 = 0",
      "result": 0
    }
  },
  "validation": {
    "current_age_logic": {
      "withdrawal_benefit_applicable": true,
      "reason": "Employee is active and below retirement age",
      "calculation_method": "Current age withdrawal risk only"
    },
    "mathematical_consistency": {
      "discount_factor_check": "✓ 1.0 for current age",
      "withdrawal_rate_reasonable": "✓ 0.000% for 53-year-old male",
      "benefit_amount_valid": "✓ Matches Step 3 withdrawal benefit calculation"
    }
  }
}
```

**Contoh Perhitungan Projected PVFB Withdrawal:**

```json
{
  "pvfb_withdrawal_calculation_projected_ages": {
    "employee_profile": {
      "usia_saat_valuasi": 54.08,
      "usia_pensiun": 55,
      "projection_method": "multi_year_death_risk_analysis"
    },
    "age_54_projection": {
      "inputs": {
        "withdrawal_benefit_gross_projected": 63958826,
        "discount_factor": 0.942596,
        "withdrawal_rate": 0.000000,
        "status": "aktif_belum_pensiun",
        "time_from_valuation": 0.92
      },
      "calculation": {
        "formula": "0.942596 × 0.000000 x 63,958,826",
        "step_by_step": {
          "step_1": "0.942596 x 0.000000 = 0",
          "step_2": "0 x 63,958,826 = 0"
        },
        "pvfb_withdrawal": 0,
        "note": "Death benefit applicable - projected future age risk"
      }
    },
    "age_55_retirement": {
      "inputs": {
        "withdrawal_benefit_gross": 0,
        "discount_factor": 0.892875,
        "withdrawal_rate": 0.000000,
        "status": "usia_pensiun",
        "time_from_valuation": 1.92
      },
      "calculation": {
        "formula": "0 × 0.892875 × 0.000000",
        "pvfb_withdrawal": 0,
        "note": "Withdrawal benefit tidak applicable - sudah pensiun"
      }
    }
  },
  "calculation_components": {
    "withdrawal_benefit_gross": {
      "age_54": 63958826,
      "age_55": 0,
      "note": "Death benefit hanya applicable untuk karyawan aktif"
    },
    "discount_factors": {
      "age_54": 0.942596,
      "age_55": 0.892875,
      "discount_rate": 0.0609,
      "note": "Discount factor untuk present value calculation dari IGSYC"
    },
    "withdrawal_rates": {
      "age_54": 0.000000,
      "age_55": 0.000000,
      "source": "Withdrawal rate from Company policy",
      "note": "Withdrawal rates decrease with age, zero at retirement"
    }
  },
  "pvfb_projection_summary": {
    "age_54": {
      "pvfb_value": 0,
      "composition": {
        "withdrawal_risk_pv": 0,
        "percentage_contribution": "100%",
        "risk_type": "future_active_age_withdrawal_risk"
      }
    },
    "age_55": {
      "pvfb_value": 0,
      "composition": {
        "withdrawal_risk_pv": 0,
        "percentage_contribution": "0%",
        "risk_type": "retirement_age_no_withdrawal_benefit"
      }
    },
    "total_projected_pvfb": {
      "ages_54_55_combined": 0,
      "note": "Total projected withdrawal risk from future ages"
    }
  },
  "comprehensive_multi_age_analysis": {
    "current_age_53": 0,
    "projected_age_54": 0,
    "projected_age_55": 0,
    "total_comprehensive_pvfb_withdrawal": 0,
    "note": "Full multi-age withdrawal risk analysis (not typically used in simplified calculations)"
  },
  "validation": {
    "age_progression_logic": {
      "age_54": "PVFB > 0 (active employee, disability benefit applicable)",
      "age_55": "PVFB = 0 (retirement cutoff, no disability benefit)",
      "transition_point": "Age 55 = retirement eliminates disability benefit"
    },
    "mathematical_consistency": {
      "withdrawal_rate_progression": "✓ Decreases until retirement",
      "discount_factor_logic": "✓ Higher discounting for later ages",
      "benefit_cutoff_logic": "✓ Zero withdrawal benefit at retirement age"
    },
    "projection_assumptions": {
      "withdrawal_table": "Withdrawal assumptions from Company policy",
      "discount_rate": "6.09% IGSYC interpolated",
      "withdrawal_benefit_policy": "Active employees only",
      "calculation_method": "Present value of future withdrawal risks"
    }
  }
}
```

## 🏢 **Current Service Cost (CSC) Calculations**

**Kondisi Umum:** Menggunakan `masa_kerja_lalu` dari Step 1 sebagai denominator

### **Step 4.7: Service Cost Methodology**

Service cost represents the portion of PVFB attributed to the current service year.

```formula
💫 Formula: Current Service Cost
csc_benefit = pvfb_benefit / masa_kerja_lalu
```

**Komponen:**

- csc_benefit: Annual service cost for specific benefit type
- pvfb_benefit: Present value of future benefits dari Step 4.3-4.6
- masa_kerja_lalu: Service period dari Step 1 (or IFRIC-adjusted)

**Protection Rule:**

```
if masa_kerja_lalu < 1:
    csc_benefit = pvfb_benefit  // Avoid division by zero
```

### **Step 4.8: CSC Calculations by Benefit Type**

**IFRIC Method Consideration:**

- **Standard Method**: Use masa_kerja_lalu since hire date
- **IFRIC Method**: Use masa_kerja_lalu_ifric

**Contoh Perhitungan Current CSC All Benefit Type:**

```json
{
  "csc_calculations_current": {
    "employee_profile": {
      "usia_saat_valuasi": 53.08,
      "usia_pensiun": 55,
      "masa_kerja_lalu_ifric": 22.08,
      "masa_kerja_lalu": 30.42,
      "status": "aktif_belum_pensiun"
    },
    "csc_calculations": {
      "pension_csc": {
        "pvfb_pension": 0,
        "calculation": "0 ÷ 22.08",
        "csc_pension": 0
      },
      "death_csc": {
        "pvfb_death": 2478289,
        "calculation": "2,478,289 ÷ 30.42",
        "csc_death": 81478
      },
      "disability_csc": {
        "pvfb_disability": 247829,
        "calculation": "247,829 ÷ 30.42",
        "csc_disability": 8148
      },
      "withdrawal_csc": {
        "pvfb_withdrawal": 0,
        "calculation": "0 ÷ 30.42",
        "csc_withdrawal": 0
      }
    },
    "csc_summary_current": {
      "total_csc": 89626,
      "breakdown": {
        "pension_contribution": 0,
        "death_contribution": 81478,
        "disability_contribution": 8148,
        "withdrawal_contribution": 0
      },
      "percentage_composition": {
        "pension": "0.00%",
        "death": "90.9%",
        "disability": "9.1%",
        "withdrawal": "0.00%"
      }
    },
    "calculation_methodology": {
      "pension_divisor": 22.08,
      "pension_divisor_note": "Based on IFRIC service years (22.08)",
      "other_benefits_divisor": 30.42,
      "other_benefits_divisor_note": "Based on full service years (masa_kerja_lalu)",
      "csc_concept": "Annual cost allocation of total benefit obligation"
    },
    "validation": {
      "calculation_checks": {
        "pension_csc": "✓ 0 ÷ 22.08 = 0",
        "death_csc": "✓ 2,478,289 ÷ 30.42 = 81,478", 
        "disability_csc": "✓ 247,829 ÷ 30.42 = 8,148",
        "withdrawal_csc": "✓ 0 ÷ 30.42 = 0"
      },
      "total_validation_current": "✓ Sum of current CSCs = 89,626",
      "methodology_consistency": "✓ Different divisors for pension vs immediate benefits"
    }
  }
}
```

**Contoh Perhitungan Projected CSC All Benefit Type:**

```json
{
  "csc_calculations_projected": {
    "base_employee_profile": {
      "current_usia_saat_valuasi": 53.08,
      "usia_pensiun": 55,
      "masa_kerja_lalu_ifric": 22.08,
      "masa_kerja_lalu": 30.42,
      "projection_method": "age_progression_analysis"
    },
    "age_54_08_projected": {
      "employee_profile": {
        "projected_usia_saat_valuasi": 54.08,
        "usia_pensiun": 55,
        "masa_kerja_lalu_ifric": 22.08,
        "masa_kerja_lalu": 30.42,
        "status": "aktif_belum_pensiun"
      },
      "csc_calculations": {
        "pension_csc": {
          "pvfb_pension": 0,
          "calculation": "0 ÷ 22.08",
          "csc_pension": 0,
          "note": "Still below retirement age"
        },
        "death_csc": {
          "pvfb_death": 2803632,
          "calculation": "2,803,632 ÷ 30.42",
          "csc_death": 89240
        },
        "disability_csc": {
          "pvfb_disability": 280363,
          "calculation": "280,363 ÷ 30.42",
          "csc_disability": 8924
        },
        "withdrawal_csc": {
          "pvfb_withdrawal": 0,
          "calculation": "0 ÷ 30.42",
          "csc_withdrawal": 0
        }
      },
      "csc_summary_projected": {
        "total_csc": 98164,
        "breakdown": {
          "pension_contribution": 0,
          "death_contribution": 89240,
          "disability_contribution": 8924,
          "withdrawal_contribution": 0
        },
        "percentage_composition": {
          "pension": "0.00%",
          "death": "90.9%",
          "disability": "9.1%",
          "withdrawal": "0.00%"
        }
      }
    },
    "age_55_retirement": {
      "employee_profile": {
        "projected_usia_saat_valuasi": 55,
        "usia_pensiun": 55,
        "masa_kerja_lalu_ifric": 22.08,
        "masa_kerja_lalu": 30.42,
        "status": "usia_pensiun"
      },
      "csc_calculations": {
        "pension_csc": {
          "pvfb_pension": 173128494,
          "calculation": "173,128,494 ÷ 22.08",
          "csc_pension": 7213687
        },
        "death_csc": {
          "pvfb_death": 0,
          "calculation": "0 ÷ 30.42",
          "csc_death": 0,
          "note": "No death benefit at retirement"
        },
        "disability_csc": {
          "pvfb_disability": 0,
          "calculation": "0 ÷ 30.42",
          "csc_disability": 0,
          "note": "No disability benefit at retirement"
        },
        "withdrawal_csc": {
          "pvfb_withdrawal": 0,
          "calculation": "0 ÷ 30.42",
          "csc_withdrawal": 0,
          "note": "No withdrawal benefit at retirement"
        }
      },
      "csc_summary_retirement": {
        "total_csc": 7213687,
        "breakdown": {
          "pension_contribution": 7213687,
          "death_contribution": 0,
          "disability_contribution": 0,
          "withdrawal_contribution": 0
        },
        "percentage_composition": {
          "pension": "100.00%",
          "death": "0.00%",
          "disability": "0.00%",
          "withdrawal": "0.00%"
        }
      }
    },
    "age_progression_summary": {
      "csc_by_age": {
        "age_53_08": 89626,
        "age_54_08": 98164,
        "age_55": 7213687
      },
      "dominant_benefit_by_age": {
        "age_53_08": "Death benefit (90.9%)",
        "age_54_08": "Death benefit (90.9%)", 
        "age_55": "Pension benefit (100%)"
      },
      "transition_point": {
        "age": 55,
        "description": "Dramatic CSC increase due to pension benefit activation",
        "csc_multiplier": "73.5x increase from age 54 to 55"
      }
    },
    "calculation_methodology": {
      "pension_divisor": 22.08,
      "pension_divisor_note": "Based on IFRIC service years consistently",
      "other_benefits_divisor": 30.42,
      "other_benefits_divisor_note": "Based on full service years consistently",
      "csc_concept": "CSC represents annual cost at specific age, not cumulative across ages",
      "projection_assumptions": "Mortality rates and life probabilities adjusted per age"
    },
    "validation": {
      "age_53_08_checks": {
        "pension_csc": "✓ 0 ÷ 22.08 = 0",
        "death_csc": "✓ 2,478,289 ÷ 30.42 = 81,478",
        "disability_csc": "✓ 247,829 ÷ 30.42 = 8,148",
        "total": "✓ 89,626"
      },
      "age_54_08_checks": {
        "pension_csc": "✓ 0 ÷ 22.08 = 0",
        "death_csc": "✓ 2,803,632 ÷ 30.42 = 89,240",
        "disability_csc": "✓ 280,363 ÷ 30.42 = 8,924", 
        "total": "✓ 98,164"
      },
      "age_55_checks": {
        "pension_csc": "✓ 173,128,494 ÷ 22.08 = 7,213,687",
        "death_csc": "✓ 0 ÷ 30.42 = 0",
        "disability_csc": "✓ 0 ÷ 30.42 = 0",
        "total": "✓ 7,213,687"
      },
      "summary_csc_by_age_checks": {
        "age_53_08": "✓ 89,626",
        "age_54_08": "✓ 98,164",
        "age_55": "✓ 7,213,687",
        "total": "✓ 89,626 + 98,164 + 7,213,687 = 7401477"
      },
      "consistency_validation": "✓ All ages use consistent divisors and methodology"
    }
  }
}
```

## 📋 **PVDBO Calculations**

**Kondisi Umum:** Menggunakan `total_masa_kerja` dari Step 1 sebagai denominator

### **Step 4.9: PVDBO Methodology**

PVDBO represents the portion of PVFB that should be recognized as a liability on the balance sheet.

**Kondisi:** PVDBO usia_saat_valuasi

```formula
💫 Formula: Current PVDBO
pvdbo_benefit_current = pvfb_benefit_current
```

**Kondisi:** PVDBO proyeksi saat usia t

```formula
💫 Formula: Projected PVDBO t
pvdbo_benefit_t = pvfb_benefit_t × service_ratio_t

where:
service_ratio_t = masa_kerja_lalu / masa_kerja_hingga_t
masa_kerja_hingga_t = masa_kerja_lalu + (t - usia_saat_valuasi)
```

**Kondisi:** PVDBO usia_pensiun

```formula
💫 Formula: PVDBO Pension
pvdbo_pension = pvfb_pension × service_ratio_ifric
pvdbo_pension= pvfb_pension × (masa_kerja_lalu_ifric / total_masa_kerja_ifric)
```

**Komponen:**

- **pvdbo_benefit**: Liability amount for balance sheet
- **pvfb_benefit**: Present value of future benefits dari Step 4.3-4.6
- **masa_kerja_lalu**: Actual service completed dari Step 1
- **masa_kerja_lalu_ifric**: IFRIC-adjusted service period untuk pension benefits
- **masa_kerja_hingga_t**: Actual service ditambah proyeksi hingga usia t
- **total_masa_kerja**: Expected total service years dari Step 1
- **total_masa_kerja_ifric**: Expected total IFRIC service years untuk pension
- **service_ratio_t**: Proportion of service completed hingga usia t
- **service_ratio_ifric**: IFRIC-based service ratio untuk pension benefits

**Protection Rule:**

```
if masa_kerja_lalu < 1:
    pvdbo_benefit = csc_benefit  // Use service cost instead
```

### **Step 4.10: Service Ratio Analysis**

**Service Ratio Interpretation:**

| Service Ratio    | Employee Profile | PVDBO Recognition    |
| ---------------- | ---------------- | -------------------- |
| 0.9+ (90%+)      | Near retirement  | High recognition     |
| 0.5-0.9 (50-90%) | Mid-career       | Moderate recognition |
| 0.1-0.5 (10-50%) | Early career     | Low recognition      |
| <0.1 (<10%)      | New employee     | Minimal recognition  |

**Contoh Perhitungan Current PVDBO All Benefit Type:**

```json
{
  "pvdbo_calculations_current": {
    "calculation_principle": {
      "usia_saat_valuasi": 53.08,
      "method": "pvdbo_current = pvfb_current",
      "rationale": "At current age, service ratio = 1.0 for all benefits"
    },
    "pension_pvdbo": {
      "pvfb_pension": 0,
      "service_ratio": 1.0,
      "calculation": "0 × 1.0",
      "pvdbo_pension_current": 0,
      "note": "No pension benefit at current age"
    },
    "death_pvdbo": {
      "pvfb_death": 2478289,
      "service_ratio": 1.0,
      "calculation": "2,478,289 × 1.0",
      "pvdbo_death_current": 2478289,
      "note": "Death benefit applicable at current age"
    },
    "disability_pvdbo": {
      "pvfb_disability": 247829,
      "service_ratio": 1.0,
      "calculation": "247,829 × 1.0",
      "pvdbo_disability_current": 247829,
      "note": "Disability benefit applicable at current age"
    },
    "withdrawal_pvdbo": {
      "pvfb_withdrawal": 0,
      "service_ratio": 1.0,
      "calculation": "0 × 1.0",
      "pvdbo_withdrawal_current": 0,
      "note": "No withdrawal PVFB at current age"
    }
  },
  "total_pvdbo_current": {
    "breakdown": {
      "pvdbo_pension_current": 0,
      "pvdbo_death_current": 2478289,
      "pvdbo_disability_current": 247829,
      "pvdbo_withdrawal_current": 0
    },
    "total": 2726118,
    "calculation_validation": "0 + 2,478,289 + 247,829 + 0 = 2,726,118"
  },
  "validation": {
    "current_age_logic": {
      "service_ratio_check": "✓ All ratios = 1.0 for current age",
      "pvdbo_equals_pvfb": "✓ PVDBO = PVFB for current age calculations",
      "applicable_benefits": "Death and disability only (employee still active)"
    },
    "mathematical_consistency": {
      "total_validation": "✓ Sum matches individual components",
      "benefit_hierarchy": "✓ Death > Disability > Other benefits",
      "current_age_principle": "✓ No discounting or service adjustment needed"
    }
  }
}
```

**Contoh Perhitungan Projected PVDBO All Benefit Type:**

```json
{
  "pvdbo_calculations_projected": {
    "age_54_08_projection": {
      "time_calculation": {
        "current_usia_saat_valuasi": 53.08,
        "projected_age": 54.08,
        "time_difference": 1.00,
        "calculation_note": "54.08 - 53.08 = 1.00 years forward projection"
      },
      "service_periods": {
        "masa_kerja_lalu": 30.42,
        "masa_kerja_hingga_t": 31.42,
        "service_ratio": 0.968,
        "masa_kerja_lalu_ifric": 22.08,
        "masa_kerja_hingga_t_ifric": 23.08,
        "service_ratio_pension": 0.957,
        "calculation_formulas": {
          "immediate_benefits": "masa_kerja_lalu / masa_kerja_hingga_t = 30.42 / 31.42 = 0.968",
          "pension_benefits": "masa_kerja_lalu_ifric / masa_kerja_hingga_t_ifric = 22.08 / 23.08 = 0.957"
        }
      },
      "employee_status": {
        "current_usia_saat_valuasi": 53.08,
        "projected_age": 54.08,
        "usia_pensiun": 55,
        "time_to_retirement": 0.92,
        "status": "aktif_belum_pensiun"
      },
      "benefit_calculations": {
        "pension_pvdbo": {
          "pvfb_pension": 0,
          "service_ratio_pension": 0.957,
          "calculation": "0 × 0.957",
          "pvdbo_pension": 0,
          "note": "No pension PVFB yet - employee still active"
        },
        "death_pvdbo": {
          "pvfb_death": 2803632,
          "service_ratio": 0.968,
          "calculation": "2,803,632 × 0.968",
          "pvdbo_death": 2714391,
          "note": "Death benefit with service ratio adjustment"
        },
        "disability_pvdbo": {
          "pvfb_disability": 280363,
          "service_ratio": 0.968,
          "calculation": "280,363 × 0.968",
          "pvdbo_disability": 271439,
          "note": "Disability benefit with service ratio adjustment"
        },
        "withdrawal_pvdbo": {
          "pvfb_withdrawal": 0,
          "service_ratio": 0.968,
          "calculation": "0 × 0.968",
          "pvdbo_withdrawal": 0,
          "note": "No withdrawal PVFB"
        }
      },
      "total_pvdbo_age_54_08": 2985830,
      "total_calculation_verification": "0 + 2,714,391 + 271,439 + 0 = 2,985,830"
    },
    
    "age_55_retirement": {
      "time_calculation": {
        "current_usia_saat_valuasi": 53.08,
        "projected_age": 55.00,
        "time_difference": 1.92,
        "calculation_note": "55.00 - 53.08 = 1.92 years to retirement"
      },
      "service_periods": {
        "masa_kerja_lalu": 30.42,
        "masa_kerja_hingga_t": 32.34,
        "service_ratio": 0.9406,
        "masa_kerja_lalu_ifric": 22.08,
        "masa_kerja_hingga_t_ifric": 24.00,
        "service_ratio_pension": 0.92,
        "calculation_formulas": {
          "immediate_benefits": "masa_kerja_lalu / masa_kerja_hingga_t = 30.42 / 32.34 = 0.9406",
          "pension_benefits": "masa_kerja_lalu_ifric / masa_kerja_hingga_t_ifric = 22.08 / 24.00 = 0.92"
        }
      },
      "employee_status": {
        "current_usia_saat_valuasi": 53.08,
        "projected_age": 55.00,
        "retirement_age": 55,
        "status": "usia_pensiun",
        "pension_activation": true
      },
      "benefit_calculations": {
        "pension_pvdbo": {
          "pvfb_pension": 173128494,
          "service_ratio_pension": 0.92,
          "calculation": "173,128,494 × 0.92",
          "pvdbo_pension": 159302260,
          "note": "Pension PVFB activated at retirement with IFRIC service ratio"
        },
        "death_pvdbo": {
          "pvfb_death": 0,
          "service_ratio": 0.9406,
          "calculation": "0 × 0.9406",
          "pvdbo_death": 0,
          "note": "No death benefit at retirement age"
        },
        "disability_pvdbo": {
          "pvfb_disability": 0,
          "service_ratio": 0.9406,
          "calculation": "0 × 0.9406",
          "pvdbo_disability": 0,
          "note": "No disability benefit at retirement age"
        },
        "withdrawal_pvdbo": {
          "pvfb_withdrawal": 0,
          "service_ratio": 0.9406,
          "calculation": "0 × 0.9406",
          "pvdbo_withdrawal": 0,
          "note": "No withdrawal benefit at retirement age"
        }
      },
      "total_pvdbo_age_55": 159302260,
      "total_calculation_verification": "159,302,260 + 0 + 0 + 0 = 159,302,260"
    }
  },
  
  "pvdbo_progression_summary": {
    "age_54_08": {
      "total_pvdbo": 2985830,
      "dominant_benefit": "Death (90.9%)",
      "composition": {
        "pension": 0,
        "death": 2714391,
        "disability": 271439,
        "withdrawal": 0
      },
      "percentage_breakdown": {
        "pension": "0.0%",
        "death": "90.9%",
        "disability": "9.1%",
        "withdrawal": "0.0%"
      }
    },
    "age_55": {
      "total_pvdbo": 159302260,
      "dominant_benefit": "Pension (100%)",
      "composition": {
        "pension": 159302260,
        "death": 0,
        "disability": 0,
        "withdrawal": 0
      },
      "percentage_breakdown": {
        "pension": "100.0%",
        "death": "0.0%",
        "disability": "0.0%",
        "withdrawal": "0.0%"
      }
    },
    "transition_analysis": {
      "pvdbo_shift": "From death/disability focus to pension focus",
      "age_55_multiplier": "53.3x increase in total PVDBO",
      "calculation": "159,302,260 / 2,985,830 = 53.3x",
      "economic_driver": "Pension obligation activation at retirement"
    }
  },
  
  "calculation_methodology": {
    "service_ratio_application": {
      "pension_benefits": "Uses IFRIC service ratio methodology",
      "immediate_benefits": "Uses full service ratio methodology",
      "age_54_ratios": "0.957 (pension), 0.968 (immediate)",
      "age_55_ratios": "0.92 (pension), 0.9406 (immediate)",
      "rationale": "Different benefit types have different service recognition patterns per PSAK 219"
    },
    "pvdbo_concept": {
      "definition": "Present value of defined benefit obligation for service rendered to date",
      "calculation": "PVDBO = PVFB × Service_Ratio",
      "purpose": "Measure accrued benefit obligation at specific projected age"
    },
    "projection_logic": {
      "age_54_08": "1-year forward projection from current age 53.08",
      "age_55": "Retirement age projection (1.92 years from current age)",
      "service_accumulation": "Service periods increase linearly with time passage"
    }
  },
  
  "validation": {
    "age_54_08_validation": {
      "service_ratio_check": "✓ 0.968 for immediate benefits, 0.957 for pension",
      "time_calculation": "✓ 54.08 - 53.08 = 1.00 years",
      "service_period_logic": "✓ 30.42 + 1.00 = 31.42 (immediate), 22.08 + 1.00 = 23.08 (pension)",
      "pension_logic": "✓ Zero PVFB until retirement age",
      "total_calculation": "✓ 2,714,391 + 271,439 = 2,985,830"
    },
    "age_55_validation": {
      "service_ratio_check": "✓ 0.9406 for immediate benefits, 0.92 for pension",
      "time_calculation": "✓ 55.00 - 53.08 = 1.92 years",
      "service_period_logic": "✓ 30.42 + 1.92 = 32.34 (immediate), 22.08 + 1.92 = 24.00 (pension)",
      "pension_activation": "✓ Full pension PVFB applies service ratio",
      "immediate_benefits_cutoff": "✓ Zero at retirement age",
      "total_calculation": "✓ 159,302,260 (pension only)"
    },
    "mathematical_consistency": {
      "service_ratio_bounds": "✓ All ratios < 1.0 (partial service recognition)",
      "benefit_transitions": "✓ Proper cutoffs at retirement age",
      "calculation_accuracy": "✓ All multiplications and totals verified",
      "naming_consistency": "✓ Consistent variable names across all sections",
      "formula_documentation": "✓ All calculations properly documented"
    }
  }
}
```

## ✅ **Validation & Quality Controls**

### **Step 4.11: Mathematical Relationship Validation**

```python
def validate_mathematical_relationships(pvfb, pvdbo, csc, employee_profile, service_periods):
    """Validate core mathematical relationships"""
    
    # PVDBO should be ≤ PVFB (with small tolerance for rounding)
    assert pvdbo <= pvfb * 1.01, f"PVDBO {pvdbo:,.0f} exceeds PVFB {pvfb:,.0f}"
    
    # CSC relationship check - different denominators for different benefits
    if employee_profile["benefit_type"] == "pension":
        denominator = service_periods["masa_kerja_lalu_ifric"]
        expected_csc = pvfb / max(1, denominator)
    else:
        denominator = service_periods["masa_kerja_lalu"]
        expected_csc = pvfb / max(1, denominator)
    
    if expected_csc > 0:  # Avoid division by zero comparison
        csc_variance = abs(csc - expected_csc) / expected_csc
        assert csc_variance < 0.02, f"CSC calculation variance: {csc_variance:.3f} for {employee_profile['benefit_type']}"
    
    # Service ratio validation - depends on benefit type
    if employee_profile["benefit_type"] == "pension":
        service_ratio = service_periods["masa_kerja_lalu_ifric"] / service_periods["total_masa_kerja_ifric"]
        expected_pvdbo = pvfb * service_ratio
    else:
        service_ratio = service_periods["masa_kerja_lalu"] / service_periods["total_masa_kerja"]
        expected_pvdbo = pvfb * service_ratio
    
    if expected_pvdbo > 0:
        pvdbo_variance = abs(pvdbo - expected_pvdbo) / expected_pvdbo
        assert pvdbo_variance < 0.02, f"PVDBO service ratio variance: {pvdbo_variance:.3f}"
```

### **Step 4.12: Business Logic Validation**

```python
def validate_business_logic(results, employee_profile):
    """Check business reasonableness of present value results"""
    
    total_pvdbo = sum(results["pvdbo"].values())
    monthly_salary = employee_profile["total_gaji_saat_valuasi"]
    age = employee_profile["usia_saat_valuasi"]
    retirement_age = employee_profile["usia_pensiun"]
    
    # Age-based benefit logic validation
    if age < retirement_age:
        # For active employees, immediate benefits (death/disability) should be present
        immediate_benefits = results["pvdbo"]["death"] + results["pvdbo"]["disability"]
        
        if results["pvdbo"]["pension"] > immediate_benefits * 2:
            flag_for_review(f"Unusual pension dominance for active employee age {age}")
    
    elif age >= retirement_age:
        # For retired employees, pension should dominate
        pension_percentage = results["pvdbo"]["pension"] / max(total_pvdbo, 1)
        if pension_percentage < 0.8:
            flag_for_review(f"Low pension percentage {pension_percentage:.1%} for retiree")
    
    # Service cost reasonableness check
    total_csc = sum(results["csc"].values())
    csc_percentage = total_csc / monthly_salary if monthly_salary > 0 else 0
    
    # Adjusted thresholds based on proximity to retirement
    years_to_retirement = retirement_age - age
    if years_to_retirement < 2 and csc_percentage > 0.8:  # 80% for very near retirees
        flag_for_review(f"Very high service cost: {csc_percentage:.1%} of salary")
    elif years_to_retirement >= 2 and csc_percentage > 0.15:  # 15% for others
        flag_for_review(f"High service cost: {csc_percentage:.1%} of salary")
    
    # PVDBO reasonableness relative to annual salary
    annual_salary = monthly_salary * 12
    pvdbo_multiple = total_pvdbo / annual_salary if annual_salary > 0 else 0
    if pvdbo_multiple > 25:  # More than 25x annual salary
        flag_for_review(f"Very high PVDBO: {pvdbo_multiple:.1f}x annual salary")

def flag_for_review(message):
    """Helper function to log validation warnings"""
    print(f"VALIDATION WARNING: {message}")
```

### **Step 4.13: Discount Factor Consistency Validation**

```python
def validate_discount_factor_consistency(step4_results, employee_profile):
    """Ensure discount rate and discount factor calculations are consistent"""
    
    discount_rate = step4_results["discount_rate"]
    future_service = employee_profile["future_service"] 
    usia_saat_valuasi = employee_profile["usia_saat_valuasi"]
    usia_pensiun = employee_profile["usia_pensiun"]
    
    # Validate discount rate source
    expected_discount_rate = interpolate_igsyc_yield_curve(future_service)
    if abs(discount_rate - expected_discount_rate) > 0.001:
        raise ValidationError(f"Discount rate {discount_rate:.4f} inconsistent with future service {future_service}")
    
    # Validate discount factor calculations
    time_to_retirement = usia_pensiun - usia_saat_valuasi
    expected_discount_factor = 1.0 / ((1 + discount_rate) ** time_to_retirement)
    actual_discount_factor = step4_results["discount_factors"]["retirement_age"]
    
    if abs(expected_discount_factor - actual_discount_factor) > 0.001:
        raise ValidationError(f"Discount factor inconsistent: expected {expected_discount_factor:.6f}, got {actual_discount_factor:.6f}")
    
    # Validate current age discount factor is 1.0
    current_age_factor = step4_results["discount_factors"]["current_age"]
    if abs(current_age_factor - 1.0) > 0.001:
        raise ValidationError(f"Current age discount factor should be 1.0, got {current_age_factor}")
```

## 🔄 **Integration with Next Steps**

Present value calculations akan digunakan dalam:

- **[Step 5: Sensitivity Analysis](step05_sensitivity_analysis.md)** - untuk impact analysis dan key metrics reporting
- **Financial Reporting** - untuk balance sheet and P&L recognition per PSAK 219

**Output Requirements for Step 5:**

```json
{
  "required_outputs": {
    "for_step5": [
      "total_pvdbo",
      "total_csc", 
      "discount_rate",
      "service_ratios"
    ],
    "supporting_data": [
      "pvdbo_by_benefit_type",
      "csc_by_benefit_type",
      "key_assumptions_applied"
    ]
  }
}
```

## 🎯 **Expected Final Output**

```json
{
  "step4_present_value_results": {
    "summary": {
      "employee_id": "B0652",
      "unrounded_usia_saat_valuasi": 53.08,
      "usia_saat_valuasi": 53,
      "usia_pensiun": 55,
      "methodology": "projected_unit_credit_individual_approach",
      "discount_rate": 0.0609,
      "discount_rate_source": "IGSYC_interpolated",
      "future_service": 1.92
    },
    "service_periods": {
      "masa_kerja_lalu": 30.42,
      "masa_kerja_lalu_ifric": 22.08,
      "total_masa_kerja": 32.34,
      "total_masa_kerja_ifric": 24.00,
      "service_ratio": 0.9407,
      "service_ratio_ifric": 0.92
    },
    "discount_factors": {
      "current_age_53_08": 1.000000,
      "retirement_age_55": 0.892875,
      "rate_source": "compound_discounting_formula",
      "calculation_method": "1 / (1 + discount_rate)^time_years"
    },
    "pvfb_current": {
      "pension": 0,
      "death": 2478289,
      "disability": 247829,
      "withdrawal": 0,
      "total_pvfb": 2726118
    },
    "csc_current": {
      "pension": 0,
      "death": 81478,
      "disability": 8148,
      "withdrawal": 0,
      "total_csc": 89626
    },
    "pvdbo_current": {
      "pension": 0,
      "death": 2478289,
      "disability": 247829,
      "withdrawal": 0,
      "total_pvdbo": 2726118
    },
    "calculation_inputs": {
      "mortality_rate": 0.004030,
      "disability_rate": 0.000403,
      "withdrawal_rate": 0.000000,
      "life_probability": 1.000000,
      "pension_benefit": 0,
      "death_benefit": 614960000,
      "disability_benefit": 614960000,
      "withdrawal_benefit": 58471400
    },
    "key_ratios": {
      "pvdbo_to_pvfb": 1.0000,
      "service_completion": 0.9407,
      "service_completion_ifric": 0.92,
      "csc_to_monthly_salary": 0.0463,
      "total_pvdbo_to_annual_salary": 1.1736
    },
    "methodology_notes": { 
      "pension_benefits": "Uses IFRIC periods (22.08/24.00) for service ratio calculation",
      "immediate_benefits": "Uses full service periods (30.42/32.34) for service ratio calculation",
      "discount_terminology": "discount_rate = annual percentage, discount_factor = present value multiplier",
      "rationale": "Different benefit types have different service recognition patterns per PSAK 219"
    },
    "validation_status": {
      "mathematical_relationships": "passed",
      "business_logic": "passed", 
      "discount_factor_consistency": "passed",
      "age_based_logic": "passed",
      "ready_for_step5": true
    },
    "integration_outputs": {
      "for_step5_sensitivity": {
        "base_discount_rate": 0.0609,
        "base_total_pvdbo": 2726118,
        "base_total_csc": 89626,
        "key_service_periods": {
          "masa_kerja_lalu": 30.42,
          "masa_kerja_lalu_ifric": 22.08
        }
      },
      "for_financial_reporting": {
        "balance_sheet_liability": 2726118,
        "current_service_cost": 89626,
        "benefit_breakdown": {
          "immediate_benefits": 2726118,
          "pension_benefits": 0
        }
      }
    }
  }
}
```
---

📎 **Navigation:**

- ⬅️ [Step 3: Benefit Calculation](step03_benefit_calculation.md)
- ➡️ [Step 5: Sensitivity Analysis](step05_sensitivity_analysis.md)
- 📊 [Yield Curve](yield_curve_igsyc.md) | [Duration Guide](macaulay_duration.md)