---
title: Troubleshooting Guide - PSAK 219 Calculations
description: Common issues, error handling, and problem resolution for actuarial calculations
keywords:
  - troubleshooting
  - error handling
  - validation
  - debugging
  - common issues
  - PSAK 219
  - actuarial errors
  - calculation problems
  - data validation
  - RAG errors
difficulty:
  - intermediate
estimated_reading: 25 minutes
target_audience:
  - actuaries
  - developers
  - finance_team
  - system_administrators
  - RAG_users
document_type: troubleshooting_guide
---

---

# 🛠️ Troubleshooting Guide - PSAK 219 Steps 01-06

## 🎯 **Overview**

Step-specific troubleshooting guide untuk mendiagnosis dan menyelesaikan masalah dalam setiap tahap perhitungan valuasi aktuaria PSAK 219, dari employee data foundation hingga sensitivity analysis.

## 📋 **Step-by-Step Error Diagnosis**

### **🔧 Step 1: Employee Data Foundation**

#### **Issue 1.1: Age Calculation Precision Errors**

**Symptoms:**

```
Warning: usia_saat_valuasi 53.083 rounds to 53, but calculation shows 52
OR
Error: usia_mulai_masa_kerja 22.58 inconsistent with dates
```

**Root Causes & Solutions:**

```python
def diagnose_age_calculation_errors(tanggal_lahir, tanggal_masuk_kerja, tanggal_valuasi):
    # Check month difference calculation precision
    months_diff_valuasi = calculate_months_difference(tanggal_lahir, tanggal_valuasi)
    unrounded_sia_saat_valuasi = months_diff_valuasi / 12
    usia_saat_valuasi = int(unrounded_usia_saat_valuasi)
    
    months_diff_hire = calculate_months_difference(tanggal_lahir, tanggal_masuk_kerja)
    usia_mulai_masa_kerja = months_diff_hire / 12
    
    # Common precision issues
    if abs(usia_saat_valuasi - usia_saat_valuasi_rounded) > 0.1:
        return {
            "issue": "age_precision_boundary",
            "solution": "Use consistent rounding: int(usia_saat_valuasi) for integer ages",
            "example": f"{usia_saat_valuasi:.3f} should round to {usia_saat_valuasi_rounded}"
        }
    
    # Date format validation
    date_formats = [tanggal_lahir, tanggal_masuk_kerja, tanggal_valuasi]
    for date_str in date_formats:
        if not validate_date_format(date_str, "YYYY-MM-DD"):
            return {
                "issue": "invalid_date_format",
                "solution": f"Convert {date_str} to YYYY-MM-DD format",
                "required_format": "2026-01-01"
            }
```

#### **Issue 1.2: IFRIC vs Standard Service Calculation Mix-up**

**Symptoms:**

```
Error: masa_kerja_lalu_ifric 22.08 > masa_kerja_lalu 30.42 (impossible)
OR
Warning: Using masa_kerja_lalu for pension calculation (should use masa_kerja_lalu_ifric)
```

**Diagnostic Logic:**

```python
def validate_ifric_calculation(usia_pensiun, usia_mulai_masa_kerja, usia_saat_valuasi, 
                              masa_kerja_lalu, tanggal_masuk_kerja, tanggal_valuasi):
    min_usia = usia_pensiun - 24
    
    # Calculate standard service period
    months_diff = calculate_months_difference(tanggal_masuk_kerja, tanggal_valuasi)
    calculated_masa_kerja_lalu = months_diff / 12
    
    # IFRIC Condition 1: min_usia <= usia_mulai_masa_kerja
    if min_usia <= usia_mulai_masa_kerja:
        expected_masa_kerja_lalu_ifric = calculated_masa_kerja_lalu
        method = "full_service_recognition"
    
    # IFRIC Condition 2: usia_mulai_masa_kerja < min_usia <= usia_saat_valuasi  
    elif usia_mulai_masa_kerja < min_usia <= usia_saat_valuasi:
        expected_masa_kerja_lalu_ifric = usia_saat_valuasi - min_usia
        method = "ifric_minimum_age_adjustment"
    
    else:
        return {
            "error": "ifric_conditions_not_met",
            "min_usia": min_usia,
            "usia_mulai_masa_kerja": usia_mulai_masa_kerja,
            "usia_saat_valuasi": usia_saat_valuasi,
            "recommendation": "Check usia_pensiun and employment dates"
        }
    
    # Validate that IFRIC service <= standard service
    if expected_masa_kerja_lalu_ifric > calculated_masa_kerja_lalu + 0.1:
        return {
            "error": "masa_kerja_lalu_ifric_exceeds_masa_kerja_lalu",
            "expected_ifric": expected_masa_kerja_lalu_ifric,
            "calculated_standard": calculated_masa_kerja_lalu,
            "fix": "masa_kerja_lalu_ifric cannot exceed masa_kerja_lalu"
        }
    
    return {
        "expected_masa_kerja_lalu_ifric": expected_masa_kerja_lalu_ifric,
        "method_applied": method,
        "validation": "passed"
    }
```

#### **Issue 1.3: Salary Projection Inconsistencies**

**Symptoms:**

```
Error: total_gaji_pensiun < total_gaji_saat_valuasi (salary decreased?)
OR
Warning: Salary growth 150% over future_service 2 years (unrealistic)
```

**Resolution Script:**

```python
def diagnose_salary_projection(total_gaji_saat_valuasi, total_gaji_pensiun, 
                              future_service, tingkat_kenaikan_gaji):
    # Calculate implied growth rate
    if future_service > 0:
        implied_rate = (total_gaji_pensiun / total_gaji_saat_valuasi) ** (1/future_service) - 1
    else:
        implied_rate = 0
    
    # Validate growth rate reasonableness
    if implied_rate > 0.25:  # >25% annual growth
        return {
            "issue": "excessive_growth_rate",
            "implied_rate": f"{implied_rate:.1%}",
            "tingkat_kenaikan_gaji_used": f"{tingkat_kenaikan_gaji:.1%}",
            "recommendation": "Check tingkat_kenaikan_gaji parameter",
            "typical_range": "5-12% for Indonesia"
        }
    
    # Check calculation accuracy
    expected_total_gaji_pensiun = total_gaji_saat_valuasi * ((1 + tingkat_kenaikan_gaji) ** future_service)
    if abs(total_gaji_pensiun - expected_total_gaji_pensiun) > 1000:
        return {
            "issue": "salary_projection_calculation_error",
            "expected_total_gaji_pensiun": expected_total_gaji_pensiun,
            "actual_total_gaji_pensiun": total_gaji_pensiun,
            "formula": f"{total_gaji_saat_valuasi} × (1 + {tingkat_kenaikan_gaji})^{future_service}",
            "future_service_years": future_service
        }
    
    # Validate total_masa_kerja calculation
    expected_total_masa_kerja = usia_pensiun - usia_mulai_masa_kerja
    if abs(total_masa_kerja - expected_total_masa_kerja) > 0.1:
        return {
            "issue": "total_masa_kerja_inconsistent",
            "expected": expected_total_masa_kerja,
            "calculated": total_masa_kerja,
            "formula": f"{usia_pensiun} - {usia_mulai_masa_kerja}"
        }
```

---

### **🔧 Step 2: Multiple Decrement Analysis**

#### **Issue 2.1: TMI IV Mortality Rate Lookup Failures**

**Symptoms:**

```
KeyError: Age 53.08 not found in TMI_IV_Male table
OR
IndexError: list index out of range for mortality_rate_dasar lookup
```

**Safe Lookup Implementation:**

```python
def safe_tmi_iv_lookup(usia_saat_valuasi, gender):
    # Round age to integer for table lookup
    age_lookup = int(round(usia_saat_valuasi))
    
    # Boundary validation
    if age_lookup < 0:
        return {
            "error": "negative_age",
            "usia_saat_valuasi": usia_saat_valuasi,
            "solution": "Check tanggal_lahir calculation"
        }
    
    if age_lookup > 111:  # TMI IV maximum
        age_lookup = 111
        log_warning(f"Using maximum TMI IV age 111 for usia_saat_valuasi {usia_saat_valuasi}")
    
    # Gender validation
    gender_code = gender.upper() if gender in ['M', 'F'] else 'M'
    if gender not in ['M', 'F']:
        log_warning("Gender defaulted to Male for TMI_IV lookup")
    
    # Perform lookup
    try:
        if gender_code == 'M':
            mortality_rate_dasar = TMI_IV_Male[age_lookup]
        else:
            mortality_rate_dasar = TMI_IV_Female[age_lookup]
            
        return {
            "mortality_rate_dasar": mortality_rate_dasar,
            "age_used": age_lookup,
            "gender_used": gender_code,
            "status": "success"
        }
    except KeyError:
        return {
            "error": "table_lookup_failed",
            "age_lookup": age_lookup,
            "available_ages": "0-111 for TMI IV 2019"
        }
```

#### **Issue 2.2: Life Probability Progression Errors**

**Symptoms:**

```
Error: life_probability_t > life_probability_previous (increased over time)
OR
Warning: total_decrement_rate > 1.0 (impossible probability)
```

**Validation Logic:**

```python
def validate_life_probability_progression(life_prob_by_age, decrement_rates_by_age):
    errors = []
    
    ages = sorted(life_prob_by_age.keys())
    
    for i in range(1, len(ages)):
        current_age = ages[i]
        previous_age = ages[i-1]
        
        current_life_prob = life_prob_by_age[current_age]
        previous_life_prob = life_prob_by_age[previous_age]
        
        # Life probability should decrease or stay same
        if current_life_prob > previous_life_prob + 0.001:  # Small tolerance
            errors.append({
                "age": current_age,
                "issue": "life_probability_increased",
                "previous_life_probability": previous_life_prob,
                "current_life_probability": current_life_prob,
                "fix": "Check total_decrement_rate calculation"
            })
        
        # Validate total decrement rate
        if current_age in decrement_rates_by_age:
            rates = decrement_rates_by_age[current_age]
            total_decrement_rate = (rates.get("mortality_rate", 0) + 
                                  rates.get("disability_rate", 0) +
                                  rates.get("withdrawal_rate", 0) + 
                                  rates.get("pension_rate", 0))
            
            if total_decrement_rate > 1.0:
                errors.append({
                    "age": current_age,
                    "issue": "total_decrement_rate_exceeds_1",
                    "total_decrement_rate": total_decrement_rate,
                    "breakdown": rates,
                    "fix": "Check corrected rate calculations"
                })
        
        # Probability cannot be negative
        if current_life_prob < 0:
            errors.append({
                "age": current_age,
                "issue": "negative_life_probability",
                "value": current_life_prob,
                "fix": "Verify decrement rate calculations"
            })
    
    return errors
```

#### **Issue 2.3: Corrected vs Base Rate Confusion**

**Symptoms:**

```
Error: mortality_rate > mortality_rate_dasar (corrected rate higher than base)
OR
Warning: pension_rate_dasar = 1.0 but usia_saat_valuasi < usia_pensiun
```

**Correction Formula Validation:**

```python
def validate_corrected_rates(usia_saat_valuasi, usia_pensiun, life_probability, 
                           base_rates, corrected_rates):
    # Pension rate logic validation
    if usia_saat_valuasi < usia_pensiun:
        assert corrected_rates["pension_rate"] == 0, "No pension_rate before retirement"
        assert base_rates["pension_rate_dasar"] == 0, "pension_rate_dasar should be 0"
    elif usia_saat_valuasi == usia_pensiun:
        assert base_rates["pension_rate_dasar"] == 1.0, "Mandatory retirement"
        assert corrected_rates["pension_rate"] == 1.0, "Full pension_rate"
    
    # Corrected rates formula validation
    correction_factor = life_probability - base_rates.get("pension_rate_dasar", 0)
    
    for rate_type in ["mortality", "disability", "withdrawal"]:
        base_rate_key = f"{rate_type}_rate_dasar"
        corrected_rate_key = f"{rate_type}_rate"
        
        if base_rate_key in base_rates and corrected_rate_key in corrected_rates:
            expected_corrected = correction_factor * base_rates[base_rate_key]
            actual_corrected = corrected_rates[corrected_rate_key]
            
            if abs(actual_corrected - expected_corrected) > 0.001:
                return {
                    "error": f"corrected_{rate_type}_rate_formula_error",
                    "expected": expected_corrected,
                    "actual": actual_corrected,
                    "formula": f"({life_probability} - {base_rates.get('pension_rate_dasar', 0)}) × {base_rates[base_rate_key]}",
                    "correction_factor": correction_factor
                }
    
    return {"validation": "corrected_rates_passed"}
```

---

### **🔧 Step 3: Benefit Calculation**

#### **Issue 3.1: Benefit Factor Table Selection Errors**

**Symptoms:**

```
Error: UUK13 pension_factor 32.2 used for employee hired 2021 (should use UUCK)
OR
KeyError: Service years 42 not in benefit_factors_pp table (max 40)
```

**Program Selection Validation:**

```python
def validate_program_selection(tanggal_masuk_kerja, program_selected):
    # UUCK effective date
    uuck_effective = "2020-11-02"
    
    if tanggal_masuk_kerja <= uuck_effective and program_selected == "UUCK":
        return {
            "error": "incorrect_program_selection",
            "tanggal_masuk_kerja": tanggal_masuk_kerja,
            "eligible_programs": ["UUK13", "Company_Policy"],
            "fix": "Employees hired before Nov 2, 2020 use UUK13 or benefit_factors_pp"
        }
    
    if tanggal_masuk_kerja > uuck_effective and program_selected == "UUK13":
        return {
            "error": "grandfathering_issue",
            "tanggal_masuk_kerja": tanggal_masuk_kerja,
            "required_program": "UUCK or benefit_factors_pp",
            "fix": "UUK13 only for grandfathered employees"
        }
    
    return {"status": "program_selection_valid"}

def safe_factor_lookup(service_years, benefit_type, program):
    # Round service years to integer
    lookup_years = min(int(round(service_years)), 40)  # Cap at table maximum
    
    # Table boundaries
    table_bounds = {
        "benefit_factors_uuk13": (0, 40),
        "benefit_factors_uuck": (0, 40), 
        "benefit_factors_pp": (0, 40)
    }
    
    if lookup_years > table_bounds[program][1]:
        lookup_years = table_bounds[program][1]
        log_warning(f"Service years {service_years} capped at {lookup_years}")
    
    # Factor lookup keys
    factor_keys = {
        "pension": "pension_factor",
        "death": "death_factor", 
        "disability": "disability_factor",
        "withdrawal": "withdrawal_factor"
    }
    
    return BENEFIT_FACTORS[program][factor_keys[benefit_type]][lookup_years]
```

#### **Issue 3.2: Service Years Mapping Confusion**

**Symptoms:**

```
Error: Pension using masa_kerja_lalu instead of masa_kerja_lalu_ifric
OR
Warning: All benefits using same service years (should differ for pension)
```

**Service Years Validation:**

```python
def validate_service_years_mapping(benefit_calculations, step1_data):
    expected_mapping = {
        "pension": step1_data["masa_kerja_lalu_ifric"],
        "death": step1_data["masa_kerja_lalu"],
        "disability": step1_data["masa_kerja_lalu"],
        "withdrawal": step1_data["masa_kerja_lalu"]
    }
    
    errors = []
    for benefit_type, calculation in benefit_calculations.items():
        expected_years = expected_mapping[benefit_type]
        actual_years = calculation["factor_lookup"]["service_years_value"]
        
        if abs(expected_years - actual_years) > 0.1:
            errors.append({
                "benefit_type": benefit_type,
                "expected_service_years": expected_years,
                "actual_service_years": actual_years,
                "expected_variable": "masa_kerja_lalu_ifric" if benefit_type == "pension" else "masa_kerja_lalu",
                "fix": f"Use {expected_mapping[benefit_type]} for {benefit_type} benefits"
            })
    
    # Validate benefit_salary_base usage
    for benefit_type, calculation in benefit_calculations.items():
        if benefit_type == "pension":
            expected_salary_base = step1_data.get("total_gaji_pensiun", step1_data["total_gaji_saat_valuasi"])
        else:
            expected_salary_base = step1_data["total_gaji_saat_valuasi"]
        
        actual_salary_base = calculation.get("benefit_salary_base", 0)
        if abs(actual_salary_base - expected_salary_base) > 1000:
            errors.append({
                "benefit_type": benefit_type,
                "salary_base_error": "incorrect_salary_base",
                "expected": expected_salary_base,
                "actual": actual_salary_base
            })
    
    return errors
```

#### **Issue 3.3: Progressive Tax Calculation Errors**

**Symptoms:**

```
Error: total_tax calculated 125% of benefit amount
OR
Warning: Benefit in bracket 4 but tax rate applied from bracket 2
```

**Tax Bracket Validation:**

```python
def validate_progressive_tax(benefit_amount, calculated_total_tax):
    # Indonesian tax brackets for employee benefits
    tax_brackets = [
        {"min": 0, "max": 50_000_000, "rate": 0.00},
        {"min": 50_000_000, "max": 100_000_000, "rate": 0.05},
        {"min": 100_000_000, "max": 500_000_000, "rate": 0.15},
        {"min": 500_000_000, "max": float('inf'), "rate": 0.25}
    ]
    
    # Calculate expected total_tax
    expected_total_tax = 0
    remaining_amount = benefit_amount
    
    for bracket in tax_brackets:
        if remaining_amount <= 0:
            break
            
        taxable_in_bracket = min(remaining_amount, bracket["max"] - bracket["min"])
        if taxable_in_bracket > 0:
            expected_total_tax += taxable_in_bracket * bracket["rate"]
            remaining_amount -= taxable_in_bracket
    
    # Validate calculated vs expected
    if abs(calculated_total_tax - expected_total_tax) > 1000:
        return {
            "error": "progressive_tax_calculation_mismatch",
            "benefit_amount": benefit_amount,
            "expected_total_tax": expected_total_tax,
            "calculated_total_tax": calculated_total_tax,
            "variance": calculated_total_tax - expected_total_tax,
            "formula": "Sum of (taxable_amount_per_bracket × bracket_rate)"
        }
    
    # Calculate effective_tax_rate
    effective_tax_rate = calculated_total_tax / benefit_amount if benefit_amount > 0 else 0
    if effective_tax_rate > 0.25:  # Maximum bracket rate
        return {
            "error": "effective_tax_rate_exceeds_maximum",
            "effective_tax_rate": f"{effective_tax_rate:.1%}",
            "maximum_rate": "25%",
            "check": "Verify progressive tax bracket application"
        }
    
    # Validate gross_benefit calculation
    expected_gross_benefit = benefit_amount + calculated_total_tax
    return {
        "validation": "progressive_tax_passed",
        "expected_gross_benefit": expected_gross_benefit,
        "effective_tax_rate": effective_tax_rate
    }
```

---

### **🔧 Step 4: Present Value Calculations**

#### **Issue 4.1: Discount Rate Interpolation Problems**

**Symptoms:**

```
Error: Interpolated discount_rate 15.2% unreasonable for future_service 1.92 years
OR
ValueError: Cannot interpolate between IGSYC tenors 1.5 and 2.0
```

**IGSYC Interpolation Validation:**

```python
def safe_discount_rate_interpolation(future_service):
    # Boundary conditions
    if future_service < 0.5:
        return {
            "discount_rate": IGSYC_YIELD_CURVE[0.5],
            "method": "minimum_tenor_used",
            "original_future_service": future_service,
            "note": "future_service below minimum IGSYC tenor"
        }
    
    if future_service > 30.0:
        return {
            "discount_rate": IGSYC_YIELD_CURVE[30.0],
            "method": "maximum_tenor_used", 
            "original_future_service": future_service,
            "note": "future_service above maximum IGSYC tenor"
        }
    
    # Find interpolation boundaries
    available_tenors = sorted(IGSYC_YIELD_CURVE.keys())
    
    # Exact match
    if future_service in available_tenors:
        return {
            "discount_rate": IGSYC_YIELD_CURVE[future_service],
            "method": "exact_tenor_match"
        }
    
    # Linear interpolation
    lower_tenor = max([t for t in available_tenors if t <= future_service])
    upper_tenor = min([t for t in available_tenors if t >= future_service])
    
    lower_rate = IGSYC_YIELD_CURVE[lower_tenor]
    upper_rate = IGSYC_YIELD_CURVE[upper_tenor]
    
    # Interpolation calculation
    weight = (future_service - lower_tenor) / (upper_tenor - lower_tenor)
    interpolated_rate = lower_rate + weight * (upper_rate - lower_rate)
    
    # Reasonableness check
    if not (0.01 <= interpolated_rate <= 0.20):
        return {
            "error": "unreasonable_interpolated_discount_rate",
            "discount_rate": interpolated_rate,
            "future_service": future_service,
            "boundaries": f"{lower_tenor}-{upper_tenor}",
            "boundary_rates": f"{lower_rate:.4f}-{upper_rate:.4f}",
            "check": "Verify IGSYC yield curve data"
        }
    
    return {
        "discount_rate": interpolated_rate,
        "method": "linear_interpolation",
        "boundaries_used": f"{lower_tenor}-{upper_tenor}",
        "interpolation_weight": weight
    }
```

#### **Issue 4.2: PVFB vs PVDBO Relationship Errors**

**Symptoms:**

```
Error: pvdbo_pension 250,000,000 > pvfb_pension 200,000,000 (impossible)
OR
Warning: service_ratio 1.25 > 1.0 (invalid)
```

**Relationship Validation:**

```python
def validate_pvfb_pvdbo_relationship(pvfb_results, pvdbo_results, service_data, dplk_balance=0):
    errors = []
    
    for benefit_type in ["pension", "death", "disability", "withdrawal"]:
        pvfb_value = pvfb_results.get(f"pvfb_{benefit_type}", 0)
        pvdbo_value = pvdbo_results.get(f"pvdbo_{benefit_type}", 0)
        
        # Special handling for pension with DPLK offset
        if benefit_type == "pension" and pvfb_value == 0:
            # Pension PVFB might be 0 due to DPLK offset, but check gross calculation
            gross_pension_pvfb = pvfb_value + dplk_balance
            if pvdbo_value > gross_pension_pvfb * 1.01:
                errors.append({
                    "benefit_type": benefit_type,
                    "issue": "pvdbo_exceeds_gross_pvfb",
                    "pvfb_net": pvfb_value,
                    "pvfb_gross": gross_pension_pvfb,
                    "pvdbo": pvdbo_value,
                    "dplk_balance": dplk_balance
                })
        else:
            # PVDBO should not exceed PVFB
            if pvdbo_value > pvfb_value * 1.01:  # Small tolerance for rounding
                errors.append({
                    "benefit_type": benefit_type,
                    "issue": "pvdbo_exceeds_pvfb",
                    "pvfb": pvfb_value,
                    "pvdbo": pvdbo_value,
                    "fix": "Check service_ratio calculation"
                })
        
        # Service ratio validation
        if benefit_type == "pension":
            if service_data.get("total_masa_kerja_ifric", 0) > 0:
                service_ratio_ifric = service_data["masa_kerja_lalu_ifric"] / service_data["total_masa_kerja_ifric"]
            else:
                service_ratio_ifric = 0
            
            if service_ratio_ifric > 1.0:
                errors.append({
                    "benefit_type": benefit_type,
                    "issue": "service_ratio_ifric_exceeds_1",
                    "service_ratio_ifric": service_ratio_ifric,
                    "masa_kerja_lalu_ifric": service_data["masa_kerja_lalu_ifric"],
                    "total_masa_kerja_ifric": service_data.get("total_masa_kerja_ifric", 0),
                    "fix": "Check IFRIC service period calculations in Step 1"
                })
        else:
            if service_data.get("total_masa_kerja", 0) > 0:
                service_ratio = service_data["masa_kerja_lalu"] / service_data["total_masa_kerja"]
            else:
                service_ratio = 0
            
            if service_ratio > 1.0:
                errors.append({
                    "benefit_type": benefit_type,
                    "issue": "service_ratio_exceeds_1",
                    "service_ratio": service_ratio,
                    "masa_kerja_lalu": service_data["masa_kerja_lalu"],
                    "total_masa_kerja": service_data.get("total_masa_kerja", 0),
                    "fix": "Check service period calculations in Step 1"
                })
    
    return errors
```

#### **Issue 4.3: Service Cost Division by Zero**

**Symptoms:**

```
ZeroDivisionError: division by zero in csc_pension calculation
OR
Warning: csc_death 50,000,000 represents 500% of monthly salary
```

**Safe CSC Calculation:**

```python
def safe_service_cost_calculation(pvfb_value, service_denominator, benefit_type, monthly_salary=0):
    # Handle zero denominator
    if service_denominator <= 0:
        return {
            f"csc_{benefit_type}": 0,
            "warning": f"Zero service denominator for {benefit_type}",
            "service_denominator": service_denominator,
            "recommendation": "Check employment dates and service calculations"
        }
    
    # Handle very small denominators (< 1 year service)
    if service_denominator < 1.0:
        csc_value = pvfb_value  # Use full PVFB as service cost
        return {
            f"csc_{benefit_type}": csc_value,
            "method": "full_pvfb_allocation",
            "reason": f"Service period {service_denominator:.2f} < 1 year",
            "service_denominator": service_denominator
        }
    
    # Normal calculation
    csc_value = pvfb_value / service_denominator
    
    # Reasonableness check against monthly salary
    if monthly_salary > 0:
        csc_to_salary_ratio = csc_value / monthly_salary
        if csc_to_salary_ratio > 5.0:  # 500% of monthly salary
            return {
                f"csc_{benefit_type}": csc_value,
                "warning": f"Very high service cost: {csc_value:,.0f} = {csc_to_salary_ratio:.1f}x monthly salary",
                "pvfb_value": pvfb_value,
                "service_denominator": service_denominator,
                "monthly_salary": monthly_salary,
                "recommendation": "Verify PVFB and service period calculations"
            }
    
    # High absolute CSC check
    if csc_value > 100_000_000:  # 100M threshold
        return {
            f"csc_{benefit_type}": csc_value,
            "warning": f"Very high absolute service cost: {csc_value:,.0f}",
            "pvfb_value": pvfb_value,
            "service_denominator": service_denominator,
            "recommendation": "Verify PVFB and service period calculations"
        }
    
    return {f"csc_{benefit_type}": csc_value, "status": "normal"}
```

---

### **🔧 Step 5: Sensitivity Analysis & Reporting**

#### **Issue 5.1: Sensitivity Impact Out of Range**

**Symptoms:**

```
Warning: discount_rate +1% causes +45% PVDBO change (expected -8% to -15%)
OR
Error: salary_increase -1% results in negative pension benefit
```

**Sensitivity Bounds Validation:**

```python
def validate_sensitivity_ranges(sensitivity_results, base_total_pvdbo, base_csc_pension):
    expected_ranges = {
        "discount_rate_sensitivity": {
            "upward_shock": {"min": -0.18, "max": -0.05},  # -18% to -5%
            "downward_shock": {"min": 0.06, "max": 0.20}   # +6% to +20%
        },
        "salary_increase_sensitivity": {
            "higher_salary_increase": {"min": 0.02, "max": 0.15},    # +2% to +15%
            "lower_salary_increase": {"min": -0.15, "max": -0.02}    # -15% to -2%
        }
    }
    
    issues = []
    
    # Validate discount rate sensitivity
    if "discount_rate_sensitivity" in sensitivity_results:
        dr_sens = sensitivity_results["discount_rate_sensitivity"]
        
        if "upward_shock" in dr_sens:
            impact = dr_sens["upward_shock"]["percentage_impact_pvdbo"] / 100
            expected = expected_ranges["discount_rate_sensitivity"]["upward_shock"]
            
            if not (expected["min"] <= impact <= expected["max"]):
                issues.append({
                    "factor": "discount_rate_sensitivity",
                    "scenario": "upward_shock",
                    "impact": f"{impact:.1%}",
                    "expected_range": f"{expected['min']:.1%} to {expected['max']:.1%}",
                    "base_pvdbo": base_total_pvdbo,
                    "sensitivity_pvdbo": dr_sens["upward_shock"]["sensitivity_pvdbo"],
                    "diagnosis": "Check discount_rate calculation and PVFB recalculation"
                })
    
    # Validate salary increase sensitivity  
    if "salary_increase_sensitivity" in sensitivity_results:
        si_sens = sensitivity_results["salary_increase_sensitivity"]
        
        if "higher_salary_increase" in si_sens:
            impact = si_sens["higher_salary_increase"]["percentage_impact_csc"] / 100
            expected = expected_ranges["salary_increase_sensitivity"]["higher_salary_increase"]
            
            if not (expected["min"] <= impact <= expected["max"]):
                issues.append({
                    "factor": "salary_increase_sensitivity", 
                    "scenario": "higher_salary_increase",
                    "impact": f"{impact:.1%}",
                    "expected_range": f"{expected['min']:.1%} to {expected['max']:.1%}",
                    "base_csc_pension": base_csc_pension,
                    "csc_sensitivity": si_sens["higher_salary_increase"]["csc_sensitivity"],
                    "diagnosis": "Check tingkat_kenaikan_gaji recalculation impact"
                })
    
    return issues
```

#### **Issue 5.2: Maturity Profile Calculation Errors**

**Symptoms:**

```
Error: mp_discounted > mp_undiscounted (discounted exceeds undiscounted)
OR
Warning: Portfolio maturity profile sum mismatch with individual calculations
```

**Maturity Profile Validation:**

```python
def validate_maturity_profile(individual_profiles, portfolio_aggregate):
    # Individual profiles should sum to portfolio
    calculated_portfolio = {
        "mp_undiscounted_sum": sum(emp["mp_undiscounted"] for emp in individual_profiles),
        "mp_discounted_sum": sum(emp["mp_discounted"] for emp in individual_profiles)
    }
    
    errors = []
    
    # Portfolio aggregation check
    for profile_type in ["mp_undiscounted_sum", "mp_discounted_sum"]:
        expected = calculated_portfolio[profile_type]
        actual = portfolio_aggregate.get(profile_type, 0)
        
        if abs(expected - actual) > 1000:  # 1K tolerance
            errors.append({
                "issue": f"portfolio_{profile_type}_mismatch",
                "expected": expected,
                "actual": actual,
                "variance": actual - expected,
                "formula": "Sum of individual employee maturity profiles"
            })
    
    # Individual employee validations
    for emp in individual_profiles:
        # mp_discounted should be <= mp_undiscounted
        if emp["mp_discounted"] > emp["mp_undiscounted"] * 1.01:
            errors.append({
                "employee_id": emp.get("employee_id", "unknown"),
                "issue": "mp_discounted_exceeds_mp_undiscounted", 
                "mp_undiscounted": emp["mp_undiscounted"],
                "mp_discounted": emp["mp_discounted"],
                "discount_factor_pension": emp.get("discount_factor_pension", "missing"),
                "check": "Verify discount_factor calculation in Step 4"
            })
        
        # Validate discount calculation
        expected_discounted = emp["mp_undiscounted"] * emp.get("discount_factor_pension", 1.0)
        if abs(emp["mp_discounted"] - expected_discounted) > 1000:
            errors.append({
                "employee_id": emp.get("employee_id", "unknown"),
                "issue": "mp_discounted_calculation_error",
                "expected_discounted": expected_discounted,
                "actual_discounted": emp["mp_discounted"],
                "discount_factor_pension": emp.get("discount_factor_pension"),
                "formula": "mp_undiscounted × discount_factor_pension"
            })
    
    return errors
```

---

### **🔧 Step 6: Macaulay Duration**

#### **Issue 6.1: Duration Calculation Inconsistencies**

**Symptoms:**

```
Error: Macaulay_Duration 25.5 years exceeds maximum cash flow timing
OR
Warning: Duration-based discount_rate differs significantly from individual approach
```

**Duration Bounds Validation:**

```python
def validate_duration_calculation(cash_flows, calculated_Macaulay_Duration):
    # Duration cannot exceed maximum cash flow timing
    max_timing = max(cash_flows.keys()) if cash_flows else 0
    
    if calculated_Macaulay_Duration > max_timing:
        return {
            "error": "Macaulay_Duration_exceeds_max_timing",
            "calculated_Macaulay_Duration": calculated_Macaulay_Duration,
            "max_cash_flow_timing": max_timing,
            "fix": "Check PV(t.CFt) weighting calculation"
        }
    
    # Calculate simple weighted average for comparison
    total_pv_tcf = sum(pv for pv in cash_flows.values())  # Sum of PV(TCFt)
    if total_pv_tcf > 0:
        weighted_avg = sum(t * pv for t, pv in cash_flows.items()) / total_pv_tcf
    else:
        weighted_avg = 0
    
    # Macaulay_Duration should be close to weighted average
    if total_pv_tcf > 0 and abs(calculated_Macaulay_Duration - weighted_avg) > 1.0:  # 1 year tolerance
        return {
            "warning": "Macaulay_Duration_differs_from_weighted_average",
            "calculated_Macaulay_Duration": calculated_Macaulay_Duration,
            "weighted_average": weighted_avg,
            "difference": calculated_Macaulay_Duration - weighted_avg,
            "total_pv_tcf": total_pv_tcf,
            "check": "Verify Macaulay Duration formula: sum(PV(t.CFt)) / sum(PV(TCFt))"
        }
    
    # Duration should be within reasonable bounds
    if not (1.0 <= calculated_Macaulay_Duration <= 25.0):
        return {
            "warning": "Macaulay_Duration_outside_reasonable_range",
            "calculated_Macaulay_Duration": calculated_Macaulay_Duration,
            "reasonable_range": "1.0 to 25.0 years",
            "check": "Review cash flow profile and present value calculations"
        }
    
    return {"status": "Macaulay_Duration_calculation_valid"}
```

#### **Issue 6.2: Individual vs Portfolio Approach Discrepancies**

**Symptoms:**

```
Warning: Portfolio Duration_Based_Discount_Rate 6.65% vs individual average 6.89% (large gap)
OR
Error: Duration approach results in total_pvdbo variance > 10%
```

**Cross-Method Validation:**

```python
def compare_duration_vs_individual(portfolio_results, individual_results):
    # Calculate weighted average of individual discount rates
    total_pvdbo_individual = sum(emp.get("total_pvdbo", 0) for emp in individual_results)
    
    if total_pvdbo_individual > 0:
        weighted_avg_discount_rate = sum(
            emp.get("discount_rate", 0) * emp.get("total_pvdbo", 0) 
            for emp in individual_results
        ) / total_pvdbo_individual
    else:
        weighted_avg_discount_rate = 0
    
    Duration_Based_Discount_Rate = portfolio_results.get("Duration_Based_Discount_Rate", 0)
    
    # Rate comparison
    rate_difference = abs(Duration_Based_Discount_Rate - weighted_avg_discount_rate)
    if rate_difference > 0.005:  # 50 basis points
        return {
            "warning": "significant_discount_rate_difference",
            "Duration_Based_Discount_Rate": f"{Duration_Based_Discount_Rate:.3%}",
            "individual_weighted_avg_discount_rate": f"{weighted_avg_discount_rate:.3%}",
            "difference": f"{rate_difference:.1%}",
            "Macaulay_Duration": portfolio_results.get("Macaulay_Duration"),
            "recommendation": "Consider method appropriateness for portfolio"
        }
    
    # PVDBO impact comparison
    portfolio_total_pvdbo = portfolio_results.get("total_pvdbo", 0)
    pvdbo_variance = abs(portfolio_total_pvdbo - total_pvdbo_individual)
    
    if total_pvdbo_individual > 0:
        variance_percentage = pvdbo_variance / total_pvdbo_individual
    else:
        variance_percentage = 0
    
    if variance_percentage > 0.05:  # 5% variance
        return {
            "issue": "material_total_pvdbo_variance",
            "portfolio_total_pvdbo": portfolio_total_pvdbo,
            "individual_sum": total_pvdbo_individual,
            "variance_percentage": f"{variance_percentage:.1%}",
            "materiality": "material" if variance_percentage > 0.10 else "attention_needed",
            "method_comparison": "Duration vs Individual approach"
        }
    
    return {"status": "duration_individual_methods_aligned"}
```

---

## 🚀 **Quick Diagnostic Workflows**

### **End-to-End Validation Sequence**

```python
def comprehensive_validation_workflow(calculation_data):
    validation_sequence = [
        ("Step 1", validate_employee_data_foundation),
        ("Step 2", validate_multiple_decrement_analysis),
        ("Step 3", validate_benefit_calculations),
        ("Step 4", validate_present_value_calculations),
        ("Step 5", validate_sensitivity_analysis),
        ("Step 6", validate_duration_methodology)
    ]
    
    results = {"passed": [], "warnings": [], "errors": []}
    
    for step_name, validator in validation_sequence:
        try:
            step_result = validator(calculation_data)
            
            if step_result.get("status") == "passed":
                results["passed"].append(step_name)
            elif step_result.get("warnings"):
                results["warnings"].append({step_name: step_result["warnings"]})
            
        except ValidationError as e:
            results["errors"].append({step_name: str(e)})
            break  # Stop validation on critical error
    
    return results
```

### **Step-Specific Quick Checks**

```python
def quick_step_validation(step_number, step_data):
    """Quick validation checks for each step using exact variable names"""
    
    quick_checks = {
        1: lambda data: (
            data.get("usia_saat_valuasi", 0) < data.get("usia_pensiun", 100) and
            data.get("masa_kerja_lalu", 0) > 0 and
            data.get("total_gaji_saat_valuasi", 0) > 0 and
            data.get("masa_kerja_lalu_ifric", 0) <= data.get("masa_kerja_lalu", 0)
        ),
        2: lambda data: (
            0 <= sum([
                data.get("mortality_rate", 0),
                data.get("disability_rate", 0), 
                data.get("withdrawal_rate", 0),
                data.get("pension_rate", 0)
            ]) <= 1.0 and
            data.get("life_probability", 0) > 0
        ),
        3: lambda data: all(
            calc.get("benefit_amount", 0) >= 0 
            for calc in data.get("benefit_calculations", {}).values()
        ),
        4: lambda data: (
            data.get("total_pvdbo", 0) <= data.get("total_pvfb", float('inf')) and
            data.get("discount_rate", 0) > 0
        ),
        5: lambda data: (
            -0.5 <= data.get("sensitivity", {}).get("discount_rate_impact", 0) <= 0.5
        ),
        6: lambda data: (
            1.0 <= data.get("Macaulay_Duration", 0) <= 25.0
        )
    }
    
    return quick_checks.get(step_number, lambda x: True)(step_data)
```

### **Variable Consistency Cross-Check**

```python
def cross_step_variable_consistency(all_step_data):
    """Check variable consistency across steps"""
    
    consistency_checks = []
    
    # Step 1 → Step 2: Age consistency
    step1_age = all_step_data["step1"].get("usia_saat_valuasi")
    step2_age_used = all_step_data["step2"].get("age_for_mortality_lookup")
    
    if step1_age and step2_age_used:
        if abs(int(step1_age) - step2_age_used) > 0:
            consistency_checks.append({
                "issue": "age_inconsistency_step1_step2",
                "step1_usia_saat_valuasi": step1_age,
                "step2_age_used": step2_age_used,
                "fix": "Use int(usia_saat_valuasi) for TMI IV lookup"
            })
    
    # Step 3 → Step 4: Benefit amounts consistency
    step3_benefits = all_step_data["step3"].get("gross_benefits", {})
    step4_benefits = all_step_data["step4"].get("benefit_inputs", {})
    
    for benefit_type in ["pension", "death", "disability", "withdrawal"]:
        step3_key = f"{benefit_type}_benefit_gross"
        step4_key = f"{benefit_type}_benefit"
        
        if step3_key in step3_benefits and step4_key in step4_benefits:
            if abs(step3_benefits[step3_key] - step4_benefits[step4_key]) > 1000:
                consistency_checks.append({
                    "issue": f"{benefit_type}_benefit_inconsistency_step3_step4",
                    "step3_value": step3_benefits[step3_key],
                    "step4_value": step4_benefits[step4_key],
                    "fix": f"Ensure {benefit_type}_benefit_gross from Step 3 used in Step 4"
                })
    
    return consistency_checks
```

## 📞 **Escalation & Support Matrix**

### **Error Severity Classification**

|Severity|Description|Response Time|Escalation Level|Variable Types|
|---|---|---|---|---|
|**Critical**|Blocking calculation errors|Immediate|Level 3 - Actuarial Review|Core variables: usia_saat_valuasi, masa_kerja_lalu, discount_rate|
|**High**|Material result variances|2 hours|Level 2 - Technical Review|Calculation results: total_pvdbo, csc_pension, Macaulay_Duration|
|**Medium**|Business logic warnings|1 day|Level 1 - Self-Service|Factor lookups: pension_factor, mortality_rate_dasar|
|**Low**|Documentation issues|3 days|Self-Service Guide|Supporting data: benefit_salary_base, effective_tax_rate|

### **Step-Specific Support Resources**

```yaml
Step 1 Issues:
  Primary_Variables: ["usia_saat_valuasi", "masa_kerja_lalu", "masa_kerja_lalu_ifric", "total_gaji_saat_valuasi"]
  Resources: ["Date calculation guide", "IFRIC methodology", "Salary projection formulas"]
  
Step 2 Issues:
  Primary_Variables: ["mortality_rate_dasar", "life_probability", "corrected_rates"]
  Resources: ["TMI IV 2019 documentation", "Multiple decrement theory", "Life probability progression"]
  
Step 3 Issues:
  Primary_Variables: ["pension_factor", "benefit_salary_base", "total_tax", "gross_benefit"]
  Resources: ["Benefit factor tables", "Indonesian tax brackets", "Program selection rules"]
  
Step 4 Issues:
  Primary_Variables: ["discount_rate", "pvfb_pension", "pvdbo_pension", "csc_pension", "service_ratio"]
  Resources: ["IGSYC yield curve", "Present value methodology", "Service cost allocation"]
  
Step 5-6 Issues:
  Primary_Variables: ["sensitivity_discount", "mp_discounted", "Macaulay_Duration", "Duration_Based_Discount_Rate"]
  Resources: ["Sensitivity analysis bounds", "Maturity profile calculations", "Duration methodology"]
```

---

🔍 **Related Resources:**

- 📘 [Master Guide](INDEX.md) - Complete calculation methodology and steps
- 🛠️ [Reference Tables](assumptions_reference.md) - All supporting data tables