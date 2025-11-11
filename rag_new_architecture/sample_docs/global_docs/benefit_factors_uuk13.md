---
title: UUK13 Benefit Factor Table
description: Complete benefit multipliers under UU No. 13/2003 legacy regulation
keywords:
  - UUK13
  - benefit factors
  - severance pay
  - service pay
  - legacy regulation
table_type: benefit_factors
source: UU No. 13 Tahun 2003
regulation_status: grandfathered
usage_frequency: medium
---

# 📊 UUK13 Benefit Factor Table

## 🎯 **Regulation Overview**

**Source:** Undang-Undang Ketenagakerjaan No. 13 Tahun 2003
**Status:** Legacy regulation, grandfathered for existing employees  
**Coverage:** Permanent employees (PKWTT) with employment contracts predating UUCK

## 📋 **AI Processing Metadata**

```yaml
table_type: benefit_factors
regulation: UUK13_legacy
dimensions: [service_years, benefit_type]
service_range: [0, 40]
benefit_types: [pension, death, disability, withdrawal]
calculation_base: gross_salary
````

## 🧮 **Calculation Formula Framework**

### **Base Components**

- **Severance Pay (B)**: Basic termination payment multiplier
- **Service Pay (C)**: Long service recognition multiplier
- **Enhancement Factor**: 115% multiplier for most benefits

### **Benefit Type Formulas**

| Benefit Type   | Formula                             | Enhancement      |
| -------------- | ----------------------------------- | ---------------- |
| **Pension**    | `(2B + 1C) × 115%`                  | Yes              |
| **Death**      | `(2B + 1C) × 115%`                  | Yes              |
| **Disability** | `(2B + 2C) × 115%`                  | Yes              |
| **Withdrawal** | `(1B + 1C) × 115% + Separation_Pay` | Yes + Additional |
### 📊 **Complete UUK13 Benefit Factor Table**

### **Chunk 1: Early Career (0-10 Years Service)**

|Service Years|Severance (B)|Service (C)|Pension Factor|Death Factor|Disability Factor|Withdrawal Factor|
|---|---|---|---|---|---|---|
|0|1|0|2.30|2.30|2.30|0.15|
|1|2|0|4.60|4.60|4.60|0.30|
|2|3|0|6.90|6.90|6.90|0.45|
|3|4|2|11.50|11.50|13.80|0.90|
|4|5|2|13.80|13.80|16.10|1.05|
|5|6|2|16.10|16.10|18.40|1.20|
|6|7|3|19.55|19.55|23.00|1.50|
|7|8|3|21.85|21.85|25.30|1.65|
|8|9|3|24.15|24.15|27.60|1.80|
|9|9|4|25.30|25.30|29.90|1.95|
|10|9|4|25.30|25.30|29.90|1.95|
### **Chunk 2: Mid-Career (11-20 Years Service)**

|Service Years|Severance (B)|Service (C)|Pension Factor|Death Factor|Disability Factor|Withdrawal Factor|
|---|---|---|---|---|---|---|
|11|9|4|25.30|25.30|29.90|1.95|
|12|9|5|26.45|26.45|32.20|2.10|
|13|9|5|26.45|26.45|32.20|2.10|
|14|9|5|26.45|26.45|32.20|2.10|
|15|9|6|27.60|27.60|34.50|2.25|
|16|9|6|27.60|27.60|34.50|2.25|
|17|9|6|27.60|27.60|34.50|2.25|
|18|9|7|28.75|28.75|36.80|2.40|
|19|9|7|28.75|28.75|36.80|2.40|
|20|9|7|28.75|28.75|36.80|2.40|
### **Chunk 3: Senior Career (21-30 Years Service)**

|Service Years|Severance (B)|Service (C)|Pension Factor|Death Factor|Disability Factor|Withdrawal Factor|
|---|---|---|---|---|---|---|
|21|9|8|29.90|29.90|39.10|2.55|
|22|9|8|29.90|29.90|39.10|2.55|
|23|9|8|29.90|29.90|39.10|2.55|
|24|9|10|32.20|32.20|43.70|2.85|
|25|9|10|32.20|32.20|43.70|2.85|
|26|9|10|32.20|32.20|43.70|2.85|
|27|9|10|32.20|32.20|43.70|2.85|
|28|9|10|32.20|32.20|43.70|2.85|
|29|9|10|32.20|32.20|43.70|2.85|
|30|9|10|32.20|32.20|43.70|2.85|
### **Chunk 4: Long Service (31-40 Years Service)**

|Service Years|Severance (B)|Service (C)|Pension Factor|Death Factor|Disability Factor|Withdrawal Factor|
|---|---|---|---|---|---|---|
|31|9|10|32.20|32.20|43.70|2.85|
|32|9|10|32.20|32.20|43.70|2.85|
|33|9|10|32.20|32.20|43.70|2.85|
|34|9|10|32.20|32.20|43.70|2.85|
|35|9|10|32.20|32.20|43.70|2.85|
|36|9|10|32.20|32.20|43.70|2.85|
|37|9|10|32.20|32.20|43.70|2.85|
|38|9|10|32.20|32.20|43.70|2.85|
|39|9|10|32.20|32.20|43.70|2.85|
|40|9|10|32.20|32.20|43.70|2.85|

> *Severance Pay dan Service Pay dihitung dalam kelipatan gaji (multiple of wages).  
> **YoS = Years of Service / Masa Kerja.

## 🧮 **Factor Lookup Algorithm**

### **UUK13 Quick Factor Lookup Process:**

**Step 1: Common Factors Check**

1. **Check Common Service Years**: 5, 10, 15, 20, 25
2. **Direct Return**: If exact match found in common factors table
3. **Benefit Types**: pension, death, disability, withdrawal

**Step 2: Full Table Lookup**

1. **Input**: Service years (0-40), Benefit type
2. **Process**: Direct table lookup from complete UUK13 table
3. **Output**: Corresponding factor

### **Common Factors Quick Reference:**

| Service Years | Pension Factor | Death Factor | Disability Factor | Withdrawal Factor |
| ------------- | -------------- | ------------ | ----------------- | ----------------- |
| 5 years       | 16.10          | 16.10        | 18.40             | 1.20              |
| 10 years      | 25.30          | 25.30        | 29.90             | 1.95              |
| 15 years      | 27.60          | 27.60        | 34.50             | 2.25              |
| 20 years      | 28.75          | 28.75        | 36.80             | 2.40              |
| 25 years      | 32.20          | 32.20        | 43.70             | 2.85              |
## 📋 **Implementation Example**

json

```json
{
  "employee_profile": {
    "service_years": 12,
    "gross_monthly_salary": 15000000
  },
  "uuk13_calculation": {
    "severance_component": "9 × 15,000,000 = 135,000,000",
    "service_component": "5 × 15,000,000 = 75,000,000", 
    "death_benefit": "(2×135M + 1×75M) × 115% = 345M × 115% = 396,750,000"
  }
}
```

## ✅ **Validation Rules**

### **Maximum Factor Validation:**

- **Rule**: Pension factor caps at 32.20 for 24+ years service
- **Test**: If service_years ≥ 24 AND pension_factor ≠ 32.20
- **Action**: Raise error "UUK13 pension factor caps at 32.20 for 24+ years"

### **Minimum Factor Validation:**

- **Rule**: Minimum pension factor is 4.60 for 1+ years service
- **Test**: If service_years ≥ 1 AND pension_factor < 4.60
- **Action**: Raise error "UUK13 minimum pension factor is 4.60 for 1+ years"

### **New Employee Factor Validation:**

- **Rule**: New employee (0 years) factor must be 2.30
- **Test**: If service_years = 0 AND pension_factor ≠ 2.30
- **Action**: Raise error "UUK13 new employee factor must be 2.30"

### **Enhancement Factor Check:**

- **Rule**: All factors (except withdrawal) include 115% enhancement
- **Test**: Verify calculated factors match table values
- **Action**: Flag discrepancies for review

## 🏛️ **Regulatory Notes**

### **Grandfathered Status**

- Applies only to employees with contracts predating UU Cipta Kerja
- New employees must use UUCK benefit structure
- Companies may choose higher benefits but not lower than UUK13

### **Enhancement Factor (115%)**

- Mandatory 15% increase over base severance + service pay
- Applies to pension, death, and disability benefits
- Withdrawal benefits have different calculation (includes separation pay)

### **Service Year Brackets**

- Years 0-2: Linear increase in severance pay
- Years 3-8: Service pay introduction and increase
- Years 9-23: Plateau periods with step increases
- Years 24+: Maximum benefit levels reached

## 🛠️ **Developer Implementation Notes**

### **Required System Functions:**

**1. Factor Lookup Function:**

- **Input**: Service years (integer), Benefit type (string)
- **Process**: Common factors check → Full table lookup
- **Output**: Benefit factor (decimal)

**2. Component Calculator:**

- **Input**: Service years, Gross monthly salary
- **Process**: Calculate severance and service components
- **Output**: Severance amount, Service amount, Enhancement factor

**3. Validation Function:**

- **Input**: Service years, Benefit type, Calculated factor
- **Process**: Apply all UUK13 validation rules
- **Output**: Validation result (pass/fail) and error messages

**4. Enhancement Calculator:**

- **Input**: Base severance, Service pay, Benefit type
- **Process**: Apply 115% enhancement for applicable benefits
- **Output**: Final benefit amount

### **Testing Requirements:**

- Validate factor lookups against provided table
- Test enhancement factor calculations (115% for most benefits)
- Verify maximum and minimum factor limits
- Ensure grandfathered status logic is correctly applied

---

📎 **Related Resources:**

- 🔙 [Benefit Calculation Guide](step03_benefit_calculation.md)
- 📊 [UUCK Factors](benefit_factors_uuck.md) | [Company Policy](benefit_factors_pp.md)
- 🛠️ [Troubleshooting](troubleshooting_guide.md)