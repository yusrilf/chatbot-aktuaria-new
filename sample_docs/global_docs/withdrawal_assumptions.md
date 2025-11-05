
# 📊 Withdrawal Assumptions Table

## 🎯 **Overview**

Tabel asumsi withdrawal (qx_w) yang digunakan dalam multiple decrement analysis untuk menentukan probabilitas karyawan berhenti kerja secara sukarela atau tidak sukarela sebelum usia pensiun.

## 📋 **Table Characteristics**

- **Coverage**: Ages 15-60 years (working age span)
- **Rate Type**: Annual probability of voluntary/involuntary termination
- **Source Authority**: Company Policy and Industry Experience
- **Pattern Type**: Age-based lifecycle withdrawal behavior
- **Usage**: Multiple decrement analysis, HR planning

## 📊 **Complete Withdrawal Assumptions Table**

### **Ages 15-19: Entry Level Protection (Low Usage)**

x = age

| x   | Withdrawal Rate (qx_w) | x   | Withdrawal Rate (qx_w) |
| --- | ---------------------- | --- | ---------------------- |
| 15  | 0.00%                  | 18  | 0.00%                  |
| 16  | 0.00%                  | 19  | 0.00%                  |
| 17  | 0.00%                  |     |                        |

### **Ages 20-29: High Mobility Period (Very High Usage)**

|x|Withdrawal Rate (qx_w)|x|Withdrawal Rate (qx_w)|
|---|---|---|---|
|20|6.00%|25|6.00%|
|21|6.00%|26|6.00%|
|22|6.00%|27|6.00%|
|23|6.00%|28|6.00%|
|24|6.00%|29|6.00%|

### **Ages 30-39: Career Establishment (High Usage)**

|x|Withdrawal Rate (qx_w)|x|Withdrawal Rate (qx_w)|
|---|---|---|---|
|30|3.00%|35|1.80%|
|31|3.00%|36|1.80%|
|32|3.00%|37|1.80%|
|33|3.00%|38|1.80%|
|34|3.00%|39|1.80%|

### **Ages 40-53: Career Stability (High Usage)**

|x|Withdrawal Rate (qx_w)|x|Withdrawal Rate (qx_w)|
|---|---|---|---|
|40|1.20%|47|1.20%|
|41|1.20%|48|1.20%|
|42|1.20%|49|1.20%|
|43|1.20%|50|1.20%|
|44|1.20%|51|0.60%|
|45|1.20%|52|0.60%|
|46|1.20%|53|0.60%|

### **Ages 54-60: Pre-Retirement (Medium Usage)**

|x|Withdrawal Rate (qx_w)|x|Withdrawal Rate (qx_w)|
|---|---|---|---|
|54|0.00%|58|0.00%|
|55|0.00%|59|0.00%|
|56|0.00%|60|0.00%|
|57|0.00%|||

## 🧮 **Usage in Calculations**

### **Direct Withdrawal Rate Lookup**

```formula
💫 Formula: Withdrawal Rate Lookup
withdrawal_rate_dasar = withdrawal_table[x]
```

**Kondisi Aplikasi:**

- Untuk 0 < x ≤ 14 dan x = usia_pensiun: withdrawal_rate_dasar = 0
- Untuk 15 ≤ x ≤ usia_pensiun - 1: ambil dari tabel withdrawal assumption

### **Multiple Decrement Integration**

```formula
💫 Formula: Withdrawal Rate Terkoreksi
withdrawal_rate = (life_probability - pension_rate_dasar) × withdrawal_rate_dasar
```

### **Common Working Age Quick Reference**

| x   | Withdrawal Rate | Career Stage    | Typical Employee Profile   |
| --- | --------------- | --------------- | -------------------------- |
| 20  | 6.00%           | High Mobility   | New graduates, exploration |
| 25  | 6.00%           | High Mobility   | Job market testing         |
| 30  | 3.00%           | Stabilization   | Career establishment       |
| 35  | 1.80%           | Mid-Career      | Management transition      |
| 45  | 1.20%           | Retention       | Senior professionals       |
| 53  | 0.60%           | Pre-Retirement  | Pension planning           |
| 55  | 0.00%           | Retirement Prep | Benefit maximization       |

## 🏭 **Industry-Specific Variations**

### **Technology Sector**

- **Ages 20-29**: 8-12% (higher mobility for better opportunities)
- **Ages 30-39**: 4-6% (stock option vesting impact)
- **Ages 40+**: 0.5-2% (senior positions, equity retention)

### **Manufacturing Sector**

- **Ages 20-29**: 4-5% (stable employment culture)
- **Ages 30-49**: 1-2% (union protections, pension benefits)
- **Ages 50+**: 0.1% (strong retirement benefits)

### **Financial Services**

- **Ages 25-35**: 3-5% (regulatory examinations, bonus cycles)
- **Ages 35-50**: 1-3% (compensation structures, long-term incentives)
- **Ages 50+**: 0.5% (regulatory/compliance expertise value)

### **Healthcare Sector**

- **Ages 25-40**: 2-4% (license portability, demand stability)
- **Ages 40+**: 1% (established patient relationships, partnership tracks)

## ✅ **Validation Rules**

### **Lookup Validation**

```python
def validate_withdrawal_lookup(age):
    # Age bounds check
    if not (15 <= age <= 60):
        if age < 15:
            return 0.0  # Below working age
        elif age > 60:
            return 0.0  # Post-retirement
        else:
            raise ValueError(f"Age {age} outside withdrawal table range")
    
    # Get rate from table
    rate = get_withdrawal_rate(age)
    
    # Rate reasonableness
    if rate > 0.15:  # 15% seems unreasonably high
        flag_for_review(f"High withdrawal rate {rate} for age {age}")
    
    if rate < 0:
        raise ValueError(f"Withdrawal rate cannot be negative")
    
    return rate
```

### **Business Context Validation**

```python
def validate_withdrawal_context(age, rate):
    # Age-specific expectations
    if 20 <= age <= 29 and rate != 0.06:
        flag_inconsistency(f"Expected 6% withdrawal for age {age}, got {rate}")
    
    if age >= 54 and rate > 0:
        raise ValueError(f"No withdrawal expected at pre-retirement age {age}")
    
    # Pattern validation
    if 30 <= age <= 34 and rate != 0.03:
        flag_inconsistency(f"Expected 3% stabilization rate for age {age}")
    
    if 40 <= age <= 50 and rate != 0.012:
        flag_inconsistency(f"Expected 1.2% retention rate for age {age}")
```

### **Pattern Consistency Validation**

```python
def validate_withdrawal_patterns():
    # Check decreasing trend with age (generally)
    age_groups = [
        (20, 29, 0.06),  # High mobility
        (30, 34, 0.03),  # Stabilization  
        (35, 39, 0.018), # Mid-career
        (40, 50, 0.012), # Retention
        (51, 53, 0.006)  # Pre-retirement
    ]
    
    for i in range(1, len(age_groups)):
        current_rate = age_groups[i][2]
        previous_rate = age_groups[i-1][2]
        
        if current_rate >= previous_rate:
            flag_inconsistency("Withdrawal rates should generally decrease with age")
```

## 🎯 **RAG Query Examples**

### **Simple Lookup Queries**

- "Withdrawal rate for age 25" → Direct: 6.00%
- "What is resignation probability for 35-year-old?" → Direct: 1.80%
- "Turnover rate age 45" → Direct: 1.20%

### **Pattern Analysis Queries**

- "Why high withdrawal rates in twenties?" → Career exploration peak, job mobility
- "Withdrawal pattern by age group" → 6% → 3% → 1.8% → 1.2% → 0.6% → 0%
- "Pre-retirement withdrawal behavior" → Zero withdrawal (pension maximization)

### **Business Context Queries**

- "Technology sector withdrawal adjustments" → Higher mobility rates (8-12% vs 6%)
- "Manufacturing vs standard rates" → Lower due to union protection
- "Why zero withdrawal age 54+" → Retirement preparation, benefit optimization

### **Validation Queries**

- "Is 8% reasonable for age 25?" → High but possible for tech sector
- "Withdrawal rate maximum age coverage" → 60 years
- "Why zero rates for ages 15-19?" → Training period, skill development

---

📎 **Related Resources:**

- 🔙 [Step 2: Multiple Decrement](step02_multiple_decrement.md)
- 📊 [TMI IV Mortality Table](tmi_iv_mortality.md)
- 🛠️ [Industry Variations](assumptions_reference.md) | [Validation Guidelines](troubleshooting_guide.md)