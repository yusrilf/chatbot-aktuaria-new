
# 📊 Tabel Mortalita Indonesia IV 2019

## 🎯 **Overview**

Tabel Mortalita Indonesia IV (TMI IV) 2019 adalah standar tabel mortalitas resmi yang digunakan untuk perhitungan aktuaria di Indonesia, diterbitkan oleh Asosiasi Aktuaris Indonesia.

## 📋 **Table Characteristics**

- **Coverage**: Ages 0-111 years
- **Gender Split**: Separate rates for Male and Female
- **Source Authority**: Asosiasi Aktuaris Indonesia
- **Base Year**: 2019 Indonesian population data
- **Usage**: Multiple decrement analysis, disability rate calculation

## 📊 **Complete TMI IV Mortality Table**

### **Ages 0-20: Child and Adolescent (Low Usage)**

|x|TMI_IV_Male|TMI_IV_Female|x|TMI_IV_Male|TMI_IV_Female|
|---|---|---|---|---|---|
|0|0.00524|0.00266|11|0.00019|0.00018|
|1|0.00053|0.00041|12|0.00019|0.00020|
|2|0.00042|0.00031|13|0.00020|0.00022|
|3|0.00034|0.00024|14|0.00023|0.00023|
|4|0.00029|0.00021|15|0.00027|0.00023|
|5|0.00026|0.00020|16|0.00031|0.00024|
|6|0.00023|0.00022|17|0.00037|0.00024|
|7|0.00021|0.00023|18|0.00043|0.00025|
|8|0.00020|0.00022|19|0.00047|0.00026|
|9|0.00010|0.00021|20|0.00049|0.00027|
|10|0.00019|0.00019||||

### **Ages 21-40: Young Adult Working Age (High Usage)**

|x|TMI_IV_Male|TMI_IV_Female|x|TMI_IV_Male|TMI_IV_Female|
|---|---|---|---|---|---|
|21|0.00049|0.00028|31|0.00081|0.00060|
|22|0.00049|0.00030|32|0.00087|0.00064|
|23|0.00049|0.00032|33|0.00093|0.00069|
|24|0.00050|0.00034|34|0.00099|0.00074|
|25|0.00052|0.00038|35|0.00107|0.00080|
|26|0.00055|0.00042|36|0.00116|0.00086|
|27|0.00060|0.00046|37|0.00127|0.00093|
|28|0.00065|0.00049|38|0.00139|0.00100|
|29|0.00070|0.00052|39|0.00155|0.00108|
|30|0.00075|0.00056|40|0.00175|0.00118|

### **Ages 41-60: Mid-Career Working Age (Very High Usage)**

|x|TMI_IV_Male|TMI_IV_Female|x|TMI_IV_Male|TMI_IV_Female|
|---|---|---|---|---|---|
|41|0.00193|0.00128|51|0.00556|0.00335|
|42|0.00216|0.00141|52|0.00609|0.00368|
|43|0.00241|0.00154|53|0.00667|0.00403|
|44|0.00270|0.00169|54|0.00727|0.00442|
|45|0.00302|0.00187|55|0.00789|0.00483|
|46|0.00338|0.00209|56|0.00847|0.00524|
|47|0.00377|0.00230|57|0.00898|0.00563|
|48|0.00418|0.00253|58|0.00939|0.00601|
|49|0.00461|0.00277|59|0.00971|0.00637|
|50|0.00508|0.00305|60|0.00999|0.00671|

### **Ages 61-80: Retirement Transition (Medium Usage)**

|x|TMI_IV_Male|TMI_IV_Female|x|TMI_IV_Male|TMI_IV_Female|
|---|---|---|---|---|---|
|61|0.01024|0.00707|71|0.01574|0.01314|
|62|0.01046|0.00746|72|0.01670|0.01406|
|63|0.01071|0.00788|73|0.01777|0.01508|
|64|0.01104|0.00833|74|0.01895|0.01620|
|65|0.01146|0.00883|75|0.02026|0.01743|
|66|0.01199|0.00940|76|0.02369|0.01879|
|67|0.01260|0.01005|77|0.02738|0.02030|
|68|0.01329|0.01076|78|0.03130|0.02326|
|69|0.01405|0.01150|79|0.03693|0.02880|
|70|0.01485|0.01229|80|0.04518|0.03569|

### **Ages 81-111: Advanced Age (Low Usage)**

|x|TMI_IV_Male|TMI_IV_Female|x|TMI_IV_Male|TMI_IV_Female|
|---|---|---|---|---|---|
|81|0.05527|0.04208|96|0.25715|0.19155|
|82|0.06732|0.04907|97|0.27419|0.20596|
|83|0.08228|0.05520|98|0.29249|0.22227|
|84|0.09478|0.06086|99|0.31215|0.23736|
|85|0.10465|0.06715|100|0.33331|0.25810|
|86|0.11533|0.07318|101|0.35163|0.28068|
|87|0.12698|0.08155|102|0.37132|0.30562|
|88|0.13947|0.09045|103|0.39250|0.33315|
|89|0.15271|0.10001|104|0.41527|0.36369|
|90|0.16659|0.10913|105|0.43973|0.39318|
|91|0.17991|0.11521|106|0.46602|0.42883|
|92|0.19390|0.12499|107|0.49429|0.46604|
|93|0.20874|0.13826|108|0.52467|0.50427|
|94|0.22451|0.15451|109|0.55733|0.54477|
|95|0.24126|0.17429|110|0.59244|0.58702|
||||111|1.00000|1.00000|

## 🧮 **Usage in Calculations**

### **Direct Mortality Rate Lookup**

```formula
💫 Formula: TMI IV Mortality Rate
mortality_rate_base = TMI_IV_Male(age)    // untuk karyawan pria
mortality_rate_base = TMI_IV_Female(age)  // untuk karyawan wanita
```

### **Disability Rate Calculation**

```formula
💫 Formula: Disability Rate from TMI IV
disability_rate_base = disability_multiplier × TMI_IV_rate(age, gender)
```

**Where:** disability_multiplier = 5% to 10%

### **Common Working Age Quick Reference**

|Age|Male Rate|Female Rate|Typical Employee Profile|
|---|---|---|---|
|25|0.00052|0.00038|Young professionals|
|35|0.00107|0.00080|Mid-career|
|45|0.00302|0.00187|Senior professionals|
|55|0.00789|0.00483|Pre-retirement|
|65|0.01146|0.00883|Retirement age|

## ✅ **Validation Rules**

### **Lookup Validation**

```python
def validate_tmi_iv_lookup(age, gender):
    # Age bounds check
    if not (0 <= age <= 111):
        raise ValueError(f"Age {age} outside TMI IV range [0-111]")
    
    # Gender validation
    if gender not in ["M", "F"]:
        raise ValueError(f"Gender must be 'M' or 'F', got '{gender}'")
    
    # Rate reasonableness
    rate = get_tmi_iv_rate(age, gender)
    if not (0.0001 <= rate <= 1.0):
        flag_unusual_rate(f"TMI IV rate {rate} unusual for age {age} gender {gender}")
    
    return rate
```

### **Business Context Validation**

```python
def validate_working_age_context(age, rate):
    # Working age mortality expectations
    if 25 <= age <= 65:
        if rate > 0.02:  # 2% seems high for working age
            flag_for_review(f"High mortality rate {rate} for working age {age}")
        
        if rate < 0.0001:  # Too low seems unrealistic
            flag_for_review(f"Unusually low mortality rate {rate} for age {age}")
```

## 🎯 **RAG Query Examples**

### **Simple Lookup Queries**

- "TMI IV mortality rate for 45-year-old male" → Direct: 0.00302
- "Female mortality rate age 35 TMI IV" → Direct: 0.00080
- "What is TMI IV rate for age 55?" → Requires gender specification

### **Calculation Support Queries**

- "Calculate disability rate from TMI IV for 40-year-old female" → 0.00080 × 0.10 = 0.000080
- "Compare male vs female mortality at age 50" → Male: 0.00508, Female: 0.00305
- "TMI IV rates for working age population" → Table chunk ages 21-65

### **Validation Queries**

- "Is 0.005 reasonable TMI IV rate for age 30?" → No, actual rate ~0.00075
- "TMI IV maximum age coverage" → 111 years (rate = 1.0)
- "Gender-specific vs unisex rates" → TMI IV provides gender-specific only

---

📎 **Related Resources:**

- 🔙 [Step 2: Multiple Decrement](step02_multiple_decrement.md)
- 📊 [Withdrawal Assumptions](withdrawal_assumptions.md)
- 🛠️ [Calculation Dependencies](step04_pvfb_pvdbo.md)