---
title: "IGSYC Zero Coupon Yield Curve"
description: "Indonesia Government Securities Yield Curve for discount rate determination"
keywords: [IGSYC, yield curve, discount rates, government bonds, zero coupon]
table_type: yield_curve
source: "PHEI (PT Penilai Harga Efek Indonesia)"
usage_context: discount_factor_calculation
update_frequency: daily
---

# 📊 IGSYC Zero Coupon Yield Curve

## 🎯 **Overview**

Indonesia Government Securities Yield Curve (IGSYC) Zero Coupon rates digunakan sebagai basis penentuan tingkat diskonto untuk valuasi aktuaria imbalan kerja sesuai standar Asosiasi Aktuaris Indonesia.

## 📋 **Yield Curve Framework**

|Method|Symbol|Description|Usage|Application Context|
|---|---|---|---|---|
|**Individual**|Future Service Matching|Direct tenor matching|⭐⭐⭐ High|Employee-specific rates|
|**Interpolation**|Linear Interpolation|Between-tenor calculation|⭐⭐⭐ High|Non-standard service periods|
|**Duration**|Macaulay Duration|Portfolio weighted rate|⭐⭐ Medium|Aggregate calculations|

**Key Integration Points:**

- 📊 **[Macaulay Duration Guide](macaulay_duration.md)** - Duration-based discount rate determination
- 📈 **[Step 4: Present Value Calculations](step04_pvfb_pvdbo.md)** - Discount factor applications

## 📈 **Yield Curve Characteristics**

### **Key Properties**

- **Risk-Free Rate**: Indonesian government bond yields
- **Zero Coupon**: Pure discount rates without coupon effects
- **Daily Updates**: Market-driven rate changes from PHEI
- **Maximum Tenor**: 30 years for actuarial calculations
- **Source Authority**: PHEI as official pricing agency

### **Usage in Actuarial Calculations**

**Kondisi Umum:** Menggunakan `future_service` dari Step 1 untuk menentukan tingkat diskonto

- **Individual Approach**: Match employee future service to yield curve tenor
- **Duration Matching**: Use Macaulay duration for weighted average rate
- **Interpolation**: Linear interpolation between available tenors
- **Maximum Cap**: 30-year rate used for longer service periods

## ⭐ **Current IGSYC Rates (as of June 30, 2025)**

### **Short-Term Rates (High Usage)**

|Tenor|Yield (%)|Usage Frequency|Typical Employee Profile|
|---|---|---|---|
|0.5|5.7165|⭐⭐ Medium|Near-retirement (6 months)|
|1.0|5.8843|⭐⭐⭐ High|Very near retirement|
|2.0|6.0313|⭐⭐⭐ High|Near retirement (2 years)|
|3.0|6.1608|⭐⭐⭐ High|Short remaining service|
|5.0|6.3913|⭐⭐⭐ High|Mid-senior employees|

### **Medium-Term Rates (Most Common)**

|Tenor|Yield (%)|Usage Frequency|Typical Employee Profile|
|---|---|---|---|
|7.0|6.5930|⭐⭐⭐ High|Mid-career employees|
|10.0|6.8363|⭐⭐⭐ High|Established career|
|12.0|6.9565|⭐⭐⭐ High|Mid-career professionals|
|15.0|7.0800|⭐⭐⭐ High|Experienced staff|

### **Long-Term Rates (Moderate Usage)**

|Tenor|Yield (%)|Usage Frequency|Typical Employee Profile|
|---|---|---|---|
|20.0|7.1758|⭐⭐ Medium|Young professionals|
|25.0|7.1973|⭐⭐ Medium|Early career|
|30.0|7.1902|⭐⭐ Medium|Very young employees|

## 📊 **Complete IGSYC Zero Coupon Table**

|Tenor|Yield (%)|Tenor|Yield (%)|
|---|---|---|---|
|0.5|5.7165|15.5|7.0949|
|1.0|5.8843|16.0|7.1084|
|1.5|5.9606|16.5|7.1206|
|2.0|6.0313|17.0|7.1315|
|2.5|6.0976|17.5|7.1413|
|3.0|6.1608|18.0|7.1500|
|3.5|6.2214|18.5|7.1578|
|4.0|6.2798|19.0|7.1646|
|4.5|6.3364|19.5|7.1706|
|5.0|6.3913|20.0|7.1758|
|5.5|6.4444|20.5|7.1803|
|6.0|6.4958|21.0|7.1841|
|6.5|6.5453|21.5|7.1874|
|7.0|6.5930|22.0|7.1901|
|7.5|6.6387|22.5|7.1923|
|8.0|6.6824|23.0|7.1940|
|8.5|6.7240|23.5|7.1953|
|9.0|6.7636|24.0|7.1963|
|9.5|6.8010|24.5|7.1970|
|10.0|6.8363|25.0|7.1973|
|10.5|6.8694|25.5|7.1974|
|11.0|6.9005|26.0|7.1972|
|11.5|6.9295|26.5|7.1969|
|12.0|6.9565|27.0|7.1963|
|12.5|6.9815|27.5|7.1956|
|13.0|7.0047|28.0|7.1948|
|13.5|7.0260|28.5|7.1938|
|14.0|7.0457|29.0|7.1927|
|14.5|7.0636|29.5|7.1915|
|15.0|7.0800|30.0|7.1902|

## 🧮 **Discount Rate Determination Methods**

**Kondisi Umum:** Menggunakan `future_service` dari Step 1 untuk semua rate determination methods

### **Method 1: Direct Future Service Matching**

```formula
💫 Formula: Direct Tenor Matching
discount_rate = IGSYC_rate[future_service_years]
```

**Usage:** When future service exactly matches available tenor

**Kondisi Aplikasi:**

- Untuk future_service = exact tenor (0.5, 1.0, 1.5, ..., 30.0): gunakan direct lookup
- Untuk future_service > 30.0: gunakan rate 30.0 years
- Untuk future_service < 0.5: gunakan rate 0.5 years

**Contoh Perhitungan:**

```json
{
  "direct_matching_example": {
    "future_service": 7.0,
    "lookup_result": {
      "tenor": "7.0 years",
      "yield_rate": 6.5930,
      "source": "IGSYC_direct_lookup"
    }
  }
}
```

### **Method 2: Linear Interpolation**

```formula
💫 Formula: Linear Interpolation
discount_rate = rate_lower + (rate_upper - rate_lower) × (tenor - tenor_lower) / (tenor_upper - tenor_lower)
```

**Usage:** When future service falls between available tenors

**Kondisi Aplikasi:**

- Untuk non-standard future_service (e.g., 7.62 years): gunakan interpolation
- Find closest lower and upper tenors
- Apply linear interpolation formula

**Contoh Perhitungan:**

```json
{
  "interpolation_example": {
    "future_service": 7.62,
    "boundaries": {
      "lower_tenor": 7.0,
      "upper_tenor": 8.0,
      "lower_rate": 6.5930,
      "upper_rate": 6.6824
    },
    "calculation": {
      "proportion": 0.62,
      "formula": "6.5930 + (6.6824 - 6.5930) × 0.62",
      "rate_difference": 0.0894,
      "interpolated_amount": 0.0555,
      "final_rate": 6.6485
    }
  }
}
```

### **Method 3: Macaulay Duration Approach**

For portfolio-level calculations using duration matching:

📊 **[Macaulay Duration Guide](macaulay_duration.md)** - Duration-based weighted average rates

```formula
💫 Formula: Duration-Weighted Rate
discount_rate = [Desimal_Macaulay × Yield(Macaulay_Atas)] + [Sisa_Desimal × Yield(Macaulay_Bawah)]
```

**Kondisi Aplikasi:**

- Untuk portfolio-level calculations: gunakan Macaulay duration approach
- Calculate weighted average based on cash flow timing
- Apply to aggregate discount rate determination

## 🤖 **Rate Lookup Algorithm**

### **IGSYC Rate Lookup Process:**

**Step 1: Boundary Validation**

```python
def validate_future_service_bounds(future_service):
    """Validate future service input bounds"""
    if future_service <= 0.5:
        return 0.5, "minimum_tenor"
    elif future_service >= 30.0:
        return 30.0, "maximum_tenor"
    else:
        return future_service, "within_bounds"
```

**Step 2: Exact Match Check**

```python
def check_exact_tenor_match(future_service):
    """Check if future service matches exact tenor"""
    available_tenors = [0.5, 1.0, 1.5, 2.0, ..., 30.0]  # All 0.5 increments
    
    if future_service in available_tenors:
        return IGSYC_TABLE[future_service], "direct_lookup"
    else:
        return None, "interpolation_needed"
```

**Step 3: Linear Interpolation**

```python
def interpolate_yield_rate(future_service):
    """Perform linear interpolation between tenors"""
    lower_tenor = floor(future_service * 2) / 2  # Round down to 0.5 increment
    upper_tenor = lower_tenor + 0.5
    
    lower_rate = IGSYC_TABLE[lower_tenor]
    upper_rate = IGSYC_TABLE[upper_tenor]
    
    proportion = (future_service - lower_tenor) / (upper_tenor - lower_tenor)
    interpolated_rate = lower_rate + (upper_rate - lower_rate) * proportion
    
    return interpolated_rate, "interpolated"
```

### **Common Rate Quick Reference:**

|Future Service|Direct Rate|Interpolated Rate|Most Common Usage|
|---|---|---|---|
|1.0 year|5.8843%|N/A|Near retirement|
|5.0 years|6.3913%|N/A|Mid-senior employees|
|7.62 years|N/A|6.6485%|Duration matching example|
|10.0 years|6.8363%|N/A|Established career|
|15.0 years|7.0800%|N/A|Experienced staff|
|20.0 years|7.1758%|N/A|Young professionals|
|30.0 years|7.1902%|N/A|Very young employees|

## 📊 **Market Context & Rate Environment**

### **Current Rate Environment Analysis**

```json
{
  "rate_environment_june_2025": {
    "curve_characteristics": {
      "shape": "slightly_inverted_long_end",
      "short_term_trend": "stable_around_6_percent",
      "long_term_trend": "stabilized_around_7_percent",
      "inversion_point": "25_year_tenor"
    },
    "actuarial_implications": {
      "discount_sensitivity": "high_for_young_employees",
      "duration_impact": "significant_for_long_service", 
      "rate_volatility": "moderate_daily_changes",
      "update_frequency": "monitor_daily_apply_monthly"
    },
    "economic_context": {
      "inflation_environment": "controlled",
      "monetary_policy": "accommodative",
      "government_fiscal": "stable_bond_issuance"
    }
  }
}
```

### **Historical Context & Trends**

**Recent Rate Evolution:**

- **2019-2021**: Declining rate environment due to pandemic response
- **2022-2024**: Rate normalization and monetary policy adjustment
- **2025**: Current stabilized environment in 6-7% range
- **Long-term outlook**: Structural stability with moderate volatility

## ✅ **Quality Assurance & Validation**

### **Rate Validation Process:**

**Step 1: Rate Reasonableness Validation**

```python
def validate_rate_reasonableness(yield_curve):
    """Validate all rates in yield curve for reasonableness"""
    for tenor, rate in yield_curve.items():
        # Rate bounds check
        if rate < 0.03 or rate > 0.12:  # 3%-12% reasonable range
            flag_unusual_rate(f"Tenor {tenor}: Rate {rate:.4f} outside normal range")
        
        # Zero or negative rate check
        if rate <= 0:
            raise ValidationError(f"Invalid rate {rate} for tenor {tenor}")
```

**Step 2: Curve Smoothness Validation**

```python
def validate_curve_smoothness(yield_curve):
    """Check for unrealistic jumps between consecutive tenors"""
    sorted_tenors = sorted(yield_curve.keys())
    
    for i in range(1, len(sorted_tenors)):
        current_tenor = sorted_tenors[i]
        previous_tenor = sorted_tenors[i-1]
        
        rate_change = abs(yield_curve[current_tenor] - yield_curve[previous_tenor])
        tenor_difference = current_tenor - previous_tenor
        
        # Rate change per year should be reasonable
        if rate_change / tenor_difference > 0.02:  # >2% per year
            flag_for_review(f"Large rate jump between {previous_tenor}Y and {current_tenor}Y")
```

**Step 3: Data Freshness Validation**

```python
def validate_data_freshness(data_timestamp):
    """Ensure yield curve data is current"""
    import datetime
    
    current_date = datetime.date.today()
    data_age = (current_date - data_timestamp).days
    
    if data_age > 7:  # More than 1 week old
        flag_stale_data(f"Yield curve data is {data_age} days old")
    elif data_age > 3:  # More than 3 days old
        flag_for_attention(f"Yield curve data is {data_age} days old")
```

### **Interpolation Quality Checks**

**Monotonic Progression Validation:**

```python
def validate_monotonic_progression(yield_curve):
    """Check medium-term rates follow expected progression"""
    medium_term_tenors = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    
    for i in range(1, len(medium_term_tenors)):
        current_rate = yield_curve[medium_term_tenors[i]]
        previous_rate = yield_curve[medium_term_tenors[i-1]]
        
        if current_rate < previous_rate:
            flag_inversion(f"Rate inversion at {medium_term_tenors[i]}Y")
```

**Long-End Inversion Handling:**

```python
def validate_long_end_inversion(yield_curve):
    """Handle normal long-end yield curve inversion"""
    rate_25y = yield_curve[25.0]
    rate_30y = yield_curve[30.0]
    
    if rate_25y > rate_30y:
        # This is normal for long-end of curve
        log_info("Normal long-end inversion detected (25Y > 30Y)")
    else:
        log_info("No long-end inversion (25Y ≤ 30Y)")
```

## 🛠️ **Implementation Guidelines**

### **System Integration Requirements:**

**Daily Rate Updates:**

- **Source**: PHEI official published rates
- **Validation**: Run all quality checks before applying new rates
- **Backup**: Maintain previous day rates for comparison and fallback
- **Audit Trail**: Log all rate changes and validation results

**Interpolation Engine:**

- **Method**: Linear interpolation between consecutive tenors
- **Precision**: Minimum 4 decimal places for rate calculations
- **Edge Cases**: Handle boundary conditions (< 0.5Y, > 30Y) appropriately
- **Performance**: Cache frequently used interpolations

**Performance Optimization:**

- **Cache Strategy**: Pre-calculate common rates (1Y, 5Y, 10Y, 15Y, 20Y)
- **Batch Processing**: Process multiple employees efficiently
- **Update Strategy**: Minimize calculation overhead during rate updates

### **Error Handling Protocols**

**Missing Tenor Data:**

```python
def handle_missing_tenor(requested_tenor):
    """Handle requests for unavailable tenors"""
    try:
        return interpolate_yield_rate(requested_tenor)
    except InterpolationError:
        nearest_tenor = find_nearest_available_tenor(requested_tenor)
        log_warning(f"Using nearest tenor {nearest_tenor} for requested {requested_tenor}")
        return IGSYC_TABLE[nearest_tenor]
```

**Extreme Rate Scenarios:**

```python
def handle_extreme_rates(calculated_rate):
    """Handle unusually high or low calculated rates"""
    if calculated_rate > 0.15:  # >15%
        log_critical(f"Extremely high rate {calculated_rate:.4f} - manual review required")
        return apply_rate_cap(calculated_rate, 0.15)
    elif calculated_rate < 0.01:  # <1%
        log_critical(f"Extremely low rate {calculated_rate:.4f} - manual review required")
        return apply_rate_floor(calculated_rate, 0.01)
    else:
        return calculated_rate
```

## 🔄 **Integration with Calculation Steps**

### **Step 4 Integration:**

Rate determination feeds directly into present value calculations:

📈 **[Step 4: Present Value Calculations](step04_pvfb_pvdbo.md)** - Discount factor applications

```json
{
  "integration_example": {
    "step1_output": {
      "future_service": 7.62
    },
    "yield_curve_processing": {
      "method": "linear_interpolation",
      "discount_rate": 6.6485
    },
    "step4_input": {
      "discount_rate": 6.6485,
      "discount_factor_retirement": 0.8929
    }
  }
}
```

### **Duration Method Integration:**

Portfolio-level approach using Macaulay duration:

📊 **[Macaulay Duration Guide](macaulay_duration.md)** - Duration-based calculations

```json
{
  "duration_integration": {
    "portfolio_duration": 7.62,
    "interpolation_weights": {
      "7_year_weight": 0.38,
      "8_year_weight": 0.62
    },
    "weighted_discount_rate": 6.6485
  }
}
```

---

📎 **Related Resources:**

- 📊 [Macaulay Duration Guide](macaulay_duration.md) - Duration-based discount rate determination
- 📈 [Step 4: Present Value Calculations](step04_pvfb_pvdbo.md) - Discount factor applications
- 🔍 [Step 5: Sensitivity Analysis](step05_sensitivity_analysis.md) - Rate sensitivity testing