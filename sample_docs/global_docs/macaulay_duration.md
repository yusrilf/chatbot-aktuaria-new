
# 📊 Macaulay Duration Guide

## 🎯 **Overview**

Macaulay Duration adalah rata-rata tertimbang waktu pembayaran arus kas masa depan, yang digunakan untuk menentukan tingkat diskonto yang sesuai dengan profil kewajiban imbalan kerja dalam PSAK 219.

## 📋 **Duration Methodology Framework**

|Approach|Symbol|Description|Usage|Application Context|
|---|---|---|---|---|
|**Individual**|Employee-Specific|Future service matching|⭐⭐⭐ High|Single employee calculations|
|**Portfolio**|Macaulay Duration|Weighted average approach|⭐⭐ Medium|Company-wide liability|
|**Aggregate**|Duration Matching|Asset-liability matching|⭐⭐ Medium|Investment strategy|

**Key Reference Tables:**

- 📊 **[IGSYC Yield Curve](yield_curve_igsyc.md)** - Government bond zero-coupon rates
- 📊 **[Present Value Calculations](step04_pvfb_pvdbo.md)** - Integration with individual approach

## 💡 **Duration Concept & Purpose**

### **Definisi Macaulay Duration**

Macaulay Duration adalah rata-rata tertimbang waktu pembayaran arus kas masa depan, yang digunakan untuk menentukan tingkat diskonto yang sesuai dengan profil kewajiban imbalan kerja.

### **Tujuan Penggunaan**

- Menentukan **tingkat diskonto** yang tepat untuk valuasi kewajiban imbalan kerja
- **Mencocokkan durasi** kewajiban dengan yield curve pemerintah
- Memastikan **akurasi pengukuran** present value kewajiban
- **Portfolio-level** discount rate determination

## 🧮 **Duration Calculation Methodology**

### **Dua Pendekatan Sub-Metode:**

**Metode 1: Historical Weighted Average**

- Menggunakan informasi rata-rata tertimbang dari **laporan periode sebelumnya**
- Cocok untuk entitas dengan **data historis lengkap**
- **Consistency** dengan periode sebelumnya

**Metode 2: Current Period Calculation**

- Menggunakan informasi dari **hitungan periode sekarang** dengan tingkat diskonto tentative
- Cocok untuk **new valuations** atau **significant changes**
- **Real-time** reflection of current liability profile

### **Macaulay Duration Formula**

```formula
💫 Formula: Macaulay Duration
Macaulay_Duration = sum(PV(t_CFt)) / sum(PV(TCFt))

Where t > 0 (exclude period_0):
CFt = sum(cash_flow_i[period_t]) for all employees i where period_t > 0
```

**Komponen:**

- **t**: Waktu pembayaran (tahun)
- **PV(t_CFt)**: Present Value Cash Flow pada tahun t (time-weighted)
- **CFt**: Cash Flow pada tahun t all employees
- **PV(TCFt)**: Present Value Total Cash Flow
- **IGSYC_t**: Tingkat diskonto dari yield curve pada tahun ke t

### **Present Value Calculation Steps**

#### **Step 1: Calculate Present Value of Each Cash Flow**

```formula
💫 Formula: Present Value Cash Flow
PV(TCFt) = CFt / (1 + IGSYC_t)^t
```

#### **Step 2: Calculate Time-Weighted Present Value**

```formula
💫 Formula: Time-Weighted Present Value
PV(t_CFt) = t × (CFt / [1 + IGSYC_t]^t)
```

#### **Step 3: Calculate Macaulay Duration**

```formula
💫 Formula: Final Duration Calculation
macaulay_duration = sum(PV[t_CFt]) / sum(PV[TCFt])
```

## 📊 **Practical Calculation Example**

### **Portfolio Cash Flow Table**

| t   | Cash Flow (CFt) | IGSYC Rate | PV(TCFt)      | PV(t_CFt)      |
| --- | --------------- | ---------- | ------------- | -------------- |
| 1   | 4,973,680,262   | 0.0588     | 4,697,280,266 | 4,697,280,266  |
| 2   | 1,394,232,515   | 0.0603     | 1,240,130,517 | 2,480,261,035  |
| 3   | 2,234,573,209   | 0.0616     | 1,867,679,741 | 5,603,039,223  |
| 4   | 1,343,992,550   | 0.0628     | 1,053,399,991 | 4,213,599,962  |
| 5   | 919,411,614     | 0.0639     | 674,496,552   | 3,372,482,762  |
| 6   | 1,182,058,709   | 0.0650     | 810,298,932   | 4,861,793,589  |
| 7   | 1,685,006,157   | 0.0659     | 1,079,627,864 | 7,557,395,045  |
| 8   | 334,430,942     | 0.0666     | 199,359,990   | 1,594,847,920  |
| 9   | 2,066,963,593   | 0.0676     | 1,146,897,969 | 10,322,081,724 |
| 10  | 2,265,002,575   | 0.0684     | 1,169,179,754 | 11,691,797,538 |
| ... | ...             | ...        | ...           | ...            |
| 23  | 4,274,871,578   | 0.0719     | 864,972,098   | 19,894,358,245 |
| 24  | 65,250,651      | 0.0720     | 12,310,304    | 295,447,307    |
| 25  | 52,112,845      | 0.0720     | 9,169,559     | 229,238,969    |

### **Duration Calculation Results**

```json
{
  "duration_calculation": {
    "total_pv_weighted": 148209050506,
    "total_pv_cashflows": 19445926887,
    "macaulay_duration": 7.62,
    "duration_years": "7 years 7 months",
    "calculation": "148,209,050,506 ÷ 19,445,926,887 = 7.62 years"
  }
}
```

## 🎯 **Discount Rate Determination**

### **Linear Interpolation Method**

Setelah mendapat Macaulay Duration = 7.62 tahun, langkah selanjutnya adalah menentukan tingkat diskonto tertimbang menggunakan interpolasi linear pada yield curve.

```formula
💫 Formula: Duration-Based Discount Rate
discount_rate = (desimal_macaulay × yield[macaulay_atas]) + (sisa_desimal × yield[macaulay_bawah])
```

**Pembulatan:** 2 desimal sesuai standar Asosiasi Aktuaris Indonesia

### **Interpolation Steps**

#### **Step 1: Determine Boundaries**

```formula
💫 Formula: Duration Boundaries
macaulay_bawah = floor(macaulay_duration)
macaulay_atas = floor(macaulay_duration) + 1
```

**Contoh:**

```
Macaulay Duration = 7.62 tahun
macaulay_bawah = 7 tahun
macaulay_atas = 8 tahun
```

#### **Step 2: Calculate Decimal Components**

```formula
💫 Formula: Decimal Components
desimal_macaulay = macaulay_duration - macaulay_bawah
sisa_desimal = 1 - desimal_macaulay
```

**Contoh:**

```
desimal_macaulay = 7.62 - 7 = 0.62
sisa_desimal = 1 - 0.62 = 0.38
```

#### **Step 3: Apply Interpolation**

```json
{
  "interpolation_example": {
    "macaulay_duration": 7.62,
    "yield_inputs": {
      "yield_7_years": 0.065930,
      "yield_8_years": 0.066824
    },
    "weights": {
      "weight_8_years": 0.62,
      "weight_7_years": 0.38
    },
    "calculation": {
      "formula": "(0.62 × 0.066824) + (0.38 × 0.065930)",
      "component_1": 0.041431,
      "component_2": 0.025053,
      "total": 0.066484,
      "final_rate": "6.65%"
    }
  }
}
```

_Final Rate ini sebagai Tingkat Diskonto Akhir Tahun yang digunakan pada [Tabel 1 Laporan Aktuaria](02f1_valuasi_tabel_utama.md#tabel-1-data-asumsi)_

## 📊 **Integration with IGSYC Yield Curve**

### **Data Source Standards**

Tingkat diskonto mengikuti **Juknis Asosiasi Aktuaris Indonesia** dengan data dari:

- **Source**: PHEI (PT Penilai Harga Efek Indonesia)
- **Type**: Yield obligasi zero coupon
- **Basis**: Risk-free rate sesuai standar aktuaria
- **Maximum Tenor**: 30 tahun untuk perhitungan aktuaria

📊 **[Complete IGSYC Yield Curve](yield_curve_igsyc.md)** - Current market rates

### **Aggregate vs Individual Approach**

**Duration Approach (Aggregate):**

- Menggunakan **satu tingkat diskonto tunggal** untuk seluruh portofolio
- Berdasarkan **durasi agregat** dari total arus kas kewajiban
- Cocok untuk **portfolio-level** reporting dan **asset-liability matching**

**Individual Approach:**

- Setiap karyawan memiliki tingkat diskonto berbeda
- Berdasarkan **future service** individual
- Cocok untuk **detailed employee-level** calculations

## ⚖️ **Materiality & Rounding Considerations**

### **Rounding Policy Guidelines**

|Liability Size|Rounding Precision|Usage Context|
|---|---|---|
|**Large obligations**|25 basis points|Major company liabilities|
|**Medium sensitivity**|10 basis points|Standard calculations|
|**High sensitivity**|5 basis points|Precision-critical calculations|

### **Materiality Considerations**

**Key Factors:**

1. **Besaran kewajiban** yang dihitung
2. **Tingkat sensitivitas** hasil perhitungan
3. **Konsistensi** penerapan di periode mendatang
4. **Kesepakatan dengan auditor** untuk audit trail

## ✅ **Validation & Quality Assurance**

### **Duration Reasonableness Checks**

```python
def validate_duration_calculation(macaulay_duration, total_pv_weighted, total_pv_cashflows, pv_components):
    """Validate duration calculation using actual markdown variables"""
    
    # Duration should be within reasonable bounds
    if not (0.5 <= macaulay_duration <= 25.0):
        flag_for_review(f"Duration {macaulay_duration:.2f} outside normal range")
    
    # Validate core calculation consistency
    calculated_duration = total_pv_weighted / total_pv_cashflows
    duration_variance = abs(macaulay_duration - calculated_duration) / calculated_duration
    
    if duration_variance > 0.001:  # 0.1% tolerance
        flag_for_review(f"Duration calculation inconsistent: {macaulay_duration:.6f} vs {calculated_duration:.6f}")
    
    # Validate component summation
    sum_pv_tcft = sum(pv_components["PV_TCFt"])
    sum_pv_t_cft = sum(pv_components["PV_t_CFt"])
    
    if abs(sum_pv_tcft - total_pv_cashflows) > 1:
        flag_for_review("PV(TCFt) summation inconsistent with total_pv_cashflows")
    
    if abs(sum_pv_t_cft - total_pv_weighted) > 1:
        flag_for_review("PV(t.CFt) summation inconsistent with total_pv_weighted")

def flag_for_review(message):
    """Helper function to log validation warnings"""
    print(f"VALIDATION WARNING: {message}")
```

### **Interpolation Validation**

```python
def validate_interpolation(macaulay_duration, discount_rate, igsyc_yield_curve):
    """Validate interpolation using markdown-defined variables"""
    
    # Calculate interpolation components per markdown formula
    macaulay_bawah = int(macaulay_duration)  # floor function
    macaulay_atas = macaulay_bawah + 1
    
    desimal_macaulay = macaulay_duration - macaulay_bawah
    sisa_desimal = 1 - desimal_macaulay
    
    # Get yield curve rates
    yield_bawah = igsyc_yield_curve[macaulay_bawah]
    yield_atas = igsyc_yield_curve[macaulay_atas]
    
    # Validate interpolation formula: (desimal_macaulay × yield_atas) + (sisa_desimal × yield_bawah)
    expected_rate = (desimal_macaulay * yield_atas) + (sisa_desimal * yield_bawah)
    interpolation_variance = abs(discount_rate - expected_rate)
    
    if interpolation_variance > 0.0001:  # 1 basis point tolerance
        flag_for_review(f"Interpolation calculation error: {discount_rate:.6f} vs {expected_rate:.6f}")
    
    # Final rate should be between yield curve bounds
    if not (min(yield_bawah, yield_atas) <= discount_rate <= max(yield_bawah, yield_atas)):
        raise ValidationError("Interpolated rate outside yield curve bounds")
    
    # Rate should be within reasonable IGSYC range
    if discount_rate < 0.04 or discount_rate > 0.10:
        flag_for_review(f"Unusual interpolated rate: {discount_rate:.4f}")
    
    # Validate decimal components sum to 1
    if abs((desimal_macaulay + sisa_desimal) - 1.0) > 0.0001:
        raise ValidationError(f"Decimal components don't sum to 1: {desimal_macaulay} + {sisa_desimal}")
```

### **Comprehend Portfolio Checks**

```python
def validate_portfolio_duration(cash_flow_periods, igsyc_rates):
    """Validate portfolio-level duration calculation"""
    
    total_pv_weighted = 0
    total_pv_cashflows = 0
    
    for t, CFt in cash_flow_periods.items():
        if t > 0:  # Exclude period_0 per markdown constraint
            igsyc_t = igsyc_rates[t]
            
            # Calculate PV(TCFt) = CFt / (1 + IGSYC_t)^t
            pv_tcft = CFt / ((1 + igsyc_t) ** t)
            
            # Calculate PV(t.CFt) = t × [CFt / (1 + IGSYC_t)^t]
            pv_t_cft = t * pv_tcft
            
            total_pv_cashflows += pv_tcft
            total_pv_weighted += pv_t_cft
    
    # Validate against expected totals
    portfolio_duration = total_pv_weighted / total_pv_cashflows if total_pv_cashflows > 0 else 0
    
    return {
        "portfolio_duration": portfolio_duration,
        "total_pv_weighted": total_pv_weighted,
        "total_pv_cashflows": total_pv_cashflows,
        "validation_passed": total_pv_cashflows > 0 and 0.5 <= portfolio_duration <= 25.0
    }
```


## 🔄 **Implementation Guidelines**

### **Method Selection Criteria**

**Choose Duration Approach When:**

- Portfolio-level discount rate needed
- Asset-liability matching strategy
- Aggregate reporting requirements
- Multiple employee calculations

**Choose Individual Approach When:**

- Employee-specific calculations required
- Detailed actuarial analysis needed
- Individual benefit projections
- Sensitivity analysis by employee profile

### **Consistency Requirements**

**Period-to-Period:**

- Maintain same methodology unless justified change
- Document any method modifications
- Ensure audit trail completeness

**Cross-Validation:**

- Compare results with individual approach
- Verify duration calculation accuracy
- Validate interpolation results

## 🎯 **PSAK 219 Implementation Notes**

### **Regulatory Compliance**

**Key Requirements:**

- **Matching duration** liabilitas dengan yield curve
- **Present value accuracy** melalui metode yang tepat
- **Consistency check** antar metode untuk validasi
- **Documentation** yang jelas dan terstruktur

### **Audit Considerations**

**Required Documentation:**

- **Justifikasi pemilihan** metode yang dapat diaudit
- **Sensitivitas analysis** untuk validasi hasil
- **Methodology consistency** across periods
- **Quality control procedures** implemented

### **Flexibility Guidelines**

**Implementation Approach:**

- Entitas dapat memilih metode sesuai **kompleksitas** dan **ketersediaan data**
- **Konsistensi** penerapan lebih penting daripada kompleksitas metode
- **Pemilihan metode harus disesuaikan dengan kapabilitas entitas**
- Tetap mempertahankan **akurasi dan konsistensi** dalam perhitungan

---

📎 **Related Resources:**

- 📊 [IGSYC Yield Curve](yield_curve_igsyc.md) - Current market rates
- 📈 [Step 4: Present Value Calculations](step04_pvfb_pvdbo.md) - Individual approach integration
- 🔍 [Sensitivity Analysis](step05_sensitivity_analysis.md) - Duration impact testing