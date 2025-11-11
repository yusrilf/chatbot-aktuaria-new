# PSAK 219

Standar akuntansi yang mengatur pengakuan, pengukuran, penyajian, dan pengungkapan imbalan kerja. PSAK 219 merupakan adopsi dari IAS 19 (International Accounting Standard 19) yang mengatur tentang Employee Benefits.

_Catatan: Semua perhitungan dalam dokumen ini mengikuti ketentuan PSAK 219 dan dapat ditemukan implementasinya di [Perhitungan Valuasi](INDEX.md)_

# Konsep Dasar
## Usia Pensiun 

Usia pensiun adalah **umur di mana seseorang secara resmi berhenti bekerja**, sesuai kebijakan perusahaan atau ketentuan regulasi, dan mulai memasuki masa pensiun.

## Tanggal Valuasi

**Tanggal valuasi** adalah tanggal spesifik saat dilakukan penilaian atau perhitungan nilai kini (present value) dari kewajiban dan aset dalam suatu program manfaat atau kewajiban aktuaria.

Tanggal ini menjadi acuan utama untuk mengukur nilai kewajiban dan aset pada waktu tertentu, sehingga hasil valuasi merefleksikan kondisi keuangan yang aktual pada tanggal tersebut.

## Min Usia 

**Min usia** adalah **usia minimum saat seseorang mulai berhak mendapatkan manfaat tertentu** dalam program manfaat pensiun.

**Rumus menurut standar IFRIC:**

```
Min_Usia = usia_pensiun - 24
```

# Asumsi Keuangan

## Tingkat Kenaikan Gaji 

Asumsi tingkat kenaikan gaji atau **salary increase rate** yang digunakan mengacu pada **rata-rata kenaikan gaji historis** dalam kurun waktu tertentu, dengan ketentuan sebagai berikut:

- Menggunakan **rata-rata kenaikan gaji maksimal 5 tahun ke belakang**.
- Jika data 5 tahun tidak tersedia, minimal menggunakan data **3 tahun terakhir**.

## Tingkat Diskonto 

Tingkat diskonto atau **discount rate** adalah tingkat pengembalian yang digunakan untuk menghitung nilai kini (present value) dari kewajiban imbalan pasca kerja per karyawan.

Pendekatan dimana **setiap karyawan** atau kelompok homogen karyawan memiliki tingkat diskonto yang **berbeda-beda** berdasarkan karakteristik demografis dan proyeksi arus kas individual mereka (individual approach).

**Rumus penentuan:**

```
tingkat_diskonto = f(Future_Service)
```

Discount rate ini diambil berdasarkan pada tabel Spot Rate per periode tahun berjalan.
### Spot Rate Table

| TENOR | Spot Rate 2024 | Spot Rate 31 Dec 2024 |
| ----- | -------------- | --------------------- |
| 0.50  | 6.93563%       | 5.76463%              |
| 1.00  | 6.98240%       | 5.86426%              |
| 1.50  | 6.93649%       | 5.93062%              |
| 2.00  | 6.95366%       | 6.05126%              |
| 2.50  | 6.91658%       | 6.09763%              |
| 3.00  | 6.96152%       | 6.15077%              |
| 3.50  | 6.95853%       | 6.22486%              |
| 4.00  | 6.93544%       | 6.27384%              |
| 4.50  | 7.00236%       | 6.33540%              |
| 5.00  | 7.00580%       | 6.39792%              |
| 5.50  | 7.01076%       | 6.44400%              |
| 6.00  | 7.03246%       | 6.51457%              |
| 6.50  | 7.03298%       | 6.54530%              |
| 7.00  | 7.01332%       | 6.59327%              |
| 7.50  | 7.04374%       | 6.63859%              |
| 8.00  | 7.06506%       | 6.68240%              |
| 8.50  | 7.06620%       | 6.72404%              |
| 9.00  | 7.07339%       | 6.76358%              |
| 9.50  | 7.08136%       | 6.80093%              |
| 10.00 | 7.08926%       | 6.83629%              |
| 10.50 | 7.09485%       | 6.86344%              |
| 11.00 | 7.09047%       | 6.90050%              |
| 11.50 | 7.10576%       | 6.92551%              |
| 12.00 | 7.11622%       | 6.95520%              |
| 12.50 | 7.11259%       | 6.98170%              |
| 13.00 | 7.11639%       | 7.00469%              |
| 13.50 | 7.12632%       | 7.02676%              |
| 14.00 | 7.12467%       | 7.04555%              |
| 14.50 | 7.12740%       | 7.06360%              |
| 15.00 | 7.12324%       | 7.07398%              |
| 15.50 | 7.13102%       | 7.08488%              |
| 16.00 | 7.13252%       | 7.10937%              |
| 16.50 | 7.13370%       | 7.12055%              |
| 17.00 | 7.13475%       | 7.13151%              |
| 17.50 | 7.13534%       | 7.14179%              |
| 18.00 | 7.13616%       | 7.15005%              |
| 18.50 | 7.13662%       | 7.15779%              |
| 19.00 | 7.13693%       | 7.16462%              |
| 19.50 | 7.13716%       | 7.17066%              |
| 20.00 | 7.13735%       | 7.17599%              |
| 20.50 | 7.13750%       | 7.18030%              |
| 21.00 | 7.13763%       | 7.18413%              |
| 21.50 | 7.13776%       | 7.18737%              |
| 22.00 | 7.13790%       | 7.19006%              |
| 22.50 | 7.13581%       | 7.19225%              |
| 23.00 | 7.13553%       | 7.19400%              |
| 23.50 | 7.13532%       | 7.19534%              |
| 24.00 | 7.13595%       | 7.19631%              |
| 24.50 | 7.13575%       | 7.19696%              |
| 25.00 | 7.13544%       | 7.19730%              |
| 25.50 | 7.13513%       | 7.19739%              |
| 26.00 | 7.13481%       | 7.19723%              |
| 26.50 | 7.13446%       | 7.19685%              |
| 27.00 | 7.13409%       | 7.19625%              |
| 27.50 | 7.13371%       | 7.19549%              |
| 28.00 | 7.13560%       | 7.19475%              |
| 28.50 | 7.13336%       | 7.19377%              |
| 29.00 | 7.13286%       | 7.19268%              |
| 29.50 | 7.13255%       | 7.19150%              |
| 30.00 | 7.13224%       | 7.19023%              |

# Asumsi Demografis
## Radix

**Radix** adalah nilai dasar atau angka awal yang digunakan dalam perhitungan aktuaria atau matematis sebagai acuan untuk menghitung nilai-nilai lain.

Dalam konteks aktuaria, radix bisa merujuk pada jumlah awal karyawan, populasi dasar, atau nilai referensi yang menjadi titik awal perhitungan mortalita, kecacatan, atau manfaat lainnya.

## Multiple Decrement 

### Mortality Rate

Asumsi kematian atau **Mortality Rate** adalah probabilitas seseorang meninggal dunia pada masa kerja `x` dalam periode satu tahun berikutnya. Nilai ini digunakan untuk menghitung **kemungkinan bahwa karyawan akan meninggal sebelum mencapai usia pensiun** dan untuk menentukan besarnya manfaat yang dibayarkan akibat kematian dini. Asumsi ini yang digunakan mengacu pada [Tabel Mortalita Indonesia IV 2019](tmi_iv_mortality.md).

**Notasi:** `qx_d`

**Pemilihan Tabel:**

- **Individual Calculation**: Gunakan tabel sesuai gender karyawan
- **Portfolio/Aggregate**: Gunakan weighted average berdasarkan komposisi gender workforce

### Disability Rate

Asumsi kecacatan atau **Disability Rate** adalah asumsi aktuaria yang digunakan untuk memperkirakan **kemungkinan seorang karyawan mengalami cacat tetap** (permanen) selama masa kerja aktif, sehingga tidak dapat lagi melanjutkan pekerjaannya.

**Notasi:** `qx_i`

**Rumus estimasi:**

```
qx_i = 5% hingga 10% × qx_d
```

### Withdrawal Rate

Asumsi resign atau **Withdrawal Rate** menyatakan probabilitas seorang karyawan berhenti bekerja **(bukan karena meninggal)** pada masa kerja tertentu sebelum usia pensiun yang telah ditentukan perusahaan. Ini mencerminkan risiko karyawan:

- Resign
- Diberhentikan
- Tidak melanjutkan kerja hingga pensiun

**Notasi:** `qx_w`

[Withdrawal assumptions](withdrawal_assumptions.md) bisa berbeda di setiap perusahaan karena sesuai kebijakan perusahaan terkait.

## Pension Rate

Asumsi pensiun atau **Pension Rate** adalah peluang seorang karyawan memasuki masa pensiun pada masa kerja `x` ternte. Nilai ini menunjukkan bahwa pada titik usia pensiun, karyawan akan keluar dari skema imbalan kerja karena pensiun normal.

**Notasi:** `qx_r`

**Rumus:**

```
qx_r = 0  untuk x < usia_pensiun
qx_r = 1  untuk x = usia_pensiun
```

# Program Manfaat

**Program manfaat** adalah jenis program imbalan pasca kerja yang diatur dan digunakan sesuai standar akuntansi, khususnya PSAK 219, dan regulasi terkait seperti:

### 1. Undang-Undang Ketenagakerjaan No. 13 Tahun 2003 [UUK13](benefit_factors_uuk13.md)

### 2. Undang-Undang Cipta Kerja [UUCK](benefit_factors_uuck.md)

### 3. Peraturan Perusahaan (PP) yang berlaku di masing-masing entitas perusahaan [PP](benefit_factors_pp.md)

# Pajak 

Pajak adalah kewajiban atas manfaat yang diterima karyawan, termasuk imbalan kerja, seperti pensiun, atau pesangon. Dalam perhitungan aktuaria, pajak **tidak dimasukkan langsung dalam nilai kewajiban** karena perhitungan dilakukan atas nilai bruto manfaat.

Pajak atas imbalan kerja bisa ditanggung oleh karyawan melalui pemotongan pembayaran manfaat, atau oleh **perusahaan** sesuai kebijakan. Meskipun tidak memengaruhi nilai kewajiban imbalan kerja (DBO), pajak tetap dicatat secara terpisah dalam laporan keuangan, misalnya melalui pengakuan **pajak tangguhan.**

### Tarif Pajak Progresif

Pajak penghasilan (**PPh 21**) atas manfaat pensiun dihitung secara progresif berdasarkan total manfaat yang diterima sebelum pajak. Perhitungan ini mengikuti lapisan tarif pajak yang ditetapkan pemerintah, dengan persentase yang meningkat seiring besarnya manfaat.

Berikut adalah lapisan tarif yang berlaku:

- **0 – 50 juta rupiah**: dikenakan tarif **0%**
- **50 juta – 100 juta rupiah**: dikenakan tarif **5%**
- **100 juta – 500 juta rupiah**: dikenakan tarif **15%**
- **> 500 juta rupiah**: dikenakan tarif **25%**

---