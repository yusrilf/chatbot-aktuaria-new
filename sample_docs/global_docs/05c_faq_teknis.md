---
keywords:
  - aktuaria
  - perhitungan
  - metodologi
  - PVDBO
  - CSC
  - PSC
  - OCI
  - diskonto
  - asumsi
  - PUC
  - projected unit credit
  - multiple decrement
  - curtailment
difficulty:
  - basic
  - intermediate
  - advanced
  - expert
estimated_reading: 90 minutes
target_audience:
  - actuaries
  - finance_professionals
  - consultants
  - technical_specialists
  - quantitative_analysts
document_type: faq_specialized
last_updated: 2025-07-28
version: "2025.1"
domain: imbalan_kerja_psak219
scope: technical_actuarial_calculation
exclude_domains:
  - asuransi_umum
  - perbankan_komersial
  - insurance_analytics
context_boundary: strict
semantic_focus:
  - aktuaria
  - perhitungan
  - metodologi
  - PVDBO
  - CSC
  - PSC
  - OCI
  - diskonto
  - asumsi
  - PUC
  - projected unit credit
  - multiple decrement
  - curtailment
  - sains aktuaria
  - matematis aktuaria
  - PSAK 219
  - imbalan kerja
semantic_exclude:
  - outstanding claims reserve
  - insurance underwriting
  - claim settlement
  - premium calculation
  - general insurance
  - life insurance products
  - bancassurance
  - RBNS
  - IBNR
  - insurance marketing
fallback_response: Informasi tidak tersedia dalam konteks sains aktuaria dan imbalan kerja. Silakan ajukan pertanyaan terkait PSAK 219, valuasi aktuaria, atau imbalan kerja karyawan.
---

---
# BAB 5C: Frequently Asked Questions (FAQ) - TEKNIS & PERHITUNGAN

**📖 Bagian dari:** [PANDUAN LENGKAP IMBALAN KERJA PSAK 219 - BAB 5](05_faq.md)

---

## 🎯 **Tentang FAQ Ini**

FAQ Teknis & Perhitungan ini menyajikan panduan mendalam mengenai metodologi, asumsi aktuaria, dan implementasi teknis perhitungan imbalan kerja sesuai PSAK 219. FAQ ini dirancang untuk praktisi aktuaria, konsultan, dan profesional keuangan yang memerlukan pemahaman teknis yang komprehensif.

## 🚀 **Quick Search Guide**

**Tips Pencarian Cepat:**

- Gunakan `Ctrl+F` (Windows) atau `Cmd+F` (Mac) untuk mencari kata kunci
- Kata kunci umum: "CSC", "PVDBO", "OCI", "diskonto", "asumsi", "metodologi"
- Nomor pertanyaan: "#T001", "#T102", "#T203" untuk referensi cepat
- Difficulty levels: 🟢 Basic, 🟡 Intermediate, 🟠 Advanced, 🔴 Expert

## 🔍 **Quick Glossary for RAG**

- **PVDBO**: Present Value of Defined Benefit Obligation
- **CSC**: Current Service Cost (Biaya Jasa Kini)  
- **BJS**: Past Service Cost (Biaya Jasa Lalu)
- **OCI**: Other Comprehensive Income
- **PUC**: Projected Unit Credit Method
- **IFRIC**: IFRS Interpretations Committee

## 🔍 **Quick Lookup Table - Technical & Calculations**
| Query Category | FAQ Range | Key Concepts | Complexity | Audience |
|:---------------|:----------|:-------------|:-----------|:---------|
| **Data Setup** | T001-T050 | data requirements, employee status, salary basis | 🟢 | All users |
| **Actuarial Methodology** | T051-T100 | PUC method, assumptions, discount rate, salary growth | 🟡 | Practitioners |
| **Accounting & Reporting** | T101-T150 | CSC, PSC, OCI, PVDBO, financial statements | 🟠 | Finance professionals |
| **Complex Scenarios** | T151-T200 | curtailment, DPLK, OLTEB, edge cases | 🟠 | Specialists |
| **Expert Level** | T201-T250 | advanced methodology, multiple rates, interpolation | 🔴 | Actuaries |

| Technical Concepts | Core FAQs | Formula/Method | Dependencies |
|:-------------------|:----------|:---------------|:-------------|
| **PUC Method** | T051, T103 | CSC = PVFB / Service Years | Employee data, assumptions |
| **Discount Rate** | T056, T058 | Multiple vs single rate | Yield curve, duration |
| **PVDBO Calculation** | T109, T164 | Present value of obligations | All actuarial assumptions |
| **Service Costs** | T103, T165 | Current vs Past service costs | Employee movements, plan changes |

| Calculation Components | Related FAQs | Input Requirements | Output Impact |
|:----------------------|:-------------|:------------------|:--------------|
| **Current Service Cost** | T067, T103, T112 | Employee data, salary, service | P&L expense |
| **Past Service Cost** | T064, T103, T165 | Plan changes, amendments | P&L expense |
| **Interest Cost** | T102, T112 | PVDBO beginning, discount rate | P&L expense |
| **OCI Components** | T104, T105, T106 | Actuarial gains/losses | OCI reporting |

| Common Technical Queries                         | Target FAQ | Required Expertise | Typical Answer Pattern                       |
| :----------------------------------------------- | :--------- | :----------------- | :------------------------------------------- |
| "Apa itu metode PUC?"                            | T051       | Basic actuarial    | "Projected Unit Credit menggunakan..."       |
| "Bagaimana hitung CSC?"                          | T103       | Intermediate       | "CSC = PVFB / Total masa kerja..."           |
| "Mengapa PVDBO naik padahal karyawan berkurang?" | T164       | Advanced           | "PVDBO tergantung usia, masa kerja, gaji..." |
| "Perbedaan multiple vs single discount rate?"    | T056, T202 | Expert             | "Multiple rate per employee duration..."     |

| Data Quality Checks    | Validation FAQs  | Red Flags                       | Resolution                 |
| :--------------------- | :--------------- | :------------------------------ | :------------------------- |
| **Employee data**      | T001, T011, T016 | Missing dates, incorrect status | Data completion            |
| **Salary information** | T006, T015, T204 | Inconsistent components         | Clarify total compensation |
| **Benefit payments**   | T010, T161, T163 | Unreported severances           | Include all movements      |
| **Assumptions**        | T054, T062, T063 | Unrealistic salary growth       | Management justification   |

---

## Daftar Isi

- [Data dan Setup Perhitungan](#data-dan-setup-perhitungan) (T001-T050) 🟢
- [Metodologi dan Asumsi Aktuaria](#metodologi-dan-asumsi-aktuaria) (T051-T100) 🟡
- [Accounting dan Pelaporan](#accounting-dan-pelaporan) (T101-T150) 🟠
- [Skenario Kompleks dan Edge Cases](#skenario-kompleks-dan-edge-cases) (T151-T200) 🟠
- [Advanced Topics dan Expert Level](#advanced-topics-dan-expert-level) (T201-T250) 🔴

**Note:**

- 'kami' pada pertanyaan merujuk pada point of view user atau klien (penanya)
- 'kami' pada jawaban merujuk pada point of view KKA Nirmala (penjawab)
- Difficulty indicators: 🟢 Basic, 🟡 Intermediate, 🟠 Advanced, 🔴 Expert

---

## #data-dan-setup-perhitungan

#### T001. Data apa saja yang dibutuhkan untuk perhitungan? 🟢

Kami memiliki 2 file template data untuk dilengkapi. Pertama, berupa file template excel yang berisikan data pribadi karyawan seperti **tanggal lahir, tanggal masuk kerja, gaji, dan lainnya.** Kedua, berupa file word yang berisikan informasi perusahaan, seperti **nama perusahaan, alamat perusahaan, rata-rata kenaikan gaji, penanggung pajak oleh karyawan/perusahaan dan lainnya.**

Selain 2 template tersebut, kami membutuhkan data pendukung lainnya seperti **peraturan perusahaan, NPWP perusahaan, lembar persetujuan penawaran yang telah ditandatangani,** laporan aktuaria tahun lalu (jika ada) serta saldo & **iuran** DPLK (jika ada).

Hal-hal yang perlu dipastikan meliputi:

- Pastikan jumlah karyawan yang dihitung mencakup karyawan aktif dan tidak aktif, dilengkapi status tetap dan kontrak (jika ada)
- Pastikan untuk tidak memasukkan anggota Board of Directors (BOD) dan Tenaga Kerja Asing (TKA) (jika ada) dalam perhitungan

#### T002. Bagaimana cara memastikan saldo imbalan pasca kerja pada laporan keuangan sebelumnya? 🟢

Pastikan untuk memeriksa apakah pada laporan keuangan sebelumnya sudah ada saldo **beban** imbalan pasca kerja yang dicatat. Jika ini adalah pertama kali perusahaan menghitung, tetapi sebelumnya sudah ada perhitungan internal, dibutuhkan data detail perhitungannya, setidaknya saldo yang digunakan pada laporan keuangan sebelumnya.

#### T003. Bagaimana pencatatan saldo imbalan pasca kerja di neraca? 🟢

- Umumnya dicatat sebagai **Utang** jika imbalan kerja masih berupa kewajiban **yang belum dibayarkan**, seperti Biaya Jasa Kini dan Biaya Jasa Lalu
- Dalam kondisi tertentu dicatat sebagai **Aset** jika perusahaan **memiliki dana pensiun** atau **aset investasi yang dikelola secara khusus** untuk membayar manfaat tersebut di masa depan

Dengan bantuan Kantor Akuntan Publik (KAP), pencatatan ini dilakukan sesuai standar akuntansi seperti PSAK 219 untuk memastikan transparansi laporan keuangan.

#### T004. Periode perhitungan IPK dilakukan per tahun, per bulan, atau per kuartal? 🟢

Perhitungan IPK **biasanya dilakukan per tahun**, mengikuti periode pelaporan audit, karena merupakan salah satu komponen yang diperlukan auditor dalam laporan keuangan. Namun, bisa juga dihitung per kuartal **tergantung kebutuhan perusahaan dan auditor** untuk mendapatkan nilai estimasi imbalan kerja yang **lebih proporsional secara berkala.**

#### T005. Bagaimana dengan status karyawan harian? 🟢

Karyawan harian tidak masuk ke dalam perhitungan kecuali ada ketentuan khusus yang mengatur manfaat imbalan kerja untuk karyawan harian.

#### T006. Bagaimana ketentuan gaji yang digunakan dalam perhitungan IPK? 🟢

Gaji yang digunakan untuk menghitung Imbalan Pasca Kerja (IPK) terdiri dari gaji pokok dan tunjangan tetap, yang keduanya dihitung dalam jumlah bruto (sebelum dipotong pajak atau iuran lain) untuk masing-masing karyawan.

#### T007. Bagaimana jika karyawan kontrak berubah setiap proyek? 🟢

Perhitungan dilakukan berdasarkan masing-masing karyawan kontrak per proyek, dengan masa kerja dihitung sejak tanggal awal kontrak hingga tanggal proyek selesai.

#### T008. Apa yang harus dilakukan jika klien sudah pernah melakukan perhitungan tetapi tidak memiliki data rincian terkait perhitungan kewajiban / PVDBO? 🟢

Jika klien tidak memiliki data rincian PVDBO, maka **dibutuhkan data karyawan yang digunakan pada perhitungan sebelumnya** untuk melihat asumsi yang relevan yang akan digunakan pada perhitungan saat ini.

#### T009. Bagaimana ketentuan mengenai realisasi pembayaran karyawan kontrak? 🟢

Untuk karyawan kontrak, perlu dipastikan apakah realisasi pembayaran kompensasi dilakukan **setiap akhir kontrak**—baik bagi karyawan yang kontraknya diperpanjang maupun yang tidak—atau **hanya saat karyawan keluar** dari perusahaan.

#### T010. Untuk periode perhitungan, bagaimana tahunnya? 🟢

Periode perhitungan diisi **sesuai dengan periode audit perusahaan**. Jika audit perusahaan dilakukan per Desember, maka perhitungan aktuaria juga dilakukan per Desember.

**See also:** T004 (Periode Perhitungan), A031 (Waktu Perhitungan)

#### T011. Apakah dasar gaji yang digunakan untuk menghitung imbalan kerja mencakup gaji pokok, tunjangan tetap, dan PPh? 🟢

Dasar gaji yang digunakan dalam perhitungan imbalan kerja bergantung pada kebijakan perusahaan dan regulasi ketenagakerjaan yang berlaku. Secara umum, dalam perhitungan aktuaria sesuai PSAK 219, yang digunakan sebagai dasar adalah **Gaji Pokok + Tunjangan Tetap.**

- Gaji Pokok → Komponen utama yang diperhitungkan dalam manfaat pasca kerja.
- Tunjangan Tetap → Tunjangan yang diberikan secara rutin, nilainya tetap dan tidak bergantung pada kehadiran atau kinerja, seperti tunjangan jabatan, tunjangan keluarga, atau tunjangan fungsional.
- _PPh 21 (Pajak Penghasilan Pasal 21)_ → tidak dimasukkan dalam data perhitungan imbalan kerja. **Jika PPh 21 ditanggung oleh Perusahaan, maka perhitungan manfaat akan di gross up sesuai dengan ketentuan PPh 21.**

#### T012. Apakah data karyawan yang mengundurkan diri tetap harus dicatat, meskipun mereka tidak dihitung dalam jumlah total karyawan untuk perhitungan? 🟢

Ya, data mereka tetap perlu dicatat dalam file template excel. Hal ini penting untuk **mendokumentasikan movement (perubahan status) karyawan**, seperti pengunduran diri, sehingga aktuaris dapat melacak perubahan populasi karyawan dari waktu ke waktu. Data ini juga relevan untuk perhitungan terkait biaya jasa lalu atau _curtailment_, apabila ada kompensasi yang diberikan pada saat pengunduran diri.

#### T013. Apakah karyawan ekspatriat dimasukkan ke dalam perhitungan? 🟢

Tidak dimasukkan ke dalam perhitungan, kecuali ada SK tersendiri / tertuang dalam kontrak kerja manfaat / kompensasi yang akan diterima jika kontrak kerja berakhir.

**See also:** L41 (TKA dalam Perhitungan)

#### T014. Apakah masih diperlukan penghitungan imbalan kerja untuk perusahaan yang sudah lama tidak aktif & tak ada karyawan? 🟢

Untuk perusahaan yang sudah tidak aktif / tidak ada karyawannya, maka tidak perlu ada perhitungan imbalan pasca kerja karena imbalan tersebut muncul sebagai nilai biaya yang harus dicadangkan perusahaan ketika nanti karyawannya pensiun / resign.

#### T015. Untuk karyawan outsource, apakah dimasukkan ke dalam perhitungan? 🟢

Tidak, biasanya karyawan outsource sudah dihitung oleh perusahaannya (vendor outsourcing).

#### T016. Kami ada driver (supir) dengan jenis kontrak kemitraan, apakah ini juga dimasukkan ke dalam perhitungan? 🟢

Tidak, biasanya perusahaan mitra sudah melakukan perhitungan imbalan pasca kerja sendiri.


---

## #metodologi-dan-asumsi-aktuaria
#### T051. Metode apa yang digunakan dalam perhitungan? 🟡

Perhitungan IPK menggunakan metode Projected Unit Credit (PUC) sesuai dengan yang dipersyaratkan dalam PSAK 219. Metode ini juga sesuai dengan IFRS/IAS 19 (Employee Benefits), sebagai standar perhitungan aktuaria internasional.

#### T052. Bagaimana cara menghitung nilai estimated future working dalam PSAK 219? 🟡

Estimated future working dapat dihitung berdasarkan sisa masa kerja untuk setiap karyawan. Untuk karyawan tetap dihitung hingga karyawan tersebut mencapai usia pensiunnya, sedangkan untuk karyawan kontrak dihitung hingga berakhirnya kontrak karyawan tersebut (diasumsikan satu tahun).

#### T053. Apa perbedaan perhitungan untuk karyawan tetap dan kontrak? 🟡

- **Karyawan tetap (PKWTT)**: Perhitungannya mencakup beberapa komponen, seperti cadangan untuk pensiun, meninggal dunia, cacat, dan resign, yang formulanya mengikuti peraturan perundang-undangan yang berlaku
- **Karyawan kontrak (PKWT)**: Perhitungannya didasarkan pada masa kerja (proporsional), yaitu masa kerja (bulan) / 12 x gaji

**Catatan**: Setelah IFRIC diterapkan, nilai manfaat PKWTT berusia < 34 tahun umumnya lebih tinggi dibandingkan PKWT. Ini karena perhitungan PKWT hanya mencakup manfaat kematian, cacat, dan pengunduran diri, **tanpa manfaat pensiun**.

#### T054. Bagaimana cara menentukan tingkat kenaikan gaji dalam perhitungan IPK? 🟡

Tingkat kenaikan gaji umumnya ditentukan berdasarkan:

- Rata-rata kenaikan gaji historis perusahaan dalam beberapa tahun terakhir
- Kenaikan UMK di wilayah operasional perusahaan
- Tingkat inflasi dalam beberapa tahun terakhir
- Rencana kenaikan gaji ke depan, berdasarkan kebijakan internal atau surat resmi dari manajemen perusahaan (management letter)
- Asumsi tahun sebelumnya, bila masih relevan dan belum berubah signifikan

#### T055. Apakah terdapat perbedaan perhitungan yang digunakan jika dibandingkan dengan KKA lain? 🟡

Setiap KKA, dalam melakukan perhitungan IPK dihitung menggunakan metode yang sama, sesuai dengan yang dipersyaratkan dalam PSAK 219, namun umumnya terdapat perbedaan dalam pendekatan menghitung usia yang akan digunakan dalam menghitung kewajiban.

#### T056. Apa tingkat diskonto yang menggunakan single rate atau multiple rate? 🟡

Untuk **karyawan tetap**, penetapan diskonto menggunakan **multiple rate**, dimana besarannya akan berbeda untuk setiap karyawan yang ditentukan berdasarkan rata-rata sisa masa kerja yang diprakirakan untuk setiap karyawan, sedangkan untuk penyajiannya dalam laporan menggunakan tingkat diskonto setelah perhitungan **durasi rata-rata tertimbang menggunakan Macaulay Duration.**

Sedangkan untuk **karyawan kontrak**, menggunakan rata-rata diskonto yang diperoleh dari interpolasi sisa masa kerja setiap karyawan.

#### T057. Berapa suku bunga diskonto yang digunakan dalam perhitungan aktuaria untuk tahun valuasi? 🟡

Suku bunga diskonto bervariasi tergantung pada kebijakan perusahaan dan regulasi yang berlaku, karena tidak ada standar tunggal yang berlaku di tiap tahun valuasi untuk semua entitas.

#### T058. Mengapa ada perhitungan aktuaria untuk laporan di tahun berjalan, baru bisa dilakukan di awal tahun selanjutnya? 🟡

Perhitungan aktuaria mengandalkan tingkat diskonto yang biasanya didasarkan pada yield obligasi pemerintah per tanggal valuasi (misalnya 31 Desember 2024). Namun, data resmi tingkat diskonto ini baru tersedia di awal tahun berikutnya (Januari 2025). Untuk kebutuhan internal, perusahaan dapat menggunakan estimasi tingkat diskonto sementara (misalnya per 30 November), tetapi hasil akhir tetap harus diperbarui setelah data resmi tersedia.

Kami mengacu kepada spot rate yang dikeluarkan oleh pemerintah melalui PHEI (PT Penilai Harga Efek Indonesia) setiap bulannya. Hal ini dilakukan agar kewajiban imbalan kerja mencerminkan nilai wajar sesuai PSAK 219.

#### T059. Apa risiko jika perhitungan aktuaria dilakukan menggunakan estimasi tingkat diskonto sebelum data resmi dirilis? 🟡

Risiko utamanya adalah perbedaan antara hasil estimasi dan hasil akhir setelah tingkat diskonto resmi dirilis. Jika selisihnya signifikan, laporan keuangan bisa dianggap tidak mencerminkan kewajiban sebenarnya, berpotensi mendapat catatan audit dan menurunkan kepercayaan pemangku kepentingan. Akibatnya, perusahaan harus merevisi laporan dan menghadapi keterlambatan pelaporan ke regulator dan pemegang saham.

#### T060. Apa dasar penentuan rata-rata kenaikan gaji yang digunakan dalam perhitungan? 🟡

Kami mengikuti informasi yang diberikan oleh klien sesuai dengan pengalaman perusahaan. Namun, jika klien tidak mengisi, maka kami akan mengikuti tingkat inflasi yang berlaku saat ini.

**See also:** T054 (Tingkat Kenaikan Gaji), A33 (Perubahan Kenaikan Gaji)

#### T061. Apakah indikator harus tetap menggunakan tingkat kenaikan gaji? 🟡

Bagaimanapun kondisi entitas perusahaan, perhitungan aktuaria tetap harus ada tingkat kenaikan gaji dengan minimal di angka 5%.

#### T062. Apakah ada perbedaan antara tingkat kenaikan gaji aktual dengan asumsi tingkat kenaikan gaji? 🟡

Kami menggunakan asumsi estimasi sesuai yang diinformasikan dari perusahaan, tingkat kenaikan gaji yang dimaksud merupakan asumsi untuk tahun-tahun ke depannya, bukan yang lalu. Adapun jika terdapat perbedaan antara aktual dan yang diasumsikan, dampaknya akan diakui pada perubahan asumsi keuangan.

**See also:** T054 (Tingkat Kenaikan Gaji)

#### T063. Mengapa ada perbedaan antara salary increase rate yang digunakan dalam laporan dan aktual kenaikan gaji? 🟡

Perbedaan tersebut bisa terjadi karena dalam laporan merupakan asumsi jangka panjang, sedangkan kenaikan gaji aktual dapat dipengaruhi oleh faktor-faktor eksternal dan kebijakan perusahaan dalam jangka pendek. Dalam perhitungan, rate yang digunakan untuk memproyeksikan kewajiban di masa depan dengan mempertimbangkan:

1. **Tren historis** – Pola kenaikan gaji rata-rata dalam beberapa tahun terakhir
2. **Inflasi & kondisi ekonomi** – Perkiraan pertumbuhan ekonomi yang mempengaruhi upah
3. **Historis karyawan** – kenaikan gaji berdasarkan tingkat jabatan, performa, dan masa kerja
4. **Kebijakan perusahaan** – Rencana kompensasi jangka panjang

Oleh karena itu, jika rate mengalami kenaikan/penurunan aktual dari asumsi dalam laporan aktuaria, maka dapat terjadi deviasi yang akan dikoreksi dalam penyesuaian aktuaria berikutnya.

#### T064. Mengapa nilai biaya jasa lalu bisa muncul dari perbedaan usia pensiun dan mutasi karyawan? 🟡

Perbedaan usia pensiun dan mutasi karyawan **dapat berdampak langsung** pada perhitungan kewajiban aktuaria karena mengubah periode kerja yang diperhitungkan dan jumlah karyawan yang berhak atas manfaat.

Jika **usia pensiun diperpanjang**, maka **nilai kini kewajiban (PVDBO) cenderung menurun**, karena manfaat dibayarkan lebih lambat dan dicicil selama masa kerja yang lebih panjang. Namun, **manfaat total yang akan diterima di akhir masa kerja bisa meningkat** akibat masa kerja yang lebih panjang. Perubahan ini diakui sebagai biaya jasa lalu.

Jika terjadi **mutasi atau pengurangan karyawan**, maka cadangan aktuaria untuk karyawan tersebut **dikeluarkan (take-out) dari kewajiban.** Apabila perusahaan kemudian membayar manfaat, maka selisih antara cadangan yang dilepas dan pembayaran aktual akan **menimbulkan biaya jasa lalu.**

#### T065. Bagaimana menangani data gaji karyawan dalam perhitungan imbalan kerja, terutama dengan adanya kenaikan gaji di bulan tertentu sesuai PP No. 5 Tahun 2024? 🟡

Dalam perhitungan aktuaria, **yang digunakan adalah gaji posisi valuasi** (misalnya, Desember 2024) karena mencerminkan kondisi laporan keuangan. Jika ada kenaikan gaji di bulan tertentu tahun 2025, maka sesuai PP No. 5 Tahun 2024, data tersebut **tidak langsung digunakan dalam perhitungan kewajiban, tetapi tetap relevan untuk memvalidasi asumsi kenaikan gaji di masa depan.** Meskipun gaji Desember menjadi dasar utama, data kenaikan gaji Februari bisa diberikan sebagai tambahan untuk analisis lebih lanjut, seperti mengevaluasi dampaknya terhadap kewajiban atau menyesuaikan asumsi aktuaria.

#### T066. Apa yang dimaksud dengan Perubahan Program Manfaat? 🟡

Perubahan dalam ketentuan program imbalan kerja yang menyebabkan penyesuaian atas kewajiban, seperti perubahan formula manfaat atau syarat kelayakan.

#### T067. Jika ada perubahan tanggal masuk kerja karyawan, apakah mempengaruhi perhitungan pada bagian CSC? 🟡

Perubahan data yang mempengaruhi nilai dari CSC / Biaya Jasa Kini di antaranya adalah:

- Tanggal Lahir
- Tanggal Masuk kerja
- Gaji Karyawan

Perubahan pada tanggal masuk kerja mempengaruhi lama masa kerja karyawan, sehingga berpengaruh kepada perhitungan liabilitas karyawan tersebut.

**See also:** T103 (Biaya Jasa Kini)

#### T068. Apabila pada peraturan perusahaan terdapat penjelasan secara spesifik berupa benefit yang akan didapat karyawan selama sakit dan sakit berkepanjangan seperti upah, apakah masuk ke dalam perhitungan PSAK 219? 🟡

Jika manfaat tersebut diberikan untuk karyawan yang masih bekerja, maka tidak masuk ke dalam perhitungan. Hal tersebut dikarenakan manfaat yang dihitung dalam PSAK 219 adalah benefit untuk pasca kerja, yaitu ketika karyawan tersebut sudah tidak bekerja atau benefit ketika terjadi pemutusan hubungan kerja.

#### T069. Kalau dalam ilmu aktuaria, uang duka seperti apa yang bisa masuk dan yang tidak masuk ke perhitungan aktuaria? 🟡

**Yang Masuk Perhitungan:** Uang duka / santunan duka yang diberikan kepada karyawan karena merupakan bagian dari manfaat meninggal untuk karyawan (termasuk imbalan pasca kerja).

**Yang Tidak Masuk Perhitungan:** Uang duka untuk keluarga karyawan. Misalkan, bantuan kematian untuk ayah ibu dan anak, itu tidak perlu diperhitungkan.

#### T070. Jika kompensasi untuk PKWT yang sudah habis kontraknya tidak dibayarkan oleh perusahaan, maka uang kompensasinya akan menggulung atau diakui sebagai pendapatan lain-lain? 🟡

Ya, uang kompensasinya akan menggulung dari awal kontrak kerja. Sedangkan jika dibayarkan kompensasinya maka tanggal masuk kerja karyawan tersebut menggunakan tanggal kontrak terbarunya.

**See also:** L047 (Kompensasi PKWT)

#### T071. Jika jumlah karyawan bertambah tetapi liabilitas dari perhitungan aktuaria lebih kecil dibandingkan tahun sebelumnya, bagaimana cara memastikan Uang Penghargaan Hak (UPH) sudah dihitung dengan benar? 🟡

Dalam perhitungan manfaat UUK, jika jumlah karyawan bertambah tetapi liabilitas lebih kecil dibanding tahun sebelumnya, **perlu dipastikan bahwa Uang Penghargaan Hak (UPH) sudah dimasukkan dalam formula 15% x (Pesangon + Penghargaan Masa Kerja).** Seringkali, laporan aktuaria sebelumnya hanya menyebutkan total perhitungan tanpa secara eksplisit menuliskan UPH secara terpisah. Oleh karena itu, penting untuk memeriksa kembali perumusan manfaat guna memastikan UPH sudah diperhitungkan dengan benar dalam kewajiban aktuaria.

#### T072. Apakah jenis industri perusahaan memengaruhi perhitungan imbalan pasca kerja?🟡

Ya, Jenis industri dapat berpengaruh pada asumsi yang akan digunakan pada perhitungan imbalan pasca kerja, seperti disability rate dan withdrawal rate. 

**See also:** T054 (Tingkat Kenaikan Gaji), T055 (Perbedaan KKA), T062 (Asumsi vs Aktual), L44 (Aktuaria Perusahaan Sekuritas)

#### T073. Kalau karyawan resign pada bulan Desember 2024, apakah masuk ke perhitungan? Karena mereka masih mendapatkan gaji prorate	🟡

Kalau periode perhitungan per 31 Desember 2024, maka karyawan yang tidak aktif pada Desember tidak dihitung maupun dimasukkan ke laporan aktuaria.
Akan tetap dihitung jika tanggal henti kerja karyawan tersebut setelah tanggal valuasi, yaitu tahun 2025.

**See also:** T111 (Karyawan Keluar Tahun Valuasi), T163 (Realisasi PHK), T016 (Data Karyawan Resign), A62 (Data Karyawan Aktif)

#### T074. Kenapa angka perhitungan kami besar sedangkan karyawan kami semuanya PKWT (kontrak)? bukannya karyawan kontrak hanya 1x gaji untuk kompensasinya🟡

Untuk kompensasi, besarannya disesuaikan dengan masa kerja. Jika masa kerja 1 tahun, maka diberikan sebesar 1 kali gaji. Jika masa kerja 2 tahun, maka diberikan sebesar 2 kali gaji, begitupun seterusnya.

**See also:** T053 (Perhitungan Tetap vs Kontrak), T223 (PKWT Angka Besar), T205 (IFRIC Age Requirements), T226 (Kontrak Lebih Besar Awal)

#### T075. Kalau seandainya ini karyawan permanen, angka liabilitasnya lebih besar mana jika dibandingkan dengan angka dari karyawan kontrak? 🟡

Hasilnya tentatif, karena untuk karyawan tetap perhitungan manfaat diproyeksikan hingga usia pensiun. Sementara untuk karyawan kontrak dengan jatuh tempo singkat, cadangan akan dibentuk penuh sekaligus dan membuat nilainya lebih besar ketika awal-awal perhitungan. Jadi nilai manfaat karyawan tetap akan lebih besar dibandingkan karyawan kontrak apabila peserta mencapai usia pensiun. 

**See also:** T225 (Liability Tetap vs Kontrak), T053 (Perhitungan Tetap vs Kontrak), T070 (Kompensasi PKWT), T052 (Estimated Future Working)



---

## #accounting-dan-pelaporan

#### T101. Bagaimana memahami pengakuan beban/pendapatan yang memisahkan Biaya Jasa Kini dan Biaya Jasa Lalu dalam pencatatan imbalan kerja? 🟠

Pengakuan beban/pendapatan dalam pencatatan imbalan kerja, penting dipisahkan antara:

1. **Debit** imbalan kerja sebagai total beban yang timbul selama tahun berjalan dan harus diakui di laporan laba rugi:
    - **Biaya Jasa Kini (Current Service Cost)** → hak karyawan yang bekerja aktif selama tahun valuasi
    - **Biaya Jasa Lalu (Past Service Cost)** → koreksi atas manfaat masa lalu, yang sifatnya mengurangi kewajiban perusahaan secara signifikan
2. **Kredit** imbalan kerja sebagai jumlah kewajiban (utang) yang harus dibayar oleh perusahaan kepada karyawan di masa mendatang

#### T102. Apa saja komponen utama penyusun beban pada laporan IPK? 🟠

Terdiri dari Biaya Jasa Kini, Biaya Jasa Lalu dan Biaya Bunga. Bisa ditambahkan juga komponen tambahan lainnya seperti: pengaruh Kurtailmen atau Penyelesaian, jika ada penghentian program pensiun, dan (Dikurangi) Hasil Investasi Aset Program, bila perusahaan punya aset dana pensiun.

#### T103. Apa yang dimaksud dengan Biaya Jasa Lalu dan Biaya Jasa Kini? Bagaimana cara perhitungannya menurut PSAK 219 untuk karyawan tetap? 🟠

1. **Biaya Jasa Lalu (Past Service Cost)** adalah biaya yang muncul ketika perusahaan baru pertama kali mencatat kewajiban imbalan pasca kerja/IPK secara menyeluruh atau terjadi perubahan besar, seperti perubahan kebijakan cuti besar, restruktur kompensasi, mutasi karyawan, perubahan pada metode/asumsi, atau revisi program pensiun. Berikut adalah ketentuan mengenai Biaya Jasa Lalu untuk penerapan pertama kali:
    - Jika masa kerja < 1 tahun: Biaya Jasa Lalu = 0
    - Jika masa kerja ≥ 1 tahun: Biaya Jasa Lalu = **PVDBO** – **CSC**
2. **Biaya Jasa Kini (Current Service Cost)** adalah biaya atas manfaat pensiun yang "ditabung" perusahaan untuk karyawan selama periode berjalan.

Rumus sederhana Biaya Jasa Kini/**CSC**: 
CSC= PVFB / MK
dimana: 
- PVFB = present value of Future Benefit atau nilai sekarang dari estimasi manfaat masa depan
- MK = total masa kerja hingga karyawan mencapai usia pensiun normal

#### T104. Mengapa nilai OCI hanya mencakup karyawan tetap, dan bagaimana perlakuan keuntungan/kerugian aktuarial dalam laporan keuangan? 🟠

Nilai **OCI** hanya mencakup karyawan tetap karena merekalah yang umumnya menerima manfaat jangka panjang seperti pensiun atau imbalan pasca kerja, yang menjadi sumber timbulnya keuntungan atau kerugian aktuaria.

#### T105. Apakah karyawan kontrak/PKWT berdampak kepada OCI? 🟠

Pada dasarnya perhitungan karyawan kontrak tidak diatur jelas dalam standar praktik imbalan kerja. Sebagian KKA tidak memakai **OCI** karena menganggap itu bagian dari **OLTEB**. Kelebihan atau kekurangan kewajiban masuk di laporan laba rugi.

Tapi kami, KKA Nirmala disini memakai nilai OCI karena menilai kompensasi karyawan kontrak masuk kategori imbalan pasca kerja. Perhitungannya dengan metode PUC dengan nilai manfaat sebesar proporsi masa kerja x gaji yang menggunakan asumsi diskonto dan probabilitas. Perbedaan asumsi juga akan menimbulkan OCI.

Umumnya, perlakuannya disesuaikan dengan ketentuan kontrak setiap perusahaan.

#### T106. Bagaimana mutasi karyawan akan berdampak pada nilai Biaya Jasa Lalu dan OCI suatu perusahaan? 🟠

- Untuk **karyawan keluar**, jika **cadangan yang dicatat > realisasi**, selisihnya akan diakui pada **kurtailmen** sebagai penyelesaian. Namun, jika **cadangan yang dicatat < realisasi**, selisihnya akan diakui pada **biaya jasa lalu**.
- Untuk **karyawan masuk**, akan diakui pada biaya jasa lalu sebesar **PVDBO - CSC**

#### T107. Nilai manakah dari draft laporan aktuaria yang harus ditambahkan pada laporan laba rugi? 🟠

Nilai dengan keterangan 'Kewajiban/(Kekayaan) atas imbalan pasca kerja pada akhir periode' pada draft laporan. Nilai ini juga mencerminkan total beban yang diakui dalam periode berjalan.

Beban pada lampiran Tabel 5 tersebut seluruhnya diakui dalam laporan laba rugi dan terdiri dari dua komponen utama, yaitu:

- **Biaya Jasa Kini** yaitu biaya/kewajiban yang timbul mulai dari 1 Januari 2024 s.d. 31 Desember 2024 (untuk satu tahun berjalan)
- **Biaya Jasa Lalu** yaitu biaya yang seharusnya sudah dicadangkan pada tahun lalu (biaya atas kewajiban sebelumnya)

**See also:** T107 (Laporan Laba Rugi), A07 (Pencatatan Akuntansi)

#### T108. Bagaimana cara menghitung nilai kewajiban suatu perusahaan? 🟠

Nilai kewajiban akhir suatu perusahaan diperoleh dari:

- Nilai kewajiban awal (jika ada) (+)
- Beban tahun berjalan (+)
- Realisasi tahun berjalan (-)
- Nilai **OCI** tahun berjalan (+)
- Iuran DPLK porsi perusahaan tahun berjalan (jika ada) (-)

#### T109. Apa yang dimaksud dengan Nilai Kini Kewajiban (PVDBO) perusahaan per tanggal valuasi? 🟠

Nilai Kini Kewajiban (**PVDBO**) adalah estimasi atau proyeksi dari nilai total kewajiban yang harus dicadangkan oleh perusahaan pada tanggal perhitungan (tanggal valuasi) untuk memenuhi seluruh manfaat karyawan di masa mendatang. Perhitungan **PVDBO** mempertimbangkan faktor diskonto, tingkat kenaikan gaji, harapan hidup, dan probabilitas karyawan tetap bekerja hingga menerima manfaat.

Komponen **PVDBO** mencakup:

- **Manfaat Pensiun** – Nilai kini dari seluruh kewajiban pensiun yang harus disediakan oleh perusahaan
- **Manfaat Kematian** – Nilai kini dari manfaat yang akan diberikan kepada ahli waris jika karyawan meninggal sebelum pensiun
- **Manfaat Cacat/Sakit Berkepanjangan** – Nilai kini dari kewajiban perusahaan jika karyawan mengalami kecacatan/sakit berkepanjangan
- **Manfaat Pengunduran Diri** – Nilai kini dari manfaat yang akan diberikan kepada karyawan yang berhenti sebelum pensiun

#### T110. Untuk analisa jatuh tempo pembayaran manfaat, terdiskonto atau tidak terdiskonto? 🟠

Analisa jatuh tempo adalah pembayaran manfaat yang tidak terdiskonto.

#### T111. Jika terdapat karyawan yang keluar di tahun valuasi, namun realisasinya baru akan dibayarkan di tahun berikutnya, bagaimana perhitungan IPK-nya? 🟠

1. **Take-out cadangan saat ini, lalu biaya realisasi di tahun berikutnya**
    - Karyawan yang keluar dianggap sudah tidak menjadi bagian dari program pada tahun ini sehingga Cadangan (PVDBO) di-take-out dari kewajiban pada laporan tahun berjalan.
    - Pembayaran aktual di tahun berikutnya dicatat sebagai biaya langsung di laporan laba rugi tahun tersebut, tanpa lagi memengaruhi PVDBO, **atau**
2. **Tetap akui dalam PVDBO sampai manfaat dibayarkan**
    - Estimasi pembayaran manfaat tetap dipertahankan dalam PVDBO hingga benar-benar dibayarkan.
    - Kewajiban diakui pada periode saat karyawan keluar.
    - Penyesuaian dilakukan di tahun berikutnya saat pembayaran aktual dilakukan (misalnya jika nilai realisasi berbeda dari estimasi).

#### T112. Mengapa CSC naik, namun interest cost malah menurun? Sedangkan tingkat diskonto dan gaji naik dari tahun lalu sebelum valuasi, apakah ada faktor lain yang mempengaruhi interest cost? 🟠

Faktor yang mempengaruhi interest cost, antara lain:

- Jumlah karyawan
- Rata-rata masa kerja
- Perbedaan data karyawan tahun valuasi dengan tahun sebelumnya

Lalu, untuk perhitungan interest cost (Biaya Bunga) sendiri didapatkan dengan rumus: **Interest Cost = PVDBO awal periode × tingkat diskonto awal periode**

**See also:** T102 (Komponen Beban)

#### T113. Di tahun valuasi menunjukkan pendapatan aktuaria sangat besar. Apa penyebab utama hal ini? 🟠

Terdapat dua penyebab utama:  <br>1. Biaya jasa kini menurun tajam → bisa karena pengurangan jumlah peserta aktif atau penghapusan sebagian hak manfaat tahun berjalan.  <br>2. Reversal besar dari biaya jasa lalu → menunjukkan penghapusan kewajiban masa lalu karena perubahan struktur program.

**See also:** T103 (Biaya Jasa Lalu dan Biaya Jasa Kini), T156 (Curtailment dalam PSAK 219), T106 (Mutasi karyawan)


---

## #skenario-kompleks-dan-edge-cases

#### T151. Apakah karyawan yang sudah pensiun tetapi masih bekerja sebagai komisaris dan menerima gaji harus tetap dimasukkan dalam perhitungan PSAK 219? 🟠

Tidak. Karyawan yang telah pensiun dan diangkat sebagai komisaris tidak lagi masuk dalam perhitungan aktuaria PSAK 219 karena hubungan mereka dengan perusahaan berubah menjadi hubungan profesional, bukan lagi karyawan.

Namun, jika perusahaan memiliki kebijakan memberikan tunjangan pasca kerja kepada komisaris, kewajiban tersebut tetap harus dihitung, tetapi secara terpisah dari kewajiban aktuaria karyawan. Sesuai Pasal 15 UU 40/2007, komisaris dan direksi bukan bagian dari karyawan, sehingga tidak berhak atas imbalan kerja sesuai PSAK 219, kecuali ada kebijakan perusahaan yang secara khusus mengatur tunjangan mereka.

#### T152. Mengapa karyawan yang di atas usia pensiun masih masuk dalam perhitungan? 🟠

Selama karyawan belum menerima pesangon atas usia pensiunnya dan/memang masih bekerja, perusahaan masih memiliki kewajiban untuk mencadangkan pesangon untuk karyawan tersebut.

#### T153. Bila memiliki DPLK, data terkait apa saja yang dibutuhkan untuk perhitungan? 🟠

- Jika **DPLK DKP (Dana Kompensasi Pesangon)** : Kami membutuhkan data berupa saldo akhir DPLK per periode perhitungan, realisasi pembayaran manfaat, iuran DPLK, dan ROI rate dari DPLK.
- Jika **DPLK PPIP (Program Pensiun Iuran Pasti)** : Kami membutuhkan data berupa saldo akhir DPLK per periode perhitungan, iuran DPLK, dan ROI rate dari DPLK.

#### T154. Apa yang perlu diperhatikan jika perhitungan IPK termasuk imbalan jangka panjang lainnya (OLTEB) dan DPLK? 🟠

1. Pastikan aturan mengenai imbalan jangka panjang lainnya/OLTEB, seperti Cuti Besar (CBS) dan Penghargaan Emas, **sudah ditetapkan dalam kebijakan perusahaan**
2. Jika karyawan terdaftar dalam DPLK, kewajiban pensiun sebagian dapat ditanggung oleh DPLK, sehingga perusahaan hanya perlu mencatat kewajiban tambahan yang belum ditanggung oleh dana pensiun tersebut
3. Imbalan Jangka Panjang Lainnya yang belum digunakan, juga harus diperhitungkan dalam kewajiban perusahaan
4. Jika perusahaan sudah pernah melakukan perhitungan valuasi sebelumnya, pastikan bahwa perhitungan terbaru mencakup seluruh komponen ini agar nilai kewajiban yang diakui dalam laporan keuangan tetap sesuai dengan standar PSAK 219

#### T155. Bagaimana perhitungan kewajiban apabila karyawan terdaftar dalam DPLK dan pencatatannya dalam laporan keuangan? 🟠

1. **DPLK DKP** : Total saldo DPLK akhir periode dapat digunakan pengurang kewajiban, sehingga yang dicatat dalam laporan IPK hanya sisa kewajiban setelah dikurangi saldo DPLK yang tersedia. Imbal hasil dari DPLK akan dicatat sebagai pengurang di laba rugi.
    - Jika **liabilitas > DPLK DKP**, selisihnya masih harus dicatat sebagai **kewajiban di neraca.**
    - Jika **liabilitas ≤ DPLK DKP**, maka kewajiban di neraca bisa **dianggap nol**, karena sudah ter-cover oleh dana yang terkumpul di DPLK dan akan muncul dampak batas atas aset.
2. **DPLK PPIP** : Saldo DPLK akhir periode masing-masing karyawan akan diproyeksikan sampai usia pensiun dan dapat digunakan pengurang kewajiban, sehingga yang dicatat dalam laporan IPK hanya **sisa kewajiban setelah dikurangi saldo DPLK yang tersedia.**

#### T156. Apa itu Curtailment dalam perhitungan PSAK 219? Dan apa dampaknya pada laporan keuangan perusahaan? 🟠

Kurtailmen (curtailment) terjadi ketika perusahaan melakukan tindakan yang secara signifikan mengurangi jumlah karyawan yang berhak atas imbalan pasca kerja. Nilai ini berdampak pada estimasi nilai kewajiban / PVDBO yang mengakibatkan keuntungan atau kerugian yang harus segera diakui dalam laporan laba rugi.

**Kurtailmen** merupakan bagian dari biaya jasa lalu ketika ada penurunan signifikan yang dilakukan oleh entitas dalam hal jumlah pekerja yang ditanggung oleh program.

#### T157. Apakah nilai kewajiban dapat diperkecil? 🟠

Ya, bisa diperkecil dengan beberapa strategi, tetapi harus tetap sesuai dengan prinsip aktuaria dan standar akuntansi. Berikut adalah beberapa faktor yang dapat mengurangi nilai kewajiban:

1. **Mengubah Asumsi Aktuaria**
    - Meningkatkan tingkat diskonto
    - Menurunkan tingkat kenaikan gaji
    - Menyesuaikan asumsi mortalita & withdrawal
2. **Menyesuaikan Kebijakan Manfaat**
    - Mengurangi atau membatasi manfaat (contohnya, membatasi pesangon atau manfaat kesehatan pasca kerja)
    - Menerapkan batas maksimal manfaat
3. **Menggunakan Curtailment atau Settlement**
    - Curtailment → Penurunan signifikan yang dilakukan oleh entitas dalam hal jumlah pekerja yang ditanggung oleh program
    - Settlement → Menyelesaikan kewajiban lebih awal, misalnya dengan pembayaran lump sum atau membeli polis asuransi pensiun.

#### T158. Apakah ada keuntungan bagi perusahaan dalam menghitung CBS atau mendaftarkan karyawan ke dalam DPLK terkait perhitungan kewajiban? 🟠

Menghitung CBS dapat meningkatkan kewajiban imbalan jangka panjang lainnya, tetapi juga **dapat membantu dalam retensi karyawan**. Selain itu, mendaftarkan karyawan ke DPLK dapat mengurangi kewajiban perusahaan karena sebagian dari beban pensiun ditanggung oleh pihak ketiga, yang bisa **mengurangi beban finansial perusahaan dalam jangka panjang.**

#### T159. Jika ada karyawan yang diangkat menjadi direktur, apakah efeknya dalam perhitungan? 🟠

**Prinsip Umum:** Umumnya direktur memang **tidak dihitung** dalam perhitungan aktuaria reguler. Namun, dalam prakteknya terdapat beberapa skenario yang perlu dipertimbangkan:

**Skenario Perhitungan:**

**1. Masa Kerja Sebagai Karyawan → Direktur** Jika awalnya karyawan lalu diangkat menjadi direktur, maka masa kerja selama menjadi karyawan sampai diangkat menjadi direktur **dibayarkan dulu manfaat pensiun dini**. Lalu masa kerja mulai dari awal sejak diangkat menjadi direktur. Ini bisa saja manfaatnya masih mengikuti UU tapi dengan masa kerja baru atau bisa juga diatur terpisah manfaat/kompensasinya.

**2. Langsung Diangkat Sebagai Direktur** Dari awal bukan karyawan tapi langsung sebagai direktur, biasanya yang seperti ini sesuai dengan masa jabatan direksi. Apakah ada manfaat/kompensasi saat selesai menjabat diatur di perjanjian sendiri.

**3. Direktur yang Ditunjuk Langsung oleh Owner** Misal direktur ditunjuk langsung owner (misal keluarga sendiri), kami mengikuti informasi dari perusahaan. Namun, biasanya kalau owner merangkap jadi direktur ini yang **tidak dihitung**.

**Perlakuan Aktuaria:** Dalam praktiknya, saat karyawan diangkat menjadi Direksi maka perlu diperhitungkan sebagai **pensiun dini** dengan dibayarkan manfaat pensiun sesuai masa kerja dari awal bekerja sampai cut off sebelum diangkat sebagai direksi.

**See also:** L009 (Direksi dan Pesangon), L048 (Imbalan Direktur), T159 (Direktur Diangkat)

#### T160. Apakah pensiun dini tersebut bisa dihitungkan oleh aktuaris? 🟠

Bisa, namun kembali lagi kepada manajemen ingin menggunakan perhitungan aktuaris terkait pensiun dini atau memang mengacu pada pencadangan terakhir yang sudah pernah dihitung oleh aktuaris.

#### T161. Bagaimana jika terdapat realisasi pembayaran untuk karyawan yang sebelumnya belum masuk pada perhitungan IPK? 🟠

Pembayaran tersebut tetap diakui dalam perhitungan tahun berjalan. Namun, karena karyawan tersebut tidak termasuk dalam perhitungan IPK sebelumnya, maka **nominalnya langsung diakui sebagai biaya jasa lalu.**

#### T162. Bagaimana penjelasannya jika jumlah karyawan berkurang dan asumsi kenaikan gaji tidak signifikan, namun alokasi biaya justru meningkat cukup besar? 🟠

1. Karyawan tetap: Dalam perhitungan aktuaria, kenaikan gaji menjadi salah satu faktor utama yang meningkatkan nilai kewajiban, karena estimasi manfaat di masa depan bergantung pada besaran gaji terakhir karyawan. Maka dari itu, beban imbalan kerja tetap bisa meningkat walau jumlah karyawannya berkurang.
2. Karyawan kontrak: Terdapat perbedaan antara asumsi awal dan realisasi, baik dari sisi kenaikan gaji maupun komposisi data karyawan. Beberapa kemungkinan penyebab lonjakan biaya di antaranya:
    - Penambahan jumlah karyawan kontrak.
    - Bertambahnya masa kerja rata-rata yang berpengaruh pada nilai manfaat.
    - Perubahan asumsi atau parameter aktuaria seperti tingkat diskonto atau tingkat keluar-masuk karyawan.
    - Penggunaan data aktual menggantikan estimasi yang sebelumnya lebih rendah.

_Rekomendasi:_ Melakukan pengecekan kembali terhadap data gaji tahun sebelumnya, apakah saat itu sudah mencakup seluruh komponen tetap yang relevan. Jika sebelumnya hanya menggunakan gaji pokok sebagai dasar perhitungan, maka hasil perhitungan tahun tersebut kemungkinan terlalu rendah. Ketika datanya diperbaiki di tahun berikutnya menggunakan total gaji, maka kenaikan hasil perhitungan adalah hal yang wajar.

#### T163. Apakah realisasi untuk PHK di tahun berjalan harus tetap dihitung? 🟠

Tidak selalu. Jika karyawan sudah di PHK dan manfaat pesangonnya sudah dibayar penuh sebelum tanggal valuasi, maka tidak perlu dimasukkan dalam perhitungan aktuaria karena kewajibannya sudah direalisasikan. Namun, jika manfaatnya belum dibayar penuh, maka tetap harus dihitung dalam valuasi aktuaria sebagai bagian dari kewajiban perusahaan.

**Contoh Praktis:** Karyawan A di PHK pada 1 Juli 2024 dan dibayar pesangon sebesar Rp 150 juta pada tanggal 5 Juli 2024. → Karena pembayaran sudah dilakukan dan tercatat, maka tidak perlu dimasukkan dalam PVDBO per tanggal valuasi 31 Desember 2024.

**See also:** T156 (Curtailment), A051 (Restrukturisasi)

#### T164. Mengapa PVDBO bisa sangat besar walaupun hanya untuk sedikit jumlah karyawan? 🟠

PVDBO tergantung pada usia, masa kerja, dan gaji karyawan, bukan hanya jumlah orang. Karyawan yang mendekati pensiun dengan gaji tinggi dan masa kerja panjang akan memiliki nilai DBO yang besar karena manfaatnya lebih besar dan waktunya lebih dekat.

**Contoh Kalkulasi:** Dua karyawan berusia 58 dan 60, masing-masing bergaji Rp50 juta/bulan dan masa kerja >20 tahun. Jika pesangon dihitung 2 x gaji x masa kerja, maka total kewajiban bisa mencapai >Rp 1 M, didiskontokan menjadi PVDBO sekitar Rp 800 juta – Rp 1 M, tergantung asumsi diskonto dan mortalita. → Walau hanya dua orang, nilainya bisa besar jika profil mereka "mahal".

**See also:** T109 (Definisi PVDBO)

#### T165. Apa penyebab Biaya Jasa Lalu naik secara signifikan di tahun valuasi? 🟠

Biaya Jasa Lalu muncul karena ada perubahan manfaat yang berlaku surut (retrospektif), seperti:

1. Penambahan manfaat penghargaan masa kerja
2. Perubahan skema pesangon (misalnya dari 1x ke 2x gaji)
3. Penyesuaian aturan internal (misalnya mengacu ke PP No. 35 Tahun 2021)

Jika manfaat tersebut bersifat vested (tidak dapat ditarik kembali), maka biayanya langsung diakui di tahun valuasi.

**Contoh Implementasi:** Perusahaan di tahun 2024 mengubah kebijakan: karyawan yang bekerja ≥10 tahun berhak mendapat 1x gaji tambahan saat pensiun. (Seluruh karyawan yang sudah ≥10 tahun otomatis menciptakan Biaya Jasa Lalu, karena hak tersebut diberikan atas masa kerja sebelumnya) 20 orang karyawan terkena dampak, rata-rata manfaat tambahan Rp20 juta → total Biaya Jasa Lalu = Rp400 juta.

**See also:** T103 (Biaya Jasa Lalu vs Kini)

#### T166. Apakah karyawan yang akan di PHK tahun berikutnya (setelah tahun valuasi) perlu dimasukkan dalam perhitungan aktuaria? 🟠

Ya. Perhitungan aktuaria berbasis kondisi existing per tanggal laporan (cut-off date). Jadi meskipun manajemen sudah merencanakan PHK tahun berikutnya, selama belum terjadi, belum diumumkan resmi, dan belum dibayar, maka semua karyawan aktif tetap harus dihitung.

**Contoh Scenario:** Perusahaan berencana merumahkan 30 orang pada Maret 2025, tetapi per tanggal valuasi 31 Desember 2024, status mereka masih aktif dan belum ada surat resmi PHK. → Semua 30 orang tetap harus dimasukkan dalam valuasi aktuaria per 31 Desember 2024.

#### T167. Bagaimana cara aktuaris memperlakukan realisasi manfaat non PHK (pensiun, meninggal, cacat) dalam valuasi aktuaria? 🟠

Jika realisasi manfaat seperti pensiun atau meninggal terjadi di tahun valuasi, perlakuannya tergantung:

1. Jika manfaat sudah dibayar penuh sebelum tanggal valuasi, maka karyawan tersebut tidak masuk perhitungan valuasi
2. Jika masih dalam proses pembayaran atau tercatat sebagai utang perusahaan, maka tetap masuk dalam perhitungan valuasi

**Contoh Praktis:** Karyawan pensiun per 15 November 2024, namun pembayaran pensiunnya dijadwalkan pada Februari 2025. → Aktuaris akan tetap memasukkannya karena perusahaan masih punya kewajiban yang belum diselesaikan secara keuangan hingga tanggal valuasi 31 Desember 2024.

#### T168. Mengapa nilai kewajiban OLTEB per tahun valuasi turun drastis dibandingkan tahun sebelumnya? 🟠

Penurunan kewajiban OLTEB (Other Long-Term Employee Benefits) pada tahun valuasi kemungkinan besar terjadi karena:

1. Penghapusan atau penghentian program loyalitas atau imbalan jangka panjang
2. Penyesuaian kebijakan manfaat yang menyebabkan pembalikan (reversal) kewajiban masa lalu
3. Koreksi atas data karyawan atau manfaat vested (Biaya Jasa Lalu) yang tidak lagi berlaku

Dalam PSAK 219, OLTEB tidak menggunakan OCI, sehingga seluruh dampak penurunan langsung dicatat sebagai pendapatan di laporan laba rugi.

**See also:** T154 (OLTEB)

#### T169. Perhitungan untuk imbalan pasca kerja dimulai dari tahun karyawan pertama kali masuk kerja saat menjadi karyawan kontrak atau sudah diangkat menjadi karyawan tetap? 🟠

**Scenario 1:** Jika saat kontrak habis mendapatkan kompensasi dan baru diangkat menjadi karyawan tetap, maka menggunakan tanggal pengangkatan saat menjadi karyawan tetap.

**Scenario 2:** Jika saat kontrak habis tidak dibayarkan kompensasi kepada karyawan dan langsung diangkat menjadi karyawan tetap, maka menggunakan tanggal pertama kali karyawan tersebut bergabung.

**See also:** T053 (Karyawan Tetap vs Kontrak)

#### T170. Kenapa di laporan aktuaria kami terdapat nilai kurtailmen?	🟠

Jika perusahaan mengeluarkan realisasi atas PHK yang dilakukan kepada karyawan, yang ada di luar dari cakupan manfaat yang diatur/dihitung dalam PSAK 24 (Pensiun, Meninggal dunia, Cacat/sakit berkepanjangan, dan Resign), maka realisasi tersebut diakui sebagai kurtailmen.

**See also:** T156 (Curtailment dalam PSAK 219), T218 (Kompensasi PHK Kurtailmen), T106 (Mutasi Karyawan), A51 (Restrukturisasi PHK)

#### T171. Kenapa nilai kompensasi PHK tidak dimasukkan pada realisasi pembayaran dan malah masuk ke kurtailmen? 🟠

PSAK 24 hanya menghitung manfaat pasca kerja untuk karyawan yang Pensiun, Meninggal Dunia, Cacat/sakit berkepanjangan, dan Resign. Realisasi pada PSAK 24 dimaksudkan sebagai realisasi untuk mengurangi liabilitas/kewaijban yang terbentuk hanya untuk manfaat-manfaat yang diatur dalam PSAK 24. Apabila perusahaan melakukan PHK selain dari cakupan yang dihitung dari PSAK 24, dan perusahaan tersebut memberikan realisasi atas PHK nya, maka nilai realisasi tersebut masuk dalam Kurtailmen.

**See also:** T156 (Curtailment), T217 (Nilai Kurtailmen), T163 (Realisasi PHK)


---

## #advanced-topics-dan-expert-level
#### T201. Mengapa hasil perhitungan tahun valuasi lebih tinggi dibandingkan tahun sebelumnya? 🔴

Beberapa faktor yang dapat menyebabkan kenaikan nilai kewajiban:

- Bertambahnya usia setiap karyawan
- Bertambahnya masa kerja setiap karyawan
- Penurunan asumsi usia pensiun normal
- Perbedaan pengakuan usia pada tanggal valuasi
- Penurunan tingkat diskonto
- Kenaikan gaji aktual yang lebih besar daripada asumsi
- Penambahan komponen pajak dalam perhitungan
- Perubahan manfaat IPK yang digunakan (penambahan manfaat)

#### T202. Selain perbedaan data tanggal lahir atau mulai kerja, apakah ada perbedaan asumsi aktuaria atau metodologi yang digunakan dengan KKA sebelumnya? 🔴

Terkait asumsi: hanya tingkat diskonto yang berubah menyesuaikan dengan tanggal valuasi. Terkait metodologi: tidak ada perubahan, masih sama menggunakan metode PUC dengan penerapan IFRIC sesuai dengan yang dipersyaratkan PSAK.

Namun mungkin di antara KKA terdapat beberapa pendekatan yang berbeda yang digunakan dalam menentukan komponen berikut:

1. **Pengakuan Usia:** Karyawan berusia 40 tahun 5 bulan, ada KKA yang menggunakan usia 40 tahun, ada yang menggunakan 41 tahun, sedangkan kami menggunakan interpolasi dimana kami menghitung usia 40 dan 41 lalu di interpolasi sehingga tepat di usia 40 Tahun 5 bulan.
2. **Perhitungan Peluang Hidup:** Kami menggunakan Multiple Decrement dimana Peluang Meninggal, Peluang Sakit, Peluang Resign dan Peluang Pensiun saling keterkaitan satu sama lainnya.
3. **Penerapan Bunga Diskonto:** Pada umumnya perhitungan menggunakan single rate dimana diambil dari rata-rata sisa masa kerja, sedangkan kami menggunakan multiple rate, dimana tiap karyawan memiliki bunga diskonto yang berbeda-beda.

**Contoh Multiple Rate:** Karyawan berusia 40 tahun, jika usia pensiun normal 56 tahun, maka bunga diskonto yang diambil adalah yield ke 16 (56-40), sedangkan karyawan berusia 50 tahun, maka bunga diskonto yang diambil adalah yield ke 6 (56-50).

**See also:** T056 (Multiple vs Single Rate), T055 (Diferensiasi KKA)

#### T203. Apakah tunjangan variabel juga perlu dihitung jika menjadi bagian dari total gaji pada beberapa karyawan? 🔴

Tunjangan variabel, seperti bonus tahunan atau insentif berbasis kinerja, umumnya tidak dihitung dalam perhitungan aktuaria imbalan kerja karena sifatnya tidak tetap dan tidak bisa diproyeksikan secara konsisten. Namun, jika perusahaan secara jelas menyatakan bahwa tunjangan variabel ini termasuk dalam perhitungan imbalan pasca kerja, maka harus dimasukkan dalam total gaji.

#### T204. Apakah tingkat dan rata-rata kenaikan gaji dihitung dari Gaji Pokok saja atau Total Gaji (gaji pokok + tunjangan tetap)? 🔴

**Contoh Perhitungan:**

Seorang karyawan memiliki Gaji Pokok sebesar Rp10 juta dan Tunjangan Tetap Rp2 juta. Total gaji menjadi Rp12 juta. Setelah satu tahun, Gaji Pokok naik menjadi Rp11 juta, dan Tunjangan Tetap tetap Rp2 juta. Total gaji baru menjadi Rp13 juta.

Pertanyaan: Berapa tingkat Kenaikan Gaji berdasarkan Gaji Pokok? Berapa tingkat Kenaikan Gaji berdasarkan Total Gaji?

Jawaban:

- Tingkat Kenaikan Gaji Pokok: Tingkat Kenaikan = (Gaji Pokok Baru − Gaji Pokok Lama) / Gaji Pokok Lama × 100% Tingkat Kenaikan = (11.000.000 − 10.000.000) / 10.000.000 × 100% Tingkat Kenaikan = 10%
- Tingkat Kenaikan Total Gaji: Tingkat Kenaikan = (Total Gaji Baru − Total Gaji Lama) / Total Gaji Lama × 100% Tingkat Kenaikan = (13.000.000 − 12.000.000) / 12.000.000 × 100% Tingkat Kenaikan = 8,33%

Total gaji lebih relevan karena imbalan pasca kerja seperti pesangon atau manfaat pensiun sering dihitung berdasarkan **gaji terakhir yang mencakup gaji pokok dan tunjangan tetap**.

#### T205. Ada karyawan yang di bawah 31 tahun masuk ke dalam perhitungan employment benefit. Berdasarkan IFRIC update 2022, persyaratan usia karyawan tetap yang boleh diakui adalah di atas 31 tahun atau maksimal sisa masa kerja 24 tahun. Bagaimana penjelasannya? 🔴

Pertama-tama, adanya perbedaan antara Provisional Employment Benefit (PEB) dan Post Employment Benefit (imbalan pasca kerja/IPK).

**PEB:**

- Standar PSAK 57/IAS 37
- Jangka pendek
- Belum pasti
- Timing selama masa kerja
- Perhitungan relatif sederhana

**IPK:**

- Standar PSAK 219/IAS 19
- Jangka panjang
- Relatif pasti
- Timing setelah masa kerja berakhir
- Perhitungan kompleks sehingga membutuhkan aktuaris

IFRIC mengatur hanya untuk atribusi manfaat pensiun normal terkait imbalan pasca kerja. Berikut ketentuan manfaat IPK berdasarkan usia karyawan:

- **Di atas 31 tahun**: mencakup manfaat pensiun normal, meninggal dunia, sakit berkepanjangan, dan mengundurkan diri
- **Di bawah 31 tahun**: hanya untuk manfaat meninggal dunia, sakit berkepanjangan, dan mengundurkan diri

#### T206. Apabila basis perhitungan dengan diketahui data: Rata-rata kenaikan gaji karyawan tetap di 2023 sebesar Rp9.304.058, di 2024 sebesar Rp10.096.908 dengan kondisi karyawan tetap berkurang 1 orang namun perhitungan kewajibannya justru meningkat, lalu rata-rata kenaikan gaji karyawan kontrak di 2023 sebesar Rp5.236.600, di 2024 sebesar Rp5.297.424 → kenaikan gaji periode 2024 sekitar 1,16% dibandingkan dengan 2023 yang sebesar 3% namun hasil alokasi biayanya meningkat hingga di Rp1,64 miliar. Bagaimana penjelasannya? 🔴

**Analisis Karyawan Tetap:** Meskipun jumlah karyawan tetap berkurang 1 orang, nilai rata-rata kenaikan gaji meningkat sekitar 8,52%, dari Rp9,3 juta ke Rp10 juta. Hal ini menyebabkan total beban imbalan kerja tetap meningkat, walaupun headcount turun. Dalam perhitungan aktuaria, kenaikan gaji memiliki pengaruh besar, terutama karena:

- **Proyeksi kewajiban dihitung berdasarkan gaji masa depan**
- Kenaikan 8,52% cukup tinggi dan akan berdampak pada nilai kini (present value) kewajiban

**Analisis Karyawan Kontrak:** Ada perbedaan antara asumsi awal (kenaikan gaji 3%) dan aktual (kenaikan hanya 1,16%), tetapi hasil alokasi biaya meningkat ke Rp1,64 miliar.

Kemungkinan penyebabnya:

- Ada **penambahan jumlah karyawan kontrak** yang signifikan
- **Masa kerja atau usia karyawan kontrak bertambah** → meningkatkan nilai manfaat
- Terjadi perbedaan asumsi aktuaria atau parameter lain (misalnya tingkat diskonto, turnover, dan lainnya) antara 2023 dan 2024
- Penggunaan parameter aktual menggantikan asumsi estimasi (misal: realisasi lebih besar dari ekspektasi)

**Rekomendasi**: Perlu dilakukan **pengecekan ulang terhadap data penghasilan** tahun 2023 – apakah saat itu sudah mencakup seluruh komponen tetap yang relevan dengan manfaat pasca kerja. Bila hanya gaji pokok yang digunakan, maka perlu dilakukan rekonsiliasi dan normalisasi data untuk memastikan konsistensi tahun ke tahun.

Jika benar tahun **2023 hanya memakai gaji pokok** sebagai basis valuasi, maka perhitungan manfaat tahun tersebut understated (terlalu rendah). Saat data **2024 diperbaiki dengan total gaji**, wajar jika manfaat melonjak.

**See also:** T054 (Tingkat Kenaikan Gaji), T204 (Dasar Gaji)


---

**Disclaimer Teknis:** Informasi dalam FAQ ini berdasarkan standar PSAK 219, IAS 19, dan praktik aktuaria terbaik. Untuk implementasi spesifik, disarankan berkonsultasi dengan aktuaris bersertifikat dan mengikuti perkembangan regulasi terbaru.

**Update Terakhir:** Januari 2025  
**Referensi Teknis:** PSAK 219, IAS 19, IFRIC Interpretations, SPAI (Standar Profesi Aktuaris Indonesia)

---

**Catatan**: FAQ ini disusun berdasarkan pengalaman praktis KKA Nirmala dan standar internasional. Untuk kasus kompleks atau situasi khusus, silakan hubungi tim aktuaris kami untuk konsultasi mendalam.

---