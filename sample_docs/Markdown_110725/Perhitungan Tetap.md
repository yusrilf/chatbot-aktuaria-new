
## **Step 1**
adalah potongan dari sheet calculate 

## #Usia-Mulai-Masa-Kerja 

Usia Mulai Masa Kerja = selisih antara `Tanggal masuk kerja` dan `Tanggal lahir` dibagi 365,25 dan dibulatkan 2 desimal.

- Jika hasilnya ≥ [[Asumsi#Usia-Pensiun]], maka gunakan `Usia Pensiun`.
- Jika hasilnya <  [[Asumsi#Usia-Pensiun]], gunakan **hasil perhitungan** tersebut.

## #Usia-Saat-Valuasi 

Usia Saat Valuasi = selisih antara `Tanggal valuasi` dan `Tanggal lahir` dibagi 365,25 dan dibulatkan 2 desimal.

- Jika hasilnya ≥  [[Asumsi#Usia-Pensiun]], maka gunakan `Usia Pensiun`.
- Jika hasilnya <  [[Asumsi#Usia-Pensiun]], gunakan **hasil perhitungan** tersebut.

## #Total-Gaji 

Total Gaji= Total `Gaji pokok` dan `Tunjangan tetap` (jika ada)

## #Batas-Usia-Valuasi

Dalam praktik aktuaria, usia valuasi yang tidak bulat sering kali perlu diinterpolasi antara #Batas-Bawah dan #Batas-Bawah. Berikut adalah penjabaran perhitungan masing-masing batas usia dan turunannya:


### #Batas-Bawah (Down)

- #Bagian-desimal-usia  
  Menggambarkan bagian pecahan dari usia, misalnya jika usia valuasi adalah 35,8 maka bagian desimalnya adalah 0,8.

- #Usia-Valuasi-bulat-bawah  
  Merupakan usia yang dibulatkan ke bawah dari usia valuasi. Dalam contoh 35,8 maka batas bawah adalah 35 tahun.

- #Masa-kerja-hingga-usia-valuasi-batas-bawah  
  Merupakan selisih antara #Usia-Valuasi-bulat-bawah dengan #Usia-Mulai-Masa-Kerja .

- #Sisa-masa-kerja-hingga-usia-pensiun (berdasarkan batas bawah)  
  Dihitung dari selisih antara [[Asumsi#Usia-Pensiun]] dan #Usia-Valuasi-bulat-bawah .

- #Total-masa-kerja-hingga-pensiun-dari-batas-bawah-usia  
  Merupakan jumlah dari #Masa-Kerja-Lalu hingga #Usia-Valuasi-bulat-bawah dan #Future-Service hingga pensiun.

---

### #Batas-Atas (Up)

- #Sisa-pecahan-menuju-usia-bulat-atas  
  Menggambarkan seberapa dekat usia valuasi dengan batas atas, dihitung dari 1 dikurangi #Bagian-desimal-usia .

- #Usia-Valuasi-bulat-atas  
  Merupakan usia yang dibulatkan ke atas dari usia valuasi. Misalnya, jika usia valuasi adalah 35,8 maka batas atasnya adalah 36 tahun.

- #Masa-kerja-hingga-usia-valuasi-batas-atas  
  Biasanya dihitung dengan **menambahkan 1 tahun** pada #Masa-kerja-hingga-usia-valuasi-batas-bawah 

- #Sisa-masa-kerja-hingga-usia-pensiun (berdasarkan batas atas)  
  Dihitung dari selisih antara [[Asumsi#Usia-Pensiun]] dan #Usia-Valuasi-bulat-atas .

- #Total-masa-kerja-hingga-pensiun-dari-batas-atas-usia  
  Merupakan jumlah #Masa-Kerja-Lalu hingga #Usia-Valuasi-bulat-atas  dan sisa masa kerja setelahnya sampai dengan usia pensiun.


## #Masa-Kerja-Lalu 

Masa kerja lalu =  Selisih #Usia-Saat-Valuasi dan #Usia-Saat-Masa-Kerja , lalu dibulatkan 2 desimal.

## #Future-Service

Future Service (Sisa Masa Kerja) = Selisih [[Asumsi#Usia-Pensiun]] dan #Usia-Saat-Valuasi  dan dibulatkan 2 desimal.

## #Total-Masa-Kerja

Total Masa Kerja = penjumlahan dari #Masa-Kerja-Lalu dan #Future-Service

## #Sx

**Sx** adalah faktor kenaikan gaji di masa depan yang digunakan untuk menghitung proyeksi gaji sampai usia pensiun atau akhir masa kerja.

**Rumus:**

Sx = (1 + #Future-Service) ^ ([[Asumsi#Kenaikan-Gaji-Konversi-ke-bulanan]] × 12)


## #PV 

**PV (Present Value)** adalah nilai kini dari sejumlah uang atau manfaat yang akan diterima di masa depan, yang dihitung dengan mendiskontokan nilai tersebut menggunakan tingkat diskonto yang berlaku.

**Rumus**:

PV = (1 + #Future-Service) ^ (–[[Asumsi#Tingkat-Diskonto]])

## #Total-Gaji 

Total Gaji= Total `Gaji pokok` dan `Tunjangan tetap` (jika ada)

## #Total-Manfaat-Pensiun


**Total Manfaat Pensiun** adalah **jumlah keseluruhan hak finansial** yang diterima peserta saat memasuki masa pensiun.

Jika manfaat pensiun **ditanggung oleh perusahaan**, maka nilai yang digunakan adalah:

> #Manfaat-Pensiun-Sebelum-Pajak + #Pajak 
> (Pajak dianggap sebagai beban Perusahaan)

Namun, jika manfaat pensiun **ditanggung oleh pegawai, maka nilai yang digunakan cukup:

> #Manfaat-Pensiun-Sebelum-Pajak
> (Pajak dianggap sebagai beban pegawai)

## #Manfaat-Pensiun-Sebelum-Pajak

Manfaat pensiun sebelum pajak dihitung dengan mengambil persentase manfaat pensiun dari [[Asumsi#Program-Manfaat]] berdasarkan #Total-Masa-Kerja. Nilai tersebut kemudian dikalikan dengan #Total-Gaji  dan  #sx

## #Pajak

Pajak penghasilan (PPh 21) atas manfaat pensiun dihitung secara **progresif** berdasarkan total manfaat pensiun yang diterima peserta sebelum pajak. Penghitungan ini mengikuti lapisan tarif yang telah ditetapkan pemerintah, dengan tarif yang semakin tinggi seiring meningkatnya jumlah manfaat.

Berikut adalah lapisan tarif yang berlaku:

- **0 – 60 juta rupiah**: dikenakan tarif **5%**
    
- **60 juta – 250 juta rupiah**: dikenakan tarif **15%**
    
- **250 juta – 500 juta rupiah**: dikenakan tarif **25%**
    
- **500 juta – 5 miliar rupiah**: dikenakan tarif **30%**
    
- **Di atas 5 miliar rupiah**: dikenakan tarif **35%**

Menjumlahkan total pajak dari masing-masing lapisan, menghasilkan total pajak atas manfaat pensiun yang diterima. 

## **Step 2**
## #qₓʷ  

Nilai [[Asumsi#Asumsi-Resign]] atau angka **Withdrawal Rate** `qₓʷ` dengan kondisi:

- Untuk usia kerja `x` = 0 hingga `x` = 14, maka `qₓᵈ` = 0
- Untuk `x` = 15 hingga 1 tahun sebelum `Usia Pensiun` (misal pensiun di `x` = 60, maka sampai `x` = 59) → **ambil dari tabel Withdrawal Assumption**
- Untuk `x` = `Usia Pensiun` maka `qₓᵈ` = **0%** (karena tidak ada peluang berhenti secara sukarela pada usia pensiun)

## #qₓᵈ 

Tabel tingkat mortalita [[Asumsi#Asumsi-Kematian]] `qₓᵈ` yang digunakan sebagai referensi adalah **TMI IV 2019 (Tabel Mortalita Indonesia IV Tahun 2019)**. Ini adalah salah satu tabel mortalita resmi yang disusun oleh Persatuan Aktuaris Indonesia (PAI) dan digunakan luas dalam praktik aktuaria di Indonesia.

Asumsi **Mortality Rate** diambil nilai `qₓ` dari Tabel Mortalita TMI IV dengan kondisi x = 15 tahun hingga #Usia-Pensiun 

## #qₓⁱ 

[[Asumsi#Asumsi-Kecacatan]] atau angka disability rate `qₓⁱ` biasanya ditentukan secara asumtif.
Mengacu pada **5% hingga 10%** dari `qₓⁱ` per tiap usia kerjanya hingga pensiun.

## #qₓʳ

 [[Asumsi#Probabilitas-Pensiun]] Nilai asumsi **Pension Rate** `qₓʳ` dengan kondisi:
- Untuk usia kerja `x` = 0 hingga 1 tahun sebelum `Usia Pensiun`, maka `qₓʳ` = 0 
- Untuk `x` = `Usia Pensiun`, maka `qₓʳ` = **1** (karena dianggap pensiun normal)

## #qₓʷ-aksen 

Undur diri terkoreksi `qₓ′ʷ` merupakan **koreksi peluang berhenti kerja atau pensiun dini**, dengan mempertimbangkan bahwa karyawan bisa keluar dari skema karena sebab lainnya, seperti kematian, cacat / sakit berkepanjangan, dan pensiun normal.

**Rumus:**

`qₓ′ʷ` = #qₓʷ x (1 - (0,5 x #qₓᵈ )) x (1 - (0,5 x #qₓⁱ )) x (1 - (0,5 x #qₓʳ ))

## #qₓᵈ-aksen 

 Mortalita terkoreksi `qₓ′ᵈ` merupakan **koreksi peluang meninggal dunia**, dengan mempertimbangkan bahwa karyawan bisa keluar dari skema karena sebab lainnya, seperti berhenti kerja atau pensiun dini, cacat / sakit berkepanjangan, dan pensiun normal.
 
 **Rumus:**

`qₓ′ᵈ` = #qₓᵈ x (1 - (0,5 x #qₓʷ )) x (1 - (0,5 x #qₓⁱ )) x (1 - (0,5 x #qₓʳ ))

## #qₓⁱ-aksen

Cacat terkoreksi `qₓ′ⁱ` merupakan **koreksi peluang cacat / sakit berkepanjangan**, dengan mempertimbangkan bahwa karyawan bisa keluar dari skema karena sebab lainnya, seperti berhenti kerja atau pensiun dini, meninggal dunia, dan pensiun normal.

**Rumus:**

`qₓ'ⁱ` = #qₓⁱ x (1 - (0,5 x #qₓʷ )) x (1 - (0,5 x #qₓᵈ )) x (1 - (0,5 x #qₓʳ ))

## #qₓʳ-aksen

Pensiun terkoreksi `qₓ′ʳ` merupakan **koreksi peluang pensiun normal**, dengan mempertimbangkan bahwa karyawan bisa keluar dari skema karena sebab lainnya..

**Rumus:**

`qₓ′ʳ` =  #qₓʳ x (1 - (0,5 x #qₓʷ )) x (1 - (0,5 x #qₓᵈ )) x (1 - (0,5 x #qₓⁱ ))

yang mana nilai yang dihasilkan selalu **1**

## #lₓᵀ  

 `lₓᵀ` adalah jumlah karyawan yang **masih hidup dan aktif** pada usia kerja `x` dalam model **multiple decrement**.
 
 Nilai ini menurun tiap usia karena ada pengurangan akibat kematian, cacat, pensiun dini, dan berhenti kerja (withdrawal). 
 
 - Untuk usia kerja `x` = 0 hingga `x` = 14, maka `lₓᵀ` = 0
 - Untuk `x` = 15, maka `lₓᵀ` = [[#Radix]] (sebagai awal perhitungan)
 - Untuk `x` = 16 hingga `Usia Pensiun`, gunakan rumus perhitungan:
 
   `lₓᵀ` = #l₍ₓ₋₁₎ᵀ − ( #d₍ₓ₋₁₎ᵀ )

### #l₍ₓ₋₁₎ᵀ

Jumlah peserta (karyawan) yang **masih hidup dan aktif pada usia kerja sebelumnya**, yaitu usia (x−1)
 
## #d₍ₓ₋₁₎ᵀ

**Total decrement** pada usia kerja sebelumnya, yaitu usia **(x−1)**

**Rumus:**

`d₍ₓ₋₁₎ᵀ` = #d₍ₓ₋₁₎ + #i₍ₓ₋₁₎ + #r₍ₓ₋₁₎ + #w₍ₓ₋₁₎

## #wₓ 

**Multiple decrement resign** `wₓ` adalah jumlah undur diri pada usia `x` .

**Rumus**:

`wₓ` = #lₓᵀ x #qₓʷ-aksen  

## #dₓ 

**Multiple decrement kematian** `dₓ` adalah jumlah **kematian** pada usia `x` dalam model multiple decrement.

**Rumus**:

`dₓ` = #lₓᵀ x #qₓᵈ-aksen 

## #iₓ 

**Multiple decrement cacat** `iₓ` adalah jumlah **kecacatan** pada usia `x` dalam model multiple decrement.

**Rumus**:

`iₓ` = #lₓᵀ x #qₓⁱ-aksen  

## #rₓ 

**Multiple decrement pensiun** `rₓ` adalah jumlah **pensiun dini** pada usia `x`.

**Rumus**:

`iₓ` = #lₓᵀ x #qₓⁱ-aksen  

## #dₓᵀ  

 `dₓᵀ` adalah **total decrement** pada usia `x` 

**Rumus**:

`dₓᵀ` = #wₓ + #dₓ + #iₓ + #rₓ 

## #vˣ

Faktor `vˣ` untuk menghitung **nilai sekarang** dari Rp1 yang akan dibayarkan di masa depan pada usia tertentu (`x`).

**Rumus:**

`vˣ` = (1 + [[Asumsi#Tingkat-Diskonto]]) ^ -x

Nilai #vˣ  **bersifat unik** dan akan berbeda untuk setiap karyawan, karena:

- Tiap karyawan memiliki **usia kerja yang berbeda**
- Jarak waktu antara **hari ini** dan [[Asumsi#Usia-Pensiun]] tidak sama untuk semua karyawan

Maka, perhitungan #vˣ harus mempertimbangkan **jangka waktu spesifik** tiap karyawan. Dengan kata lain:
- Karyawan muda akan memiliki nilai `vˣ` yang **lebih kecil** karena uang akan dibayarkan lebih lama lagi di masa depan → **butuh diskonto lebih besar**
- Karyawan yang lebih dekat dengan pensiun akan memiliki nilai `vˣ` yang **lebih besar** (lebih mendekati 1)

## #Dₓ

`Dₓ`​ adalah **nilai sekarang** dari karyawan hidup pada usia **x**

**Rumus:** 

`Dₓ` = [[Step 2#lₓᵀ]] x #vˣ 

## #Dᵣ

`Dᵣ`​ adalah nilai #Dₓ  dari karyawan pada [[Asumsi#Usia-Pensiun]] `x`

## #Dᵣ-per-Dₓ

 `Dᵣ/Dₓ` adalah rasio nilai diskonto pada #Usia-Pensiun (`Dᵣ`) dibandingkan dengan nilai diskonto pada usia kerja tertentu (`Dₓ`​). 
 
 Rasio ini digunakan untuk **mengalokasikan bobot proporsi beban jasa kini** (Current Service Cost / CSC). CSC tahun berjalan dihitung sebagai persentase kewajiban yang belum diakui pada usia `x`.
 
 Cek nilai #Dₓ :
 - apabila `Dₓ` = 0 , maka nilai `Dᵣ/Dₓ` = 0 (untuk hindari value error)
 - apabila `Dₓ` ≠ 0 , maka gunakan hasil perhitungan dari `Dᵣ/Dₓ` tersebut


## **Step 3**
## #Tabel-tpₓ

Tabel ini umum digunakan untuk menghitung faktor pengali manfaat dengan mempertimbangkan **kemungkinan seseorang masih hidup** saat ini di usia `x` selama `t` tahun ke depan.

Faktor pengali ini berperan penting dalam menentukan **expected Present Value** (estimasi nilai kini) dari berbagai manfaat, tergantung pada masa kerja, [[Asumsi#Usia-Pensiun]], dan [[Asumsi#Asumsi-Kematian]].

Kolom tabel diawali dengan **1pₓ** sebagai peluang hidup individu x+1, kemudian berlanjut secara urut hingga kolom **45pₓ** sebagai peluang hidup individu x+45.

> **Contoh:**  
> Jika usia saat valuasi = 25 tahun, maka `ₜpₓ` = 40p₂₅ yang menunjukkan probabilitas bahwa karyawan tersebut akan hidup sampai usia 65 tahun (25 + 40).

**Rumus:**

`ₜpₓ` = #lₓ₊ₜ / #lₓᵀ

dengan:
- #lₓᵀ ​: peluang individu hidup di usia `x` [[Step 2#lₓᵀ]]
- #lₓ₊ₜ ​: peluang individu hidup di usia `x + t`

syarat:
- `lₓᵀ` = 0 , maka nilai ₜpₓ = 0 (untuk hindari value error)
 - `lₓᵀ` ≠ 0 , maka gunakan hasil perhitungan dari ₜpₓ
 
## #Tabel-qₓᵈ

Tabel ini digunakan untuk menunjukkan **kemungkinan karyawan yang berusia `x` saat ini akan meninggal dunia** dalam waktu `t` tahun ke depan, dengan kondisi bahwa manfaat yang akan diterima **ditangguhkan (delayed benefit)**.

Tabel ini berisi faktor pengali untuk perhitungan **nilai kini manfaat (Present Value)** yang berkaitan dengan risiko kematian dan manfaat tertunda.

**Rumus:**

ₜqₓᵈ = [[#Tabel-tpₓ]] × `qₓ₊ₜᵈ`

dengan:
- `tpₓ`: peluang bahwa karyawan berusia `x` akan **masih hidup hingga awal tahun ke-t**.
- `qₓ₊ₜᵈ` ​: peluang karyawan meninggal dunia tepat pada tahun ke-**t**, dengan skema **penundaan manfaat**, mengacu ke nilai dari [[Step 2#qₓᵈ]]

> **Contoh:**  
> Jika usia saat valuasi = 30 tahun, dan perusahaan ingin menghitung **kemungkinan karyawan tersebut meninggal dunia pada usia 35 tahun** (setelah hidup 5 tahun sejak usia 30), maka gunakan `ₜqₓᵈ` = 5p₃₀ × q₃₅ᵈ

## #Tabel-qₓⁱ 

Tabel ini digunakan untuk menunjukkan **kemungkinan seseorang yang berusia `x` saat ini akan mengalami cacat / sakit berkepanjangan** dalam waktu `t` tahun ke depan, berdasarkan **asumsi individual** atau kondisi khusus yang telah ditetapkan — seperti status kesehatan, pekerjaan, gaya hidup, atau klasifikasi risiko lainnya.

Tabel ini berisi faktor pengali untuk perhitungan **nilai kini manfaat (Present Value)** yang berkaitan dengan risiko kecacatan dan manfaat tertunda.

**Rumus:**

ₜqₓⁱ = [[#Tabel-tpₓ]] × `qₓ₊ₜⁱ` 

dengan:
- `tpₓ`: peluang bahwa karyawan berusia `x` akan **masih hidup hingga awal tahun ke-t**.
- `qₓ₊ₜⁱ` ​: peluang karyawan mengalami kecacatan tepat pada tahun ke-**t**, dengan skema **penundaan manfaat**, mengacu ke nilai dari [[Step 2#qₓⁱ]]

> **Contoh:**  
> Jika usia saat valuasi = 40 tahun. Berdasarkan **profil risiko individu** (misalnya karyawan dengan pekerjaan berisiko tinggi), perusahaan ingin menghitung **kemungkinan karyawan tersebut mengalami kecacatan / sakit berkepanjangan pada usia 45 tahun** (mengalami cacat 5 tahun setelah usia 40), maka gunakan `ₜqₓⁱ` = 5p₄₀ × q₄₅ⁱ

## #Tabel-qₓʷ 

Tabel ini digunakan untuk menunjukkan **kemungkinan seseorang yang berusia `x` saat ini akan mengundurkan diri atau resign** dalam waktu `t` tahun ke depan (sebelum [[Asumsi#Usia-Pensiun]] normal), berdasarkan **profil atau kondisi tertentu** — seperti tingkat kepuasan kerja, kontrak kerja, atau faktor lain yang memengaruhi keputusan resign.

Tabel ini berfungsi sebagai faktor pengali dalam perhitungan **nilai kini manfaat (Present Value)** yang berhubungan dengan risiko resign dan manfaat yang mungkin tertunda.

**Rumus:**

ₜqₓʷ = [[#Tabel-tpₓ]] × `qₓ₊ₜʷ`

dengan:
- `tpₓ`: peluang bahwa karyawan berusia `x` akan **masih hidup hingga awal tahun ke-t**.
- `qₓ₊ₜʷ` : peluang karyawan resign tepat pada tahun ke-**t**, dengan skema **penundaan manfaat**, mengacu ke nilai dari [[Step 2#qₓʷ]]

> **Contoh:**  
> Jika usia saat valuasi = 20 tahun, sementara usia pensiun pada perusahaan adalah 57, dan perusahaan ingin menghitung kemungkinan karyawan tersebut **mengundurkan diri sebelum** `Usia Pensiun` yaitu pada usia 35 tahun (15 tahun setelah usia 20), maka kita menggunakan `ₜqₓʷ` = 15p₂₀ × q₃₅ʷ

## **Step 4**
## #TMK1

TMK1 merupakan suatu nilai yang dihitung berdasarkan perbandingan antara #Total-masa-kerja-hingga-pensiun-dari-batas-bawah-usia hingga #Masa-kerja-hingga-usia-valuasi-batas-bawah, ditambah dengan variabel usia `x` (rentang 1-40) terhadap #Total-masa-kerja-hingga-pensiun-dari-batas-bawah-usia  hingga [[Perhitungan Valuasi Aktuaria/Asumsi#Usia-Pensiun|Usia Pensiun]] dari #Masa-kerja-hingga-usia-valuasi-batas-bawah.


## #TMK2

TMK2 merupakan nilai yang dihitung berdasarkan dua kondisi utama terkait #TMK1 dan #Total-masa-kerja-hingga-pensiun-dari-batas-atas-usia .
#### Aturan Perhitungan TMK2:

1. Jika #TMK1 = 0, maka TMK2 = 0.
    
    - Artinya, jika sebelumnya #TMK1 sudah melebihi batas yang diizinkan (sehingga bernilai 0), maka TMK2 otomatis dianggap 0.
        
2. Jika #TMK1 = #Total-masa-kerja-hingga-pensiun-dari-batas-bawah-usia , maka TMK2 = 0.
    
    - Ini berarti jika masa kerja #TMK1 sudah mencapai batas maksimal  [[Perhitungan Valuasi Aktuaria/Asumsi#Usia-Pensiun|Usia Pensiun]], maka tidak ada penambahan lagi (TMK2 = 0).
        
3. Jika #TMK1  ≠ 0 dan belum mencapai #Total-masa-kerja-hingga-pensiun-dari-batas-bawah-usia, maka TMK2 = #TMK1 + 1.
    
    - Dalam kondisi normal (masih dalam rentang masa kerja yang diizinkan), TMK2 akan menambahkan 1 tahun ke #TMK1.


## #v1

 Faktor diskonto yang mengkonversi nilai manfaat masa depan menjadi nilai sekarang dengan memperhitungkan [[Asumsi#Tingkat-Diskonto]]  yang berlaku untuk tahun `x` selama masa #Future-Service 

**Rumus**:

v1ₓ = (1 + [[Asumsi#Tingkat-Diskonto]]) ^ -x

- Jika TMK1 pada `x` = 0 , maka v1 = 0.
- Jika tidak, maka **gunakan hasil perhitungan** rumus v1.

## #sₓ 

Menunjukkan nilai saldo gaji yang disesuaikan dengan asumsi #Tingkat-Kenaikan-Gaji tahunan pada tahun ke `x` masa kerja karyawan hingga #Usia-Pensiun 

Nilai ini merepresentasikan **perkiraan gaji aktual pada tahun ke `x`** di masa #Future-Service yang digunakan sebagai dasar perhitungan manfaat aktual, seperti pensiun atau manfaat imbalan kerja lainnya.

**Rumus:**

sₓ = (1 + [[Asumsi#Tingkat-Kenaikan-Gaji]]) ^ x-1

## #Manfaat
-  #Manfaat-Pengali-Meninggal  Manfaat ini didapat ketika meninggal dihitung :
	  - Jika #TMK1 = 0, maka nilai manfaat adalah **0**.
	  - Jika #TMK1 ≥ 40, maka nilai manfaat menggunakan #TMK1 = 40.
	  - Jika tidak memenuhi dua kondisi di atas, maka nilai manfaat ditentukan berdasarkan nilai #TMK1 yang merujuk pada bagian *death/meninggal* dalam tabel [[Asumsi#Program-Manfaat]].

- #Manfaat-Pengali-Cacat  Manfaat ini didapat ketika cacat :
	- Jika #TMK1 = 0, maka nilai manfaat adalah **0**.
	- Jika #TMK1 ≥ 40, maka nilai manfaat menggunakan #TMK1 = 40.
	- Jika tidak memenuhi dua kondisi di atas, maka nilai manfaat ditentukan berdasarkan nilai #TMK1 yang merujuk pada bagian *cacat* dalam tabel [[Asumsi#Program-Manfaat]].

- #Manfaat-Pengali-Resign Manfaat ini didapat ketika resign :
	- Jika #TMK1 = 0, maka nilai manfaat adalah **0**.
	- Jika #TMK1 ≥ 40, maka nilai manfaat menggunakan #TMK1 = 40.
	- Jika tidak memenuhi dua kondisi di atas, maka nilai manfaat ditentukan berdasarkan nilai #TMK1 yang merujuk pada bagian *resign* dalam tabel [[Asumsi#Program-Manfaat]].


## #Qx

Nilai **Qx** merupakan probabilitas seorang karyawan akan mengalami suatu risiko. Terdapat tiga jenis probabilitas berdasarkan kondisi karyawan, yaitu: **Meninggal**, **Cacat**, dan **Resign**, dengan ketentuan umum sebagai berikut:

- Untuk **tahun pertama**, nilai **Qx** diambil dari [[Step 2]] sesuai jenis probabilitas berdasarkan #Usia-Valuasi-bulat-bawah.
- Untuk **tahun kedua dan seterusnya** (hingga maksimal tahun ke-40), nilai **Qx** diambil dari [[Step 3]] sesuai jenis probabilitas berdasarkan #Usia-Valuasi-bulat-bawah. Kolom dalam tabel akan dipilih secara dinamis tergantung pada tahun ke berapa (tahun ke `x`).

---

### Jenis Probabilitas

#### #qₓᵈ – Risiko Meninggal

- Tahun pertama: nilai Qx diambil dari [[Step 2#qₓᵈ-aksen]],berdasarkan #Usia-Valuasi-bulat-bawah 
- Tahun kedua dan seterusnya: nilai Qx diambil dari [[Step 3#Tabel-qₓᵈ]]  berdasarkan #Usia-Valuasi-bulat-bawah , tergantung tahun ke `x`

#### #qₓʷ – Risiko Resign

- Tahun pertama: nilai Qx diambil dari [[Step 2#qₓʷ-aksen]],berdasarkan #Usia-Valuasi-bulat-bawah
- Tahun kedua dan seterusnya: nilai Qx diambil dari [[Step 3#Tabel-qₓʷ]] berdasarkan #Usia-Valuasi-bulat-bawah, tergantung tahun ke `x`

#### #qₓⁱ – Risiko Cacat

- Tahun pertama: nilai Qx diambil dari [[Step 2#qₓⁱ-aksen]],berdasarkan #Usia-Valuasi-bulat-bawah
- Tahun kedua dan seterusnya: nilai Qx diambil dari [[Step 3#Tabel-qₓⁱ]] berdasarkan #Usia-Valuasi-bulat-bawah, tergantung tahun ke `x`

## #Manfaat-Meninggal

Manfaat meninggal merupakan manfaat yang diberikan kepada karyawan atau ahli waris ketika karyawan meninggal dunia. Perhitungan manfaat ini terdiri dari dua komponen utama:

1. **Manfaat dasar berdasarkan Program Manfaat**
2. **Tambahan manfaat berupa uang duka**, jika tercantum dalam Peraturan Perusahaan.

---

### Perhitungan Dasar

Setelah memperoleh nilai **#Manfaat-Pengali-Meninggal**, manfaat dasar dihitung dengan ketentuan berikut:

- Jika #TMK1 = **0**, maka #Manfaat-Meninggal = **0**
- Jika #TMK1> **0***, maka:

> **Manfaat Meninggal =  #Total-Gaji × #Manfaat-Pengali-Meninggal × #sₓ 

---

### Tambahan Uang Duka

Beberapa perusahaan memberikan **tambahan manfaat** berupa uang duka atau bantuan meninggal. Ketentuannya dapat bersifat:

#### 1. Sama Rata (Flat)

##### Contoh A: Nilai Tetap
> Jika tertulis: *"Diberikan bantuan uang duka sebesar Rp100.000"*, maka:

> **Manfaat Meninggal = ( #Total-Gaji × #Manfaat-Pengali-Meninggal × #sₓ ) + Rp100.000**

##### Contoh B: 1 Kali Gaji
> Jika tertulis: *"Diberikan bantuan sebesar 1 kali gaji"*, maka:

> **Manfaat Meninggal = (Program Manfaat pada bagian *death/meninggal*) + 1 × Gaji**

Referensi:  
[[Perhitungan Valuasi Aktuaria/Perhitungan Valuasi Aktuaria/Asumsi#Program-Manfaat]]

---

#### 2. Berjenjang Berdasarkan Masa Kerja

Jika ketentuan uang duka bergantung pada masa kerja, maka digunakan tabel berikut:

| Masa Kerja   | Uang Duka   |
|--------------|-------------|
| 1 – 2 tahun  | Rp1.000.000 |
| 3 – 4 tahun  | Rp1.500.000 |
| 5 – 6 tahun  | Rp2.000.000 |
| 7 – 8 tahun  | Rp2.500.000 |
| ≥10 tahun    | Rp3.000.000 |

Maka perhitungannya menjadi:

> **Manfaat Meninggal = ( #Total-Gaji × #Manfaat-Pengali-Meninggal × #sₓ ) + Uang Duka Sesuai Masa Kerja (berdasarkan #TMK1)**

Jika disebut sebagai "beberapa kali gaji" sesuai masa kerja, maka:

> **Manfaat Meninggal = (Program Manfaat pada bagian *death/meninggal*) + Tambahan Gaji Sesuai Masa Kerja**

- - -
### Referensi Terkait

- [[Perhitungan Valuasi Aktuaria/Perhitungan Valuasi Aktuaria/Asumsi#Program-Manfaat]]


## #Manfaat-Cacat

Manfaat cacat merupakan manfaat yang diberikan kepada karyawan jika mengalami cacat tetap. Perhitungan manfaat ini terdiri dari dua komponen utama:

1. **Manfaat dasar berdasarkan Program Manfaat**
2. **Tambahan manfaat berupa bantuan cacat**, jika tercantum dalam Peraturan Perusahaan.

---

### Perhitungan Dasar

Setelah memperoleh nilai **#Manfaat-Pengali-Cacat**, manfaat dasar dihitung dengan ketentuan berikut:

- Jika #TMK1 = **0**, maka #Manfaat-Cacat = **0**
- Jika #TMK1 > **0**, maka:

> **Manfaat Cacat = #Total-Gaji × #Manfaat-Pengali-Cacat × #sₓ 

---

### Tambahan Bantuan Cacat

Beberapa perusahaan memberikan **tambahan manfaat** berupa bantuan cacat. Ketentuannya dapat bersifat:

#### 1. Sama Rata (Flat)

##### Contoh A: Nilai Tetap
> Jika tertulis: *"Diberikan bantuan cacat sebesar Rp100.000"*, maka:

> **Manfaat Cacat = ( #Total-Gaji × #Manfaat-Pengali-Cacat × #sₓ ) + Rp100.000**

##### Contoh B: 1 Kali Gaji
> Jika tertulis: *"Diberikan bantuan sebesar 1 kali gaji"*, maka:

> **Manfaat Cacat = (Program Manfaat pada bagian *cacat*) + 1 × Gaji**

Referensi:  
[[Perhitungan Valuasi Aktuaria/Perhitungan Valuasi Aktuaria/Asumsi#Program-Manfaat]]

---

#### 2. Berjenjang Berdasarkan Masa Kerja

Jika ketentuan bantuan cacat mengikuti masa kerja karyawan, maka digunakan tabel berikut:

| Masa Kerja   | Bantuan Cacat |
|--------------|----------------|
| 1 – 2 tahun  | Rp1.000.000     |
| 3 – 4 tahun  | Rp1.500.000     |
| 5 – 6 tahun  | Rp2.000.000     |
| 7 – 8 tahun  | Rp2.500.000     |
| ≥10 tahun    | Rp3.000.000     |

Maka perhitungannya menjadi:

> **Manfaat Cacat = ( #Total-Gaji × #Manfaat-Pengali-Cacat × #sₓ ) + Bantuan Cacat Sesuai Masa Kerja (berdasarkan #TMK1)**

Jika bantuan cacat berupa pengali gaji berdasarkan masa kerja, maka:

> **Manfaat Cacat = (Program Manfaat pada bagian *cacat*) + Tambahan Gaji Sesuai Masa Kerja**

- - -
### Referensi Terkait

- [[Perhitungan Valuasi Aktuaria/Perhitungan Valuasi Aktuaria/Asumsi#Program-Manfaat]]


## #Manfaat-Resign

Manfaat resign merupakan manfaat yang diberikan kepada karyawan ketika mengundurkan diri (resign). Perhitungan manfaat ini terdiri dari dua komponen utama:

1. **Manfaat dasar berdasarkan Program Manfaat**
2. **Tambahan manfaat berupa uang pisah**, jika tercantum dalam Peraturan Perusahaan.

---

### Perhitungan Dasar

Setelah memperoleh nilai **#Manfaat-Pengali-Resign**, manfaat dasar dihitung dengan ketentuan berikut:

- Jika #TMK1 = **0**, maka #Manfaat-Resign = **0
- Jika #TMK1 > **0**, maka:

> **Manfaat Resign = #Total-Gaji  × #Manfaat-Pengali-Resign × #sₓ 

---

### Tambahan Uang Pisah

Beberapa perusahaan memberikan tambahan manfaat berupa uang pisah. Ketentuannya dapat bersifat:

#### 1. Sama Rata (Flat)

##### Contoh A: Nilai Tetap
> Jika tertulis: *"Diberikan bantuan uang pisah sebesar Rp100.000"*, maka:

> **Manfaat Resign = ( #Total-Gaji  × #Manfaat-Pengali-Resign × #sₓ ) + Rp100.000**

##### Contoh B: 1 Kali Gaji
> Jika tertulis: *"Diberikan uang pisah sebesar 1 kali gaji"*, maka:

> **Manfaat Resign = (Program Manfaat pada bagian *resign*) + 1 × Gaji**

Referensi:  
[[Perhitungan Valuasi Aktuaria/Perhitungan Valuasi Aktuaria/Asumsi#Program-Manfaat]]

---

#### 2. Berjenjang Berdasarkan Masa Kerja

Jika ketentuan uang pisah mengikuti masa kerja karyawan, maka digunakan tabel berikut:

| Masa Kerja   | Uang Pisah  |
|--------------|-------------|
| 1 – 2 tahun  | Rp1.000.000 |
| 3 – 4 tahun  | Rp1.500.000 |
| 5 – 6 tahun  | Rp2.000.000 |
| 7 – 8 tahun  | Rp2.500.000 |
| ≥10 tahun    | Rp3.000.000 |

Maka perhitungannya menjadi:

> **Manfaat Resign = ( #Total-Gaji × #Manfaat-Pengali-Resign × #sₓ ) + Uang Pisah Sesuai Masa Kerja (berdasarkan #TMK1)**

Jika manfaat berupa pengali gaji berdasarkan masa kerja, maka:

> **Manfaat Resign = (Program Manfaat pada bagian *resign*) + Tambahan Gaji Sesuai Masa Kerja**

- - -
### Referensi Terkait

- [[Perhitungan Valuasi Aktuaria/Perhitungan Valuasi Aktuaria/Asumsi#Program-Manfaat]]


## #Pajak-manfaat-meninggal

Setelah #Manfaat-Meninggal dihitung, selanjutnya Pajak atas manfaat dihitung menggunakan tarif progresif sesuai ketentuan PPh 21

## #Pajak-manfaat-cacat

Setelah #Manfaat-Cacat dihitung, selanjutnya Pajak atas manfaat dihitung menggunakan tarif progresif sesuai ketentuan PPh 21

## #Pajak-manfaat-resign

Setelah #Manfaat-Resign dihitung, selanjutnya Pajak atas manfaat dihitung menggunakan tarif progresif sesuai ketentuan PPh 21

## #PVFB1-Meninggal

**PVFB1-Meninggal** adalah nilai kini dari #Manfaat-Meninggal yang diperkirakan akan dibayarkan kepada ahli waris seorang karyawan apabila karyawan tersebut **meninggal dunia sebelum pensiun**. Nilai ini dihitung dengan mempertimbangkan besar manfaat yang akan diterima, perlakuan pajak, kemungkinan terjadinya kematian, serta nilai waktu dari uang.

Dalam perhitungannya, terdapat dua skenario terkait status pajak:

1. **Jika pajak ditanggung perusahaan**, maka ( #Manfaat-Meninggal + #Pajak-manfaat-meninggal ) x #v1 x #qₓᵈ
    
2. **Jika pajak tidak ditanggung perusahaan**, maka  #Manfaat-Meninggal  x #v1 x #qₓᵈ

## #PVFB1-Cacat

**PVFB1-Cacat** adalah nilai kini dari #Manfaat-Cacat yang diperkirakan akan dibayarkan kepada karyawan apabila karyawan tersebut **cacat tetap sebelum pensiun**. Nilai ini dihitung dengan mempertimbangkan besar manfaat yang akan diterima, perlakuan pajak, kemungkinan terjadinya kematian, serta nilai waktu dari uang.

Dalam perhitungannya, terdapat dua skenario terkait status pajak:

1. **Jika pajak ditanggung perusahaan**, maka ( #Manfaat-Cacat+ #Pajak-manfaat-cacat ) x #v1 x #qₓⁱ 
    
2. **Jika pajak tidak ditanggung perusahaan**, maka  #Manfaat-Cacat  x #v1 x #qₓⁱ 

## #PVFB1-Resign

**PVFB1-Resign** adalah nilai kini dari #Manfaat-Resign yang diperkirakan akan dibayarkan kepada karyawan apabila **karyawan tersebut resign sebelum pensiun**. Nilai ini dihitung dengan mempertimbangkan besar manfaat yang akan diterima, perlakuan pajak, kemungkinan terjadinya kematian, serta nilai waktu dari uang.

Dalam perhitungannya, terdapat dua skenario terkait status pajak:

1. **Jika pajak ditanggung perusahaan**, maka ( #Manfaat-Resign+ #Pajak-manfaat-resign ) x #v1 x #qₓʷ
    
2. **Jika pajak tidak ditanggung perusahaan**, maka  #Manfaat-Resign  x #v1 x #qₓʷ 

## #CSC1-Meninggal

**CSC1-Meninggal** merupakan nilai kewajiban aktuaria rata-rata per individu untuk manfaat meninggal. Nilai ini diperoleh dengan kondisi jika #TMK1 = 0 maka #CSC1-Meninggal = 0 , selain itu  membagi #PVFB1-Meninggal  dengan  #TMK1 .

## #CSC1-Cacat

**CSC1-Cacat** merupakan nilai kewajiban aktuaria rata-rata per individu untuk manfaat cacat. Nilai ini diperoleh dengan kondisi jika #TMK1 = 0 maka #CSC1-Cacat = 0 , selain itu membagi #PVFB1-Cacat dengan  #TMK1 .

## #CSC1-Resign

**CSC1-Resign** merupakan nilai kewajiban aktuaria rata-rata per individu untuk manfaat resign. Nilai ini diperoleh dengan kondisi jika #TMK1 = 0 maka #CSC1-Resign = 0 , selain itu membagi #PVFB1-Resign  dengan  #TMK1 .

## #PVDBO1-Meninggal

**PVDBO1-Meninggal** adalah total kewajiban aktuaria yang harus dicatat oleh perusahaan untuk manfaat meninggal, dengan perhitungan #TMK1 dikali dengan #CSC1-Meninggal

## #PVDBO1-Cacat

**PVDBO1-Cacat** adalah total kewajiban aktuaria yang harus dicatat oleh perusahaan untuk manfaat cacat, dengan perhitungan #TMK1 dikali dengan #CSC1-Cacat 

## #PVDBO1-Resign

**PVDBO1-Resign** adalah total kewajiban aktuaria yang harus dicatat oleh perusahaan untuk manfaat resign, dengan perhitungan #TMK1 dikali dengan #CSC1-Resign 


## #Manfaat2
-  #Manfaat-Pengali-Meninggal2  Manfaat ini didapat ketika meninggal dihitung :
	  - Jika #TMK2= 0, maka nilai manfaat adalah **0**.
	  - Jika #TMK2 ≥ 40, maka nilai manfaat menggunakan #TMK2 = 40.
	  - Jika tidak memenuhi dua kondisi di atas, maka nilai manfaat ditentukan berdasarkan nilai #TMK2 yang merujuk pada bagian *death/meninggal* dalam tabel [[Asumsi#Program-Manfaat]].

- #Manfaat-Pengali-Cacat2  Manfaat ini didapat ketika cacat :
	- Jika #TMK2 = 0, maka nilai manfaat adalah **0**.
	- Jika #TMK2 ≥ 40, maka nilai manfaat menggunakan #TMK1 = 40.
	- Jika tidak memenuhi dua kondisi di atas, maka nilai manfaat ditentukan berdasarkan nilai #TMK2 yang merujuk pada bagian *cacat* dalam tabel [[Asumsi#Program-Manfaat]].

- #Manfaat-Pengali-Resign2 Manfaat ini didapat ketika resign :
	- Jika #TMK2 = 0, maka nilai manfaat adalah **0**.
	- Jika #TMK2 ≥ 40, maka nilai manfaat menggunakan #TMK2 = 40.
	- Jika tidak memenuhi dua kondisi di atas, maka nilai manfaat ditentukan berdasarkan nilai #TMK2 yang merujuk pada bagian *resign* dalam tabel [[Asumsi#Program-Manfaat]].


## #Qx2

Nilai **Qx**2 merupakan probabilitas seorang karyawan akan mengalami suatu risiko. Terdapat tiga jenis probabilitas berdasarkan kondisi karyawan, yaitu: **Meninggal**, **Cacat**, dan **Resign**, dengan ketentuan umum sebagai berikut:

- Untuk **tahun pertama**, nilai **Qx2** diambil dari [[Step 2]] sesuai jenis probabilitas berdasarkan #Usia-Valuasi-bulat-atas.
- Untuk **tahun kedua dan seterusnya** (hingga maksimal tahun ke-40), nilai **Qx2** diambil dari [[Step 3]] sesuai jenis probabilitas berdasarkan #Usia-Valuasi-bulat-atas. Kolom dalam tabel akan dipilih secara dinamis tergantung pada tahun ke berapa (tahun ke `x`).

---

### Jenis Probabilitas

#### #qₓᵈ2 – Risiko Meninggal

- Tahun pertama: nilai Qx2 diambil dari [[Step 2#qₓᵈ-aksen]], berdasarkan #Usia-Valuasi-bulat-atas
- Tahun kedua dan seterusnya: nilai Qx2 diambil dari [[Step 3#Tabel-qₓᵈ]] berdasarkan #Usia-Valuasi-bulat-atas, tergantung tahun ke `x`

#### #qₓʷ2 – Risiko Resign

- Tahun pertama: nilai Qx2 diambil dari [[Step 2#qₓʷ-aksen]], berdasarkan #Usia-Valuasi-bulat-atas
- Tahun kedua dan seterusnya: nilai Qx2 diambil dari [[Step 3#Tabel-qₓʷ]] berdasarkan #Usia-Valuasi-bulat-atas, tergantung tahun ke `x`

#### #qₓⁱ2 – Risiko Cacat

- Tahun pertama: nilai Qx2 diambil dari [[Step 2#qₓⁱ-aksen]], berdasarkan #Usia-Valuasi-bulat-atas
- Tahun kedua dan seterusnya: nilai Qx2 diambil dari [[Step 3#Tabel-qₓⁱ]] berdasarkan #Usia-Valuasi-bulat-atas , tergantung tahun ke `x`


## #PVFB2-Meninggal

**PVFB2-Meninggal** adalah nilai kini dari #Manfaat-Meninggal yang diperkirakan akan dibayarkan kepada ahli waris seorang karyawan apabila karyawan tersebut **meninggal dunia sebelum pensiun**. Nilai ini dihitung dengan mempertimbangkan besar manfaat yang akan diterima, perlakuan pajak, kemungkinan terjadinya kematian, serta nilai waktu dari uang.

Dalam perhitungannya, terdapat dua skenario terkait status pajak:

1. **Jika pajak ditanggung perusahaan**, maka ( #Manfaat-Meninggal + #Pajak-manfaat-meninggal ) x #v1 x #qₓᵈ2 
    
2. **Jika pajak tidak ditanggung perusahaan**, maka  #Manfaat-Meninggal  x #v1 x #qₓᵈ2  

## #PVFB2-Cacat

**PVFB2-Cacat** adalah nilai kini dari #Manfaat-Cacat yang diperkirakan akan dibayarkan kepada karyawan apabila karyawan tersebut **cacat tetap sebelum pensiun**. Nilai ini dihitung dengan mempertimbangkan besar manfaat yang akan diterima, perlakuan pajak, kemungkinan terjadinya kematian, serta nilai waktu dari uang.

Dalam perhitungannya, terdapat dua skenario terkait status pajak:

1. **Jika pajak ditanggung perusahaan**, maka ( #Manfaat-Cacat+ #Pajak-manfaat-cacat ) x #v1 x #qₓⁱ2 
    
2. **Jika pajak tidak ditanggung perusahaan**, maka  #Manfaat-Cacat  x #v1 x #qₓⁱ2 

## #PVFB2-Resign

**PVFB2-Resign** adalah nilai kini dari #Manfaat-Resign yang diperkirakan akan dibayarkan kepada karyawan a**pabila karyawan tersebut resign sebelum pensiun**. Nilai ini dihitung dengan mempertimbangkan besar manfaat yang akan diterima, perlakuan pajak, kemungkinan terjadinya kematian, serta nilai waktu dari uang.

Dalam perhitungannya, terdapat dua skenario terkait status pajak:

1. **Jika pajak ditanggung perusahaan**, maka ( #Manfaat-Resign+ #Pajak-manfaat-resign ) x #v1 x #qₓʷ2 
    
2. **Jika pajak tidak ditanggung perusahaan**, maka  #Manfaat-Resign  x #v1 x #qₓʷ2 


## #CSC2-Meninggal

**CSC2-Meninggal** merupakan nilai kewajiban aktuaria rata-rata per individu untuk manfaat meninggal. Nilai ini diperoleh dengan kondisi jika #TMK2 = 0 maka #CSC2-Meninggal = 0 , selain itu membagi #PVFB2-Meninggal dengan  #TMK2 .

## #CSC2-Cacat

**CSC2-Cacat** merupakan nilai kewajiban aktuaria rata-rata per individu untuk manfaat cacat. Nilai ini diperoleh dengan kondisi jika #TMK2 = 0 maka #CSC2-Cacat = 0 , selain itu membagi #PVFB2-Cacat dengan  #TMK2 .

## #CSC2-Resign

**CSC2-Resign** merupakan nilai kewajiban aktuaria rata-rata per individu untuk manfaat resign. Nilai ini diperoleh dengan kondisi jika #TMK2 = 0 maka #CSC2-Resign= 0 , selain itu membagi #PVFB2-Resign dengan  #TMK2 .

## #PVDBO2-Meninggal

**PVDBO2-Meninggal** adalah total kewajiban aktuaria yang harus dicatat oleh perusahaan untuk manfaat meninggal, dengan perhitungan #TMK2 dikali dengan #CSC2-Meninggal

## #PVDBO2-Cacat

**PVDBO2-Cacat** adalah total kewajiban aktuaria yang harus dicatat oleh perusahaan untuk manfaat cacat, dengan perhitungan #TMK2 dikali dengan #CSC2-Cacat 

## #PVDBO2-Resign

**PVDBO2-Resign** adalah total kewajiban aktuaria yang harus dicatat oleh perusahaan untuk manfaat resign, dengan perhitungan #TMK2 dikali dengan #CSC2-Resign 
