## #Usia-Mulai-Masa-Kerja  

Usia Mulai Masa Kerja = selisih antara `Tanggal masuk kerja` dan `Tanggal lahir` dibagi 365,25 dan dibulatkan 2 desimal.

## #Usia-Saat-Valuasi 

Usia Saat Valuasi = selisih antara `Tanggal valuasi` dan `Tanggal lahir` dibagi 365,25 dan dibulatkan 2 desimal.

## #Total-Gaji 

Total Gaji= Total `Gaji pokok` dan `Tunjangan tetap`

## #Masa-Kerja-Lalu-Kontrak 

Masa kerja lalu kontrak =  Selisih `Tanggal valuasi` dan  `Tanggal awal kontrak` dibagi 365,25 , lalu dikalikan 12

## #Future-Service-Kontrak

Future service kontrak = Selisih `Tanggal akhir kontrak` dan `Tanggal valuasi` dibagi 365,25
## #Total-Masa-Kontrak

Tahun masa kontrak = Selisih `Tanggal akhir kontrak` dan `Tanggal awal kontrak` lalu dibagi 365,25 , lalu dibulatkan 2 desimal.

## #Sisa-Masa-Kontrak

Sisa masa kontrak = 12 dikalikan #Total-Masa-Kontrak , lalu dikurangkan dengan #Masa-Kerja-Lalu-Kontrak 

## #Kompensasi

Kompensasi = Nilai dari #Total-Gaji dikalikan dengan #Tahun-Masa-Kontrak 

## #Batas-Diskonto-Bawah

Batas Diskonto Bawah = Pembulatan ke bawah untuk nilai #Future-Service-Kontrak 

## #Batas-Diskonto-Atas

Batas Diskonto Atas = Pembulatan ke atas untuk nilai #Future-Service-Kontrak 

## #Interpolasi-Bunga-Diskonto  

Interpolasi Bunga Diskonto = Selisih #Future-Service-Kontrak dengan #Batas-Diskonto-Bawah  

## #Sisa-Interpolasi-Bunga

Sisa Interpolasi Bunga = Selisih #Batas-Diskonto-Atas  dengan #Interpolasi-Bunga-Diskonto 

## #Faktor-CSC

**Rumus:**

Faktor Pengali CSC = Selisih #Masa-Kerja-Lalu-Kontrak dengan nilai (12 dikali #Batas-Bunga-Bawah)

- Jika hasilnya = 0, maka **Faktor = 12**
- Jika hasilnya ≠ 0, gunakan **hasil perhitungan** tersebut.

## #Tingkat-Diskonto 

[[Asumsi#Tingkat-Diskonto]] yaitu tingkat diskonto tahunan yang akan digunakan untuk menghitung nilai sekarang dari manfaat kerja (PVDBO).

## #Sx-Kontrak

**Sx-Kontrak** adalah faktor kenaikan gaji selama masa kontrak kerja, bersifat terbatas (misalnya 1 hingga 2 tahun).

Sx kontrak = ((1 + #Tingkat-Diskonto ) ^ (1 / 12)) - 1

## #PV 

**PV (Present Value)** adalah nilai kini dari sejumlah uang atau manfaat yang akan diterima di di sisa masa kontrak, yang dihitung dengan mendiskontokan nilai tersebut menggunakan tingkat diskonto yang berlaku.

**Rumus**:

PV = (1 + #Sx-Kontrak ) ^ (–[[#Sisa-Masa-Kontrak]])

## #PVDBO

**PVDBO (Present Value Defined Benefit Obligation)** adalah nilai kini dari total manfaat selama masa kontrak yang dihitung dengan mendiskontokan nilai tersebut menggunakan tingkat diskonto yang berlaku.

**Rumus**:

PVDBO = #PV * #Kompensasi 

## #CSC

**CSC (Current Service Cost)** adalah kontribusi biaya jasa berjalan selama masa kontrak.

**Rumus:**

CSC = #PVDBO / #Total-Masa-Kontrak 

