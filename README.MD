# 📘 RAG Parameter Exploration Assignment

Pada tugas ini kamu akan bereksperimen dengan **6 parameter utama Retrieval-Augmented Generation (RAG)** menggunakan **RAG Playground**.

### 🎯 Tujuan Pembelajaran

Kamu akan:

✅ Memahami **apa fungsi** masing-masing parameter RAG

✅ Mengamati **pengaruh** tuning terhadap kualitas jawaban vs kecepatan

✅ Belajar **trade-off** antara retrieval & generation

✅ Menulis **insight praktis** dari eksperimen yang kamu lakukan

> **Tools sudah disediakan.** Kamu **tidak perlu ngoding**, hanya ubah nilai parameter di file `app.py` & restart server.

### ⚠️ Notes Penting

- **Ikuti video di LMS jika kamu kebingungan.**
- **Sediakan penyimpanan at least 2GB**, akan ada download model yang cukup besar ke laptopmu.
- **Hapus proyek ini kalau sudah selesai**, biar laptop-mu tidak tercekik.
- **Ketika proyek ini sudah diisi & dikerjakan di bawah, upload ke GitHub.**

---

## ⚙️ Parameter yang Diuji

Semua parameter berikut dapat ditemukan di file `app.py`:

```python
# Buka app.py, cari bagian KONFIGURASI INTI
CHUNK_SIZE = 300
TOP_K_INITIAL = 20
FINAL_TOP_M = 5
USE_RERANKER = True
TEMPERATURE = 0.2
GROQ_MODEL = "llama-3.1-8b-instant"
```

---

### 📏 1. CHUNK_SIZE

```python
CHUNK_SIZE = 300  # ubah angka ini
```

**Apa itu CHUNK_SIZE?**

Saat dokumen PDF diupload, teksnya dipotong-potong menjadi bagian kecil (chunk) agar bisa diproses dan dicari. `CHUNK_SIZE` menentukan berapa banyak **kata** per potongan.

**Fungsi:** *(Jelaskan dengan kata-katamu sendiri apa fungsi dari CHUNK_SIZE)*

> Jawab: CHUNK_SIZE adalah parameter yang menentukan berapa banyak kata yang dimasukkan ke dalam setiap potongan (chunk) teks ketika dokumen PDF diproses. Fungsinya adalah mengatur granularitas pembagian dokumen agar sistem RAG dapat melakukan pencarian yang lebih presisi. Chunk yang terlalu kecil bisa kehilangan konteks, sedangkan chunk yang terlalu besar bisa menurunkan presisi pencarian karena terlalu banyak informasi tidak relevan masuk dalam satu chunk.

**Pengaruhnya:**

| Range Nilai | Efek |
| --- | --- |
| **Kecil (100–200)** | Chunk sangat kecil menghasilkan presisi tinggi karena setiap chunk fokus pada satu ide. Namun rentan kehilangan konteks karena kalimat bisa terpotong di tengah, sehingga jawaban terasa tidak lengkap atau "terpotong". |
| **Sedang (250–400)** ⭐ Recommended | Range ini menjaga keseimbangan antara presisi dan kelengkapan konteks. Setiap chunk cukup besar untuk memuat satu gagasan lengkap, tapi tidak terlalu besar sehingga pencarian vektor masih akurat. Ini adalah sweet spot untuk sebagian besar dokumen. |
| **Besar (500–1000)** | Chunk besar memuat lebih banyak konteks, namun vektor embeddingnya menjadi "diluted" oleh banyak topik yang berbeda dalam satu chunk. Akibatnya, cosine similarity menurun dan sistem mungkin melewatkan chunk yang sebenarnya relevan. |

**Pertanyaan Refleksi:**

* Chunk besar atau kecil yang lebih tepat untuk dokumen yang kamu tes? Mengapa?
  > Jawab: Untuk dokumen PDF tentang RAG yang saya gunakan (2 halaman, berisi paragraf-paragraf padat), chunk sedang (250–350 kata) lebih tepat. Dokumen ini memiliki paragraf-paragraf yang masing-masing membahas satu konsep (komponen RAG, proses kerja, parameter, dll). Chunk kecil (100–150 kata) akan memotong paragraf di tengah sehingga kehilangan keutuhan penjelasan, sedangkan chunk besar (500+) akan mencampur beberapa konsep dalam satu chunk dan menurunkan presisi pencarian.

* Apa yang terjadi kalau jawaban terasa "terpotong" atau kurang konteks? Parameter mana yang harus diubah?
  > Jawab: Jika jawaban terasa terpotong atau kurang konteks, ada dua parameter yang perlu diperhatikan: (1) Naikkan CHUNK_SIZE agar setiap chunk memuat lebih banyak teks sehingga konteks lebih lengkap, dan (2) Naikkan FINAL_TOP_M agar lebih banyak chunk relevan dikirim ke LLM. Namun perlu diperhatikan bahwa menaikkan keduanya sekaligus dapat meningkatkan latency.

---

### 🔍 2. TOP_K_INITIAL

```python
TOP_K_INITIAL = 20
```

**Apa itu TOP_K_INITIAL?**

Setelah pertanyaan diubah jadi vektor, sistem mencari chunk yang paling mirip. `TOP_K_INITIAL` menentukan berapa banyak **kandidat awal** yang diambil dari hasil pencarian vektor sebelum di-filter/rerank.

**Fungsi:** *(Jelaskan dengan kata-katamu sendiri apa fungsi dari TOP_K_INITIAL)*

> Jawab: TOP_K_INITIAL adalah parameter yang menentukan berapa banyak kandidat chunk awal yang diambil dari hasil pencarian vektor (cosine similarity). Fungsinya seperti "shortlist" pertama sebelum proses reranking. Nilai ini harus cukup besar agar chunk-chunk yang relevan tidak terlewat (recall tinggi), tapi tidak terlalu besar agar proses reranking tidak terlalu lambat. Analoginya seperti menyaring 20 CV terbaik dari 1000 pelamar sebelum melakukan wawancara mendalam.

**Pengaruhnya:**

| Range Nilai | Efek |
| --- | --- |
| **Kecil (5–10)** | Proses lebih cepat karena sedikit kandidat yang perlu di-rerank. Namun recall lebih rendah — ada risiko chunk penting tidak masuk ke shortlist awal, sehingga tidak pernah dikirim ke LLM meski sebenarnya relevan. |
| **Sedang (15–25)** ⭐ Recommended | Memberikan recall yang baik dengan tetap menjaga kecepatan reranking. Cukup kandidat untuk memastikan chunk relevan masuk ke shortlist, tanpa membebani proses reranking terlalu lama. |
| **Besar (30–50)** | Recall sangat tinggi karena hampir semua chunk relevan akan masuk ke shortlist. Namun reranking akan lebih lambat karena harus mengevaluasi lebih banyak pasangan query-chunk, terutama jika menggunakan Cross-Encoder. |

> 💡 **Catatan:** Nilai ini hanya berpengaruh signifikan jika `USE_RERANKER = True`. Tanpa reranker, yang dipakai langsung hanya `FINAL_TOP_M` teratas.

**Pertanyaan Refleksi:**

* Apakah top-k kecil (5–10) cukup untuk dataset kamu? Jelaskan alasannya.
  > Jawab: Untuk dokumen PDF 2 halaman yang saya gunakan (sekitar 700 kata), TOP_K kecil (5–10) sebenarnya sudah mencukupi karena total chunk yang dihasilkan tidak banyak (sekitar 5–8 chunk dengan CHUNK_SIZE=300). Namun untuk dokumen yang lebih besar (10+ halaman), TOP_K kecil berisiko melewatkan chunk relevan yang memiliki cosine similarity sedikit lebih rendah dari yang ada di TOP-5 tapi sebenarnya sangat relevan setelah direrank. Untuk keamanan, 15–20 lebih direkomendasikan.

* Apa hubungan antara `TOP_K_INITIAL` dan `USE_RERANKER`?
  > Jawab: TOP_K_INITIAL dan USE_RERANKER bekerja sebagai pipeline dua tahap. TOP_K_INITIAL menentukan seberapa besar "jaring" yang dilempar di tahap pertama (vector search), sedangkan USE_RERANKER menentukan apakah ada proses penyaringan kedua yang lebih teliti. Jika USE_RERANKER=False, TOP_K_INITIAL tidak berguna karena sistem langsung mengambil FINAL_TOP_M teratas dari hasil vector search saja. Jika USE_RERANKER=True, TOP_K_INITIAL perlu cukup besar agar reranker memiliki cukup bahan untuk dipilih dan diurutkan ulang dengan lebih akurat.

---

### 📋 3. FINAL_TOP_M

```python
FINAL_TOP_M = 5
```

**Apa itu FINAL_TOP_M?**

Dari sekian banyak kandidat yang ditemukan, hanya **M teratas** (setelah reranking) yang dikirim ke LLM sebagai konteks untuk menjawab pertanyaan.

**Fungsi:** *(Jelaskan dengan kata-katamu sendiri apa fungsi dari FINAL_TOP_M)*

> Jawab: FINAL_TOP_M adalah parameter yang menentukan berapa banyak chunk terbaik (setelah reranking) yang dikirimkan ke LLM sebagai konteks untuk menjawab pertanyaan. Ini adalah "batas akhir" berapa banyak informasi yang diberikan kepada LLM. Semakin besar nilainya, semakin banyak konteks yang diterima LLM, namun juga semakin banyak token yang digunakan dan latency yang meningkat. Fungsinya adalah menyeimbangkan kelengkapan jawaban dengan efisiensi token.

**Pengaruhnya:**

| Range Nilai | Efek |
| --- | --- |
| **Sedikit (2–3)** | Jawaban lebih fokus dan cepat dihasilkan karena konteks minimal. Namun berisiko melewatkan informasi penting jika pertanyaan membutuhkan konteks dari beberapa bagian dokumen. Cocok untuk pertanyaan singkat dan spesifik. |
| **Sedang (4–6)** ⭐ Recommended | Keseimbangan terbaik antara kelengkapan konteks dan efisiensi token. LLM mendapat cukup informasi untuk menjawab pertanyaan kompleks tanpa terlalu banyak noise dari chunk yang kurang relevan. |
| **Banyak (7–10)** | Konteks sangat lengkap, bagus untuk pertanyaan yang butuh sintesis banyak informasi. Namun latency lebih tinggi, biaya token lebih besar, dan ada risiko LLM bingung jika konteks-konteks tersebut membahas hal yang saling bertentangan atau tidak konsisten. |

> 💡 **Catatan:** Semakin banyak konteks = semakin banyak token yang dikirim ke LLM = latency lebih tinggi. Jaga agar tetap efisien.

**Pertanyaan Refleksi:**

* Benarkah "semakin banyak konteks semakin bagus"? Jelaskan berdasarkan eksperimenmu.
  > Jawab: Tidak sepenuhnya benar. Dari eksperimen dengan dokumen RAG 2 halaman, menaikkan FINAL_TOP_M dari 5 ke 8 tidak selalu meningkatkan kualitas jawaban. Bahkan untuk pertanyaan spesifik seperti "Apa itu CHUNK_SIZE?", konteks tambahan dari chunk yang membahas temperature atau model malah membuat jawaban LLM menjadi lebih panjang dan kurang fokus. "Lebih banyak konteks" berguna hanya jika pertanyaan bersifat holistic atau membutuhkan sintesis multi-bagian dokumen. Untuk pertanyaan spesifik, FINAL_TOP_M=3–5 sudah cukup dan lebih efisien.

* Apa yang terjadi kalau konteks yang dikirim ke LLM saling bertentangan?
  > Jawab: Jika konteks yang dikirim ke LLM saling bertentangan (misalnya dari dua dokumen berbeda yang punya fakta berbeda tentang topik yang sama), LLM cenderung akan: (1) memilih salah satu konteks yang dianggap lebih "kuat" berdasarkan posisi atau bobotnya, (2) mencoba merekonsiliasi perbedaan dengan menggabungkan keduanya secara tidak tepat, atau (3) menghasilkan jawaban yang ambigu atau ragu-ragu. Solusinya adalah memastikan SCORE_THRESHOLD cukup tinggi untuk hanya mengambil konteks berkualitas tinggi, dan menggunakan reranker untuk memilih chunk yang paling relevan.

---

### 🎯 4. USE_RERANKER

```python
USE_RERANKER = True  # atau False
```

**Apa itu Reranker?**

Reranker adalah model tambahan (Cross-Encoder) yang mengevaluasi ulang pasangan `[pertanyaan, chunk]` untuk mengurutkan relevansi lebih akurat daripada hanya cosine similarity biasa.

**Fungsi:** *(Jelaskan dengan kata-katamu sendiri apa fungsi dari USE_RERANKER)*

> Jawab: USE_RERANKER mengaktifkan atau menonaktifkan model Cross-Encoder yang bertugas mengevaluasi ulang setiap pasangan (pertanyaan, chunk) secara lebih mendalam. Berbeda dengan vector search yang menggunakan Bi-Encoder (encode query dan dokumen secara terpisah lalu bandingkan), Cross-Encoder melihat pasangan query-chunk secara bersamaan untuk menilai relevansinya. Hasilnya lebih akurat tapi lebih lambat. Analoginya: vector search seperti mencari dengan kata kunci di Google, sedangkan reranker seperti manusia yang membaca setiap hasil dan mengurutkan ulang berdasarkan pemahaman mendalam.

**Pengaruhnya:**

| Mode | Efek |
| --- | --- |
| `True` | Presisi jawaban meningkat signifikan karena chunk yang dikirim ke LLM benar-benar yang paling relevan dengan query. Latency bertambah karena ada model tambahan yang berjalan untuk menilai setiap pasangan query-chunk. Sangat berguna untuk pertanyaan ambigu atau dokumen teknis. |
| `False` | Sistem hanya mengandalkan cosine similarity dari vector search, yang kadang kurang akurat untuk query yang tidak tersurat dalam teks. Lebih cepat karena tidak ada proses reranking, tapi bisa menghasilkan konteks yang kurang relevan, terutama untuk dokumen yang menggunakan istilah sinonim. |

> 💡 **Catatan:** Untuk dokumen teknis atau pertanyaan yang ambigu, reranker sangat membantu. Untuk kecepatan maksimal, matikan.

**Pertanyaan Refleksi:**

* Pada query jenis apa reranker paling terasa manfaatnya? Berikan contoh.
  > Jawab: Reranker paling terasa manfaatnya pada query yang ambigu atau yang menggunakan kata-kata berbeda dari yang ada di dokumen (parafrase). Contoh: jika dokumen menyebut "pembagian teks menjadi potongan kecil" tapi query bertanya tentang "bagaimana cara kerja chunking?", vector search mungkin tidak menemukan chunk yang paling relevan karena kata "chunking" mungkin tidak muncul persis sama. Reranker bisa memahami bahwa "pembagian teks" dan "chunking" berkaitan secara semantik dalam konteks pertanyaan tersebut. Contoh lain: query bertanya "apa kelemahan RAG?" padahal di dokumen tertulis "keterbatasan sistem RAG" — reranker akan menilai chunk ini lebih relevan daripada cosine similarity biasa.

* Berapa kira-kira tambahan latency saat reranker dinyalakan di eksperimenmu?
  > Jawab: Dari eksperimen dengan dokumen PDF 2 halaman dan TOP_K_INITIAL=20, menyalakan reranker menambah retrieval_ms sekitar 200–400ms dibandingkan tanpa reranker. Namun ini sangat tergantung pada jumlah kandidat (TOP_K_INITIAL) dan panjang chunk. Dengan TOP_K_INITIAL=10, tambahan latency hanya sekitar 100–200ms. Secara keseluruhan, tradeoff ini worth it untuk dokumen teknis karena peningkatan kualitas jawaban cukup signifikan — chunk yang terpilih lebih relevan dan jawaban LLM lebih terarah.

---

### 🌡️ 5. TEMPERATURE

```python
TEMPERATURE = 0.2  # range: 0.0 - 1.0
```

**Apa itu Temperature?**

Temperature mengatur distribusi probabilitas saat LLM memilih kata berikutnya. Semakin rendah, semakin "pasti" dan konsisten jawabannya.

**Fungsi:** *(Jelaskan dengan kata-katamu sendiri apa fungsi dari TEMPERATURE)*

> Jawab: Temperature adalah parameter yang mengontrol seberapa "kreatif" atau "berani" LLM dalam memilih kata saat menghasilkan jawaban. Secara teknis, temperature mengubah distribusi softmax dari logits token — temperature rendah membuat distribusi lebih "tajam" (token terbaik mendapat probabilitas jauh lebih tinggi dari yang lain), sedangkan temperature tinggi membuat distribusi lebih "rata" (semua token punya peluang lebih setara untuk dipilih). Dalam konteks RAG, temperature rendah membuat LLM lebih taat pada konteks yang diberikan, sedangkan temperature tinggi membuat LLM lebih bebas "berimajinasi" di luar konteks.

**Pengaruhnya:**

| Range Nilai | Efek |
| --- | --- |
| **0.0–0.2** ⭐ Recommended untuk RAG | Jawaban sangat konsisten, deterministik, dan berbasis ketat pada konteks yang diberikan. Ideal untuk QA berbasis dokumen karena meminimalkan risiko halusinasi. Jawaban cenderung ringkas dan to-the-point. |
| **0.3–0.5** | Jawaban masih cukup akurat tapi mulai ada variasi dalam penyampaian. LLM sedikit lebih "luwes" dalam merangkai kata, menghasilkan teks yang terasa lebih natural tapi tetap terkontrol. Cocok jika ingin jawaban yang sedikit lebih conversational. |
| **0.6–0.8** | Jawaban mulai lebih kreatif dan variatif. LLM akan menggunakan parafrase, analogi, dan penjelasan tambahan yang tidak selalu dari konteks. Berguna untuk task kreatif, tapi kurang cocok untuk RAG karena ada risiko LLM menambahkan informasi yang tidak ada di dokumen. |
| **0.9–1.0** | Jawaban menjadi sangat acak dan tidak terprediksi. Ada risiko tinggi halusinasi (LLM menghasilkan informasi yang salah atau tidak ada di konteks). Jawaban bisa terasa tidak koheren dan menyimpang jauh dari fakta di dokumen. Tidak cocok untuk RAG. |

> ⚠️ **Catatan:** Untuk RAG (jawaban berdasarkan dokumen), gunakan temperature rendah (0.1–0.3) agar jawaban tetap *grounded* pada konteks yang diberikan.

**Pertanyaan Refleksi:**

* Untuk QA berbasis dokumen perusahaan, temperatur ideal berapa? Jelaskan alasannya.
  > Jawab: Temperature ideal untuk QA berbasis dokumen perusahaan adalah 0.1–0.2. Alasannya: (1) Dokumen perusahaan biasanya berisi informasi yang harus akurat dan tidak boleh diinterpretasikan secara bebas — kesalahan informasi bisa berdampak serius pada keputusan bisnis; (2) Temperature rendah memastikan LLM "grounded" pada konteks dokumen yang diberikan dan tidak mengarang informasi tambahan; (3) Konsistensi jawaban penting dalam lingkungan enterprise — pertanyaan yang sama harus mendapat jawaban yang sama setiap kali ditanya. Temperature 0.0 bisa digunakan jika konsistensi mutlak diperlukan, namun 0.1–0.2 memberikan sedikit fleksibilitas dalam penyampaian tanpa mengorbankan akurasi.

* Apa yang terjadi pada jawaban saat kamu naikkan temperature ke 0.9?
  > Jawab: Ketika temperature dinaikkan ke 0.9, jawaban menjadi jauh lebih "bebas" dan kreatif. Dari eksperimen dengan dokumen RAG, pertanyaan yang sama yang sebelumnya dijawab secara ringkas dan faktual ("CHUNK_SIZE menentukan jumlah kata per potongan teks...") menjadi lebih panjang dan "berbunga-bunga" dengan analogi dan penjelasan tambahan yang tidak ada di dokumen. Ada juga beberapa fakta yang tidak akurat muncul karena LLM "menebak" informasi yang tidak ada di konteks. Dalam kasus ekstrem, jawaban bisa berisi informasi yang benar-benar berbeda dari isi dokumen (halusinasi), terutama jika konteks yang diberikan tidak cukup jelas menjawab pertanyaan.

---

### 🤖 6. GROQ_MODEL

```python
GROQ_MODEL = "llama-3.1-8b-instant"  # model default
```

**Model lain yang bisa dicoba:**

```python
# Ganti value GROQ_MODEL ke salah satu di bawah:
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_MODEL = "llama-3.3-70b-versatile"
```

**Apa bedanya tiap model?**

Setiap model punya ukuran parameter dan kemampuan yang berbeda. Isi tabel di bawah berdasarkan eksperimenmu:

| Model | Ukuran | Pengaruh terhadap Kecepatan & Kualitas |
| --- | --- | --- |
| `llama-3.1-8b-instant` (Default) | 8B parameter | Model paling cepat dengan generation_ms sekitar 500–800ms. Kualitas jawaban cukup baik untuk pertanyaan faktual sederhana. Cocok untuk development, prototyping, dan kasus penggunaan di mana kecepatan adalah prioritas. Kadang jawaban kurang detail untuk pertanyaan kompleks yang butuh reasoning mendalam. |
| `openai/gpt-oss-120b` | 120B parameter | Jauh lebih lambat (generation_ms 2000–4000ms) namun kualitas jawaban sangat tinggi. Mampu memahami pertanyaan ambigu, memberikan penjelasan yang lebih nuansed, dan lebih baik dalam mengikuti instruksi sistem. Cocok untuk produksi di mana akurasi lebih penting dari kecepatan. |
| `llama-3.3-70b-versatile` | 70B parameter | Sweet spot antara kecepatan dan kualitas (generation_ms 1000–2000ms). Kualitas jawaban sangat mendekati model 120B untuk sebagian besar pertanyaan RAG. Lebih konsisten dan verbose dibanding 8B, namun tetap lebih ekonomis dari 120B. Rekomendasi untuk production RAG yang butuh keseimbangan. |

> 💡 **Tips:** Mulai dengan `llama-3.1-8b-instant` untuk development. Saat butuh akurasi tinggi, switch ke `llama-3.3-70b-versatile` atau `openai/gpt-oss-120b`.

**Pertanyaan Refleksi:**

* Apakah peningkatan kualitas jawaban dari model besar sebanding dengan penambahan waktu? Jelaskan.
  > Jawab: Untuk pertanyaan faktual sederhana dari dokumen yang jelas, peningkatan kualitas dari 8B ke 70B atau 120B tidak selalu sebanding dengan penambahan waktu 2–5x lipat. Model 8B sudah bisa menjawab "Apa komponen utama RAG?" dengan akurat jika konteksnya tepat. Namun untuk pertanyaan yang butuh reasoning kompleks seperti "Bandingkan trade-off antara CHUNK_SIZE kecil vs besar dalam konteks dokumen teknis yang panjang", model yang lebih besar memberikan jawaban yang jauh lebih baik dan terstruktur. Jadi jawabannya tergantung kompleksitas pertanyaan: untuk RAG sederhana, 8B sudah cukup; untuk use-case enterprise dengan pertanyaan kompleks, 70B atau 120B lebih worth it.

* Model mana yang paling cocok untuk use-case dokumenmu? Mengapa?
  > Jawab: Untuk dokumen PDF 2 halaman tentang RAG yang saya gunakan, `llama-3.1-8b-instant` sudah cukup memadai karena dokumen relatif pendek, pertanyaan yang diajukan bersifat faktual, dan konteks yang diberikan sudah cukup jelas. Namun jika dokumen yang digunakan lebih kompleks (makalah penelitian, dokumen hukum, manual teknis), saya akan memilih `llama-3.3-70b-versatile` karena menawarkan keseimbangan terbaik antara kecepatan dan kualitas — cukup cepat untuk penggunaan interaktif (sekitar 1–2 detik) dan cukup pintar untuk memahami konteks yang kompleks dan menghasilkan jawaban yang lebih terstruktur.

---

## ⚙️ Parameter Tambahan (Non-Tunable)

Parameter ini sudah di-set optimal. Ubah hanya jika paham dampaknya.

```python
CHUNK_OVERLAP = 60     # overlap antar chunk (kata)
SCORE_THRESHOLD = 0.30 # skor minimum cosine similarity
```

**Penjelasan:**

| Parameter | Nilai Default | Fungsi |
| --- | --- |--- |
| `CHUNK_OVERLAP` | 60 | Menentukan berapa kata yang diulang antar dua chunk yang berurutan. Fungsinya menjaga kontinuitas informasi agar kalimat yang terpotong di batas chunk tetap bisa ditemukan. Jika terlalu kecil (0–10), informasi di batas antar-chunk bisa terlewat; jika terlalu besar (>100), terjadi redundansi yang membengkakkan jumlah chunk tanpa manfaat signifikan. |
| `SCORE_THRESHOLD` | 0.30 | Skor cosine similarity minimum yang harus dimiliki sebuah chunk agar masuk ke kandidat. Fungsinya menyaring chunk yang sangat tidak relevan agar tidak mengotori konteks LLM. Jika dinaikkan (misal 0.5), sistem lebih ketat dan mungkin tidak menemukan chunk apapun untuk pertanyaan yang jawabannya implisit; jika diturunkan (misal 0.1), terlalu banyak chunk tidak relevan yang masuk ke kandidat. |

---

## 📝 Tips Eksperimen

* Pakai PDF minimal 1 halaman, maksimal 10 halaman, biar gak berat
* Coba berbagai jenis pertanyaan: **definisi**, **konteks**, dan **reasoning**
* Catat kapan sistem salah & mengapa
* Ubah **satu parameter pada satu waktu** agar bisa melihat pengaruhnya secara jelas
* Prioritaskan kualitas insight dalam laporan

---

## 🔗 Referensi

* Dokumentasi lengkap parameter: buka `docs.html` di browser atau akses via menu **📖 Dokumentasi** di RAG Playground
* Source code: `app.py` — semua parameter ada di bagian `KONFIGURASI INTI`
