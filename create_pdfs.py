"""Buat 2 PDF tambahan untuk eksperimen RAG dengan topik berbeda."""


def make_pdf(filename: str, content: str) -> None:
    def make_stream(text):
        ops = "BT\n/F1 10 Tf\n40 800 Td\n13 TL\n"
        max_chars = 90
        for para in text.strip().split("\n\n"):
            words = para.split()
            line = ""
            for word in words:
                if len(line) + len(word) + 1 <= max_chars:
                    line += (" " if line else "") + word
                else:
                    esc = line.replace("\\","\\\\").replace("(","\\(").replace(")","\\)")
                    ops += f"({esc}) Tj T*\n"
                    line = word
            if line:
                esc = line.replace("\\","\\\\").replace("(","\\(").replace(")","\\)")
                ops += f"({esc}) Tj T*\n"
            ops += "() Tj T*\n"
        ops += "ET\n"
        return ops.encode("latin-1", errors="replace")

    mid = len(content) // 2
    cut = content.rfind(". ", 0, mid) + 2
    pages = [content[:cut], content[cut:]]
    streams = [make_stream(p) for p in pages]

    objs = []
    objs.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objs.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>\nendobj\n")
    objs.append(b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R /Resources << /Font << /F1 7 0 R >> >> >>\nendobj\n")
    objs.append(f"4 0 obj\n<< /Length {len(streams[0])} >>\nstream\n".encode() + streams[0] + b"\nendstream\nendobj\n")
    objs.append(b"5 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 6 0 R /Resources << /Font << /F1 7 0 R >> >> >>\nendobj\n")
    objs.append(f"6 0 obj\n<< /Length {len(streams[1])} >>\nstream\n".encode() + streams[1] + b"\nendstream\nendobj\n")
    objs.append(b"7 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")

    pdf = b"%PDF-1.4\n"
    offsets = []
    for obj in objs:
        offsets.append(len(pdf))
        pdf += obj

    xref = len(pdf)
    pdf += b"xref\n" + f"0 {len(objs)+1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        pdf += f"{off:010d} 00000 n \n".encode()
    pdf += f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()

    with open(filename, "wb") as f:
        f.write(pdf)
    print(f"Dibuat: {filename}")


PYTHON_CONTENT = """Python adalah bahasa pemrograman tingkat tinggi yang diciptakan oleh Guido van Rossum dan pertama kali dirilis pada tahun 1991. Python dikenal karena sintaksnya yang bersih dan mudah dibaca, sehingga sangat populer untuk pemula maupun profesional. Python mendukung paradigma pemrograman prosedural, berorientasi objek, dan fungsional.

Tipe data dasar dalam Python meliputi integer (bilangan bulat), float (bilangan desimal), string (teks), boolean (True/False), list (daftar yang bisa diubah), tuple (daftar yang tidak bisa diubah), dictionary (pasangan kunci-nilai), dan set (kumpulan nilai unik). Python menggunakan indentasi sebagai penanda blok kode, bukan kurung kurawal seperti C atau Java.

Fungsi dalam Python didefinisikan menggunakan kata kunci def. Python mendukung fungsi lambda (fungsi anonim satu baris), fungsi rekursif, dan fungsi dengan argumen default. Python juga mendukung list comprehension, yaitu cara ringkas untuk membuat list baru dari iterable yang sudah ada dengan kondisi tertentu.

Library populer Python meliputi NumPy untuk komputasi numerik, Pandas untuk analisis data, Matplotlib dan Seaborn untuk visualisasi, Scikit-learn untuk machine learning, TensorFlow dan PyTorch untuk deep learning, FastAPI dan Flask untuk web API, serta Requests untuk HTTP client. Python Package Index (PyPI) menyediakan lebih dari 400.000 paket yang bisa diinstall menggunakan pip.

Virtual environment dalam Python digunakan untuk mengisolasi dependensi proyek agar tidak konflik antar proyek. Tools yang umum digunakan adalah venv (bawaan Python), virtualenv, conda, dan uv. Setiap proyek sebaiknya memiliki virtual environment sendiri dengan file requirements.txt untuk mencatat dependensinya.

Error handling dalam Python menggunakan blok try-except-finally. Python memiliki hierarki exception yang luas, mulai dari BaseException di paling atas, hingga exception spesifik seperti ValueError, TypeError, FileNotFoundError, dan KeyError. Custom exception bisa dibuat dengan mewarisi kelas Exception.

Python memiliki fitur OOP (Object-Oriented Programming) yang lengkap, termasuk kelas, inheritance (pewarisan), encapsulation, dan polymorphism. Dekorator adalah fitur Python yang memungkinkan modifikasi perilaku fungsi atau kelas tanpa mengubah kode aslinya. Generator dan iterator memungkinkan pemrosesan data secara lazy (satu per satu) yang hemat memori.
"""

DATASCIENCE_CONTENT = """Data Science adalah bidang interdisipliner yang menggabungkan statistik, matematika, pemrograman, dan domain knowledge untuk mengekstrak wawasan dan pengetahuan dari data. Data Scientist bertugas mengumpulkan, membersihkan, menganalisis, dan memvisualisasikan data untuk mendukung pengambilan keputusan bisnis.

Proses Data Science mengikuti siklus yang disebut CRISP-DM (Cross-Industry Standard Process for Data Mining): Business Understanding (memahami tujuan bisnis), Data Understanding (eksplorasi data awal), Data Preparation (pembersihan dan transformasi data), Modeling (membangun model), Evaluation (mengevaluasi performa model), dan Deployment (penerapan model ke produksi).

Eksploratory Data Analysis (EDA) adalah langkah awal dalam analisis data untuk memahami struktur, distribusi, dan hubungan antar variabel. EDA meliputi perhitungan statistik deskriptif (mean, median, modus, standar deviasi), visualisasi distribusi dengan histogram dan boxplot, analisis korelasi dengan heatmap, dan deteksi outlier dengan IQR atau z-score.

Feature Engineering adalah proses membuat fitur baru atau mentransformasi fitur yang ada untuk meningkatkan performa model machine learning. Teknik umum meliputi encoding variabel kategorikal (one-hot encoding, label encoding), normalisasi dan standarisasi data numerik, penanganan missing values (imputasi rata-rata, median, atau KNN imputer), dan pembuatan fitur interaksi.

Model machine learning dibagi menjadi dua kategori utama: supervised learning (pembelajaran terawasi) dan unsupervised learning (pembelajaran tidak terawasi). Supervised learning mencakup regresi (untuk prediksi nilai kontinu) dan klasifikasi (untuk prediksi kategori). Algoritma populer meliputi Linear Regression, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, dan Neural Network.

Evaluasi model menggunakan metrik yang berbeda tergantung jenis masalah. Untuk klasifikasi: Accuracy, Precision, Recall, F1-Score, dan AUC-ROC. Untuk regresi: MSE, RMSE, MAE, dan R-squared. Cross-validation digunakan untuk mendapatkan estimasi performa model yang lebih robust dengan membagi data ke dalam beberapa fold.

Bias-variance tradeoff adalah konsep fundamental dalam machine learning. Model dengan variance tinggi cenderung overfit (terlalu hafal data training). Model dengan bias tinggi cenderung underfit (terlalu simpel). Regularisasi (L1/Lasso dan L2/Ridge) digunakan untuk mengurangi overfitting. Teknik lain termasuk dropout untuk neural network dan pruning untuk decision tree.
"""

if __name__ == "__main__":
    make_pdf("pdf_python.pdf", PYTHON_CONTENT)
    make_pdf("pdf_datascience.pdf", DATASCIENCE_CONTENT)
