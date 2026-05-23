"""Script sederhana untuk membuat PDF test tentang RAG (Retrieval-Augmented Generation)"""


def create_minimal_pdf(filename: str) -> None:
    content = """Retrieval-Augmented Generation (RAG) adalah teknik dalam Natural Language Processing (NLP) yang menggabungkan sistem pencarian informasi (retrieval) dengan model bahasa generatif (generation). RAG bekerja dengan cara mengambil konteks relevan dari basis data dokumen sebelum menghasilkan jawaban, sehingga jawaban yang dihasilkan lebih akurat dan berbasis pada fakta yang ada.

Komponen utama RAG adalah: (1) Retriever, yaitu sistem yang mencari dokumen atau potongan teks yang relevan dengan pertanyaan pengguna; (2) Generator, yaitu model bahasa besar (LLM) yang menggunakan konteks yang diambil untuk menghasilkan jawaban; dan (3) Knowledge Base, yaitu basis data dokumen yang diindeks menggunakan embedding vektor.

Proses kerja RAG dimulai dengan encoding pertanyaan pengguna menjadi vektor embedding. Selanjutnya, sistem mencari chunk-chunk dokumen yang memiliki kemiripan tertinggi dengan vektor pertanyaan menggunakan cosine similarity. Chunk-chunk terpilih kemudian dijadikan konteks untuk LLM, dan LLM menghasilkan jawaban berdasarkan konteks tersebut.

Parameter penting dalam RAG meliputi CHUNK_SIZE yang menentukan ukuran potongan teks, TOP_K_INITIAL yang menentukan berapa banyak kandidat awal yang diambil, FINAL_TOP_M yang menentukan berapa chunk yang dikirim ke LLM, USE_RERANKER yang mengaktifkan model reranking untuk meningkatkan presisi, dan TEMPERATURE yang mengontrol kreativitas jawaban LLM.

Keunggulan RAG dibandingkan fine-tuning model adalah: RAG tidak memerlukan pelatihan ulang model, lebih mudah diperbarui dengan data baru, memberikan transparansi sumber informasi, dan lebih hemat biaya komputasi. RAG sangat berguna untuk aplikasi QA (Question Answering) berbasis dokumen perusahaan, chatbot layanan pelanggan, dan sistem pencarian informasi teknis.

Model embedding yang umum digunakan dalam RAG antara lain sentence-transformers, OpenAI text-embedding-ada-002, dan multilingual-e5. Untuk reranking, biasanya digunakan Cross-Encoder model yang mengevaluasi relevansi pasangan query-dokumen secara lebih akurat dibandingkan Bi-Encoder biasa.

Evaluasi sistem RAG dapat dilakukan menggunakan metrik seperti Faithfulness (seberapa akurat jawaban berdasarkan konteks), Answer Relevancy (seberapa relevan jawaban dengan pertanyaan), Context Precision (presisi konteks yang diambil), dan Context Recall (kelengkapan konteks yang diambil).
"""

    # Encode content
    content_bytes = content.encode("latin-1", errors="replace")
    content_len = len(content_bytes)

    lines = content.split(". ")
    page1_lines = lines[: len(lines) // 2]
    page2_lines = lines[len(lines) // 2 :]

    def make_page_stream(text_lines):
        ops = "BT\n/F1 11 Tf\n50 750 Td\n12 TL\n"
        max_chars = 80
        for sentence in text_lines:
            words = sentence.split()
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 <= max_chars:
                    current_line += (" " if current_line else "") + word
                else:
                    escaped = current_line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
                    ops += f"({escaped}) Tj T*\n"
                    current_line = word
            if current_line:
                escaped = current_line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
                ops += f"({escaped}) Tj T*\n"
        ops += "ET\n"
        return ops.encode("latin-1", errors="replace")

    stream1 = make_page_stream(page1_lines)
    stream2 = make_page_stream(page2_lines)

    objects = []
    offsets = []

    # Obj 1: Catalog
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    # Obj 2: Pages
    objects.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>\nendobj\n")
    # Obj 3: Page 1
    objects.append(b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R /Resources << /Font << /F1 7 0 R >> >> >>\nendobj\n")
    # Obj 4: Content stream page 1
    objects.append(f"4 0 obj\n<< /Length {len(stream1)} >>\nstream\n".encode() + stream1 + b"\nendstream\nendobj\n")
    # Obj 5: Page 2
    objects.append(b"5 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 6 0 R /Resources << /Font << /F1 7 0 R >> >> >>\nendobj\n")
    # Obj 6: Content stream page 2
    objects.append(f"6 0 obj\n<< /Length {len(stream2)} >>\nstream\n".encode() + stream2 + b"\nendstream\nendobj\n")
    # Obj 7: Font
    objects.append(b"7 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")

    pdf = b"%PDF-1.4\n"
    for i, obj in enumerate(objects):
        offsets.append(len(pdf))
        pdf += obj

    xref_offset = len(pdf)
    pdf += b"xref\n"
    pdf += f"0 {len(objects) + 1}\n".encode()
    pdf += b"0000000000 65535 f \n"
    for off in offsets:
        pdf += f"{off:010d} 00000 n \n".encode()

    pdf += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()

    with open(filename, "wb") as f:
        f.write(pdf)

    print(f"PDF berhasil dibuat: {filename}")


if __name__ == "__main__":
    create_minimal_pdf("test_rag_document.pdf")
