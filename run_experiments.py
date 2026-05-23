"""
Script eksperimen otomatis: ubah parameter app.py, restart server, upload PDF, tanya, catat hasil.
Dijalankan DARI LUAR venv (pakai python biasa), karena ia mengontrol server sebagai subprocess.
"""
import subprocess, time, json, re, sys, os, signal
import urllib.request, urllib.parse

BASE = r"C:\Users\PT SITC Indonesia\Documents\IL\rag-playground"
VENV_PYTHON = os.path.join(BASE, "venv", "Scripts", "python.exe")
APP = os.path.join(BASE, "app.py")
URL = "http://127.0.0.1:7860"

PDFS = {
    "rag":  os.path.join(BASE, "test_rag_document.pdf"),
    "python": os.path.join(BASE, "pdf_python.pdf"),
    "datascience": os.path.join(BASE, "pdf_datascience.pdf"),
}

QUESTIONS = {
    "rag": "Apa komponen utama RAG dan bagaimana proses kerjanya?",
    "python": "Apa saja tipe data dasar dalam Python dan apa itu virtual environment?",
    "datascience": "Apa itu CRISP-DM dan bagaimana cara mengevaluasi model machine learning?",
}

DEFAULT_PARAMS = {
    "CHUNK_SIZE": 300,
    "TOP_K_INITIAL": 20,
    "FINAL_TOP_M": 5,
    "USE_RERANKER": True,
    "TEMPERATURE": 0.2,
    "GROQ_MODEL": '"openai/gpt-oss-120b"',
}

# Eksperimen: (label, param_overrides, pdf_key)
EXPERIMENTS = [
    # --- 1. CHUNK_SIZE kecil ---
    ("1a_chunk_kecil_100",   {"CHUNK_SIZE": 100}, "rag"),
    ("1b_chunk_sedang_300",  {"CHUNK_SIZE": 300}, "rag"),
    ("1c_chunk_besar_600",   {"CHUNK_SIZE": 600}, "rag"),
    # --- 2. TOP_K_INITIAL ---
    ("2a_topk_kecil_5",      {"TOP_K_INITIAL": 5},  "python"),
    ("2b_topk_sedang_20",    {"TOP_K_INITIAL": 20}, "python"),
    ("2c_topk_besar_40",     {"TOP_K_INITIAL": 40}, "python"),
    # --- 3. FINAL_TOP_M ---
    ("3a_topm_sedikit_2",    {"FINAL_TOP_M": 2}, "datascience"),
    ("3b_topm_sedang_5",     {"FINAL_TOP_M": 5}, "datascience"),
    ("3c_topm_banyak_8",     {"FINAL_TOP_M": 8}, "datascience"),
    # --- 4. USE_RERANKER ---
    ("4a_reranker_true",     {"USE_RERANKER": True},  "rag"),
    ("4b_reranker_false",    {"USE_RERANKER": False}, "rag"),
    # --- 5. TEMPERATURE ---
    ("5a_temp_001",          {"TEMPERATURE": 0.1}, "python"),
    ("5b_temp_050",          {"TEMPERATURE": 0.5}, "python"),
    ("5c_temp_090",          {"TEMPERATURE": 0.9}, "python"),
    # --- 6. GROQ_MODEL ---
    ("6a_model_8b",   {"GROQ_MODEL": '"llama-3.1-8b-instant"'},   "datascience"),
    ("6b_model_70b",  {"GROQ_MODEL": '"llama-3.3-70b-versatile"'}, "datascience"),
    ("6c_model_120b", {"GROQ_MODEL": '"openai/gpt-oss-120b"'},     "datascience"),
]


def patch_app(overrides: dict):
    """Tulis ulang parameter di app.py."""
    with open(APP, encoding="utf-8") as f:
        src = f.read()
    params = {**DEFAULT_PARAMS, **overrides}
    for key, val in params.items():
        if isinstance(val, bool):
            src = re.sub(rf"^{key}\s*=.*", f"{key} = {val}", src, flags=re.MULTILINE)
        elif isinstance(val, str):  # sudah dibungkus tanda kutip
            src = re.sub(rf"^{key}\s*=.*", f"{key} = {val}", src, flags=re.MULTILINE)
        else:
            src = re.sub(rf"^{key}\s*=.*", f"{key} = {val}", src, flags=re.MULTILINE)
    with open(APP, "w", encoding="utf-8") as f:
        f.write(src)


def start_server():
    proc = subprocess.Popen(
        [VENV_PYTHON, APP],
        cwd=BASE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    # Tunggu server siap
    for _ in range(60):
        try:
            urllib.request.urlopen(f"{URL}/health", timeout=2)
            return proc
        except Exception:
            time.sleep(1)
    proc.terminate()
    raise RuntimeError("Server tidak mau start dalam 60 detik")


def stop_server(proc):
    try:
        if sys.platform == "win32":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
    time.sleep(2)


def ingest_pdf(pdf_path: str):
    import urllib.request
    boundary = "----FormBoundary7MA4YWxkTrZu0gW"
    with open(pdf_path, "rb") as f:
        file_data = f.read()
    fname = os.path.basename(pdf_path)
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="replace"\r\n\r\ntrue\r\n'
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{fname}"\r\n'
        f"Content-Type: application/pdf\r\n\r\n"
    ).encode() + file_data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{URL}/ingest",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=30)
    return json.loads(resp.read())


def ask(question: str):
    data = json.dumps({"query": question}).encode()
    req = urllib.request.Request(
        f"{URL}/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=60)
    elapsed = round((time.time() - t0) * 1000)
    result = json.loads(resp.read())
    return result, elapsed


def main():
    results = {}
    total = len(EXPERIMENTS)
    for i, (label, overrides, pdf_key) in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/{total}] === {label} ===")
        print(f"  PDF: {pdf_key}  |  override: {overrides}")

        patch_app(overrides)
        proc = start_server()
        print("  Server UP")

        try:
            ing = ingest_pdf(PDFS[pdf_key])
            print(f"  Ingested: {ing['ingested_chunks']} chunks")
            q = QUESTIONS[pdf_key]
            res, elapsed = ask(q)
            ans = res.get("answer", "")
            metrics = res.get("metrics", {})
            top_score = res["contexts"][0]["score"] if res.get("contexts") else 0
            print(f"  Q: {q[:60]}...")
            print(f"  A: {ans[:120]}...")
            print(f"  retrieval={metrics.get('retrieval_ms')}ms  gen={metrics.get('generation_ms')}ms  top_score={top_score}")
            results[label] = {
                "params": {**DEFAULT_PARAMS, **overrides},
                "pdf": pdf_key,
                "question": q,
                "answer": ans,
                "chunks": ing["ingested_chunks"],
                "retrieval_ms": metrics.get("retrieval_ms"),
                "generation_ms": metrics.get("generation_ms"),
                "top_score": top_score,
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            results[label] = {"error": str(e), "params": overrides}
        finally:
            stop_server(proc)
            print("  Server stopped")

    out = os.path.join(BASE, "experiment_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSemua selesai. Hasil disimpan di: {out}")


if __name__ == "__main__":
    main()
