
import fitz
import requests
import base64
import time
import re
from pathlib import Path
from jiwer import wer, cer

BOOKS_DIR = Path("books")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

MODEL = "qwen2.5vl:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"

pdf_files = list(BOOKS_DIR.glob("*.pdf"))
if not pdf_files:
    raise FileNotFoundError("Put your PDF inside the books folder.")

doc = fitz.open(pdf_files[0])

def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()

def extract_with_qwen(image_bytes):
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": (
                "Transcribe all visible text on this page exactly. "
                "Preserve words, punctuation, and paragraph order. "
                "Do not summarize or explain."
            ),
            "images": [image_base64],
            "stream": False,
            "options": {"temperature": 0}
        },
        timeout=600
    )
    response.raise_for_status()
    return response.json()["response"]

results = []

for i in range(min(2, len(doc))):
    page = doc[i]
    reference = page.get_text().strip()

    if not reference:
        print(f"Page {i + 1}: skipped (no reference text)")
        continue

    pix = page.get_pixmap(dpi=150, alpha=False)
    image_bytes = pix.tobytes("png")

    start = time.perf_counter()

    try:
        output = extract_with_qwen(image_bytes)
    except Exception as error:
        print(f"Page {i + 1}: ERROR - {error}")
        continue

    elapsed = time.perf_counter() - start
    reference_clean = normalize(reference)
    output_clean = normalize(output)

    word_error = wer(reference_clean, output_clean)
    char_error = cer(reference_clean, output_clean)

    results.append((i + 1, elapsed, word_error, char_error, output, reference))

    print(
        f"Page {i + 1}: WER={word_error:.2%}, "
        f"CER={char_error:.2%}, Time={elapsed:.2f}s"
    )

report = [
    "QWEN2.5-VL OCR BENCHMARK",
    f"Model: {MODEL}",
    f"PDF: {pdf_files[0].name}",
    "Image resolution: 150 DPI",
    ""
]

for page_num, elapsed, word_error, char_error, output, reference in results:
    report.append(
        f"PAGE {page_num}\n"
        f"Time: {elapsed:.2f}s\n"
        f"WER: {word_error:.2%}\n"
        f"CER: {char_error:.2%}\n\n"
        f"MODEL OUTPUT:\n{output}\n\n"
        f"REFERENCE TEXT:\n{reference}\n"
    )

if results:
    report += [
        "\nAVERAGE RESULTS",
        f"Average WER: {sum(r[2] for r in results) / len(results):.2%}",
        f"Average CER: {sum(r[3] for r in results) / len(results):.2%}",
        f"Average time per page: {sum(r[1] for r in results) / len(results):.2f}s"
    ]

output_path = RESULTS_DIR / "qwen_benchmark.txt"
output_path.write_text("\n".join(report), encoding="utf-8")
doc.close()

print(f"\nReport saved to: {output_path}")
