
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

MODEL = "gemma3:4b"
OLLAMA_URL = "http://localhost:11434/api/generate"

pdf_files = list(BOOKS_DIR.glob("*.pdf"))
if not pdf_files:
    raise FileNotFoundError("Put your PDF inside the books folder.")

doc = fitz.open(pdf_files[0])


def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_with_gemma(image_bytes):
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": (
                "Transcribe all visible text in this book page exactly. "
                "Preserve the original words, punctuation, and paragraph order. "
                "Do not summarize, explain, or add anything."
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

    # Render the same PDF page for vision-model input
    pix = page.get_pixmap(dpi=150, alpha=False)
    image_bytes = pix.tobytes("png")

    start = time.perf_counter()

    try:
        output = extract_with_gemma(image_bytes)
    except Exception as error:
        print(f"Page {i + 1}: ERROR - {error}")
        continue

    elapsed = time.perf_counter() - start

    reference_clean = normalize(reference)
    output_clean = normalize(output)

    word_error = wer(reference_clean, output_clean)
    char_error = cer(reference_clean, output_clean)

    results.append({
        "page": i + 1,
        "time": elapsed,
        "wer": word_error,
        "cer": char_error,
        "output": output,
        "reference": reference,
    })

    print(
        f"Page {i + 1}: "
        f"WER={word_error:.2%}, "
        f"CER={char_error:.2%}, "
        f"Time={elapsed:.2f}s"
    )

report = []

for item in results:
    report.append(
        f"PAGE {item['page']}\n"
        f"Time: {item['time']:.2f}s\n"
        f"WER: {item['wer']:.2%}\n"
        f"CER: {item['cer']:.2%}\n\n"
        f"MODEL OUTPUT:\n{item['output']}\n\n"
        f"REFERENCE TEXT:\n{item['reference']}\n"
    )

output_path = RESULTS_DIR / "gemma_benchmark.txt"
output_path.write_text("\n".join(report), encoding="utf-8")

print(f"\nReport saved to: {output_path}")
