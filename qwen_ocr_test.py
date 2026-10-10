
import pymupdf
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
DPI = 150
PAGES_TO_TEST = 2


def normalize(text):
    """Normalize whitespace and case, preserving punctuation."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_with_qwen(image_bytes):
    """Send a page image to Qwen through Ollama."""
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": (
                "Transcribe all visible text on this page exactly. "
                "Include every heading, paragraph, and visible word. "
                "Preserve original wording, punctuation, and reading order. "
                "Do not summarize, omit text, or add explanations."
            ),
            "images": [image_base64],
            "stream": False,
            "options": {"temperature": 0}
        },
        timeout=600
    )

    response.raise_for_status()
    return response.json()["response"]


pdf_files = sorted(BOOKS_DIR.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError("Put your PDF inside the books folder.")

pdf_path = pdf_files[0]
doc = pymupdf.open(pdf_path)
results = []

try:
    for i in range(min(PAGES_TO_TEST, len(doc))):
        page = doc[i]
        reference = page.get_text().strip()

        if not reference:
            print(f"Page {i + 1}: skipped (no reference text)")
            continue

        # Render the PDF page for vision-model OCR.
        pix = page.get_pixmap(dpi=DPI, alpha=False)
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

        # Calculate error rates using the complete strings.
        word_error = wer(reference_clean, output_clean)
        char_error = cer(reference_clean, output_clean)

        reference_words = len(reference_clean.split())
        output_words = len(output_clean.split())
        reference_chars = len(reference_clean)
        output_chars = len(output_clean)

        results.append({
            "page": i + 1,
            "time": elapsed,
            "wer": word_error,
            "cer": char_error,
            "reference_words": reference_words,
            "output_words": output_words,
            "reference_chars": reference_chars,
            "output_chars": output_chars,
            "output": output,
            "reference": reference,
        })

        print(f"\nPage {i + 1}")
        print(f"WER: {word_error:.2%}")
        print(f"CER: {char_error:.2%}")
        print(f"Time: {elapsed:.2f} seconds")
        print(f"Reference words: {reference_words}")
        print(f"Output words: {output_words}")
        print(f"Reference characters: {reference_chars}")
        print(f"Output characters: {output_chars}")

        if output_words < reference_words * 0.8:
            print("WARNING: Output may be incomplete.")

finally:
    doc.close()


# Build detailed report.
report = [
    "QWEN2.5-VL OCR BENCHMARK",
    f"Model: {MODEL}",
    f"PDF: {pdf_path.name}",
    f"Image resolution: {DPI} DPI",
    f"Pages requested: {PAGES_TO_TEST}",
    ""
]

for item in results:
    report.append(
        f"PAGE {item['page']}\n"
        f"Time: {item['time']:.2f}s\n"
        f"WER: {item['wer']:.2%}\n"
        f"CER: {item['cer']:.2%}\n"
        f"Reference words: {item['reference_words']}\n"
        f"Output words: {item['output_words']}\n"
        f"Reference characters: {item['reference_chars']}\n"
        f"Output characters: {item['output_chars']}\n\n"
        f"MODEL OUTPUT:\n{item['output']}\n\n"
        f"REFERENCE TEXT:\n{item['reference']}\n"
    )

if results:
    report.extend([
        "\nAVERAGE RESULTS",
        f"Average WER: {sum(x['wer'] for x in results) / len(results):.2%}",
        f"Average CER: {sum(x['cer'] for x in results) / len(results):.2%}",
        f"Average time per page: {sum(x['time'] for x in results) / len(results):.2f}s",
        f"Pages successfully tested: {len(results)}"
    ])
else:
    report.append("No pages were successfully benchmarked.")

output_path = RESULTS_DIR / "qwen_benchmark.txt"
output_path.write_text("\n".join(report), encoding="utf-8")

print(f"\nReport saved to: {output_path}")
