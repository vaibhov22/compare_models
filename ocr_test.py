
import fitz
import pytesseract
from PIL import Image
from pathlib import Path
from jiwer import wer, cer
import time
import re

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

BOOKS_DIR = Path("books")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

pdf_files = list(BOOKS_DIR.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError("No PDF found in books/")

doc = fitz.open(pdf_files[0])
results = []

def normalize(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

for i in range(min(5, len(doc))):
    page = doc[i]
    reference = page.get_text().strip()

    pix = page.get_pixmap(dpi=200, alpha=False)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    start = time.perf_counter()
    output = pytesseract.image_to_string(image, lang="eng")
    elapsed = time.perf_counter() - start

    ref = normalize(reference)
    hyp = normalize(output)

    # Skip pages without usable reference text
    if not ref:
        print(f"Page {i + 1}: skipped (no reference text)")
        continue

    word_error = wer(ref, hyp)
    char_error = cer(ref, hyp)

    results.append({
        "page": i + 1,
        "seconds": elapsed,
        "wer": word_error,
        "cer": char_error,
        "reference": reference,
        "ocr": output,
    })

    print(
        f"Page {i + 1}: "
        f"WER={word_error:.2%}, "
        f"CER={char_error:.2%}, "
        f"Time={elapsed:.2f}s"
    )

if results:
    report = []

    for item in results:
        report.append(
            f"Page: {item['page']}\n"
            f"Time: {item['seconds']:.2f}s\n"
            f"WER: {item['wer']:.2%}\n"
            f"CER: {item['cer']:.2%}\n\n"
            f"OCR OUTPUT:\n{item['ocr']}\n\n"
            f"REFERENCE:\n{item['reference']}\n"
        )

    report_path = RESULTS_DIR / "tesseract_benchmark.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print(f"\nReport saved to: {report_path}")
else:
    print("No pages with reference text were available.")
