
import fitz
from pathlib import Path

BOOKS_DIR = Path("books")
OUTPUT_DIR = Path("extracted_text")
OUTPUT_DIR.mkdir(exist_ok=True)

pdf_files = list(BOOKS_DIR.glob("*.pdf"))

if not pdf_files:
    print("No PDF found in the books folder.")
else:
    pdf_path = pdf_files[0]
    doc = fitz.open(pdf_path)

    all_text = []

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text()
        all_text.append(f"\n--- Page {page_number} ---\n{text}")

    output_path = OUTPUT_DIR / f"{pdf_path.stem}_reference.txt"
    output_path.write_text("\n".join(all_text), encoding="utf-8")

    print(f"Book: {pdf_path.name}")
    print(f"Total pages: {len(doc)}")
    print(f"Extracted characters: {sum(len(t) for t in all_text)}")
    print(f"Reference text saved to: {output_path}")



from pathlib import Path
import re
import statistics

results_dir = Path("results")

models = {
    "Tesseract OCR": "tesseract_benchmark.txt",
    "Gemma 3 (4B)": "gemma_benchmark.txt",
    "MiniCPM-V (1B)": "minicpm_benchmark.txt",
    "Qwen2.5-VL (3B)": "qwen_benchmark.txt",
}

report = [
    "OCR MODEL BENCHMARK COMPARISON",
    "=" * 60,
    f"{'Model':<20} {'Avg WER':>10} {'Avg CER':>10} {'Avg Time':>12}",
    "-" * 60,
]

summary = []

for model, filename in models.items():
    path = results_dir / filename

    if not path.exists():
        print(f"Missing report: {path}")
        continue

    text = path.read_text(encoding="utf-8")

    wers = [float(x) for x in re.findall(
        r"WER\s*[:=]\s*([\d.]+)%", text, re.I
    )]
    cers = [float(x) for x in re.findall(
        r"CER\s*[:=]\s*([\d.]+)%", text, re.I
    )]
    times = [float(x) for x in re.findall(
        r"Time\s*[:=]\s*([\d.]+)\s*s", text, re.I
    )]

    if not wers or not cers or not times:
        print(f"Could not read metrics from: {filename}")
        continue

    avg_wer = statistics.mean(wers)
    avg_cer = statistics.mean(cers)
    avg_time = statistics.mean(times)

    summary.append((model, avg_wer, avg_cer, avg_time))

    report.append(
        f"{model:<20} {avg_wer:>9.2f}% "
        f"{avg_cer:>9.2f}% {avg_time:>9.2f} sec"
    )

if summary:
    best = min(summary, key=lambda item: item[1])
    report.extend([
        "-" * 60,
        f"Lowest average WER: {best[0]}",
        "",
        "Note: Lower WER and CER are better.",
        "Compare models on the same pages for a fair evaluation.",
    ])

    output = results_dir / "final_comparison.txt"
    output.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nSaved to: {output}")
else:
    print("No valid benchmark results found.")
