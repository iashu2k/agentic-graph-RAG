from app.config import settings
from app.ingestion.parser import parse_filing
from app.ingestion.chunker import chunk_by_section
from app.ingestion.loader import load_chunks
import sys
import os
import re
import glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


TICKER_TO_COMPANY = {
    "AAPL": "Apple",
    "AMZN": "Amazon",
    "INTC": "Intel",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
}

FILENAME_PATTERN = re.compile(
  r"(\d{4})\s+(Q[1-3])\s+([A-Z]+)\.pdf", re.IGNORECASE)


def parse_filename(filename: str):
  """Extract year, quarter, ticker from 'YYYY QN TICKER.pdf' format."""
  match = FILENAME_PATTERN.search(os.path.basename(filename))
  if not match:
    raise ValueError(f"Filename doesn't match expected pattern: {filename}")
  year, quarter, ticker = match.groups()
  return int(year), quarter.upper(), ticker.upper()


def main():
  pdf_files = sorted(glob.glob(os.path.join(settings.docs_dir, "*.pdf")))
  print(f"Found {len(pdf_files)} PDFs in {settings.docs_dir}")

  for filepath in pdf_files:
    try:
      fiscal_year, fiscal_quarter, ticker = parse_filename(filepath)
      company = TICKER_TO_COMPANY.get(ticker, ticker)

      print(
        f"\nProcessing {os.path.basename(filepath)} -> {company} {fiscal_quarter} FY{fiscal_year}")
      elements = parse_filing(filepath)
      print(f"  Parsed {len(elements)} elements.")

      chunks = chunk_by_section(elements)
      print(f"  Created {len(chunks)} chunks.")

      load_chunks(chunks, company, "10-Q", fiscal_year, fiscal_quarter)

    except Exception as e:
      print(f"  ERROR processing {filepath}: {e}")
      continue

  print("\nIngestion complete.")


if __name__ == "__main__":
  main()
