import time
from sec_edgar_downloader import Downloader

dl = Downloader("AshutoshMishra", "iashu2k@gmail.com", "./sec_filings")

tech_tickers = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "NVDA",   # NVIDIA
    "GOOGL",  # Alphabet
    "AMZN",   # Amazon
    "META",   # Meta
    "CRM",    # Salesforce
    "ORCL",   # Oracle
]

start_date = "2023-01-01"
end_date = "2026-01-01"

for ticker in tech_tickers:
  print(f"Downloading filings for {ticker}...")

  dl.get("10-K", ticker, after=start_date, before=end_date, limit=3)
  time.sleep(0.5)

  dl.get("10-Q", ticker, after=start_date, before=end_date, limit=3)
  time.sleep(0.5)

print("Done. Filings saved under ./sec_filings")
