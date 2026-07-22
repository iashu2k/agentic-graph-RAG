from unstructured.partition.pdf import partition_pdf


def parse_filing(filepath: str):
  """Parse a docugami 10-Q PDF, preserving section titles and tables."""
  elements = partition_pdf(
      filename=filepath,
      strategy="hi_res",          # better table/layout detection
      infer_table_structure=True  # preserve table structure as HTML
  )

  parsed = []
  current_section = "UNKNOWN"
  for el in elements:
    el_type = el.category
    text = str(el).strip()
    if not text:
      continue
    if el_type == "Title":
      current_section = text
    parsed.append({
        "section": current_section,
        "type": el_type,
        "text": text
    })
  return parsed
