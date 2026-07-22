def chunk_by_section(parsed_elements, max_chars=1500, overlap_ratio=0.125):
  """Group elements by section, then split long sections into overlapping chunks."""
  sections = {}
  for el in parsed_elements:
    sections.setdefault(el["section"], []).append(el["text"])

  chunks = []
  for section, texts in sections.items():
    full_text = "\n".join(texts)
    if len(full_text) <= max_chars:
      chunks.append({"section": section, "content": full_text})
      continue

    overlap = int(max_chars * overlap_ratio)
    start = 0
    while start < len(full_text):
      end = min(start + max_chars, len(full_text))
      chunk_text = full_text[start:end]
      chunks.append({"section": section, "content": chunk_text})
      start += max_chars - overlap

  return chunks
