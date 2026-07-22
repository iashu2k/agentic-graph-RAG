from groq import Groq
from app.config import settings

client = Groq(api_key=settings.groq_api_key)


def generate_answer(query: str, context_chunks: list[dict]):
  context_str = "\n\n".join(
      f"[Source: {c['company']} {c['filing_type']} {c['fiscal_quarter']} FY{c['fiscal_year']}, Section: {c['section']}]\n{c['content']}"
      for c in context_chunks
  )

  prompt = f"""Answer the question using ONLY the context below. Cite the source (company, filing type, quarter, fiscal year, section) for every claim.

Context:
{context_str}

Question: {query}

Answer:"""

  response = client.chat.completions.create(
      model="llama-3.3-70b-versatile",
      messages=[{"role": "user", "content": prompt}],
      temperature=0.1,
  )
  return response.choices[0].message.content
