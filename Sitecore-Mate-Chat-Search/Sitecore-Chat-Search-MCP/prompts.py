# prompts.py

RAG_PROMPT_TEMPLATE = """
You are a helpful assistant for a website.
Using only the information provided in the context below, please answer the user's question.

- If the context does not contain the answer, respond with: "I'm sorry, but the provided context does not contain the information needed to answer your question."
- Write in a friendly, professional tone.
- Begin with a brief summary, then elaborate with relevant details as needed.
- Use your own words; do not repeat the context verbatim.
- Ensure your answer is clear, concise, and uses all relevant information from the context.
- Do not include information that is not supported by the context.

Context:
---
{context}
---

User Question: {query}

Answer:
"""

# Dynamic Prompt for Page-Level Fields
PAGE_SUMMARY_PROMPT_TEMPLATE = """
Below are structured fields for a web page.
Please combine them into a single, clear, natural-language summary suitable for semantic search.

Fields:
{field_lines}

Summary:
"""

# Dynamic Prompt for Component Fields
COMPONENT_SUMMARY_PROMPT_TEMPLATE = """
Below are the fields from a page component ("{component_name}").
Please combine them into a single readable paragraph.

Fields:
{field_lines}

Component Description:
"""