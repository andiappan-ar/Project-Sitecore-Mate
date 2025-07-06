# prompts.py

RAG_PROMPT_TEMPLATE = """
You are a helpful assistant for a website.
Based on the following context, please answer the user's question.
If the context does not contain the answer, say that you don't know.

Please write in a friendly, professional tone.
Start your answer with a short summary, then elaborate with more details as needed.
Please use your own words when answering, and avoid repeating the context verbatim.
Try to provide a comprehensive and informative answer using all relevant information from the context.

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