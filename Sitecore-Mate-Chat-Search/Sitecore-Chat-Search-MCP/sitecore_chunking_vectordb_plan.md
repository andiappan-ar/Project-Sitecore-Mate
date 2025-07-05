# Sitecore Content Chunking & VectorDB Implementation Plan

## Overview

This document outlines the strategy to chunk Sitecore content (pages and components) and store it efficiently in a Vector Database (VectorDB) to enable semantic search and accurate retrieval.

---

## 1. Chunking Strategy

### Inputs

- Sitecore **Page Fields**: can be simple text or rich text (HTML).
- Sitecore **Component Fields**: components referenced by pages, also simple or rich text.
- Components may be reused across multiple pages.

### Chunking Steps

- For each **Page Field**:
    - If the field content is **simple text or short**, create a single chunk with the full field content.
    - If the field content is **rich text (HTML)** or larger text:
        1. **Use a robust HTML parsing library to clean the content, removing all HTML tags (`<p>`, `<div>`, `<span>`, etc.) to extract pure text.**
        2. Parse the extracted text into logical blocks (e.g., split by newlines that corresponded to paragraphs or headings).
        3. For each block:
            - If the block's length exceeds the token limit (~300 tokens), split it further into sentences.
        4. Combine smaller consecutive blocks/sentences to reach a sensible chunk size (100–300 tokens).
    - Store each chunk with metadata: `page_id`, `field_name`, `chunk_index`.

- For each **Component referenced** by a page:
    - Process component fields similarly (simple text = 1 chunk; rich text = **HTML stripping** + paragraph/sentence chunking).
    - Store each chunk with metadata: `page_id`, `component_id`, `field_name`, `chunk_index`.

### Chunk Size and Overlap

- **Target chunk size**: 100–300 tokens.
- **Optional overlap** of 10–20 tokens between adjacent chunks to maintain context.
- Avoid chunks exceeding model input token limits.

---

## 2. VectorDB Metadata Schema

Each chunk is stored with the following metadata fields:

| Field Name     | Description                                                                          |
| :------------- | :----------------------------------------------------------------------------------- |
| `id`           | **Deterministic, unique chunk identifier** (e.g., hash of `page_id` + `field_name` + `chunk_index`) |
| `page_id`      | Sitecore page identifier                                                             |
| `page_path`    | Page URL                                                                             |
| `page_title`   | Title of the page (either from a 'pageTitle' field or the Sitecore item name)        |
| `component_id` | Component ID if the chunk belongs to a component (optional)                          |
| `field_name`   | Name of the field the chunk was created from                                         |
| `chunk_index`  | Order index of the chunk within the field                                            |
| `chunk_text`   | The actual text content of the chunk                                                 |
| `language`     | Content language (if multilingual)                                                   |
| `created_at`   | Timestamp of when the chunk was created                                              |

---

## 3. Small Page Example (Single Chunk)

```json
{
  "id": "b1b2b3b4b5b6b7b8b9b0c1c2c3c4c5c6",
  "page_id": "page001",
  "page_path": "/sitecore/content/home",
  "page_title": "Home Page",
  "component_id": null,
  "field_name": "MainContent",
  "chunk_index": 1,
  "chunk_text": "Welcome to our home page. We offer excellent services for all your needs.",
  "language": "en",
  "created_at": "2025-07-05T10:00:00Z"
}
```

---

## 4. Big Page Example (Multiple Chunks)

```json
{
  "id": "d1d2d3d4d5d6d7d8d9d0e1e2e3e4e5e6",
  "page_id": "page002",
  "page_path": "/sitecore/content/about",
  "page_title": "About Us",
  "component_id": null,
  "field_name": "Body",
  "chunk_index": 1,
  "chunk_text": "Our company was founded in 1990 with a vision to innovate...",
  "language": "en",
  "created_at": "2025-07-05T10:05:00Z"
}
```json
{
  "id": "f1f2f3f4f5f6f7f8f9f0g1g2g3g4g5g6",
  "page_id": "page002",
  "page_path": "/sitecore/content/about",
  "page_title": "About Us",
  "component_id": null,
  "field_name": "Body",
  "chunk_index": 2,
  "chunk_text": "We specialize in delivering high-quality products that exceed expectations...",
  "language": "en",
  "created_at": "2025-07-05T10:06:00Z"
}
```json
{
  "id": "h1h2h3h4h5h6h7h8h9h0i1i2i3i4i5i6",
  "page_id": "page002",
  "page_path": "/sitecore/content/about",
  "page_title": "About Us",
  "component_id": "comp456",
  "field_name": "Description",
  "chunk_index": 1,
  "chunk_text": "This component describes our core values and mission...",
  "language": "en",
  "created_at": "2025-07-05T10:07:00Z"
}
```

---

## 5. Data Flow

```text
Sitecore Page
    ├── Page Fields
    │     ├── Chunking (simple or rich text parsing)
    │     └── Store chunk + metadata + embedding in VectorDB
    └── Components (referenced)
          ├── Component Fields
          ├── Chunking (same as page fields)
          └── Store chunk + metadata + embedding in VectorDB (per page reference)
```

---

## 6. Retrieval and User Display

- Retrieve top matching chunks by vector similarity.
- Aggregate chunks by `page_id` to present unified search results at the page level.
- Display to user:
  - Page Title
  - Page URL (Sitecore path)
  - Snippet(s) from relevant chunks
- Hide internal component or field details from user-facing results.

---

## 7. Notes

- Components are stored per page reference to correctly associate search results with pages.
- Chunk size is optimized for embedding model limits.
- Metadata enables filtering, grouping, and traceability.
- Overlap chunks can improve context continuity for embeddings.

---

*End of Document*
