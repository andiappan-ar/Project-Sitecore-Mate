// src/app/api/context/route.ts
import { NextResponse } from 'next/server';

interface ContextRequest {
    query: string;
    filters?: Record<string, unknown>; // Changed 'any' to 'unknown'
    limit?: number;
}

interface RetrievedContextItem {
    id: string;
    title: string;
    text: string;
}

export async function POST(request: Request) {
    const { query, filters, limit }: ContextRequest = await request.json();
    console.log(`Received context request with query: "${query}"`, { filters, limit });

    // --- Context Retrieval Logic Here (from Vector DB) ---
    // This is where you would implement the core RAG retrieval:
    // 1. Embed the `query` using your embedding model.
    // 2. Query your vector database with the embedded query and apply any `filters` (e.g., content type, language).
    // 3. Retrieve the top `limit` relevant content chunks/documents.
    // 4. Return the raw text content or structured data of these chunks.
    //    This retrieved context will then be sent to an LLM (like Claude) to generate the final answer.

    // Simulate retrieving context from a vector DB
    const simulatedContext: RetrievedContextItem[] = [
        { id: 'item1', title: 'Sitecore Headless Architecture', text: 'Sitecore headless architecture separates the presentation layer from the content management system, allowing for greater flexibility in front-end development using frameworks like Next.js.' },
        { id: 'item2', title: 'Sitecore Experience Platform (XP)', text: 'Sitecore XP is a comprehensive platform that combines content management, customer data, and marketing automation to deliver personalized digital experiences.' },
        { id: 'item3', title: 'GraphQL in Sitecore', text: 'Sitecore provides a GraphQL API for efficient content delivery, enabling developers to query specific content fields and structures without over-fetching data.' }
    ];

    // Filter based on a simple keyword match for simulation
    const relevantContext = simulatedContext.filter(item =>
        item.text.toLowerCase().includes(query.toLowerCase()) ||
        item.title.toLowerCase().includes(query.toLowerCase())
    );

    return NextResponse.json({
        query: query,
        retrievedContext: relevantContext.slice(0, limit || 3), // Return top N results
        message: "Context retrieved successfully (simulated from vector DB)."
    });
}
