// src/app/api/query/route.ts

import { NextResponse } from 'next/server';

interface QueryRequestPayload {
    query: string;
    n_results?: number;
    environment: string; // Changed from environmentId to environment
}

interface QueryResultItem {
    content: string;
    metadata: {
        id: string;
        name: string;
        path: string;
        url: string;
        language: string;
        environmentId: string; // This is metadata from the Python backend, keep as is
        [key: string]: any; // Allow other metadata properties
    };
    distance: number;
}

interface QueryResponsePayload {
    status: string;
    results: QueryResultItem[];
    message?: string;
}

// URL of your Python FastAPI indexing service's query endpoint
const PYTHON_BASE_API_URL = process.env.NEXT_PUBLIC_PYTHON_BASE_API_URL || "http://localhost:8001";
const PYTHON_QUERY_SERVICE_URL = `${PYTHON_BASE_API_URL}/query-content`;

export async function POST(request: Request) {
    const { query, n_results, environment }: QueryRequestPayload = await request.json(); // Changed from environmentId to environment
    console.log(`Next.js API: Received query request for: "${query}" (n_results: ${n_results}) from environment: ${environment}`); // Log environment

    if (!environment) { // Validate environment (name)
        return NextResponse.json(
            { status: "error", message: "Environment name is required for query." },
            { status: 400 }
        );
    }

    try {
        const response = await fetch(PYTHON_QUERY_SERVICE_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query, n_results, environment }), // Pass environment (name)
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error(`Next.js API: Python query service failed: ${response.status} - ${errorText}`);
            return NextResponse.json(
                { status: "error", message: `Query service failed: ${response.status} - ${errorText}` },
                { status: response.status }
            );
        }

        const result: QueryResponsePayload = await response.json();
        console.log(`Next.js API: Python query service responded with ${result.results.length} results.`);
        return NextResponse.json(result);

    } catch (error: any) {
        console.error(`Next.js API: Error forwarding query to Python service:`, error);
        return NextResponse.json(
            { status: "error", message: `An error occurred while querying: ${error.message}` },
            { status: 500 }
        );
    }
}
