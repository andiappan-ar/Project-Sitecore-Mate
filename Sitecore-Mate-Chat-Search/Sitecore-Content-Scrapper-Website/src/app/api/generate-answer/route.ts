// src/app/api/generate-answer/route.ts

import { NextResponse } from 'next/server';

interface GenerateAnswerRequestPayload {
    query: string;
    n_results?: number; // Number of documents to retrieve for context, default 5
    environment: string; // Changed from environmentId to environment
}

interface GenerateAnswerResponsePayload {
    status: string;
    answer?: string;
    retrieved_context?: Array<{
        content: string;
        metadata: {
            id: string;
            name: string;
            path: string;
            url: string;
            language: string;
            environmentId: string; // This is metadata from the Python backend, keep as is
            [key: string]: any;
        };
        distance: number;
    }>;
    message?: string;
}

// URL of your Python FastAPI indexing service's generate-answer endpoint
const PYTHON_BASE_API_URL = process.env.NEXT_PUBLIC_PYTHON_BASE_API_URL || "http://localhost:8001";
const PYTHON_GENERATE_ANSWER_SERVICE_URL = `${PYTHON_BASE_API_URL}/generate-answer`;

export async function POST(request: Request) {
    const { query, n_results, environment }: GenerateAnswerRequestPayload = await request.json(); // Changed from environmentId to environment
    console.log(`Next.js API: Received generate answer request for: "${query}" (n_results: ${n_results}) from environment: ${environment}`); // Log environment

    if (!environment) { // Validate environment (name)
        return NextResponse.json(
            { status: "error", message: "Environment name is required for answer generation." },
            { status: 400 }
        );
    }

    try {
        const response = await fetch(PYTHON_GENERATE_ANSWER_SERVICE_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query, n_results, environment }), // Pass environment (name)
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error(`Next.js API: Python generate answer service failed: ${response.status} - ${errorText}`);
            return NextResponse.json(
                { status: "error", message: `Generate answer service failed: ${response.status} - ${errorText}` },
                { status: response.status }
            );
        }

        const result: GenerateAnswerResponsePayload = await response.json();
        console.log(`Next.js API: Python generate answer service responded.`);
        return NextResponse.json(result);

    } catch (error: any) {
        console.error(`Next.js API: Error forwarding generate answer request to Python service:`, error);
        return NextResponse.json(
            { status: "error", message: `An error occurred while generating answer: ${error.message}` },
            { status: 500 }
        );
    }
}
