// src/app/api/log-stream/route.ts

import { NextRequest, NextResponse } from 'next/server';

// The base URL of your Python FastAPI backend, where the /log-stream endpoint is located.
const PYTHON_BASE_API_URL = process.env.NEXT_PUBLIC_PYTHON_BASE_API_URL || 'http://localhost:8001';
const PYTHON_LOG_STREAM_URL = `${PYTHON_BASE_API_URL}/log-stream`;

/**
 * Next.js API route to proxy the Server-Sent Events (SSE) stream
 * from the Python FastAPI backend.
 * This allows the frontend to connect to a Next.js endpoint, which then
 * maintains the connection to the Python backend and forwards the events.
 */
export async function GET(request: NextRequest) {
  try {
    // Forward the request to the Python backend's SSE endpoint
    const pythonResponse = await fetch(PYTHON_LOG_STREAM_URL, {
      method: 'GET',
      // Ensure headers are set to maintain a streaming connection
      headers: {
        'Accept': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
      // Pass through any query parameters if needed
       body: request.body, // Not needed for GET, but useful for POST proxies
      // signal: request.signal, // Propagate abort signal from client to backend
    });

    if (!pythonResponse.ok) {
      const errorText = await pythonResponse.text();
      console.error(`Error proxying log stream from Python backend: ${pythonResponse.status} - ${errorText}`);
      return new NextResponse(
        `Error connecting to log stream: ${pythonResponse.status} - ${errorText}`,
        { status: pythonResponse.status, headers: { 'Content-Type': 'text/plain' } }
      );
    }

    // Return the Python backend's streaming response directly to the client
    // Next.js will handle the streaming headers automatically if the body is a ReadableStream
    return new NextResponse(pythonResponse.body, {
      status: 200,
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
    });

  } catch (error: any) {
    console.error('Error in Next.js /api/log-stream proxy:', error);
    return new NextResponse(
      `Internal Server Error: ${error.message}`,
      { status: 500, headers: { 'Content-Type': 'text/plain' } }
    );
  }
}
