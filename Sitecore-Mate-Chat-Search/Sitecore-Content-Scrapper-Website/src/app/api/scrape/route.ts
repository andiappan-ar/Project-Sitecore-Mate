// src/app/api/scrape/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { getEnvironmentById } from '@/lib/environments';
import { scrape } from '@/graphql/scraping';
import { processAndIndexContent } from '@/index/process-index';

/**
 * Sanitizes a string to be a valid ChromaDB collection name.
 * ChromaDB collection names must contain 3-512 characters from [a-zA-Z0-9._-],
 * starting and ending with a character in [a-zA-Z0-9].
 * @param name The original string to sanitize.
 * @returns A sanitized string suitable for a ChromaDB collection name.
 */
function sanitizeChromaDbName(name: string): string {
  // Replace spaces with hyphens
  let sanitized = name.replace(/\s+/g, '-');
  // Remove any characters not allowed by ChromaDB (keep a-z, A-Z, 0-9, ., _, -)
  sanitized = sanitized.replace(/[^a-zA-Z0-9._-]/g, '');
  // Ensure it starts and ends with an alphanumeric character
  sanitized = sanitized.replace(/^[^a-zA-Z0-9]+/, ''); // Remove non-alphanumeric from start
  sanitized = sanitized.replace(/[^a-zA-Z0-9]+$/, ''); // Remove non-alphanumeric from end
  // Ensure minimum length (ChromaDB requires at least 3 chars)
  if (sanitized.length < 3) {
    // If too short after sanitization, append a hash or unique identifier
    // For simplicity, we'll just append 'id' if it's too short.
    // In a real app, you might want a more robust unique suffix.
    sanitized = sanitized + 'id';
  }
  return sanitized.toLowerCase(); // Often good practice for consistency
}

/**
 * API route handler for scraping content.
 * Now handles POST requests.
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { environmentId } = body;

    if (!environmentId) {
      return NextResponse.json({ error: 'environmentId is required' }, { status: 400 });
    }

    const environment = getEnvironmentById(environmentId);

    if (!environment) {
      return NextResponse.json({ error: 'Environment not found' }, { status: 404 });
    }

    console.log(`Starting scrape process for environment: ${environment.name}`);

    // Sanitize the environment name for ChromaDB
    const sanitizedEnvironmentName = sanitizeChromaDbName(environment.name);
    console.log(`Original environment name: "${environment.name}", Sanitized for ChromaDB: "${sanitizedEnvironmentName}"`);

    const structuredContent = await scrape(
      environment.graphql_endpoint,
      environment.api_key,
      environment.root_path,
      environment.language,
      // Use the sanitized environment name when scraping and sending to indexing service
      sanitizedEnvironmentName
    );

    const indexingResponse = await processAndIndexContent(structuredContent);

    return NextResponse.json({
      message: 'Scraping and indexing initiated successfully.',
      data: indexingResponse,
    });
  } catch (error) {
    console.error('Error in /api/scrape:', error);
    const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred';
    return NextResponse.json({ error: 'Failed to scrape content', details: errorMessage }, { status: 500 });
  }
}
