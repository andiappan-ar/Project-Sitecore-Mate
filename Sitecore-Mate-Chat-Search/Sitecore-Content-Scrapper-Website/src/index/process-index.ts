// src/index/process-index.ts

import { ContentPayload } from "@/graphql/scraping";

// The base URL of your Python indexing service, now read from environment variables
// with a fallback to a default value for convenience.
// This should be the base URL, e.g., 'http://127.0.0.1:8001'
const PYTHON_BASE_API_URL = process.env.NEXT_PUBLIC_PYTHON_BASE_API_URL || 'http://127.0.0.1:8001';

/**
 * Sends the structured content payload to the Python backend for chunking and indexing.
 * @param payload The structured data containing pages and environment name.
 * @returns The response from the indexing service.
 */
export async function processAndIndexContent(payload: ContentPayload) {
  // Filter out pages that have no fields and no components to avoid sending empty data.
  const pagesWithContent = payload.pages.filter(
    page => (page.fields && page.fields.length > 0) || (page.components && page.components.length > 0)
  );

  if (pagesWithContent.length === 0) {
    console.log('--- No pages with content found to index. ---');
    return { message: 'Scraping complete, but no new content was found to index.' };
  }

  const filteredPayload: ContentPayload = {
    ...payload,
    pages: pagesWithContent,
  };

  console.log(`--- Sending ${filteredPayload.pages.length} pages (with content) to Python Indexing Service for environment: ${filteredPayload.environment} ---`);
  
  // Re-added: Log the full payload for debugging
  console.log('--- Full Payload to be sent to Python indexing service: ---');
  console.log(JSON.stringify(filteredPayload, null, 2)); // <-- Re-added this log

  // Construct the full URL for the indexing endpoint
  const INDEXING_SERVICE_URL = `${PYTHON_BASE_API_URL}/index-content`;

  try {
    const response = await fetch(INDEXING_SERVICE_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(filteredPayload),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`Indexing service failed with status ${response.status}:`, errorText);
      throw new Error(`Indexing service failed: ${response.status} - ${errorText}`);
    }

    console.log('--- Successfully sent content to indexing service ---');
    return await response.json();

  } catch (error: any) {
    // --- IMPROVED ERROR HANDLING ---
    // Specifically check for the connection refused error.
    if (error.cause && error.cause.code === 'ECONNREFUSED') {
      const friendlyError = new Error(
        `Connection to Python backend failed. Please ensure the Python server is running on ${PYTHON_BASE_API_URL} before scraping.`
      );
      console.error('******************************************************************');
      console.error('FETCH ERROR:', friendlyError.message);
      console.error('******************************************************************');
      throw friendlyError;
    }
    
    console.error('Error sending content to indexing service:', error);
    throw error; // Re-throw other errors
  }
}
