// src/index/process-index.ts

/**
 * Sends a scraped content item, including its layout data and environment details,
 * to the Python backend for processing and indexing.
 * @param item The scraped item data, which now includes layout fields.
 * @param environment The full environment configuration object.
 */
export async function processAndIndexContent(
  item: {
    id: string;
    name: string;
    path: string;
    url: string;
    language: string;
    content: string;
    sharedLayout?: any; // The shared layout data from the GraphQL query
    finalLayout?: any;  // The final layout data from the GraphQL query
  },
  environment: {
    id: string;
    url: string;      // This will be used as the graphql_endpoint
    apiKey: string;
  }
) {
  console.log('--- Sending Item to Python Indexing Service ---');
  console.log(`  ID: ${item.id}`);
  console.log(`  Name: ${item.name}`);
  console.log(`  Path: ${item.path}`);

  try {
    // Construct the request body with all the fields required by the Python backend's
    // IndexRequestItem model.
    const requestBody = {
      id: item.id,
      name: item.name,
      path: item.path,
      url: item.url,
      language: item.language,
      content: item.content,
      environmentId: environment.id,
      // NEW: Add the required fields for datasource fetching
      graphql_endpoint: environment.url,
      api_key: environment.apiKey,
      sharedLayout: item.sharedLayout,
      finalLayout: item.finalLayout,
    };

    console.log('--- Request Body to be Sent ---');
    console.log(JSON.stringify(requestBody, null, 2));

    const response = await fetch(`${process.env.NEXT_PUBLIC_PYTHON_API_URL}/index-content`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Indexing service failed: ${response.status} - ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error(`Error sending item ${item.id} to indexing service:`, error);
    throw error;
  }
}
