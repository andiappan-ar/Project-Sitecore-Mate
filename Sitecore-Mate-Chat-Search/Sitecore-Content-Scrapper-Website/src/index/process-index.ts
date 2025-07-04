// src/index/process-index.ts

/**
 * Sends a scraped content item to the Python backend for processing and indexing.
 * @param item The scraped item data.
 * @param environment The environment configuration.
 * @returns The response from the indexing service.
 */
export async function processAndIndexContent(
  item: {
    id: string;
    name: string;
    path: string;
    url: string;
    language: string;
    content: string;
    children: any[]; // Consider defining a more specific type for children
  },
  environment: any // Consider defining a more specific type for environment
) {
  // Log the details of the item being sent to the backend.
  console.log('--- Sending Item to Python Indexing Service ---');
  console.log(`  ID: ${item.id.replace(/-/g, '').toUpperCase()}`);
  console.log(`  Name: ${item.name}`);
  console.log(`  Path: ${item.path}`);
  console.log(`  Language: ${item.language}`);
  console.log(`  Content Length: ${item.content.length} characters`);

  try {
    // Create the request body object to be sent.
    const requestBody = {
      id: item.id.replace(/-/g, '').toUpperCase(),
      name: item.name,
      path: item.path,
      url: item.url,
      language: item.language,
      content: item.content,
      childrenPaths: (item.children || []).map((child) => ({ name: child.name, path: child.path })),
      environmentId: environment.id,
    };

    // DEBUGGING STEP: Log the request body to the console before sending.
    // This will confirm exactly what data is being sent to the Python server.
    console.log('--- Request Body to be Sent ---');
    console.log(JSON.stringify(requestBody, null, 2));

    // Send a POST request to the Python backend's /index-content endpoint.
    const response = await fetch(`${process.env.NEXT_PUBLIC_PYTHON_API_URL}/index-content`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody),
    });

    // Check if the request was successful.
    if (!response.ok) {
      // If not, read the error message from the response and throw an error.
      const errorText = await response.text();
      throw new Error(`Indexing service failed: ${response.status} - ${errorText}`);
    }

    // Parse the JSON response from the backend.
    const result = await response.json();
    return result;
  } catch (error) {
    // Log any errors that occur during the fetch operation.
    console.error(`Error sending item ${item.id.replace(/-/g, '').toUpperCase()} to indexing service:`, error);
    // Re-throw the error so it can be handled by the calling function.
    throw error;
  }
}
