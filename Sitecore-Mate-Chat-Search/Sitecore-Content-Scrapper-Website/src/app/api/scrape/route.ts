// src/app/api/scrape/route.ts
import { NextResponse } from 'next/server';
import { processAndIndexContent } from '../../../index/process-index'; // Import the new indexing function

interface ScrapeRequest {
    envId: number;
    environment: {
        id: number;
        name: string;
        url: string; // Sitecore GraphQL endpoint URL
        apiKey: string; // Sitecore API Key (sc_apikey)
        status: string;
        rootPath: string; // Sitecore root path for this environment
        languages: string[]; // List of languages to scrape for this environment
    };
}

// Define the GraphQL query
const GET_PAGE_CONTENT_QUERY = `
query GetPageContent($path: String!, $language: String!) {
  item(path: $path, language: $language) {
    id
    name
    displayName
    path
    url
    hasChildren
    children {
      name
      path
    }
    ownFields: fields(ownFields: false, excludeStandardFields: true) {
      __typename
      name
      value
    }
    sharedLayout: field(name: "__Renderings") {
      name
      value
    }
    finalLayout: field(name: "__Final Renderings") {
      name
      value
    }
  }
}
`;

// Define the structure for a scraped item to be returned to the UI
interface ScrapedItem {
    id: string;
    name: string;
    path: string;
    url: string;
    language: string;
    content: string; // Aggregated text content from relevant fields
    childrenPaths?: { name: string; path: string }[];
}

/**
 * Fetches content for a given Sitecore item using GraphQL.
 * @param apiUrl The Sitecore GraphQL endpoint URL.
 * @param apiKey The Sitecore API Key.
 * @param path The Sitecore item path.
 * @param language The language version to fetch.
 * @returns The item data from the GraphQL response, or null if an error occurs.
 */
async function fetchSitecoreContent(
    apiUrl: string,
    apiKey: string,
    path: string,
    language: string
): Promise<any | null> {
    try {
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'sc_apikey': apiKey, // Sitecore API Key in header
            },
            body: JSON.stringify({
                query: GET_PAGE_CONTENT_QUERY,
                variables: { path, language },
            }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error(`GraphQL request failed for path: ${path}, language: ${language}, status: ${response.status}, error: ${errorText}`);
            return null;
        }

        const jsonResponse = await response.json();

        if (jsonResponse.errors) {
            console.error(`GraphQL errors for path: ${path}, language: ${language}`, jsonResponse.errors);
            return null;
        }

        return jsonResponse.data?.item || null;

    } catch (error: any) {
        console.error(`Error fetching Sitecore content for path: ${path}, language: ${language}`, error);
        return null;
    }
}

/**
 * Recursively scrapes Sitecore content, extracts relevant data, and sends it via stream.
 * @param currentPath The current Sitecore item path to scrape.
 * @param language The language version.
 * @param sitecoreApiUrl The Sitecore GraphQL endpoint URL.
 * @param sitecoreApiKey The Sitecore API Key.
 * @param controller The ReadableStreamDefaultController to enqueue data.
 * @param encoder The TextEncoder to encode data.
 * @param depth Current recursion depth for logging indentation.
 */
async function scrapeSitecoreItem(
    currentPath: string,
    language: string,
    sitecoreApiUrl: string,
    sitecoreApiKey: string,
    controller: ReadableStreamDefaultController, // Corrected type to ReadableStreamDefaultController
    encoder: TextEncoder, // Encoder for stream
    depth: number = 0
) {
    const indent = '  '.repeat(depth);
    console.log(`${indent}Scraping: ${currentPath} (Language: ${language})`);

    const item = await fetchSitecoreContent(sitecoreApiUrl, sitecoreApiKey, currentPath, language);

    if (!item) {
        console.log(`${indent}  Failed to retrieve item or item is null for path: ${currentPath}`);
        // Send an error event for the UI
        controller.enqueue(encoder.encode(`event: error\ndata: ${JSON.stringify({ path: currentPath, language, message: 'Failed to retrieve item.' })}\n\n`));
        return;
    }

    // Extract page details
    const pageName = item.name;
    const pageUrl = item.url;
    const itemId = item.id;
    const itemPath = item.path;

    // Read ownFields: name/value if text field then do the indexing
    let pageContent = '';
    if (item.ownFields && Array.isArray(item.ownFields)) {
        item.ownFields.forEach((field: any) => {
            if (field.value) {
                switch (field.__typename) {
                    case 'TextField':
                    case 'RichTextField':     // Added RichTextField
                    case 'MultiLineTextField': // Added MultiLineTextField
                    case 'NumberField':
                    case 'DateField':
                        pageContent += `${field.name}: ${field.value}\n`;
                        break;
                    case 'LinkField': // Extract text and URL from LinkField
                        if (field.text) pageContent += `${field.name} Text: ${field.text}\n`;
                        if (field.url) pageContent += `${field.name} URL: ${field.url}\n`;
                        break;
                    case 'ImageField': // Extract alt text from ImageField
                        if (field.alt) pageContent += `${field.name} Alt Text: ${field.alt}\n`;
                        break;
                    // You can add more cases for other field types that contain valuable text
                    // For example, if you have custom field types that store text.
                }
            }
        });
    }

    // Create a ScrapedItem object
    const scrapedItem: ScrapedItem = {
        id: itemId,
        name: pageName,
        path: itemPath,
        url: pageUrl,
        language: language,
        content: pageContent.trim(), // Ensure content is trimmed
        childrenPaths: item.children?.map((child: any) => ({ name: child.name, path: child.path })) || []
    };

    // Send the scraped item to the frontend immediately
    controller.enqueue(encoder.encode(`event: update\ndata: ${JSON.stringify(scrapedItem)}\n\n`));

    // Log the extracted details (for server-side debugging)
    console.log(`${indent}  Page Name: ${pageName}`);
    console.log(`${indent}  Page URL: ${pageUrl}`);
    console.log(`${indent}  Page Content (from text fields - first 100 chars):\n${indent}    ${scrapedItem.content.substring(0, 100).replace(/\n/g, `\n${indent}    `)}...`);

    // --- Call the indexing logic here ---
    // This will now process the scraped item for your vector DB.
    await processAndIndexContent(scrapedItem);

    // Check for children and recurse
    const hasValidLayout = (item.sharedLayout?.value && item.sharedLayout.value.trim() !== '') ||
                           (item.finalLayout?.value && item.finalLayout.value.trim() !== '');

    if (item.hasChildren && item.children && Array.isArray(item.children) && item.children.length > 0 && hasValidLayout) {
        console.log(`${indent}  Has children and valid layout. Descending...`);
        for (const child of item.children) {
            await scrapeSitecoreItem(child.path, language, sitecoreApiUrl, sitecoreApiKey, controller, encoder, depth + 1);
        }
    } else {
        console.log(`${indent}  No children, children array is empty, or no valid layout found. Not descending.`);
    }
}

// Main API route handler
export async function POST(request: Request) {
    const { envId, environment }: ScrapeRequest = await request.json();
    console.log(`Received scrape request for environment ID: ${envId}`, environment);

    if (!environment || !environment.url || !environment.apiKey || !environment.rootPath || !environment.languages || environment.languages.length === 0) {
        return NextResponse.json({ error: 'Missing environment details (URL, API Key, Root Path, or Languages).', message: 'Scraping failed.' }, { status: 400 });
    }

    const initialPath = environment.rootPath;
    const languagesToScrape = environment.languages;

    // Create a ReadableStream to send data to the client
    const encoder = new TextEncoder();
    const readableStream = new ReadableStream({
        async start(controller) {
            try {
                for (const language of languagesToScrape) {
                    console.log(`Starting deep scrape for environment: ${environment.name}, Root: ${initialPath}, Language: ${language}`);
                    // Send a "start" event for each language
                    controller.enqueue(encoder.encode(`event: start\ndata: ${JSON.stringify({ language, rootPath: initialPath })}\n\n`));

                    await scrapeSitecoreItem(initialPath, language, environment.url, environment.apiKey, controller, encoder);

                    // Send a "language_complete" event
                    controller.enqueue(encoder.encode(`event: language_complete\ndata: ${JSON.stringify({ language, rootPath: initialPath })}\n\n`));
                    console.log(`Finished deep scrape for environment: ${environment.name}, Root: ${initialPath}, Language: ${language}`);
                }
                // Send a "complete" event when all scraping is done
                // Ensure the data is a valid JSON string
                controller.enqueue(encoder.encode(`event: complete\ndata: ${JSON.stringify({ message: 'All scraping complete.' })}\n\n`));
                controller.close(); // Close the stream when done
            } catch (error: any) {
                console.error(`Error during deep scraping for environment ${environment.name}:`, error);
                // Send an error event and close the stream
                controller.enqueue(encoder.encode(`event: error\ndata: ${JSON.stringify({ message: `Scraping failed: ${error.message}` })}\n\n`));
                controller.close();
            }
        },
    });

    return new Response(readableStream, {
        headers: {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache, no-transform',
            'Connection': 'keep-alive',
        },
    });
}
