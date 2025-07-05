// src/app/api/scrape/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { processAndIndexContent } from '@/index/process-index';

const scrapingStatus: { [key: string]: { status: string; log: string[] } } = {};

function extractTextFromFields(fields: any[]): string {
  let combinedText = '';
  if (!fields) return '';
  fields.forEach((field) => {
    if (field && field.value) {
      combinedText += `${field.name}: ${field.value}\n`;
    }
  });
  return combinedText;
}

async function scrapeGraphQL(
  itemPath: string,
  language: string,
  graphqlEndpoint: string,
  apiKey: string
) {
  const query = `
    query GetItemData($itemPath: String!, $language: String!) {
      item(path: $itemPath, language: $language) {
        id
        name
        path
        url 
        hasChildren
        children {
          name
          path
        }
        ownFields: fields(ownFields: true, excludeStandardFields: true) {
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

  try {
    const response = await fetch(graphqlEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'sc_apikey': apiKey },
      body: JSON.stringify({ query, variables: { itemPath, language } }),
      cache: 'no-store',
    });

    if (!response.ok) {
      console.error(`GraphQL request failed: ${response.statusText}`);
      return null;
    }

    const jsonResponse = await response.json();
    const item = jsonResponse.data?.item;

    if (!item) {
      return null;
    }
    
    const content = extractTextFromFields(item.ownFields);

    return {
      id: item.id.replace(/-/g, ''),
      name: item.name,
      path: item.path,
      url: item.url,
      language: language,
      content: content.trim(),
      children: item.hasChildren ? item.children.map((child: any) => ({ name: child.name, path: child.path })) : [],
      sharedLayout: item.sharedLayout,
      finalLayout: item.finalLayout,
    };
  } catch (error) {
    console.error('Error during GraphQL fetch:', error);
    return null;
  }
}

/**
 * Recursively scrapes an item and its descendants, aggregating component content
 * and processing pages as they are found.
 * @returns The aggregated content of any non-page items found.
 */
async function scrapeAndAggregate(
  path: string,
  language: string,
  environment: any,
  log: (message: string) => void
): Promise<string> {
  log(`Attempting to scrape: ${path} (${language})`);
  const itemData = await scrapeGraphQL(path, language, environment.url, environment.apiKey);

  if (!itemData) {
    log(`  INFO: No data returned from GraphQL for path: ${path}`);
    return ""; // Return empty string if item not found
  }

  // Check if the current item is a "page" by verifying it has layout details.
  const isPage = !!(itemData.sharedLayout?.value || itemData.finalLayout?.value);

  let aggregatedComponentContent = "";

  // If the current item is NOT a page, its own content is considered component content.
  if (!isPage) {
    log(`  INFO: Found component: ${itemData.name}. Aggregating its content.`);
    aggregatedComponentContent += itemData.content + "\n\n";
  }

  // Recurse into children to gather their component content.
  if (itemData.children && itemData.children.length > 0) {
    for (const child of itemData.children) {
      // The content returned from the recursive call is appended.
      aggregatedComponentContent += await scrapeAndAggregate(child.path, language, environment, log);
    }
  }

  // If the current item IS a page, we process it now.
  if (isPage) {
    log(`  SUCCESS: Identified as a page. Aggregating content for: ${itemData.name}`);
    
    // The final content for indexing is the page's own content plus all descendant component content.
    const finalContentForIndexing = itemData.content + "\n\n--- Contained Components ---\n" + aggregatedComponentContent;

    // Create a new item object with the aggregated content.
    const pageToIndex = {
        ...itemData,
        content: finalContentForIndexing,
    };

    // Send the complete page object for indexing.
    await processAndIndexContent(pageToIndex, environment);
    
    // Since this page and all its components have been processed and indexed,
    // we return an empty string. This stops its content from being passed further up the recursion tree.
    return ""; 
  } else {
    // If this is not a page, we return all content found under it to be handled by a parent page.
    return aggregatedComponentContent;
  }
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const environmentId = searchParams.get('environmentId');
  if (!environmentId) return NextResponse.json({ error: 'Environment ID is required' }, { status: 400 });
  const status = scrapingStatus[environmentId];
  if (!status) return NextResponse.json({ error: 'Process not found.' }, { status: 404 });
  return NextResponse.json(status);
}

export async function POST(request: NextRequest) {
  try {
    const { environment } = await request.json();
    const environmentId = environment.id;

    if (!environmentId) return NextResponse.json({ error: 'Environment ID is required' }, { status: 400 });
    if (scrapingStatus[environmentId]?.status === 'In Progress') return NextResponse.json({ message: 'Scraping already in progress.' }, { status: 409 });

    scrapingStatus[environmentId] = { status: 'In Progress', log: [`Scraping started for ${environment.name}`] };
    
    const log = (message: string) => {
        console.log(message);
        if (scrapingStatus[environmentId]) {
            scrapingStatus[environmentId].log.push(message);
        }
    };

    (async () => {
      try {
        for (const lang of environment.languages) {
          // Start the new aggregation process from the root path.
          await scrapeAndAggregate(environment.rootPath, lang, environment, log);
        }
        scrapingStatus[environmentId].status = 'Completed';
        log('Scraping completed successfully.');
      } catch (error: any) {
        const errorMessage = `Error during scraping: ${error.message}`;
        log(errorMessage);
        console.error(errorMessage, error);
        scrapingStatus[environmentId].status = 'Failed';
      }
    })();

    return NextResponse.json({ message: 'Scraping process started.', environmentId });
  } catch (error: any) {
    return NextResponse.json({ error: 'Failed to start scraping process', details: error.message }, { status: 500 });
  }
}
