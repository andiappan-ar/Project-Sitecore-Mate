// src/app/api/scrape/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { processAndIndexContent } from '@/index/process-index';

// In-memory status tracking object. This will hold the logs and status for each scraping job.
const scrapingStatus: { [key: string]: { status: string; log: string[] } } = {};

// Helper function to recursively extract text from various field types
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

/**
 * Scrapes a Sitecore item using a GraphQL query.
 */
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
        displayName
        path
        url 
        hasChildren
        children {
          name
          path
        }
        ownFields: fields(ownFields: true, excludeStandardFields: true) {
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

  try {
    const response = await fetch(graphqlEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'sc_apikey': apiKey,
      },
      body: JSON.stringify({
        query,
        variables: { itemPath, language },
      }),
      cache: 'no-store',
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`GraphQL request failed with status ${response.status}: ${errorText}`);
      return null;
    }

    const jsonResponse = await response.json();
    const item = jsonResponse.data?.item;

    if (!item) {
      if (jsonResponse.errors) {
        console.error('--- GraphQL Errors from Sitecore ---');
        console.error(JSON.stringify(jsonResponse.errors, null, 2));
      }
      return null;
    }

    let content = extractTextFromFields(item.ownFields);
    if (item.sharedLayout && item.sharedLayout.value) {
        content += `${item.sharedLayout.name}: ${item.sharedLayout.value}\n`;
    }
    if (item.finalLayout && item.finalLayout.value) {
        content += `${item.finalLayout.name}: ${item.finalLayout.value}\n`;
    }

    return {
      id: item.id.replace(/-/g, ''),
      name: item.name,
      path: item.path,
      url: item.url,
      language: language,
      content: content.trim(),
      children: item.hasChildren ? item.children.map((child: any) => ({ name: child.name, path: child.path })) : [],
    };
  } catch (error) {
    console.error('Error during GraphQL fetch:', error);
    return null;
  }
}


/**
 * Recursively scrapes a Sitecore item and its children.
 */
async function scrapeSitecoreItem(
  path: string,
  language: string,
  environment: any,
  apiKey: string,
  environmentId: string
) {
  const log = (message: string) => {
    console.log(message);
    if (scrapingStatus[environmentId]) {
      scrapingStatus[environmentId].log.push(message);
    }
  };

  log(`Scraping: ${path} (Language: ${language})`);

  try {
    const itemData = await scrapeGraphQL(path, language, environment.url, apiKey);

    if (itemData) {
      log(`  Page Name: ${itemData.name}`);
      log(`  Page URL: ${itemData.url}`);
      if (itemData.content) {
        log(`  Page Content (from text fields - first 100 chars):\n    ${itemData.content.substring(0, 100).replace(/\n/g, '\n    ')}`);
        await processAndIndexContent(itemData, environment);
      } else {
        log('  No content fields found to index.');
      }

      if (itemData.children && itemData.children.length > 0) {
        log('  Has children to scrape. Descending...');
        for (const child of itemData.children) {
          await scrapeSitecoreItem(child.path, language, environment, apiKey, environmentId);
        }
      } else {
        log('  No children to scrape.');
      }
    } else {
      log(`  No data returned from GraphQL for path: ${path}`);
    }
  } catch (error: any) {
    const errorMessage = `  Error scraping ${path}: ${error.message}`;
    log(errorMessage);
    throw new Error(errorMessage);
  }
}

/**
 * FIX: Handles GET requests to fetch the status of a scraping job.
 * This function was missing, causing the 405 Method Not Allowed error.
 */
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const environmentId = searchParams.get('environmentId');

  if (!environmentId) {
    return NextResponse.json({ error: 'Environment ID is required' }, { status: 400 });
  }

  const status = scrapingStatus[environmentId];

  if (!status) {
    return NextResponse.json({ error: 'Scraping process not found for this environment ID.' }, { status: 404 });
  }

  return NextResponse.json(status);
}


/**
 * Handles POST requests to start the scraping process.
 */
export async function POST(request: NextRequest) {
  try {
    const { environment } = await request.json();
    const environmentId = environment.id;

    if (!environmentId) {
      return NextResponse.json({ error: 'Environment ID is required' }, { status: 400 });
    }

    if (scrapingStatus[environmentId] && scrapingStatus[environmentId].status === 'In Progress') {
        return NextResponse.json({ message: 'A scraping process is already in progress for this environment.' }, { status: 409 });
    }

    console.log(`Received scrape request for environment ID: ${environmentId}`, environment);

    scrapingStatus[environmentId] = {
      status: 'In Progress',
      log: [`Scraping started for environment: ${environment.name}`],
    };

    (async () => {
      try {
        for (const lang of environment.languages) {
          await scrapeSitecoreItem(environment.rootPath, lang, environment, environment.apiKey, environmentId);
        }
        scrapingStatus[environmentId].status = 'Completed';
        scrapingStatus[environmentId].log.push('Scraping completed successfully.');
        console.log(`Deep scraping finished for environment: ${environment.name}`);
      } catch (error: any) {
        const errorMessage = `Error during deep scraping for environment ${environment.name}: ${error.message}`;
        console.error(errorMessage, error);
        scrapingStatus[environmentId].status = 'Failed';
        scrapingStatus[environmentId].log.push('Scraping failed.');
        scrapingStatus[environmentId].log.push(errorMessage);
      }
    })();

    return NextResponse.json({
      message: 'Scraping process started.',
      environmentId: environmentId,
    });
  } catch (error: any) {
    console.error('Error in /api/scrape:', error);
    return NextResponse.json({ error: 'Failed to start scraping process', details: error.message }, { status: 500 });
  }
}
