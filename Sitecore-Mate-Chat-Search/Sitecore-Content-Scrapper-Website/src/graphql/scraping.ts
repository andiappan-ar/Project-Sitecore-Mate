// src/graphql/scraping.ts

import { GraphQLClient, gql } from 'graphql-request';
import { XMLParser } from 'fast-xml-parser'; // For parsing XML layout fields

// Define the structure of a field as expected by the backend
export interface Field {
  fieldName: string;
  fieldValue: string;
  componentId?: string; // Added to link field to its originating component data source
}

// Define the structure of a component as expected by the backend
// This interface will still exist but the 'components' array in Page will be empty
// as component fields are now flattened into the Page's fields.
export interface Component {
  componentId: string;
  componentName: string;
  fields: Field[]; // These fields will be moved to the parent Page's fields
}

// Define the structure of a page/item as expected by the backend
export interface Page {
  pageId: string;
  pagePath: string;
  pageTitle: string;
  language: string;
  fields: Field[]; // This will now contain both page's own fields and component fields
  components: Component[]; // This array will now always be empty as component fields are flattened
  itemType?: 'page' | 'component'; // 'page' for content pages, 'component' for data source items
}

// Define the overall payload structure for the indexing service
export interface ContentPayload {
  pages: Page[];
  environment: string;
}

// GraphQL query to get page content and layout information
const GetPageContent = gql`
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
        id # Added id to children for potential future use
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

// GraphQL query to get data source content
const GetDataSourceContent = gql`
  query GetDataSourceContent($id: String!, $language: String!) {
    item(path: $id, language: $language) { # Using path as ID as per your query structure
      id
      name
      displayName
      path
      url
      hasChildren
      ownFields: fields(ownFields: false, excludeStandardFields: true) {
        name
        value
      }
    }
  }
`;

// Helper to extract data source IDs from layout XML
function extractDataSourceIds(layoutXml: string): string[] {
  const parser = new XMLParser({
    ignoreAttributes: false,
    attributeNamePrefix: "@_",
    allowBooleanAttributes: true,
  });
  const jsonObj = parser.parse(layoutXml);
  const ids: string[] = [];

  // Function to recursively find data source IDs
  const findIds = (obj: any) => {
    if (obj && typeof obj === 'object') {
      // Check for rendering elements
      if (obj.r && obj.r.d && obj.r.d.r) {
        const renderings = Array.isArray(obj.r.d.r) ? obj.r.d.r : [obj.r.d.r];
        for (const rendering of renderings) {
          // Check for both '@_ds' and '@_s:ds' (with namespace prefix)
          if (rendering['@_ds']) {
            ids.push(rendering['@_ds']);
          } else if (rendering['@_s:ds']) { // Added check for namespace prefixed attribute
            ids.push(rendering['@_s:ds']);
          }
          // Recursively check children renderings
          if (rendering.d && rendering.d.r) {
            findIds(rendering.d.r);
          }
        }
      }
      // General recursion for other objects/arrays
      for (const key in obj) {
        if (Object.prototype.hasOwnProperty.call(obj, key)) {
          findIds(obj[key]);
        }
      }
    }
  };

  findIds(jsonObj);
  return ids.filter(id => id); // Filter out any empty or null IDs
}

/**
 * Recursively scrapes content from Sitecore, including pages and their associated data sources.
 * @param graphqlEndpoint The Sitecore GraphQL endpoint URL.
 * @param apiKey The Sitecore API key.
 * @param currentPath The current path to scrape.
 * @param language The language to scrape.
 * @param environmentName The name of the environment for the backend payload.
 * @param scrapedPages Accumulator for all scraped page/component data.
 * @param processedItemIds Set to keep track of processed item IDs to avoid infinite loops.
 * @param parentPageId Optional: The ID of the primary content page that this item's content should be associated with.
 * @returns A promise that resolves to the structured content payload.
 */
export async function scrape(
  graphqlEndpoint: string,
  apiKey: string,
  currentPath: string,
  language: string,
  environmentName: string,
  scrapedPages: Page[] = [],
  processedItemIds: Set<string> = new Set(),
  parentPageId?: string // New parameter: ID of the page that owns this content
): Promise<ContentPayload> {
  const client = new GraphQLClient(graphqlEndpoint, {
    headers: {
      sc_apikey: apiKey,
    },
  });

  // Avoid re-processing items to prevent infinite loops in circular references
  if (processedItemIds.has(currentPath)) {
    console.log(`Skipping already processed item: ${currentPath}`);
    return { pages: scrapedPages, environment: environmentName };
  }
  processedItemIds.add(currentPath);

  try {
    console.log(`Scraping for ${currentPath} in ${language}...`);
    const pageResponse: any = await client.request(GetPageContent, {
      path: currentPath,
      language: language,
    });

    const item = pageResponse?.item;

    if (!item) {
      console.warn(`No item found for path: ${currentPath}`);
      return { pages: scrapedPages, environment: environmentName };
    }

    // Determine if this item is a primary content page (has layout)
    const isPrimaryContentPage = item.sharedLayout?.value || item.finalLayout?.value;
    const itemType: 'page' | 'component' = isPrimaryContentPage ? 'page' : 'component';
    
    console.log(`Determined itemType for ${item.displayName || item.name} (${item.id}): ${itemType}`);

    const currentItemFields: Field[] = item.ownFields.map((field: any) => ({
      fieldName: "PageField."+field.name,
      fieldValue: field.value || '',
      componentId: parentPageId ? item.id : undefined, // If this is a component data source, set its ID
    }));

    const dataSourceIds: string[] = [];

    // Extract data source IDs from sharedLayout
    if (item.sharedLayout?.value) {
      const ids = extractDataSourceIds(item.sharedLayout.value);
      dataSourceIds.push(...ids);
    }
    // Extract data source IDs from finalLayout
    if (item.finalLayout?.value) {
      const ids = extractDataSourceIds(item.finalLayout.value);
      dataSourceIds.push(...ids);
    }

    // Process data sources: recursively scrape them and collect their fields
    const componentFieldsFromDataSources: Field[] = [];
    for (const dsId of [...new Set(dataSourceIds)]) {
      if (dsId === item.id) { // Avoid self-referencing data sources
          continue;
      }
      try {
        const dsResponse: any = await client.request(GetDataSourceContent, {
          id: dsId,
          language: language,
        });

        const dsItem = dsResponse?.item;
        if (dsItem) {
          // Recursively scrape the data source, passing its ID as the parentPageId
          // This call will return a ContentPayload, but we only care about the fields it collects
          // We pass the current 'item.id' as the parentPageId for the data source's content
          // const tempScrapedPages: Page[] = []; // Temporary array for recursive call
          // await scrape(graphqlEndpoint, apiKey, dsItem.path, language, environmentName, tempScrapedPages, processedItemIds, item.id);
          
          // Extract fields from the tempScrapedPages (which will contain only the dsItem as a 'page' with its fields)
          // and add the componentId to them.
          // if (tempScrapedPages.length > 0) {
            //const dsPage = tempScrapedPages[0]; // Assuming only one item is returned for a data source ID
            const componentName = "{"+dsItem.path+"}"; // Use pageTitle as component name, fallback to ID
            dsItem.ownFields.forEach(field => {
              componentFieldsFromDataSources.push({
                fieldName: `${componentName}.${field.name}`, // Prepend component name to fieldName
                fieldValue: field.value,
                componentId: item.id, // This is the ID of the data source item itself
              });
            });
          // }
        }
      } catch (dsError: any) {
        console.error(`Error scraping data source ${dsId}:`, dsError.message);
      }
    }

    // If the current item is a primary content page, add it to scrapedPages
    if (itemType === 'page') {
      scrapedPages.push({
        pageId: item.id,
        pagePath: item.path,
        pageTitle: item.displayName || item.name,
        language: language,
        fields: [...currentItemFields, ...componentFieldsFromDataSources], // Combine page's own fields with component fields
        components: [], // This array is now always empty
        itemType: itemType,
      });
    } else {
      // If it's a component data source item itself (not a primary page),
      // and it has a parentPageId (meaning it was called recursively from a page),
      // its fields are already collected and pushed into componentFieldsFromDataSources by its parent.
      // We don't push it as a top-level page here.
      // However, if it's a standalone component item being scraped directly (no parentPageId),
      // we still need to add its content. This case might need further refinement based on your exact content structure.
      // For now, if it's a component and has no parentPageId, it will be treated as a top-level page for indexing.
      if (!parentPageId) {
        scrapedPages.push({
          pageId: item.id,
          pagePath: item.path,
          pageTitle: item.displayName || item.name,
          language: language,
          fields: currentItemFields, // Only its own fields
          components: [],
          itemType: itemType,
        });
      }
    }


    // Recursively scrape children pages
    if (item.hasChildren && item.children && item.children.length > 0) {
      for (const child of item.children) {
        // Recursively scrape children. If a child is a primary content page,
        // it will be added to scrapedPages. If it's a component data source,
        // its fields will be collected by its parent (if the parent is a page).
        // The parentPageId for children is the current item's ID if it's a page.
        await scrape(graphqlEndpoint, apiKey, child.path, language, environmentName, scrapedPages, processedItemIds, item.id);
      }
    }

    return { pages: scrapedPages, environment: environmentName };

  } catch (error: any) {
    console.error(`Error scraping ${currentPath}:`, error.message);
    throw error;
  }
}
