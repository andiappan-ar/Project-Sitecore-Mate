// src/graphql/scraping.ts

import { GraphQLClient, gql } from "graphql-request";
import { XMLParser } from "fast-xml-parser"; // For parsing XML layout fields
import { processAndIndexContent } from "../index/process-index"; // Import the indexing function

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
  itemType?: "page" | "component"; // 'page' for content pages, 'component' for data source items
  url: string; // Added to capture the public-facing URL
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
      url # Fetch the URL
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
    item(path: $id, language: $language) {
      # Using path as ID as per your query structure
      id
      name
      displayName
      path
      url # Fetch the URL
      hasChildren
      ownFields: fields(ownFields: false, excludeStandardFields: true) {
        __typename
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
    if (obj && typeof obj === "object") {
      // Check for rendering elements
      if (obj.r && obj.r.d && obj.r.d.r) {
        const renderings = Array.isArray(obj.r.d.r) ? obj.r.d.r : [obj.r.d.r];
        for (const rendering of renderings) {
          // Check for both '@_ds' and '@_s:ds' (with namespace prefix)
          if (rendering["@_ds"]) {
            ids.push(rendering["@_ds"]);
          } else if (rendering["@_s:ds"]) {
            // Added check for namespace prefixed attribute
            ids.push(rendering["@_s:ds"]);
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
  return ids.filter((id) => id); // Filter out any empty or null IDs
}

function mapField(fieldName, fieldValue, componentId) {
  return {
    fieldName: fieldName,
    fieldValue: fieldValue,
    componentId: componentId,
  };
}

/**
 * Recursively scrapes content from Sitecore and indexes it page by page.
 * @param graphqlEndpoint The Sitecore GraphQL endpoint URL.
 * @param apiKey The Sitecore API key.
 * @param currentPath The current path to scrape.
 * @param language The language to scrape.
 * @param environmentName The name of the environment for the backend payload.
 * @param processedItemIds Set to keep track of processed item IDs to avoid infinite loops.
 * @param parentPageId Optional: The ID of the primary content page that this item's content should be associated with.
 * @returns A promise that resolves when scraping and indexing for the current path is complete.
 */
export async function scrape(
  graphqlEndpoint: string,
  apiKey: string,
  currentPath: string,
  language: string,
  environmentName: string,
  processedItemIds: Set<string> = new Set(),
  parentPageId?: string // New parameter: ID of the page that owns this content
): Promise<void> {
  // Changed return type to void
  const client = new GraphQLClient(graphqlEndpoint, {
    headers: {
      sc_apikey: apiKey,
    },
  });

  // Avoid re-processing items to prevent infinite loops in circular references
  if (processedItemIds.has(currentPath)) {
    console.log(`Skipping already processed item: ${currentPath}`);
    return; // No return value needed
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
      return; // No return value needed
    }

    // Determine if this item is a primary content page (has layout)
    const isPrimaryContentPage =
      item.sharedLayout?.value || item.finalLayout?.value;
    const itemType: "page" | "component" = isPrimaryContentPage
      ? "page"
      : "component";

    console.log(
      `Determined itemType for ${item.displayName || item.name} (${
        item.id
      }): ${itemType}`
    );

    const currentItemFields = item.ownFields
      .filter(field => field.value && field.__typename === "TextField")
      .map((field) =>
        mapField(
          "PageField." + field.name, // fieldname
          field.value, // fieldvalue
          parentPageId ? item.id : undefined // componentId
        )
      );

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
      if (dsId === item.id) {
        // Avoid self-referencing data sources
        continue;
      }
      try {
        const dsResponse: any = await client.request(GetDataSourceContent, {
          id: dsId,
          language: language,
        });

        const dsItem = dsResponse?.item;
        if (dsItem) {
          const componentName = dsItem.displayName || dsItem.name || dsItem.id; // Use display name, then name, then ID
          let aggregatedComponentValue = "";

          // Aggregate all text field values from the component
          dsItem.ownFields
            .filter(field => field.value && field.__typename === "TextField")
            .forEach((field) => {
              aggregatedComponentValue += `${field.value} `; // Concatenate values with a space
            });

          if (aggregatedComponentValue.trim().length > 0) {
            componentFieldsFromDataSources.push(
              mapField(
                `${componentName}.AllFields`, // New fieldName: ComponentName.AllFields
                aggregatedComponentValue.trim(), // Aggregated fieldValue
                item.id // componentId (parent page's ID)
              )
            );
          }
        }
      } catch (dsError: any) {
        console.error(`Error scraping data source ${dsId}:`, dsError.message);
      }
    }

    // Determine the URL for the current item. Use item.url directly.
    const itemUrl = item.url || "";

    // If the current item is a primary content page, index it immediately
    if (itemType === "page") {
      const pageToProcess: Page = {
        pageId: item.id,
        pagePath: item.path,
        pageTitle: item.displayName || item.name,
        language: language,
        fields: [...currentItemFields, ...componentFieldsFromDataSources], // Combine page's own fields with component fields
        components: [], // This array is now always empty
        itemType: itemType,
        url: itemUrl, // Include the URL here
      };
      await processAndIndexContent({
        pages: [pageToProcess],
        environment: environmentName,
      });
      console.log(`Indexed page: ${pageToProcess.pagePath}`);
    } else {
      // If it's a component data source item itself (not a primary page),
      // and it has no parentPageId (meaning it's a top-level component being scraped directly), index it.
      if (!parentPageId) {
        const componentToProcess: Page = {
          pageId: item.id,
          pagePath: item.path,
          pageTitle: item.displayName || item.name,
          language: language,
          fields: currentItemFields, // Only its own fields
          components: [],
          itemType: itemType,
          url: itemUrl, // Include the URL here
        };
        await processAndIndexContent({
          pages: [componentToProcess],
          environment: environmentName,
        });
        console.log(`Indexed component: ${componentToProcess.pagePath}`);
      }
    }

    // Recursively scrape children pages
    if (item.hasChildren && item.children && item.children.length > 0) {
      for (const child of item.children) {
        // Recursively scrape children. The indexing will happen inside the recursive call
        // if the child is a primary content page.
        // The parentPageId for children is the current item's ID if it's a page.
        await scrape(
          graphqlEndpoint,
          apiKey,
          child.path,
          language,
          environmentName,
          processedItemIds,
          item.id
        );
      }
    }

    // No return value needed as indexing is done incrementally
  } catch (error: any) {
    console.error(`Error scraping ${currentPath}:`, error.message);
    throw error;
  }
}
