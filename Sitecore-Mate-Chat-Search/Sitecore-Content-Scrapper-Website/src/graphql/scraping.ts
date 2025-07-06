// src/graphql/scraping.ts

import { GraphQLClient, gql } from "graphql-request";
import { XMLParser } from "fast-xml-parser"; // For parsing XML layout fields
import { processAndIndexContent } from "../index/process-index"; // Import the indexing function

// Define the structure of a field as expected by the backend
export interface Field {
  fieldName: string;
  fieldValue: string;
  componentId?: string; // Added to link field to its originating component data source
  renderingUid?:string; // Optional: to capture the rendering UID if needed
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

// New interface for detailed data source information
export interface DataSourceDetail {
  dsId: string;
  renderingName?: string;
  renderingUid?: string;
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

// Helper to extract data source details from layout XML
function extractDataSourceDetails(layoutXml: string): DataSourceDetail[] {
  const parser = new XMLParser({
    ignoreAttributes: false,
    attributeNamePrefix: "@_",
    allowBooleanAttributes: true,
  });
  const jsonObj = parser.parse(layoutXml);
  const details: DataSourceDetail[] = [];

  // Function to recursively find data source details
  const findDetails = (obj: any) => {
    if (obj && typeof obj === "object") {
      // Check for rendering elements (r.d.r structure)
      if (obj.r && obj.r.d && obj.r.d.r) {
        const renderings = Array.isArray(obj.r.d.r) ? obj.r.d.r : [obj.r.d.r];
        for (const rendering of renderings) {
          const dsId = rendering["@_ds"] || rendering["@_s:ds"]; // Check for both
          if (dsId) {
            details.push({
              dsId: dsId,
              renderingName: rendering["@_name"] || undefined, // Access rendering name
              renderingUid: rendering["@_uid"] || undefined, // Access rendering UID
            });
          }
          // Recursively check children renderings
          if (rendering.d && rendering.d.r) {
            findDetails(rendering.d.r);
          }
        }
      }
      // General recursion for other objects/arrays
      for (const key in obj) {
        if (Object.prototype.hasOwnProperty.call(obj, key)) {
          findDetails(obj[key]);
        }
      }
    }
  };

  findDetails(jsonObj);
  // Filter out any entries without a valid dsId
  return details.filter((detail) => detail.dsId);
}

function mapField(fieldName: string, fieldValue: string, componentId?: string, renderingUid?: string): Field {
  return {
    fieldName: fieldName,
    fieldValue: fieldValue,
    componentId: componentId,
    renderingUid: renderingUid, // Optional: include rendering UID if available
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
  const client = new GraphQLClient(graphqlEndpoint, {
    headers: {
      sc_apikey: apiKey,
    },
  });

  // Avoid re-processing items to prevent infinite loops in circular references
  if (processedItemIds.has(currentPath)) {
    console.log(`Skipping already processed item: ${currentPath}`);
    return;
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
      return;
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

    let aggregatedPageValue = "";
    const seenPageFieldValues = new Set<string>();

    item.ownFields
      .filter((field: any) => field.value && field.__typename === "TextField")
      .forEach((field: any) => {
        if (!seenPageFieldValues.has(field.value)) {
          aggregatedPageValue += `${field.value} `;
          seenPageFieldValues.add(field.value);
        }
      });

    const currentItemFields: Field[] = [];
    if (aggregatedPageValue.trim().length > 0) {
      currentItemFields.push(
        mapField(
          "Page.AllFields",
          aggregatedPageValue.trim(),
          item.id
        )
      );
    }

    // Changed to DataSourceDetail[]
    const dataSourceCollection: DataSourceDetail[] = []; 

    // Extract data source details from sharedLayout
    if (item.sharedLayout?.value) {
      const details = extractDataSourceDetails(item.sharedLayout.value);
      dataSourceCollection.push(...details);
    }
    // Extract data source details from finalLayout
    if (item.finalLayout?.value) {
      const details = extractDataSourceDetails(item.finalLayout.value);
      dataSourceCollection.push(...details);
    }

    // Process data sources: recursively scrape them and collect their fields
    const componentFieldsFromDataSources: Field[] = [];
    // Use a Set to store unique dsIds to avoid processing the same data source multiple times
    const processedDataSourceIds = new Set<string>();

    for (const dsDetail of dataSourceCollection) { // Loop over DataSourceDetail
      if (dsDetail.dsId === item.id) {
        // Avoid self-referencing data sources
        continue;
      }
      if (processedDataSourceIds.has(dsDetail.dsId)) {
        continue; // Skip if already processed
      }
      processedDataSourceIds.add(dsDetail.dsId);

      try {
        const dsResponse: any = await client.request(GetDataSourceContent, {
          id: dsDetail.dsId, // Use dsId from the detail object
          language: language,
        });

        const dsItem = dsResponse?.item;
        if (dsItem) {
          const componentName = dsItem.displayName || dsItem.name || dsItem.id;
          let aggregatedComponentValue = "";
          const seenComponentFieldValues = new Set<string>();

          dsItem.ownFields
            .filter((field: any) => field.value && field.__typename === "TextField")
            .forEach((field: any) => {
              if (!seenComponentFieldValues.has(field.value)) {
                aggregatedComponentValue += `${field.value} `;
                seenComponentFieldValues.add(field.value);
              }
            });

          if (aggregatedComponentValue.trim().length > 0) {
            componentFieldsFromDataSources.push(
              mapField(
                `${componentName}.AllFields`,
                aggregatedComponentValue.trim(),
                item.id,
                `${componentName}.AllFields.${item.id}.${dsDetail.dsId}.${dsDetail.renderingUid}` // Optional: include rendering UID if available
              )
            );
          }
        }
      } catch (dsError: any) {
        console.error(`Error scraping data source ${dsDetail.dsId}:`, dsError.message);
      }
    }

    const itemUrl = item.url || "";

    if (itemType === "page") {
      const pageToProcess: Page = {
        pageId: item.id,
        pagePath: item.path,
        pageTitle: item.displayName || item.name,
        language: language,
        fields: [...currentItemFields, ...componentFieldsFromDataSources],
        components: [],
        itemType: itemType,
        url: itemUrl,
      };
      await processAndIndexContent({
        pages: [pageToProcess],
        environment: environmentName,
      });
      console.log(`Indexed page: ${pageToProcess.pagePath}`);
    } else {
      if (!parentPageId) {
        const componentToProcess: Page = {
          pageId: item.id,
          pagePath: item.path,
          pageTitle: item.displayName || item.name,
          language: language,
          fields: currentItemFields,
          components: [],
          itemType: itemType,
          url: itemUrl,
        };
        await processAndIndexContent({
          pages: [componentToProcess],
          environment: environmentName,
        });
        console.log(`Indexed component: ${componentToProcess.pagePath}`);
      }
    }

    if (item.hasChildren && item.children && item.children.length > 0) {
      for (const child of item.children) {
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
  } catch (error: any) {
    console.error(`Error scraping ${currentPath}:`, error.message);
    throw error;
  }
}