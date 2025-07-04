// src/graphql/scraping.ts

// Define types for Sitecore environment, matching the interface in page.tsx
interface SitecoreEnvironment {
    id: string; // Changed to string for consistency with localStorage and backend
    name: string;
    url: string;
    apiKey: string;
    status: string;
    rootPath: string;
    languages: string[];
}

// Define the structure for a scraped item (must match the backend)
interface ScrapedItem {
    id: string;
    name: string;
    path: string;
    url: string;
    language: string;
    content: string; // Aggregated text content from relevant fields
    childrenPaths?: { name: string; path: string }[];
}

// Define specific data types for each type of scrape update event
interface ScrapeStartData {
    language: string;
    rootPath: string;
}

interface ScrapeLanguageCompleteData {
    language: string;
}

interface ScrapeCompleteErrorData {
    message: string;
}

// Define a union type for the 'data' parameter in handleScrapeUpdate
type ScrapeUpdateData = ScrapedItem | ScrapeStartData | ScrapeLanguageCompleteData | ScrapeCompleteErrorData;

// Define a callback type for live updates, using the specific union type
type ScrapeUpdateCallback = (type: 'update' | 'start' | 'language_complete' | 'complete' | 'error', data: ScrapeUpdateData) => void;

/**
 * Initiates the Sitecore content scraping process via SSE, providing live updates.
 * @param environment The Sitecore environment details to scrape.
 * @param onUpdate Callback function to receive live updates.
 * @param environmentId The ID of the environment being scraped, used for backend collection selection.
 * @returns A promise that resolves when the stream ends (either successfully or with an error).
 */
export const startScrapingApi = async (environment: SitecoreEnvironment, onUpdate: ScrapeUpdateCallback, environmentId: string): Promise<ApiResponse> => {
    try {
        // The /api/scrape route in Next.js will then forward this to the Python backend.
        const response = await fetch('/api/scrape', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            // Pass the entire environment object and the specific environmentId
            body: JSON.stringify({
                envId: environment.id, // This might be redundant if environmentId is the primary identifier
                environment: { // Ensure all environment details are passed as expected by your Next.js API route
                    id: environment.id,
                    name: environment.name,
                    url: environment.url,
                    apiKey: environment.apiKey,
                    status: environment.status,
                    rootPath: environment.rootPath,
                    languages: environment.languages,
                },
                environmentId: environmentId // Crucial for backend ChromaDB collection selection
            })
        });

        if (!response.ok || !response.body) {
            const errorText = await response.text();
            onUpdate('error', { message: `Failed to start scraping stream: ${errorText}` });
            return { success: false, message: `Failed to start scraping stream: ${errorText}`, error: errorText };
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) {
                onUpdate('complete', 'Stream finished.');
                break;
            }

            buffer += decoder.decode(value, { stream: true });

            let newlineIndex;
            while ((newlineIndex = buffer.indexOf('\n\n')) !== -1) {
                const eventChunk = buffer.substring(0, newlineIndex);
                buffer = buffer.substring(newlineIndex + 2);

                const lines = eventChunk.split('\n');
                let eventType = 'message';
                let eventData: ScrapeUpdateData = { message: 'Unknown event data' } as ScrapeCompleteErrorData; // Default to a safe type

                for (const line of lines) {
                    if (line.startsWith('event: ')) {
                        eventType = line.substring('event: '.length);
                    } else if (line.startsWith('data: ')) {
                        try {
                            eventData = JSON.parse(line.substring('data: '.length));
                        } catch (e) {
                            console.error('Error parsing SSE data:', e);
                            eventData = { message: `Parsing failed: ${line.substring('data: '.length)}` };
                        }
                    }
                }
                onUpdate(eventType as 'update' | 'start' | 'language_complete' | 'complete' | 'error', eventData);
            }
        }

        return { success: true, message: 'Scraping stream completed.' };

    } catch (error: any) {
        console.error('Error during SSE scraping:', error);
        onUpdate('error', { message: `An error occurred during scraping: ${error.message}` });
        return { success: false, message: `An error occurred during scraping: ${error.message}`, error: error.message };
    }
};

/**
 * Initiates the Sitecore index update process by calling the Next.js /api/update-index endpoint.
 * @param environment The Sitecore environment details for which to update indexes.
 * @param environmentId The ID of the environment being updated, used for backend collection selection.
 * @returns A promise that resolves to an ApiResponse indicating success or failure.
 */
export const updateIndexesApi = async (environment: SitecoreEnvironment, environmentId: string): Promise<ApiResponse> => {
    try {
        // The /api/update-index route in Next.js will then forward this to the Python backend.
        const response = await fetch('/api/update-index', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            // Pass the specific environmentId
            body: JSON.stringify({
                envId: environment.id, // This might be redundant if environmentId is the primary identifier
                environment: { // Ensure all environment details are passed as expected by your Next.js API route
                    id: environment.id,
                    name: environment.name,
                    url: environment.url,
                    apiKey: environment.apiKey,
                    status: environment.status,
                    rootPath: environment.rootPath,
                    languages: environment.languages,
                },
                environmentId: environmentId // Crucial for backend ChromaDB collection selection
            })
        });

        const result = await response.json();

        if (response.ok) {
            return { success: true, message: result.message };
        } else {
            return { success: false, message: `Index update failed: ${result.error || 'Unknown error'}`, error: result.error };
        }
    } catch (error: any) {
        console.error('Error calling /api/update-index:', error);
        return { success: false, message: `An error occurred while initiating index update: ${error.message}`, error: error.message };
    }
};
