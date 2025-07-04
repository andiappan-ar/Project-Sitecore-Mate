// src/graphql/scraping.ts

// Define types for Sitecore environment, matching the interface in page.tsx
interface SitecoreEnvironment {
    id: number;
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

// Define a common return type for API operations (for updateIndexesApi, startScrapingApi will be different)
interface ApiResponse {
    success: boolean;
    message: string;
    error?: string;
}

// Define a callback type for live updates
type ScrapeUpdateCallback = (type: 'update' | 'start' | 'language_complete' | 'complete' | 'error', data: any) => void;

/**
 * Initiates the Sitecore content scraping process via SSE, providing live updates.
 * @param environment The Sitecore environment details to scrape.
 * @param onUpdate Callback function to receive live updates.
 * @returns A promise that resolves when the stream ends (either successfully or with an error).
 */
export const startScrapingApi = async (environment: SitecoreEnvironment, onUpdate: ScrapeUpdateCallback): Promise<ApiResponse> => {
    try {
        // Use fetch with a stream to send the initial request body
        const response = await fetch('/api/scrape', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ envId: environment.id, environment })
        });

        if (!response.ok || !response.body) {
            const errorText = await response.text();
            onUpdate('error', { message: `Failed to start scraping stream: ${errorText}` });
            return { success: false, message: `Failed to start scraping stream: ${errorText}`, error: errorText };
        }

        // Use EventSource for SSE. Note: EventSource only supports GET requests.
        // For POST requests with SSE, we need to manually read the stream from the fetch response.
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
                let eventData = {};

                for (const line of lines) {
                    if (line.startsWith('event: ')) {
                        eventType = line.substring('event: '.length);
                    } else if (line.startsWith('data: ')) {
                        try {
                            eventData = JSON.parse(line.substring('data: '.length));
                        } catch (e) {
                            console.error('Error parsing SSE data:', e);
                            eventData = { raw: line.substring('data: '.length), error: 'Parsing failed' };
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
 * @returns A promise that resolves to an ApiResponse indicating success or failure.
 */
export const updateIndexesApi = async (environment: SitecoreEnvironment): Promise<ApiResponse> => {
    try {
        const response = await fetch('/api/update-index', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ envId: environment.id, environment })
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
    // Note: updateIndexesApi remains a single-response function as it's not a streaming process.
};
