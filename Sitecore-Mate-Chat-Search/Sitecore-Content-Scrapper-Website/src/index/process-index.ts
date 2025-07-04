// src/index/process-index.ts

// No longer need fs or path as we're sending to a service
// import * as fs from 'fs';
// import * as path from 'path';

// Define the structure for a scraped item (must match the backend and frontend)
interface ScrapedItem {
    id: string;
    name: string;
    path: string;
    url: string;
    language: string;
    content: string; // Aggregated text content from relevant fields
    childrenPaths?: { name: string; path: string }[];
}

// Define the URL of your Python FastAPI indexing service
// Ensure this matches the host and port where your FastAPI server is running
// You might want to put this in an environment variable for different deployments
const PYTHON_INDEXING_SERVICE_URL = process.env.PYTHON_INDEXING_SERVICE_URL || "http://localhost:8001/index-content";


/**
 * Processes a scraped content item by sending it to a Python FastAPI service for indexing.
 * @param item The scraped content item to process and send.
 * @returns A promise that resolves when the indexing process is complete.
 */
export async function processAndIndexContent(item: ScrapedItem): Promise<void> {
    console.log(`--- Sending Item to Python Indexing Service ---`);
    console.log(`  ID: ${item.id}`);
    console.log(`  Name: ${item.name}`);
    console.log(`  Path: ${item.path}`);
    console.log(`  Language: ${item.language}`);
    console.log(`  Content Length: ${item.content.length} characters`);

    try {
        const response = await fetch(PYTHON_INDEXING_SERVICE_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(item), // Send the scraped item directly
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Indexing service failed: ${response.status} - ${errorText}`);
        }

        const result = await response.json();
        console.log(`--- Indexing Service Response for Item ${item.id}:`, result);
        console.log(`--- Successfully Sent Item: ${item.id} to Python Indexing Service ---`);

    } catch (error: any) {
        console.error(`Error sending item ${item.id} to indexing service:`, error);
        // Handle error, e.g., retry, log to a failure queue
    }
}
