// app/page.tsx
// This is the main frontend page for your Next.js application using the App Router.

'use client'; // This directive makes the component a Client Component

import React, { useState } from 'react';
import { startScrapingApi, updateIndexesApi } from '../graphql/scraping'; // Import the new functions

// Define types for Sitecore environment
interface SitecoreEnvironment {
    id: number;
    name: string;
    url: string;
    apiKey: string;
    status: string;
    rootPath: string;
    languages: string[];
}

// Define the structure for a scraped item (must match the backend and scraping.ts)
interface ScrapedItem {
    id: string;
    name: string;
    path: string;
    url: string;
    language: string;
    content: string;
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


const Home: React.FC = () => {
    // State for managing Sitecore environments with default values from environment variables
    const [environments, setEnvironments] = useState<SitecoreEnvironment[]>([]);
    const [newEnvName, setNewEnvName] = useState<string>('dev'); // Default value
    const [newEnvUrl, setNewEnvUrl] = useState<string>(process.env.NEXT_PUBLIC_SITECORE_GRAPHQL_URL || ''); // Read from .env
    const [newEnvApiKey, setNewEnvApiKey] = useState<string>(process.env.NEXT_PUBLIC_SITECORE_API_KEY || ''); // Read from .env
    const [newEnvRootPath, setNewEnvRootPath] = useState<string>(process.env.NEXT_PUBLIC_SITECORE_ROOT_PATH || ''); // Read from .env
    const [newEnvLanguages, setNewEnvLanguages] = useState<string>(process.env.NEXT_PUBLIC_SITECORE_LANGUAGES || 'en'); // Read from .env

    // State to store scraped data for display (now updated incrementally)
    const [scrapedData, setScrapedData] = useState<ScrapedItem[] | null>(null);
    const [isScrapingLoading, setIsScrapingLoading] = useState<boolean>(false);
    const [currentScrapingStatus, setCurrentScrapingStatus] = useState<string>(''); // For live status messages
    const [uiMessage, setUiMessage] = useState<{ type: 'success' | 'error' | 'info', text: string } | null>(null); // For general UI messages

    // Function to display a temporary message in the UI
    const showUiMessage = (type: 'success' | 'error' | 'info', text: string) => {
        setUiMessage({ type, text });
        setTimeout(() => setUiMessage(null), 5000); // Clear message after 5 seconds
    };

    // Function to add a new Sitecore environment
    const addEnvironment = (): void => {
        if (newEnvName && newEnvUrl && newEnvApiKey && newEnvRootPath && newEnvLanguages) {
            const languagesArray = newEnvLanguages.split(',').map(lang => lang.trim()).filter(lang => lang.length > 0);
            if (languagesArray.length === 0) {
                showUiMessage('error', 'Please enter at least one language.');
                return;
            }

            setEnvironments([...environments, {
                id: Date.now(), // Simple unique ID for frontend display
                name: newEnvName,
                url: newEnvUrl,
                apiKey: newEnvApiKey,
                status: 'Not Indexed',
                rootPath: newEnvRootPath,
                languages: languagesArray,
            }]);
            setNewEnvName('');
            setNewEnvUrl('');
            setNewEnvApiKey('');
            setNewEnvRootPath('');
            setNewEnvLanguages('en');
            showUiMessage('success', 'Environment added successfully!');
        } else {
            showUiMessage('error', 'Please fill in all environment details.');
        }
    };

    // Function to initiate scraping via the new API utility with live updates
    const startScraping = async (envId: number): Promise<void> => {
        const targetEnv = environments.find(e => e.id === envId);
        if (!targetEnv) return;

        setIsScrapingLoading(true); // Set loading state
        setScrapedData([]); // Initialize as empty array for incremental updates
        setCurrentScrapingStatus('Starting scraping process...');
        setUiMessage(null); // Clear any previous UI messages

        // Update status for the specific environment in the list
        setEnvironments(prevEnvs => prevEnvs.map(env =>
            env.id === envId ? { ...env, status: 'Indexing in progress...' } : env
        ));

        // Define the callback for live updates
        const handleScrapeUpdate = (
            type: 'update' | 'start' | 'language_complete' | 'complete' | 'error',
            data: ScrapeUpdateData // Changed from 'any' to a union type
        ) => {
            if (type === 'update') {
                // Add new scraped item to the list
                setScrapedData(prev => [...(prev || []), data as ScrapedItem]);
                setCurrentScrapingStatus(`Scraped: ${(data as ScrapedItem).path} (${(data as ScrapedItem).language})`);
            } else if (type === 'start') {
                const startData = data as ScrapeStartData;
                setCurrentScrapingStatus(`Starting language: ${startData.language} from ${startData.rootPath}`);
            } else if (type === 'language_complete') {
                const langCompleteData = data as ScrapeLanguageCompleteData;
                setCurrentScrapingStatus(`Finished language: ${langCompleteData.language}`);
            } else if (type === 'complete') {
                const completeData = data as ScrapeCompleteErrorData; // Reusing for message property
                setCurrentScrapingStatus('Scraping complete!');
                setIsScrapingLoading(false);
                setEnvironments(prevEnvs => prevEnvs.map(env =>
                    env.id === envId ? { ...env, status: 'Indexed' } : env
                ));
                showUiMessage('success', completeData.message || 'Scraping process completed successfully.');
            } else if (type === 'error') {
                const errorData = data as ScrapeCompleteErrorData;
                setCurrentScrapingStatus(`Error during scraping: ${errorData.message}`);
                setIsScrapingLoading(false);
                setEnvironments(prevEnvs => prevEnvs.map(env =>
                    env.id === envId ? { ...env, status: 'Scraping Failed' } : env
                ));
                showUiMessage('error', `Scraping error: ${errorData.message}`);
            }
        };

        // Call the SSE-enabled scraping API
        await startScrapingApi(targetEnv, handleScrapeUpdate);
    };

    // Function to update indexes via the new API utility (remains non-streaming)
    const updateIndexes = async (envId: number): Promise<void> => {
        const targetEnv = environments.find(e => e.id === envId);
        if (!targetEnv) return;

        setUiMessage(null); // Clear any previous UI messages

        // Update status to "in progress" immediately
        setEnvironments(prevEnvs => prevEnvs.map(env =>
            env.id === envId ? { ...env, status: 'Updating index...' } : env
        ));

        const result = await updateIndexesApi(targetEnv);

        if (result.success) {
            setEnvironments(prevEnvs => prevEnvs.map(env =>
                env.id === envId ? { ...env, status: 'Indexed (Updated)' } : env
            ));
            showUiMessage('success', `Index update initiated for environment ID: ${envId}. Message: ${result.message}`);
        } else {
            setEnvironments(prevEnvs => prevEnvs.map(env =>
                env.id === envId ? { ...env, status: 'Update Failed' } : env
            ));
            showUiMessage('error', `Index update failed for environment ID: ${envId}. Error: ${result.error}`);
        }
    };

    return (
        <div className="min-h-screen bg-gray-100 font-sans antialiased text-gray-800 p-4 flex flex-col items-center">
            <div className="max-w-4xl w-full bg-white shadow-lg rounded-xl p-8 space-y-8">
                <h1 className="text-4xl font-extrabold text-center text-indigo-700 mb-8">
                    Sitecore Content Scrapper MCP (Next.js App Router)
                </h1>

                {/* Navigation Link to Scraped Content Log */}
                <div className="text-center mb-4">
                    <a href="#scraped-content-log" className="text-indigo-600 hover:underline text-lg font-medium">
                        Jump to Scraped Content Log
                    </a>
                </div>

                {/* General UI Message Display */}
                {uiMessage && (
                    <div className={`p-3 rounded-lg text-center font-medium ${
                        uiMessage.type === 'success' ? 'bg-green-100 text-green-800' :
                        uiMessage.type === 'error' ? 'bg-red-100 text-red-800' :
                        'bg-blue-100 text-blue-800'
                    }`}>
                        {uiMessage.text}
                    </div>
                )}

                {/* Sitecore Environment Management Section */}
                <section className="bg-gradient-to-br from-blue-50 to-indigo-100 p-6 rounded-lg shadow-md border border-blue-200">
                    <h2 className="text-2xl font-bold text-indigo-600 mb-4">Manage Sitecore Environments</h2>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                        <input
                            type="text"
                            placeholder="Environment Name (e.g., 'Dev Site')"
                            className="p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-400 focus:border-transparent transition duration-200"
                            value={newEnvName}
                            onChange={(e) => setNewEnvName(e.target.value)}
                        />
                        <input
                            type="url"
                            placeholder="Sitecore GraphQL URL (e.g., 'https://your-site.com/sitecore/api/graph/edge')"
                            className="p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-400 focus:border-transparent transition duration-200"
                            value={newEnvUrl}
                            onChange={(e) => setNewEnvUrl(e.target.value)}
                        />
                        <input
                            type="text"
                            placeholder="Sitecore API Key"
                            className="p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-400 focus:border-transparent transition duration-200"
                            value={newEnvApiKey}
                            onChange={(e) => setNewEnvApiKey(e.target.value)}
                        />
                         <input
                            type="text"
                            placeholder="Sitecore Root Path (e.g., '/sitecore/content/my-site/home')"
                            className="p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-400 focus:border-transparent transition duration-200"
                            value={newEnvRootPath}
                            onChange={(e) => setNewEnvRootPath(e.target.value)}
                        />
                        <input
                            type="text"
                            placeholder="Languages (e.g., 'en,fr,de')"
                            className="p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-400 focus:border-transparent transition duration-200"
                            value={newEnvLanguages}
                            onChange={(e) => setNewEnvLanguages(e.target.value)}
                        />
                    </div>
                    <button
                        onClick={addEnvironment}
                        className="w-full bg-indigo-600 text-white py-3 rounded-lg font-semibold hover:bg-indigo-700 transition duration-300 ease-in-out transform hover:scale-105 shadow-lg"
                    >
                        Add Environment
                    </button>

                    <div className="mt-8 space-y-4">
                        {environments.length === 0 ? (
                            <p className="text-center text-gray-500 italic">No environments added yet.</p>
                        ) : (
                            environments.map(env => (
                                <div key={env.id} className="flex flex-col md:flex-row items-center justify-between bg-white p-4 rounded-lg shadow-sm border border-gray-200">
                                    <div className="flex-1 mb-2 md:mb-0 md:mr-4">
                                        <h3 className="text-lg font-semibold text-gray-900">{env.name}</h3>
                                        <p className="text-sm text-gray-600 truncate">{env.url}</p>
                                        <p className="text-sm text-gray-500">Root Path: <span className="font-medium text-gray-700">{env.rootPath}</span></p>
                                        <p className="text-sm text-gray-500">Languages: <span className="font-medium text-gray-700">{env.languages.join(', ')}</span></p>
                                        <p className="text-sm text-gray-500">Status: <span className={`font-medium ${env.status.includes('Indexed') ? 'text-green-600' : 'text-yellow-600'}`}>{env.status}</span></p>
                                    </div>
                                    <div className="flex space-x-2">
                                        <button
                                            onClick={() => startScraping(env.id)}
                                            className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition duration-200 text-sm font-medium shadow-md"
                                            disabled={isScrapingLoading} // Disable button while loading
                                        >
                                            {isScrapingLoading ? (
                                                <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                                </svg>
                                            ) : (
                                                'Start Scraping'
                                            )}
                                        </button>
                                        <button
                                            onClick={() => updateIndexes(env.id)}
                                            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition duration-200 text-sm font-medium shadow-md"
                                        >
                                            Update Indexes
                                        </button>
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </section>

                {/* Scraped Data Display Section */}
                <section id="scraped-content-log" className="bg-gradient-to-br from-gray-50 to-gray-100 p-6 rounded-lg shadow-md border border-gray-200 mt-8">
                    <h2 className="text-2xl font-bold text-gray-700 mb-4">Scraped Content Log</h2>
                    {isScrapingLoading && (
                        <p className="text-center text-indigo-500 mb-4 flex items-center justify-center">
                            <svg className="animate-spin h-5 w-5 mr-3 text-indigo-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            {currentScrapingStatus || 'Scraping in progress...'}
                        </p>
                    )}
                    {scrapedData && scrapedData.length > 0 ? (
                        <div className="space-y-4 max-h-96 overflow-y-auto bg-white p-4 rounded-lg border border-gray-300 shadow-inner">
                            {scrapedData.map((item) => (
                                <div key={item.id + item.language} className="bg-blue-50 p-3 rounded-md border border-blue-200">
                                    <h4 className="font-semibold text-blue-800 text-lg">{item.name} ({item.language})</h4>
                                    <p className="text-sm text-gray-700 break-all">Path: {item.path}</p>
                                    <p className="text-sm text-gray-700 break-all">URL: <a href={item.url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">{item.url}</a></p>
                                    <p className="text-sm text-gray-700 mt-2">Content:</p>
                                    <pre className="bg-gray-100 p-2 rounded-md text-xs overflow-auto max-h-24 whitespace-pre-wrap">{item.content || 'No text content found.'}</pre>
                                    {item.childrenPaths && item.childrenPaths.length > 0 && (
                                        <p className="text-xs text-gray-600 mt-1">Children: {item.childrenPaths.map(c => c.name).join(', ')}</p>
                                    )}
                                </div>
                            ))}
                        </div>
                    ) : (
                        !isScrapingLoading && <p className="text-center text-gray-500 italic">No scraped data to display yet. Start scraping an environment!</p>
                    )}
                </section>
            </div>
            {/* The following CSS is assumed to be handled by your global stylesheet (globals.css) */}
            {/* and Tailwind's PostCSS setup, along with font imports in layout.tsx or globals.css. */}
        </div>
    );
};

export default Home;
