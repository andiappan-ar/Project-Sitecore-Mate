// src/app/query/page.tsx
'use client';

import React, { useState } from 'react';
import Link from 'next/link'; // For navigation back to home

interface QueryResultItem {
    content: string;
    metadata: {
        id: string;
        name: string;
        path: string;
        url: string;
        language: string;
        [key: string]: any; // Allow other metadata properties
    };
    distance: number;
}

const QueryPage: React.FC = () => {
    const [query, setQuery] = useState<string>('');
    const [searchResults, setSearchResults] = useState<QueryResultItem[] | null>(null);
    const [generatedAnswer, setGeneratedAnswer] = useState<string | null>(null); // State for LLM generated answer
    const [isLoadingSearch, setIsLoadingSearch] = useState<boolean>(false);
    const [isLoadingGenerate, setIsLoadingGenerate] = useState<boolean>(false); // Loading state for LLM generation
    const [errorMessage, setErrorMessage] = useState<string | null>(null);

    const handleQuerySubmit = async () => {
        if (!query.trim()) {
            setErrorMessage('Please enter a query.');
            return;
        }

        setIsLoadingSearch(true);
        setErrorMessage(null);
        setSearchResults(null); // Clear previous search results
        setGeneratedAnswer(null); // Clear previous generated answer

        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, n_results: 5 }), // Request top 5 results
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.message || 'Failed to fetch search results.');
            }

            const data = await response.json();
            if (data.status === 'success') {
                setSearchResults(data.results);
            } else {
                setErrorMessage(data.message || 'An unknown error occurred during search.');
            }
        } catch (error: any) {
            console.error('Error fetching search results:', error);
            setErrorMessage(error.message || 'Failed to connect to the search service.');
        } finally {
            setIsLoadingSearch(false);
        }
    };

    const handleGenerateAnswer = async () => {
        if (!query.trim()) {
            setErrorMessage('Please enter a query to generate an answer.');
            return;
        }

        setIsLoadingGenerate(true);
        setErrorMessage(null);
        setGeneratedAnswer(null); // Clear previous generated answer
        setSearchResults(null); // Also clear search results when generating answer

        try {
            const response = await fetch('/api/generate-answer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, n_results: 5 }), // Send query to LLM endpoint
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.message || 'Failed to generate answer.');
            }

            const data = await response.json();
            if (data.status === 'success') {
                setGeneratedAnswer(data.answer);
                // Optionally, display the retrieved context for the generated answer
                // if data.retrieved_context is available and you want to show it.
                // For this example, we'll just show the answer.
                setSearchResults(data.retrieved_context || []); // Show context used for generation
            } else {
                setErrorMessage(data.message || 'An unknown error occurred during answer generation.');
            }
        } catch (error: any) {
            console.error('Error generating answer:', error);
            setErrorMessage(error.message || 'Failed to connect to the answer generation service.');
        } finally {
            setIsLoadingGenerate(false);
        }
    };

    return (
        <div className="min-h-screen bg-gray-100 font-sans antialiased text-gray-800 p-4 flex flex-col items-center">
            <div className="max-w-4xl w-full bg-white shadow-lg rounded-xl p-8 space-y-8">
                <Link href="/" className="text-indigo-600 hover:underline mb-4 block">
                    &larr; Back to Environment Management
                </Link>
                <h1 className="text-4xl font-extrabold text-center text-indigo-700 mb-8">
                    Query Vector Database & Generate Answer
                </h1>

                <section className="bg-gradient-to-br from-blue-50 to-indigo-100 p-6 rounded-lg shadow-md border border-blue-200">
                    <h2 className="text-2xl font-bold text-indigo-600 mb-4">Enter Your Query</h2>
                    <textarea
                        className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-400 focus:border-transparent transition duration-200 resize-y min-h-[100px]"
                        placeholder="e.g., 'What are the details about the HeroBanner component?' or 'Tell me about Sitecore's headless capabilities.'"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        rows={4}
                    ></textarea>
                    <div className="flex flex-col sm:flex-row gap-4 mt-4">
                        <button
                            onClick={handleQuerySubmit}
                            className="flex-1 bg-indigo-600 text-white py-3 rounded-lg font-semibold hover:bg-indigo-700 transition duration-300 ease-in-out transform hover:scale-105 shadow-lg"
                            disabled={isLoadingSearch || isLoadingGenerate}
                        >
                            {isLoadingSearch ? (
                                <svg className="animate-spin h-5 w-5 mx-auto text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                            ) : (
                                'Search Vector DB (Retrieve)'
                            )}
                        </button>
                        <button
                            onClick={handleGenerateAnswer}
                            className="flex-1 bg-purple-600 text-white py-3 rounded-lg font-semibold hover:bg-purple-700 transition duration-300 ease-in-out transform hover:scale-105 shadow-lg"
                            disabled={isLoadingSearch || isLoadingGenerate}
                        >
                            {isLoadingGenerate ? (
                                <svg className="animate-spin h-5 w-5 mx-auto text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                            ) : (
                                'Generate Answer (RAG)'
                            )}
                        </button>
                    </div>
                </section>

                {errorMessage && (
                    <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mt-4" role="alert">
                        <strong className="font-bold">Error!</strong>
                        <span className="block sm:inline"> {errorMessage}</span>
                    </div>
                )}

                {/* LLM Generated Answer Section */}
                <section className="bg-gradient-to-br from-green-50 to-teal-100 p-6 rounded-lg shadow-md border border-green-200 mt-8">
                    <h2 className="text-2xl font-bold text-teal-700 mb-4">Generated Answer</h2>
                    {isLoadingGenerate && !generatedAnswer && (
                        <p className="text-center text-teal-500 italic">Generating answer...</p>
                    )}
                    {generatedAnswer ? (
                        <div className="bg-white p-4 rounded-lg border border-gray-300 shadow-inner">
                            <pre className="whitespace-pre-wrap font-mono text-sm text-gray-800">{generatedAnswer}</pre>
                        </div>
                    ) : (
                        !isLoadingGenerate && !isLoadingSearch && !errorMessage && <p className="text-center text-gray-500 italic">No answer generated yet. Use the "Generate Answer" button!</p>
                    )}
                </section>

                {/* Search Results Section (now also shows context for generated answer) */}
                <section className="bg-gradient-to-br from-gray-50 to-gray-100 p-6 rounded-lg shadow-md border border-gray-200 mt-8">
                    <h2 className="text-2xl font-bold text-gray-700 mb-4">Retrieved Context / Search Results</h2>
                    {(isLoadingSearch || isLoadingGenerate) && !searchResults && (
                        <p className="text-center text-indigo-500 italic">Retrieving context...</p>
                    )}
                    {searchResults && searchResults.length > 0 ? (
                        <div className="space-y-4 max-h-96 overflow-y-auto bg-white p-4 rounded-lg border border-gray-300 shadow-inner">
                            {searchResults.map((result, index) => (
                                <div key={result.metadata.id + result.metadata.language + index} className="bg-green-50 p-3 rounded-md border border-green-200">
                                    <h4 className="font-semibold text-green-800 text-lg">
                                        {result.metadata.name} ({result.metadata.language}) - Distance: {result.distance.toFixed(4)}
                                    </h4>
                                    <p className="text-sm text-gray-700 break-all">Path: {result.metadata.path}</p>
                                    <p className="text-sm text-gray-700 break-all">URL: <a href={result.metadata.url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">{result.metadata.url}</a></p>
                                    <p className="text-sm text-gray-700 mt-2">Content Snippet:</p>
                                    <pre className="bg-gray-100 p-2 rounded-md text-xs overflow-auto max-h-24 whitespace-pre-wrap">{result.content || 'No text content found.'}</pre>
                                </div>
                            ))}
                        </div>
                    ) : (
                        !isLoadingSearch && !isLoadingGenerate && !errorMessage && <p className="text-center text-gray-500 italic">No results to display yet. Enter a query and search or generate an answer!</p>
                    )}
                </section>
            </div>
        </div>
    );
};

export default QueryPage;
