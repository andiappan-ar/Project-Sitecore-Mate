'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';

// Define the structure for a Sitecore Environment
interface SitecoreEnvironment {
  id: string;
  name: string;
  // Add other properties as needed if you want to display them
}

// Define the structure for a search result document
interface SearchResultDocument {
  id: string;
  content: string;
  metadata: {
    language: string;
    name: string;
    path: string;
    url: string;
  };
}

const QueryPage = () => {
  const [environments, setEnvironments] = useState<SitecoreEnvironment[]>([]);
  const [selectedEnvironmentId, setSelectedEnvironmentId] = useState<string>('');
  const [query, setQuery] = useState<string>('');
  const [searchResults, setSearchResults] = useState<SearchResultDocument[] | null>(null);
  const [generatedAnswer, setGeneratedAnswer] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Load environments from localStorage
    // FIX: Changed localStorage key from 'sitecoreEnvironments' to 'environments'
    // to match the key used on the main environment management page.
    const storedEnvs = localStorage.getItem('environments');
    if (storedEnvs) {
      const parsedEnvs = JSON.parse(storedEnvs);
      setEnvironments(parsedEnvs);
      // Default to the first environment if available
      if (parsedEnvs.length > 0) {
        setSelectedEnvironmentId(parsedEnvs[0].id);
      }
    }
  }, []);

  const resetState = () => {
    setSearchResults(null);
    setGeneratedAnswer('');
    setError(null);
  };

  const handleVectorSearch = async () => {
    if (!query || !selectedEnvironmentId) {
      setError('Please select an environment and enter a query.');
      return;
    }
    setIsLoading(true);
    resetState();

    try {
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          environmentId: selectedEnvironmentId,
          query: query,
        }),
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || 'Failed to fetch search results.');
      }
      const data = await response.json();
      setSearchResults(data.results);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleGenerateAnswer = async () => {
    if (!query || !selectedEnvironmentId) {
      setError('Please select an environment and enter a query.');
      return;
    }
    setIsLoading(true);
    resetState();

    try {
      const response = await fetch('/api/generate-answer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          environmentId: selectedEnvironmentId,
          query: query,
        }),
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || 'Failed to generate answer.');
      }
      const data = await response.json();
      setGeneratedAnswer(data.answer);
      // Optionally, you can also display the context documents
      setSearchResults(data.context);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="container mx-auto p-4 sm:p-6 lg:p-8 font-sans bg-gray-50 min-h-screen">
      <div className="max-w-4xl w-full mx-auto">
        <h1 className="text-4xl font-extrabold text-center text-indigo-700 mb-2">AI Content Search</h1>
        <p className="text-center text-gray-500 mb-6">Query your indexed Sitecore content using vector search or generative AI.</p>
        
        <div className="text-center mb-6">
            <Link href="/" className="text-blue-600 hover:underline text-lg font-medium">
                &larr; Back to Environment Management
            </Link>
        </div>

        {/* Query Input Section */}
        <div className="bg-white p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-2xl font-semibold text-gray-700 mb-4">Select Environment & Enter Query</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <select
              value={selectedEnvironmentId}
              onChange={(e) => setSelectedEnvironmentId(e.target.value)}
              className="p-3 border rounded-md focus:ring-2 focus:ring-indigo-500 w-full bg-white"
              disabled={environments.length === 0 || isLoading}
            >
              {environments.length > 0 ? (
                environments.map(env => (
                  <option key={env.id} value={env.id}>{env.name}</option>
                ))
              ) : (
                <option>No environments available. Please add some on the Environment Management page.</option>
              )}
            </select>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a question about your content..."
              className="p-3 border rounded-md focus:ring-2 focus:ring-indigo-500 w-full col-span-1 md:col-span-2"
              disabled={isLoading}
            />
          </div>
          <div className="flex flex-col sm:flex-row space-y-2 sm:space-y-0 sm:space-x-2">
            <button onClick={handleVectorSearch} disabled={isLoading} className="flex-1 px-6 py-3 bg-blue-600 text-white font-semibold rounded-md hover:bg-blue-700 transition-colors disabled:bg-gray-400">
              {isLoading ? 'Searching...' : 'Search Vector DB (Retrieve)'}
            </button>
            <button onClick={handleGenerateAnswer} disabled={isLoading} className="flex-1 px-6 py-3 bg-green-600 text-white font-semibold rounded-md hover:bg-green-700 transition-colors disabled:bg-gray-400">
              {isLoading ? 'Generating...' : 'Generate Answer (RAG)'}
            </button>
          </div>
        </div>

        {/* Results Section */}
        {error && <div className="p-4 mb-6 rounded-lg text-center font-medium bg-red-100 text-red-800">{error}</div>}

        {isLoading && (
            <div className="text-center">
                <svg className="animate-spin h-8 w-8 text-indigo-600 mx-auto" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
            </div>
        )}

        {generatedAnswer && (
            <div className="bg-green-50 border border-green-200 p-6 rounded-lg mb-8">
                <h3 className="text-xl font-bold text-green-800 mb-2">Generated Answer</h3>
                <p className="text-gray-700 whitespace-pre-wrap">{generatedAnswer}</p>
            </div>
        )}

        {searchResults && searchResults.length > 0 && (
            <div>
                <h3 className="text-xl font-bold text-gray-800 mb-4">{generatedAnswer ? 'Retrieved Context:' : 'Search Results:'}</h3>
                <div className="space-y-4">
                    {searchResults.map((doc) => (
                        <div key={doc.id} className="bg-white p-4 rounded-lg shadow-md border">
                            <h4 className="font-semibold text-indigo-700">{doc.metadata.name} ({doc.metadata.language})</h4>
                            <a href={doc.metadata.url} target="_blank" rel="noopener noreferrer" className="text-sm text-blue-500 hover:underline break-all">{doc.metadata.url}</a>
                            <pre className="mt-2 bg-gray-100 p-3 rounded-md text-sm whitespace-pre-wrap max-h-48 overflow-y-auto">{doc.content}</pre>
                        </div>
                    ))}
                </div>
            </div>
        )}
      </div>
    </main>
  );
};

export default QueryPage;
