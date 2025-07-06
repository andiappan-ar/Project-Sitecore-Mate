// src/app/page.tsx

'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';

// Define the structure for a Sitecore Environment
interface SitecoreEnvironment {
  id: string;
  name: string;
  graphql_endpoint: string;
  api_key: string;
  root_path: string;
  language: string;
  status: string;
}

// Define the structure for a document's metadata
interface DocumentMetadata {
  language: string;
  name: string;
  path: string;
  url: string;
}

// Define the structure for a search result document (from vector search)
interface VectorSearchResult {
  content: string;
  metadata: DocumentMetadata;
  distance?: number; // Include the distance score from the vector DB
}

// Define the structure for context used in RAG
interface ContextDocument {
  content: string;
  metadata: DocumentMetadata;
}

const QueryPage = () => {
  const [environments, setEnvironments] = useState<SitecoreEnvironment[]>([]);
  const [selectedEnvironmentId, setSelectedEnvironmentId] = useState<string>('');
  const [query, setQuery] = useState<string>('');

  // State for different types of results
  const [vectorSearchResults, setVectorSearchResults] = useState<VectorSearchResult[] | null>(null);
  const [retrievedContext, setRetrievedContext] = useState<ContextDocument[] | null>(null);
  const [generatedAnswer, setGeneratedAnswer] = useState<string>('');

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Function to fetch environments from the API
  const fetchEnvironments = async () => {
    try {
      const response = await fetch('/api/environments');
      if (!response.ok) {
        throw new Error(`Error fetching environments: ${response.statusText}`);
      }
      const data: SitecoreEnvironment[] = await response.json();
      setEnvironments(data);
      if (data.length > 0) {
        setSelectedEnvironmentId(data[0].id.toString());
      }
    } catch (err: any) {
      setError(err.message);
      console.error("Failed to fetch environments:", err);
    }
  };

  useEffect(() => {
    fetchEnvironments(); // Call the fetch function on component mount
  }, []);

  // Function to clear previous results before a new search
  const resetState = () => {
    setVectorSearchResults(null);
    setRetrievedContext(null);
    setGeneratedAnswer('');
    setError(null);
  };

  // Helper function to get the selected environment's name
  const getSelectedEnvironmentName = (): string | undefined => {
    if (!selectedEnvironmentId) return undefined;
    const selectedEnv = environments.find(env => env.id.toString() === selectedEnvironmentId);
    return selectedEnv?.name;
  };

  // Handler for the "Search Vector DB" button
  const handleVectorSearch = async () => {
    const environmentName = getSelectedEnvironmentName();

    if (!query || !environmentName) {
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
          environment: environmentName,
          query: query,
        }),
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || 'Failed to fetch search results.');
      }
      const data = await response.json();
      setVectorSearchResults(data.results);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  // Main handler for the "Generate Answer" button
  const handleGenerateAnswer = async () => {
    const environmentName = getSelectedEnvironmentName();

    if (!query || !environmentName) {
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
          environment: environmentName,
          query: query,
        }),
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || 'Failed to generate answer.');
      }
      const data = await response.json();
      setGeneratedAnswer(data.answer);
      setRetrievedContext(data.sources.map((source: any) => ({
        content: '',
        metadata: {
          id: source.path,
          name: source.title,
          path: source.path,
          url: source.url,
          language: '',
        }
      })));
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    
     <div className="min-h-screen bg-neutral-50 p-8 font-sans"> {/* Changed to light gray background */}
        <div className="max-w-7xl mx-auto bg-white p-6 rounded-lg shadow-lg"> {/* Main container, keeping white */}

          {/* H1 changed to darker gray for prominence */}
          <h1 className="text-4xl font-extrabold text-center text-gray-800 mb-2">AI Content Search</h1>
          <p className="text-center text-gray-600 mb-6">Query your indexed Sitecore content using vector search or generative AI.</p>

          <div className="text-center mb-6">
            {/* Link changed to dark text */}
            <Link href="/" className="text-gray-700 hover:text-gray-900 hover:underline text-lg font-medium">
              &larr; Back to Environment Management
            </Link>
          </div>

          <h2 className="text-2xl font-semibold text-gray-700 mb-4">Select Environment & Enter Query</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <select
              value={selectedEnvironmentId}
              onChange={(e) => setSelectedEnvironmentId(e.target.value)}
              className="p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 w-full bg-white text-gray-800"
              disabled={environments.length === 0 || isLoading}
            >
              {environments.length > 0 ? (
                environments.map(env => (
                  <option key={env.id} value={env.id.toString()}>{env.name}</option>
                ))
              ) : (
                <option value="">No environments available. Please add some first.</option>
              )}
            </select>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a question about your content..."
              className="p-3 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 w-full col-span-1 md:col-span-2 text-gray-800"
              disabled={isLoading}
              onKeyDown={(e) => e.key === 'Enter' && handleGenerateAnswer()}
            />
          </div>
          <div className="flex flex-col sm:flex-row space-y-2 sm:space-y-0 sm:space-x-2">
            <button onClick={handleVectorSearch} disabled={isLoading}
              className="flex-1 px-6 py-3 bg-gray-800 text-white font-semibold rounded-md hover:bg-gray-900 transition-colors disabled:bg-gray-400"> {/* Button color */}
              {isLoading ? 'Searching...' : 'Search Vector DB (Retrieve)'}
            </button>
            <button onClick={handleGenerateAnswer} disabled={isLoading}
              className="flex-1 px-6 py-3 bg-gray-800 text-white font-semibold rounded-md hover:bg-gray-900 transition-colors disabled:bg-gray-400"> {/* Button color */}
              {isLoading ? 'Generating...' : 'Generate Answer (RAG)'}
            </button>
          </div>
        </div>

        {/* Results Section */}
        {error && <div className="p-4 mb-6 rounded-lg text-center font-medium bg-red-100 text-red-800">{error}</div>}

        {isLoading && (
          <div className="text-center">
            <svg className="animate-spin h-8 w-8 text-blue-600 mx-auto" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
          </div>
        )}

        {/* RAG Answer Display */}
        {generatedAnswer && (
          <div className="bg-white border border-gray-200 p-6 rounded-lg mb-8 shadow-sm"> {/* Changed to white background, subtle border */}
            <h3 className="text-xl font-bold text-gray-800 mb-2">Summary</h3>
            <p className="text-gray-700 whitespace-pre-wrap">{generatedAnswer}</p>
          </div>
        )}

        {/* RAG References Display */}
        {retrievedContext && retrievedContext.length > 0 && (
          <div className="mb-8">
            <h3 className="text-xl font-bold text-gray-800 mb-4">References</h3>
            <div className="space-y-2">
              {retrievedContext.map((doc, index) => (
                <div key={index} className="bg-white p-3 rounded-lg shadow-sm border border-gray-200">
                  <a href={doc.metadata.url} target="_blank" rel="noopener noreferrer" className="font-semibold text-blue-600 hover:underline">
                    {doc.metadata.name}
                  </a>
                  <p className="text-sm text-gray-500">{doc.metadata.url}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Vector Search Results Display */}
        {vectorSearchResults && vectorSearchResults.length > 0 && (
          <div>
            <h3 className="text-xl font-bold text-gray-800 mb-4">Vector Search Results</h3>
            <div className="space-y-4">
              {vectorSearchResults.map((doc, index) => (
                <div key={doc.metadata.path + index} className="bg-white p-4 rounded-lg shadow-md border border-gray-200">
                  <div className="flex justify-between items-start">
                    <h4 className="font-semibold text-gray-800">{doc.metadata.name} ({doc.metadata.language})</h4> {/* Changed to gray-800 */}
                    {doc.distance && <span className="text-xs bg-neutral-100 text-gray-700 font-medium px-2 py-1 rounded-full">Distance: {doc.distance.toFixed(4)}</span>} {/* Changed to neutral colors */}
                  </div>
                  <a href={doc.metadata.url} target="_blank" rel="noopener noreferrer" className="text-sm text-blue-600 hover:underline break-all">{doc.metadata.url}</a>
                  <pre className="mt-2 bg-gray-100 p-3 rounded-md text-sm whitespace-pre-wrap max-h-48 overflow-y-auto text-gray-700">{doc.content}</pre>
                </div>
              ))}
            </div>
          </div>
        )}

      </div>
   
  );
};

export default QueryPage;