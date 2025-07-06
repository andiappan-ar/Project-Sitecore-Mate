// src/app/page.tsx

'use client'; // This directive makes the component a Client Component

import { useState, useEffect, FormEvent } from 'react';
import { Environment } from '@/lib/environments'; // Assuming Environment type is exported from here
import { v4 as uuidv4 } from 'uuid'; // For generating unique IDs for new environments
import Link from 'next/link'; // Import Link for navigation

export default function Home() {
  const [environments, setEnvironments] = useState<Environment[]>([]);
  const [newEnvName, setNewEnvName] = useState('');
  const [newEnvGraphQLEndpoint, setNewEnvGraphQLEndpoint] = useState(process.env.NEXT_PUBLIC_SITECORE_GRAPHQL_URL || '');
  const [newEnvApiKey, setNewEnvApiKey] = useState(process.env.NEXT_PUBLIC_SITECORE_API_KEY || '');
  const [newEnvRootPath, setNewEnvRootPath] = useState(process.env.NEXT_PUBLIC_SITECORE_ROOT_PATH || '/sitecore/content/demo-mate-d-jss-xp/home');
  const [newEnvLanguage, setNewEnvLanguage] = useState(process.env.NEXT_PUBLIC_SITECORE_LANGUAGES?.split(',')[0] || 'en'); // Takes the first language if multiple
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [selectedEnvironmentId, setSelectedEnvironmentId] = useState<number | null>(null);
  const [logMessages, setLogMessages] = useState<string[]>([]); // State to store log messages
  const [isScraping, setIsScraping] = useState(false); // State to indicate if scraping is active

  // Fetch environments on component mount
  useEffect(() => {
    fetchEnvironments();
  }, []);

  // Effect for SSE log streaming
  useEffect(() => {
    if (isScraping) {
      // Connect to the Next.js API proxy for the log stream
      const eventSource = new EventSource('/api/log-stream');

      eventSource.onmessage = (event) => {
        setLogMessages((prevLogs) => [...prevLogs, event.data]);
        // Scroll to the bottom of the log window
        const logWindow = document.getElementById('log-window');
        if (logWindow) {
          logWindow.scrollTop = logWindow.scrollHeight;
        }
      };

      eventSource.onerror = (err) => {
        console.error('EventSource failed:', err);
        eventSource.close();
        setLogMessages((prevLogs) => [...prevLogs, 'Error: Log stream disconnected.']);
      };

      // Clean up the EventSource connection when component unmounts or scraping stops
      return () => {
        console.log('Closing EventSource connection.');
        eventSource.close();
      };
    }
  }, [isScraping]); // Re-run when isScraping changes

  const fetchEnvironments = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/environments');
      if (!response.ok) {
        throw new Error(`Error: ${response.status} - ${await response.text()}`);
      }
      const data = await response.json();
      setEnvironments(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleAddEnvironment = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setMessage(null);

    const newEnvironment: Environment = {
      id: environments.length > 0 ? Math.max(...environments.map(env => env.id)) + 1 : 1,
      name: newEnvName,
      graphql_endpoint: newEnvGraphQLEndpoint,
      api_key: newEnvApiKey,
      root_path: newEnvRootPath,
      language: newEnvLanguage,
      status: 'Ready', // Default status
    };

    const updatedEnvironments = [...environments, newEnvironment];

    try {
      const response = await fetch('/api/environments', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updatedEnvironments),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.status} - ${await response.text()}`);
      }

      setMessage('Environment added successfully!');
      setEnvironments(updatedEnvironments);
      setNewEnvName('');
      // Keep the form fields populated with env values after adding
      // setNewEnvGraphQLEndpoint(''); 
      // setNewEnvApiKey('');
      // setNewEnvRootPath('/sitecore/content/demo-mate-d-jss-xp/home');
      // setNewEnvLanguage('en');
    } catch (err: any) {
      setError(err.message);
    }
  };

  const handleDeleteEnvironment = async (id: number) => {
    setError(null);
    setMessage(null);

    const updatedEnvironments = environments.filter(env => env.id !== id);

    try {
      const response = await fetch('/api/environments', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updatedEnvironments),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.status} - ${await response.text()}`);
      }

      setMessage('Environment deleted successfully!');
      setEnvironments(updatedEnvironments);
      if (selectedEnvironmentId === id) {
        setSelectedEnvironmentId(null);
      }
    } catch (err: any) {
      setError(err.message);
    }
  };

  const handleScrapeContent = async () => {
    if (selectedEnvironmentId === null) {
      setError('Please select an environment to scrape.');
      return;
    }

    setError(null);
    setMessage(null);
    setLogMessages([]); // Clear previous logs
    setIsScraping(true); // Start scraping, which triggers SSE useEffect

    try {
      const response = await fetch('/api/scrape', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ environmentId: selectedEnvironmentId }),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.status} - ${await response.text()}`);
      }

      const data = await response.json();
      setMessage(data.message || 'Scraping and indexing completed.');
    } catch (err: any) {
      setError(err.message);
      setLogMessages((prevLogs) => [...prevLogs, `Scraping failed: ${err.message}`]);
    } finally {
      setIsScraping(false); // Stop scraping, which cleans up SSE
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-8 font-sans">
      <div className="max-w-7xl mx-auto bg-white p-6 rounded-lg shadow-lg">
        <h1 className="text-3xl font-bold text-gray-800 mb-6 text-center">Sitecore Content Scraper & Indexer</h1>

        {/* Link to Query Page */}
        <div className="text-center mb-6">
            <Link href="/query" className="text-blue-600 hover:underline text-lg font-medium">
                Go to AI Content Search &rarr;
            </Link>
        </div>

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4" role="alert">
            <strong className="font-bold">Error:</strong>
            <span className="block sm:inline"> {error}</span>
          </div>
        )}
        {message && (
          <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded relative mb-4" role="alert">
            <strong className="font-bold">Success:</strong>
            <span className="block sm:inline"> {message}</span>
          </div>
        )}

        {/* Add New Environment Form */}
        <div className="mb-8 p-6 border rounded-lg bg-gray-50">
          <h2 className="text-2xl font-semibold text-gray-700 mb-4">Add New Sitecore Environment</h2>
          <form onSubmit={handleAddEnvironment} className="space-y-4">
            <div>
              <label htmlFor="envName" className="block text-sm font-medium text-gray-700">Environment Name</label>
              <input
                type="text"
                id="envName"
                className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                value={newEnvName}
                onChange={(e) => setNewEnvName(e.target.value)}
                required
              />
            </div>
            <div>
              <label htmlFor="graphqlEndpoint" className="block text-sm font-medium text-gray-700">GraphQL Endpoint</label>
              <input
                type="url"
                id="graphqlEndpoint"
                className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                value={newEnvGraphQLEndpoint}
                onChange={(e) => setNewEnvGraphQLEndpoint(e.target.value)}
                placeholder="e.g., https://your-sitecore-instance/sitecore/api/graph/edge"
                required
              />
            </div>
            <div>
              <label htmlFor="apiKey" className="block text-sm font-medium text-gray-700">API Key</label>
              <input
                type="text"
                id="apiKey"
                className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                value={newEnvApiKey}
                onChange={(e) => setNewEnvApiKey(e.target.value)}
                required
              />
            </div>
            <div>
              <label htmlFor="rootPath" className="block text-sm font-medium text-gray-700">Root Path</label>
              <input
                type="text"
                id="rootPath"
                className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                value={newEnvRootPath}
                onChange={(e) => setNewEnvRootPath(e.target.value)}
                placeholder="/sitecore/content/your-site/home"
                required
              />
            </div>
            <div>
              <label htmlFor="language" className="block text-sm font-medium text-gray-700">Language</label>
              <input
                type="text"
                id="language"
                className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                value={newEnvLanguage}
                onChange={(e) => setNewEnvLanguage(e.target.value)}
                placeholder="e.g., en"
                required
              />
            </div>
            <button
              type="submit"
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-md shadow-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
            >
              Add Environment
            </button>
          </form>
        </div>

        {/* Existing Environments List */}
        <div className="mb-8">
          <h2 className="text-2xl font-semibold text-gray-700 mb-4">Existing Environments</h2>
          {loading ? (
            <p className="text-gray-600">Loading environments...</p>
          ) : environments.length === 0 ? (
            <p className="text-gray-600">No environments added yet. Add one above!</p>
          ) : (
            <ul className="space-y-4">
              {environments.map((env) => (
                <li
                  key={env.id}
                  className={`p-4 border rounded-lg flex justify-between items-center transition-all duration-200 ${
                    selectedEnvironmentId === env.id ? 'bg-blue-50 border-blue-400 shadow-md' : 'bg-white border-gray-200'
                  }`}
                >
                  <div>
                    <h3 className="text-lg font-medium text-gray-900">{env.name}</h3>
                    <p className="text-sm text-gray-600">URL: {env.graphql_endpoint}</p>
                    <p className="text-sm text-gray-600">Root Path: {env.root_path}</p>
                    <p className="text-sm text-gray-600">Language: {env.language}</p>
                    <p className="text-sm text-gray-600">Status: {env.status}</p>
                  </div>
                  <div className="flex space-x-2">
                    <button
                      onClick={() => setSelectedEnvironmentId(env.id === selectedEnvironmentId ? null : env.id)}
                      className={`px-3 py-1 rounded-md text-sm font-medium ${
                        selectedEnvironmentId === env.id
                          ? 'bg-blue-500 text-white hover:bg-blue-600'
                          : 'bg-gray-200 text-gray-800 hover:bg-gray-300'
                      }`}
                    >
                      {selectedEnvironmentId === env.id ? 'Selected' : 'Select'}
                    </button>
                    <button
                      onClick={() => handleDeleteEnvironment(env.id)}
                      className="px-3 py-1 bg-red-500 hover:bg-red-600 text-white rounded-md text-sm font-medium"
                    >
                      Delete
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* Actions for Selected Environment */}
        {selectedEnvironmentId !== null && (
          <div className="mb-8 p-6 border rounded-lg bg-blue-50">
            <h2 className="text-2xl font-semibold text-blue-800 mb-4">Actions for Selected Environment</h2>
            <p className="text-lg text-blue-700 mb-4">
              Selected: <span className="font-bold">{environments.find(e => e.id === selectedEnvironmentId)?.name}</span>
            </p>
            <div className="flex space-x-4">
              <button
                onClick={handleScrapeContent}
                disabled={isScraping}
                className={`flex-1 py-3 px-6 rounded-md text-lg font-semibold shadow-md focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 transition-colors duration-200 ${
                  isScraping ? 'bg-gray-400 cursor-not-allowed' : 'bg-green-600 hover:bg-green-700 text-white'
                }`}
              >
                {isScraping ? 'Scraping & Indexing...' : 'Scrape & Index Content'}
              </button>
              {/* You might add other actions here, e.g., "Update Index" */}
            </div>
          </div>
        )}

        {/* Real-time Log Window */}
        <div className="mb-8 p-6 border rounded-lg bg-gray-800 text-gray-100 font-mono">
          <h2 className="text-2xl font-semibold text-white mb-4">Real-time Scraping Logs</h2>
          <div id="log-window" className="h-64 overflow-y-auto bg-gray-900 p-4 rounded-md text-sm">
            {logMessages.length === 0 ? (
              <p className="text-gray-500">Logs will appear here during scraping...</p>
            ) : (
              logMessages.map((log, index) => (
                <p key={index} className="mb-1 last:mb-0 break-words whitespace-pre-wrap">{log}</p>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}