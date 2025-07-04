'use client';

import { useState, useEffect, useRef } from 'react';
import Link from 'next/link'; // Import Link for navigation

// Define the structure of an Environment
interface Environment {
  id: string;
  name: string;
  url: string;
  apiKey: string;
  rootPath: string;
  languages: string[];
  status?: string; // Optional status from the backend
}

// Define the structure for UI messages
interface UiMessage {
    type: 'success' | 'error' | 'info';
    text: string;
}

export default function Home() {
  // State for managing environments
  const [environments, setEnvironments] = useState<Environment[]>([]);
  
  // State for the "Add Environment" form, now with default values from .env
  const [newEnv, setNewEnv] = useState<Omit<Environment, 'id'>>({
    name: process.env.NEXT_PUBLIC_SITECORE_NAME || 'dev',
    url: process.env.NEXT_PUBLIC_SITECORE_GRAPHQL_URL || '',
    apiKey: process.env.NEXT_PUBLIC_SITECORE_API_KEY || '',
    rootPath: process.env.NEXT_PUBLIC_SITECORE_ROOT_PATH || '',
    languages: (process.env.NEXT_PUBLIC_SITECORE_LANGUAGES || 'en').split(','),
  });

  // State for managing scraping status and logs
  const [scrapingEnvironmentId, setScrapingEnvironmentId] = useState<string | null>(null);
  const [scrapingStatus, setScrapingStatus] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  
  // State for showing user-friendly messages
  const [uiMessage, setUiMessage] = useState<UiMessage | null>(null);

  // Ref to hold the interval ID for polling
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  
  // NEW: Ref for the log display container to enable auto-scrolling
  const logContainerRef = useRef<HTMLPreElement>(null);

  // Load environments from localStorage on initial render
  useEffect(() => {
    const storedEnvs = localStorage.getItem('environments');
    if (storedEnvs) {
      setEnvironments(JSON.parse(storedEnvs));
    }
  }, []);

  // NEW: useEffect to automatically scroll the log container to the bottom
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]); // This effect runs every time the logs array changes

  // Function to display a temporary message in the UI
  const showUiMessage = (type: 'success' | 'error' | 'info', text: string) => {
    setUiMessage({ type, text });
    setTimeout(() => setUiMessage(null), 5000); // Clear message after 5 seconds
  };

  // Function to save environments to localStorage
  const saveEnvironments = (envs: Environment[]) => {
    localStorage.setItem('environments', JSON.stringify(envs));
    setEnvironments(envs);
  };

  // Handler for input changes in the "Add Environment" form
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    if (name === 'languages') {
      setNewEnv({ ...newEnv, languages: value.split(',').map(lang => lang.trim()) });
    } else {
      setNewEnv({ ...newEnv, [name]: value });
    }
  };

  // Handler for adding a new environment
  const handleAddEnvironment = () => {
    if (newEnv.name && newEnv.url && newEnv.apiKey && newEnv.rootPath) {
      const newEnvironmentWithId: Environment = { ...newEnv, id: Date.now().toString() };
      saveEnvironments([...environments, newEnvironmentWithId]);
      // Reset only some fields, keeping others for convenience
      setNewEnv({ ...newEnv, name: '', rootPath: '' }); 
      showUiMessage('success', 'Environment added successfully!');
    } else {
      showUiMessage('error', 'Please fill all fields.');
    }
  };

  // Handler for deleting an environment
  const handleDeleteEnvironment = (id: string) => {
    saveEnvironments(environments.filter(env => env.id !== id));
    showUiMessage('info', 'Environment removed.');
  };

  // Function to fetch the status and logs from the backend
  const fetchStatus = async (environmentId: string) => {
    try {
      const response = await fetch(`/api/scrape?environmentId=${environmentId}`);
      if (response.ok) {
        const data = await response.json();
        setScrapingStatus(data.status);
        setLogs(data.log);

        // Stop polling if the job is done
        if (data.status === 'Completed' || data.status === 'Failed') {
          if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
          }
          if (data.status === 'Completed') {
              showUiMessage('success', 'Scraping process completed successfully.');
          } else {
              showUiMessage('error', 'Scraping process failed. Check logs for details.');
          }
          setScrapingEnvironmentId(null); // Reset for the next job
        }
      }
    } catch (error) {
      console.error('Error fetching scrape status:', error);
       if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
    }
  };
  
  // useEffect to manage the polling interval
  useEffect(() => {
    if (scrapingEnvironmentId) {
      if (intervalRef.current) clearInterval(intervalRef.current);
      intervalRef.current = setInterval(() => fetchStatus(scrapingEnvironmentId), 2000);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [scrapingEnvironmentId]);

  // Handler for the "Scrape" button
  const handleScrape = async (environment: Environment) => {
    setLogs([`Starting scrape for ${environment.name}...`]);
    setScrapingStatus('In Progress');
    setScrapingEnvironmentId(environment.id);
    showUiMessage('info', `Scraping process initiated for ${environment.name}.`);

    try {
      const response = await fetch('/api/scrape', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ environment }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to start scraping process.');
      }
      
      console.log('Scraping process started successfully.');

    } catch (error: any) {
      console.error('Error starting scrape:', error);
      showUiMessage('error', `Error: ${error.message}`);
      setScrapingStatus('Failed');
       if (intervalRef.current) clearInterval(intervalRef.current);
       setScrapingEnvironmentId(null);
    }
  };

  return (
    <main className="container mx-auto p-4 sm:p-6 lg:p-8 font-sans bg-gray-50 min-h-screen">
      <div className="max-w-4xl w-full mx-auto">
        <h1 className="text-4xl font-extrabold text-center text-indigo-700 mb-2">Sitecore Content Scraper</h1>
        <p className="text-center text-gray-500 mb-6">Manage environments and scrape content for vector indexing.</p>
        
        {/* NEW: Navigation Link */}
        <div className="text-center mb-6">
            <Link href="/query" className="text-blue-600 hover:underline text-lg font-medium">
                Go to Query Page &rarr;
            </Link>
        </div>

        {/* NEW: General UI Message Display */}
        {uiMessage && (
            <div className={`p-3 mb-6 rounded-lg text-center font-medium ${
                uiMessage.type === 'success' ? 'bg-green-100 text-green-800' :
                uiMessage.type === 'error' ? 'bg-red-100 text-red-800' :
                'bg-blue-100 text-blue-800'
            }`}>
                {uiMessage.text}
            </div>
        )}

        {/* Add New Environment Form */}
        <div className="bg-gradient-to-br from-blue-50 to-indigo-100 p-6 rounded-lg shadow-md mb-8 border border-blue-200">
          <h2 className="text-2xl font-bold text-indigo-600 mb-4">Add New Environment</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <input type="text" name="name" value={newEnv.name} onChange={handleInputChange} placeholder="Environment Name (e.g., 'dev')" className="p-3 border rounded-md focus:ring-2 focus:ring-blue-500" />
            <input type="text" name="url" value={newEnv.url} onChange={handleInputChange} placeholder="GraphQL Endpoint URL" className="p-3 border rounded-md focus:ring-2 focus:ring-blue-500" />
            <input type="text" name="apiKey" value={newEnv.apiKey} onChange={handleInputChange} placeholder="Sitecore API Key" className="p-3 border rounded-md focus:ring-2 focus:ring-blue-500" />
            <input type="text" name="rootPath" value={newEnv.rootPath} onChange={handleInputChange} placeholder="Root Item Path" className="p-3 border rounded-md focus:ring-2 focus:ring-blue-500" />
            <input type="text" name="languages" value={newEnv.languages.join(', ')} onChange={handleInputChange} placeholder="Languages (comma-separated)" className="p-3 border rounded-md focus:ring-2 focus:ring-blue-500 col-span-1 md:col-span-2" />
          </div>
          <button onClick={handleAddEnvironment} className="mt-4 px-6 py-2 bg-indigo-600 text-white font-semibold rounded-md hover:bg-indigo-700 transition-colors w-full md:w-auto">
            Add Environment
          </button>
        </div>

        {/* Environments List */}
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold text-gray-700 mb-4">Configured Environments</h2>
          {environments.length === 0 ? (
                <p className="text-center text-gray-500 italic">No environments added yet.</p>
            ) : environments.map(env => (
            <div key={env.id} className="bg-white p-4 rounded-lg shadow-md flex flex-col md:flex-row justify-between items-start md:items-center">
              <div className="flex-1 mb-4 md:mb-0">
                <h3 className="text-xl font-bold text-gray-800">{env.name}</h3>
                <p className="text-sm text-gray-500 truncate">URL: {env.url}</p>
                <p className="text-sm text-gray-500">Root: {env.rootPath}</p>
                <p className="text-sm text-gray-500">Languages: {env.languages.join(', ')}</p>
              </div>
              <div className="flex space-x-2">
                <button onClick={() => handleScrape(env)} disabled={scrapingStatus === 'In Progress'} className="px-4 py-2 bg-green-600 text-white font-semibold rounded-md hover:bg-green-700 transition-colors disabled:bg-gray-400 flex items-center justify-center w-36">
                  {scrapingEnvironmentId === env.id ? (
                      <>
                        <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Scraping...
                      </>
                  ) : 'Scrape Content'}
                </button>
                <button onClick={() => handleDeleteEnvironment(env.id)} className="px-4 py-2 bg-red-600 text-white font-semibold rounded-md hover:bg-red-700 transition-colors">
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
        
        {/* Log Display Area */}
        {logs.length > 0 && (
          <div className="mt-8 bg-gray-900 text-white font-mono p-4 rounded-lg shadow-lg">
              <h3 className="text-lg font-semibold mb-2 border-b border-gray-700 pb-2">Scraping Log ({scrapingStatus || 'Waiting...'})</h3>
              {/* Attach the ref to the pre element */}
              <pre ref={logContainerRef} className="overflow-x-auto whitespace-pre-wrap text-sm h-64 overflow-y-scroll">
                  {logs.join('\n')}
              </pre>
          </div>
        )}
      </div>
    </main>
  );
}
