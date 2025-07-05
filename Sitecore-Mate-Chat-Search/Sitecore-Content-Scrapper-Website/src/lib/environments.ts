// src/lib/environments.ts

import fs from 'fs';
import path from 'path';

export interface Environment {
  id: string;
  name: string;
  graphql_endpoint: string;
  api_key: string;
  root_path: string;
  language: string;
}

// Path to the JSON file that will act as our simple database.
const envFilePath = path.join(process.cwd(), 'environments.json');

/**
 * Ensures the JSON file exists. If not, it creates an empty one.
 */
function ensureEnvFile() {
  if (!fs.existsSync(envFilePath)) {
    fs.writeFileSync(envFilePath, JSON.stringify([]));
  }
}

/**
 * Reads all environments from the JSON file.
 * @returns An array of Environment objects.
 */
export function getEnvironments(): Environment[] {
  ensureEnvFile();
  const fileContents = fs.readFileSync(envFilePath, 'utf8');
  try {
    const data = JSON.parse(fileContents);
    return Array.isArray(data) ? data : [];
  } catch (e) {
    console.error("Could not parse environments.json, returning empty array.");
    return [];
  }
}

/**
 * Writes an array of environments to the JSON file.
 * @param environments The array of Environment objects to save.
 */
export function saveEnvironments(environments: Environment[]) {
  ensureEnvFile();
  fs.writeFileSync(envFilePath, JSON.stringify(environments, null, 2));
}

/**
 * Finds a single environment by its ID.
 * @param id The ID of the environment to find.
 * @returns The found Environment object or undefined.
 */
export function getEnvironmentById(id: string): Environment | undefined {
    const environments = getEnvironments();
    return environments.find(env => env.id === id);
}
