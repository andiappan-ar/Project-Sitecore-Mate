// src/app/api/environments/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { getEnvironments, saveEnvironments, Environment } from '@/lib/environments';

/**
 * API handler to get all environments from the environments.json file.
 */
export async function GET() {
  try {
    const environments = getEnvironments();
    return NextResponse.json(environments);
  } catch (error) {
    console.error('Failed to get environments:', error);
    return NextResponse.json({ message: 'Error fetching environments' }, { status: 500 });
  }
}

/**
 * API handler to save the entire list of environments to the environments.json file.
 */
export async function POST(request: NextRequest) {
  try {
    // The body is expected to be the full array of environments
    const environments: Environment[] = await request.json();
    
    if (!Array.isArray(environments)) {
      return NextResponse.json({ message: 'Invalid payload, an array of environments is required.' }, { status: 400 });
    }

    saveEnvironments(environments);
    return NextResponse.json({ message: 'Environments saved successfully.' });
  } catch (error) {
    console.error('Failed to save environments:', error);
    return NextResponse.json({ message: 'Error saving environments' }, { status: 500 });
  }
}
