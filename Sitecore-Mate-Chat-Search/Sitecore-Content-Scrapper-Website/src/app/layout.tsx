// src/app/layout.tsx

import './globals.css'; // This line is crucial for loading global styles

import { Inter } from 'next/font/google'; // Import the Inter font from next/font/google

// Configure the Inter font
const inter = Inter({ subsets: ['latin'], variable: '--font-inter' });

// Metadata for your application (optional, but good practice)
export const metadata = {
  title: 'Sitecore Content Scrapper MCP',
  description: 'Manage Sitecore content and RAG',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      {/* Apply the font to the body using its className */}
      <body className={inter.className}>{children}</body>
    </html>
  );
}
