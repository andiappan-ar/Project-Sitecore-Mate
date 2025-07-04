// next.config.ts

import type { NextConfig } from 'next'; // Import NextConfig type

const nextConfig: NextConfig = {
  // Set reactStrictMode to true for development warnings and best practices
  reactStrictMode: true,
  // Removed 'swcMinify' as it's an unrecognized key in Next.js 15.3.4
  // Next.js 15+ handles minification by default.
  compiler: {
    // Keep this empty or add other valid compiler options if you have them.
    // Ensure any options here are recognized by your specific Next.js version.
  },
  webpack: (config, { dev, webpack }) => {
    if (dev) {
      config.experiments = {
        ...config.experiments,
        topLevelAwait: true,
      };
    }
    return config;
  },
};

export default nextConfig;
