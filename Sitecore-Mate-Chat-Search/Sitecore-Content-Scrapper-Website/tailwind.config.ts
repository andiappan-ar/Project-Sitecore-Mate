// tailwind.config.ts
import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./src/**/*.{js,ts,jsx,tsx,mdx}'],
  theme: {
    extend: {
      // You can add custom theme configurations here if needed
      // For example, custom colors, fonts, spacing, etc.
    },
  },
  plugins: [
    // Add any Tailwind CSS plugins here, e.g., require('@tailwindcss/typography')
  ],
};
export default config;