import { dirname } from "path";
import { fileURLToPath } from "url";
import { FlatCompat } from "@eslint/eslintrc";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const compat = new FlatCompat({
  baseDirectory: __dirname,
});

const eslintConfig = [
  ...compat.extends("next/core-web-vitals", "next/typescript"),
  {
    // This object allows you to specify rules for the entire configuration
    // or apply them to specific files using the 'files' property.
    // For disabling a rule globally, you typically put it in an object like this.
    rules: {
      "@typescript-eslint/no-explicit-any": "off", // Disable the no-explicit-any rule
    },
  },
];

export default eslintConfig;
