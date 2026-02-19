import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  async rewrites() {
    const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
    return [
      {
        source: '/api/:path*',
        destination: `${BACKEND_URL}/api/:path*`,
      },
      {
        source: '/docs',
        destination: `${BACKEND_URL}/docs`,
      },
      {
        source: '/openapi.json',
        destination: `${BACKEND_URL}/openapi.json`,
      },
    ];
  },
};

export default nextConfig;
