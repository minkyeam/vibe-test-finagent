import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    // 백엔드가 배포된 URL (Render 등)을 NEXT_PUBLIC_API_URL로 설정하세요.
    // 설정되지 않은 경우 로컬 서버를 바라봅니다.
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
