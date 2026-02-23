/** @type {import('next').NextConfig} */
const nextConfig = {
    async rewrites() {
        const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

        return [
            // Exclude /api/auth from being proxied to the python backend
            {
                source: '/api/:path((?!auth).*)',
                destination: `${BACKEND_URL}/api/:path`,
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
