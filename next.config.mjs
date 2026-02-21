/** @type {import('next').NextConfig} */
const nextConfig = {
    async rewrites() {
        const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

        return [
            // Exclude /api/auth from being proxied to the python backend
            {
                source: '/api/((?!auth).*)',
                destination: `${BACKEND_URL}/api/:1`,
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
