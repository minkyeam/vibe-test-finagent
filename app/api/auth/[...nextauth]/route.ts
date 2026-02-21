import NextAuth from "next-auth"
import GoogleProvider from "next-auth/providers/google"
import NaverProvider from "next-auth/providers/naver"

const handler = NextAuth({
    providers: [
        GoogleProvider({
            clientId: process.env.GOOGLE_CLIENT_ID || "invalid_id",
            clientSecret: process.env.GOOGLE_CLIENT_SECRET || "invalid_secret",
        }),
        NaverProvider({
            clientId: process.env.NAVER_CLIENT_ID || "invalid_id",
            clientSecret: process.env.NAVER_CLIENT_SECRET || "invalid_secret",
        }),
    ],
    secret: process.env.NEXTAUTH_SECRET || "finagent-secret-4b4d6b67-9c60-4b2a-bd1e-06ea8f5a28b0",
    debug: true,
})

export { handler as GET, handler as POST }
