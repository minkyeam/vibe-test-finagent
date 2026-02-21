import NextAuth from "next-auth"
import GoogleProvider from "next-auth/providers/google"
import NaverProvider from "next-auth/providers/naver"

const handler = NextAuth({
    providers: [
        GoogleProvider({
            clientId: process.env.GOOGLE_CLIENT_ID || "",
            clientSecret: process.env.GOOGLE_CLIENT_SECRET || "",
        }),
        NaverProvider({
            clientId: process.env.NAVER_CLIENT_ID || "",
            clientSecret: process.env.NAVER_CLIENT_SECRET || "",
        }),
    ],
    callbacks: {
        async session({ session, token }) {
            if (session.user && token.sub) {
                // You can attach user id here if needed
            }
            return session;
        },
    },
    secret: process.env.NEXTAUTH_SECRET || "finagent-secret-4b4d6b67-9c60-4b2a-bd1e-06ea8f5a28b0",
    debug: true,
})

export { handler as GET, handler as POST }
