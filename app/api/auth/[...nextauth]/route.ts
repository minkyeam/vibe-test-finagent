import NextAuth from "next-auth"
import GoogleProvider from "next-auth/providers/google"
import NaverProvider from "next-auth/providers/naver"

const providers = [];

// 환경 변수가 Vercel 런타임에 제대로 주입되었는지 확인 후 추가 (안 되어있으면 공급자 목록에서 제외되어 오류 방지)
if (process.env.GOOGLE_CLIENT_ID && process.env.GOOGLE_CLIENT_SECRET) {
    providers.push(GoogleProvider({
        clientId: process.env.GOOGLE_CLIENT_ID,
        clientSecret: process.env.GOOGLE_CLIENT_SECRET,
    }));
}

if (process.env.NAVER_CLIENT_ID && process.env.NAVER_CLIENT_SECRET) {
    providers.push(NaverProvider({
        clientId: process.env.NAVER_CLIENT_ID,
        clientSecret: process.env.NAVER_CLIENT_SECRET,
    }));
}

const handler = NextAuth({
    providers,
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
