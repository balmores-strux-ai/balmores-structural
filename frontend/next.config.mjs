/** @type {import('next').NextConfig} */
function parseBackendEnv() {
  const rawProxy = process.env.BACKEND_PROXY_URL?.trim() || "";
  // Render users sometimes paste:
  //   BACKEND_PROXY_URL=https://...
  //   BACKEND_API_KEY=...
  // into the BACKEND_PROXY_URL value. Extract the URL/key so the build does
  // not fail with an invalid rewrite destination.
  const embeddedUrl = rawProxy.match(/https?:\/\/[^\s]+/)?.[0];
  const embeddedKey = rawProxy.match(/BACKEND_API_KEY\s*=\s*([^\s]+)/)?.[1];
  const backend = (embeddedUrl || rawProxy || "http://127.0.0.1:8000").replace(/\/$/, "");
  const apiKey = process.env.BACKEND_API_KEY?.trim() || embeddedKey || "";

  return {
    backend,
    useSecureApiProxy: Boolean(apiKey),
  };
}

const { backend, useSecureApiProxy } = parseBackendEnv();

const nextConfig = {
  experimental: { serverActions: { allowedOrigins: ["*"] } },
  async redirects() {
    return [
      {
        source: "/:path*",
        has: [{ type: "host", value: "balmoreslab.com" }],
        destination: "https://www.balmoreslab.com/:path*",
        permanent: true,
      },
    ];
  },
  async rewrites() {
    // When BACKEND_API_KEY is set, the App Router proxy at
    // app/api/backend/[...path]/route.ts handles /api/backend/* and injects
    // the key server-side. Do not rewrite directly, otherwise the browser
    // would have to carry a public NEXT_PUBLIC_API_KEY.
    if (useSecureApiProxy) return [];
    return [
      {
        source: "/api/backend/:path*",
        destination: `${backend}/:path*`,
      },
    ];
  },
  async headers() {
    if (process.env.NODE_ENV !== "production") return [];
    return [
      {
        source: "/:path*",
        headers: [
          {
            key: "Strict-Transport-Security",
            value: "max-age=63072000; includeSubDomains; preload",
          },
        ],
      },
    ];
  },
};
export default nextConfig;
