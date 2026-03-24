/** @type {import('next').NextConfig} */
const backend =
  process.env.BACKEND_PROXY_URL?.replace(/\/$/, "") ||
  "http://127.0.0.1:8000";

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
