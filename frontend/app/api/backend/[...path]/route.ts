import { type NextRequest } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function parseBackendEnv(): { proxyUrl: string; apiKey: string } {
  const rawProxy = process.env.BACKEND_PROXY_URL?.trim() || "";
  // Be forgiving if both values were accidentally pasted into BACKEND_PROXY_URL.
  const embeddedUrl = rawProxy.match(/https?:\/\/[^\s]+/)?.[0];
  const embeddedKey = rawProxy.match(/BACKEND_API_KEY\s*=\s*([^\s]+)/)?.[1];

  return {
    proxyUrl: (embeddedUrl || rawProxy || "http://127.0.0.1:8000").replace(/\/$/, ""),
    apiKey: process.env.BACKEND_API_KEY?.trim() || embeddedKey || "",
  };
}

const { proxyUrl: BACKEND_PROXY_URL, apiKey: BACKEND_API_KEY } = parseBackendEnv();

function upstreamUrl(path: string[], search: string): string {
  const safePath = path.map((part) => encodeURIComponent(part)).join("/");
  return `${BACKEND_PROXY_URL}/${safePath}${search}`;
}

function proxyHeaders(req: NextRequest): Headers {
  const headers = new Headers();
  const contentType = req.headers.get("content-type");
  const accept = req.headers.get("accept");

  if (contentType) headers.set("content-type", contentType);
  if (accept) headers.set("accept", accept);
  if (BACKEND_API_KEY) headers.set("x-api-key", BACKEND_API_KEY);

  return headers;
}

async function proxy(
  req: NextRequest,
  context: { params: { path?: string[] } },
): Promise<Response> {
  const method = req.method.toUpperCase();
  const url = upstreamUrl(context.params.path ?? [], req.nextUrl.search);
  const body = method === "GET" || method === "HEAD" ? undefined : await req.arrayBuffer();

  const upstream = await fetch(url, {
    method,
    headers: proxyHeaders(req),
    body,
    cache: "no-store",
  });

  const headers = new Headers(upstream.headers);
  headers.delete("content-encoding");
  headers.delete("content-length");

  return new Response(upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers,
  });
}

export async function GET(
  req: NextRequest,
  context: { params: { path?: string[] } },
): Promise<Response> {
  return proxy(req, context);
}

export async function POST(
  req: NextRequest,
  context: { params: { path?: string[] } },
): Promise<Response> {
  return proxy(req, context);
}

export async function PUT(
  req: NextRequest,
  context: { params: { path?: string[] } },
): Promise<Response> {
  return proxy(req, context);
}

export async function PATCH(
  req: NextRequest,
  context: { params: { path?: string[] } },
): Promise<Response> {
  return proxy(req, context);
}

export async function DELETE(
  req: NextRequest,
  context: { params: { path?: string[] } },
): Promise<Response> {
  return proxy(req, context);
}
