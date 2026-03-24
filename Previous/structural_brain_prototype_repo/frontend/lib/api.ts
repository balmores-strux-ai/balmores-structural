import { ApiEnvelope, CodeOption } from '@/types';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? 'http://localhost:8000';

export async function sendProjectPrompt(prompt: string, code: CodeOption): Promise<ApiEnvelope> {
  const response = await fetch(`${API_BASE}/api/chat/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, code })
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || 'Failed to analyze project');
  }

  return response.json();
}

export async function requestVerification(project: unknown) {
  const response = await fetch(`${API_BASE}/api/verification/jobs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(project)
  });

  if (!response.ok) {
    throw new Error('Failed to start verification job');
  }

  return response.json();
}
