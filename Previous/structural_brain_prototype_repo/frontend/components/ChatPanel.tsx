'use client';

import { Message } from '@/types';
import { FormEvent, useState } from 'react';

interface Props {
  messages: Message[];
  onSend: (text: string) => Promise<void>;
  loading: boolean;
}

const starter = `Design a 4 storey reinforced concrete building 12m by 18m, 3 bays each way, fc 28 MPa, fy 415 MPa, SBC 150 kPa, fixed supports, rigid diaphragm. Tell me beam shears, beam moments, column axial forces, joint reactions, beam deflections and storey drift in mm.`;

export default function ChatPanel({ messages, onSend, loading }: Props) {
  const [draft, setDraft] = useState(starter);

  async function submit(e: FormEvent) {
    e.preventDefault();
    const text = draft.trim();
    if (!text || loading) return;
    await onSend(text);
    setDraft('');
  }

  return (
    <section className="panel chatPanel">
      <div className="panelTitle">Project conversation</div>
      <div className="messages">
        {messages.map((m) => (
          <div key={m.id} className={`message ${m.role}`}>
            <div className="messageRole">{m.role === 'user' ? 'You' : 'Structural Brain'}</div>
            <div className="messageText">{m.text}</div>
          </div>
        ))}
      </div>

      <form onSubmit={submit} className="composer">
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Describe the building in plain English, coordinates, or mixed text..."
          rows={7}
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Thinking…' : 'Analyze'}
        </button>
      </form>
    </section>
  );
}
