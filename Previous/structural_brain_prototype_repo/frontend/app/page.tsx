'use client';

import { useMemo, useState } from 'react';
import HeaderBar from '@/components/HeaderBar';
import ChatPanel from '@/components/ChatPanel';
import ViewerPanel from '@/components/ViewerPanel';
import ResultPanel from '@/components/ResultPanel';
import AssumptionPanel from '@/components/AssumptionPanel';
import { ApiEnvelope, CodeOption, Message } from '@/types';
import { sendProjectPrompt } from '@/lib/api';

function uid() {
  return Math.random().toString(36).slice(2, 10);
}

export default function HomePage() {
  const [code, setCode] = useState<CodeOption>('Canada');
  const [messages, setMessages] = useState<Message[]>([
    {
      id: uid(),
      role: 'assistant',
      text: 'Describe your building in plain language, mixed text, or coordinates. I will parse it, make reasonable assumptions where needed, show the 3D structure, and return structural results immediately.'
    }
  ]);
  const [payload, setPayload] = useState<ApiEnvelope | undefined>();
  const [loading, setLoading] = useState(false);

  async function onSend(text: string) {
    setMessages((prev) => [...prev, { id: uid(), role: 'user', text }]);
    setLoading(true);
    try {
      const response = await sendProjectPrompt(text, code);
      setPayload(response);
      setMessages((prev) => [
        ...prev,
        {
          id: uid(),
          role: 'assistant',
          text:
            `I parsed a ${response.project.stories}-storey ${response.project.material_system} building ` +
            `with plan ${response.project.plan_x_m}m × ${response.project.plan_y_m}m. ` +
            `Predicted max beam moment is ${response.analysis.beam_moment_max_kNm.toFixed(1)} kNm and ` +
            `max storey drift is ${response.analysis.story_drift_max_mm.toFixed(1)} mm.`
        }
      ]);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      setMessages((prev) => [...prev, { id: uid(), role: 'assistant', text: `Error: ${message}` }]);
    } finally {
      setLoading(false);
    }
  }

  const assumptions = useMemo(() => payload?.analysis.assumptions ?? [], [payload]);

  return (
    <main className="workspace">
      <HeaderBar code={code} setCode={setCode} />

      <div className="contentGrid">
        <ChatPanel messages={messages} onSend={onSend} loading={loading} />

        <div className="centerColumn">
          <ViewerPanel project={payload?.project} />
          <AssumptionPanel
            project={payload?.project}
            assumptions={assumptions}
            followUp={payload?.follow_up_question}
          />
        </div>

        <ResultPanel analysis={payload?.analysis} />
      </div>
    </main>
  );
}
