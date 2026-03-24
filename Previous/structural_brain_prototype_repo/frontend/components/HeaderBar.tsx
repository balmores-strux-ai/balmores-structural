'use client';

import { CodeOption } from '@/types';

interface Props {
  code: CodeOption;
  setCode: (code: CodeOption) => void;
}

export default function HeaderBar({ code, setCode }: Props) {
  return (
    <header className="header">
      <div>
        <div className="eyebrow">Prototype workspace</div>
        <h1>Structural Brain</h1>
        <p className="subtle">Chat-first structural analysis with live 3D preview, assumptions, and report-ready outputs.</p>
      </div>

      <div className="codeSelector">
        <label htmlFor="code">Building code</label>
        <select id="code" value={code} onChange={(e) => setCode(e.target.value as CodeOption)}>
          <option value="Canada">Canada</option>
          <option value="US">US</option>
          <option value="Philippines">Philippines</option>
        </select>
      </div>
    </header>
  );
}
