"use client";

import { useEffect, useRef, useState } from "react";
import type { FeaProgressEvent } from "@/lib/api";

type LogLine = { id: string; text: string; ts: number };

type Props = {
  /** Latest progress event from the stream (set by parent). */
  event: FeaProgressEvent | null;
  /** Whether the analysis is currently running. */
  active: boolean;
  /** Optional fallback total seconds when the server doesn't broadcast it. */
  fallbackTotal?: number;
};

/**
 * STAAD-style progress overlay. Shows:
 *   - A live counter "12.4 s of est. 38 s"
 *   - A wide gradient progress bar (0..1)
 *   - A scrolling log of stage transitions
 *   - The current named stage in bold
 *
 * Designed so the user knows the engine is *working* even when a 60-storey
 * model takes 45 seconds to crunch.
 */
export default function AnalysisProgress({ event, active, fallbackTotal }: Props) {
  const [progress, setProgress] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [estimated, setEstimated] = useState<number | undefined>(fallbackTotal);
  const [stage, setStage] = useState<string>("Initialising solver");
  const [stageKey, setStageKey] = useState<string>("init");
  const [log, setLog] = useState<LogLine[]>([]);
  const startedAtRef = useRef<number | null>(null);
  const tickerRef = useRef<number | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!active) {
      if (tickerRef.current) {
        window.clearInterval(tickerRef.current);
        tickerRef.current = null;
      }
      startedAtRef.current = null;
      return;
    }
    if (startedAtRef.current == null) {
      startedAtRef.current = performance.now();
      setProgress(0);
      setElapsed(0);
      setStage("Connecting to PyNite kernel");
      setStageKey("connect");
      setLog([
        { id: `${Date.now()}-init`, text: "Connecting to PyNite kernel\u2026", ts: Date.now() },
      ]);
    }
    if (tickerRef.current == null) {
      tickerRef.current = window.setInterval(() => {
        if (startedAtRef.current == null) return;
        const e = (performance.now() - startedAtRef.current) / 1000;
        setElapsed(e);
        setProgress((p) => {
          if (estimated && estimated > 0) {
            const target = Math.min(0.97, e / estimated);
            return Math.max(p, target);
          }
          return Math.min(0.95, p + 0.005);
        });
      }, 250);
    }
    return () => {
      if (tickerRef.current) {
        window.clearInterval(tickerRef.current);
        tickerRef.current = null;
      }
    };
  }, [active, estimated]);

  useEffect(() => {
    if (!event) return;
    if (event.type === "stage") {
      setStage(event.label);
      setStageKey(event.stage);
      if (typeof event.progress === "number") setProgress(event.progress);
      if (typeof event.elapsed_seconds === "number") setElapsed(event.elapsed_seconds);
      if (typeof event.estimated_total_seconds === "number")
        setEstimated(event.estimated_total_seconds);
      setLog((prev) => [
        ...prev,
        { id: `${Date.now()}-${event.stage}`, text: event.label, ts: Date.now() },
      ]);
    } else if (event.type === "tick") {
      if (typeof event.progress === "number") setProgress(event.progress);
      if (typeof event.elapsed_seconds === "number") setElapsed(event.elapsed_seconds);
      if (typeof event.estimated_total_seconds === "number")
        setEstimated(event.estimated_total_seconds);
    } else if (event.type === "error") {
      setStage(`Solver error: ${event.message}`);
      setStageKey("error");
    }
  }, [event]);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [log.length]);

  if (!active) return null;

  const pct = Math.round(Math.min(1, Math.max(0, progress)) * 100);
  const showEstimated =
    typeof estimated === "number" && Number.isFinite(estimated) && estimated > 0;

  return (
    <div className="analysis-progress" role="status" aria-live="polite">
      <div className="ap-head">
        <div className="ap-title">
          <span className="ap-spinner" aria-hidden />
          Running PyNite FEM
        </div>
        <div className="ap-meta">
          <span className="ap-pct">{pct}%</span>
          <span className="ap-clock">
            {elapsed.toFixed(1)} s
            {showEstimated ? <span className="ap-est"> / est. {estimated!.toFixed(0)} s</span> : null}
          </span>
        </div>
      </div>

      <div className="ap-bar" aria-label="Solver progress">
        <div
          className="ap-bar-fill"
          style={{ width: `${pct}%` }}
          data-stage={stageKey}
        />
      </div>

      <div className="ap-stage">
        <span className="ap-stage-dot" data-stage={stageKey} aria-hidden />
        <strong>{stage}</strong>
      </div>

      <ol className="ap-log" aria-label="Solver activity">
        {log.map((l) => (
          <li key={l.id}>
            <code className="ap-log-time">
              {new Date(l.ts).toLocaleTimeString([], { hour12: false })}
            </code>
            <span>{l.text}</span>
          </li>
        ))}
        <li ref={logEndRef as unknown as React.RefObject<HTMLLIElement>} />
      </ol>

      <p className="ap-hint small-muted">
        Tip: large 3D buildings (40 +&nbsp;storeys with P-Δ) can take 30–60 s on shared
        hosting. Stay on this page — every member force, drift and reaction will appear here
        the moment PyNite finishes solving.
      </p>
    </div>
  );
}
