'use client';

import { AnalysisResults } from '@/types';

interface Props {
  analysis?: AnalysisResults;
}

export default function ResultPanel({ analysis }: Props) {
  if (!analysis) {
    return (
      <section className="panel resultPanel">
        <div className="panelTitle">Analysis output</div>
        <div className="emptyState">Results will appear here immediately after inference.</div>
      </section>
    );
  }

  return (
    <section className="panel resultPanel">
      <div className="panelTitle">Analysis output</div>

      <div className="metricGrid">
        <div className="metric"><span>Roof drift X</span><strong>{analysis.roof_drift_x_mm.toFixed(1)} mm</strong></div>
        <div className="metric"><span>Roof drift Y</span><strong>{analysis.roof_drift_y_mm.toFixed(1)} mm</strong></div>
        <div className="metric"><span>Storey drift max</span><strong>{analysis.story_drift_max_mm.toFixed(1)} mm</strong></div>
        <div className="metric"><span>Base shear X</span><strong>{analysis.base_shear_x_kN.toFixed(1)} kN</strong></div>
        <div className="metric"><span>Base shear Y</span><strong>{analysis.base_shear_y_kN.toFixed(1)} kN</strong></div>
        <div className="metric"><span>Beam moment max</span><strong>{analysis.beam_moment_max_kNm.toFixed(1)} kNm</strong></div>
        <div className="metric"><span>Beam shear max</span><strong>{analysis.beam_shear_max_kN.toFixed(1)} kN</strong></div>
        <div className="metric"><span>Column axial max</span><strong>{analysis.column_axial_max_kN.toFixed(1)} kN</strong></div>
        <div className="metric"><span>Joint reaction max</span><strong>{analysis.joint_reaction_max_kN.toFixed(1)} kN</strong></div>
        <div className="metric"><span>Period T1</span><strong>{analysis.period_1_s.toFixed(2)} s</strong></div>
        <div className="metric"><span>Governed by</span><strong>{analysis.governing_direction}</strong></div>
        <div className="metric"><span>Confidence</span><strong>{analysis.confidence}</strong></div>
      </div>

      <div className="sectionBlock">
        <h3>Recommendations</h3>
        <ul className="bulletList">
          {analysis.recommendations.map((item, idx) => <li key={idx}>{item}</li>)}
        </ul>
      </div>

      <div className="sectionBlock">
        <h3>Conclusion</h3>
        <p>{analysis.conclusion}</p>
      </div>

      <div className="sectionBlock">
        <h3>Member summary</h3>
        <div className="memberTableWrap">
          <table className="memberTable">
            <thead>
              <tr>
                <th>Name</th>
                <th>Type</th>
                <th>Shear (kN)</th>
                <th>Moment (kNm)</th>
                <th>Axial (kN)</th>
                <th>Deflection (mm)</th>
                <th>Group</th>
              </tr>
            </thead>
            <tbody>
              {analysis.members.map((m) => (
                <tr key={m.name}>
                  <td>{m.name}</td>
                  <td>{m.type}</td>
                  <td>{m.max_shear_kN.toFixed(1)}</td>
                  <td>{m.max_moment_kNm.toFixed(1)}</td>
                  <td>{m.axial_kN.toFixed(1)}</td>
                  <td>{(m.deflection_mm ?? 0).toFixed(1)}</td>
                  <td>{m.group ?? '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
