'use client';

import { Assumption, ProjectState } from '@/types';

interface Props {
  project?: ProjectState;
  assumptions: Assumption[];
  followUp?: string;
}

export default function AssumptionPanel({ project, assumptions, followUp }: Props) {
  return (
    <section className="panel assumptionsPanel">
      <div className="panelTitle">Project state and assumptions</div>

      {project ? (
        <div className="kvGrid">
          <div><span>Name</span><strong>{project.project_name}</strong></div>
          <div><span>Stories</span><strong>{project.stories}</strong></div>
          <div><span>Plan X</span><strong>{project.plan_x_m} m</strong></div>
          <div><span>Plan Y</span><strong>{project.plan_y_m} m</strong></div>
          <div><span>Material</span><strong>{project.material_system}</strong></div>
          <div><span>fc</span><strong>{project.fc_mpa} MPa</strong></div>
          <div><span>fy</span><strong>{project.fy_mpa} MPa</strong></div>
          <div><span>SBC</span><strong>{project.sbc_kpa} kPa</strong></div>
          <div><span>Supports</span><strong>{project.support_type}</strong></div>
          <div><span>Diaphragm</span><strong>{project.diaphragm_type}</strong></div>
        </div>
      ) : (
        <div className="emptyState">Submit a project prompt to populate the canonical project state.</div>
      )}

      <div className="sectionBlock">
        <h3>Assumptions used</h3>
        {assumptions.length ? assumptions.map((a, i) => (
          <div className="assumption" key={`${a.field}-${i}`}>
            <strong>{a.field}:</strong> {a.value}
            <div className="subtle">{a.reason}</div>
          </div>
        )) : <div className="emptyState">No assumptions yet.</div>}
      </div>

      {followUp ? (
        <div className="sectionBlock">
          <h3>Suggested follow-up</h3>
          <div className="followUp">{followUp}</div>
        </div>
      ) : null}
    </section>
  );
}
