export type CodeOption = 'Canada' | 'US' | 'Philippines';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  text: string;
}

export interface Assumption {
  field: string;
  value: string;
  reason: string;
}

export interface ProjectState {
  project_name: string;
  code: CodeOption;
  stories: number;
  plan_x_m: number;
  plan_y_m: number;
  story_heights_m: number[];
  grid_x_m: number[];
  grid_y_m: number[];
  material_system: string;
  fc_mpa: number;
  fy_mpa: number;
  sbc_kpa: number;
  support_type: string;
  diaphragm_type: string;
  coordinate_mode: boolean;
  raw_input: string;
}

export interface MemberSummary {
  name: string;
  type: 'beam' | 'column';
  max_shear_kN: number;
  max_moment_kNm: number;
  axial_kN: number;
  deflection_mm?: number;
  group?: string;
}

export interface AnalysisResults {
  roof_drift_x_mm: number;
  roof_drift_y_mm: number;
  story_drift_max_mm: number;
  base_shear_x_kN: number;
  base_shear_y_kN: number;
  beam_moment_max_kNm: number;
  beam_shear_max_kN: number;
  column_axial_max_kN: number;
  joint_reaction_max_kN: number;
  period_1_s: number;
  governing_direction: string;
  confidence: string;
  recommendations: string[];
  conclusion: string;
  members: MemberSummary[];
  assumptions: Assumption[];
}

export interface ApiEnvelope {
  project: ProjectState;
  analysis: AnalysisResults;
  follow_up_question: string;
}
