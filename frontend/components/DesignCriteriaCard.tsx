"use client";

import type { FeaDesignCriteria } from "@/lib/api";

type Props = { criteria: FeaDesignCriteria };

function fmt(v: number, decimals = 2): string {
  if (!Number.isFinite(v)) return "—";
  return v.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

/** Render the location-resolved design criteria as four small tables. */
export default function DesignCriteriaCard({ criteria }: Props) {
  const { matched_location, country, is_assumed, loads, wind, seismic, soil, combos, notes } =
    criteria;

  return (
    <div className="design-criteria">
      <div className="dc-header">
        <div>
          <strong className="dc-title">Design criteria</strong>
          <div className="small-muted">
            {matched_location ? (
              <>
                Resolved for <strong>{matched_location}</strong>
                {country ? `, ${country}` : ""} {is_assumed ? " · ASSUMED FALLBACK" : ""}
              </>
            ) : (
              <>Generic moderate parameters · ASSUMED FALLBACK (no location supplied)</>
            )}
          </div>
        </div>
        {is_assumed ? (
          <span className="dc-badge dc-badge-assumed" title="Generic fallback values">
            assumed
          </span>
        ) : (
          <span className="dc-badge dc-badge-resolved" title="Loaded from built-in code table">
            location-resolved
          </span>
        )}
      </div>

      <div className="dc-grid">
        {loads ? (
          <section className="dc-section">
            <h4 className="dc-h">Gravity loads</h4>
            <table className="dc-table">
              <tbody>
                <tr>
                  <th>Dead load (DL)</th>
                  <td>{fmt(loads.dl_kpa, 2)}</td>
                  <td className="dc-unit">kPa</td>
                </tr>
                <tr>
                  <th>Live load (LL)</th>
                  <td>{fmt(loads.ll_kpa, 2)}</td>
                  <td className="dc-unit">kPa</td>
                </tr>
                <tr>
                  <th>Snow load</th>
                  <td>{fmt(loads.snow_kpa, 2)}</td>
                  <td className="dc-unit">kPa</td>
                </tr>
              </tbody>
            </table>
            {loads.notes ? <p className="dc-note small-muted">{loads.notes}</p> : null}
          </section>
        ) : null}

        {wind ? (
          <section className="dc-section">
            <h4 className="dc-h">Wind</h4>
            <table className="dc-table">
              <tbody>
                <tr>
                  <th>Design wind speed V</th>
                  <td>{fmt(wind.design_wind_speed_mps, 0)}</td>
                  <td className="dc-unit">m/s (3-s gust, 50-yr)</td>
                </tr>
                <tr>
                  <th>Velocity pressure q</th>
                  <td>{fmt(wind.pressure_kpa, 2)}</td>
                  <td className="dc-unit">kPa</td>
                </tr>
                <tr>
                  <th>Exposure category</th>
                  <td>{wind.exposure_category}</td>
                  <td className="dc-unit">—</td>
                </tr>
                <tr>
                  <th>Importance factor I</th>
                  <td>{fmt(wind.importance_factor, 2)}</td>
                  <td className="dc-unit">—</td>
                </tr>
              </tbody>
            </table>
            <p className="dc-source small-muted">
              Source: <em>{wind.code_basis || "—"}</em>
            </p>
          </section>
        ) : null}

        {seismic ? (
          <section className="dc-section">
            <h4 className="dc-h">Seismic</h4>
            <table className="dc-table">
              <tbody>
                <tr>
                  <th>Seismic zone</th>
                  <td>{seismic.zone}</td>
                  <td className="dc-unit">—</td>
                </tr>
                <tr>
                  <th>Peak ground acc.</th>
                  <td>{fmt(seismic.pga_g, 2)}</td>
                  <td className="dc-unit">g</td>
                </tr>
                <tr>
                  <th>Base shear V/W</th>
                  <td>{fmt(seismic.base_shear_coeff, 3)}</td>
                  <td className="dc-unit">—</td>
                </tr>
                <tr>
                  <th>Site class</th>
                  <td>{seismic.site_class}</td>
                  <td className="dc-unit">—</td>
                </tr>
              </tbody>
            </table>
            <p className="dc-source small-muted">
              Source: <em>{seismic.code_basis || "—"}</em>
            </p>
          </section>
        ) : null}

        {soil ? (
          <section className="dc-section">
            <h4 className="dc-h">Soil bearing</h4>
            <table className="dc-table">
              <tbody>
                <tr>
                  <th>Allowable SBC</th>
                  <td>{fmt(soil.sbc_kpa, 0)}</td>
                  <td className="dc-unit">kPa</td>
                </tr>
                <tr>
                  <th>Description</th>
                  <td colSpan={2}>{soil.description || "—"}</td>
                </tr>
              </tbody>
            </table>
            <p className="dc-source small-muted">
              Source: <em>{soil.code_basis || "—"}</em>
            </p>
          </section>
        ) : null}
      </div>

      {combos ? (
        <section className="dc-section dc-combos">
          <h4 className="dc-h">Load combinations</h4>
          <ul className="dc-combo-list">
            {combos.uls.map((c) => (
              <li key={c} className="dc-combo dc-combo-uls" title="Ultimate limit state">
                <span className="dc-combo-tag">ULS</span> {c}
              </li>
            ))}
            {combos.sls.map((c) => (
              <li key={c} className="dc-combo dc-combo-sls" title="Serviceability limit state">
                <span className="dc-combo-tag">SLS</span> {c}
              </li>
            ))}
          </ul>
          <p className="dc-source small-muted">
            Solver runs the governing envelope: <strong>{combos.governing}</strong>
          </p>
        </section>
      ) : null}

      {notes && notes.length ? (
        <ul className="dc-notes small-muted">
          {notes.map((n, i) => (
            <li key={i}>{n}</li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
