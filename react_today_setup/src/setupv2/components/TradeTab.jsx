import { useState } from 'react';
import { gexB, longDate, pct, pctInterval, pts, shortDate, signedPts, strike } from '../format';
import DetailPanel from './DetailPanel';

// ── Price gauge ───────────────────────────────────────────────────────────────

/** Quote, max and fair on one track. Scale: 0 → 1.25 × max(fair, quote, max). */
export function PriceGauge({ quote, max, fair }) {
  const vals = [quote, max, fair].filter(v => v != null).map(Number);
  if (!vals.length) return null;
  const top = Math.max(...vals) * 1.25 || 1;
  const x = v => (v == null ? null : `${Math.max(0, Math.min(100, (Number(v) / top) * 100))}%`);
  return (
    <div className="gauge" data-testid="price-gauge">
      <div className="track">
        {max != null && <div className="under" style={{ width: x(max) }} />}
        {max != null && <div className="over" style={{ left: x(max) }} />}
      </div>
      {quote != null && (
        <>
          <div className="tick q" style={{ left: x(quote) }} />
          <div className="lab strong" style={{ left: x(quote) }}>quote now {pts(quote)}</div>
        </>
      )}
      {max != null && (
        <>
          <div className="tick" style={{ left: x(max) }} />
          <div className="lab below" style={{ left: x(max) }}>max {pts(max, 1)}</div>
        </>
      )}
      {fair != null && (
        <>
          <div className="tick fair" style={{ left: x(fair) }} />
          <div className="lab below" style={{ left: x(fair), marginTop: 14 }}>fair {pts(fair, 1)}</div>
        </>
      )}
    </div>
  );
}

// ── Verdict ───────────────────────────────────────────────────────────────────

export function Verdict({ verdict }) {
  if (!verdict) return null;
  const cls = verdict.code === 'enter' ? '' : verdict.code === 'skip' ? ' skip' : ' none';
  return <span className={`verdict${cls}`} data-testid="verdict" data-code={verdict.code}>{verdict.text}</span>;
}

// ── Band strip ────────────────────────────────────────────────────────────────

const BAND_LABEL = { near: 'near <1.5 IM', mid: 'mid 1.5–2.0', far: 'far >2.0' };

function bandClass(rows, row) {
  // best win rate → s3 (green), worst → s1 (red), middle → s2
  const wins = rows.map(r => r.win_rate ?? -1);
  const best = Math.max(...wins);
  const worst = Math.min(...wins);
  if (row.win_rate === best) return 's3';
  if (row.win_rate === worst) return 's1';
  return 's2';
}

export function BandStrip({ band }) {
  const rows = band?.rows || [];
  if (!rows.length) return <p className="fact muted" data-testid="band-strip">No reference cells persisted yet.</p>;
  return (
    <div className="band" data-testid="band-strip">
      {rows.map(r => (
        <div key={r.band} className={`bk ${bandClass(rows, r)}${band.today === r.band ? ' today' : ''}`} data-testid={`band-${r.band}`}>
          {BAND_LABEL[r.band] || r.band}
          <span>{signedPts(r.mean_pnl)} · {pct(r.win_rate)} win · n {r.n}</span>
        </div>
      ))}
    </div>
  );
}

function bandFact(band) {
  const today = band?.today_cell;
  if (!band?.today || !today) {
    return band?.sigma != null
      ? `Today's wall is ${pts(band.sigma, 2)} IM from the open — no reference cell for that band yet.`
      : 'No band today — the wall distance could not be measured against the open straddle.';
  }
  const rows = band.rows || [];
  const bestWin = Math.max(...rows.map(r => r.win_rate ?? -1));
  const rank = today.win_rate === bestWin ? 'the band with the best win rate' : 'not the band with the best win rate';
  const lb = today.wilson_lo != null ? Math.round(today.wilson_lo * 100) : null;
  const coin = lb == null ? '' : lb > 50 ? ', interval clear of a coin flip' : ', interval includes a coin flip';
  return `Today is ${band.today} (wall ${pts(band.sigma, 2)} IM above the open): ${rank}${coin}. Mean P&L per trade in points, hold to close, threshold ${pts(today.threshold, 2)}.`;
}

// ── Trade tab ─────────────────────────────────────────────────────────────────

export default function TradeTab({ card, apiBase }) {
  const { structure = {}, quote = {}, fair_value: fv = {}, pnl = {}, verdict, band = {}, analogues = {}, manage = {}, stamp = {}, wall = {} } = card;
  const legs = structure.legs || [];
  const shortLeg = legs.find(l => l.side === 'short');
  const longLeg = legs.find(l => l.side === 'long');
  const listed = !!structure.listed;
  const dir = structure.direction === 'put' ? 'put' : 'call';
  const n = fv.n ?? analogues.k_with_outcomes ?? 0;
  const k = analogues.k ?? 0;
  const quoteTag = quote.net_debit == null
    ? { cls: 'bad', text: `No valid quote at ${quote.entry_minute_pt || '06:34'}` }
    : quote.stale_quote
      ? { cls: 'warn', text: `Quote ${quote.quote_minute} (stale)` }
      : { cls: '', text: `Quote ${quote.quote_minute || quote.entry_minute_pt}, clean` };
  const expiry = structure.expiry;
  const holdCell = band.today_cell;

  return (
    <div className="pane on" data-testid="trade-tab">
      <div className="top">
        <div>
          <div className="kicker">Debit {dir} spread to the wall</div>
          {listed ? (
            <>
              <div className="name" data-testid="structure-name">
                Buy {strike(longLeg?.strike_spx)} / sell {strike(shortLeg?.strike_spx)}, {shortDate(expiry)}
              </div>
              <table className="legs">
                <tbody>
                  <tr><td>Buy</td><td>{strike(longLeg?.strike_spx)} {dir}</td><td>ES {pts(longLeg?.strike_es, 0)}</td><td className="muted">{longLeg?.mid != null ? `mid ${pts(longLeg.mid)}` : ''}</td></tr>
                  <tr><td>Sell</td><td>{strike(shortLeg?.strike_spx)} {dir}</td><td>ES {pts(shortLeg?.strike_es, 0)}</td><td className="muted">{shortLeg?.mid != null ? `mid ${pts(shortLeg.mid)}` : ''}</td></tr>
                </tbody>
              </table>
              <span className="tag ok">Listed on today's chain</span>
              <span className="tag">Width {pts(structure.width_actual ?? structure.width_nominal, 0)}{structure.width_actual != null && structure.width_nominal != null && structure.width_actual !== structure.width_nominal ? ` (intent ${pts(structure.width_nominal, 0)})` : ''}</span>
              <span className="tag">{structure.dte_calendar} DTE</span>
              <span className={`tag ${quoteTag.cls}`} data-testid="quote-tag">{quoteTag.text}</span>
            </>
          ) : (
            <div className="name" data-testid="no-listed-structure">No listed structure at this expiry</div>
          )}
          <div className="kv">
            <span className="k">Wall</span>
            <span className="v">{pts(wall.price_es, 0)} · {gexB(wall.gex_b)} · {wall.sigma != null ? `${pts(wall.sigma, 2)} implied move ${wall.above_spot === false ? 'below' : 'above'} spot ${pts(card.context?.spot, 0)}` : `spot ${pts(card.context?.spot, 0)}`}</span>
            {listed && (
              <>
                <span className="k">Pays in full if</span>
                <span className="v">SPX {dir === 'call' ? '≥' : '≤'} {strike(shortLeg?.strike_spx)} at close on {shortDate(expiry)}</span>
                <span className="k">Pays partly if</span>
                <span className="v">SPX between {strike(Math.min(longLeg?.strike_spx ?? 0, shortLeg?.strike_spx ?? 0))} and {strike(Math.max(longLeg?.strike_spx ?? 0, shortLeg?.strike_spx ?? 0))}</span>
              </>
            )}
          </div>
        </div>

        <div>
          <div className="kicker">Max price to pay</div>
          <div className="big" data-testid="max-price">{fv.max_price != null ? pts(fv.max_price, 1) : '—'}<span className="u">pt · conservative fair value</span></div>
          <PriceGauge quote={quote.net_debit} max={fv.max_price} fair={fv.fair} />
          <Verdict verdict={verdict} />
          <div className="ev">
            <div className="box">
              <div className="l">Fair value at close</div>
              <div className="n" data-testid="fair-value">{fv.fair != null ? pts(fv.fair, 1) : '—'}</div>
              <div className="s">
                mean of this spread's value at the {n} analogue closes at T+15 sessions — {k} analogues within the similarity ceiling, {fv.n_computed ?? analogues.k_with_outcomes ?? n} with a computed outcome
                {fv.n_no_t15_close ? `, ${fv.n_no_t15_close} without a T+15 close excluded` : ''}
                {fv.n_no_horizon_close ? `, ${fv.n_no_horizon_close} without a horizon close excluded` : ''}
              </div>
            </div>
            <div className="box">
              <div className="l">Expected P&amp;L at {quote.net_debit != null ? pts(quote.net_debit) : '—'}</div>
              <div className="n" data-testid="expected-pnl">
                {pnl.expected != null ? signedPts(pnl.expected, 1) : '—'}
                {pnl.lo != null && pnl.hi != null && <span className="ci"> [{signedPts(pnl.lo, 1)}, {signedPts(pnl.hi, 1)}]</span>}
              </div>
              <div className="s">fair value minus quote, fees included</div>
            </div>
            <div className="box">
              <div className="l">Market-implied</div>
              <div className="n" data-testid="market-implied">{quote.market_implied != null ? pct(quote.market_implied) : '—'}</div>
              <div className="s">quote ÷ width — the chance the price is charging for</div>
            </div>
          </div>
        </div>
      </div>

      <div className="mid">
        <div>
          <h3>Distance band — where the evidence is</h3>
          <BandStrip band={band} />
          <p className="fact" data-testid="band-fact">{bandFact(band)}</p>
        </div>
        <div>
          <h3>What the {n} analogues did (all from before today)</h3>
          <div className="kv" data-testid="analogue-facts">
            <span className="k">Reached the wall</span>
            <span className="v">{pct(analogues.touch_rate)} <span className="muted">{pctInterval(analogues.touch_ci?.[0], analogues.touch_ci?.[1])}</span>{analogues.mean_days_to_reach != null ? ` · mean ${pts(analogues.mean_days_to_reach, 1)} sessions` : ''}</span>
            <span className="k">Finished above {strike(shortLeg?.strike_spx)}</span>
            <span className="v">{pct(analogues.full_payout_rate)} <span className="muted">— full payout days (at or above the wall at T+15)</span></span>
            <span className="k">Finished above {strike(longLeg?.strike_spx)}</span>
            <span className="v">{pct(analogues.any_payout_rate)} <span className="muted">— any payout (above the long strike at T+15)</span></span>
            <span className="k">Closed at the wall</span>
            <span className="v">{pct(analogues.close_at_wall_rate)} <span className="muted">— within ±0.25 IM</span></span>
            <span className="k">Spread worth at close</span>
            <span className="v">{fv.fair != null ? `${pts(fv.fair, 1)} avg` : '—'} <span className="muted">· at T+15 sessions — that's where "fair value" comes from</span></span>
          </div>
        </div>
      </div>

      <div className="manage">
        <h3>How to run it</h3>
        <ul data-testid="manage">
          <li><span className="dot ok" /><b>Enter at the open if the quote is under {fv.max_price != null ? pts(fv.max_price, 1) : '—'}.</b> Don't wait for a better price{holdCell?.beat_baseline != null ? ` — in the reference cell the edge gate changed the mean by ${signedPts(holdCell.beat_baseline)} pt against first-minute entry` : ''}.</li>
          <li><span className="dot ok" /><b>Hold to the close on {expiry ? longDate(expiry) : '—'}.</b>{holdCell?.mean_pnl != null ? ` The ${band.today} cell made ${signedPts(holdCell.mean_pnl)} pt per trade held to close (n ${holdCell.n}).` : ' No reference cell for today\'s band.'}</li>
          <li><span className="dot ok" /><b>{structure.dte_calendar ?? '—'} DTE.</b> The tested expiry.</li>
          {(manage.watches || []).map(w => (
            <li key={w.key} className="untested" data-testid={`watch-${w.key}`}>
              <span className="dot none" /><b>{w.label} ({w.status}):</b>{' '}
              {w.key === 'vol_state' && w.value?.implied_move_percentile != null
                ? `implied move ${pts(w.value.implied_move, 1)} pt, ${Math.round(w.value.implied_move_percentile)}th percentile of the magnet-above corpus. `
                : ''}
              {w.note}
            </li>
          ))}
        </ul>
      </div>

      <div className="stamp" data-testid="stamp">
        <span>Reference <b>{stamp.cr_id || '—'}</b> · walk-forward, clean quotes, listed strikes</span>
        <span>cell: <b>{stamp.cell || '—'}</b>{stamp.n != null ? ` · n ${stamp.n}` : ''}</span>
        <span>re-run <b>{stamp.rerun_date ? longDate(stamp.rerun_date) : '—'}</b> · next <b>{stamp.next_run ? shortDate(stamp.next_run) : '—'}</b></span>
        <span>fees <b>${pts(stamp.fee_per_contract_per_leg)} / contract / leg</b> included</span>
        <span>quote <b>{stamp.quote_minute_pt} PT</b></span>
      </div>

      <DetailToggle card={card} apiBase={apiBase} />
    </div>
  );
}

function DetailToggle({ card, apiBase }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="detail">
      <button type="button" className="toggle" onClick={() => setOpen(o => !o)} aria-expanded={open} data-testid="detail-toggle">
        {open ? '▾' : '▸'} Detail — post-touch behaviour, quote by minute
      </button>
      {open && <DetailPanel card={card} apiBase={apiBase} />}
    </div>
  );
}
