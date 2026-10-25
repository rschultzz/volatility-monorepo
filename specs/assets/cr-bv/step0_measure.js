// CR-BV Step 0 — DOM measurement snippets (paste into the page console; read-only).
// Price chart (react_price_preview, landscape panel open). Reads the CURRENT React tree from #root
// (a DOM node's __reactFiber may be a stale alternate), the candlestick series from PriceChart's hooks,
// and compares rendered panel elements (tick labels: text y = yOf + 3; amber spot line y1) with
// series.priceToCoordinate on the chart pane (lightweight-charts pane <td>, getBoundingClientRect).
window.__crbvRendered = () => {
  const rootEl = document.getElementById('root'); const ck = Object.keys(rootEl).find(k => k.startsWith('__reactContainer'));
  const stack = [rootEl[ck].stateNode.current]; let gl = null;
  while (stack.length) { const f = stack.pop(); if (!f) continue; if (f.type && f.type.name === 'GexLandscape') { gl = f.memoizedProps; break; } if (f.sibling) stack.push(f.sibling); if (f.child) stack.push(f.child); }
  const el = document.querySelector('.tv-lightweight-charts'); const key = Object.keys(el.parentElement).find(k => k.startsWith('__reactFiber'));
  const s = (() => { let f = el.parentElement[key], d = 0; while (f && d < 30) { let h = f.memoizedState; while (h) { const v = h.memoizedState; const c = v && typeof v === 'object' && 'current' in v ? v.current : null; if (c && typeof c.priceToCoordinate === 'function' && c.seriesType() === 'Candlestick') return c; h = h.next; } f = f.return; d++; } })();
  const paneRect = [...el.querySelectorAll('td')][1].getBoundingClientRect();
  const svg = [...document.querySelectorAll('svg')].find(x => x.getBoundingClientRect().width > 200 && x.getBoundingClientRect().left > paneRect.right - 5);
  const svgTop = svg.getBoundingClientRect().top;
  const ticks = [...svg.querySelectorAll('text')].filter(t => /^\d{4}(\.\d+)?$/.test(t.textContent.trim())).map(t => {
    const p = +t.textContent; const yPanel = svgTop + (+t.getAttribute('y')) - 3; const yChart = paneRect.top + s.priceToCoordinate(p);
    return { price: p, y_panel_page: +yPanel.toFixed(2), y_chart_page: +yChart.toFixed(2), err_px: +(yPanel - yChart).toFixed(2) };
  });
  const spotLine = [...svg.querySelectorAll('line')].find(l => l.getAttribute('stroke') === '#fbbf24' || l.getAttribute('stroke-dasharray'));
  const spot = gl.data.spot;
  const spotErr = spotLine ? +((svgTop + (+spotLine.getAttribute('y1'))) - (paneRect.top + s.priceToCoordinate(spot))).toFixed(2) : null;
  return { viewport: [innerWidth, innerHeight], trade_date: gl.data.trade_date, chartPaneTop: paneRect.top, chartPaneHeight: paneRect.height,
    panelSvgTop: svgTop, bodyOffsetTop: svg.parentElement.offsetTop, published: gl.visiblePriceRange, ticks, spot, spotErr_px: spotErr, accuracy: gl.data.accuracy };
};
JSON.stringify(window.__crbvRendered());
