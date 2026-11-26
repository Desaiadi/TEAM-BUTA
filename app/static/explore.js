/* BUTA Explore: pick one or two NYC buildings on the map, then see what to
 * expect inside them next year and what gets reported on the street around them.
 *
 *   Map      MapLibre with the BUTA style, 2D/3D switch, draggable A/B pins and
 *            the radius circles - the original "Across the street" map.
 *   Snap     A click, drag or search moves a pin to the nearest apartment
 *            building in our data (/api/nearest).
 *   Check    Loads each building's report (/api/building/<bbl>) and the live
 *            311 requests around each pin (NYC Open Data), then renders both.
 *
 * Settings come from window.BUTA, which explore.html fills in from Flask.
 * Chart drawing uses butaChart() and PALETTE from charts.js.
 */
(() => {
  'use strict';

  const CFG = window.BUTA;
  const $ = (id) => document.getElementById(id);
  const KEYS = ['A', 'B'];
  const CATS = ['Noise', 'Housing', 'Sanitation', 'Streets', 'Other'];
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

  const state = {
    mode: 'two',                 // 'one' | 'two'
    active: 'A',                 // the pin a map click moves
    points: {
      A: { lat: 40.7265, lng: -73.9815, name: 'East Village', bbl: null },
      B: { lat: 40.7208, lng: -73.9845, name: 'Lower East Side', bbl: null },
    },
    snapSeq: { A: 0, B: 0 },     // newest nearest-building request per pin
    pending: { A: false, B: false },
    priority: 'all',
    loading: false,
    revision: 0,                 // bumps on every change, so stale results are dropped
    controller: null,
    result: null,
  };

  const activeKeys = () => (state.mode === 'one' ? ['A'] : KEYS);
  const other = (k) => (k === 'A' ? 'B' : 'A');
  const lower = (k) => k.toLowerCase();

  // ------------------------------------------------------------ formatting
  const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ESC[c]);
  const pct = (v) => (v == null ? '–' : `${Math.round(v * 100)}%`);
  const titleCase = (s) => String(s ?? '').toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase());
  const shortLabel = (label) => String(label).split(' (')[0];
  const pinDot = (k) => `<span class="pin-dot ${lower(k)}">${k}</span>`;
  const byType = (report) => Object.fromEntries(report.risks.map((r) => [r.type, r]));

  // ------------------------------------------------------------ map (original implementation)
  const map = new maplibregl.Map({
    container: 'map', style: CFG.styleUrl, center: [-73.983, 40.724], zoom: 15.3, pitch: 52,
    bearing: -24, maxPitch: 65, minZoom: 10, maxZoom: 19,
    maxBounds: [[-74.3, 40.45], [-73.65, 40.95]], attributionControl: true,
  });
  map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'top-right');
  map.addControl(new maplibregl.ScaleControl({ maxWidth: 100, unit: 'metric' }), 'bottom-left');

  function radiusFeature(k) {
    const p = state.points[k], r = Number($('radius').value), ring = [];
    for (let n = 0; n <= 96; n++) {
      const angle = (n / 96) * Math.PI * 2;
      ring.push([
        p.lng + (Math.cos(angle) * r) / (111320 * Math.cos((p.lat * Math.PI) / 180)),
        p.lat + (Math.sin(angle) * r) / 111320,
      ]);
    }
    return { type: 'Feature', properties: { key: k }, geometry: { type: 'Polygon', coordinates: [ring] } };
  }

  function refreshCircles() {
    const src = map.getSource('comparison-areas');
    if (src) src.setData({ type: 'FeatureCollection', features: activeKeys().map(radiusFeature) });
  }

  const markers = {};
  for (const k of KEYS) {
    const el = document.createElement('div');
    el.className = 'marker-shell';
    el.innerHTML = `<div class="map-pin ${lower(k)}">${k}</div><span class="pin-stem"></span><span class="pin-shadow"></span>`;
    el.setAttribute('aria-label', `Building ${k}, drag to move`);
    markers[k] = new maplibregl.Marker({ element: el, draggable: true, anchor: 'bottom' })
      .setLngLat([state.points[k].lng, state.points[k].lat]).addTo(map);
    markers[k].on('dragend', () => place(k, markers[k].getLngLat()));
    $('pick' + k).onclick = () => select(k);
  }

  map.on('load', () => {
    map.addSource('comparison-areas', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
    const before = map.getStyle().layers.find((l) => l.type === 'fill-extrusion')?.id;
    map.addLayer({
      id: 'area-fill', type: 'fill', source: 'comparison-areas',
      paint: { 'fill-color': ['match', ['get', 'key'], 'A', '#178570', '#d87844'], 'fill-opacity': 0.16 },
    }, before);
    map.addLayer({
      id: 'area-outline', type: 'line', source: 'comparison-areas',
      paint: { 'line-color': ['match', ['get', 'key'], 'A', '#178570', '#d87844'], 'line-width': 2, 'line-opacity': 0.8 },
    }, before);
    refreshCircles();
    $('mapnotice').textContent = '';
  });
  map.on('error', () => {
    if (!map.isStyleLoaded()) $('mapnotice').textContent = 'Map tiles are taking longer to load. Check your connection and refresh.';
  });

  function setPerspective(is3d) {
    map.easeTo({ pitch: is3d ? 52 : 0, bearing: is3d ? -24 : 0, duration: reducedMotion ? 0 : 850 });
    for (const [id, on] of [['view2d', !is3d], ['view3d', is3d]]) {
      $(id).classList.toggle('active', on);
      $(id).setAttribute('aria-pressed', on);
    }
  }
  $('view2d').onclick = () => setPerspective(false);
  $('view3d').onclick = () => setPerspective(true);

  $('fitpins').onclick = () => {
    const ks = activeKeys();
    if (ks.length === 1) {
      const p = state.points.A;
      map.flyTo({ center: [p.lng, p.lat], zoom: 16.5, duration: reducedMotion ? 0 : 900 });
      return;
    }
    const bounds = new maplibregl.LngLatBounds();
    ks.forEach((k) => bounds.extend([state.points[k].lng, state.points[k].lat]));
    map.fitBounds(bounds, { padding: { top: 110, bottom: 90, left: 75, right: 75 }, maxZoom: 16, duration: reducedMotion ? 0 : 900 });
  };

  // ------------------------------------------------------------ choosing buildings
  function updateLabel(k) {
    const p = state.points[k];
    $('name' + k).textContent = p.name;
    let line = `${p.lat.toFixed(4)}, ${p.lng.toFixed(4)}`;
    if (state.pending[k]) line = 'Finding the nearest building…';
    else if (p.bbl) line = [p.borough && titleCase(p.borough), p.units && `${p.units} apartments`, `BBL ${p.bbl}`].filter(Boolean).join(' · ');
    else if (CFG.dataReady) line = `No apartment building within 150 m · ${line}`;
    $('coords' + k).textContent = line;
  }

  function select(k) {
    if (state.mode === 'one') k = 'A';
    state.active = k;
    for (const x of KEYS) {
      $('pick' + x).classList.toggle('selected', x === k);
      $('pick' + x).setAttribute('aria-pressed', x === k);
    }
    $('maphint').textContent = state.mode === 'one'
      ? 'Click a building on the map to check it'
      : `Click a building to move location ${k}`;
  }

  function setMode(mode) {
    if (mode === state.mode) return;
    state.mode = mode;
    for (const [id, on] of [['modeOne', mode === 'one'], ['modeTwo', mode === 'two']]) {
      $(id).classList.toggle('active', on);
      $(id).setAttribute('aria-pressed', on);
    }
    $('pickB').hidden = mode === 'one';
    $('fitpins').textContent = mode === 'one' ? '⌖ Center building' : '⌖ Fit both pins';
    $('pickhelp').textContent = mode === 'one'
      ? 'Click a building on the map, drag the pin, or search an address.'
      : 'Choose a pin, then click a building on the map.';
    $('export').textContent = mode === 'one' ? 'Download report' : 'Download comparison';
    if (mode === 'one') markers.B.remove();
    else {
      markers.B.addTo(map);
      if (CFG.dataReady && !state.points.B.bbl && !state.pending.B) snap('B');
    }
    refreshCircles();
    select('A');
    invalidate('Press Check to see ' + (mode === 'one' ? 'this building.' : 'both buildings side by side.'));
  }
  $('modeOne').onclick = () => setMode('one');
  $('modeTwo').onclick = () => setMode('two');

  function checkLabel() {
    return state.mode === 'one' ? 'Check this building' : 'Check these buildings';
  }

  function updateCheckButton() {
    const finding = activeKeys().some((k) => state.pending[k]);
    $('check').disabled = finding || state.loading;
    $('check').textContent = state.loading ? 'Loading reports…' : finding ? 'Finding buildings…' : checkLabel();
  }

  function invalidate(message = 'Your selection changed. Press Check to refresh the results.') {
    state.revision++;
    state.controller?.abort();
    state.loading = false;
    state.result = null;
    $('export').disabled = true;
    $('results').hidden = true;
    $('resultsEmpty').hidden = false;
    $('status').textContent = message;
    updateCheckButton();
  }

  const inNYC = (p) => !(p.lat < 40.49 || p.lat > 40.92 || p.lng < -74.26 || p.lng > -73.68);

  /** Move pin k to a point. `info` is a known building; otherwise snap to one. */
  function place(k, ll, info) {
    if (!inNYC(ll)) {
      $('status').textContent = 'Please choose a location within New York City.';
      markers[k].setLngLat([state.points[k].lng, state.points[k].lat]);
      return;
    }
    state.snapSeq[k]++;                       // drop any snap still in flight
    state.pending[k] = false;
    state.points[k] = info ? { ...info } : { lat: ll.lat, lng: ll.lng, name: 'Selected NYC location', bbl: null };
    markers[k].setLngLat([state.points[k].lng, state.points[k].lat]);
    refreshCircles();
    invalidate();
    if (!info && CFG.dataReady) snap(k);
    else updateLabel(k);
  }

  /** Ask the server for the apartment building nearest pin k and move the pin onto it. */
  async function snap(k) {
    const seq = ++state.snapSeq[k];
    const p = state.points[k];
    state.pending[k] = true;
    updateLabel(k);
    updateCheckButton();
    try {
      const u = new URL(CFG.api.nearest, location.origin);
      u.searchParams.set('lat', p.lat);
      u.searchParams.set('lng', p.lng);
      const r = await fetch(u, { signal: AbortSignal.timeout(15000) });
      if (!r.ok) throw new Error(`nearest building returned ${r.status}`);
      const hit = await r.json();
      if (seq !== state.snapSeq[k]) return;
      if (hit.found) {
        state.points[k] = { lat: hit.lat, lng: hit.lon, name: hit.address || 'Apartment building', bbl: hit.bbl, borough: hit.borough, units: hit.units };
        markers[k].setLngLat([hit.lon, hit.lat]);
        refreshCircles();
      } else {
        state.points[k] = { ...p, name: 'No apartment building here', bbl: null };
      }
    } catch {
      if (seq === state.snapSeq[k]) state.points[k] = { ...p, bbl: null };
    } finally {
      if (seq === state.snapSeq[k]) {
        state.pending[k] = false;
        updateLabel(k);
        updateCheckButton();
      }
    }
  }

  map.on('click', (e) => {
    const k = state.active;
    place(k, e.lngLat);
    if (state.mode === 'two') select(other(k));
  });

  for (const id of ['radius', 'period']) {
    $(id).onchange = () => { refreshCircles(); invalidate(); };
  }

  // ------------------------------------------------------------ address search
  async function searchBuildings(q) {
    const u = new URL(CFG.api.search, location.origin);
    u.searchParams.set('q', q);
    const r = await fetch(u, { signal: AbortSignal.timeout(15000) });
    if (!r.ok) throw new Error();
    return (await r.json()).filter((h) => h.lat != null).map((h) => ({
      lat: h.lat, lng: h.lon, label: h.address,
      detail: `${titleCase(h.borough)} ${h.zipcode ?? ''} · ${h.units} apartments`,
      info: { lat: h.lat, lng: h.lon, name: h.address, bbl: h.bbl, borough: h.borough, units: h.units },
    }));
  }

  // Used only when the building data isn't built: any address, no report.
  async function searchGeocoder(q) {
    const u = new URL('https://geosearch.planninglabs.nyc/v2/search');
    u.searchParams.set('text', q);
    const r = await fetch(u, { signal: AbortSignal.timeout(15000) });
    if (!r.ok) throw new Error();
    const data = await r.json();
    return (data.features || []).map((f) => {
      const [lng, lat] = f.geometry.coordinates;
      return { lat, lng, label: f.properties.label, info: { lat, lng, name: f.properties.name || f.properties.label, bbl: null } };
    });
  }

  $('searchForm').onsubmit = async (e) => {
    e.preventDefault();
    const k = state.active, q = $('search').value.trim(), box = $('searchResults');
    if (!q) return;
    box.textContent = 'Searching NYC addresses…';
    try {
      const hits = CFG.dataReady ? await searchBuildings(q) : await searchGeocoder(q);
      box.replaceChildren();
      if (!hits.length) {
        box.textContent = CFG.dataReady
          ? 'No apartment building found. Try the house number and street, e.g. "1 W 72 St", or click the map.'
          : 'No matches. Try a full street address or click the map.';
        return;
      }
      for (const h of hits.slice(0, 6)) {
        const b = document.createElement('button');
        b.type = 'button';
        b.innerHTML = `<strong>${esc(h.label)}</strong>${h.detail ? `<span>${esc(h.detail)}</span>` : ''}`;
        b.onclick = () => {
          place(k, h, h.info);
          map.flyTo({ center: [h.lng, h.lat], zoom: 16.5, duration: reducedMotion ? 0 : 1000 });
          box.replaceChildren();
          $('search').value = '';
          if (state.mode === 'two') select(other(k));
        };
        box.append(b);
      }
    } catch {
      box.textContent = 'Address search is unavailable right now. You can still click buildings on the map.';
    }
  };

  // ------------------------------------------------------------ live 311 (original logic)
  function category(r) {
    const t = (r.complaint_type || '').toLowerCase();
    if (t.includes('noise')) return 'Noise';
    if (r.agency === 'HPD' || /heat|hot water|plumbing|paint|mold|elevator/.test(t)) return 'Housing';
    if (/sanitation|dirty|rodent|garbage|refuse|recycling|litter|sweeping|dumping|pest/.test(t)) return 'Sanitation';
    if (/street|sidewalk|pothole|traffic|parking|curb|highway/.test(t)) return 'Streets';
    return 'Other';
  }

  function summarize(rows) {
    const counts = Object.fromEntries(CATS.map((c) => [c, 0]));
    let open = 0;
    const hours = [];
    for (const r of rows) {
      counts[category(r)]++;
      if ((r.status || '').toLowerCase() !== 'closed') open++;
      if (r.closed_date && r.created_date) {
        const h = (new Date(r.closed_date) - new Date(r.created_date)) / 36e5;
        if (h >= 0 && Number.isFinite(h)) hours.push(h);
      }
    }
    hours.sort((a, b) => a - b);
    const n = hours.length;
    return {
      total: rows.length, counts, open,
      median: n ? (n % 2 ? hours[(n - 1) / 2] : (hours[n / 2 - 1] + hours[n / 2]) / 2) : null,
      closedSample: n,
    };
  }

  async function getRows(p, start, end, rad, signal) {
    const rows = [];
    let offset = 0;
    const size = 5000;
    while (true) {
      const u = new URL('https://data.cityofnewyork.us/resource/erm2-nwe9.json');
      u.search = new URLSearchParams({
        $select: 'unique_key,complaint_type,agency,created_date,closed_date,status',
        $where: `created_date >= '${start}' AND created_date < '${end}' AND within_circle(location, ${p.lat}, ${p.lng}, ${rad})`,
        $order: 'unique_key ASC', $limit: size, $offset: offset,
      });
      const response = await fetch(u, { signal });
      if (!response.ok) throw new Error(`NYC Open Data returned ${response.status}`);
      const batch = await response.json();
      if (!Array.isArray(batch)) throw new Error('Unexpected data response');
      rows.push(...batch);
      if (batch.length < size) break;
      offset += size;
      if (offset >= 100000) throw new Error('This area has too many reports. Choose a smaller radius or shorter period.');
    }
    return summarize(rows);
  }

  const dateOnly = (d) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;

  // ------------------------------------------------------------ check
  async function loadBuilding(bbl, signal) {
    if (!CFG.dataReady || !bbl) return null;
    const r = await fetch(CFG.api.building + encodeURIComponent(bbl), { signal });
    if (!r.ok) throw new Error(`the building report returned ${r.status}`);
    return r.json();
  }

  async function runCheck() {
    state.controller?.abort();
    const controller = new AbortController();
    state.controller = controller;
    const signal = controller.signal, rev = ++state.revision, ks = activeKeys();
    state.result = null;
    state.loading = true;
    updateCheckButton();
    $('export').disabled = true;
    $('status').textContent = ks.length === 1
      ? 'Loading the building report and the live 311 reports around it…'
      : 'Loading both building reports and the live 311 reports around them…';

    const end = dateOnly(new Date()), s = new Date(end + 'T12:00:00');
    s.setDate(s.getDate() - Number($('period').value));
    const start = dateOnly(s), radius = Number($('radius').value);
    const points = structuredClone(state.points);
    const timer = setTimeout(() => controller.abort(), 90000);
    try {
      // Settle separately: a slow 311 API shouldn't hide the building reports.
      const [bRes, sRes] = await Promise.all([
        Promise.allSettled(ks.map((k) => loadBuilding(points[k].bbl, signal))),
        Promise.allSettled(ks.map((k) => getRows(points[k], start, end, radius, signal))),
      ]);
      if (rev !== state.revision) return;
      if (signal.aborted) throw new Error('timeout');
      const buildings = {}, street = {};
      ks.forEach((k, i) => {
        buildings[k] = bRes[i].status === 'fulfilled' ? bRes[i].value : { error: bRes[i].reason?.message || 'request failed' };
        street[k] = sRes[i].status === 'fulfilled' ? sRes[i].value : { error: sRes[i].reason?.message || 'request failed' };
      });
      state.result = { ks, points, buildings, street, start, end, radius };
      render();
      const streetOk = ks.every((k) => !street[k].error);
      const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      $('status').textContent = (CFG.dataReady ? 'Building predictions for the next 12 months · ' : '')
        + (streetOk ? `Live NYC 311 · ${start} through ${end} (end exclusive) · ${radius} m radius` : 'Live 311 data unavailable')
        + ` · Fetched ${time}.`;
      $('export').disabled = false;
    } catch {
      if (rev !== state.revision) return;
      $('status').textContent = signal.aborted
        ? 'The request timed out. Try a shorter time period or a smaller radius.'
        : 'Could not load the results. Please try again.';
    } finally {
      clearTimeout(timer);
      if (rev === state.revision) {
        state.loading = false;
        updateCheckButton();
      }
    }
  }
  $('check').onclick = runCheck;

  // ------------------------------------------------------------ render
  function render() {
    const R = state.result;
    const have = R.ks.filter((k) => R.buildings[k] && !R.buildings[k].error);
    $('resultsEmpty').hidden = true;
    $('results').hidden = false;
    $('reportTitle').textContent = R.ks.length === 1 ? "What you'd be signing up for." : 'Two buildings, side by side.';
    renderHeads(R);
    renderExpect(R, have);
    renderCharts(R, have);
    renderStreet(R);
  }

  function headCard(k, R, name, body) {
    const tag = R.ks.length === 1 ? 'THE BUILDING' : `BUILDING ${k}`;
    return `<article class="bhead bhead-${lower(k)}">
      <div class="bhead-top"><span class="pin ${lower(k)}">${k}</span>
        <div><small>${tag}</small><h3>${name}</h3></div></div>${body}</article>`;
  }

  function renderHeads(R) {
    $('bheads').innerHTML = R.ks.map((k) => {
      const B = R.buildings[k], p = R.points[k];
      if (!CFG.dataReady) {
        return headCard(k, R, esc(p.name), '<p class="muted small">Building reports aren\'t built on this server yet, so only the street-level picture is shown.</p>');
      }
      if (!B) {
        return headCard(k, R, esc(p.name), '<p class="muted small">There\'s no apartment building (3+ units) in our data within 150 m of this pin. Click directly on a residential building or search its address. The street-level picture below still applies.</p>');
      }
      if (B.error) {
        return headCard(k, R, esc(p.name), `<p class="muted small">This building's report could not be loaded (${esc(B.error)}).</p>`);
      }
      const b = B.report;
      const meta = [`${titleCase(b.borough)} ${b.zipcode ?? ''}`.trim(), `built ${b.yearbuilt}`, `${b.units} apartments`,
        b.floors != null ? `${Math.round(b.floors)} floors` : null, b.kind].filter(Boolean).join(' · ');
      let who = '';
      if (b.landlord) {
        who = `Managed/owned by <strong>${esc(b.landlord)}</strong>`
          + (b.portfolio > 1 ? `, who runs <strong>${b.portfolio}</strong> registered buildings.` : '.');
      } else if (!b.registered) {
        who = 'Not registered with HPD, so no landlord information is available.';
      }
      const top = b.risks[0];
      return headCard(k, R, esc(b.address), `
        <p class="muted small">${esc(meta)}</p>
        ${who ? `<p class="small">${who}</p>` : ''}
        <div class="bhead-stats">
          <div><strong>${pct(top.prob)}</strong><span>top risk: ${esc(shortLabel(top.label).toLowerCase())}</span></div>
          <div><strong>${b.complaints_2y}</strong><span>housing complaints, last 2 years</span></div>
          <div><strong>${b.open_violations}</strong><span>open violations</span></div>
        </div>
        <a class="bhead-link" href="${esc(B.url)}">Open the full report ↗</a>`);
    }).join('');
  }

  function riskTile(r) {
    return `<article class="risk risk-${lower(r.level)}">
      <header><span class="badge">${esc(r.level)}</span><span class="prob">${pct(r.prob)}</span></header>
      <h3>${esc(r.label)}</h3>
      <p class="muted">${r.times_city.toFixed(1)}× the typical building</p>
      ${r.reasons.length ? `<ul>${r.reasons.map((x) => `<li>${esc(x)}</li>`).join('')}</ul>` : ''}
    </article>`;
  }

  function expectCell(r, k) {
    const width = Math.min(100, r.prob * 100).toFixed(1);
    const tick = Math.min(100, (r.typical ?? 0) * 100).toFixed(1);
    return `<div class="expect-cell risk-${lower(r.level)}" data-label="BUILDING ${k}">
      <div class="expect-top"><span class="prob">${pct(r.prob)}</span><span class="badge">${esc(r.level)}</span></div>
      <div class="meter meter-${lower(k)}"><i style="width:${width}%"></i><b style="left:${tick}%" title="Typical building"></b></div>
      <span class="muted small">${r.times_city.toFixed(1)}× the typical building</span>
      ${r.reasons.length ? `<details><summary>Why</summary><ul>${r.reasons.map((x) => `<li>${esc(x)}</li>`).join('')}</ul></details>` : ''}
    </div>`;
  }

  function renderExpect(R, have) {
    $('expectSection').hidden = have.length === 0;
    if (!have.length) return;

    if (have.length === 1) {
      const k = have[0], b = R.buildings[k].report;
      $('expectLead').textContent = R.ks.length === 2
        ? `Only building ${k} has a report, so it is shown on its own.`
        : '';
      $('expect').innerHTML = `<div class="risk-grid">${b.risks.map(riskTile).join('')}</div>`;
      return;
    }

    const [kA, kB] = have, A = R.buildings[kA].report, B = R.buildings[kB].report;
    const a = byType(A), b = byType(B);
    const types = (CFG.types.length ? CFG.types : A.risks.map((r) => r.type))
      .filter((t) => a[t] && b[t])
      .sort((x, y) => Math.max(a[y].prob, b[y].prob) - Math.max(a[x].prob, b[x].prob));

    // Differences under 2 points are treated as a tie.
    let aLower = 0, bLower = 0;
    for (const t of types) {
      const d = a[t].prob - b[t].prob;
      if (d <= -0.02) aLower++;
      else if (d >= 0.02) bLower++;
    }
    const n = types.length;
    $('expectLead').innerHTML = aLower === bLower
      ? `Neither building stands out: each has the lower predicted risk for ${aLower} of ${n} problems.`
      : `${pinDot(aLower > bLower ? kA : kB)}<strong>Building ${aLower > bLower ? kA : kB}</strong> has the lower predicted risk for <strong>${Math.max(aLower, bLower)} of ${n}</strong> problems${Math.min(aLower, bLower) ? `; building ${aLower > bLower ? kB : kA} is lower for ${Math.min(aLower, bLower)}` : ''}.`;

    $('expect').innerHTML = `<div class="expect-table">
      <div class="expect-row expect-head">
        <span>Problem</span>
        <span>${pinDot(kA)}${esc(A.address)}</span>
        <span>${pinDot(kB)}${esc(B.address)}</span>
      </div>
      ${types.map((t) => `<div class="expect-row">
        <div class="expect-label"><strong>${esc(shortLabel(a[t].label))}</strong>
          <span class="muted small">Typical building: ${pct(a[t].typical)}</span></div>
        ${expectCell(a[t], kA)}${expectCell(b[t], kB)}
      </div>`).join('')}
    </div>
    <p class="expect-legend">Bars show each building's predicted chance; the dark tick marks the typical NYC apartment building.</p>`;
  }

  // Shared axis top for side-by-side stacked charts, so equal heights mean equal counts.
  function stackMax(seriesList) {
    let m = 0;
    for (const series of seriesList) {
      const n = series[0]?.length || 0;
      for (let i = 0; i < n; i++) m = Math.max(m, series.reduce((s, arr) => s + (arr[i] || 0), 0));
    }
    if (!m) return undefined;
    const p = 10 ** Math.floor(Math.log10(m));
    return (Math.ceil((m / p) * 2) * p) / 2;
  }

  function chartCard(id, heading, note, height = 300) {
    return `<div class="card"><h2>${heading}</h2><p class="muted small">${note}</p>
      <div class="chart" style="height:${height}px"><canvas id="${id}" role="img" aria-label="${esc(heading.replace(/<[^>]+>/g, ''))}"></canvas></div></div>`;
  }

  function flagsCard(b, heading) {
    const list = b.flags.length
      ? `<ul class="flags">${b.flags.map((f) => `<li>${esc(f)}</li>`).join('')}</ul>`
      : '<p>No open serious violations, recent bedbugs or evictions on record.</p>';
    return `<div class="card"><h2>${heading}</h2>${list}
      <p class="muted small">${b.complaints_2y} housing complaints in the last 2 years · ${b.open_violations} open violations of any class.</p></div>`;
  }

  const HISTORY_NOTE = (C) => `Tenant complaints to HPD per year, by problem type. *${C.years.at(-1).replace('*', '')} is the year so far.`;
  const VIOLATION_NOTE = 'Violations inspectors confirmed per year: A non-hazardous, B hazardous, C immediately hazardous.';

  function drawHistory(id, C, yMax) {
    butaChart(id, 'bar', C.years,
      CFG.types.map((t) => ({ label: t, data: C.complaints[t], color: PALETTE.typeColors[t] })),
      { stacked: true, format: 'int', labelHeader: 'Year', yMax });
  }

  function drawViolations(id, C, yMax) {
    butaChart(id, 'bar', C.years, [
      { label: 'Class A', data: C.violations.A, color: PALETTE.ordinal[0] },
      { label: 'Class B', data: C.violations.B, color: PALETTE.ordinal[1] },
      { label: 'Class C', data: C.violations.C, color: PALETTE.ordinal[2] },
    ], { stacked: true, format: 'int', labelHeader: 'Year', yMax });
  }

  function renderCharts(R, have) {
    const box = $('charts');
    if (!have.length) { box.innerHTML = ''; return; }
    if (have.length === 1) renderOneChart(box, R.buildings[have[0]]);
    else renderTwoCharts(box, have, have.map((k) => R.buildings[k]));
  }

  function renderOneChart(box, B) {
    const b = B.report, C = B.charts;
    const hasLandlord = b.portfolio > 1;
    box.innerHTML = `
      <section>${chartCard('cRisk', 'This building vs a typical building', 'Predicted chance of each problem next year, against the share of all apartment buildings that had it in 2025.')}</section>
      <section class="two-col">
        ${chartCard('cHist', 'Complaint history', HISTORY_NOTE(C))}
        ${chartCard('cViol', 'Violations issued', VIOLATION_NOTE)}
      </section>
      <section class="two-col">
        ${flagsCard(b, 'Red flags right now')}
        ${hasLandlord
          ? chartCard('cLandlord', "The landlord's other buildings", 'Complaints per 100 apartments over the last 2 years, in the landlord\'s other buildings vs all NYC apartment buildings.')
          : '<div class="card"><h2>The landlord\'s other buildings</h2><p class="muted">This landlord has no other registered buildings to compare.</p></div>'}
      </section>
      ${C.crime.labels.length ? `<section>${chartCard('cCrime', 'Reported crime within 200 m', 'Crimes per year reported to the NYPD, 2025–26, located at the nearest street segment. Reporting also reflects policing patterns; shown as context only.', 220)}</section>` : ''}`;

    butaChart('cRisk', 'bar', C.risk.labels, [
      { label: 'This building', data: C.risk.building, color: PALETTE.teal },
      { label: 'Typical building', data: C.risk.typical, color: PALETTE.grey },
    ], { horizontal: true, format: 'pct', yMax: 1, labelHeader: 'Problem' });
    drawHistory('cHist', C);
    drawViolations('cViol', C);
    if (hasLandlord) {
      butaChart('cLandlord', 'bar', C.landlord.labels, [
        { label: "Landlord's other buildings", data: C.landlord.landlord, color: PALETTE.teal },
        { label: 'All NYC buildings', data: C.landlord.city, color: PALETTE.grey },
      ], { horizontal: true, format: 'one', labelHeader: 'Problem' });
    }
    if (C.crime.labels.length) {
      butaChart('cCrime', 'bar', C.crime.labels, [
        { label: 'Within 200 m of this building', data: C.crime.here, color: PALETTE.teal },
        { label: 'Median NYC building', data: C.crime.city, color: PALETTE.grey },
      ], { horizontal: true, format: 'int', labelHeader: 'Crimes per year' });
    }
  }

  function renderTwoCharts(box, keys, [BA, BB]) {
    const [kA, kB] = keys, A = BA.report, B = BB.report, CA = BA.charts, CB = BB.charts;
    const ra = byType(A), rb = byType(B);
    const types = (CFG.types.length ? CFG.types : A.risks.map((r) => r.type)).filter((t) => ra[t] && rb[t]);
    const histMax = stackMax([CFG.types.map((t) => CA.complaints[t]), CFG.types.map((t) => CB.complaints[t])]);
    const classes = ['A', 'B', 'C'];
    const violMax = stackMax([classes.map((c) => CA.violations[c]), classes.map((c) => CB.violations[c])]);
    const crimeLabels = CA.crime.labels.filter((l) => CB.crime.labels.includes(l));
    const crimeOf = (C) => Object.fromEntries(C.crime.labels.map((l, i) => [l, C.crime.here[i]]));
    const crimeA = crimeOf(CA), crimeB = crimeOf(CB);
    const cityCrime = Object.fromEntries(CA.crime.labels.map((l, i) => [l, CA.crime.city[i]]));
    const noLandlord = keys.filter((k, i) => [A, B][i].portfolio <= 1);
    const tag = (k, b) => `${pinDot(k)}${esc(b.address)}`;

    box.innerHTML = `
      <section>${chartCard('cRisk', 'Building A vs building B vs a typical building', 'Predicted chance of each problem next year. Grey is the share of all apartment buildings that had it in 2025.', 340)}</section>
      <h2 class="section-title">Complaint and violation history</h2>
      <p class="muted small">Both buildings share one scale, so taller bars mean more complaints.</p>
      <section class="two-col">
        ${chartCard('cHistA', `${tag(kA, A)}`, HISTORY_NOTE(CA))}
        ${chartCard('cHistB', `${tag(kB, B)}`, HISTORY_NOTE(CB))}
      </section>
      <section class="two-col">
        ${chartCard('cViolA', `${pinDot(kA)}Violations issued`, VIOLATION_NOTE)}
        ${chartCard('cViolB', `${pinDot(kB)}Violations issued`, VIOLATION_NOTE)}
      </section>
      <section class="two-col">
        ${flagsCard(A, `${pinDot(kA)}Red flags right now`)}
        ${flagsCard(B, `${pinDot(kB)}Red flags right now`)}
      </section>
      <section>${chartCard('cLandlord', "The landlords' other buildings",
        'Complaints per 100 apartments over the last 2 years, in each landlord\'s other buildings vs all NYC apartment buildings.'
        + (noLandlord.length ? ` Building ${noLandlord.join(' and ')}'s landlord has no other registered buildings, so it has no bars.` : ''))}</section>
      ${crimeLabels.length ? `<section>${chartCard('cCrime', 'Reported crime within 200 m', 'Crimes per year reported to the NYPD, 2025–26. Reporting also reflects policing patterns; shown as context only.', 240)}</section>` : ''}`;

    butaChart('cRisk', 'bar', types.map((t) => shortLabel(ra[t].label)), [
      { label: `Building ${kA}`, data: types.map((t) => ra[t].prob), color: PALETTE.teal },
      { label: `Building ${kB}`, data: types.map((t) => rb[t].prob), color: PALETTE.orange },
      { label: 'Typical building', data: types.map((t) => ra[t].typical), color: PALETTE.grey },
    ], { horizontal: true, format: 'pct', yMax: 1, labelHeader: 'Problem' });
    drawHistory('cHistA', CA, histMax);
    drawHistory('cHistB', CB, histMax);
    drawViolations('cViolA', CA, violMax);
    drawViolations('cViolB', CB, violMax);
    butaChart('cLandlord', 'bar', CA.landlord.labels, [
      { label: `Building ${kA}'s landlord`, data: CA.landlord.landlord, color: PALETTE.teal },
      { label: `Building ${kB}'s landlord`, data: CB.landlord.landlord, color: PALETTE.orange },
      { label: 'All NYC buildings', data: CA.landlord.city, color: PALETTE.grey },
    ], { horizontal: true, format: 'one', labelHeader: 'Problem' });
    if (crimeLabels.length) {
      butaChart('cCrime', 'bar', crimeLabels, [
        { label: `Within 200 m of building ${kA}`, data: crimeLabels.map((l) => crimeA[l]), color: PALETTE.teal },
        { label: `Within 200 m of building ${kB}`, data: crimeLabels.map((l) => crimeB[l]), color: PALETTE.orange },
        { label: 'Median NYC building', data: crimeLabels.map((l) => cityCrime[l]), color: PALETTE.grey },
      ], { horizontal: true, format: 'int', labelHeader: 'Crimes per year' });
    }
  }

  function renderStreet(R) {
    const ks = R.ks, S = R.street, two = ks.length === 2;
    const ok = (k) => S[k] && !S[k].error;
    const box = $('street');
    $('streetLead').textContent = two
      ? `Live NYC 311 service requests within ${R.radius} m of each pin, ${R.start} through ${R.end}.`
      : `Live NYC 311 service requests within ${R.radius} m of the building, ${R.start} through ${R.end}.`;
    if (!ks.some(ok)) {
      box.innerHTML = `<p class="empty">Live 311 data is unavailable right now (${esc(S[ks[0]]?.error || 'no response')}). The building results above are unaffected.</p>`;
      return;
    }
    const fmtH = (n) => (n == null ? 'N/A' : n < 24 ? `${n.toFixed(1)}h` : `${(n / 24).toFixed(1)}d`);
    const openPct = (x) => (x.total ? `${Math.round((x.open / x.total) * 100)}%` : 'N/A');
    const val = (k, f) => (ok(k) ? f(S[k]) : 'N/A');
    const metric = (label, f, note) => `<article><span>${label}</span>
      <strong>${two ? `${val('A', f)} <i>vs</i> <b>${val('B', f)}</b>` : val('A', f)}</strong><p>${note}</p></article>`;
    const closure = `Valid closure dates: ${ks.map((k) => `${two ? k + ' ' : ''}${ok(k) ? S[k].closedSample.toLocaleString() : '–'}`).join(' · ')}`;
    const max = Math.max(1, ...ks.filter(ok).flatMap((k) => Object.values(S[k].counts)));
    const bars = CATS.map((c) => `<div class="barrow"><span>${c}</span>
      <div class="tracks">${ks.map((k) => `<div class="track"><i style="width:${ok(k) ? (100 * S[k].counts[c]) / max : 0}%"></i></div>`).join('')}</div>
      <span class="counts">${ks.map((k) => (ok(k) ? S[k].counts[c] : '–')).join(' / ')}</span></div>`).join('');

    box.innerHTML = `
      <div class="metrics">
        ${metric('REPORTED PROBLEMS', (x) => x.total.toLocaleString(), `311 requests within ${R.radius} m`)}
        ${metric('MEDIAN CLOSURE TIME', (x) => fmtH(x.median), closure)}
        ${metric('STILL OPEN', openPct, 'Share not marked closed')}
      </div>
      <div class="detailgrid${two ? '' : ' single'}">
        <article class="breakdown">
          <div class="sectiontop"><h3>What gets reported?</h3>
            ${two ? '<div class="legend"><span>A</span> Building A <span>B</span> Building B</div>' : ''}</div>
          <p class="muted">Request counts by category. Fewer reports may mean fewer issues.</p>
          <div class="bars">${bars}</div>
        </article>
        ${two ? `<article class="insight">
          <span class="eyebrow">YOUR PRIORITIES MATTER</span>
          <h3 id="verdict">Which block fits you?</h3>
          <p id="reason"></p>
          <label for="priority">What matters most?</label>
          <select id="priority">
            <option value="all">A balanced view</option>
            <option value="Noise">A quieter home</option>
            <option value="Housing">Housing conditions</option>
            <option value="Sanitation">Cleaner surroundings</option>
            <option value="Streets">Street conditions</option>
          </select>
          <div class="insightfoot">A 311-based signal about the street, not a complete livability rating.
            The building predictions above look inside each address.</div>
        </article>` : ''}
      </div>`;

    if (two) {
      $('priority').value = state.priority;
      $('priority').onchange = () => { state.priority = $('priority').value; recommend(); };
      recommend();
    }
  }

  function recommend() {
    const R = state.result;
    if (!R || R.ks.length !== 2) return;
    const SA = R.street.A, SB = R.street.B;
    if (!SA || SA.error || !SB || SB.error) {
      $('verdict').textContent = 'Comparison unavailable';
      $('reason').textContent = 'We need both locations’ live data before suggesting a result.';
      return;
    }
    const p = state.priority;
    const a = p === 'all' ? SA.total : SA.counts[p], b = p === 'all' ? SB.total : SB.counts[p];
    const gap = Math.max(a, b) ? Math.abs(a - b) / Math.max(a, b) : 0;
    if (a + b < 20) {
      $('verdict').textContent = 'Too little data to lean either way.';
      $('reason').textContent = `Only ${a + b} combined ${p === 'all' ? '' : p.toLowerCase() + ' '}reports. Try a longer time window. An absence of reports is not proof of an absence of issues.`;
    } else if (gap < 0.1) {
      $('verdict').textContent = 'A similar reporting picture.';
      $('reason').textContent = 'The difference is below 10% for your chosen priority. These reports do not provide a clear preference.';
    } else {
      const winner = a < b ? 'A' : 'B';
      $('verdict').textContent = `The block around ${winner} has fewer reports.`;
      $('reason').textContent = `${Math.round(gap * 100)}% fewer ${p === 'all' ? 'total' : p.toLowerCase()} requests (${Math.min(a, b)} vs ${Math.max(a, b)}) in the same radius and period. This may fit your priorities, but reporting and density differences can influence the result.`;
    }
  }

  // ------------------------------------------------------------ export
  $('export').onclick = () => {
    const R = state.result;
    if (!R) return;
    const B = (k) => (R.buildings[k] && !R.buildings[k].error ? R.buildings[k].report : null);
    const S = (k) => (R.street[k] && !R.street[k].error ? R.street[k] : null);
    const col = (f) => R.ks.map((k) => f(k) ?? '');
    const lines = [
      [R.ks.length === 1 ? 'BUTA building report' : 'BUTA building comparison'],
      ['311 start inclusive', R.start], ['311 end exclusive', R.end], ['311 radius meters', R.radius],
      ['Measure', ...R.ks.map((k) => `Building ${k}`)],
      ['Address', ...col((k) => B(k)?.address ?? R.points[k].name)],
      ['BBL', ...col((k) => R.points[k].bbl)],
      ['Latitude', ...col((k) => R.points[k].lat)],
      ['Longitude', ...col((k) => R.points[k].lng)],
      ...CFG.types.map((t) => [`Predicted chance next year: ${t}`, ...col((k) => B(k)?.risks.find((r) => r.type === t)?.prob)]),
      ['Housing complaints, last 2 years', ...col((k) => B(k)?.complaints_2y)],
      ['Open violations', ...col((k) => B(k)?.open_violations)],
      ['Red flags', ...col((k) => B(k)?.flags.join('; '))],
      ['311 total requests', ...col((k) => S(k)?.total)],
      ['311 median closure hours', ...col((k) => S(k)?.median)],
      ['311 valid closure sample', ...col((k) => S(k)?.closedSample)],
      ['311 requests not closed', ...col((k) => S(k)?.open)],
      ...CATS.map((c) => [`311 ${c}`, ...col((k) => S(k)?.counts[c])]),
      ['Sources', 'NYC Open Data: 311 (live), HPD, PLUTO, NYPD, DOI evictions; predictions by Team BUTA'],
      ['Limitations', 'Predictions are estimates of reported problems. 311 counts are not population adjusted; reports and closures do not prove actual conditions or resolution.'],
    ];
    const csv = lines.map((row) => row.map((v) => `"${String(v).replaceAll('"', '""')}"`).join(',')).join('\r\n');
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = R.ks.length === 1 ? 'buta-building-report.csv' : 'buta-building-comparison.csv';
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  // ------------------------------------------------------------ start
  KEYS.forEach(updateLabel);
  select('A');
  if (new URLSearchParams(location.search).get('mode') === 'one') setMode('one');   // shareable: /?mode=one
  updateCheckButton();
  (async () => {
    if (CFG.dataReady) await Promise.all(KEYS.map(snap));   // land both pins on real buildings
    runCheck();
  })();
})();
