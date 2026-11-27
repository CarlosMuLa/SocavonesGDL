/* SocavonesGDL — Mapa interactivo Leaflet + Galería de hallazgos */

const AMG_CENTER = [20.668, -103.355];
const AMG_ZOOM = 12;

const map = L.map("riskMap", {
  center: AMG_CENTER,
  zoom: AMG_ZOOM,
  preferCanvas: true,
});

L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; CARTO',
  maxZoom: 19,
}).addTo(map);

const layerMuyAlto = L.layerGroup().addTo(map);
const layerAlto = L.layerGroup().addTo(map);
const layerSinkholes = L.layerGroup().addTo(map);

let allSinkholeFeatures = [];

const statusEl = document.getElementById("mapStatus");
const countEl = document.getElementById("countSocavones");

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function presionLabel(val) {
  if (val >= 3) return "Alta";
  if (val >= 2) return "Media";
  if (val >= 1) return "Baja";
  return "Sin dato";
}

function inundacionLabel(val) {
  const labels = { 1: "Muy bajo", 2: "Bajo", 3: "Medio", 4: "Alto", 5: "Muy alto" };
  return labels[Math.round(val)] || "Sin dato";
}

function cellPopupHtml(p) {
  const isVeryHigh = p.risk_tier === "Muy Alto";
  const drivers = (p.top_drivers || "Sin desglose")
    .split("|")
    .map((d) => `<li>${escapeHtml(d.trim())}</li>`)
    .join("");

  return `
    <div>
      <span class="popup-badge ${isVeryHigh ? "popup-muy-alto" : "popup-alto"}">
        Riesgo ${escapeHtml(p.risk_tier)} · ${Number(p.risk_score).toFixed(1)}/100
      </span>
      <div class="popup-title">Celda urbana 250 m #${escapeHtml(p.cell_id)}</div>
      <div class="popup-meta">
        <strong>Principales factores aquí:</strong>
        <ul style="margin:4px 0 8px; padding-left:16px;">${drivers}</ul>
        <strong>Presión drenaje:</strong> ${presionLabel(p.presion_drenaje_max)}<br/>
        <strong>Riesgo inundación:</strong> ${inundacionLabel(p.riesgo_inundacion_max)}<br/>
        <strong>Intensidad CUS máx:</strong> ${Number(p.cus_intensidad_max || 0).toFixed(1)}<br/>
        <strong>Socavones registrados aquí:</strong> ${p.sinkhole_count || 0}
      </div>
    </div>
  `;
}

function sinkholePopupHtml(p) {
  const calle = [p.calle, p.cruces].filter(Boolean).join(" / ") || "Ubicación registrada";
  const lugar = [p.colonia, p.municipio].filter(Boolean).join(", ");
  return `
    <div>
      <div class="popup-title">Socavón registrado (${escapeHtml(p.fecha || p.anio)})</div>
      <div class="popup-meta">
        <strong>Ubicación:</strong> ${escapeHtml(calle)}<br/>
        <strong>Zona:</strong> ${escapeHtml(lugar || "AMG")}<br/>
        <strong>Causa aparente:</strong> ${escapeHtml(p.causa_raw || "Sin dato")}<br/>
        <strong>Tamaño:</strong> ${escapeHtml(p.tamano || "Sin dato")}
      </div>
    </div>
  `;
}

function renderSinkholes() {
  layerSinkholes.clearLayers();
  const yearFilter = document.getElementById("filterYear").value;
  const causeFilter = document.getElementById("filterCause").value;

  let shown = 0;
  for (const feat of allSinkholeFeatures) {
    const p = feat.properties || {};
    const anio = Number(p.anio);
    if (yearFilter === "2020_2025" && anio > 2025) continue;
    if (yearFilter === "2026" && anio !== 2026) continue;
    if (causeFilter !== "all" && p.causa_normalizada !== causeFilter) continue;

    const [lon, lat] = feat.geometry.coordinates;
    const is2026 = anio === 2026;
    const marker = L.circleMarker([lat, lon], {
      radius: 5.5,
      fillColor: is2026 ? "#0284c7" : "#1e3a8a",
      color: "#ffffff",
      weight: 1.5,
      fillOpacity: 0.92,
    });
    marker.bindPopup(sinkholePopupHtml(p));
    marker.addTo(layerSinkholes);
    shown += 1;
  }
  countEl.textContent = String(shown);
}

async function initMapData() {
  try {
    const [riskResp, sinkResp] = await Promise.all([
      fetch("./data/risk_cells.geojson"),
      fetch("./data/sinkholes.geojson"),
    ]);
    const [riskData, sinkData] = await Promise.all([riskResp.json(), sinkResp.json()]);

    for (const feat of riskData.features) {
      const p = feat.properties || {};
      const isVeryHigh = p.risk_tier === "Muy Alto";
      const layer = L.geoJSON(feat, {
        style: {
          color: isVeryHigh ? "#7f1d1d" : "#c2410c",
          weight: 0.4,
          fillColor: isVeryHigh ? "#991b1b" : "#ea580c",
          fillOpacity: isVeryHigh ? 0.62 : 0.42,
        },
      });
      layer.bindPopup(cellPopupHtml(p));
      layer.addTo(isVeryHigh ? layerMuyAlto : layerAlto);
    }

    allSinkholeFeatures = sinkData.features || [];
    renderSinkholes();
    statusEl.textContent = `${riskData.features.length.toLocaleString("es-MX")} zonas de riesgo cargadas`;
    setTimeout(() => {
      statusEl.style.opacity = "0";
      statusEl.style.transition = "opacity 0.5s";
    }, 2500);
  } catch (err) {
    statusEl.textContent = "Error al cargar las capas del mapa.";
    console.error(err);
  }
}

document.getElementById("toggleMuyAlto").addEventListener("change", (e) => {
  e.target.checked ? map.addLayer(layerMuyAlto) : map.removeLayer(layerMuyAlto);
});
document.getElementById("toggleAlto").addEventListener("change", (e) => {
  e.target.checked ? map.addLayer(layerAlto) : map.removeLayer(layerAlto);
});
document.getElementById("toggleSocavones").addEventListener("change", (e) => {
  e.target.checked ? map.addLayer(layerSinkholes) : map.removeLayer(layerSinkholes);
});
document.getElementById("filterYear").addEventListener("change", renderSinkholes);
document.getElementById("filterCause").addEventListener("change", renderSinkholes);
document.getElementById("btnResetMap").addEventListener("click", () => {
  map.setView(AMG_CENTER, AMG_ZOOM);
});

// Pestañas de gráficas + Lightbox
const chartImg = document.getElementById("activeChartImg");
const chartCaption = document.getElementById("activeChartCaption");
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    chartImg.src = btn.dataset.img;
    chartCaption.textContent = btn.dataset.caption;
  });
});

const lightbox = document.getElementById("lightbox");
const lightboxImg = document.getElementById("lightboxImg");
const lightboxCaption = document.getElementById("lightboxCaption");

chartImg.addEventListener("click", () => {
  lightboxImg.src = chartImg.src;
  lightboxCaption.textContent = chartCaption.textContent;
  lightbox.hidden = false;
});
document.getElementById("lightboxClose").addEventListener("click", () => {
  lightbox.hidden = true;
});
lightbox.addEventListener("click", (e) => {
  if (e.target === lightbox) lightbox.hidden = true;
});

initMapData();
