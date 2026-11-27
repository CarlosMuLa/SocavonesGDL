/* SocavonesGDL — Visor geográfico estilo SIGmetro/IMEPLAN + Galería */

const AMG_CENTER = [20.668, -103.355];
const AMG_ZOOM = 12;

// Mapas base 100% libres sin API key
const baseOSM = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  maxZoom: 19,
});

const baseSat = L.tileLayer(
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  {
    attribution: "Tiles &copy; Esri &mdash; Maxar, Earthstar Geographics",
    maxZoom: 19,
  }
);

const map = L.map("riskMap", {
  center: AMG_CENTER,
  zoom: AMG_ZOOM,
  layers: [baseOSM],
  preferCanvas: true,
});

// Panes ordenados para que overlays queden debajo de celdas/puntos clicables
map.createPane("rasterOverlays");
map.getPane("rasterOverlays").style.zIndex = 350;

const layerMuyAlto = L.layerGroup().addTo(map);
const layerAlto = L.layerGroup().addTo(map);
const layerSinkholes = L.layerGroup().addTo(map);

const OVERLAY_SPECS = {
  delta_ndvi: {
    title: "Pérdida de vegetación (ΔNDVI 2020–26)",
    file: "./layers/delta_ndvi.png",
    legend: [
      ["#fdba74", "Moderada (-0.08)"],
      ["#ea580c", "Fuerte (-0.15)"],
      ["#991b1b", "Severa (-0.25)"],
    ],
  },
  delta_ndbi: {
    title: "Nuevo pavimento / obra (ΔNDBI 2020–26)",
    file: "./layers/delta_ndbi.png",
    legend: [
      ["#fbbf24", "Leve (+0.03)"],
      ["#dc2626", "Obra / pavimento (+0.08)"],
      ["#7e22ce", "Impermeabilización alta"],
    ],
  },
  mndwi_wet: {
    title: "Humedad lluvias 2026 (MNDWI)",
    file: "./layers/mndwi_wet.png",
    legend: [
      ["#38bdf8", "Humedad superficial"],
      ["#0284c7", "Acumulación alta"],
      ["#1e3a8a", "Cuerpo de agua"],
    ],
  },
  tpi_depressions: {
    title: "Depresiones y arroyos (TPI 500 m)",
    file: "./layers/tpi_depressions.png",
    legend: [
      ["#2dd4bf", "Bajo leve (-2 m)"],
      ["#0d9488", "Cauce / arroyo (-5 m)"],
      ["#115e59", "Depresión profunda (-10 m)"],
    ],
  },
  imeplan_presion_drenaje: {
    title: "Presión sobre el drenaje (IMEPLAN)",
    file: "./layers/imeplan_presion_drenaje.png",
    legend: [
      ["#fef08a", "Baja"],
      ["#f97316", "Media"],
      ["#991b1b", "Alta"],
    ],
  },
  imeplan_inundaciones: {
    title: "Riesgo de inundaciones (IMEPLAN)",
    file: "./layers/imeplan_inundaciones.png",
    legend: [
      ["#7dd3fc", "Medio"],
      ["#2563eb", "Alto"],
      ["#312e81", "Muy alto"],
    ],
  },
  imeplan_cus: {
    title: "Densificación / Incremento CUS (IMEPLAN)",
    file: "./layers/imeplan_cus.png",
    legend: [
      ["#e879f9", "Moderado"],
      ["#a21caf", "Alto"],
      ["#581c87", "Muy alto"],
    ],
  },
};

const activeOverlays = {};
let overlayBounds = null;
let allSinkholeFeatures = [];

const statusEl = document.getElementById("mapStatus");
const countEl = document.getElementById("countSocavones");
const legendEl = document.getElementById("dynamicLegend");

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

function updateLegend() {
  const blocks = [];
  const muyAltoOn = document.getElementById("toggleMuyAlto").checked;
  const altoOn = document.getElementById("toggleAlto").checked;
  if (muyAltoOn || altoOn) {
    const items = [];
    if (muyAltoOn) items.push(`<span class="legend-chip"><span class="swatch swatch-muy-alto"></span>Muy Alto (95–100)</span>`);
    if (altoOn) items.push(`<span class="legend-chip"><span class="swatch swatch-alto"></span>Alto (85–95)</span>`);
    blocks.push(`<div><div class="legend-block-title">Riesgo predictivo</div><div class="legend-swatches">${items.join("")}</div></div>`);
  }

  for (const [key, layer] of Object.entries(activeOverlays)) {
    if (!map.hasLayer(layer)) continue;
    const spec = OVERLAY_SPECS[key];
    const chips = spec.legend
      .map(([color, lbl]) => `<span class="legend-chip"><span class="swatch" style="background:${color}"></span>${lbl}</span>`)
      .join("");
    blocks.push(`<div><div class="legend-block-title">${spec.title}</div><div class="legend-swatches">${chips}</div></div>`);
  }

  legendEl.innerHTML = blocks.length
    ? blocks.join("")
    : `<span>Activa cualquier capa arriba para ver su simbología.</span>`;
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
      fillOpacity: 0.94,
    });
    marker.bindPopup(sinkholePopupHtml(p));
    marker.addTo(layerSinkholes);
    shown += 1;
  }
  countEl.textContent = String(shown);
}

async function initMapData() {
  try {
    const [riskResp, sinkResp, metaResp] = await Promise.all([
      fetch("./data/risk_cells.geojson"),
      fetch("./data/sinkholes.geojson"),
      fetch("./layers/layers_meta.json"),
    ]);
    const [riskData, sinkData, layersMeta] = await Promise.all([
      riskResp.json(),
      sinkResp.json(),
      metaResp.json(),
    ]);

    overlayBounds = layersMeta.bounds;

    for (const feat of riskData.features) {
      const p = feat.properties || {};
      const isVeryHigh = p.risk_tier === "Muy Alto";
      const layer = L.geoJSON(feat, {
        style: {
          color: isVeryHigh ? "#7f1d1d" : "#c2410c",
          weight: 0.4,
          fillColor: isVeryHigh ? "#991b1b" : "#ea580c",
          fillOpacity: isVeryHigh ? 0.58 : 0.38,
        },
      });
      layer.bindPopup(cellPopupHtml(p));
      layer.addTo(isVeryHigh ? layerMuyAlto : layerAlto);
    }

    allSinkholeFeatures = sinkData.features || [];
    renderSinkholes();
    updateLegend();
    statusEl.textContent = `${riskData.features.length.toLocaleString("es-MX")} zonas + 7 capas listas`;
    setTimeout(() => {
      statusEl.style.opacity = "0";
      statusEl.style.transition = "opacity 0.5s";
    }, 2500);
  } catch (err) {
    statusEl.textContent = "Error al cargar las capas del mapa.";
    console.error(err);
  }
}

// Cambio de mapa base (OSM / Satélite Esri)
document.querySelectorAll(".bm-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".bm-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    if (btn.dataset.bm === "sat") {
      map.removeLayer(baseOSM);
      baseSat.addTo(map);
    } else {
      map.removeLayer(baseSat);
      baseOSM.addTo(map);
    }
  });
});

// Toggles de riesgo y socavones
document.getElementById("toggleMuyAlto").addEventListener("change", (e) => {
  e.target.checked ? map.addLayer(layerMuyAlto) : map.removeLayer(layerMuyAlto);
  updateLegend();
});
document.getElementById("toggleAlto").addEventListener("change", (e) => {
  e.target.checked ? map.addLayer(layerAlto) : map.removeLayer(layerAlto);
  updateLegend();
});
document.getElementById("toggleSocavones").addEventListener("change", (e) => {
  e.target.checked ? map.addLayer(layerSinkholes) : map.removeLayer(layerSinkholes);
});
document.getElementById("filterYear").addEventListener("change", renderSinkholes);
document.getElementById("filterCause").addEventListener("change", renderSinkholes);
document.getElementById("btnResetMap").addEventListener("click", () => {
  map.setView(AMG_CENTER, AMG_ZOOM);
});

// Toggles y sliders de opacidad de capas satelitales + IMEPLAN
document.querySelectorAll(".overlay-toggle").forEach((chk) => {
  chk.addEventListener("change", (e) => {
    const key = e.target.dataset.layer;
    const itemEl = e.target.closest(".overlay-item");
    const opWrap = itemEl.querySelector(".opacity-wrap");
    const slider = itemEl.querySelector(".opacity-slider");
    const spec = OVERLAY_SPECS[key];

    if (!overlayBounds || !spec) return;

    if (e.target.checked) {
      if (!activeOverlays[key]) {
        activeOverlays[key] = L.imageOverlay(spec.file, overlayBounds, {
          opacity: parseFloat(slider.value),
          pane: "rasterOverlays",
          interactive: false,
        });
      }
      activeOverlays[key].addTo(map);
      opWrap.hidden = false;
    } else {
      if (activeOverlays[key]) map.removeLayer(activeOverlays[key]);
      opWrap.hidden = true;
    }
    updateLegend();
  });
});

document.querySelectorAll(".opacity-slider").forEach((sl) => {
  sl.addEventListener("input", (e) => {
    const key = e.target.dataset.layer;
    if (activeOverlays[key]) {
      activeOverlays[key].setOpacity(parseFloat(e.target.value));
    }
  });
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
