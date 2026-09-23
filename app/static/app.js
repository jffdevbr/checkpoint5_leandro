// Helpers compartilhados pelas páginas.

export const fmt = (v, d = 2) =>
  v === null || v === undefined || Number.isNaN(v)
    ? "—"
    : Number(v).toLocaleString("pt-BR", { minimumFractionDigits: d, maximumFractionDigits: d });

export const brl = (v) =>
  v === null || v === undefined || Number.isNaN(v)
    ? "—"
    : Number(v).toLocaleString("pt-BR", { style: "currency", currency: "BRL", maximumFractionDigits: 0 });

export async function api(path, body) {
  const opts = body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : {};
  const r = await fetch(path, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `Erro ${r.status}`);
  return data;
}

/** Mostra um erro de API de forma legível (inclusive o 503 de "modelo não treinado"). */
export function mostrarErro(el, err) {
  el.innerHTML = `<div class="note bad"><b>Não foi possível calcular.</b><br>${err.message}</div>`;
}

/** Liga um <input type=range> ao seu <b> de leitura. */
export function slider(id, casas = 1) {
  const input = document.getElementById(id);
  const out = document.getElementById(id + "-val");
  const sync = () => (out.textContent = fmt(input.value, casas));
  input.addEventListener("input", sync);
  sync();
  return input;
}

export function lerContexto() {
  return {
    Catalyst_Age_Days: +document.getElementById("age").value,
    Sensor_Health_Index: +document.getElementById("health").value,
    Vibration_Level_mm_s: +document.getElementById("vib").value,
    Ambient_Temp_C: +document.getElementById("amb").value,
  };
}

export function badgeNivel(nivel) {
  const cls = nivel === "L2" ? "ok" : nivel === "L1" ? "warn" : "bad";
  return `<span class="badge ${cls}">${nivel}</span>`;
}
