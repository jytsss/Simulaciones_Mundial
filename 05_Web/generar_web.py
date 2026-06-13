# -*- coding: utf-8 -*-
"""Genera la web estática de predicciones (docs/index.html, lista para GitHub Pages).

Para cada partido de la fase de grupos calcula con el modelo del repo:
  - Probabilidades 1X2 calibradas (T=1, sin afilar: las honestas)
  - Marcador más probable y goles esperados (xG del modelo)
  - Córners esperados y P(más de 7.5 / 8.5 / 9.5)  [Poisson sobre medias por equipo]
  - Tarjetas amarillas esperadas y P(más de 3.5 / 4.5)

La sección de partidos tiene un selector de jornadas que arranca en la fecha
local del visitante, así la página se actualiza sola cada día. Un workflow de
GitHub Actions la regenera a diario por si cambian los datos.

Uso:  python 05_Web/generar_web.py   (requiere los .pkl entrenados; si no
      existen, ejecuta antes 04_Prediccion/prediccion_mundial.py)
"""

import os
import sys
import json
import math
from datetime import datetime, timezone

import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RAIZ)
sys.path.insert(0, os.path.join(RAIZ, '04_Prediccion'))

import prediccion_mundial as pm  # noqa: E402  (reutilizamos el pipeline del repo)

# Códigos ISO para las banderas (flagcdn.com)
BANDERAS_ISO = {
    'México': 'mx', 'Sudáfrica': 'za', 'Corea del Sur': 'kr', 'República Checa': 'cz',
    'Suiza': 'ch', 'Bosnia-Herzegovina': 'ba', 'Canadá': 'ca', 'Catar': 'qa',
    'Escocia': 'gb-sct', 'Brasil': 'br', 'Haití': 'ht', 'Marruecos': 'ma',
    'Turquía': 'tr', 'Paraguay': 'py', 'EE. UU.': 'us', 'Australia': 'au',
    'Alemania': 'de', 'Ecuador': 'ec', 'Costa de Marfil': 'ci', 'Curazao': 'cw',
    'Suecia': 'se', 'Países Bajos': 'nl', 'Túnez': 'tn', 'Japón': 'jp',
    'Bélgica': 'be', 'Egipto': 'eg', 'Irán': 'ir', 'Nueva Zelanda': 'nz',
    'España': 'es', 'Uruguay': 'uy', 'Cabo Verde': 'cv', 'Arabia Saudí': 'sa',
    'Francia': 'fr', 'Noruega': 'no', 'Senegal': 'sn', 'Irak': 'iq',
    'Austria': 'at', 'Argentina': 'ar', 'Argelia': 'dz', 'Jordania': 'jo',
    'Portugal': 'pt', 'Colombia': 'co', 'RD Congo': 'cd', 'Uzbekistán': 'uz',
    'Croacia': 'hr', 'Inglaterra': 'gb-eng', 'Ghana': 'gh', 'Panamá': 'pa',
}


def p_poisson_mas_de(lam, linea):
    """P(N >= ceil(linea)) con N ~ Poisson(lam)."""
    k_min = int(linea) + 1
    return 1 - sum(math.exp(-lam) * lam ** k / math.factorial(k) for k in range(k_min))


def construir_datos():
    df_mundial_grupos, df_vars, grupos, fechas_reales = pm.cargar_mundial()

    # Probabilidades calibradas sin temperatura: las adecuadas para mostrar
    pred = pm.pipeline_prediccion(df_mundial_grupos, sede_neutral=True, T=1.0)
    pred['Grupo'] = df_mundial_grupos['Grupo'].values

    stats = df_vars.set_index('Equipo')
    mapa_fechas = {(r['Equipo_Local'], r['Equipo_Visitante']): r['Fecha']
                   for _, r in fechas_reales.iterrows()}

    partidos = []
    for _, r in pred.iterrows():
        a, b = r['Equipo_Local'], r['Equipo_Visitante']
        p1, px, p2 = r['Prob_Local'], r['Prob_Empate'], r['Prob_Visitante']
        s = p1 + px + p2
        p1, px, p2 = p1 / s, px / s, p2 / s
        res = ['1', 'X', '2'][int(max(range(3), key=[p1, px, p2].__getitem__))]
        gl, gv = pm.marcador_mas_probable(r['xG_Modelo_Local'], r['xG_Modelo_Visitante'], res)

        # Córners y tarjetas: promedio de la estimación reciente (últimos 5) e histórica
        lam_cor = (stats.loc[a, 'avg_Córneres_5'] + stats.loc[b, 'avg_Córneres_5'] +
                   stats.loc[a, 'avg_Córneres_total'] + stats.loc[b, 'avg_Córneres_total']) / 2
        lam_tar = (stats.loc[a, 'avg_Tarjetas_amarillas_5'] + stats.loc[b, 'avg_Tarjetas_amarillas_5'] +
                   stats.loc[a, 'avg_Tarjetas_amarillas_total'] + stats.loc[b, 'avg_Tarjetas_amarillas_total']) / 2

        partidos.append({
            'fecha': mapa_fechas.get((a, b), ''),
            'grupo': r['Grupo'], 'local': a, 'visitante': b,
            'marcador': f'{gl}-{gv}',
            'p1': round(p1 * 100, 1), 'px': round(px * 100, 1), 'p2': round(p2 * 100, 1),
            'xgl': round(float(r['xG_Modelo_Local']), 2), 'xgv': round(float(r['xG_Modelo_Visitante']), 2),
            'cor': round(lam_cor, 1),
            'c75': round(p_poisson_mas_de(lam_cor, 7.5) * 100), 'c85': round(p_poisson_mas_de(lam_cor, 8.5) * 100),
            'c95': round(p_poisson_mas_de(lam_cor, 9.5) * 100),
            'tar': round(lam_tar, 1),
            't35': round(p_poisson_mas_de(lam_tar, 3.5) * 100), 't45': round(p_poisson_mas_de(lam_tar, 4.5) * 100),
        })
    partidos.sort(key=lambda p: (p['fecha'], p['grupo']))

    mc = pd.read_csv('Predicciones/probabilidades_montecarlo.csv', index_col=0)
    campeon = [{'equipo': eq, 'r32': float(r['R32']), 'octavos': float(r['Octavos']),
                'cuartos': float(r['Cuartos']), 'semis': float(r['Semis']),
                'final': float(r['Final']), 'campeon': float(r['Campeon'])}
               for eq, r in mc.iterrows()]

    elim = pd.read_csv('Predicciones/predicciones_eliminatorias.csv')
    cuadro = [{'fase': r['Fase'], 'fechas': r['Fechas'], 'local': r['Local'], 'visitante': r['Visitante'],
               'marcador': r['Marcador_Predicho'], 'avanza': r['Avanza'],
               'p1': float(r['Prob_1']), 'px': float(r['Prob_X']), 'p2': float(r['Prob_2'])}
              for _, r in elim.iterrows()]

    clasif = pd.read_csv('Predicciones/clasificacion_grupos.csv')
    tablas = {}
    for _, r in clasif.iterrows():
        tablas.setdefault(r['Grupo'], []).append({
            'pos': int(r['Posicion']), 'equipo': r['Equipo'],
            'pts': int(r['Pts']), 'dg': round(float(r['DG']), 2)})
    for g in tablas:
        tablas[g].sort(key=lambda x: x['pos'])

    return partidos, campeon, cuadro, tablas


PLANTILLA = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mundial 2026 · Predicciones diarias del modelo</title>
<meta name="description" content="Probabilidades de ganar, córners, tarjetas y marcador predicho para cada partido del Mundial 2026, generadas con XGBoost y simulación de Monte Carlo.">
<meta property="og:title" content="Mundial 2026 · Predicciones diarias del modelo">
<meta property="og:description" content="Probabilidades de victoria, córners y tarjetas de cada partido + cuadro completo hasta la final. XGBoost + 10.000 mundiales simulados.">
<meta property="og:type" content="website">
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>⚽</text></svg>">
<style>
:root{--bg:#0d1220;--panel:#161d31;--panel2:#1c2540;--tx:#eef1f8;--tx2:#9aa5c0;--lin:#2a3554;
--v:#34d399;--e:#fbbf24;--d:#fb7185;--ac:#60a5fa}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth;scroll-padding-top:64px}
body{background:var(--bg);color:var(--tx);font:16px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;padding-bottom:48px}
.wrap{max-width:980px;margin:0 auto;padding:0 16px}
nav{position:sticky;top:0;z-index:50;background:rgba(13,18,32,.92);backdrop-filter:blur(8px);
border-bottom:1px solid var(--lin)}
nav .wrap{display:flex;gap:4px;overflow-x:auto;padding:10px 16px;-webkit-overflow-scrolling:touch}
nav a{color:var(--tx2);text-decoration:none;font-size:.85rem;font-weight:600;padding:5px 12px;
border-radius:999px;white-space:nowrap}
nav a:hover,nav a:focus{color:var(--tx);background:var(--panel2)}
header{padding:30px 0 8px;text-align:center}
header h1{font-size:1.65rem;font-weight:700}
header p{color:var(--tx2);font-size:.95rem;max-width:640px;margin:8px auto 0}
.badge{display:inline-block;background:var(--panel2);border:1px solid var(--lin);color:var(--tx2);
border-radius:999px;padding:2px 12px;font-size:.8rem;margin-top:10px}
h2{font-size:1.15rem;margin:34px 0 14px;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
h2 small{color:var(--tx2);font-weight:400;font-size:.85rem}
.fl{width:20px;height:15px;border-radius:2px;vertical-align:-2px;object-fit:cover;background:var(--panel2)}
.dias{display:flex;gap:6px;overflow-x:auto;padding:2px 2px 10px;-webkit-overflow-scrolling:touch;scrollbar-width:none}
.dias::-webkit-scrollbar,nav .wrap::-webkit-scrollbar{display:none}
nav .wrap{scrollbar-width:none}
.dias button{flex:0 0 auto;background:var(--panel);border:1px solid var(--lin);color:var(--tx2);
border-radius:999px;padding:5px 13px;font-size:.83rem;font-weight:600;cursor:pointer}
.dias button.sel{background:var(--ac);border-color:var(--ac);color:#0d1220}
.dias button.eshoy:not(.sel){border-color:var(--ac);color:var(--ac)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:14px}
.card{background:var(--panel);border:1px solid var(--lin);border-radius:14px;padding:16px}
.card.hoy{border-color:var(--ac)}
.enc{display:flex;justify-content:space-between;align-items:baseline;font-size:.8rem;color:var(--tx2);margin-bottom:8px}
.eqs{display:flex;justify-content:space-between;align-items:center;gap:8px;font-weight:600;font-size:1rem}
.eqs span{display:flex;align-items:center;gap:6px;min-width:0}
.eqs span:last-child{justify-content:flex-end;text-align:right}
.marc{background:var(--panel2);border:1px solid var(--lin);border-radius:10px;padding:2px 10px;font-size:1.05rem;white-space:nowrap}
.barra{display:flex;height:9px;border-radius:6px;overflow:hidden;margin:12px 0 4px;background:var(--panel2)}
.barra i{display:block;height:100%}
.leyenda{display:flex;justify-content:space-between;font-size:.78rem;color:var(--tx2)}
.stats{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px;font-size:.8rem}
.stat{background:var(--panel2);border-radius:10px;padding:8px 10px}
.stat b{display:block;font-size:.86rem;color:var(--tx)}
.stat span{color:var(--tx2)}
.pasado{opacity:.55}
.etiq{font-size:.72rem;border:1px solid var(--lin);border-radius:6px;padding:1px 6px;color:var(--tx2)}
.etiq.jugado{border-color:var(--e);color:var(--e)}
table{width:100%;border-collapse:collapse;font-size:.85rem}
th,td{padding:7px 8px;text-align:left;border-bottom:1px solid var(--lin)}
th{color:var(--tx2);font-weight:600;font-size:.78rem;text-transform:uppercase;letter-spacing:.04em}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
td .fl{margin-right:5px}
.vd{color:var(--v)}.em{color:var(--e)}.dr{color:var(--d)}
details{background:var(--panel);border:1px solid var(--lin);border-radius:12px;margin-bottom:10px;overflow:hidden}
summary{cursor:pointer;padding:12px 16px;font-weight:600;list-style:none;display:flex;align-items:center;gap:8px}
summary::after{content:"+";color:var(--tx2);margin-left:auto}
details[open] summary::after{content:"–"}
details .inner{padding:0 16px 14px;overflow-x:auto}
.subtit{font-size:.78rem;text-transform:uppercase;letter-spacing:.04em;color:var(--tx2);margin:14px 0 6px;font-weight:600}
.mcbar{height:8px;background:var(--panel2);border-radius:5px;overflow:hidden;min-width:70px}
.mcbar i{display:block;height:100%;background:var(--ac)}
.oculto{display:none}
.btn{display:block;margin:12px auto 2px;background:var(--panel2);border:1px solid var(--lin);color:var(--tx);
border-radius:999px;padding:7px 18px;font-size:.85rem;font-weight:600;cursor:pointer}
.btn:hover{border-color:var(--ac)}
.aviso{background:var(--panel);border:1px solid var(--lin);border-left:4px solid var(--e);border-radius:0 12px 12px 0;
padding:14px 16px;font-size:.85rem;color:var(--tx2);margin-top:36px}
footer{text-align:center;color:var(--tx2);font-size:.8rem;margin-top:30px}
footer a{color:var(--ac);text-decoration:none}
.vacio{background:var(--panel);border:1px dashed var(--lin);border-radius:12px;padding:22px;text-align:center;color:var(--tx2);grid-column:1/-1}
@media(max-width:560px){.stats{grid-template-columns:1fr}}
</style>
</head>
<body>
<nav><div class="wrap">
  <a href="#t-dias">📅 Partidos</a>
  <a href="#t-mc">🏆 Campeón</a>
  <a href="#t-grupos">📋 Grupos</a>
  <a href="#t-cuadro">🗺️ Cuadro</a>
</div></nav>

<header class="wrap">
  <h1>⚽ Mundial 2026 — predicciones del modelo</h1>
  <p>Probabilidades de victoria, córners, tarjetas y marcador más probable para cada partido,
  generadas con dos modelos XGBoost (goles y resultado 1X2 calibrado) y 10.000 mundiales simulados por Monte Carlo.</p>
  <span class="badge">Datos regenerados: __FECHA_GEN__ · la jornada inicial se elige con tu fecha local</span>
</header>

<main class="wrap">
  <h2 id="t-dias">📅 Partidos por jornada <small id="sub-dia"></small></h2>
  <div class="dias" id="dias" role="tablist" aria-label="Elegir jornada"></div>
  <div id="dia" class="cards"></div>

  <h2 id="t-mc">🏆 Probabilidades de campeón <small>Monte Carlo · 10.000 simulaciones</small></h2>
  <div class="card"><div style="overflow-x:auto">
  <table id="tabla-mc"><thead><tr><th>#</th><th>Selección</th><th class="num">Campeón</th><th class="num">Final</th><th class="num">Semis</th><th class="num">Pasa grupos</th><th style="width:26%"></th></tr></thead><tbody></tbody></table>
  </div>
  <button class="btn" id="btn-mc" type="button">Ver las 48 selecciones</button></div>

  <h2 id="t-grupos">📋 Fase de grupos <small>partidos + clasificación predicha</small></h2>
  <div id="grupos"></div>

  <h2 id="t-cuadro">🗺️ Cuadro de eliminatorias predicho <small>resultado más probable de cada cruce</small></h2>
  <div id="cuadro"></div>

  <div class="aviso"><b>⚠️ Esto son probabilidades, no certezas.</b> El modelo acierta ~65% de los ganadores en datos
  de prueba; un favorito del 80% pierde 1 de cada 5 veces. Las estimaciones de córners y tarjetas son aproximaciones
  Poisson sobre las medias de cada selección. Nada de esta página es consejo de apuestas: si juegas, que sea solo
  entretenimiento, con límites y con dinero que no te duela perder.</div>

  <footer>Generado automáticamente con el modelo de
  <a href="https://github.com/jytsss/Simulaciones_Mundial">Simulaciones_Mundial</a> · @jyts__</footer>
</main>

<script>
const PARTIDOS = __PARTIDOS_JSON__;
const CAMPEON = __CAMPEON_JSON__;
const CUADRO = __CUADRO_JSON__;
const TABLAS = __TABLAS_JSON__;
const ISO = __ISO_JSON__;

const isoLocal = d => d.getFullYear()+'-'+String(d.getMonth()+1).padStart(2,'0')+'-'+String(d.getDate()).padStart(2,'0');
const HOY = isoLocal(new Date());
const FECHAS = [...new Set(PARTIDOS.map(p=>p.fecha))].sort();
const FIN_GRUPOS = FECHAS[FECHAS.length-1];

const fl = eq => ISO[eq] ? `<img class="fl" loading="lazy" alt="" src="https://flagcdn.com/w20/${ISO[eq]}.png" srcset="https://flagcdn.com/w40/${ISO[eq]}.png 2x">` : '';
function fmtFecha(f){
  const [y,m,d] = f.split('-');
  const meses=['ene','feb','mar','abr','may','jun','jul','ago','sep','oct','nov','dic'];
  return (+d)+' '+meses[+m-1];
}

function tarjetaPartido(p){
  const div = document.createElement('div');
  div.className = 'card' + (p.fecha === HOY ? ' hoy' : '');
  const jugado = p.fecha < HOY ? '<span class="etiq jugado">ya jugado</span>' : `<span class="etiq">xG ${p.xgl} – ${p.xgv}</span>`;
  div.innerHTML = `
    <div class="enc"><span>Grupo ${p.grupo} · ${fmtFecha(p.fecha)}</span>${jugado}</div>
    <div class="eqs"><span>${fl(p.local)}${p.local}</span><span class="marc">${p.marcador}</span><span>${p.visitante}${fl(p.visitante)}</span></div>
    <div class="barra">
      <i style="width:${p.p1}%;background:var(--v)"></i>
      <i style="width:${p.px}%;background:var(--e)"></i>
      <i style="width:${p.p2}%;background:var(--d)"></i>
    </div>
    <div class="leyenda"><span class="vd">1 · ${p.p1}%</span><span class="em">X · ${p.px}%</span><span class="dr">2 · ${p.p2}%</span></div>
    <div class="stats">
      <div class="stat"><b>⛳ Córners: ${p.cor} esperados</b>
        <span>+7.5: ${p.c75}% · +8.5: ${p.c85}% · +9.5: ${p.c95}%</span></div>
      <div class="stat"><b>🟨 Tarjetas: ${p.tar} esperadas</b>
        <span>+3.5: ${p.t35}% · +4.5: ${p.t45}%</span></div>
    </div>`;
  return div;
}

let fechaSel = FECHAS.includes(HOY) ? HOY : (FECHAS.find(f => f > HOY) || FIN_GRUPOS);

function pintarDia(){
  const cont = document.getElementById('dia');
  cont.innerHTML = '';
  const lista = PARTIDOS.filter(p => p.fecha === fechaSel);
  document.getElementById('sub-dia').textContent =
    fechaSel === HOY ? 'hoy · ' + fmtFecha(fechaSel) : fmtFecha(fechaSel);
  if (!lista.length){
    const v = document.createElement('div');
    v.className = 'vacio';
    v.textContent = 'No hay partidos de grupos este día.';
    cont.appendChild(v);
  } else {
    lista.forEach(p => cont.appendChild(tarjetaPartido(p)));
  }
  if (HOY > FIN_GRUPOS){
    const v = document.createElement('div');
    v.className = 'vacio';
    v.innerHTML = 'Fase de grupos terminada: consulta el <a href="#t-cuadro" style="color:var(--ac)">cuadro de eliminatorias predicho</a>. Los cruces reales se conocen al cerrar los grupos.';
    cont.appendChild(v);
  }
  document.querySelectorAll('#dias button').forEach(b => b.classList.toggle('sel', b.dataset.f === fechaSel));
}

const contD = document.getElementById('dias');
FECHAS.forEach(f => {
  const b = document.createElement('button');
  b.type = 'button'; b.dataset.f = f;
  b.textContent = f === HOY ? 'HOY · ' + fmtFecha(f) : fmtFecha(f);
  if (f === HOY) b.classList.add('eshoy');
  b.onclick = () => { fechaSel = f; pintarDia(); };
  contD.appendChild(b);
});
pintarDia();
const selBtn = document.querySelector('#dias button.sel');
if (selBtn) selBtn.scrollIntoView({block:'nearest', inline:'center'});

const tb = document.querySelector('#tabla-mc tbody');
CAMPEON.forEach((c,i) => {
  const tr = document.createElement('tr');
  if (i >= 12) tr.classList.add('oculto','fila-extra');
  tr.innerHTML = `<td>${i+1}</td><td>${fl(c.equipo)}${c.equipo}</td><td class="num"><b>${c.campeon.toFixed(1)}%</b></td>
  <td class="num">${c.final.toFixed(1)}%</td><td class="num">${c.semis.toFixed(1)}%</td><td class="num">${c.r32.toFixed(0)}%</td>
  <td><div class="mcbar"><i style="width:${Math.min(100, c.campeon*3.3)}%"></i></div></td>`;
  tb.appendChild(tr);
});
document.getElementById('btn-mc').onclick = function(){
  const ocultas = document.querySelectorAll('.fila-extra');
  const abrir = ocultas[0] && ocultas[0].classList.contains('oculto');
  ocultas.forEach(f => f.classList.toggle('oculto', !abrir));
  this.textContent = abrir ? 'Mostrar solo el top 12' : 'Ver las 48 selecciones';
};

const contG = document.getElementById('grupos');
[...new Set(PARTIDOS.map(p=>p.grupo))].sort().forEach(g => {
  const det = document.createElement('details');
  const filas = PARTIDOS.filter(p=>p.grupo===g).map(p => `
    <tr class="${p.fecha < HOY ? 'pasado' : ''}">
      <td>${fmtFecha(p.fecha)}</td><td>${fl(p.local)}${p.local} – ${fl(p.visitante)}${p.visitante}</td><td class="num"><b>${p.marcador}</b></td>
      <td class="num vd">${p.p1}%</td><td class="num em">${p.px}%</td><td class="num dr">${p.p2}%</td>
      <td class="num">${p.cor} <span style="color:var(--tx2)">(+7.5: ${p.c75}%)</span></td>
      <td class="num">${p.tar} <span style="color:var(--tx2)">(+3.5: ${p.t35}%)</span></td>
    </tr>`).join('');
  const tabla = (TABLAS[g]||[]).map(t => {
    const marca = t.pos <= 2 ? ' ✅' : (t.pos === 3 ? ' 🟡' : '');
    return `<tr><td>${t.pos}</td><td>${fl(t.equipo)}${t.equipo}${marca}</td>
    <td class="num">${t.pts}</td><td class="num">${t.dg > 0 ? '+' : ''}${t.dg.toFixed(2)}</td></tr>`;
  }).join('');
  det.innerHTML = `<summary>Grupo ${g}</summary><div class="inner">
    <div class="subtit">Partidos</div>
    <table><thead><tr><th>Fecha</th><th>Partido</th><th class="num">Pred.</th><th class="num">1</th>
    <th class="num">X</th><th class="num">2</th><th class="num">Córners</th><th class="num">Tarjetas</th></tr></thead>
    <tbody>${filas}</tbody></table>
    <div class="subtit">Clasificación predicha <span style="text-transform:none;font-weight:400">(✅ clasifica · 🟡 tercero: pasan los 8 mejores)</span></div>
    <table><thead><tr><th>#</th><th>Selección</th><th class="num">Pts</th><th class="num">DG (xG)</th></tr></thead>
    <tbody>${tabla}</tbody></table></div>`;
  contG.appendChild(det);
});

const contC = document.getElementById('cuadro');
[...new Set(CUADRO.map(c=>c.fase))].forEach(f => {
  const det = document.createElement('details');
  if (f === 'Final') det.open = true;
  const filas = CUADRO.filter(c=>c.fase===f).map(c => `
    <tr><td>${fl(c.local)}${c.local} – ${fl(c.visitante)}${c.visitante}</td><td class="num"><b>${c.marcador}</b></td><td><b>${fl(c.avanza)}${c.avanza}</b></td>
    <td class="num vd">${c.p1}%</td><td class="num em">${c.px}%</td><td class="num dr">${c.p2}%</td></tr>`).join('');
  det.innerHTML = `<summary>${f}<span class="etiq" style="margin-left:6px">${CUADRO.find(c=>c.fase===f).fechas}</span></summary>
    <div class="inner"><table><thead><tr><th>Cruce</th><th class="num">Pred.</th><th>Avanza</th>
    <th class="num">1</th><th class="num">X</th><th class="num">2</th></tr></thead><tbody>${filas}</tbody></table></div>`;
  contC.appendChild(det);
});
</script>
</body>
</html>
"""


if __name__ == '__main__':
    partidos, campeon, cuadro, tablas = construir_datos()
    html = (PLANTILLA
            .replace('__PARTIDOS_JSON__', json.dumps(partidos, ensure_ascii=False))
            .replace('__CAMPEON_JSON__', json.dumps(campeon, ensure_ascii=False))
            .replace('__CUADRO_JSON__', json.dumps(cuadro, ensure_ascii=False))
            .replace('__TABLAS_JSON__', json.dumps(tablas, ensure_ascii=False))
            .replace('__ISO_JSON__', json.dumps(BANDERAS_ISO, ensure_ascii=False))
            .replace('__FECHA_GEN__', datetime.now(timezone.utc).strftime('%d-%m-%Y %H:%M UTC')))
    os.makedirs('docs', exist_ok=True)
    with open('docs/index.html', 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Web generada en docs/index.html ({len(partidos)} partidos, {len(cuadro)} cruces, '
          f'{len(campeon)} equipos en la tabla MC, {len(tablas)} grupos).')
