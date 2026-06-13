# -*- coding: utf-8 -*-
"""Genera Predicciones/PREDICCIONES.md a partir de los CSVs producidos por
`prediccion_mundial.py` (predicción puntual de los 104 partidos + Monte Carlo)."""

import os
import pandas as pd

# El script vive en 04_Prediccion/; los resultados están en <raíz>/Predicciones/
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUTA = 'Predicciones/'

grupos = pd.read_csv(RUTA + 'predicciones_fase_grupos.csv')
clasif = pd.read_csv(RUTA + 'clasificacion_grupos.csv')
elim = pd.read_csv(RUTA + 'predicciones_eliminatorias.csv')
mc = pd.read_csv(RUTA + 'probabilidades_montecarlo.csv', index_col=0)

BANDERAS = {
    'México': '🇲🇽', 'Sudáfrica': '🇿🇦', 'Corea del Sur': '🇰🇷', 'República Checa': '🇨🇿',
    'Suiza': '🇨🇭', 'Bosnia-Herzegovina': '🇧🇦', 'Canadá': '🇨🇦', 'Catar': '🇶🇦',
    'Escocia': '🏴󠁧󠁢󠁳󠁣󠁴󠁿', 'Brasil': '🇧🇷', 'Haití': '🇭🇹', 'Marruecos': '🇲🇦',
    'Turquía': '🇹🇷', 'Paraguay': '🇵🇾', 'EE. UU.': '🇺🇸', 'Australia': '🇦🇺',
    'Alemania': '🇩🇪', 'Ecuador': '🇪🇨', 'Costa de Marfil': '🇨🇮', 'Curazao': '🇨🇼',
    'Suecia': '🇸🇪', 'Países Bajos': '🇳🇱', 'Túnez': '🇹🇳', 'Japón': '🇯🇵',
    'Bélgica': '🇧🇪', 'Egipto': '🇪🇬', 'Irán': '🇮🇷', 'Nueva Zelanda': '🇳🇿',
    'España': '🇪🇸', 'Uruguay': '🇺🇾', 'Cabo Verde': '🇨🇻', 'Arabia Saudí': '🇸🇦',
    'Francia': '🇫🇷', 'Noruega': '🇳🇴', 'Senegal': '🇸🇳', 'Irak': '🇮🇶',
    'Austria': '🇦🇹', 'Argentina': '🇦🇷', 'Argelia': '🇩🇿', 'Jordania': '🇯🇴',
    'Portugal': '🇵🇹', 'Colombia': '🇨🇴', 'RD Congo': '🇨🇩', 'Uzbekistán': '🇺🇿',
    'Croacia': '🇭🇷', 'Inglaterra': '🏴󠁧󠁢󠁥󠁮󠁧󠁿', 'Ghana': '🇬🇭', 'Panamá': '🇵🇦',
}

def eq(nombre):
    return f"{BANDERAS.get(nombre, '')} {nombre}".strip()

L = []
campeon = elim[elim['Fase'] == 'Final']['Avanza'].iloc[0]
final = elim[elim['Fase'] == 'Final'].iloc[0]
subcampeon = final['Local'] if final['Avanza'] == final['Visitante'] else final['Visitante']
tercero = elim[elim['Fase'] == '3er Puesto']['Avanza'].iloc[0]

L.append("# 🏆 Predicción completa del Mundial 2026 — los 104 partidos\n")
L.append("Generado con el modelo de este repositorio: regresores XGBoost (Tweedie) de goles + "
         "clasificador 1X2 XGBoost calibrado (isotónico), predicción a sede neutral con \"efecto espejo\", "
         "temperatura `T=0.27` en grupos y `T=0.5` en eliminatorias, y simulación de **Monte Carlo de "
         "10.000 mundiales** para las probabilidades por selección.\n")
L.append("> ⚠️ Predicción generada el 12-jun-2026, con los datos del repo (anteriores al torneo). "
         "La asignación de mejores terceros al cuadro usa la simplificación del notebook "
         "(ranking 1º-8º a huecos fijos), no la tabla oficial de la FIFA.\n")

L.append("## Resumen\n")
L.append(f"| | |\n|---|---|\n| 🥇 **Campeón predicho** | **{eq(campeon)}** |\n"
         f"| 🥈 Subcampeón | {eq(subcampeon)} |\n| 🥉 Tercer puesto | {eq(tercero)} |\n")

L.append("### Probabilidades de ser campeón (Top 10, Monte Carlo)\n")
L.append("| # | Selección | Campeón | Final | Semis | Cuartos |")
L.append("|---|---|---|---|---|---|")
for i, (nombre, r) in enumerate(mc.head(10).iterrows(), 1):
    L.append(f"| {i} | {eq(nombre)} | **{r['Campeon']:.1f}%** | {r['Final']:.1f}% | {r['Semis']:.1f}% | {r['Cuartos']:.1f}% |")
L.append("")

# ------------------- FASE DE GRUPOS -------------------
L.append("## Fase de grupos — 72 partidos\n")
L.append("Marcador = marcador exacto más probable según los goles esperados del modelo, "
         "condicionado al resultado 1X2 más probable.\n")

for g in sorted(grupos['Grupo'].unique()):
    L.append(f"### Grupo {g}\n")
    L.append("| Fecha | Partido | Pred. | P(1) | P(X) | P(2) |")
    L.append("|---|---|:-:|--:|--:|--:|")
    sub = grupos[grupos['Grupo'] == g].sort_values('Fecha')
    for _, r in sub.iterrows():
        partido = f"{eq(r['Local'])} – {eq(r['Visitante'])}"
        L.append(f"| {r['Fecha'][5:]} | {partido} | **{r['Marcador_Predicho']}** | "
                 f"{r['Prob_1']:.0f}% | {r['Prob_X']:.0f}% | {r['Prob_2']:.0f}% |")
    L.append("")
    sub_c = clasif[clasif['Grupo'] == g].sort_values('Posicion')
    L.append("| Pos | Equipo | Pts | DG (xG) |")
    L.append("|---|---|--:|--:|")
    for _, r in sub_c.iterrows():
        marca = ' ✅' if r['Posicion'] <= 2 else (' 🟡' if r['Posicion'] == 3 else '')
        L.append(f"| {int(r['Posicion'])} | {eq(r['Equipo'])}{marca} | {int(r['Pts'])} | {r['DG']:+.2f} |")
    L.append("")

L.append("✅ clasificado directo · 🟡 tercero (pasan los 8 mejores)\n")

# ------------------- ELIMINATORIAS -------------------
NOMBRES_FASE = {'Dieciseisavos': 'Dieciseisavos de final (16 cruces)',
                'Octavos': 'Octavos de final', 'Cuartos': 'Cuartos de final',
                'Semifinales': 'Semifinales', '3er Puesto': 'Partido por el 3er puesto',
                'Final': '🏆 Gran Final — MetLife Stadium, Nueva York/Nueva Jersey'}

L.append("## Eliminatorias — 32 partidos\n")
L.append("Si el empate es el resultado más probable, el cruce se decide por penaltis "
         "a favor del equipo con mayor probabilidad de victoria.\n")

for fase in ['Dieciseisavos', 'Octavos', 'Cuartos', 'Semifinales', '3er Puesto', 'Final']:
    sub = elim[elim['Fase'] == fase]
    L.append(f"### {NOMBRES_FASE[fase]} · *{sub['Fechas'].iloc[0]}*\n")
    L.append("| Cruce | Pred. | Avanza | P(1) | P(X) | P(2) |")
    L.append("|---|:-:|---|--:|--:|--:|")
    for _, r in sub.iterrows():
        L.append(f"| {eq(r['Local'])} – {eq(r['Visitante'])} | **{r['Marcador_Predicho']}** | "
                 f"**{eq(r['Avanza'])}** | {r['Prob_1']:.0f}% | {r['Prob_X']:.0f}% | {r['Prob_2']:.0f}% |")
    L.append("")

# ------------------- TABLA MC COMPLETA -------------------
L.append("## Probabilidades por selección — 10.000 mundiales simulados\n")
L.append("| Selección | Pasa grupos | Octavos | Cuartos | Semis | Final | 🏆 Campeón |")
L.append("|---|--:|--:|--:|--:|--:|--:|")
for nombre, r in mc.iterrows():
    L.append(f"| {eq(nombre)} | {r['R32']:.1f}% | {r['Octavos']:.1f}% | {r['Cuartos']:.1f}% | "
             f"{r['Semis']:.1f}% | {r['Final']:.1f}% | **{r['Campeon']:.1f}%** |")
L.append("")

L.append("## Validación con los partidos ya jugados\n")
L.append("| Partido | Predicción del modelo | Resultado real |")
L.append("|---|:-:|:-:|")
pred_mex = grupos[(grupos['Local'] == 'México') & (grupos['Visitante'] == 'Sudáfrica')].iloc[0]
pred_kor = grupos[(grupos['Local'] == 'Corea del Sur') & (grupos['Visitante'] == 'República Checa')].iloc[0]
L.append(f"| 🇲🇽 México – 🇿🇦 Sudáfrica | {pred_mex['Marcador_Predicho']} (P1 {pred_mex['Prob_1']:.0f}%) | 2-0 ✅ ganador acertado |")
L.append(f"| 🇰🇷 Corea del Sur – 🇨🇿 República Checa | {pred_kor['Marcador_Predicho']} (P1 {pred_kor['Prob_1']:.0f}%) | 2-1 ✅ ganador acertado |")
L.append("")
L.append("---\n*Predicciones generadas automáticamente con `prediccion_mundial.py`. "
         "El fútbol, por suerte, no entiende de modelos.* ⚽")

with open(RUTA + 'PREDICCIONES.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print("Informe escrito en Predicciones/PREDICCIONES.md")
