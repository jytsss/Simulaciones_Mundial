# -*- coding: utf-8 -*-
"""
Pipeline end-to-end del Mundial 2026 (port a script de `Modelling.ipynb`).

Pasos:
  1) Entrena los modelos del repo: 2 regresores XGBoost (Tweedie) de goles
     Local/Visitante + clasificador 1X2 XGBoost calibrado (isotónico).
  2) Predice los 72 partidos de la fase de grupos (sede neutral + espejo, T=0.27).
  3) Precalcula la matriz de cruces 48x48 con el mismo pipeline (T=0.5, como
     usa `simular_cruce` en el notebook) para acelerar las eliminatorias.
  4) Genera la PREDICCIÓN puntual de los 104 partidos: resultado más probable
     en grupos -> clasificación -> cuadro completo hasta la final (+ 3er puesto).
  5) Simulación de Monte Carlo de 10.000 mundiales para las probabilidades de
     cada selección (R32/Octavos/Cuartos/Semis/Final/Campeón).
  6) Escribe los resultados en `Predicciones/`.

Uso:  python prediccion_mundial.py
"""

import os
import math
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import defaultdict
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_predict, KFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.calibration import CalibratedClassifierCV

# El script vive en 04_Prediccion/; trabajamos desde la raíz del repo
RUTA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RUTA)
os.makedirs('Predicciones', exist_ok=True)

np.random.seed(42)

N_SIMULACIONES = 10_000   # Mundiales de Monte Carlo (como indica el README)
T_GRUPOS = 0.27           # Temperatura usada en el notebook para la fase de grupos
T_CRUCES = 0.5            # Temperatura por defecto que usa simular_cruce en el notebook

# Mejores hiperparámetros del 1X2 registrados en la salida del notebook (celda 9)
BEST_PARAMS_1X2 = {'subsample': 1.0, 'reg_lambda': 1.0, 'n_estimators': 300,
                   'max_depth': 2, 'learning_rate': 0.005, 'gamma': 0.1,
                   'colsample_bytree': 0.8}

# ====================================================================
# FUNCIONES AUXILIARES DE INGENIERÍA DE VARIABLES (idénticas al notebook)
# ====================================================================

def asignar_tier(puntos):
    if puntos >= 1700: return 1   # Élite (Francia, Argentina, España...)
    elif puntos >= 1600: return 2 # Nivel Alto (Suiza, Senegal, Japón...)
    elif puntos >= 1500: return 3 # Nivel Medio (Haití, Jordania...)
    else: return 4                # Nivel Bajo (Curazao, etc.)

mapa_continentes = {
    'República Checa': 'Europa', 'Bosnia-Herzegovina': 'Europa', 'Suiza': 'Europa',
    'Países Bajos': 'Europa', 'Alemania': 'Europa', 'Escocia': 'Europa',
    'Turquía': 'Europa', 'Suecia': 'Europa', 'España': 'Europa',
    'Bélgica': 'Europa', 'Francia': 'Europa', 'Croacia': 'Europa',
    'Austria': 'Europa', 'Portugal': 'Europa', 'Inglaterra': 'Europa',
    'Noruega': 'Europa',
    'Paraguay': 'Sudamérica', 'Brasil': 'Sudamérica', 'Ecuador': 'Sudamérica',
    'Uruguay': 'Sudamérica', 'Argentina': 'Sudamérica', 'Colombia': 'Sudamérica',
    'México': 'Norteamérica', 'Canadá': 'Norteamérica', 'EE. UU.': 'Norteamérica',
    'Haití': 'Norteamérica', 'Curazao': 'Norteamérica', 'Panamá': 'Norteamérica',
    'Sudáfrica': 'Africa', 'Marruecos': 'Africa', 'Egipto': 'Africa',
    'Túnez': 'Africa', 'Costa de Marfil': 'Africa', 'Cabo Verde': 'Africa',
    'Senegal': 'Africa', 'RD Congo': 'Africa', 'Argelia': 'Africa',
    'Ghana': 'Africa',
    'Corea del Sur': 'Asia', 'Catar': 'Asia', 'Japón': 'Asia',
    'Australia': 'Asia', 'Irán': 'Asia', 'Arabia Saudí': 'Asia',
    'Jordania': 'Asia', 'Irak': 'Asia', 'Uzbekistán': 'Asia',
    'Nueva Zelanda': 'Asia'
}

pesos_continente = {
    'Europa': 1.00, 'Sudamérica': .95, 'Norteamérica': 0.75,
    'Africa': 0.6, 'Asia': 0.7, 'Oceanía': 0.5
}

# ====================================================================
# 1. ENTRENAMIENTO (celdas 2-9 y 18 del notebook)
# ====================================================================

def entrenar_modelos():
    df = pd.read_csv('./Data/datos_historicos.csv')
    df.sort_values('Fecha', inplace=True)
    df.dropna(inplace=True)

    df['Resultado_1X2_Num'] = df['Resultado_1X2'].map({'1': 0, 'X': 1, '2': 2})
    y = df['Resultado_1X2_Num']
    X = df.drop(columns=['Fecha', 'Equipo_Local', 'Equipo_Visitante',
                         'Resultado_1X2', 'Resultado_1X2_Num', 'Goles_Local', 'Goles_Visitante'])

    split_index = int(len(df) * 0.85)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train_1X2, y_test_1X2 = y.iloc[:split_index], y.iloc[split_index:]

    y_train_goles_L = df['Goles_Local'].iloc[:split_index]
    y_train_goles_V = df['Goles_Visitante'].iloc[:split_index]

    tscv = TimeSeriesSplit(n_splits=5)

    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2, 0.5],
        'max_depth': [2, 3, 5, 8],
        'subsample': [0.8, 0.9, 1.0],
        'reg_lambda': [0.1, 0.5, 1.0, 5.0],
        'gamma': [0, 0.1],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }

    print("Fase 1: Entrenando predictores de Goles (Tweedie)...")
    pesos_train_L = np.where(y_train_goles_L >= 3, 1.5, 1.0)
    pesos_train_V = np.where(y_train_goles_V >= 3, 1.5, 1.0)

    n_iter_reg = int(os.environ.get('N_ITER_REG', '300'))
    xgb_reg_L = xgb.XGBRegressor(objective='reg:tweedie', tweedie_variance_power=1.5, random_state=42)
    xgb_reg_V = xgb.XGBRegressor(objective='reg:tweedie', tweedie_variance_power=1.5, random_state=42)
    search_L = RandomizedSearchCV(xgb_reg_L, param_grid, cv=tscv, n_iter=n_iter_reg,
                                  scoring='neg_mean_poisson_deviance', random_state=42, n_jobs=-1)
    search_V = RandomizedSearchCV(xgb_reg_V, param_grid, cv=tscv, n_iter=n_iter_reg,
                                  scoring='neg_mean_poisson_deviance', random_state=42, n_jobs=-1)
    search_L.fit(X_train, y_train_goles_L, sample_weight=pesos_train_L)
    search_V.fit(X_train, y_train_goles_V, sample_weight=pesos_train_V)
    mejor_modelo_L = search_L.best_estimator_
    mejor_modelo_V = search_V.best_estimator_
    print("  Mejores parámetros Goles L:", search_L.best_params_)
    print("  Mejores parámetros Goles V:", search_V.best_params_)

    # --- Evaluación en Test con los hiperparámetros del 1X2 del notebook ---
    print("Fase 2: Clasificador 1X2 (hiperparámetros registrados en el notebook) + calibración...")
    kf_meta = KFold(n_splits=5, shuffle=False)
    pred_goles_L_train = cross_val_predict(mejor_modelo_L, X_train, y_train_goles_L, cv=kf_meta,
                                           params={'sample_weight': pesos_train_L}) ** 1.5 * X_train['Peso_Local']
    pred_goles_V_train = cross_val_predict(mejor_modelo_V, X_train, y_train_goles_V, cv=kf_meta,
                                           params={'sample_weight': pesos_train_V}) ** 1.5 * X_train['Peso_Visitante']
    pred_goles_L_test = mejor_modelo_L.predict(X_test) ** 1.5 * X_test['Peso_Local']
    pred_goles_V_test = mejor_modelo_V.predict(X_test) ** 1.5 * X_test['Peso_Visitante']

    X_train_meta = X_train.copy(); X_test_meta = X_test.copy()
    X_train_meta['Pred_Goles_L'] = pred_goles_L_train
    X_train_meta['Pred_Goles_V'] = pred_goles_V_train
    X_test_meta['Pred_Goles_L'] = pred_goles_L_test
    X_test_meta['Pred_Goles_V'] = pred_goles_V_test

    modelo_1X2_test = xgb.XGBClassifier(objective='multi:softprob', num_class=3, base_score=0.5,
                                        random_state=42, **BEST_PARAMS_1X2)
    calibrado_test = CalibratedClassifierCV(estimator=modelo_1X2_test, method='isotonic', cv=tscv)
    calibrado_test.fit(X_train_meta, y_train_1X2)
    pred_test = calibrado_test.predict(X_test_meta)
    print("\n--- RESULTADOS DEL MODELO EN TEST (15% temporal) ---")
    print(classification_report(y_test_1X2, pred_test))
    print(f"Accuracy test: {accuracy_score(y_test_1X2, pred_test):.3f}")

    # --- FASE DE PRODUCCIÓN: reentrenamos con el 100% de los datos (celda 18) ---
    print("Fase 3: Reentrenando con el 100% de los datos (producción)...")
    y_goles_L_full = df['Goles_Local']; y_goles_V_full = df['Goles_Visitante']
    pesos_full_L = np.where(y_goles_L_full >= 3, 1.5, 1.0)
    pesos_full_V = np.where(y_goles_V_full >= 3, 1.5, 1.0)

    kf_meta_full = KFold(n_splits=5, shuffle=False)
    pred_goles_L_full = cross_val_predict(mejor_modelo_L, X, y_goles_L_full, cv=kf_meta_full,
                                          params={'sample_weight': pesos_full_L})
    pred_goles_V_full = cross_val_predict(mejor_modelo_V, X, y_goles_V_full, cv=kf_meta_full,
                                          params={'sample_weight': pesos_full_V})
    X_meta_full = X.copy()
    X_meta_full['Pred_Goles_L'] = pred_goles_L_full
    X_meta_full['Pred_Goles_V'] = pred_goles_V_full

    mejor_modelo_L.fit(X, y_goles_L_full, sample_weight=pesos_full_L)
    mejor_modelo_V.fit(X, y_goles_V_full, sample_weight=pesos_full_V)

    modelo_1X2_final = xgb.XGBClassifier(objective='multi:softprob', num_class=3, base_score=0.5,
                                         random_state=42, **BEST_PARAMS_1X2)
    clasificador_calibrado_final = CalibratedClassifierCV(estimator=modelo_1X2_final, method='isotonic', cv=tscv)
    clasificador_calibrado_final.fit(X_meta_full, y)

    joblib.dump(mejor_modelo_L, 'modelo_goles_L.pkl')
    joblib.dump(mejor_modelo_V, 'modelo_goles_V.pkl')
    joblib.dump(clasificador_calibrado_final, 'modelo_1X2_calibrado.pkl')
    joblib.dump(list(X.columns), 'columnas_entrenamiento.pkl')
    print("Modelos guardados (.pkl). ¡Modelo listo para el Mundial!")


# ====================================================================
# 2. PIPELINE DE PREDICCIÓN (celda 23 del notebook; modelos cacheados)
# ====================================================================

_CACHE_MODELOS = {}

def _modelos():
    if not _CACHE_MODELOS:
        _CACHE_MODELOS['L'] = joblib.load('modelo_goles_L.pkl')
        _CACHE_MODELOS['V'] = joblib.load('modelo_goles_V.pkl')
        _CACHE_MODELOS['1X2'] = joblib.load('modelo_1X2_calibrado.pkl')
        _CACHE_MODELOS['cols'] = joblib.load('columnas_entrenamiento.pkl')
    return _CACHE_MODELOS


def pipeline_prediccion(df_bruto, sede_neutral=True, T=0.5):
    m = _modelos()
    modelo_L, modelo_V, modelo_1X2, columnas_base = m['L'], m['V'], m['1X2'], m['cols']

    def obtener_predicciones_crudas(df_temp):
        df_calc = df_temp.copy()

        columnas_promedios_local = [col for col in df_calc.columns
                                    if col.endswith(('_5_Local', '_2_Local', '_total_Local')) and col.startswith('avg_')]
        for col_local in columnas_promedios_local:
            col_visitante = col_local.replace('_Local', '_Visitante')
            if col_visitante in df_calc.columns:
                nombre_base = col_local.replace('_Local', '')
                df_calc[f"diff_{nombre_base}"] = df_calc[col_local] - df_calc[col_visitante]

        if 'Puntos_Local' in df_calc.columns and 'Puntos_Visitante' in df_calc.columns:
            df_calc['diff_Puntos'] = df_calc['Puntos_Local'] - df_calc['Puntos_Visitante']
            df_calc['Prob_Implicita_ELO'] = 1 / (1 + 10 ** (-df_calc['diff_Puntos'] / 400))
            df_calc['diff_Tier'] = df_calc['Puntos_Local'].apply(asignar_tier) - df_calc['Puntos_Visitante'].apply(asignar_tier)

        df_calc['Continente_Local'] = df_calc['Equipo_Local'].map(mapa_continentes)
        df_calc['Continente_Visitante'] = df_calc['Equipo_Visitante'].map(mapa_continentes)
        df_calc['Peso_Local'] = df_calc['Continente_Local'].map(pesos_continente)
        df_calc['Peso_Visitante'] = df_calc['Continente_Visitante'].map(pesos_continente)
        df_calc.drop(['Continente_Local', 'Continente_Visitante'], axis=1, inplace=True)

        for col in columnas_base:
            if col not in df_calc.columns:
                df_calc[col] = 0
        X_listo = df_calc[columnas_base]

        goles_L = modelo_L.predict(X_listo)
        goles_V = modelo_V.predict(X_listo)

        X_meta = X_listo.copy()
        X_meta['Pred_Goles_L'] = goles_L
        X_meta['Pred_Goles_V'] = goles_V
        probs = modelo_1X2.predict_proba(X_meta)
        return goles_L, goles_V, probs

    df_normal = df_bruto.copy()
    cols_contexto = ['Fecha', 'Equipo_Local', 'Equipo_Visitante']
    if 'Grupo' in df_normal.columns:
        cols_contexto.append('Grupo')
    contexto = df_normal[cols_contexto].copy()

    goles_L_norm, goles_V_norm, probs_norm = obtener_predicciones_crudas(df_normal)

    if not sede_neutral:
        resultados = contexto.copy()
        resultados['xG_Modelo_Local'] = goles_L_norm.round(2)
        resultados['xG_Modelo_Visitante'] = goles_V_norm.round(2)
        resultados['Prob_Local'] = probs_norm[:, 0]
        resultados['Prob_Empate'] = probs_norm[:, 1]
        resultados['Prob_Visitante'] = probs_norm[:, 2]
        return resultados

    df_inverso = df_bruto.copy()
    nuevas_columnas = []
    for col in df_inverso.columns:
        if col.endswith('_Local'): nuevas_columnas.append(col.replace('_Local', '_Visitante'))
        elif col.endswith('_Visitante'): nuevas_columnas.append(col.replace('_Visitante', '_Local'))
        else: nuevas_columnas.append(col)
    df_inverso.columns = nuevas_columnas
    goles_L_inv, goles_V_inv, probs_inv = obtener_predicciones_crudas(df_inverso)

    resultados = contexto.copy()
    resultados['xG_Modelo_Local'] = ((goles_L_norm + goles_V_inv) / 2).round(2)
    resultados['xG_Modelo_Visitante'] = ((goles_V_norm + goles_L_inv) / 2).round(2)
    resultados['Prob_Local'] = (probs_norm[:, 0] + probs_inv[:, 2]) / 2
    resultados['Prob_Empate'] = (probs_norm[:, 1] + probs_inv[:, 1]) / 2
    resultados['Prob_Visitante'] = (probs_norm[:, 2] + probs_inv[:, 0]) / 2

    probs_afiladas = resultados[['Prob_Local', 'Prob_Empate', 'Prob_Visitante']] ** (1 / T)
    s = probs_afiladas.sum(axis=1).replace(0, 1e-12)
    resultados[['Prob_Local', 'Prob_Empate', 'Prob_Visitante']] = probs_afiladas.div(s, axis=0)
    return resultados


# ====================================================================
# 3. CARGA DEL MUNDIAL Y PREDICCIÓN DE LA FASE DE GRUPOS (celda 27)
# ====================================================================

def cargar_mundial():
    df_mundial = pd.read_csv('./Data/partidos_mundial.csv')[['Fecha', 'Equipo_Local', 'Equipo_Visitante']]
    df_vars = pd.read_csv('./Data/datos_mundial.csv').sort_values('Fecha')
    grupos = pd.read_csv('./Data/Grupos_Mundial.csv', sep=";")

    aux = df_mundial.merge(df_vars, left_on=['Equipo_Local'], right_on=['Equipo'], how='left')
    df_mundial_vars = aux.merge(df_vars, left_on=['Equipo_Visitante'], right_on=['Equipo'], how='left')
    df_mundial_vars.drop(columns=['Equipo_x', 'Equipo_y', 'Fecha_x', 'Fecha_y',
                                  'Resultado_1X2_x', 'Resultado_1X2_y', 'Tipo_Equipo_x', 'Tipo_Equipo_y'], inplace=True)
    df_mundial_vars.columns = df_mundial_vars.columns.str.replace(r'_x$', '_Local', regex=True)
    df_mundial_vars.columns = df_mundial_vars.columns.str.replace(r'_y$', '_Visitante', regex=True)
    df_mundial_grupos = pd.merge(df_mundial_vars, grupos, left_on='Equipo_Local', right_on='Equipo').drop('Equipo', axis=1)

    # Las fechas reales de cada partido (el merge pisa la columna Fecha con la de df_vars)
    fechas_reales = pd.read_csv('./Data/partidos_mundial.csv')[['Fecha', 'Equipo_Local', 'Equipo_Visitante']]
    return df_mundial_grupos, df_vars, grupos, fechas_reales


# ====================================================================
# 4. MATRIZ DE CRUCES 48x48 (equivalente vectorizado de `simular_cruce`)
# ====================================================================

def matriz_cruces(equipos, df_vars):
    """Predice todos los pares ordenados (A,B) con el mismo merge que simular_cruce
    y T=0.5 (la temperatura por defecto que usa el notebook en las eliminatorias)."""
    pares = [(a, b) for a in equipos for b in equipos if a != b]
    partido = pd.DataFrame({'Fecha': ['2026-07-01'] * len(pares),
                            'Equipo_Local': [a for a, b in pares],
                            'Equipo_Visitante': [b for a, b in pares]})
    partido_vars = partido.merge(df_vars, left_on='Equipo_Local', right_on='Equipo', how='left').drop(columns=['Equipo'])
    partido_vars = partido_vars.merge(df_vars, left_on='Equipo_Visitante', right_on='Equipo', how='left',
                                      suffixes=('_Local', '_Visitante')).drop(columns=['Equipo'])
    df_pred = pipeline_prediccion(partido_vars, sede_neutral=True, T=T_CRUCES)

    idx = {e: i for i, e in enumerate(equipos)}
    n = len(equipos)
    P1 = np.zeros((n, n)); PX = np.zeros((n, n)); P2 = np.zeros((n, n))
    XGL = np.zeros((n, n)); XGV = np.zeros((n, n))
    for (a, b), (_, r) in zip(pares, df_pred.iterrows()):
        i, j = idx[a], idx[b]
        s = r['Prob_Local'] + r['Prob_Empate'] + r['Prob_Visitante']
        P1[i, j] = r['Prob_Local'] / s; PX[i, j] = r['Prob_Empate'] / s; P2[i, j] = r['Prob_Visitante'] / s
        XGL[i, j] = r['xG_Modelo_Local']; XGV[i, j] = r['xG_Modelo_Visitante']
    # P(A avanza vs B) incluyendo la tanda de penaltis si hay empate
    M_adv = P1 + PX * (P1 / np.where(P1 + P2 == 0, 1, P1 + P2))
    np.fill_diagonal(M_adv, 0.5)
    return {'P1': P1, 'PX': PX, 'P2': P2, 'XGL': XGL, 'XGV': XGV, 'M_adv': M_adv, 'idx': idx}


# ====================================================================
# 5. UTILIDADES DE PREDICCIÓN PUNTUAL (marcador más probable)
# ====================================================================

def _poisson_pmf(k, lam):
    lam = max(lam, 0.05)
    return math.exp(-lam) * lam ** k / math.factorial(k)

def marcador_mas_probable(xg_l, xg_v, resultado):
    """Marcador exacto más probable bajo Poissons independientes con medias xG,
    condicionado al resultado 1X2 predicho ('1', 'X' o '2')."""
    mejor, p_mejor = (1, 0), -1.0
    for i in range(9):
        for j in range(9):
            if resultado == '1' and not i > j: continue
            if resultado == 'X' and i != j: continue
            if resultado == '2' and not i < j: continue
            p = _poisson_pmf(i, xg_l) * _poisson_pmf(j, xg_v)
            if p > p_mejor:
                mejor, p_mejor = (i, j), p
    return mejor


# ====================================================================
# 6. CLASIFICACIÓN DE GRUPOS (misma lógica/orden que el notebook)
# ====================================================================

CRUCES_R32 = [
    ('1E', '3_1'), ('1I', '3_2'), ('2A', '2B'), ('1F', '2C'),
    ('2K', '2L'), ('1H', '2J'), ('1D', '3_3'), ('1G', '3_4'),
    ('1C', '2F'), ('2E', '2I'), ('1A', '3_5'), ('1L', '3_6'),
    ('1J', '2H'), ('2D', '2G'), ('1B', '3_7'), ('1K', '3_8')
]

def clasificar_grupos(df, tiradas):
    """Réplica del bloque de clasificación del notebook: Pts simulados/predichos,
    GF/GC con los xG del modelo, orden por Grupo, Pts, DG, GF."""
    df = df.copy()
    tir = np.asarray(tiradas)
    df['Pts_L'] = np.where(tir == 0, 3, np.where(tir == 1, 1, 0))
    df['Pts_V'] = np.where(tir == 2, 3, np.where(tir == 1, 1, 0))

    locales = df[['Grupo', 'Equipo_Local', 'Pts_L', 'xG_Modelo_Local', 'xG_Modelo_Visitante']].rename(
        columns={'Equipo_Local': 'Equipo', 'Pts_L': 'Pts', 'xG_Modelo_Local': 'GF', 'xG_Modelo_Visitante': 'GC'})
    visitantes = df[['Grupo', 'Equipo_Visitante', 'Pts_V', 'xG_Modelo_Visitante', 'xG_Modelo_Local']].rename(
        columns={'Equipo_Visitante': 'Equipo', 'Pts_V': 'Pts', 'xG_Modelo_Visitante': 'GF', 'xG_Modelo_Local': 'GC'})

    clasif = pd.concat([locales, visitantes])
    clasif['DG'] = clasif['GF'] - clasif['GC']
    clasif = clasif.groupby(['Grupo', 'Equipo']).sum().reset_index()
    clasif = clasif.sort_values(by=['Grupo', 'Pts', 'DG', 'GF'], ascending=[True, False, False, False])
    clasif['Posicion'] = clasif.groupby('Grupo').cumcount() + 1

    terceros = clasif[clasif['Posicion'] == 3].sort_values(by=['Pts', 'DG', 'GF'], ascending=False).head(8)
    pos = {f"{r['Posicion']}{r['Grupo']}": r['Equipo'] for _, r in clasif[clasif['Posicion'] <= 2].iterrows()}
    for i, eq in enumerate(terceros['Equipo']): pos[f"3_{i + 1}"] = eq
    return clasif, terceros, pos


# ====================================================================
# 7. MONTE CARLO VECTORIZADO (10.000 mundiales)
# ====================================================================

def monte_carlo(df_pred_grupos, mc, equipos, n_sims=N_SIMULACIONES):
    idx = mc['idx']
    n_eq = len(equipos)
    grupos_de = {}   # equipo -> grupo
    df = df_pred_grupos

    eq_grupo = pd.concat([df[['Grupo', 'Equipo_Local']].rename(columns={'Equipo_Local': 'Equipo'}),
                          df[['Grupo', 'Equipo_Visitante']].rename(columns={'Equipo_Visitante': 'Equipo'})]).drop_duplicates()
    for _, r in eq_grupo.iterrows(): grupos_de[r['Equipo']] = r['Grupo']
    letras = sorted(eq_grupo['Grupo'].unique())
    miembros = {g: sorted([e for e, gg in grupos_de.items() if gg == g]) for g in letras}

    loc = df['Equipo_Local'].map(idx).to_numpy()
    vis = df['Equipo_Visitante'].map(idx).to_numpy()
    P = df[['Prob_Local', 'Prob_Empate', 'Prob_Visitante']].to_numpy()
    P = P / P.sum(axis=1, keepdims=True)
    xg_l = df['xG_Modelo_Local'].to_numpy(); xg_v = df['xG_Modelo_Visitante'].to_numpy()

    # GF/GC por equipo son constantes (el notebook usa los xG del modelo como goles)
    gf = np.zeros(n_eq); gc = np.zeros(n_eq)
    np.add.at(gf, loc, xg_l); np.add.at(gf, vis, xg_v)
    np.add.at(gc, loc, xg_v); np.add.at(gc, vis, xg_l)
    dg = gf - gc
    # Claves de desempate exactas en enteros (centésimas)
    dg_c = np.round(dg * 100).astype(np.int64)
    gf_c = np.round(gf * 100).astype(np.int64)

    # Tiradas de toda la fase de grupos: (n_sims, 72)
    u = np.random.random((n_sims, len(df)))
    cum1 = P[:, 0]; cum2 = P[:, 0] + P[:, 1]
    out = np.where(u < cum1, 0, np.where(u < cum2, 1, 2)).astype(np.int8)

    pts = np.zeros((n_sims, n_eq), dtype=np.int64)
    for m in range(len(df)):
        o = out[:, m]
        pts[:, loc[m]] += np.where(o == 0, 3, np.where(o == 1, 1, 0))
        pts[:, vis[m]] += np.where(o == 2, 3, np.where(o == 1, 1, 0))

    # Clave compuesta (Pts, DG, GF) como entero para ordenar rápido
    clave = pts * 10_000_000_000 + (dg_c[None, :] + 100_000) * 10_000 + gf_c[None, :]

    pos_eq = {}   # etiqueta '1A'.. '2L' -> array (n_sims,) de ids de equipo
    terceros_ids = np.zeros((n_sims, 12), dtype=np.int64)
    terceros_clave = np.zeros((n_sims, 12), dtype=np.int64)
    for gi, g in enumerate(letras):
        ids = np.array([idx[e] for e in miembros[g]])
        k = clave[:, ids]                              # (n_sims, 4)
        orden = np.argsort(-k, axis=1, kind='stable')  # descendente
        ordenados = ids[orden]                         # (n_sims, 4)
        pos_eq[f'1{g}'] = ordenados[:, 0]
        pos_eq[f'2{g}'] = ordenados[:, 1]
        terceros_ids[:, gi] = ordenados[:, 2]
        terceros_clave[:, gi] = np.take_along_axis(k, orden[:, 2:3], axis=1)[:, 0]

    orden_terceros = np.argsort(-terceros_clave, axis=1, kind='stable')
    mejores8 = np.take_along_axis(terceros_ids, orden_terceros[:, :8], axis=1)
    for r in range(8):
        pos_eq[f'3_{r + 1}'] = mejores8[:, r]

    # --- Eliminatorias con la matriz de avance (incluye penaltis) ---
    M = mc['M_adv']
    contadores = {f: np.zeros(n_eq, dtype=np.int64) for f in
                  ['R32', 'Octavos', 'Cuartos', 'Semis', 'Final', 'Campeon']}

    clasificados = np.stack([pos_eq[f'1{g}'] for g in letras] +
                            [pos_eq[f'2{g}'] for g in letras] +
                            [mejores8[:, r] for r in range(8)], axis=1)
    for c in range(32):
        np.add.at(contadores['R32'], clasificados[:, c], 1)

    def jugar(a_ids, b_ids):
        p = M[a_ids, b_ids]
        gana_a = np.random.random(a_ids.shape) < p
        return np.where(gana_a, a_ids, b_ids)

    r32 = [(pos_eq[a], pos_eq[b]) for a, b in CRUCES_R32]
    g32 = [jugar(a, b) for a, b in r32]
    for w in g32: np.add.at(contadores['Octavos'], w, 1)
    g16 = [jugar(g32[i], g32[i + 1]) for i in range(0, 16, 2)]
    for w in g16: np.add.at(contadores['Cuartos'], w, 1)
    gqf = [jugar(g16[i], g16[i + 1]) for i in range(0, 8, 2)]
    for w in gqf: np.add.at(contadores['Semis'], w, 1)
    sf1 = jugar(gqf[0], gqf[1]); sf2 = jugar(gqf[2], gqf[3])
    for w in (sf1, sf2): np.add.at(contadores['Final'], w, 1)
    campeon = jugar(sf1, sf2)
    np.add.at(contadores['Campeon'], campeon, 1)

    tabla = pd.DataFrame({f: contadores[f] / n_sims * 100 for f in contadores}, index=equipos)
    tabla = tabla.sort_values(by=['Campeon', 'Final', 'Semis'], ascending=False).round(1)
    return tabla


# ====================================================================
# 8. PREDICCIÓN PUNTUAL DE TODO EL MUNDIAL (104 partidos)
# ====================================================================

FASES_FECHAS = {
    'Dieciseisavos': '28 jun - 3 jul', 'Octavos': '4 - 7 jul', 'Cuartos': '9 - 11 jul',
    'Semifinales': '14 - 15 jul', '3er Puesto': '18 jul', 'Final': '19 jul'
}

def prediccion_puntual(df_pred_grupos, mc, equipos, fechas_reales):
    idx = mc['idx']

    # --- Fase de grupos: resultado más probable de cada partido ---
    probs = df_pred_grupos[['Prob_Local', 'Prob_Empate', 'Prob_Visitante']].to_numpy()
    tiradas = probs.argmax(axis=1)

    filas = []
    mapa_fechas = {(r['Equipo_Local'], r['Equipo_Visitante']): r['Fecha'] for _, r in fechas_reales.iterrows()}
    for k, (_, r) in enumerate(df_pred_grupos.iterrows()):
        res = ['1', 'X', '2'][tiradas[k]]
        gl, gv = marcador_mas_probable(r['xG_Modelo_Local'], r['xG_Modelo_Visitante'], res)
        filas.append({
            'Fecha': mapa_fechas.get((r['Equipo_Local'], r['Equipo_Visitante']), ''),
            'Grupo': r['Grupo'], 'Local': r['Equipo_Local'], 'Visitante': r['Equipo_Visitante'],
            'Marcador_Predicho': f"{gl}-{gv}", 'Resultado_1X2': res,
            'xG_L': round(r['xG_Modelo_Local'], 2), 'xG_V': round(r['xG_Modelo_Visitante'], 2),
            'Prob_1': round(r['Prob_Local'] * 100, 1), 'Prob_X': round(r['Prob_Empate'] * 100, 1),
            'Prob_2': round(r['Prob_Visitante'] * 100, 1),
        })
    df_grupos_pred = pd.DataFrame(filas).sort_values(['Grupo', 'Fecha']).reset_index(drop=True)

    clasif, terceros, pos = clasificar_grupos(df_pred_grupos, tiradas)

    # --- Eliminatorias: avanza el más probable (con penaltis si el empate es lo más probable) ---
    P1, PX, P2, XGL, XGV = mc['P1'], mc['PX'], mc['P2'], mc['XGL'], mc['XGV']
    registro = []

    def jugar_cruce(fase, eq_a, eq_b):
        i, j = idx[eq_a], idx[eq_b]
        p1, px, p2 = P1[i, j], PX[i, j], P2[i, j]
        res = ['1', 'X', '2'][int(np.argmax([p1, px, p2]))]
        if res == 'X':
            ganador = eq_a if p1 >= p2 else eq_b
            gl, gv = marcador_mas_probable(XGL[i, j], XGV[i, j], 'X')
            detalle = f"Empate {gl}-{gv}; gana en penaltis"
            marcador = f"{gl}-{gv} (pen)"
        else:
            ganador = eq_a if res == '1' else eq_b
            gl, gv = marcador_mas_probable(XGL[i, j], XGV[i, j], res)
            detalle = "Tiempo regular"
            marcador = f"{gl}-{gv}"
        registro.append({
            'Fase': fase, 'Fechas': FASES_FECHAS[fase], 'Local': eq_a, 'Visitante': eq_b,
            'Marcador_Predicho': marcador, 'Avanza': ganador,
            'xG_L': round(XGL[i, j], 2), 'xG_V': round(XGV[i, j], 2),
            'Prob_1': round(p1 * 100, 1), 'Prob_X': round(px * 100, 1), 'Prob_2': round(p2 * 100, 1),
            'Detalle': detalle,
        })
        return ganador

    g32 = [jugar_cruce('Dieciseisavos', pos[a], pos[b]) for a, b in CRUCES_R32]
    g16 = [jugar_cruce('Octavos', g32[i], g32[i + 1]) for i in range(0, 16, 2)]
    gqf = [jugar_cruce('Cuartos', g16[i], g16[i + 1]) for i in range(0, 8, 2)]
    sf = [jugar_cruce('Semifinales', gqf[0], gqf[1]), jugar_cruce('Semifinales', gqf[2], gqf[3])]
    perdedores_sf = [e for par in [(gqf[0], gqf[1]), (gqf[2], gqf[3])] for e in par if e not in sf]
    jugar_cruce('3er Puesto', perdedores_sf[0], perdedores_sf[1])   # extensión: el notebook no lo incluía
    campeon = jugar_cruce('Final', sf[0], sf[1])

    df_elim = pd.DataFrame(registro)
    return df_grupos_pred, clasif, terceros, pos, df_elim, campeon


# ====================================================================
# MAIN
# ====================================================================

if __name__ == '__main__':
    if not (os.path.exists('modelo_goles_L.pkl') and os.path.exists('modelo_1X2_calibrado.pkl')
            and os.path.exists('columnas_entrenamiento.pkl')):
        entrenar_modelos()
    else:
        print("Modelos .pkl ya presentes: se reutilizan (borra los .pkl para reentrenar).")

    print("\nCalculando proyecciones de la fase de grupos (T=%.2f)..." % T_GRUPOS)
    df_mundial_grupos, df_vars, grupos, fechas_reales = cargar_mundial()
    df_predicciones_mundial = pipeline_prediccion(df_mundial_grupos, sede_neutral=True, T=T_GRUPOS)
    df_predicciones_mundial['Grupo'] = df_mundial_grupos['Grupo'].values

    equipos = sorted(grupos['Equipo'].unique())
    print("Calculando matriz de cruces 48x48 (T=%.2f)..." % T_CRUCES)
    mc = matriz_cruces(equipos, df_vars)

    print("Generando predicción puntual de los 104 partidos...")
    df_grupos_pred, clasif, terceros, pos, df_elim, campeon = prediccion_puntual(
        df_predicciones_mundial, mc, equipos, fechas_reales)

    print(f"Simulando {N_SIMULACIONES} mundiales (Monte Carlo)...")
    tabla_mc = monte_carlo(df_predicciones_mundial, mc, equipos, N_SIMULACIONES)

    # --- Guardado ---
    df_grupos_pred.to_csv('Predicciones/predicciones_fase_grupos.csv', index=False, encoding='utf-8-sig')
    clasif[['Grupo', 'Equipo', 'Pts', 'GF', 'GC', 'DG', 'Posicion']].to_csv(
        'Predicciones/clasificacion_grupos.csv', index=False, encoding='utf-8-sig')
    df_elim.to_csv('Predicciones/predicciones_eliminatorias.csv', index=False, encoding='utf-8-sig')
    tabla_mc.to_csv('Predicciones/probabilidades_montecarlo.csv', encoding='utf-8-sig')

    print("\n=== CAMPEÓN PREDICHO:", campeon, "===")
    print("\nTop 10 probabilidades de campeón (Monte Carlo, %d sims):" % N_SIMULACIONES)
    print(tabla_mc.head(10).to_string())
    print("\nArchivos escritos en Predicciones/.")
