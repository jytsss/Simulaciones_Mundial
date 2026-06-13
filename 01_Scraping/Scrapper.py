from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time
import random
import pandas as pd
import re
import os

# ==========================================
# FUNCIONES DE EXTRACCIÓN DE PARTIDOS
# ==========================================
def extraer_partido_completo(url, driver):
    datos_partido = {
        'fecha': 'Desconocida',
        'url': url,
        'equipo_local': 'Desconocido',
        'equipo_visitante': 'Desconocido',
        'resultado': 'Desconocido',
        'estadisticas': {}
    }

    try:
        print(f"Cargando partido: {url}")
        driver.get(url)

        # Espera para que el JavaScript renderice la página
        time.sleep(5) 
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # 1. EXTRAER FECHA
        fecha_element = soup.find(class_=lambda x: x and isinstance(x, str) and 'startTime' in x)
        if fecha_element:
            datos_partido['fecha'] = fecha_element.text.strip()
            
        # 2. EXTRAER EQUIPOS
        enlaces_equipos = soup.find_all('a', class_=lambda x: x and isinstance(x, str) and 'participantName' in x)
        
        equipos_unicos = []
        for enlace in enlaces_equipos:
            nombre = enlace.text.strip()
            if nombre and nombre not in equipos_unicos:
                equipos_unicos.append(nombre)
                
        if len(equipos_unicos) >= 2:
            datos_partido['equipo_local'] = equipos_unicos[0]
            datos_partido['equipo_visitante'] = equipos_unicos[1]
            
        # 3. EXTRAER RESULTADO
        marcador = soup.find(class_=lambda x: x and isinstance(x, str) and 'detailScore' in x)
        if marcador:
            datos_partido['resultado'] = marcador.text.replace('\n', '').strip()

        # 4. EXTRAER ESTADÍSTICAS
        todos_los_divs = soup.find_all('div')
        for div in todos_los_divs:
            hijos = div.find_all('div', recursive=False)
            if len(hijos) == 3:
                valor_local = hijos[0].text.strip()
                categoria = hijos[1].text.strip()
                valor_visitante = hijos[2].text.strip()
                
                if (valor_local and categoria and valor_visitante and 
                    any(c.isalpha() for c in categoria) and 
                    len(categoria) < 30):
                    
                    datos_partido['estadisticas'][categoria] = {
                        'local': valor_local,
                        'visitante': valor_visitante
                    }

    except Exception as e:
        print(f"Ocurrió un error en la URL {url}: {e}")

    return datos_partido

def aplanar_datos(datos_partido):
    """
    Convierte el diccionario anidado en un diccionario plano de un solo nivel 
    para que Pandas pueda crear las columnas correctamente.
    """
    datos_planos = {
        'Fecha': datos_partido['fecha'],
        'URL': datos_partido['url'],
        'Equipo_Local': datos_partido['equipo_local'],
        'Equipo_Visitante': datos_partido['equipo_visitante'],
        'Resultado': datos_partido['resultado']
    }
    
    for categoria, valores in datos_partido['estadisticas'].items():
        nombre_columna = categoria.replace(" ", "_")
        datos_planos[f"{nombre_columna}_Local"] = valores['local']
        datos_planos[f"{nombre_columna}_Visitante"] = valores['visitante']
        
    return datos_planos

# ==========================================
# FUNCIONES DE EXTRACCIÓN DE RANKINGS
# ==========================================
def extraer_ranking_con_fecha(url, fecha_asignada, driver):
    """
    Navega a la URL, extrae la tabla y le asigna la fecha parametrizada.
    """
    print(f"Cargando URL para el ranking del: {fecha_asignada}")
    driver.get(url)
    time.sleep(4) 
    
    # CLICK AUTOMÁTICO EN EL BOTÓN
    try:
        boton_expandir = driver.find_element(By.XPATH, "//*[contains(text(), 'Mostrar la Clasificación Mundial completa')]")
        driver.execute_script("arguments[0].click();", boton_expandir)
        print("  Botón 'Mostrar completa' pulsado.")
        time.sleep(5) 
    except Exception:
        print("  El botón no apareció (tabla expandida por defecto).")

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    # EXTRACCIÓN DE PAÍS Y PUNTUACIÓN
    filas = soup.find_all('tr')
    datos_ranking = []
    
    for fila in filas:
        columnas = fila.find_all('td')
        if len(columnas) >= 5:
            # País (Columna 2)
            equipo = None
            textos_col_equipo = list(columnas[1].stripped_strings)
            if textos_col_equipo:
                equipo = textos_col_equipo[-1]
            
            # Puntuación (De atrás hacia adelante)
            puntos = None
            for col in reversed(columnas):
                txt = col.text.strip().replace(',', '').replace('*', '') 
                if re.match(r'^\d{3,4}\.\d{2}$', txt):
                    puntos = float(txt)
                    break 
            
            if equipo and puntos:
                datos_ranking.append({
                    'País': equipo,
                    'Puntuación': puntos,
                    'Fecha': fecha_asignada 
                })
                
    df = pd.DataFrame(datos_ranking).drop_duplicates(subset=['País']).reset_index(drop=True)
    return df


# ==========================================
# BLOQUE PRINCIPAL DE EJECUCIÓN
# ==========================================
if __name__ == "__main__":

    # Cambia estas rutas como prefieras. 
    # Si la carpeta no existe, el código la creará por ti.

    # 1. El script detecta dónde está la carpeta principal del proyecto
    # (el script vive en 01_Scraping/, así que la raíz es la carpeta padre)
    carpeta_raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 2. Entra directamente a 'Data' sin importar el sistema operativo
    RUTA_URLS = os.path.join(carpeta_raiz, "Data", "urls.txt")
    RUTA_CSV_PARTIDOS = os.path.join(carpeta_raiz, "Data", "partidos.csv")
    RUTA_CSV_RANKINGS = os.path.join(carpeta_raiz, "Data", "ranking_fifa.csv")
    
    # 1. CONEXIÓN AL NAVEGADOR
    print("Iniciando pipeline de extracción de datos...")
    print("Conectando al Google Chrome (Puerto 9222)...")
    
    opciones = Options()
    opciones.add_experimental_option("debuggerAddress", "127.0.0.1:9222")

    # Usamos la versión específica para evitar problemas de compatibilidad
    ruta_driver = ChromeDriverManager(driver_version="148.0.7778.216").install()
    servicio = Service(ruta_driver)
    
    try:
        driver = webdriver.Chrome(service=servicio, options=opciones) 
    except Exception as e:
        print(f"Error de conexión: {e}")
        exit()

    # ---------------------------------------------------------
    # FASE 1: EXTRACCIÓN DE PARTIDOS (Flashscore)
    # ---------------------------------------------------------
    try:
        with open(RUTA_URLS, 'r', encoding='utf-8') as f:
            lista_urls_partidos = [linea.strip() for linea in f if linea.strip()]
        print(f"¡Éxito! Se han cargado {len(lista_urls_partidos)} URLs de partidos.")
    except FileNotFoundError:
        print("ERROR: El archivo no existe en esa ruta de arriba.")
        lista_urls_partidos = []

    print("\n" + "="*45)
    print("FASE 1: EXTRACCIÓN DE PARTIDOS")
    print("="*45)
    
    # Leer archivo de URLs de partidos
    try:
        with open(RUTA_URLS, 'r', encoding='utf-8') as f:
            lista_urls_partidos = [linea.strip() for linea in f if linea.strip()]
            print(f"Se han cargado {len(lista_urls_partidos)} URLs.")
    except FileNotFoundError:
        print("Por favor, asegúrate de que el archivo 'urls.txt' está en la misma carpeta que este script.")
        lista_urls_partidos = []

    todos_los_partidos = []

    for idx, url in enumerate(lista_urls_partidos):
        print(f"\n--- Procesando partido {idx + 1} de {len(lista_urls_partidos)} ---")
        datos_crudos = extraer_partido_completo(url, driver)
        
        if datos_crudos and datos_crudos['equipo_local'] != 'Desconocido':
            datos_planos = aplanar_datos(datos_crudos)
            todos_los_partidos.append(datos_planos)
        else:
            print(f"Fallo en la extracción de la URL: {url}")
            
        if idx < len(lista_urls_partidos) - 1:
            pausa = random.uniform(3, 6)
            print(f"Pausa de seguridad de {pausa:.2f} segundos...")
            time.sleep(pausa)

    if todos_los_partidos:
        df_partidos = pd.DataFrame(todos_los_partidos)
        os.makedirs(os.path.dirname(RUTA_CSV_PARTIDOS), exist_ok=True)
        df_partidos.to_csv(RUTA_CSV_PARTIDOS, index=False, encoding='utf-8-sig')
        print(f"\nExtracción de partidos finalizada. Datos guardados en '{RUTA_CSV_PARTIDOS}'")
    elif lista_urls_partidos:
        print("\nNo se pudieron extraer datos de ninguna URL de partidos.")

    # ---------------------------------------------------------
    # FASE 2: EXTRACCIÓN DE RANKINGS (FIFA)
    # ---------------------------------------------------------
    print("\n" + "="*45)
    print("FASE 2: EXTRACCIÓN DE RANKINGS HISTÓRICOS")
    print("="*45)
    
    lista_urls_rankings = [
        "https://inside.fifa.com/es/fifa-world-ranking/men?dateId=FRS_Male_Football_20260119",
        "https://inside.fifa.com/es/fifa-world-ranking/men?dateId=FRS_Male_Football_20251219",
        "https://inside.fifa.com/es/fifa-world-ranking/men?dateId=FRS_Male_Football_20251119",
        "https://inside.fifa.com/es/fifa-world-ranking/men?dateId=FRS_Male_Football_20251015",
        "https://inside.fifa.com/es/fifa-world-ranking/men?dateId=FRS_Male_Football_20250910",
        "https://inside.fifa.com/es/fifa-world-ranking/men?dateId=id14870",
        "https://inside.fifa.com/es/fifa-world-ranking/men?dateId=id14800",
        "https://inside.fifa.com/es/fifa-world-ranking/men?dateId=id14702",
        "https://inside.fifa.com/es/fifa-world-ranking/men?dateId=id14597",
        "https://inside.fifa.com/es/fifa-world-ranking/men?dateId=id14576",
        "https://inside.fifa.com/es/fifa-world-ranking/men?dateId=id14541",
        "https://inside.fifa.com/es/fifa-world-ranking/men?dateId=id14506",
        "https://inside.fifa.com/es/fifa-world-ranking/men?dateId=id14443",
        "https://inside.fifa.com/es/fifa-world-ranking/men?dateId=id14415",
        "https://inside.fifa.com/es/fifa-world-ranking/men?dateId=id14338",
        "https://inside.fifa.com/es/fifa-world-ranking/men?dateId=id14289",
        "https://inside.fifa.com/es/fifa-world-ranking/men?dateId=id14233",
        "https://inside.fifa.com/es/fifa-world-ranking/men?dateId=id14212",
        "https://inside.fifa.com/es/fifa-world-ranking/men?dateId=id14177",
        "https://inside.fifa.com/es/fifa-world-ranking/men?dateId=id14142",
        "https://inside.fifa.com/es/fifa-world-ranking/men?dateId=id14079",
        "https://inside.fifa.com/es/fifa-world-ranking/men?dateId=id14058",
        "https://inside.fifa.com/es/fifa-world-ranking/men?dateId=id13974",
        "https://inside.fifa.com/es/fifa-world-ranking/men?dateId=id13869",
        "https://inside.fifa.com/es/fifa-world-ranking/men?dateId=id13792",
        "https://inside.fifa.com/es/fifa-world-ranking/men?dateId=id13750",
        "https://inside.fifa.com/es/fifa-world-ranking/men?dateId=id13687",
        "https://inside.fifa.com/es/fifa-world-ranking/men?dateId=id13603",
        "https://inside.fifa.com/es/fifa-world-ranking/men?dateId=id13554",
        "https://inside.fifa.com/es/fifa-world-ranking/men?dateId=id13505",
        "https://inside.fifa.com/es/fifa-world-ranking/men?dateId=id13197",
        "https://inside.fifa.com/es/fifa-world-ranking/men?dateId=id13127",
        "https://inside.fifa.com/es/fifa-world-ranking/men?dateId=id12770",
        "https://inside.fifa.com/es/fifa-world-ranking/men?dateId=id12406"
    ]
    
    lista_fechas_rankings = [
        "01-04-2026", "19-01-2026", "22-12-2025", "19-11-2025", "17-10-2025",
        "18-09-2025", "10-07-2025", "03-04-2025", 
        "19-12-2024", "28-11-2024", "24-10-2024", "19-09-2024", "18-07-2024", "20-06-2024", "04-04-2024", "15-02-2024",
        "21-12-2023", "30-11-2023", "26-10-2023", "21-09-2023", "20-07-2023", "29-06-2023", "06-04-2023",
        "22-12-2022", "06-10-2022", "25-08-2022", "23-06-2022", "31-03-2022", "10-02-2022",
        "23-12-2021", "18-02-2021", "10-12-2020", "19-12-2019", "20-12-2018"
    ]

    lista_dataframes_rankings = []

    for idx, (url, fecha) in enumerate(zip(lista_urls_rankings, lista_fechas_rankings)):
        print(f"\n--- Procesando enlace ranking {idx + 1} de {len(lista_urls_rankings)} ---")
        
        df_temporal = extraer_ranking_con_fecha(url, fecha, driver)
        
        if df_temporal is not None and not df_temporal.empty:
            print(f"  Registros añadidos: {len(df_temporal)}")
            lista_dataframes_rankings.append(df_temporal)
        else:
            print(f"  Fallo crítico o tabla vacía en la URL: {url}")
            
        if idx < len(lista_urls_rankings) - 1:
            pausa = random.uniform(3, 5)
            print(f"  Pausa de seguridad de {pausa:.2f} segundos...")
            time.sleep(pausa)

    if lista_dataframes_rankings:
        df_final_rankings = pd.concat(lista_dataframes_rankings, ignore_index=True)
        os.makedirs(os.path.dirname(RUTA_CSV_RANKINGS), exist_ok=True)
        df_final_rankings.to_csv(RUTA_CSV_RANKINGS, index=False, encoding='utf-8-sig')
        print(f"\nExtracción histórica de rankings finalizada. Total de registros: {len(df_final_rankings)}")
        print(f"Archivo consolidado guardado en: '{RUTA_CSV_RANKINGS}'")
    else:
        print("\nNo se extrajeron datos de los rankings.")

    print("\n" + "="*45)
    print("PROCESO COMPLETO FINALIZADO")
    print("="*45)