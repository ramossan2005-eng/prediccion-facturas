"""
Predicción de facturas que se pagarán tarde (machine learning con Python).

Pasos:
  1. Leer y limpiar el historial de facturas
  2. Crear variables útiles (incluido el historial de pago de cada cliente,
     usando SOLO información disponible en la fecha de cada factura)
  3. Entrenar y comparar dos modelos con una validación por fechas
  4. Calificar las facturas abiertas según su riesgo de atraso
  5. Exportar gráficas y un reporte en Excel listo para el área de cartera

Uso:  python3 prediccion_facturas.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Genera las gráficas sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as ImagenExcel
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ARCHIVO = Path("datos/facturas.csv")
CARPETA_FIG = Path("figuras")
REPORTE = Path("reporte_cartera.xlsx")
FECHA_CORTE = pd.Timestamp("2026-09-30")   # "Hoy" para la empresa
INICIO_PRUEBA = pd.Timestamp("2026-01-01")  # Facturas desde aquí = datos de prueba
UMBRAL = 0.5                                # Probabilidad desde la cual se alerta

AZUL, NARANJA, GRIS = "#0F4C5C", "#C2410C", "#8A9499"

NUMERICAS = ["plazo_dias", "monto_millones", "monto_vs_cupo", "antiguedad_anios",
             "cupo_millones", "temporada_alta", "facturas_previas",
             "tasa_atraso_previa", "dias_atraso_prom_previo"]
CATEGORICAS = ["sector", "ciudad"]
NOMBRES = {  # Nombres claros para el reporte
    "plazo_dias": "Plazo de pago (días)", "monto_millones": "Monto de la factura",
    "monto_vs_cupo": "Monto frente al cupo de crédito",
    "antiguedad_anios": "Antigüedad del cliente", "cupo_millones": "Cupo de crédito",
    "temporada_alta": "Emitida en diciembre o enero",
    "facturas_previas": "Número de facturas previas",
    "tasa_atraso_previa": "% de atrasos previos del cliente",
    "dias_atraso_prom_previo": "Días de atraso promedio previos",
    "sector": "Sector del cliente", "ciudad": "Ciudad",
}


# ------------------------------------------------------------- 1. Limpiar
def convertir_monto(valor):
    if isinstance(valor, str):
        valor = valor.replace("$", "").replace(",", "").strip()
    return float(valor)


def cargar_y_limpiar(archivo):
    df = pd.read_csv(archivo)
    filas = len(df)
    df = df.drop_duplicates(subset="factura_id")
    duplicadas = filas - len(df)

    for col in ["fecha_emision", "fecha_vencimiento", "fecha_pago"]:
        df[col] = pd.to_datetime(df[col])
    df["monto"] = df["monto"].map(convertir_monto)

    # El plazo faltante se puede reconstruir: vencimiento - emisión
    faltantes = df["plazo_dias"].isna().sum()
    df["plazo_dias"] = (df["fecha_vencimiento"] - df["fecha_emision"]).dt.days

    print(f"  {filas:,} filas leídas · {duplicadas} duplicadas eliminadas · "
          f"{faltantes} plazos reconstruidos")
    return df.sort_values("fecha_emision").reset_index(drop=True)


# ---------------------------------------------------- 2. Crear variables
def crear_objetivo(df):
    """Define si cada factura se pagó tarde (1), a tiempo (0) o aún no se sabe (NaN)."""
    pagada = df["fecha_pago"].notna()
    vencida = df["fecha_vencimiento"] < FECHA_CORTE
    df["dias_atraso"] = (df["fecha_pago"] - df["fecha_vencimiento"]).dt.days.clip(lower=0)
    df["atrasada"] = np.where(pagada, (df["dias_atraso"] > 0).astype(float),
                              np.where(vencida, 1.0, np.nan))  # Vencida sin pagar = atraso
    # Para facturas vencidas sin pagar, el atraso cuenta hasta hoy
    abierta_vencida = ~pagada & vencida
    df.loc[abierta_vencida, "dias_atraso"] = (FECHA_CORTE - df.loc[abierta_vencida,
                                              "fecha_vencimiento"]).dt.days
    return df


def historial_cliente(df):
    """
    Para cada factura, resume cómo había pagado el cliente ANTES de emitirla.
    Solo cuenta facturas que ya habían vencido en esa fecha, para no usar
    información del futuro (eso inflaría artificialmente la precisión).
    """
    previas, tasa, dias_prom = [], [], []
    for _, grupo in df.groupby("cliente_id", sort=False):
        venc = grupo["fecha_vencimiento"].to_numpy()
        atraso = grupo["atrasada"].fillna(0).to_numpy()
        dias = grupo["dias_atraso"].fillna(0).to_numpy()
        for idx, emision in zip(grupo.index, grupo["fecha_emision"].to_numpy()):
            conocidas = venc < emision
            n = conocidas.sum()
            previas.append((idx, n,
                            atraso[conocidas].mean() if n else np.nan,
                            dias[conocidas].mean() if n else np.nan))
    hist = pd.DataFrame(previas, columns=["idx", "facturas_previas",
                                          "tasa_atraso_previa",
                                          "dias_atraso_prom_previo"]).set_index("idx")
    df = df.join(hist)
    # Cliente sin historial: se usa el promedio general como punto de partida
    df["tasa_atraso_previa"] = df["tasa_atraso_previa"].fillna(df["atrasada"].mean())
    df["dias_atraso_prom_previo"] = df["dias_atraso_prom_previo"].fillna(0)
    return df


def crear_variables(df):
    df = historial_cliente(crear_objetivo(df))
    df["monto_millones"] = df["monto"] / 1e6
    df["cupo_millones"] = df["cupo_credito"] / 1e6
    df["monto_vs_cupo"] = df["monto"] / df["cupo_credito"]
    df["temporada_alta"] = df["fecha_emision"].dt.month.isin([12, 1]).astype(int)
    return df


# --------------------------------------------------- 3. Entrenar modelos
def crear_modelos():
    preparar = ColumnTransformer([
        ("num", StandardScaler(), NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
    ])
    return {
        "Regresión logística": Pipeline([
            ("prep", preparar),
            ("modelo", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "Random forest": Pipeline([
            ("prep", preparar),
            ("modelo", RandomForestClassifier(n_estimators=400, min_samples_leaf=10,
                                              class_weight="balanced", random_state=7,
                                              n_jobs=-1)),
        ]),
    }


def evaluar(y, prob):
    pred = (prob >= UMBRAL).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {"AUC": roc_auc_score(y, prob),
            "Precisión (alertas correctas)": precision_score(y, pred),
            "Recall (atrasos detectados)": recall_score(y, pred),
            "Verdaderos positivos": tp, "Falsos positivos": fp,
            "Falsos negativos": fn, "Verdaderos negativos": tn}


def entrenar(df):
    conocidas = df[df["atrasada"].notna()]
    train = conocidas[conocidas["fecha_emision"] < INICIO_PRUEBA]
    test = conocidas[conocidas["fecha_emision"] >= INICIO_PRUEBA]
    X_cols = NUMERICAS + CATEGORICAS
    print(f"  Entrenamiento: {len(train):,} facturas (2024–2025) · "
          f"Prueba: {len(test):,} facturas (2026)")

    resultados, probs = {}, {}
    for nombre, modelo in crear_modelos().items():
        modelo.fit(train[X_cols], train["atrasada"])
        probs[nombre] = modelo.predict_proba(test[X_cols])[:, 1]
        resultados[nombre] = evaluar(test["atrasada"], probs[nombre])
        print(f"  {nombre}: AUC = {resultados[nombre]['AUC']:.3f}")

    mejor = max(resultados, key=lambda n: resultados[n]["AUC"])
    modelos = crear_modelos()
    modelo_final = modelos[mejor].fit(train[X_cols], train["atrasada"])

    imp = permutation_importance(modelo_final, test[X_cols], test["atrasada"],
                                 scoring="roc_auc", n_repeats=10, random_state=7)
    importancia = (pd.DataFrame({"variable": X_cols, "importancia": imp.importances_mean})
                     .assign(variable=lambda d: d["variable"].map(NOMBRES))
                     .sort_values("importancia", ascending=False))

    # Modelo final reentrenado con TODO el historial conocido para calificar lo abierto
    modelo_produccion = crear_modelos()[mejor].fit(conocidas[X_cols], conocidas["atrasada"])
    return {"mejor": mejor, "resultados": resultados, "probs": probs, "test": test,
            "importancia": importancia, "modelo": modelo_produccion, "train": train}


# ------------------------------------------- 4. Calificar facturas abiertas
def calificar_abiertas(df, modelo):
    abiertas = df[df["atrasada"].isna()].copy()
    abiertas["prob_atraso"] = modelo.predict_proba(abiertas[NUMERICAS + CATEGORICAS])[:, 1]
    abiertas["riesgo"] = pd.cut(abiertas["prob_atraso"], [0, 0.35, 0.6, 1],
                                labels=["Bajo", "Medio", "Alto"], include_lowest=True)
    return abiertas.sort_values("prob_atraso", ascending=False)


# ----------------------------------------------------------- 5. Gráficas
def estilo(ax, titulo):
    ax.set_title(titulo, loc="left", fontsize=13, fontweight="bold", color="#14232B")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.spines["left"].set_color("#C9CED1")
    ax.spines["bottom"].set_color("#C9CED1")
    ax.tick_params(colors="#3A4A52")


def barras_tasa(ax, serie, titulo, promedio):
    colores = [NARANJA if v > promedio else AZUL for v in serie.values]
    ax.bar(serie.index.astype(str), serie.values * 100, color=colores, width=0.65)
    ax.axhline(promedio * 100, color=GRIS, ls="--", lw=1)
    ax.text(len(serie) - 0.5, promedio * 100, f" promedio {promedio:.0%}",
            va="bottom", ha="right", color=GRIS, fontsize=9)
    for i, v in enumerate(serie.values):
        ax.text(i, v * 100 + 1, f"{v:.0%}", ha="center", fontsize=9, color="#14232B")
    ax.set_ylabel("% de facturas pagadas tarde")
    estilo(ax, titulo)


def graficar(df, r):
    CARPETA_FIG.mkdir(exist_ok=True)
    conocidas = df[df["atrasada"].notna()]
    prom = conocidas["atrasada"].mean()
    rutas = {}

    # 1. Atraso por sector
    fig, ax = plt.subplots(figsize=(9, 4.8))
    barras_tasa(ax, conocidas.groupby("sector")["atrasada"].mean().sort_values(),
                "Construcción y restaurantes son los sectores que más se atrasan", prom)
    plt.xticks(rotation=20, ha="right")
    rutas["sector"] = guardar(fig, "1_atraso_por_sector.png")

    # 2. Atraso según historial del cliente
    tramos = pd.cut(conocidas["tasa_atraso_previa"], [-0.01, 0.2, 0.4, 0.6, 1.0],
                    labels=["0–20%", "20–40%", "40–60%", "60–100%"])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    barras_tasa(ax, conocidas.groupby(tramos, observed=True)["atrasada"].mean(),
                "Quien se atrasó antes, se vuelve a atrasar", prom)
    ax.set_xlabel("% de atrasos previos del cliente")
    rutas["historial"] = guardar(fig, "2_atraso_segun_historial.png")

    # 3. Curva ROC de ambos modelos
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for nombre, color in zip(r["probs"], [GRIS, AZUL]):
        fpr, tpr, _ = roc_curve(r["test"]["atrasada"], r["probs"][nombre])
        auc = r["resultados"][nombre]["AUC"]
        ax.plot(fpr, tpr, color=color, lw=2.2, label=f"{nombre} (AUC {auc:.2f})")
    ax.plot([0, 1], [0, 1], color="#C9CED1", ls="--", lw=1, label="Al azar (AUC 0.50)")
    ax.set_xlabel("Falsas alarmas (tasa de falsos positivos)")
    ax.set_ylabel("Atrasos detectados (tasa de verdaderos positivos)")
    ax.legend(frameon=False, loc="lower right")
    estilo(ax, "Desempeño en facturas de 2026 (datos no vistos)")
    rutas["roc"] = guardar(fig, "3_curva_roc.png")

    # 4. Importancia de variables
    imp = r["importancia"].head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(imp["variable"], imp["importancia"], color=AZUL, height=0.6)
    ax.set_xlabel("Caída del AUC al desordenar la variable")
    estilo(ax, "Qué pesa más para predecir un atraso")
    rutas["importancia"] = guardar(fig, "4_importancia_variables.png")
    return rutas


def guardar(fig, nombre):
    ruta = CARPETA_FIG / nombre
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta


# ------------------------------------------------------- 6. Reporte Excel
def exportar(df, r, abiertas, rutas):
    conocidas = df[df["atrasada"].notna()]
    m = r["resultados"][r["mejor"]]
    alto = abiertas[abiertas["riesgo"] == "Alto"]
    resumen = pd.DataFrame({
        "Indicador": [
            "Fecha de corte", "Facturas analizadas", "% histórico de facturas pagadas tarde",
            "Modelo elegido", "AUC en datos de prueba (2026)",
            f"De cada 10 alertas, cuántas son atrasos reales (umbral {UMBRAL:.0%})",
            "% de atrasos reales que el modelo detecta",
            "Facturas abiertas por vencer", "Facturas abiertas en riesgo ALTO",
            "Monto en riesgo ALTO (COP)",
        ],
        "Valor": [
            FECHA_CORTE.date(), len(df), conocidas["atrasada"].mean(), r["mejor"],
            round(m["AUC"], 3), round(m["Precisión (alertas correctas)"] * 10, 1),
            m["Recall (atrasos detectados)"], len(abiertas), len(alto), alto["monto"].sum(),
        ],
    })
    metricas = pd.DataFrame(r["resultados"]).T.reset_index(names="Modelo")
    riesgo = abiertas[["factura_id", "cliente_id", "sector", "ciudad", "fecha_emision",
                       "fecha_vencimiento", "monto", "tasa_atraso_previa",
                       "prob_atraso", "riesgo"]].copy()
    riesgo["fecha_emision"] = riesgo["fecha_emision"].dt.date
    riesgo["fecha_vencimiento"] = riesgo["fecha_vencimiento"].dt.date

    with pd.ExcelWriter(REPORTE, engine="openpyxl") as w:
        resumen.to_excel(w, sheet_name="Resumen", index=False)
        riesgo.to_excel(w, sheet_name="Facturas en riesgo", index=False)
        r["importancia"].to_excel(w, sheet_name="Importancia", index=False)
        metricas.to_excel(w, sheet_name="Comparación modelos", index=False)
    dar_formato(rutas)


def dar_formato(rutas):
    wb = load_workbook(REPORTE)
    relleno = PatternFill("solid", fgColor="0F4C5C")
    colores_riesgo = {"Alto": "FDE2D6", "Medio": "FFF4CC", "Bajo": "E3F1EC"}
    for ws in wb.worksheets:
        for c in ws[1]:
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = relleno
            c.alignment = Alignment(horizontal="center", wrap_text=True)
            c.value = str(c.value).replace("_", " ").capitalize()
        for i, col in enumerate(ws.columns, start=1):
            largo = max(len(str(c.value or "")) for c in col)
            ws.column_dimensions[get_column_letter(i)].width = min(largo + 3, 55)
        ws.freeze_panes = "A2"

    ws = wb["Facturas en riesgo"]
    encabezados = [c.value for c in ws[1]]
    col = {n: encabezados.index(n) + 1 for n in encabezados}
    ws.auto_filter.ref = ws.dimensions
    for fila in ws.iter_rows(min_row=2):
        fila[col["Monto"] - 1].number_format = '"$"#,##0'
        fila[col["Tasa atraso previa"] - 1].number_format = "0%"
        fila[col["Prob atraso"] - 1].number_format = "0%"
        nivel = fila[col["Riesgo"] - 1].value
        fila[col["Riesgo"] - 1].fill = PatternFill("solid", fgColor=colores_riesgo[nivel])

    ws = wb["Resumen"]
    ws["B4"].number_format = "0%"
    ws["B8"].number_format = "0%"
    ws["B11"].number_format = '"$"#,##0'
    ws.column_dimensions["A"].width = 62
    ws.column_dimensions["B"].width = 22
    for c in ws["B"]:
        c.alignment = Alignment(horizontal="left")

    graficas = wb.create_sheet("Gráficas", 1)
    for i, clave in enumerate(["sector", "historial", "importancia", "roc"]):
        img = ImagenExcel(str(rutas[clave]))
        img.width, img.height = img.width * 0.45, img.height * 0.45
        graficas.add_image(img, f"{'A' if i % 2 == 0 else 'K'}{1 + (i // 2) * 24}")
    wb.save(REPORTE)


# ------------------------------------------------------------------ Main
def main():
    print("1. Limpiando datos...")
    df = cargar_y_limpiar(ARCHIVO)
    print("2. Creando variables...")
    df = crear_variables(df)
    print("3. Entrenando modelos...")
    r = entrenar(df)
    print(f"   Modelo elegido: {r['mejor']}")
    print("4. Calificando facturas abiertas...")
    abiertas = calificar_abiertas(df, r["modelo"])
    print(f"   {len(abiertas)} abiertas · {(abiertas['riesgo'] == 'Alto').sum()} en riesgo alto")
    print("5. Generando gráficas y reporte...")
    rutas = graficar(df, r)
    exportar(df, r, abiertas, rutas)
    print(f"\nListo: {REPORTE} y carpeta {CARPETA_FIG}/")


if __name__ == "__main__":
    main()
