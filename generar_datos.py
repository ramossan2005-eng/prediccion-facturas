"""
Genera un historial de facturas de ejemplo (datos SINTÉTICOS) para practicar.

Simula una empresa que vende a crédito a 250 clientes entre 2024 y 2026.
Algunos clientes pagan siempre a tiempo y otros suelen atrasarse; el retraso
también depende del plazo, del monto y de la época del año, como en la vida real.
Incluye errores típicos de un archivo real: filas duplicadas, datos faltantes
y montos guardados como texto.

Uso:  python3 generar_datos.py
Crea datos/facturas.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)  # Semilla fija: siempre salen los mismos datos

N_CLIENTES = 250
INICIO = pd.Timestamp("2024-01-01")
FIN = pd.Timestamp("2026-09-30")       # Último día con facturas emitidas
FECHA_CORTE = pd.Timestamp("2026-09-30")  # "Hoy" para la empresa

SECTORES = {  # sector: efecto en el riesgo de atraso (positivo = más riesgo)
    "Retail": 0.3, "Construcción": 0.8, "Tecnología": -0.4,
    "Salud": -0.2, "Manufactura": 0.1, "Restaurantes": 0.6, "Educación": -0.1,
}
CIUDADES = ["Bogotá", "Medellín", "Cali", "Barranquilla", "Bucaramanga", "Cartagena"]


def crear_clientes():
    sectores = rng.choice(list(SECTORES), N_CLIENTES)
    return pd.DataFrame({
        "cliente_id": [f"C{i:04d}" for i in range(1, N_CLIENTES + 1)],
        "sector": sectores,
        "ciudad": rng.choice(CIUDADES, N_CLIENTES),
        "antiguedad_anios": rng.integers(0, 15, N_CLIENTES),
        "cupo_credito": rng.choice([5, 10, 20, 50, 100], N_CLIENTES) * 1_000_000,
        # Rasgo oculto: qué tan cumplido es el cliente (el modelo NO lo ve)
        "_cumplimiento": rng.normal(0, 1, N_CLIENTES),
    })


def crear_facturas(clientes):
    filas = []
    dias_totales = (FIN - INICIO).days
    for _, c in clientes.iterrows():
        n = rng.poisson(28)
        for _ in range(n):
            emision = INICIO + pd.Timedelta(days=int(rng.integers(0, dias_totales)))
            plazo = int(rng.choice([15, 30, 45, 60], p=[0.2, 0.45, 0.2, 0.15]))
            monto = round(float(rng.lognormal(np.log(c.cupo_credito * 0.12), 0.7)), -3)
            vencimiento = emision + pd.Timedelta(days=plazo)

            # Probabilidad de atraso según los factores del negocio
            z = (-1.4
                 - 0.9 * c._cumplimiento
                 + SECTORES[c.sector]
                 + 0.012 * (plazo - 30)
                 + 1.2 * min(monto / c.cupo_credito, 1.5)
                 - 0.06 * c.antiguedad_anios
                 + (0.5 if emision.month in (12, 1) else 0))   # Temporada difícil
            p_atraso = 1 / (1 + np.exp(-z))

            if rng.random() < p_atraso:
                dias = int(rng.gamma(2, 12)) + 1               # Se paga tarde
            else:
                dias = -int(rng.integers(0, min(plazo, 10)))   # A tiempo o antes
            pago = vencimiento + pd.Timedelta(days=dias)

            filas.append({
                "factura_id": None, "cliente_id": c.cliente_id,
                "fecha_emision": emision.date(), "plazo_dias": plazo,
                "fecha_vencimiento": vencimiento.date(), "monto": monto,
                # Si el pago aún no ha ocurrido a la fecha de corte, queda abierta
                "fecha_pago": pago.date() if pago <= FECHA_CORTE else None,
            })
    df = pd.DataFrame(filas).sort_values("fecha_emision").reset_index(drop=True)
    df["factura_id"] = [f"F{i:06d}" for i in range(1, len(df) + 1)]
    return df


def ensuciar(df):
    """Agrega errores típicos de un archivo exportado a mano."""
    df = df.copy()
    df["monto"] = df["monto"].astype(object)
    idx = rng.choice(len(df), 60, replace=False)
    df.loc[idx, "monto"] = df.loc[idx, "monto"].map(lambda m: f"${m:,.0f}")  # texto
    df.loc[rng.choice(len(df), 25, replace=False), "plazo_dias"] = np.nan    # faltantes
    duplicadas = df.sample(40, random_state=1)                               # duplicados
    return pd.concat([df, duplicadas]).sample(frac=1, random_state=2)


def main():
    Path("datos").mkdir(exist_ok=True)
    clientes = crear_clientes()
    facturas = ensuciar(crear_facturas(clientes))
    datos = facturas.merge(clientes.drop(columns="_cumplimiento"), on="cliente_id")
    datos.to_csv("datos/facturas.csv", index=False)
    print(f"Listo: {len(datos):,} filas en datos/facturas.csv")


if __name__ == "__main__":
    main()
