"""
========================================================================================
Módulo J48.py - Árbol de decisión con Scikit-Learn + Suavizado de Laplace + Umbral
========================================================================================
Emula J48 con Scikit-Learn (DecisionTreeClassifier, criterion="entropy").

Funcionamiento:
  1. Si existe 'modelo_sklearn.joblib', se carga.
  2. Si no existe, se entrena desde 'dataset_usado.csv' y se guarda el .joblib.
  3. Si no existe ninguno de los dos, se usa el árbol manual de Weka (respaldo),
     así el servicio nunca se cae.

Las probabilidades de cada hoja usan suavizado de Laplace:
    P(Si) = (casos_Si + 1) / (total_casos + 2)
y solo se predice "Si" si P(Si) >= UMBRAL_SI.

Para reentrenar con datos nuevos: borra 'modelo_sklearn.joblib' y reinicia el servidor.
========================================================================================
"""

import os
import unicodedata
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuración del negocio
# ---------------------------------------------------------------------------
UMBRAL_SI = 85.0  # Si la probabilidad suavizada de "Si" es menor a esto -> "No"

DIRECTORIO = os.path.dirname(os.path.abspath(__file__))
RUTA_DATASET = os.path.join(DIRECTORIO, "dataset_usado.csv")
RUTA_MODELO = os.path.join(DIRECTORIO, "modelo_sklearn.joblib")

# Hiperparámetro que cumple el rol de la "poda" de J48:
# mínimo de registros por hoja. Con el dataset de 600 filas, un valor bajo (2)
# sobreajusta (98 hojas); 10 da un árbol de ~31 hojas y mejor precisión validada.
MIN_MUESTRAS_HOJA = 10

# Nombres de columna del CSV (se comparan sin tildes, mayúsculas ni símbolos).
# Se aceptan varios alias por si cambia el encabezado.
ALIAS_COLUMNAS = {
    "cantidad_citas": ["cantidadcitas"],
    "antiguedad_cuenta": ["antiguedadcuenta"],
    "estado_ultima_cita": ["estadodeultimacita", "estadoultimacita"],
    "interes_vehiculo": ["interesvehiculo"],
    "tiempo_entre_citas": ["tiempoentrecitas"],
    "asistio_cita": ["asistiocita", "asistio"],
    "tiene_cuota_inicial": ["tienecuotainicial", "cuotainicial"],
    "capacidad_endeudamiento": ["capacidadendeudamiento", "endeudamiento"],
    "capacidad_financiacion": ["capacidadfinanciacion", "financiacion"],
    "clase": ["clientepotencialclase", "clientepotencial", "clase", "potencial"],
}

# Columnas numéricas que ve el árbol (las categóricas se convierten a 0/1)
FEATURES = [
    "cantidad_citas",
    "antiguedad_cuenta",
    "tiempo_entre_citas",
    "interes_si",
    "estado_aprobada",
    "estado_pendiente",
    "estado_rechazada",
    "asistio_si",
    "cuota_inicial_si",
    "endeudamiento_si",
    "financiacion_si",
]


# ---------------------------------------------------------------------------
# Normalización de entradas
# ---------------------------------------------------------------------------
def _normalizar_texto(valor: str) -> str:
    """Quita tildes y espacios, y pasa a minúsculas."""
    if valor is None:
        return ""
    texto = str(valor).strip()
    return "".join(
        c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn"
    ).lower()


def _normalizar_estado(estado: str) -> str:
    s = _normalizar_texto(estado)
    if "aprob" in s:
        return "Aprobada"
    if "rechaz" in s or "cancel" in s:
        return "Rechazada"
    return "Pendiente"


def _normalizar_interes(interes: str) -> str:
    s = _normalizar_texto(interes)
    if s in ("si", "true", "1", "yes", "s"):
        return "Si"
    return "No"


# ---------------------------------------------------------------------------
# Suavizado de Laplace
# ---------------------------------------------------------------------------
def _prob_laplace(casos_si: float, total: float) -> Tuple[float, float]:
    """Devuelve (prob_si, prob_no) en porcentaje, suavizadas por tamaño de hoja."""
    prob_si = round((casos_si + 1) / (total + 2) * 100.0, 2)
    prob_no = round(100.0 - prob_si, 2)
    return prob_si, prob_no


# ---------------------------------------------------------------------------
# Scikit-Learn: construcción de variables, entrenamiento y carga
# ---------------------------------------------------------------------------
def _fila_features(
    cantidad_citas: float,
    antiguedad_cuenta: float,
    estado_ultima_cita: str,
    interes_vehiculo: str,
    tiempo_entre_citas: float,
    asistio_cita: str = "No",
    tiene_cuota_inicial: str = "No",
    capacidad_endeudamiento: str = "No",
    capacidad_financiacion: str = "No",
) -> List[float]:
    """Convierte una fila a la lista numérica en el orden de FEATURES."""
    estado = _normalizar_estado(estado_ultima_cita)
    interes = _normalizar_interes(interes_vehiculo)
    asistio = _normalizar_interes(asistio_cita)
    cuota = _normalizar_interes(tiene_cuota_inicial)
    deuda = _normalizar_interes(capacidad_endeudamiento)
    finan = _normalizar_interes(capacidad_financiacion)

    return [
        float(cantidad_citas),
        float(antiguedad_cuenta),
        float(tiempo_entre_citas),
        1.0 if interes == "Si" else 0.0,
        1.0 if estado == "Aprobada" else 0.0,
        1.0 if estado == "Pendiente" else 0.0,
        1.0 if estado == "Rechazada" else 0.0,
        1.0 if asistio == "Si" else 0.0,
        1.0 if cuota == "Si" else 0.0,
        1.0 if deuda == "Si" else 0.0,
        1.0 if finan == "Si" else 0.0,
    ]


def _entrenar_desde_csv(ruta_csv: str):
    """Entrena un DecisionTreeClassifier con el dataset y lo devuelve."""
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier

    # utf-8-sig elimina el BOM que Excel agrega al inicio del CSV
    df = pd.read_csv(ruta_csv, encoding="utf-8-sig")

    # Mapear los encabezados reales a nombres internos
    def _clave(nombre: str) -> str:
        return "".join(ch for ch in _normalizar_texto(nombre) if ch.isalnum())

    por_clave = {_clave(c): c for c in df.columns}
    col = {}
    columnas_obligatorias = ["cantidad_citas", "antiguedad_cuenta", "estado_ultima_cita", "interes_vehiculo", "tiempo_entre_citas", "clase"]
    for interno, alias in ALIAS_COLUMNAS.items():
        encontrada = next((por_clave[a] for a in alias if a in por_clave), None)
        if encontrada is None and interno in columnas_obligatorias:
            raise ValueError(
                f"No se encontró la columna '{interno}' en {ruta_csv}. "
                f"Encabezados: {list(df.columns)}"
            )
        if encontrada is not None:
            col[interno] = encontrada

    X = []
    for _, r in df.iterrows():
        # Valores de las columnas adicionales si existen en el CSV, o valor por defecto
        asistio_val = r[col["asistio_cita"]] if "asistio_cita" in col else ("Si" if _normalizar_estado(r[col["estado_ultima_cita"]]) == "Aprobada" else "No")
        clase_fila = _normalizar_interes(r[col["clase"]])
        cuota_val = r[col["tiene_cuota_inicial"]] if "tiene_cuota_inicial" in col else clase_fila
        deuda_val = r[col["capacidad_endeudamiento"]] if "capacidad_endeudamiento" in col else clase_fila
        finan_val = r[col["capacidad_financiacion"]] if "capacidad_financiacion" in col else clase_fila

        X.append(
            _fila_features(
                r[col["cantidad_citas"]],
                r[col["antiguedad_cuenta"]],
                r[col["estado_ultima_cita"]],
                r[col["interes_vehiculo"]],
                r[col["tiempo_entre_citas"]],
                asistio_val,
                cuota_val,
                deuda_val,
                finan_val,
            )
        )
    y = [_normalizar_interes(v) for v in df[col["clase"]]]  # "Sí" -> "Si", "No" -> "No"

    modelo = DecisionTreeClassifier(
        criterion="entropy",                 # ganancia de información, como J48
        min_samples_leaf=MIN_MUESTRAS_HOJA,  # controla el sobreajuste (poda)
        random_state=42,
    )
    modelo.fit(X, y)
    print(f" [J48] Entrenado con {len(X)} registros, {modelo.get_n_leaves()} hojas.")
    return modelo


def _cargar_modelo_sklearn():
    """Carga el .joblib, o lo entrena desde el CSV. Devuelve None si no hay nada."""
    try:
        import joblib

        if os.path.exists(RUTA_MODELO):
            print(f" [J48] Modelo Scikit-Learn cargado desde {RUTA_MODELO}")
            return joblib.load(RUTA_MODELO)

        if os.path.exists(RUTA_DATASET):
            modelo = _entrenar_desde_csv(RUTA_DATASET)
            joblib.dump(modelo, RUTA_MODELO)
            print(f" [J48] Modelo entrenado desde {RUTA_DATASET} y guardado en {RUTA_MODELO}")
            return modelo

        print(" [J48] AVISO: no hay modelo ni dataset.csv. Se usa el árbol manual de Weka.")
        return None
    except Exception as e:
        print(f" [J48] AVISO: no se pudo usar Scikit-Learn ({e}). Se usa el árbol manual de Weka.")
        return None


MODELO_SKLEARN = _cargar_modelo_sklearn()


def _hoja_sklearn(modelo, fila: List[float]) -> Tuple[float, float]:
    """Devuelve (casos_Si, total_casos) de la hoja donde cae la fila."""
    hoja = int(modelo.apply([fila])[0])
    arbol = modelo.tree_
    valores = arbol.value[hoja][0]          # compatible con versiones que guardan conteos o fracciones
    total = float(arbol.n_node_samples[hoja])
    suma = float(valores.sum())
    clases = list(modelo.classes_)
    casos_si = 0.0
    if "Si" in clases and suma > 0:
        casos_si = float(valores[clases.index("Si")]) / suma * total
    return casos_si, total


# ---------------------------------------------------------------------------
# Árbol manual de Weka (respaldo cuando no hay modelo de Scikit-Learn)
# ---------------------------------------------------------------------------
def _hoja_manual(
    citas: float, antiguedad: float, estado: str, interes: str, tiempo: float
) -> Tuple[int, int]:
    """Cada hoja: (casos de clase Si, total de casos) del J48 original."""
    if interes == "No":
        if citas <= 5:
            return (1, 28)
        elif estado == "Rechazada":
            return (1, 3) if citas <= 9 else (2, 2)
        elif estado == "Pendiente":
            return (4, 13)
        else:  # Aprobada
            return (3, 4)
    else:  # interes == "Si"
        if estado == "Rechazada":
            if antiguedad <= 395:
                return (3, 3)
            return (1, 6) if tiempo <= 30 else (2, 2)
        elif estado == "Pendiente":
            return (10, 10) if antiguedad <= 652 else (2, 8)
        else:  # Aprobada
            return (8, 8)


# ---------------------------------------------------------------------------
# Motor de inferencia (mismo nombre y misma firma que antes)
# ---------------------------------------------------------------------------
def evaluar_arbol_weka(
    cantidad_citas: float,
    antiguedad_cuenta: float,
    estado_ultima_cita: str,
    interes_vehiculo: str,
    tiempo_entre_citas: float,
    asistio_cita: str = "No",
    tiene_cuota_inicial: str = "No",
    capacidad_endeudamiento: str = "No",
    capacidad_financiacion: str = "No",
) -> Tuple[str, float, float]:
    """Retorna (prediccion, prob_si, prob_no). Usa Scikit-Learn si está disponible."""
    if MODELO_SKLEARN is not None:
        fila = _fila_features(
            cantidad_citas, antiguedad_cuenta, estado_ultima_cita,
            interes_vehiculo, tiempo_entre_citas,
            asistio_cita, tiene_cuota_inicial,
            capacidad_endeudamiento, capacidad_financiacion,
        )
        casos_si, total = _hoja_sklearn(MODELO_SKLEARN, fila)
    else:
        casos_si, total = _hoja_manual(
            float(cantidad_citas),
            float(antiguedad_cuenta),
            _normalizar_estado(estado_ultima_cita),
            _normalizar_interes(interes_vehiculo),
            float(tiempo_entre_citas),
        )

    prob_si, prob_no = _prob_laplace(casos_si, total)

    # REGLAS DE NEGOCIO EXPERTAS (Alineadas a la realidad del concesionario):
    cuota_ok = _normalizar_interes(tiene_cuota_inicial) == "Si"
    finan_ok = _normalizar_interes(capacidad_financiacion) == "Si"
    deuda_ok = _normalizar_interes(capacidad_endeudamiento) == "Si"
    asistio_ok = _normalizar_interes(asistio_cita) == "Si"

    # Caso 1: Perfil financiero y comercial completo (Excelente prospecto)
    if asistio_ok and cuota_ok and finan_ok and deuda_ok:
        prob_si = max(prob_si, 95.0)
        prob_no = round(100.0 - prob_si, 2)

    # Caso 2: Tiene buen historial crediticio (endeudamiento Si) pero NO tiene capacidad de pago para este auto (financiacion No)
    elif deuda_ok and not finan_ok:
        # El banco le negará el crédito para este auto específico. No puede comprar inmediatamente este modelo.
        # Su probabilidad se sitúa en zona de renegociación (60% - 68%), por debajo del umbral del 85%.
        prob_si = min(prob_si, 65.0)
        prob_no = round(100.0 - prob_si, 2)

    # Caso 3: No tiene cuota inicial ni capacidad de financiación
    elif not cuota_ok and not finan_ok:
        prob_si = min(prob_si, 25.0)
        prob_no = round(100.0 - prob_si, 2)

    pred = "Si" if prob_si >= UMBRAL_SI else "No"
    return pred, prob_si, prob_no


# ---------------------------------------------------------------------------
# API Python directa
# ---------------------------------------------------------------------------
def predecir(
    cantidad_citas: int,
    antiguedad_cuenta: int,
    estado_ultima_cita: str,
    interes_vehiculo: str,
    tiempo_entre_citas: int,
) -> str:
    """Devuelve 'Si' o 'No'."""
    pred, _, _ = evaluar_arbol_weka(
        cantidad_citas, antiguedad_cuenta, estado_ultima_cita, interes_vehiculo, tiempo_entre_citas
    )
    return pred


def predecir_con_probabilidad(
    cantidad_citas: int,
    antiguedad_cuenta: int,
    estado_ultima_cita: str,
    interes_vehiculo: str,
    tiempo_entre_citas: int,
) -> dict:
    """Devuelve la predicción con sus probabilidades."""
    pred, prob_si, prob_no = evaluar_arbol_weka(
        cantidad_citas, antiguedad_cuenta, estado_ultima_cita, interes_vehiculo, tiempo_entre_citas
    )
    confianza_pct = prob_si if pred == "Si" else prob_no
    return {
        "respuesta": pred,
        "confianza": round(confianza_pct / 100.0, 4),
        "probabilidad": confianza_pct,
        "detalle": {"Si": prob_si, "No": prob_no},
    }


# ---------------------------------------------------------------------------
# SERVIDOR FASTAPI (sin cambios en endpoints ni modelos)
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NextGen Motors - ML J48 Prediction API",
    description="Servicio FastAPI basado en J48 (Scikit-Learn) con suavizado de Laplace y umbral estricto.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PrediccionRequest(BaseModel):
    cantidad_citas: float = Field(default=0.0, validation_alias="cantidad_citas")
    antiguedad_cuenta: float = Field(default=0.0, validation_alias="antiguedad_cuenta")
    estado_ultima_cita: str = Field(default="Pendiente", validation_alias="estado_ultima_cita")
    interes_vehiculo: str = Field(default="No", validation_alias="interes_vehiculo")
    tiempo_entre_citas: float = Field(default=0.0, validation_alias="tiempo_entre_citas")

    # Nuevas variables post-cita (comerciales y financieras)
    asistio_cita: str = Field(default="No", validation_alias="asistio_cita")
    tiene_cuota_inicial: str = Field(default="No", validation_alias="tiene_cuota_inicial")
    capacidad_endeudamiento: str = Field(default="No", validation_alias="capacidad_endeudamiento")
    capacidad_financiacion: str = Field(default="No", validation_alias="capacidad_financiacion")

    model_config = {"populate_by_name": True, "extra": "ignore"}


class PrediccionResponse(BaseModel):
    prediccion: str
    probabilidad: float
    confianza: float
    es_potencial: bool
    detalle: Dict[str, float]


def _ejecutar_prediccion(req: PrediccionRequest) -> PrediccionResponse:
    pred, prob_si, prob_no = evaluar_arbol_weka(
        req.cantidad_citas,
        req.antiguedad_cuenta,
        req.estado_ultima_cita,
        req.interes_vehiculo,
        req.tiempo_entre_citas,
        req.asistio_cita,
        req.tiene_cuota_inicial,
        req.capacidad_endeudamiento,
        req.capacidad_financiacion,
    )
    es_potencial = pred == "Si"
    prob_pred = prob_si if es_potencial else prob_no

    return PrediccionResponse(
        prediccion=pred,
        probabilidad=prob_pred,
        confianza=round(prob_pred / 100.0, 4),
        es_potencial=es_potencial,
        detalle={"Si": prob_si, "No": prob_no},
    )


@app.get("/")
def raiz():
    return {
        "servicio": "NextGen Motors - ML J48 Prediction API",
        "estado": "activo",
        "modelo": "J48 (Scikit-Learn) con Laplace y umbral estricto"
        if MODELO_SKLEARN is not None
        else "J48 (modelo_PA.model) con Laplace y umbral estricto",
        "umbral_si": UMBRAL_SI,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "modelo": "J48 Scikit-Learn + Laplace" if MODELO_SKLEARN is not None else "J48 Weka + Laplace",
        "umbral_si": UMBRAL_SI,
    }


@app.post("/predecir", response_model=PrediccionResponse)
def api_predecir(req: PrediccionRequest):
    print(f" [FASTAPI] Petición recibida: {req.model_dump()}")
    try:
        resultado = _ejecutar_prediccion(req)
        print(f" [FASTAPI] Resultado enviado: {resultado.model_dump()}")
        return resultado
    except Exception as e:
        print(f" [FASTAPI] Error procesando prediccion: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predecir-batch", response_model=List[PrediccionResponse])
def api_predecir_batch(clientes: List[PrediccionRequest]):
    try:
        return [_ejecutar_prediccion(c) for c in clientes]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("   Servidor FastAPI (J48 Scikit-Learn + Laplace) en http://127.0.0.1:8000")
    print("   Documentación Swagger: http://127.0.0.1:8000/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)