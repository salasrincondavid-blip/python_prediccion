"""
========================================================================================
Módulo J48.py - Árbol de decisión 100% Scikit-Learn + Suavizado de Laplace + Umbral
========================================================================================
Emula J48 con Scikit-Learn (DecisionTreeClassifier, criterion="entropy").

Funcionamiento:
  1. Si existe 'modelo_sklearn.joblib' Y está al día con 'dataset_usado.csv', se carga.
  2. Si el CSV es más nuevo que el .joblib (o el .joblib no existe), se reentrena solo.
  3. Si no hay CSV ni modelo, el servicio NO arranca (no hay árboles escritos a mano ni
     reglas "inventadas": toda predicción sale del modelo entrenado con el CSV).

Probabilidades de cada hoja con suavizado de Laplace:
    P(Si) = (casos_Si + 1) / (total_casos + 2)
y solo se predice "Si" si P(Si) >= UMBRAL_SI.

Para reentrenar con datos nuevos: reemplaza 'dataset_usado.csv' y reinicia el servidor.
========================================================================================
"""

import os
import unicodedata
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
UMBRAL_SI = 85.0  # Si la probabilidad suavizada de "Si" es menor a esto -> "No"

DIRECTORIO = os.path.dirname(os.path.abspath(__file__))
RUTA_DATASET = os.path.join(DIRECTORIO, "dataset_usado.csv")
RUTA_MODELO = os.path.join(DIRECTORIO, "modelo_sklearn.joblib")

# Candidatos de "poda" (mínimo de registros por hoja). Se elige el mejor por
# validación cruzada al entrenar.
CANDIDATOS_MIN_HOJA = [5, 10, 15, 20, 30]

# Alias de encabezados del CSV (se comparan sin tildes, mayúsculas ni símbolos).
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

# Variables que ve el árbol (las categóricas se convierten a 0/1)
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
def _normalizar_texto(valor) -> str:
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


def _normalizar_si_no(valor: str) -> str:
    """'Sí'/'Si'/true/1/yes -> 'Si'; cualquier otra cosa -> 'No'."""
    return "Si" if _normalizar_texto(valor) in ("si", "true", "1", "yes", "s") else "No"


# ---------------------------------------------------------------------------
# Suavizado de Laplace
# ---------------------------------------------------------------------------
def _prob_laplace(casos_si: float, total: float) -> Tuple[float, float]:
    """Devuelve (prob_si, prob_no) en porcentaje, suavizadas por tamaño de hoja."""
    prob_si = round((casos_si + 1) / (total + 2) * 100.0, 2)
    return prob_si, round(100.0 - prob_si, 2)


# ---------------------------------------------------------------------------
# Construcción de variables
# ---------------------------------------------------------------------------
def _fila_features(
    cantidad_citas, antiguedad_cuenta, estado_ultima_cita, interes_vehiculo,
    tiempo_entre_citas, asistio_cita, tiene_cuota_inicial,
    capacidad_endeudamiento, capacidad_financiacion,
) -> List[float]:
    """Convierte una fila a la lista numérica en el orden de FEATURES."""
    estado = _normalizar_estado(estado_ultima_cita)
    si = lambda v: 1.0 if _normalizar_si_no(v) == "Si" else 0.0
    return [
        float(cantidad_citas),
        float(antiguedad_cuenta),
        float(tiempo_entre_citas),
        si(interes_vehiculo),
        1.0 if estado == "Aprobada" else 0.0,
        1.0 if estado == "Pendiente" else 0.0,
        1.0 if estado == "Rechazada" else 0.0,
        si(asistio_cita),
        si(tiene_cuota_inicial),
        si(capacidad_endeudamiento),
        si(capacidad_financiacion),
    ]


# ---------------------------------------------------------------------------
# Entrenamiento y carga
# ---------------------------------------------------------------------------
def _entrenar_desde_csv(ruta_csv: str) -> DecisionTreeClassifier:
    """Entrena el DecisionTreeClassifier con TODAS las columnas del CSV."""
    df = pd.read_csv(ruta_csv, encoding="utf-8-sig")  # utf-8-sig quita el BOM de Excel

    def _clave(nombre: str) -> str:
        return "".join(ch for ch in _normalizar_texto(nombre) if ch.isalnum())

    por_clave = {_clave(c): c for c in df.columns}
    col = {}
    for interno, alias in ALIAS_COLUMNAS.items():
        encontrada = next((por_clave[a] for a in alias if a in por_clave), None)
        if encontrada is None:
            raise ValueError(
                f"Falta la columna '{interno}' en {ruta_csv}. Encabezados: {list(df.columns)}"
            )
        col[interno] = encontrada

    X = [
        _fila_features(*(r[col[k]] for k in ALIAS_COLUMNAS if k != "clase"))
        for _, r in df.iterrows()
    ]
    y = [_normalizar_si_no(v) for v in df[col["clase"]]]

    # "Poda": se elige min_samples_leaf por validación cruzada estratificada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    busqueda = GridSearchCV(
        DecisionTreeClassifier(criterion="entropy", random_state=42),  # ganancia de información
        {"min_samples_leaf": CANDIDATOS_MIN_HOJA},
        cv=cv, scoring="accuracy",
    )
    busqueda.fit(X, y)
    modelo = busqueda.best_estimator_  # ya reentrenado con todo el dataset

    print(f" [J48] Entrenado con {len(X)} registros | min_samples_leaf="
          f"{busqueda.best_params_['min_samples_leaf']} | {modelo.get_n_leaves()} hojas | "
          f"precisión CV={busqueda.best_score_:.3f}")
    return modelo


def _cargar_modelo() -> DecisionTreeClassifier:
    """Carga el .joblib si está al día; si no, reentrena desde el CSV."""
    hay_csv = os.path.exists(RUTA_DATASET)
    hay_modelo = os.path.exists(RUTA_MODELO)

    if hay_modelo and (not hay_csv or os.path.getmtime(RUTA_MODELO) >= os.path.getmtime(RUTA_DATASET)):
        modelo = joblib.load(RUTA_MODELO)
        if getattr(modelo, "n_features_in_", None) == len(FEATURES):
            print(f" [J48] Modelo cargado desde {RUTA_MODELO}")
            return modelo
        print(" [J48] El .joblib no coincide con las variables actuales; se reentrena.")

    if not hay_csv:
        raise FileNotFoundError(f"No existe {RUTA_DATASET} ni un modelo válido para cargar.")

    modelo = _entrenar_desde_csv(RUTA_DATASET)
    joblib.dump(modelo, RUTA_MODELO)
    print(f" [J48] Modelo guardado en {RUTA_MODELO}")
    return modelo


MODELO = _cargar_modelo()


def _hoja(modelo: DecisionTreeClassifier, fila: List[float]) -> Tuple[float, float]:
    """Devuelve (casos_Si, total_casos) de la hoja donde cae la fila."""
    hoja = int(modelo.apply([fila])[0])
    arbol = modelo.tree_
    valores = arbol.value[hoja][0]
    total = float(arbol.n_node_samples[hoja])
    suma = float(valores.sum())
    clases = list(modelo.classes_)
    casos_si = float(valores[clases.index("Si")]) / suma * total if "Si" in clases and suma > 0 else 0.0
    return casos_si, total


# ---------------------------------------------------------------------------
# Motor de inferencia (mismo nombre y firma que antes)
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
    """Retorna (prediccion, prob_si, prob_no). Todo sale del árbol + Laplace + umbral."""
    fila = _fila_features(
        cantidad_citas, antiguedad_cuenta, estado_ultima_cita, interes_vehiculo,
        tiempo_entre_citas, asistio_cita, tiene_cuota_inicial,
        capacidad_endeudamiento, capacidad_financiacion,
    )
    casos_si, total = _hoja(MODELO, fila)
    prob_si, prob_no = _prob_laplace(casos_si, total)
    pred = "Si" if prob_si >= UMBRAL_SI else "No"
    return pred, prob_si, prob_no


# ---------------------------------------------------------------------------
# API Python directa
# ---------------------------------------------------------------------------
def predecir(cantidad_citas, antiguedad_cuenta, estado_ultima_cita, interes_vehiculo,
             tiempo_entre_citas, asistio_cita="No", tiene_cuota_inicial="No",
             capacidad_endeudamiento="No", capacidad_financiacion="No") -> str:
    """Devuelve 'Si' o 'No'."""
    return evaluar_arbol_weka(
        cantidad_citas, antiguedad_cuenta, estado_ultima_cita, interes_vehiculo,
        tiempo_entre_citas, asistio_cita, tiene_cuota_inicial,
        capacidad_endeudamiento, capacidad_financiacion,
    )[0]


def predecir_con_probabilidad(cantidad_citas, antiguedad_cuenta, estado_ultima_cita,
                              interes_vehiculo, tiempo_entre_citas, asistio_cita="No",
                              tiene_cuota_inicial="No", capacidad_endeudamiento="No",
                              capacidad_financiacion="No") -> dict:
    """Devuelve la predicción con sus probabilidades."""
    pred, prob_si, prob_no = evaluar_arbol_weka(
        cantidad_citas, antiguedad_cuenta, estado_ultima_cita, interes_vehiculo,
        tiempo_entre_citas, asistio_cita, tiene_cuota_inicial,
        capacidad_endeudamiento, capacidad_financiacion,
    )
    confianza_pct = prob_si if pred == "Si" else prob_no
    return {
        "respuesta": pred,
        "confianza": round(confianza_pct / 100.0, 4),
        "probabilidad": confianza_pct,
        "detalle": {"Si": prob_si, "No": prob_no},
    }


# ---------------------------------------------------------------------------
# SERVIDOR FASTAPI (endpoints y modelos de respuesta sin cambios)
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NextGen Motors - ML J48 Prediction API",
    description="Servicio FastAPI basado en J48 (Scikit-Learn) con suavizado de Laplace y umbral estricto.",
    version="3.0.0",
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

    # Variables post-cita (comerciales y financieras)
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
        req.cantidad_citas, req.antiguedad_cuenta, req.estado_ultima_cita,
        req.interes_vehiculo, req.tiempo_entre_citas, req.asistio_cita,
        req.tiene_cuota_inicial, req.capacidad_endeudamiento, req.capacidad_financiacion,
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
        "modelo": "J48 (Scikit-Learn) con Laplace y umbral estricto",
        "umbral_si": UMBRAL_SI,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "modelo": "J48 Scikit-Learn + Laplace", "umbral_si": UMBRAL_SI}


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