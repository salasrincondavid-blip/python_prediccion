# NextGen Motors - ML J48 Prediction API

API REST construida con **FastAPI** y **Scikit-Learn** para la predicción de clientes potenciales mediante un árbol de decisión tipo J48 con suavizado de Laplace.

---

## 🚀 Despliegue en Render

Puedes desplegar este proyecto de dos formas: **Nativo (Recomendado)** o con **Docker**.

### Opción 1: Despliegue Nativo Python (Más fácil y rápido)
1. Sube este repositorio a **GitHub**.
2. Ve a [Render Dashboard](https://dashboard.render.com/) y haz clic en **New +** -> **Web Service**.
3. Conecta tu repositorio de GitHub.
4. Configura los siguientes campos:
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Haz clic en **Deploy Web Service**.

> *Nota: También puedes usar el archivo `render.yaml` incluido seleccionando **New Blueprint Instance**.*

---

### Opción 2: Despliegue con Docker
1. En Render Dashboard, selecciona **New +** -> **Web Service**.
2. Conecta tu repositorio de GitHub.
3. Render detectará automáticamente el archivo `Dockerfile`.
4. Haz clic en **Deploy Web Service**.

---

## 💻 Ejecución en Local

### 1. Crear entorno virtual
```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/macOS:
source venv/bin/activate
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Iniciar el servidor
```bash
uvicorn main:app --reload --port 8000
```
O directamente:
```bash
python main.py
```

La documentación interactiva estará disponible en:
- Swagger UI: `http://localhost:8000/docs`
- Redoc: `http://localhost:8000/redoc`

---

## 📡 Endpoints Principales

- `GET /`: Información básica de la API y versión del modelo.
- `GET /health`: Verificación del estado del servicio.
- `POST /predecir`: Realiza la predicción para un único cliente.
- `POST /predecir-batch`: Realiza predicciones en lote para una lista de clientes.
