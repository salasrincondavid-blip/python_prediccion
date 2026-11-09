"""
Punto de entrada estándar para el servidor FastAPI.
Permite ejecutar tanto 'uvicorn main:app' como 'python main.py'.
"""
import os
import uvicorn
from J48 import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
