"""
Tema 9: Building Generative AI-Powered Applications with Python
Concepto: Levantando un servidor uvicorn productivo con nuestra IA envuelta en endpoints (FastAPI/REST).
"""
def main():
    print("--- 9. Servidor Inteligente con FastAPI ---")
    print("Instrucciones: Ejecuta en terminal: uvicorn 09_building_genai_apps:app --reload")
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
        import uvicorn
        from transformers import pipeline

        # FastAPI App
        app = FastAPI(title="Generative AI API", version="1.0.0")
        
        # Modelo cargado de forma estática en la RAM al inicio del programa
        print("-> Iniciando 'gpt2' para la API...")
        generator = pipeline("text-generation", model="gpt2", device=-1)

        class InferenceRequest(BaseModel):
            texto: str
            limite_palabras: int = 50

        @app.get("/")
        def check_status():
            """Endpoint Base - Monitoreo de salud del API."""
            return {"status": "ok", "message": "El modelo base está vivo. Llama a /generate con POST."}

        @app.post("/generate")
        def generar_texto(request: InferenceRequest):
            """Endpoint Core - Toma peticiones JSON del cliente y las inyecta en Hugging Face."""
            print(f"Procesando petición entrante: {request.texto}")
            result = generator(request.texto, max_new_tokens=request.limite_palabras)
            return {"ai_response": result[0]["generated_text"].strip()}


        import sys
        if len(sys.argv) > 1 and sys.argv[1] == "run":
            # Puedes probarlo fácilmente corriendo: python 09_building_genai_apps.py run
            print("\nIniciando Uvicorn internamente en localhost:8000...")
            uvicorn.run(app, host="127.0.0.1", port=8000)
            
    except ImportError as e:
        print(f"[ERROR] Dependencias faltantes: {e}")
        print("Instala ejecutando: pip install fastapi uvicorn 'transformers[torch]'")

if __name__ == "__main__":
    main()
