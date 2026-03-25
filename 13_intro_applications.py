"""
Tema 13: Generative AI: Introduction and Applications
Concepto: Arquitectura de Petición Oficial síncrona/Asíncrona hacia OpenAI (GPT-3.5/4).
"""
def main():
    print("--- 13. API Provider Interface (Aplicaciones de la IA Generativa) ---")
    try:
        import os
        from openai import OpenAI
        from openai import AuthenticationError # Previene bloqueos por clave inválida en el test
        
        # Interfaz de seguridad: Leemos de las variables del S.O.
        clave = os.environ.get("OPENAI_API_KEY", "CLAVE_FICTICIA_SI_ESTA_VACIA")
        
        if clave == "CLAVE_FICTICIA_SI_ESTA_VACIA":
            print("\n[ALERTA] OPENAI_API_KEY no detectada. Modo Demo/Fallback activado.")
        
        print("\n-> 1. Instanciando el Endpoint del Cliente nativo de OpenAI...")
        # NOTA: Instanciará pero fallará en predict si la clave no es de verdad.
        cliente = OpenAI(api_key=clave)
        
        instruccion_usuario = "Redacta de forma extremadamente técnica y concisa qué es un Modelo de Difusión."
        print(f"\n-> Prompt Inyectado por tu software local: '{instruccion_usuario}'")
        
        print("\n-> 2. Pidiendo Completado de Chat al Servidor Remoto...")
        try:
            # Esta es la convención de software usada globalmente: Role=system + Role=user (Mensajes apilados)
            respuesta = cliente.chat.completions.create(
                model="gpt-3.5-turbo", # Puede ser gpt-4o, o cualquier modelo propietario.
                messages=[
                    # 'System' condiciona los rieles de guarda o System Prompts subyacentes
                    {"role": "system", "content": "Eres un arquitecto principal de ML, solo das definiciones de diccionario."},
                    {"role": "user", "content": instruccion_usuario}
                ],
                temperature=0.1,    # Estrictamente lógico y no creativo (Evita alucinaciones).
                max_tokens=60
            )
            
            print("\n=== RESPUESTA RECIBIDA (JSON Payload Deserializado) ===")
            texto_crudo = respuesta.choices[0].message.content
            print(texto_crudo)
            
        except AuthenticationError:
            print(f"\n[Mock] Omitiendo llamada real por clave de red no configurada.")
            print(f"      (Respuesta simulada): 'Un modelo de difusión destruye datos con ruido y entrena una red neuronal que invierte la estocástica para fabricar imágenes sintéticas y puras.'")
            
    except ImportError as e:
        print(f"\n[ERROR] Librería de transporte (Provider API) faltante: {e}")
        print("Asegúrate de ejecutar: pip install openai")

if __name__ == "__main__":
    main()
