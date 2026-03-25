"""
Tema 12: Generative AI: Prompt Engineering Basics
Concepto: Forzado de Inyección Racional Contextual usando LangChain y el mecanismo "FewShotPromptTemplate".
"""
def main():
    print("--- 12. Ingeniería de Prompts Transparente ---")
    try:
        from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
        
        print("\n-> Construyendo la técnica de Ingeniería Avanzada (Few-Shot Prompting)...")
        
        # 1. Diseñando Disparadores Analítico-Concretos.
        # Le mostramos ejemplos analíticos ("Few-Shot") para asegurar que su tokenización prediga la estructura de diccionario.
        ejemplos_semanticos = [
            {"pregunta": "El cielo estival.", "respuesta": "Soleado y despejado"},
            {"pregunta": "Tormenta de nieve alpina.", "respuesta": "Frío extremo y blanco intenso"},
            {"pregunta": "Un bosque de Oregón en octubre.", "respuesta": "Humedad y árboles caducifolios"}
        ]
        
        # 2. Esqueleto de la Micro-estructura Inyectada
        molde_mapeable = PromptTemplate(
            input_variables=["pregunta", "respuesta"],
            template="Contexto Geográfico: {pregunta}\nAnálisis Medioambiental: {respuesta}"
        )
        
        # 3. Empaquetador Dinámico
        plantilla_gen_ai = FewShotPromptTemplate(
            examples=ejemplos_semanticos,
            example_prompt=molde_mapeable,
            prefix="Comportate como un meteorólogo experto calculando condiciones atmosféricas implícitas.\n\n", # Comportamiento Persona (Act-as)
            suffix="\n\nContexto Geográfico: {usuario_input}\nAnálisis Medioambiental:", # El disparador que fuerza al LLM a continuar
            input_variables=["usuario_input"],
            example_separator="\n\n"
        )
        
        # Ejemplo Evaluativo del Usuario
        prueba_usuario = "La sabana africana a las dos de la tarde sin nubes."
        
        # Renderización 1 a 1 de lo que se enviaría al modelo.
        prompt_final = plantilla_gen_ai.format(usuario_input=prueba_usuario)
        
        print("\n=== PAYLOAD AL LLM (PROMPT GENERADO POR EL MOTOR DE INGENIERÍA) ===\n")
        print(prompt_final)
        print("\n===================================================================")
        print("\n[Éxito] Listo para invocar `.invoke()` pasándole este payload finalizado y enriquecido con ejemplos.")
        
    except ImportError as e:
        print(f"[ERROR] Faltan librerías nativas: {e}")
        print("Instala ejecutando: pip install langchain-core")

if __name__ == "__main__":
    main()
