"""
Tema 8: Fundamentals of AI Agents Using RAG and LangChain
Concepto: Despliegue de un Agente que es libre de interactuar con Funciones Python (Tools) 
siguiendo un marco mental de Razonamiento ReAct (Reason/Act).
"""
def main():
    print("--- 8. AI Agents (Razonamiento Lógico y Herramientas con LangChain) ---")
    try:
        from langchain.agents import initialize_agent, AgentType, Tool
        from langchain_community.llms import HuggingFacePipeline
        from transformers import pipeline
        import numexpr as ne # Evaluador matemático C++ (muy seguro)
        
        print("-> Inicializando el 'Cerebro Central' (LLM de Instrucciones)...")
        # google/flan-t5-large tiene capacidades sólidas de ZeroShot Instruct 
        pipe = pipeline("text2text-generation", model="google/flan-t5-large", max_new_tokens=40)
        llm = HuggingFacePipeline(pipeline=pipe)
        
        print("-> Fraguando las Herramientas Externas (Tools)...")
        def calculadora_avanzada(expresion: str) -> str:
            """Procesa matemáticas en texto puro y lo devuelve a string."""
            print(f"\n   [INTERRUPCIÓN API] El modelo invocó la calculadora con: {expresion}")
            try:
                return str(ne.evaluate(expresion))
            except Exception as e:
                return f"Error sintáctico al evaluar matemáticamente: {e}"

        # Al array `tools` podemos añadir búsqueda en Wikipedia, APIs de Clima, etc.
        tools = [
            Tool(
                name="Calculadora Inteligente",
                func=calculadora_avanzada,
                # La "description" es LEIDA por el LLM. Él decidirá usarla solo si la tarea concuerda con la descripción.
                description="Útil para realizar cálculos matemáticos complejos, sumas, restas y multiplicaciones. Recibe álgebra limpia y precisa."
            )
        ]
        
        print("-> Inyectando Patrones ReAct (Planificación de Acciones)...")
        # El Agente ZERO_SHOT asume todo por sí mismo leyendo la situación actual.
        agent = initialize_agent(
            tools=tools, 
            llm=llm, 
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
            verbose=True,            # Transparencia total. Podremos ver cómo piensa el bucle (Thought -> Action -> Observation)
            handle_parsing_errors=True 
        )
        
        pregunta = "Calcula cuánto es 257 multiplicado por 13."
        print(f"\n[Usuario envía ticket]: {pregunta}")
        print("=== EXTRACCIÓN DEL MOTOR COGNITIVO DEL AGENTE ===")
        
        try:
            respuesta = agent.invoke(pregunta)
            print("\n=== RESPUESTA FINAL ===")
            print(respuesta['output'])
        except Exception as e:
             # FLAN a veces tiene alucinaciones que rompen el parser de LangChain al no ser un GPT-4 maduro
             print(f"\n[NOTA DE DEPURACIÓN] El pipeline ha detectado una desviación lógica del LLM Open-Source: {e}")
             print("Los Agentes ReAct en open source funcionan infinitamente mejor con modelos > 7 Billones (Llama-3, Mistral, Mixtral).")
        
    except ImportError as e:
        print(f"[ERROR] Módulos ausentes: {e}")
        print("Ejecuta en consola: pip install langchain langchain-community transformers numexpr")

if __name__ == "__main__":
    main()
