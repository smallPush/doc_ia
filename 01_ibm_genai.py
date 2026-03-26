"""
Tema 1: IBM GenAI Engineering with PyTorch, LangChain & Hugging Face

Concepto:
Demostrar cómo integrar modelos open-source de Hugging Face usando PyTorch
dentro de las cadenas (Chains) de LangChain, dotándolas de MEMORIA Y CONTEXTO
para mantener un historial conversacional fluido.
"""

import sys

def main():
    print("--- 1. IBM GenAI Engineering: PyTorch + LangChain + Hugging Face ---")
    print("Cargando el entorno para construir un pipeline con memoria conversacional...\n")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        # Abstracciones de LangChain
        from langchain_community.llms import HuggingFacePipeline
        from langchain_core.prompts import PromptTemplate
        # Novedad: Importamos módulos para retención de memoria
        from langchain_classic.memory import ConversationBufferMemory
        from langchain_classic.chains import LLMChain

        print("[OK] Dependencias correctamente encontradas.")

        # === 1. Configuración de HuggingFace y PyTorch ===
        model_id = "gpt2"
        print(f"-> Inicializando el tokenizer y modelo ({model_id}) con PyTorch...")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

        generative_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=50,
            temperature=0.7,
            device=-1                # -1 es CPU; Usa 0 si tienes tarjeta de video activada
        )

        # === 2. Integración con LangChain y Memoria ===
        llm = HuggingFacePipeline(pipeline=generative_pipeline)

        # Instanciamos la memoria para guardar el contexto de la charla
        memoria = ConversationBufferMemory(memory_key="historial_chat")

        # Definimos el Prompt incluyendo explícitamente la variable de memoria
        template = (
            "You are a helpful and conversational AI assistant.\n\n"
            "Previous conversation history:\n"
            "{historial_chat}\n\n"
            "User: {pregunta}\n"
            "Assistant:"
        )
        prompt = PromptTemplate(template=template, input_variables=["historial_chat", "pregunta"])

        # Creamos la cadena (Chain) uniendo LLM, Prompt y la Memoria para orquestar los turnos
        chain = LLMChain(
            llm=llm,
            prompt=prompt,
            memory=memoria,
            verbose=False # Ponlo en True para depurar y ver al desnudo cómo inyecta el contexto LangChain
        )

        # === 3. Inferencia Evolutiva (Charla con Contexto) ===
        preguntas = [
            "Hello, my name is Ruben and I am learning to program RAG systems.",
            "That's great. Hey, do you remember what my name is and what I am studying?"
        ]

        print("\n=== INICIO DE CONVERSACIÓN CON MEMORIA ===")
        for i, pregunta in enumerate(preguntas, 1):
            print(f"\n[Turno {i}] Usuario: {pregunta}")
            print(f"Generando respuesta...")

            # Al usar predict(), la cadena inyecta {historial_chat} automáticamente extrayéndolo de memory
            respuesta = chain.predict(pregunta=pregunta)

            # El modelo tenderá a auto-completar; extraemos y mostramos el texto generado sin iterar sobre él infinitamente
            print(f"Asistente: {respuesta.strip()}")

        print("\n=== AUDITORÍA: CONTENIDO RETENIDO EN LA MEMORIA INTERNA ===")
        print(memoria.load_memory_variables({})["historial_chat"])
        print("=========================================================\n")

    except ImportError as e:
        print(f"[ERROR] Faltan dependencias para ejecutar el modo avanzado: {e}")
        print("Asegúrate de ejecutar: pip install torch transformers langchain langchain-community")

if __name__ == "__main__":
    main()
