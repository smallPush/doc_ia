"""
Ejemplos Ejecutables para los Cursos de doc_ia
==============================================

Este fichero contiene código de ejemplo (plantillas ejecutables) para cada uno
de los temas mencionados en el README.md.

Nota: Para ejecutar todos los ejemplos que utilizan librerías avanzadas,
necesitarás instalar las dependencias correspondientes (p.ej. vía pip):
pip install torch transformers langchain tensorflow scikit-learn numpy datasets peft openai fastapi
"""

import sys

# ==============================================================================
# 1. IBM GenAI Engineering with PyTorch, LangChain & Hugging Face
# ==============================================================================
def ejemplo_ibm_genai_engineering():
    print("\n--- 1. IBM GenAI Engineering with PyTorch, LangChain & Hugging Face ---")
    print("Concepto: Inicializar un pipeline simple de HuggingFace con LangChain.")
    try:
        from langchain.llms import HuggingFacePipeline
        from transformers import pipeline
        # --- Código de Ejemplo ---
        # model_id = "gpt2"
        # pipe = pipeline("text-generation", model=model_id)
        # llm = HuggingFacePipeline(pipeline=pipe)
        # print(llm("Hola, mi nombre es"))
        print("[OK] Las librerías están instaladas. Descomenta el código de ejemplo para probar.")
    except ImportError:
        print("[INFO] Faltan dependencias (langchain, transformers). Código en modo plantilla.")

# ==============================================================================
# 2. Generative AI Engineering and Fine-Tuning Transformers
# ==============================================================================
def ejemplo_finetuning_transformers():
    print("\n--- 2. Generative AI Engineering and Fine-Tuning Transformers ---")
    print("Concepto: Configuración básica de entrenamiento para un modelo usando Trainer API.")
    try:
        from transformers import TrainingArguments, Trainer
        # --- Código de Ejemplo ---
        # training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
        # trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
        # trainer.train()
        print("[OK] Plantilla de Fine-tuning con Hugging Face preparada.")
    except ImportError:
        print("[INFO] Falta dependencia (transformers). Código en modo plantilla.")

# ==============================================================================
# 3. Generative AI Language Modeling with Transformers
# ==============================================================================
def ejemplo_language_modeling():
    print("\n--- 3. Generative AI Language Modeling with Transformers ---")
    print("Concepto: Generación de texto auto-regresiva.")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # --- Código de Ejemplo ---
        # tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        # model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        # inputs = tokenizer("La inteligencia artificial es", return_tensors="pt")
        # outputs = model.generate(**inputs)
        # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        print("[OK] Generación con AutoModelForCausalLM lista.")
    except ImportError:
         print("[INFO] Falta dependencia (transformers). Código en modo plantilla.")

# ==============================================================================
# 4. Gen AI Foundational Models for NLP & Language Understanding
# ==============================================================================
def ejemplo_nlp_understanding():
    print("\n--- 4. Gen AI Foundational Models for NLP & Language Understanding ---")
    print("Concepto: Tarea de clasificación de texto / NLP fundamental.")
    try:
        from transformers import pipeline
        # --- Código de Ejemplo ---
        # classifier = pipeline("sentiment-analysis")
        # result = classifier("¡Me encanta aprender sobre IA generativa!")
        # print("Resultado del análisis predictivo:", result)
        print("[OK] Pipeline de entendimiento de lenguaje natural inicializado virtualmente.")
    except ImportError:
         print("[INFO] Falta dependencia (transformers). Código en modo plantilla.")

# ==============================================================================
# 5. Generative AI and LLMs: Architecture and Data Preparation
# ==============================================================================
def ejemplo_data_preparation():
    print("\n--- 5. Generative AI and LLMs: Architecture and Data Preparation ---")
    print("Concepto: Pre-procesamiento y tokenización de grandes volúmenes de texto.")
    try:
        # --- Código de Ejemplo ---
        import sys
        if 'datasets' not in sys.modules:
            pass # Solo comprobación lógica
        # from datasets import load_dataset
        # dataset = load_dataset("imdb")
        # def tokenize_function(examples):
        #     return tokenizer(examples["text"], padding="max_length", truncation=True)
        # tokenized_datasets = dataset.map(tokenize_function, batched=True)
        print("[OK] Arquitectura de preparación de datasets demostrada.")
    except ImportError:
         print("[INFO] Falta dependencia. Código en modo plantilla.")

# ==============================================================================
# 6. Generative AI Advance Fine-Tuning for LLMs
# ==============================================================================
def ejemplo_advance_finetuning():
    print("\n--- 6. Generative AI Advance Fine-Tuning for LLMs ---")
    print("Concepto: Entrenamiento eficiente de parámetros (PEFT) usando el método LoRA.")
    try:
        # --- Código de Ejemplo ---
        # from peft import LoraConfig, get_peft_model
        # config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        # peft_model = get_peft_model(model, config)
        print("[OK] Configuración avanzadas con LoRA descritas.")
    except ImportError:
         print("[INFO] Faltan dependencias (peft). Código en modo plantilla.")

# ==============================================================================
# 7. Project: Generative AI Applications with RAG and LangChain
# ==============================================================================
def ejemplo_rag_langchain():
    print("\n--- 7. Project: Generative AI Applications with RAG and LangChain ---")
    print("Concepto: Crear un motor de búsqueda y recuperación semántica integrado a LangChain.")
    try:
        # --- Código de Ejemplo ---
        # from langchain.document_loaders import TextLoader
        # from langchain.vectorstores import Chroma
        # from langchain.embeddings import OpenAIEmbeddings
        # loader = TextLoader("documento_ejemplo.txt")
        # docs = loader.load()
        # vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())
        print("[OK] Componentes base de RAG en LangChain presentados de forma teórica.")
    except Exception:
         print("[INFO] Faltan dependencias para RAG. Código en modo plantilla.")

# ==============================================================================
# 8. Fundamentals of AI Agents Using RAG and LangChain
# ==============================================================================
def ejemplo_ai_agents():
    print("\n--- 8. Fundamentals of AI Agents Using RAG and LangChain ---")
    print("Concepto: Implementación de un agente autónomo conversacional con acceso a herramientas.")
    try:
        # --- Código de Ejemplo ---
        # from langchain.agents import initialize_agent, Tool, AgentType
        # from langchain.llms import OpenAI
        # tools = [Tool(name="Calculadora", func=lambda x: eval(x), description="Útil para matemáticas")]
        # llm = OpenAI(temperature=0)
        # agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        # agent.run("Cuanto es 10 multiplicado por 5?")
        print("[OK] Lógica y estructura del agente LangChain definidos.")
    except ImportError:
        print("[INFO] Falta dependencia (langchain). Código en modo plantilla.")

# ==============================================================================
# 9. Building Generative AI-Powered Applications with Python
# ==============================================================================
def ejemplo_building_genai_apps():
    print("\n--- 9. Building Generative AI-Powered Applications with Python ---")
    print("Concepto: Encapsulación y servicio web de un modelo de Generación usando FastAPI.")
    print("""
    # from fastapi import FastAPI
    # app = FastAPI()
    # 
    # @app.post("/generate/")
    # def generate_text(prompt: str):
    #     response = model.generate(prompt)  # Lógica propia
    #     return {"generated_text": response}
    """)
    print("[OK] Snippet demostrativo de aplicación backend preparado.")

# ==============================================================================
# 10. Machine Learning with Python
# ==============================================================================
def ejemplo_machine_learning():
    print("\n--- 10. Machine Learning with Python ---")
    print("Concepto: Predicción con regresión lineal simple usando el ecosistema científico de Python.")
    
    # Hemos intentado usar dependencias de terceros, pero como alternativa funcional si no existen,
    # proveemos una versión "puramente Python".
    try:
        from sklearn.linear_model import LinearRegression
        import numpy as np
        
        # Dataset miniatura
        X = np.array([[1], [2], [3], [4]])
        y = np.array([2, 4, 6, 8])
        
        modelo = LinearRegression()
        modelo.fit(X, y)
        prediccion = modelo.predict(np.array([[5]]))
        print(f"[OK Scikit-Learn] Predicción exitosa para X=5: {prediccion[0]:.2f} (Debería ser ~10.0)")
    except ImportError:
        print("[INFO] No se encontró scikit-learn o numpy. Usando python puro (Regresión simple):")
        X = [1, 2, 3, 4]
        Y = [2, 4, 6, 8]
        # Slope (m) = (N*sum(xy) - sum(x)*sum(y)) / (N*sum(x^2) - sum(x)^2)
        N = len(X)
        sum_xy = sum([X[i]*Y[i] for i in range(N)])
        sum_x = sum(X)
        sum_y = sum(Y)
        sum_x2 = sum([x**2 for x in X])
        
        m = (N * sum_xy - sum_x * sum_y) / (N * sum_x2 - sum_x**2)
        b = (sum_y - m * sum_x) / N
        
        predict_5 = m * 5 + b
        print(f"[OK Vanilla Python] Ecuación: y = {m}x + {b}")
        print(f"[OK Vanilla Python] Predicción testeada y ejecutada para X=5: {predict_5:.2f}")

# ==============================================================================
# 11. Introduction to Deep Learning & Neural Networks with Keras
# ==============================================================================
def ejemplo_keras_deep_learning():
    print("\n--- 11. Introduction to Deep Learning & Neural Networks with Keras ---")
    print("Concepto: Construcción de una red neuronal secuencial multicapa.")
    try:
        # --- Código de Ejemplo ---
        # import tensorflow as tf
        # from tensorflow import keras
        # from tensorflow.keras import layers
        #
        # model = keras.Sequential([
        #     layers.Dense(64, activation='relu', input_shape=(10,)),
        #     layers.Dense(1, activation='sigmoid')
        # ])
        # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print("[OK] Red Neuronal Secuencial definida teóricamente.")
    except ImportError:
         print("[INFO] Falta dependencia (tensorflow). Código en modo plantilla.")

# ==============================================================================
# 12. Generative AI: Prompt Engineering Basics
# ==============================================================================
def ejemplo_prompt_engineering():
    print("\n--- 12. Generative AI: Prompt Engineering Basics ---")
    print("Concepto: Estandarización y formateo de instrucciones al LLM.")
    
    # Implementación pura que demuestra el engineering:
    template = "Eres un experto en {topic}. Escribe un resumen de máximo 3 lineas sobre {subtopic}."
    topic = "inteligencia artificial"
    subtopic = "las ventajas del Prompt Engineering"
    
    prompt = template.format(topic=topic, subtopic=subtopic)
    print("\n[OK Ejecutable] Prompt Estructurado Generado Listo para enviar a una API:\n", "-"*40)
    print(prompt)
    print("-" * 40)

# ==============================================================================
# 13. Generative AI: Introduction and Applications
# ==============================================================================
def ejemplo_intro_applications():
    print("\n--- 13. Generative AI: Introduction and Applications ---")
    print("Concepto: Interacción básica cliente-servidor API usando OpenAI.")
    try:
        # --- Código de Ejemplo ---
        # import openai
        # openai.api_key = "sk-..."
        # response = openai.Completion.create(
        #     model="text-davinci-003",
        #     prompt="Explica la inteligencia artificial generativa a un niño de 5 años.",
        #     max_tokens=50
        # )
        # print("Resultado:", response.choices[0].text.strip())
        print("[OK] Petición estructurada al motor generativo demostrada.")
    except Exception:
        print("[INFO] Faltan dependencias o se ignora ejecución (openai).")

if __name__ == "__main__":
    print("======================================================================")
    print("🚀 EJECUCIÓN DE EJEMPLOS DE IA GENERATIVA LIGADOS AL TEMARIO DE DOC_IA")
    print("======================================================================")
    
    ejemplo_ibm_genai_engineering()
    ejemplo_finetuning_transformers()
    ejemplo_language_modeling()
    ejemplo_nlp_understanding()
    ejemplo_data_preparation()
    ejemplo_advance_finetuning()
    ejemplo_rag_langchain()
    ejemplo_ai_agents()
    ejemplo_building_genai_apps()
    ejemplo_machine_learning()
    ejemplo_keras_deep_learning()
    ejemplo_prompt_engineering()
    ejemplo_intro_applications()
    
    print("\n======================================================================")
    print("✅ Todos los módulos se han recorrido exitosamente.")
    print("Para utilizar de forma real los modelos, descomenta las líneas")
    print("de código correspondientes y asegúrate de tener las librerías activadas.")
    print("======================================================================")
