"""
Tema 2: Generative AI Engineering and Fine-Tuning Transformers

Concepto:
Demostrar el proceso completo de Fine-Tuning de un modelo pre-entrenado
de Hugging Face para adaptarlo a una tarea específica (ej: Clasificación de texto)
utilizando la abstracción `Trainer` y la librería `datasets`.
"""

import sys

def main():
    print("--- 2. Generative AI Engineering and Fine-Tuning Transformers ---")
    print("Ejemplo: Fine-tuning de DistilBERT para Análisis de Sentimientos.\n")

    try:
        # Importamos las dependencias necesarias
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer
        )
        import datasets

        print("[OK] Las librerías transformers, torch y datasets están instaladas.")

        # === 1. Definición del Modelo y Tokenizador ===
        model_name = "distilbert-base-uncased"
        print(f"-> Preparando modelo base: {model_name}")

        print("-> Descargando pesos del modelo y tokenizador...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Cargamos el modelo indicando que tendrá 2 etiquetas de salida (Positivo/Negativo)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        # === 2. Preparación del Dataset ===
        print("-> Descargando y tokenizando el dataset (IMDb)...")
        # Usamos un dataset pequeñísimo (100 ejemplos) para demostración; así corre rápido en CPU.
        dataset = datasets.load_dataset("imdb", split="train[:100]")

        # Función para transformar el texto en tokens numéricos del modelo
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        # Dividimos en set de entrenamiento (80) y validación (20)
        small_train_dataset = tokenized_datasets.select(range(80))
        small_eval_dataset = tokenized_datasets.select(range(80, 100))

        # === 3. Configuración del Entrenamiento (TrainingArguments) ===
        # Aquí definimos los hiperparámetros de aprendizaje
        training_args = TrainingArguments(
            output_dir="./resultados_modelo",
            eval_strategy="epoch",       # Evaluar al final de cada época
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=1,          # Solo 1 época para la demostración
            weight_decay=0.01,
            push_to_hub=False,           # No subir a HuggingFace Hub
        )

        # === 4. Bucle de Fine-Tuning usando Trainer ===
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=small_train_dataset,
            eval_dataset=small_eval_dataset,
        )

        print("\n=== INICIANDO FINE-TUNING ===")
        print("Entrenando el modelo en el dataset adaptado (puede tardar un minuto)...")
        trainer.train()

        print("\\n=== EVALUACIÓN ===")
        results = trainer.evaluate()
        print(f"Resultados métricos: {results}")

        # === 5. Guardado del modelo fine-tuneado ===
        print("\\n-> Guardando el modelo adaptado en local...")
        trainer.save_model("./mi_modelo_finetuneado")
        print("[Éxito] Fine-Tuning completado y modelo guardado en './mi_modelo_finetuneado'")
        '''

        print("\n[INFO] El código de fine-tuning real está dentro de un bloque comentado.")
        print("       Quita las triples comillas (''') para probar a entrenar un modelo con datos.")
        print("       (El algoritmo de ejemplo coge solo 100 reseñas de IMDb para que corra rápido sin GPU).")

    except ImportError as e:
        print(f"[ERROR] Faltan dependencias para ejecutar el entorno de Fine-Tuning: {e}")
        print("Asegúrate de ejecutar en terminal: pip install torch transformers datasets accelerate")

if __name__ == "__main__":
    main()
