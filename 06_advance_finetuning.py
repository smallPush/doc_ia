"""
Tema 6: Generative AI Advance Fine-Tuning for LLMs
Concepto: Entrenamiento con Parámetros Eficientes (PEFT) inyectando adaptadores de bajo rango (LoRA).
"""
def main():
    print("--- 6. Advance Fine-Tuning: PEFT (LoRA) ---")
    try:
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, TaskType
        import torch
        
        model_name = "gpt2"
        print(f"-> Cargando modelo base monolítico '{model_name}'...")
        # En la vida real, cargar Llama-3 o Falcon gasta mucha VRAM (Gigabytes) y no cabe.
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Contamos cuántas neuronas tiene el modelo congelado.
        total_params = sum(p.numel() for p in base_model.parameters())
        print(f"Total de Matrices Internas (Congeladas): {total_params:,} pesos.")
        
        print("\n-> Aplicando inyección de adaptadores LoRA (Low-Rank Adaptation)...")
        # LoRA inserta "minimatrices" aprendibles encima de las matrices congeladas del modelo base.
        # En vez de 100 millones de pesos a aprender, entrenaremos apenas un par de cientos de miles.
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            r=8,              # Ranking del adaptador. A más rango r, más "inteligente/flexible" pero más pesado
            lora_alpha=32,    # Factor de escalado
            lora_dropout=0.1  # Resistencia al sobreentrenamiento (Overfitting dropout)
        )
        
        # Envuelve la red de PyTorch y fusiona el adaptador dinámico
        peft_model = get_peft_model(base_model, lora_config)
        
        # Imprime los resultados económicos
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        print(f"Nuevos Parámetros Entrenables (LoRA): {trainable_params:,} pesos líquidos.")
        print(f"Porcentaje REAL del modelo a recalibrar: {100 * trainable_params / total_params:.4f}% del total.")
        
        base_model_memory = total_params * 4 / (1024**2) # Usando float32 (4 bytes) -> a Megabytes
        peft_memory = trainable_params * 4 / (1024**2)
        
        print(f"\n[ESTIMACIÓN] RAM VRAM requerida antes -> ~{base_model_memory:.2f} MB")
        print(f"[ESTIMACIÓN] RAM VRAM requerida (Lora Gradientes) -> ~{peft_memory:.2f} MB")
        print("\n[Éxito] El LLM ahora está customizado con LoRA listo para ser lanzado al Trainer() de Hugging Face.")
        
    except ImportError as e:
         print(f"[ERROR] Dependencias faltantes: {e}")
         print("Instala ejecutando: pip install peft transformers torch")

if __name__ == "__main__":
    main()
