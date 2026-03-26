"""
Tema 3: Generative AI Language Modeling with Transformers
Mejoras: soporte multilingüe, detección de dispositivo, configuración
centralizada, manejo de errores robusto y logging claro.
"""
import logging
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuración centralizada. Fácil de modificar o serializar a JSON/YAML."""
    # Modelo: distilgpt2 (inglés, ligero) o "PlanTL-GOB-ES/gpt2-base-bne" (español nativo)
    model_name: str = "distilgpt2"
    prompt: str = "In the future, generative artificial intelligence will"
    max_new_tokens: int = 60
    temperature: float = 0.8
    num_return_sequences: int = 2
    do_sample: bool = True
    # None = detectar automáticamente (CUDA → MPS → CPU)
    device: int | None = None


def resolve_device() -> int:
    """Devuelve el índice de dispositivo óptimo disponible."""
    try:
        import torch
        if torch.cuda.is_available():
            log.info("Dispositivo: CUDA (GPU)")
            return 0
        if torch.backends.mps.is_available():    # Apple Silicon
            log.info("Dispositivo: MPS (Apple Silicon)")
            return 0
    except ImportError:
        pass
    log.info("Dispositivo: CPU")
    return -1


def build_pipeline(cfg: GenerationConfig):
    """Construye el pipeline reutilizable de text-generation."""
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
    import torch

    log.info(f"Cargando modelo: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float32,
    )
    device = cfg.device if cfg.device is not None else resolve_device()
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )


def generate(generator, cfg: GenerationConfig) -> list[str]:
    """Ejecuta la inferencia y devuelve solo el texto nuevo (sin el prompt)."""
    results = generator(
        cfg.prompt,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        num_return_sequences=cfg.num_return_sequences,
        do_sample=cfg.do_sample,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    # Extraer solo el texto generado, excluyendo el prompt original
    prompt_len = len(cfg.prompt)
    return [r["generated_text"][prompt_len:].strip() for r in results]


def main():
    log.info("=== Language Modeling with Transformers ===")

    cfg = GenerationConfig()
    # Para español, descomenta:
    # cfg = GenerationConfig(
    #     model_name="PlanTL-GOB-ES/gpt2-base-bne",
    #     prompt="En el futuro, la inteligencia artificial generativa logrará",
    # )

    try:
        generator = build_pipeline(cfg)
        log.info(f"Prompt: '{cfg.prompt}'")
        outputs = generate(generator, cfg)

        print("\n=== TEXTO GENERADO ===")
        for i, text in enumerate(outputs, 1):
            print(f"\n[Variante {i}]\n{text}\n{'─' * 40}")

    except ImportError as e:
        log.error(f"Dependencias faltantes: {e}")
        log.error("Instala con: pip install torch transformers")
    except OSError as e:
        log.error(f"Error al descargar el modelo (¿sin conexión?): {e}")
    except Exception as e:
        log.exception(f"Error inesperado: {e}")


if __name__ == "__main__":
    main()