from transformers import pipeline

class NeMoLLM:
    """
    Fallback generator using Hugging Face Transformers.
    Named NeMoLLM so app.py need not change.
    """
    def __init__(self, model_name: str = "distilgpt2", device: int = -1):
        # device=-1 for CPU/MPS, >=0 for GPU id
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            device=device,
        )

    def generate(
        self,
        prompts: list[str],
        max_length: int = 256,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
    ) -> list[str]:
        """
        Generate text for each prompt.
        Returns a list of generated strings.
        """
        outputs = self.generator(
            prompts,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            return_full_text=False,
        )
        # pipeline returns List[List[Dict]]
        return [out[0]["generated_text"] for out in outputs]
