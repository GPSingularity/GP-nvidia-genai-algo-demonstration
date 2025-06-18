import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

class NeMoLLM:
    """
    Fallback generator using Hugging Face AutoModel and AutoTokenizer for inference.
    CPU only. Avoids Gradio/pipeline multiprocessing that can segfault on macOS.
    """
    def __init__(self, model_name: str = "distilgpt2", device: str = None):
        # Select device (CPU or GPU if available)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompts: List[str],
        max_length: int = 256,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
    ) -> List[str]:
        """
        Generate text for a batch of prompts.
        Returns a list of generated strings.
        """
        # Tokenize inputs
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Generate with no_grad
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
            )
        # Decode outputs
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)