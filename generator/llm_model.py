import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional

class NeMoLLM:
    """
    GPU-accelerated generator using Hugging Face AutoModel and AutoTokenizer.
    Uses CUDA if available for inference.
    """
    def __init__(self, model_name: str = "distilgpt2"):
        # Select device (GPU if available)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Ensure tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def generate(
        self,
        prompts: List[str],
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = 256,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
    ) -> List[str]:
        """
        Generate text for a batch of prompts on GPU or CPU.
        Accepts either max_length (total length) or max_new_tokens.
        Returns a list of generated strings.
        """
        # Tokenize inputs with padding and truncation
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Build generation kwargs
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = max_new_tokens
        elif max_length is not None:
            gen_kwargs["max_length"] = max_length

        # Generate with no_grad
        with torch.no_grad():
            outputs = self.model.generate(**gen_kwargs)

        # Decode outputs
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)