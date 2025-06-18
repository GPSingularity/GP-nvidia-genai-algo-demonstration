import torch
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from typing import List, Optional

class NeMoLLM:
    """
    Wrapper around an NVIDIA NeMo Megatron GPT model for text generation.
    """
    def __init__(
        self,
        model_name: str = "gpt-j-6B",
        device: Optional[str] = None,
    ):
        # Select device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load the pretrained NeMo Megatron GPT model
        self.model: MegatronGPTModel = MegatronGPTModel.from_pretrained(model_name)
        self.model.eval().to(self.device)

    def generate(
        self,
        prompts: List[str],
        max_length: int = 256,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
    ) -> List[str]:
        """
        Generate text for each prompt.
        Returns a list of generated strings.
        """
        return self.model.generate(
            input_prompts=prompts,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
