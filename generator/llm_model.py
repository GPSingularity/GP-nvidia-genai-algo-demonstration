import os
from typing import List, Optional

import torch
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

class NeMoLLM:
    """
    Wrapper around an NVIDIA NeMo Megatron GPT model for text generation.
    """
    def __init__(
        self,
        model_name: str = "gpt-j-6B",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        # Choose device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load the pretrained NeMo Megatron GPT model
        self.model: MegatronGPTModel = MegatronGPTModel.from_pretrained(
            model_name=model_name,
            cache_dir=cache_dir,
        )
        self.model.eval()
        self.model.to(self.device)

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
        outputs = self.model.generate(
            input_prompts=prompts,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        return outputs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="gpt-j-6B")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    nemo_llm = NeMoLLM(model_name=args.model_name)
    result = nemo_llm.generate([args.prompt], max_length=args.max_length, temperature=args.temperature)
    print(result[0])
