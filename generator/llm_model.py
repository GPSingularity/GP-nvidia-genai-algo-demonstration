import torch
from pytorch_lightning import Trainer
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from typing import List, Optional

class NeMoLLM:
    """
    Wrapper for an NVIDIA NeMo Megatron GPT model for text generation.

    Uses a minimal PyTorch-Lightning Trainer to satisfy Megatron requirements.
    Defaults to the small "megatron_gpt_345m" checkpoint.
    """
    def __init__(
        self,
        model_name: str = "megatron_gpt_345m",
        device: Optional[str] = None,
    ):
        # Select device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Build minimal Lightning Trainer for inference
        trainer = Trainer(
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            max_epochs=1,
            enable_progress_bar=False,
        )

        # Load pretrained Megatron GPT model with trainer
        try:
            self.model: MegatronGPTModel = MegatronGPTModel.from_pretrained(
                pretrained_model_name=model_name,
                trainer=trainer,
            )
        except FileNotFoundError:
            available = MegatronGPTModel.list_available_models()
            raise FileNotFoundError(
                f"Model '{model_name}' not found. Available: {available}"
            )

        # Switch to eval mode and move to device
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
    parser = argparse.ArgumentParser(
        description="NeMo Megatron GPT inference demo with PyTorch-Lightning Trainer"
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="Input prompt for generation"
    )
    parser.add_argument(
        "--model-name", type=str, default="megatron_gpt_345m",
        help="NeMo model checkpoint name"
    )
    parser.add_argument(
        "--max-length", type=int, default=256, help="Max generation tokens"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    args = parser.parse_args()

    nemo_llm = NeMoLLM(model_name=args.model_name)
    result = nemo_llm.generate([
        args.prompt
    ], max_length=args.max_length, temperature=args.temperature)
    print(f"\n=== Generated Text ===\n{result[0]}")