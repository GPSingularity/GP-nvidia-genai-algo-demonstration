import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn

def export_to_onnx(
    model_name: str = "distilgpt2",
    output_path: str = "onnx/model.onnx",
    opset: int = 13,
):
    """
    Export a Hugging Face causal LM to ONNX format, wrapping to bypass past_key_values.

    Args:
        model_name: HF model identifier
        output_path: path to save ONNX
        opset: ONNX opset version
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    # Disable caching and switch off return_dict
    model.config.use_cache = False
    model.config.return_dict = False

    # Define a wrapper that only returns logits
    class GPT2Wrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            # Forward pass with no caching, returns tuple (logits, ...) when return_dict=False
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=False,
            )
            # outputs[0] are logits
            return outputs[0]

    wrapper = GPT2Wrapper(model)

    # Prepare dummy inputs
    dummy = tokenizer("Hello, ONNX!", return_tensors="pt")
    input_ids = dummy["input_ids"]
    attention_mask = dummy.get("attention_mask")

    # Export
    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"},
        },
        opset_version=opset,
    )
    print(f"ONNX model saved to {output_path}")

if __name__ == "__main__":
    export_to_onnx()
