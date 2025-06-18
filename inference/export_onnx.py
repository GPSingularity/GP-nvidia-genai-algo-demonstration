import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def export_to_onnx(
    model_name: str = "distilgpt2",
    output_path: str = "onnx/model.onnx",
    opset: int = 13,
):
    """
    Export a Hugging Face causal LM to ONNX format.

    Args:
        model_name: model identifier from HF Hub
        output_path: where to save the ONNX model
        opset: ONNX opset version
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).eval()

    # Dummy inputs
    dummy_inputs = tokenizer(
        "Hello, ONNX!", return_tensors="pt"
    )
    input_ids = dummy_inputs["input_ids"]
    attention_mask = dummy_inputs.get("attention_mask")

    # Export to ONNX
    torch.onnx.export(
        model,
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