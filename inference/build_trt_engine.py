import os
import tensorrt as trt

def build_engine(
    onnx_path: str = "onnx/model.onnx",
    engine_path: str = "trt/model.plan",
    workspace_size: int = 1 << 30,  # 1GB
    fp16: bool = True,
):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in parser.errors:
                print(f"ONNX parse error: {error}")
            raise RuntimeError("Failed to parse ONNX model")

    # Builder config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Optimization profile for batch_size=1 and seq_len 1–64
    profile = builder.create_optimization_profile()
    profile.set_shape("input_ids", (1, 1), (1, 16), (1, 64))
    profile.set_shape("attention_mask", (1, 1), (1, 16), (1, 64))
    config.add_optimization_profile(profile)

    # Always use build_serialized_network (TensorRT 8+)
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("Failed to serialize the TensorRT engine")

    # Write engine
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(engine_bytes)
    print(f"TensorRT engine saved to {engine_path}")

if __name__ == "__main__":
    build_engine()