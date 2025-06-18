import os
import tensorrt as trt


def build_engine(
    onnx_path: str = "onnx/model.onnx",
    engine_path: str = "trt/model.plan",
    workspace_size: int = 1 << 30,  # 1GB
    fp16: bool = True,
):
    """
    Build and serialize a TensorRT engine from an ONNX file.

    Args:
        onnx_path: Path to the ONNX model
        engine_path: Output path for the TensorRT engine
        workspace_size: GPU workspace size in bytes
        fp16: Enable FP16 precision
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)

    # Initialize TensorRT logger and builder
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse the ONNX model
    with open(onnx_path, "rb") as model_file:
        if not parser.parse(model_file.read()):
            for error in parser.errors:
                print(f"ONNX parse error: {error}")
            raise RuntimeError("Failed to parse ONNX model.")

    # Configure builder settings
    config = builder.create_builder_config()
    # Set workspace size
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Add optimization profile for dynamic shapes (batch_size=1, seq_len up to 64)
    profile = builder.create_optimization_profile()
    # 'input_ids' and 'attention_mask' have shape [batch, seq_len]
    profile.set_shape("input_ids", (1, 1), (1, 16), (1, 64))
    profile.set_shape("attention_mask", (1, 1), (1, 16), (1, 64))
    config.add_optimization_profile(profile)

    # Build the engine (TensorRT 8+)
    # Prefer build_engine if available, else build_serialized_network
    if hasattr(builder, 'build_engine'):
        engine = builder.build_engine(network, config)
        engine_bytes = engine.serialize()
    else:
        engine_bytes = builder.build_serialized_network(network, config)
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(engine_bytes)

    if engine_bytes is None:
        raise RuntimeError("Failed to build/serialize the TensorRT engine.")

    # Save engine bytes
    with open(engine_path, "wb") as f:
        f.write(engine_bytes)
    print(f"TensorRT engine saved to {engine_path}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Build serialized network (TensorRT 8+)
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("Failed to serialize the TensorRT engine.")

    # Deserialize engine
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError("Failed to deserialize the TensorRT engine.")

    # Save engine bytes
    with open(engine_path, "wb") as f:
        f.write(engine_bytes)
    print(f"TensorRT engine saved to {engine_path}")


if __name__ == "__main__":
    build_engine()
