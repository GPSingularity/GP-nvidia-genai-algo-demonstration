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
    # Set workspace size limit
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Build the engine
    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("Failed to build the TensorRT engine.")

    # Serialize and save
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())
    print(f"TensorRT engine saved to {engine_path}")


if __name__ == "__main__":
    build_engine()