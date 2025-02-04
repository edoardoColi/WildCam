from ultralytics import YOLO
import tensorrt as trt

model = YOLO('/usr/src/ultralytics/models/Yolo/bestSanRossore.pt')          # [print(f"Layer {i}: {layer}") for i, layer in enumerate(MODEL.model.model)]

model.export(format="onnx")


# TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# def build_engine(onnx_fp):
#     with trt.Builder(TRT_LOGGER) as builder, \
#         builder.create_network(1) as network, \
#         trt.OnnxParser(network, TRT_LOGGER) as parser:

#         builder.max_workspace_size = 1 << 30
#         builder.fp16_mode = True

#         with open(onnx_fp, "rb") as model:
#             if not parser.parse(model.read()):
#                 print ("ERROR: Failed to parse the ONNX file")
#                 return None
#         return builder.build_cuda_engine(network)

# engine = build_engine("yolov8m.onnx")

# with open("yolov8m.trt", "wb") as f:
#     f.write(engine.serialize())
