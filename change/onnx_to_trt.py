import os
import argparse

def convert_onnx_to_trt(onnx_path="onnx/crnn_model.onnx", trt_path="trt/crnn_model.trt", fp16=True, workspace=4096):
    os.makedirs(os.path.dirname(trt_path), exist_ok=True)
    fp16_flag = "--fp16" if fp16 else ""
    cmd = f"/usr/src/tensorrt/bin/trtexec --onnx={onnx_path} --saveEngine={trt_path} {fp16_flag} --workspace={workspace}"
    print(f"Running TensorRT build command:\n{cmd}")
    os.system(cmd)
    print(f"TensorRT engine saved: {trt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="onnx/crnn_model.onnx", help="Input ONNX file path")
    parser.add_argument("--trt", default="trt/crnn_model.trt", help="Output TensorRT engine file path")
    parser.add_argument("--no_fp16", action="store_true", help="Disable FP16 mode")
    args = parser.parse_args()

    convert_onnx_to_trt(
        onnx_path=args.onnx,
        trt_path=args.trt,
        fp16=(not args.no_fp16),
        workspace=4096
    )
