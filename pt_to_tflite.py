import torch
import onnx
import tensorflow as tf
from onnx_tf.backend import prepare
from ultralytics import YOLO
import os


def infer_input_size(pytorch_model_path):
    """
    PyTorch 모델의 입력 크기를 자동으로 추론합니다.
    Args:
        pytorch_model_path (str): PyTorch 모델 경로 (.pt)
    Returns:
        tuple: 추론된 입력 크기 (예: (1, 3, 640, 640))
    """
    print("Inferring input size from YOLOv8 model...")

    # YOLO 모델 로드
    model = YOLO(pytorch_model_path)

    # 입력 크기 추론
    imgsz = model.model.yaml.get('imgsz', 640)  # 기본 입력 크기 640x640
    input_size = (1, 3, imgsz, imgsz)

    print(f"Inferred input size: {input_size}")
    return input_size


def pytorch_to_tflite_auto(pytorch_model_path, onnx_model_path, tf_model_path, tflite_model_path):
    """
    PyTorch → ONNX → TensorFlow → TFLite 변환을 자동화하는 함수

    Args:
        pytorch_model_path (str): PyTorch 모델 경로 (.pt)
        onnx_model_path (str): ONNX 모델 저장 경로
        tf_model_path (str): TensorFlow 모델 저장 경로
        tflite_model_path (str): TFLite 모델 저장 경로

    Returns:
        str: 변환 완료된 TFLite 모델 경로
    """
    # Step 1: PyTorch → ONNX 변환
    print("Converting PyTorch model to ONNX format...")
    input_size = infer_input_size(pytorch_model_path)
    model = YOLO(pytorch_model_path).model  # 모델의 PyTorch 객체 가져오기
    model.eval()

    dummy_input = torch.randn(*input_size)  # 추론된 입력 크기 사용
    torch.onnx.export(model, dummy_input, onnx_model_path, export_params=True,
                      opset_version=11, do_constant_folding=True,
                      input_names=['input'], output_names=['output'])
    print(f"ONNX model saved to {onnx_model_path}")

    # Step 2: ONNX → TensorFlow 변환
    print("Converting ONNX model to TensorFlow format...")
    onnx_model = onnx.load(onnx_model_path)
    tf_rep = prepare(onnx_model)  # ONNX 모델을 TensorFlow로 변환
    tf_rep.export_graph(tf_model_path)
    print(f"TensorFlow model saved to {tf_model_path}")

    # Step 3: TensorFlow → TFLite 변환
    print("Converting TensorFlow model to TFLite format...")
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    tflite_model = converter.convert()

    # TFLite 모델 저장
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {tflite_model_path}")

    return tflite_model_path


# 실행 예제
if __name__ == "__main__":
    pytorch_model_path = "best.pt"             # PyTorch 모델 경로
    onnx_model_path = "best.onnx"              # ONNX 모델 저장 경로
    tf_model_path = "best_tf"                  # TensorFlow 모델 저장 경로
    tflite_model_path = "best.tflite"          # TFLite 모델 저장 경로

    # 경로 확인 및 디렉토리 생성
    os.makedirs(tf_model_path, exist_ok=True)

    # 변환 실행
    tflite_model_path = pytorch_to_tflite_auto(pytorch_model_path, onnx_model_path, tf_model_path, tflite_model_path)
    print(f"Conversion completed. TFLite model saved at: {tflite_model_path}")
