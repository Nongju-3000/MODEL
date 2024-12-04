import tensorflow as tf
from ultralytics import YOLO
import onnxruntime as ort
import numpy as np
from PIL import Image

# YOLO 모델 로드 (경로 수정)
model = YOLO('best.pt')

# TensorFlow Lite 모델 로드 함수
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# TensorFlow Lite 모델 추론 함수
def predict_tflite(interpreter, image_path):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 이미지 전처리
    image = Image.open(image_path).convert('RGB')
    input_height, input_width, input_channels = input_details[0]['shape'][1:4]  # 입력 텐서 크기 (1, H, W, C)에서 H, W, C 사용
    image = image.resize((input_width, input_height))
    input_data = np.array(image, dtype=np.float32).reshape(1, input_height, input_width, 3) / 255.0  # 정규화 (0~1)
    input_data = np.expand_dims(input_data, axis=0)  # (H, W, C) -> (1, H, W, C)

    # 입력 데이터 설정
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # 추론 수행
    interpreter.invoke()

    # 출력 데이터 가져오기
    output_data = interpreter.get_tensor(output_details[0]['index'])
    class_indices = np.argmax(output_data, axis=-1)  # 클래스 인덱스 추출
    print("TFLite 모델 클래스 번호:", class_indices)

# ONNX 모델 추론 함수
def predict_onnx(model_path, image_path):
    # ONNX Runtime 세션 생성
    ort_session = ort.InferenceSession(model_path)

    # 입력 이름과 모양 가져오기
    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    input_height, input_width = input_shape[2], input_shape[3]

    # 이미지 전처리
    image = Image.open(image_path).convert('RGB')
    image = image.resize((input_width, input_height))
    input_data = np.array(image, dtype=np.float32).transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
    input_data = np.expand_dims(input_data, axis=0)  # (C, H, W) -> (1, C, H, W)
    input_data = input_data / 255.0  # 정규화 (0~1)

    # 추론 수행
    outputs = ort_session.run(None, {input_name: input_data})
    class_indices = np.argmax(outputs[0], axis=-1)  # 클래스 인덱스 추출
    print("ONNX 모델 클래스 번호:", class_indices)

# 이미지 처리 함수
def process_image(image_path):
    print(f"처리 중: {image_path}")
    try:
        # YOLO 모델로 이미지 처리
        results = model.predict(source=image_path, save=True, project='./output', name='results')

        # 인식한 객체 클래스 번호 추출
        classes = results[0].boxes.cls.tolist()  # 클래스 번호 리스트
        class_numbers = [str(int(cls)) for cls in classes]  # 클래스 번호를 문자열로 변환

        # 클래스 번호 출력
        print("YOLO 클래스 번호:", " ".join(class_numbers))

        # ONNX 모델 로드 및 처리
        onnx_model_path = 'best.onnx'
        predict_onnx(onnx_model_path, image_path)

        # TensorFlow Lite 모델 로드 및 처리
        tflite_model_path = 'best.tflite'
        tflite_interpreter = load_tflite_model(tflite_model_path)
        predict_tflite(tflite_interpreter, image_path)

    except Exception as e:
        print(f"에러 발생: {e}")

# 테스트 이미지 처리
image_path = 'test_image.jpg'
process_image(image_path)
