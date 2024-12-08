import tensorflow as tf
from ultralytics import YOLO
import onnxruntime as ort
import torch
import numpy as np
from PIL import Image

# YOLO 모델 로드 (경로 수정)
yolo_model = YOLO('best.pt')
onnx_model_path = 'best.onnx'

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

def nms(boxes, scores, iou_threshold):
    """Non-Maximum Suppression (NMS) to filter overlapping boxes."""
    indices = []
    sorted_indices = np.argsort(scores)[::-1]

    while len(sorted_indices) > 0:
        current = sorted_indices[0]
        indices.append(current)
        if len(sorted_indices) == 1:
            break

        remaining = sorted_indices[1:]

        ious = compute_iou(boxes[current], boxes[remaining])
        sorted_indices = remaining[ious <= iou_threshold]

    return indices

def compute_iou(box, boxes):
    """Compute IoU between a box and a list of boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union = box_area + boxes_area - intersection
    return intersection / union

def predict_onnx(model_path, image_path, conf_threshold=0.2, iou_threshold=0.9):
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


    image.save("processed_image.jpg")
    print("전처리된 이미지를 'processed_image.jpg'로 저장했습니다.")

    # 추론 수행
    outputs = ort_session.run(None, {input_name: input_data})

    # ONNX 모델 출력 데이터
    print("ONNX 모델 출력 데이터:", outputs)
    print("ONNX 모델 출력 데이터 형상:", outputs[0].shape)

    # 박스, 점수, 클래스 데이터 분리
    boxes = outputs[0][:, :4]  # 상자 좌표
    scores = outputs[0][:, 4]  # 점수 (신뢰도)
    class_probs = outputs[0][:, 5:]  # 클래스 확률

    # 점수가 conf_threshold 이상인 상자 필터링
    valid_indices = np.where(scores > conf_threshold)[0]
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    class_probs = class_probs[valid_indices]

    # 각 박스의 클래스 선택
    class_indices = np.argmax(class_probs, axis=-1)

    # NMS 적용
    nms_indices = nms(boxes, scores, iou_threshold)
    final_boxes = boxes[nms_indices]
    final_scores = scores[nms_indices]
    final_classes = class_indices[nms_indices]

    # 결과 출력
    print(f"Detected {len(final_boxes)} objects:")
    for i, box in enumerate(final_boxes):
        print(f"Class: {final_classes[i]}, Confidence: {final_scores[i]:.2f}, Box: {box}")

    return final_boxes, final_scores, final_classes

def predict_with_yolo_preprocessing(onnx_model_path, yolo_model_path, image_path):
    # YOLO 모델 로드
    yolo_model = YOLO(yolo_model_path)

    # YOLO에서 전처리된 데이터를 추출
    results = yolo_model.predict(source=image_path, save=False)
    input_data_yolo = results[0].orig_img  # 전처리된 이미지를 numpy 배열로 가져옴
    input_data_yolo = np.expand_dims(input_data_yolo, axis=0).astype(np.float32)

    # ONNX 모델 로드
    ort_session = ort.InferenceSession(onnx_model_path)
    input_name = ort_session.get_inputs()[0].name

    # ONNX 추론 수행
    outputs = ort_session.run(None, {input_name: input_data_yolo})
    print("ONNX 모델 출력 데이터:", outputs)

    return outputs

# 이미지 처리 함수
def process_image(image_path, onnx_model_path=onnx_model_path):
    print(f"처리 중: {image_path}")
    try:
        # YOLO 모델로 이미지 처리
        results = yolo_model.predict(source=image_path, save=True, project='./output', name='results')
        #results = test_yolo_with_custom_preprocessing(image_path)

        predict_with_yolo_preprocessing(onnx_model_path, yolo_model, image_path)

        # 인식한 객체 클래스 번호 추출
        classes = results[0].boxes.cls.tolist()  # 클래스 번호 리스트
        class_numbers = [str(int(cls)) for cls in classes]  # 클래스 번호를 문자열로 변환

        # 클래스 번호 출력
        print("YOLO 클래스 번호:", " ".join(class_numbers))

        # ONNX 모델 로드 및 처리
        onnx_model_path = 'best.onnx'
        predict_onnx(onnx_model_path, image_path)

        ## TensorFlow Lite 모델 로드 및 처리
        #tflite_model_path = 'best.tflite'
        #tflite_interpreter = load_tflite_model(tflite_model_path)
        #predict_tflite(tflite_interpreter, image_path)

    except Exception as e:
        print(f"에러 발생: {e}")

# 테스트 이미지 처리
image_path = 'processed_image.jpg'
process_image(image_path)
