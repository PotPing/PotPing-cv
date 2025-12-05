from pathlib import Path
import requests
import os
import json
from datetime import datetime
from ultralytics import YOLO
import cv2
import torch
import numpy as np
from typing import List
import shutil
from threading import Thread
from queue import Queue

DATA_YAML = "../config.yaml"
MODEL_PATH = "./outputs/weights/road_anomaly_yolo_clahe/weights/best.pt"
VIDEO_PATH = 0
RESULT_DIR = "results"
POST_URL = "http://localhost:8080/api/potholes/detection"  # JSON 전송 URL

# CLAHE 전처리된 데이터셋 경로
PREPROCESSED_DATA_DIR = "./data/road_defects_clahe"

os.makedirs(RESULT_DIR, exist_ok=True)
RESULT_JSON = os.path.join(RESULT_DIR, "detections.json")


# ------------------------------
# 비동기 저장 및 POST 처리
# ------------------------------
class AsyncSaver:
    """백그라운드에서 이미지 저장과 POST 요청 처리"""
    def __init__(self, max_queue_size=10):
        self.save_queue = Queue(maxsize=max_queue_size)
        self.post_queue = Queue(maxsize=max_queue_size)
        self.running = True
        
        # 저장 스레드
        self.save_thread = Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()
        
        # POST 스레드
        self.post_thread = Thread(target=self._post_worker, daemon=True)
        self.post_thread.start()
    
    def _save_worker(self):
        """이미지 저장 워커"""
        while self.running:
            try:
                item = self.save_queue.get(timeout=1)
                if item is None:
                    break
                
                ori_path, enh_path, det_path = item['paths']
                original_frame, enhanced_frame, annotated = item['frames']
                
                cv2.imwrite(ori_path, original_frame)
                cv2.imwrite(enh_path, enhanced_frame)
                cv2.imwrite(det_path, annotated)
                
                self.save_queue.task_done()
            except:
                pass
    
    def _post_worker(self):
        """POST 요청 워커"""
        while self.running:
            try:
                payload = self.post_queue.get(timeout=1)
                if payload is None:
                    break
                
                try:
                    requests.post(POST_URL, json=payload, timeout=3)
                except Exception as e:
                    print(f"POST 오류: {e}")
                
                self.post_queue.task_done()
            except:
                pass
    
    def save_async(self, paths, frames):
        """비동기 이미지 저장"""
        try:
            self.save_queue.put_nowait({
                'paths': paths,
                'frames': frames
            })
        except:
            print("⚠️  저장 큐가 가득 찼습니다. 프레임 스킵")
    
    def post_async(self, payload):
        """비동기 POST 요청"""
        try:
            self.post_queue.put_nowait(payload)
        except:
            print("⚠️  POST 큐가 가득 찼습니다. 요청 스킵")
    
    def shutdown(self):
        """스레드 종료"""
        self.running = False
        self.save_queue.put(None)
        self.post_queue.put(None)
        self.save_thread.join(timeout=2)
        self.post_thread.join(timeout=2)


# ------------------------------
# CLAHE 전처리 함수 (학습/추론 공통)
# ------------------------------
def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) 적용
    학습과 추론에 동일하게 사용되는 전처리 함수
    """
    # YCrCb 색공간으로 변환 (Y: 밝기, Cr/Cb: 색상)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y_channel, cr, cb = cv2.split(ycrcb)
    
    # CLAHE를 Y 채널에만 적용
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_eq = clahe.apply(y_channel)
    
    # 채널 합치고 BGR로 변환
    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    enhanced = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
    
    return enhanced


# ==================== 데이터셋 전처리 ====================
def preprocess_dataset(original_data_dir="../data/road_defects", output_dir=None):
    """전체 데이터셋에 CLAHE 적용"""
    if output_dir is None:
        output_dir = PREPROCESSED_DATA_DIR
    
    original_path = Path(original_data_dir)
    output_path = Path(output_dir)
    
    if not original_path.exists():
        print(f"원본 데이터 디렉토리가 없습니다: {original_data_dir}")
        return False
    
    print("데이터셋 전처리 시작 (CLAHE 적용)...")
    
    for split in ['train', 'val']:
        img_src = original_path / "images" / split
        img_dst = output_path / "images" / split
        img_dst.mkdir(parents=True, exist_ok=True)
        
        lbl_src = original_path / "labels" / split
        lbl_dst = output_path / "labels" / split
        lbl_dst.mkdir(parents=True, exist_ok=True)
        
        if not img_src.exists():
            print(f"{img_src} 디렉토리가 없습니다. 건너뜀.")
            continue
        
        # 이미지 전처리
        image_files = list(img_src.glob("*.[jp][pn]*g"))
        print(f"{split} 세트: {len(image_files)}장 처리 중...")
        
        processed_count = 0
        for img_path in image_files:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"읽기 실패: {img_path}")
                    continue
                
                enhanced = apply_clahe(img)
                output_img_path = img_dst / img_path.name
                cv2.imwrite(str(output_img_path), enhanced)
                processed_count += 1
            except Exception as e:
                print(f"처리 실패 {img_path}: {e}")
        
        print(f"{split} 이미지: {processed_count}/{len(image_files)}장 완료")
        
        # 라벨 파일 복사
        if lbl_src.exists():
            label_files = list(lbl_src.glob("*.txt"))
            copied_count = 0
            for lbl_path in label_files:
                try:
                    shutil.copy2(str(lbl_path), str(lbl_dst / lbl_path.name))
                    copied_count += 1
                except Exception as e:
                    print(f"라벨 복사 실패 {lbl_path}: {e}")
            print(f"{split} 라벨: {copied_count}/{len(label_files)}개 복사 완료")
    
    # YAML 파일 생성
    create_preprocessed_yaml(output_dir)
    print("전처리 완료!")
    return True


def create_preprocessed_yaml(preprocessed_dir):
    """
    전처리된 데이터셋용 YAML 파일 생성
    """
    yaml_path = Path(preprocessed_dir) / "data.yaml"
    
    yaml_content = f"""# CLAHE 전처리된 포트홀 데이터셋
path: {preprocessed_dir}
train: images/train
val: images/val

# Classes
nc: 1
names: ['pothole']
"""
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"YAML 생성: {yaml_path}")
    return str(yaml_path)


# 모델 로드
def load_model():
    """모델 로드 함수"""
    if not os.path.exists(MODEL_PATH):
        print("Pretrained YOLOv8n 모델로 시작합니다.")
        model = YOLO("yolov8n.pt")
    else:
        print(f"기존 학습된 모델 로드 중: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
    return model


# ------------------------------
# IoU 계산 함수
# ------------------------------
def bbox_iou(box1: List[float], box2: List[float]) -> float:
    """
    box = [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    box1_area = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    box2_area = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    denom = box1_area + box2_area - inter_area + 1e-9
    if denom <= 0.0:
        return 0.0
    return inter_area / denom


# ------------------------------
# 모델 훈련 (CLAHE 전처리된 데이터 사용)
# ------------------------------
def train_model(use_preprocessed=True):
    """
    CLAHE 전처리된 데이터로 학습
    """
    model = load_model()
    
    if use_preprocessed:
        # 전처리된 데이터가 없으면 생성
        if not Path(PREPROCESSED_DATA_DIR).exists():
            print("전처리된 데이터셋이 없습니다. 생성 중...")
            preprocess_dataset()
        
        data_yaml = str(Path(PREPROCESSED_DATA_DIR) / "data.yaml")
        print(f"CLAHE 전처리된 데이터로 학습: {data_yaml}")
    else:
        data_yaml = DATA_YAML
        print(f"원본 데이터로 학습: {data_yaml}")
    
    model.train(
        data=data_yaml,
        epochs=150,
        imgsz=640,
        batch=16,
        
        # 학습률 최적화
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_momentum=0.8,
        
        # 포트홀 특화 데이터 증강
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.2,
        scale=0.7,
        shear=3.0,
        perspective=0.0003,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.3,
        
        # 성능 최적화
        cache=True,
        workers=8,
        patience=50,
        
        name="road_anomaly_yolo_clahe",
        project="outputs/weights",
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=True
    )
    print("학습 완료!")


def optimize_conf_threshold(
    model,
    val_images_dir="./data/road_defects/images/val",
    val_labels_dir="./data/road_defects/labels/val",
    target_precision: float = 0.85,
    target_recall: float = 0.8,
    iou_thr: float = 0.5,
    conf_range: np.ndarray = None
) -> float:
    """최적 confidence threshold 찾기"""
    if conf_range is None:
        conf_range = np.arange(0.05, 0.96, 0.05)

    val_images_dir = Path(val_images_dir)
    val_labels_dir = Path(val_labels_dir)

    # 이미지 목록
    allowed_ext = [".png", ".jpg", ".jpeg"]
    images = sorted([p for p in val_images_dir.glob("*") if p.suffix.lower() in allowed_ext])
    if len(images) == 0:
        print("검증 이미지가 없습니다:", val_images_dir)
        return 0.5

    # GT 로드
    gt_dict = {}
    for lbl_path in val_labels_dir.glob("*.txt"):
        img_path = None
        for ext in allowed_ext:
            potential_path = val_images_dir / f"{lbl_path.stem}{ext}"
            if potential_path.exists():
                img_path = potential_path
                break
        
        if img_path is None or not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        boxes = []
        with open(lbl_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                xc, yc, w, h = map(float, parts[1:5])
                x1 = (xc - w / 2.0) * img_w
                y1 = (yc - h / 2.0) * img_h
                x2 = (xc + w / 2.0) * img_w
                y2 = (yc + h / 2.0) * img_h
                boxes.append([x1, y1, x2, y2, cls])
        gt_dict[img_path.name] = boxes

    best_f1 = -1.0
    best_conf = 0.5

    # threshold sweep
    for conf_thres in conf_range:
        precision_list = []
        recall_list = []

        for img_path in images:
            img_name = img_path.name
            gt_boxes = gt_dict.get(img_name, [])

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            results = model.predict(source=img, conf=conf_thres, verbose=False)
            if len(results) == 0:
                pred_boxes = []
            else:
                res = results[0]
                pred_boxes = []
                if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    confs = res.boxes.conf.cpu().numpy()
                    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
                    for (x1, y1, x2, y2), c, cid in zip(xyxy, confs, cls_ids):
                        pred_boxes.append([float(x1), float(y1), float(x2), float(y2), int(cid), float(c)])

            tp = 0
            fp = 0
            matched_gt_indices = set()

            for p in pred_boxes:
                p_bbox = p[:4]
                p_cls = int(p[4])
                best_iou = 0.0
                best_idx = -1
                for gi, g in enumerate(gt_boxes):
                    g_bbox = g[:4]
                    g_cls = int(g[4])
                    if g_cls != p_cls:
                        continue
                    iou = bbox_iou(p_bbox, g_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = gi
                if best_iou >= iou_thr and best_idx not in matched_gt_indices:
                    tp += 1
                    matched_gt_indices.add(best_idx)
                else:
                    fp += 1

            fn = max(0, len(gt_boxes) - len(matched_gt_indices))

            p = tp / (tp + fp + 1e-9) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn + 1e-9) if (tp + fn) > 0 else 0.0

            precision_list.append(p)
            recall_list.append(r)

        p_avg = float(np.mean(precision_list)) if len(precision_list) > 0 else 0.0
        r_avg = float(np.mean(recall_list)) if len(recall_list) > 0 else 0.0
        f1 = 2 * p_avg * r_avg / (p_avg + r_avg + 1e-9) if (p_avg + r_avg) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_conf = float(conf_thres)

    print(f"검증 결과 - best_conf: {best_conf:.3f}, best_f1: {best_f1:.4f}")
    return best_conf


def predict_folder(img_dir="./data/images/test", conf_threshold=0.5):
    """폴더 내 이미지 일괄 예측"""
    model = load_model()
    results_list = []
    
    if not os.path.exists(img_dir):
        print(f"이미지 폴더가 없습니다: {img_dir}")
        return
    
    images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
            if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    
    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print("이미지를 열 수 없음:", img_path)
            continue

        # CLAHE 적용
        enhanced_img = apply_clahe(img)

        results = model.predict(source=enhanced_img, conf=conf_threshold, verbose=False)
        if len(results) == 0:
            detections = []
        else:
            res = results[0]
            detections = []
            if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                cls_ids = res.boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), c, cid in zip(xyxy, confs, cls_ids):
                    detections.append({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": float(c),
                        "class_id": int(cid),
                        "class_name": res.names[int(cid)]
                    })

        results_list.append({
            "image_path": img_path,
            "detections": detections
        })

        print(f"{os.path.basename(img_path)} 처리 완료 ({len(detections)}개 탐지됨)")

    os.makedirs(os.path.dirname(RESULT_JSON), exist_ok=True)
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(results_list, f, ensure_ascii=False, indent=4)

    print(f"결과 JSON 저장 완료: {RESULT_JSON}")


def live_detect(
    video_path=0,
    conf_threshold=0.5,
    save_interval_seconds: float = 2.0,
    min_movement_threshold: float = 100.0
):
    """실시간 포트홀 검출 (비동기 처리로 화면 멈춤 방지)"""
    model = load_model()
    
    # 비동기 저장/전송 워커 시작
    async_saver = AsyncSaver()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("비디오를 열 수 없습니다.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = 0

    print("실시간 포트홀 감지 시작... (ESC 키로 종료)")
    print("CLAHE 전처리 적용 중...")
    print("비동기 처리 활성화 - 화면이 부드럽게 유지됩니다")

    live_results = []
    stat_conf_list = []
    total_detections = 0

    # 중복 저장 방지
    recent_detections = {}
    detection_id_counter = 0

    current_conf = float(conf_threshold)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            original_frame = frame.copy()
            enhanced_frame = apply_clahe(frame)

            # YOLO 탐지
            results = model.predict(source=enhanced_frame, conf=current_conf, verbose=False)
            annotated = enhanced_frame.copy()

            if len(results) > 0:
                res = results[0]
                try:
                    annotated = res.plot()
                except Exception:
                    if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
                        xyxy = res.boxes.xyxy.cpu().numpy()
                        confs = res.boxes.conf.cpu().numpy()
                        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
                        for (x1, y1, x2, y2), c, cid in zip(xyxy, confs, cls_ids):
                            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(annotated, f"{res.names[int(cid)]}:{c:.2f}", (int(x1), int(y1) - 6),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


            # 화면 표시 (이 부분은 절대 블로킹되면 안 됨!)
            disp = cv2.resize(annotated, (960, 960))
            cv2.imshow("YOLO Road Anomaly Detection", disp)

            video_timestamp = frame_count / fps
            current_time = datetime.now()
            frame_count += 1

            # 메모리 관리
            if len(stat_conf_list) > 1000:
                stat_conf_list = stat_conf_list[-1000:]

            # 결과 처리 (비동기)
            if len(results) > 0:
                res = results[0]
                if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    confs = res.boxes.conf.cpu().numpy()
                    cls_ids = res.boxes.cls.cpu().numpy().astype(int)

                    for (x1, y1, x2, y2), conf, cid in zip(xyxy, confs, cls_ids):
                        label = res.names[int(cid)]
                        
                        if label.lower() != "pothole":
                            continue

                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        
                        w = float(x2 - x1)
                        h = float(y2 - y1)
                        area = w * h
                        
                        if area < 4000:
                            severity = "LOW"
                        elif area < 15000:
                            severity = "MEDIUM"
                        else:
                            severity = "HIGH"

                        # 중복 체크
                        should_save = True
                        matched_detection_id = None
                        distance = 0
                        
                        for det_id, det_info in list(recent_detections.items()):
                            prev_time = det_info['time']
                            prev_cx, prev_cy = det_info['center']
                            
                            time_diff = (current_time - prev_time).total_seconds()
                            distance = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                            
                            if distance < min_movement_threshold and time_diff < save_interval_seconds:
                                should_save = False
                                matched_detection_id = det_id
                                break
                            
                            if time_diff > 5.0:
                                del recent_detections[det_id]
                        
                        if not should_save:
                            continue
                        
                        # 새로운 탐지만 저장 (비동기)
                        total_detections += 1
                        stat_conf_list.append(float(conf))
                        avg_confidence = float(np.mean(stat_conf_list)) if len(stat_conf_list) > 0 else float(conf)

                        timestamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{total_detections}"

                        ori_name = f"original_{timestamp}.jpg"
                        enh_name = f"enhanced_{timestamp}.jpg"
                        det_name = f"pothole_{timestamp}.jpg"

                        ori_path = os.path.join(RESULT_DIR, ori_name)
                        enh_path = os.path.join(RESULT_DIR, enh_name)
                        det_path = os.path.join(RESULT_DIR, det_name)

                        # 프레임 복사 (원본 보존)
                        save_original = original_frame.copy()
                        save_enhanced = enhanced_frame.copy()
                        save_annotated = annotated.copy()

                        # 비동기 저장 (블로킹 없음!)
                        async_saver.save_async(
                            paths=(ori_path, enh_path, det_path),
                            frames=(save_original, save_enhanced, save_annotated)
                        )
                        
                        detection_id_counter += 1
                        recent_detections[detection_id_counter] = {
                            'time': current_time,
                            'center': (cx, cy)
                        }

                        print(f"✅ 포트홀 감지! (conf={conf:.2f}, severity={severity}, area={area:.0f})")

                        payload = {
                            "video_timestamp": video_timestamp,
                            "severity": severity,
                            "status": "DETECTED",
                            "total_detections": total_detections,
                            "average_confidence": avg_confidence,
                            "images": {
                                "original": ori_name,
                                "processed": enh_name,
                                "detected": det_name
                            }
                        }

                        # 비동기 POST (블로킹 없음!)
                        async_saver.post_async(payload)
                        live_results.append(payload)


            # ESC 키로 종료
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    finally:
        # 정리 작업
        print("\n종료 중... 남은 작업 완료 대기")
        async_saver.shutdown()
        
        # JSON 저장
        json_path = os.path.join(RESULT_DIR, "live_detections.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(live_results, f, ensure_ascii=False, indent=4)
        print(f"실시간 탐지 JSON 저장 완료: {json_path}")
        print(f"총 저장된 이미지 세트: {total_detections}개")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="predict", 
                        choices=["preprocess", "train", "predict", "live-camera"])
    parser.add_argument("--target_precision", type=float, default=0.85)
    parser.add_argument("--target_recall", type=float, default=0.8)
    args = parser.parse_args()

    best_conf = 0.5

    if args.mode == "preprocess":
        preprocess_dataset()
        
    elif args.mode == "train":
        train_model(use_preprocessed=True)
        
        model = load_model()
        best_conf = optimize_conf_threshold(
            model,
            val_images_dir=f"{PREPROCESSED_DATA_DIR}/images/val",
            val_labels_dir=f"{PREPROCESSED_DATA_DIR}/labels/val",
            target_precision=args.target_precision,
            target_recall=args.target_recall
        )
        print(f"최적 confidence threshold: {best_conf}")

        with open("best_conf.json", "w") as f:
            json.dump({"best_conf": best_conf}, f)
        print("best_conf 저장 완료")

    elif args.mode == "predict":
        predict_folder(conf_threshold=0.5)

    elif args.mode == "live-camera":
        try:
            with open("best_conf.json", "r") as f:
                best_conf = json.load(f).get("best_conf", 0.5)
            print(f"저장된 best_conf 불러옴: {best_conf:.3f}")
        except Exception:
            best_conf = 0.5
            print("best_conf.json 없음, 기본값 0.5 사용")

        live_detect(
            video_path=VIDEO_PATH,
            conf_threshold=best_conf,
        )