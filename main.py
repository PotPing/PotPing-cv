from pathlib import Path
import requests
import os
import json
from datetime import datetime
from ultralytics import YOLO
import cv2
import torch
import numpy as np
from typing import List, Tuple

DATA_YAML = "config.yaml"
MODEL_PATH = "outputs/weights/road_anomaly_yolo2/weights/best.pt"
VIDEO_PATH = 0
RESULT_DIR = "outputs/results"

FRAME_URL = "http://your-backend.com/video/frame"    #백엔드가 영상 프레임을 보내주는 URL
POST_URL  = "http://your-backend.com/api/detection"  # JSON 전송 URL


os.makedirs(RESULT_DIR, exist_ok=True)
RESULT_JSON = os.path.join(RESULT_DIR, "detections.json")

#모델 로드
if not os.path.exists(MODEL_PATH):
    print("🚀 Pretrained YOLOv8n 모델로 시작합니다.")
    model = YOLO("yolov8n.pt")
else:
    print("✅ 기존 학습된 모델 로드 중...")
    model = YOLO(MODEL_PATH)

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
    box1_area = max(0.0, box1[2]-box1[0])* max(0.0, box1[3]-box1[1])
    box2_area = max(0.0, box2[2]-box2[0])*max(0.0, box2[3]-box2[1])

    denom = box1_area + box2_area - inter_area + 1e-9
    if denom <= 0.0:
        return 0.0
    return inter_area / denom

#모델 훈련
def train_model():
    model.train(
        data=DATA_YAML,
        epochs=30,
        imgsz=640,
        batch=8,
        name="road_anomaly_yolo",
        project="outputs/weights",
        device=0 if torch.cuda.is_available() else "cpu"
    )
    print("✅ 학습 완료!")

def optimize_conf_threshold(model, val_images_dir="./data/road_defects/images/val", val_labels_dir="./data/road_defects/labels/val", target_precision: float=0.85, target_recall: float=0.8, iou_thr: float = 0.5, conf_range: np.ndarray = None) -> float:
    if conf_range is None:
        conf_range = np.arange(0.05, 0.96, 0.05)

    val_images_dir = Path(val_images_dir)
    val_labels_dir = Path(val_labels_dir)

    # 이미지 목록: png 우선, 허용되는 확장자 처리
    allowed_ext = [".png", ".jpg", ".jpeg"]
    images = sorted([p for p in val_images_dir.glob("*") if p.suffix.lower() in allowed_ext])
    if len(images) == 0:
        print("⚠️ 검증 이미지가 없습니다:", val_images_dir)
        return 0.5

    # GT 로드
    gt_dict = {}
    for lbl_path in val_labels_dir.glob("*.txt"):
        image_name = lbl_path.stem + ".jpg"  # 너는 PNG 사용한다고 했으니 그대로
        img_path = val_images_dir / image_name
        if not img_path.exists():
            # 이미지가 없으면 skip
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
                # YOLO normalized -> pixel coordinates
                x1 = (xc - w / 2.0) * img_w
                y1 = (yc - h / 2.0) * img_h
                x2 = (xc + w / 2.0) * img_w
                y2 = (yc + h / 2.0) * img_h
                boxes.append([x1, y1, x2, y2, cls])
        gt_dict[image_name] = boxes

    best_f1 = -1.0
    best_conf = 0.5

    # --- threshold sweep ---
    for conf_thres in conf_range:
        precision_list = []
        recall_list = []

        for img_path in images:
            img_name = img_path.name
            # GT for this image
            gt_boxes = gt_dict.get(img_name, [])

            # model.predict에 numpy array 전달 -> ultralytics는 결과를 원본 크기 좌표로 반환함
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Predict with current confidence threshold
            results = model.predict(source=img, conf=conf_thres, verbose=False)
            # results is a list; take first (single image)
            if len(results) == 0:
                pred_boxes = []
            else:
                res = results[0]
                pred_boxes = []
                # boxes.xyxy is tensor (N,4), boxes.conf (N,), boxes.cls (N,)
                if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    confs = res.boxes.conf.cpu().numpy()
                    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
                    for (x1, y1, x2, y2), c, cid in zip(xyxy, confs, cls_ids):
                        pred_boxes.append([float(x1), float(y1), float(x2), float(y2), int(cid), float(c)])

            # Evaluate per-image (consider only class match for TP)
            tp = 0
            fp = 0
            matched_gt_indices = set()

            # For each prediction, find best GT match (same class) with IoU >= iou_thr
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

            # Per-image precision, recall (guard division)
            p = tp / (tp + fp + 1e-9) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn + 1e-9) if (tp + fn) > 0 else 0.0

            precision_list.append(p)
            recall_list.append(r)

        # 평균 precision/recall across images
        p_avg = float(np.mean(precision_list)) if len(precision_list) > 0 else 0.0
        r_avg = float(np.mean(recall_list)) if len(recall_list) > 0 else 0.0
        f1 = 2 * p_avg * r_avg / (p_avg + r_avg + 1e-9) if (p_avg + r_avg) > 0 else 0.0

        # 선택 기준: 기본은 F1 최댓값
        # 추가 옵션: 우선적으로 precision/recall 조건을 만족하는 threshold를 선호하도록 하려면 아래 주석 해제
        # if p_avg >= target_precision and r_avg >= target_recall and f1 > best_f1:
        if f1 > best_f1:
            best_f1 = f1
            best_conf = float(conf_thres)

    print(f"🎯 검증 결과 - best_conf: {best_conf:.3f}, best_f1: {best_f1:.4f}")
    return best_conf



# 
def predict_folder(img_dir="./data/images/test", conf_threshold=0.5):
    results_list = []
    images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print("⚠️ 이미지를 열 수 없음:", img_path)
            continue

        results = model.predict(source=img, conf=conf_threshold, verbose=False)
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

    print(f"✅ 결과 JSON 저장 완료: {RESULT_JSON}")
        

def live_detect(video_path=0, session_id="session_001", conf_threshold=0.5,
                dynamic_update: bool = False, update_interval_frames: int = 300,
                val_images_dir: str = "./data/road_defects/images/val", 
                val_labels_dir: str = "./data/road_defects/labels/val",
                save_interval_seconds: float = 2.0,  # 같은 위치는 2초에 1번만 저장
                min_movement_threshold: float = 100.0):  # 최소 이동 거리 (픽셀)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 비디오를 열 수 없습니다.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = 0

    print("🎥 실시간 포트홀 감지 시작... (ESC 키로 종료)")

    live_results = []
    stat_conf_list = []
    total_detections = 0

    # 중복 저장 방지를 위한 추적 딕셔너리
    recent_detections = {}  # {detection_id: {'time': timestamp, 'center': (cx, cy)}}
    detection_id_counter = 0

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    current_conf = float(conf_threshold)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = cv2.resize(frame, (640, 640))
        original_frame = frame.copy()

        # 히스토그램 평활화
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y_channel, cr, cb = cv2.split(ycrcb)
        y_eq = clahe.apply(y_channel)
        ycrcb_eq = cv2.merge((y_eq, cr, cb))
        enhanced_frame = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

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
                        cv2.putText(annotated, f"{res.names[int(cid)]}:{c:.2f}", (int(x1), int(y1)-6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        disp = cv2.resize(annotated, (960, 960))
        cv2.imshow("YOLO Road Anomaly Detection", disp)

        video_timestamp = frame_count / fps
        current_time = datetime.now()
        frame_count += 1

        # 메모리 관리: 최근 1000개만 유지
        if len(stat_conf_list) > 1000:
            stat_conf_list = stat_conf_list[-1000:]

        # 결과 처리
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

                    # 박스 중심점 계산
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

                    # 🔥 중복 저장 방지 로직
                    should_save = True
                    matched_detection_id = None
                    
                    # 기존 탐지와 비교
                    for det_id, det_info in list(recent_detections.items()):
                        prev_time = det_info['time']
                        prev_cx, prev_cy = det_info['center']
                        
                        # 시간 간격 체크
                        time_diff = (current_time - prev_time).total_seconds()
                        
                        # 거리 체크
                        distance = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                        
                        # 같은 포트홀로 판단: 거리가 가깝고 시간이 짧으면 저장 안 함
                        if distance < min_movement_threshold and time_diff < save_interval_seconds:
                            should_save = False
                            matched_detection_id = det_id
                            break
                        
                        # 오래된 탐지 기록 삭제 (5초 이상 지난 것)
                        if time_diff > 5.0:
                            del recent_detections[det_id]
                    
                    # 저장하지 않는 경우 (중복)
                    if not should_save:
                        print(f"⏭️  중복 탐지 스킵 (ID: {matched_detection_id}, 거리: {distance:.0f}px)")
                        continue
                    
                    # 🎯 새로운 탐지이거나 충분히 이동한 경우만 저장
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

                    # 이미지 저장
                    try:
                        cv2.imwrite(ori_path, original_frame)
                        cv2.imwrite(enh_path, enhanced_frame)
                        cv2.imwrite(det_path, annotated)
                        
                        # 탐지 기록 업데이트
                        detection_id_counter += 1
                        recent_detections[detection_id_counter] = {
                            'time': current_time,
                            'center': (cx, cy)
                        }
                        
                    except Exception as e:
                        print("⚠️ 이미지 저장 오류:", e)
                        continue

                    print(f"⚠️ 포트홀 감지! (conf={conf:.2f}, severity={severity}, area={area:.0f})")
                    print(f"📷 저장: {det_name}")

                    payload = {
                        "video_timestamp": video_timestamp,
                        "severity": severity,
                        "status": "DETECTED",
                        "total_detections": total_detections,
                        "average_confidence": avg_confidence,
                        "session_id": session_id,
                        "detection_center": {"x": float(cx), "y": float(cy)},
                        "images": {
                            "original": ori_name,
                            "processed": enh_name,
                            "detected": det_name
                        }
                    }

                    # POST 전송
                    try:
                        r = requests.post(POST_URL, json=payload, timeout=5)
                        print(f"📡 서버 POST 결과 → {r.status_code}")
                    except Exception as e:
                        print("❌ POST 오류:", e)

                    live_results.append(payload)

        # dynamic_update
        if dynamic_update and update_interval_frames > 0 and (frame_count % update_interval_frames) == 0:
            print("🔁 dynamic_update: threshold 재계산 중...")
            try:
                new_best = optimize_conf_threshold(model,
                                                val_images_dir=val_images_dir,
                                                val_labels_dir=val_labels_dir)
                if isinstance(new_best, float) and 0.0 < new_best < 1.0:
                    print(f"🔔 적용: threshold {current_conf:.3f} -> {new_best:.3f}")
                    current_conf = float(new_best)
            except Exception as e:
                print("⚠️ dynamic_update 중 오류:", e)

        if cv2.waitKey(1) == 27:
            break

    # JSON 저장
    json_path = os.path.join(RESULT_DIR, "live_detections.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(live_results, f, ensure_ascii=False, indent=4)
    print(f"✅ 실시간 탐지 JSON 저장 완료: {json_path}")
    print(f"📊 총 저장된 이미지 세트: {total_detections}개 (원본 x3 = {total_detections*3}장)")

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="predict", choices=["train", "predict", "live-camera"])
    parser.add_argument("--target_precision", type=float, default=0.85)
    parser.add_argument("--target_recall", type=float, default=0.8)
    parser.add_argument("--dynamic_update", action="store_true", help="live mode에서 주기적으로 threshold 재계산")
    parser.add_argument("--update_interval_frames", type=int, default=300, help="dynamic_update일 때 몇 프레임마다 재계산할지")
    args = parser.parse_args()

    best_conf = 0.5

    if args.mode == "train":
        train_model()
        # 학습 후 검증셋으로 confidence threshold 최적화
        best_conf = optimize_conf_threshold(
            model,
            val_images_dir="./data/road_defects/images/val",
            val_labels_dir="./data/road_defects/labels/val",
            target_precision=args.target_precision,
            target_recall=args.target_recall
        )
        print(f"✅ 학습 완료 후 최적 confidence threshold: {best_conf}")

        # best_conf 저장
        with open("best_conf.json", "w") as f:
            json.dump({"best_conf": best_conf}, f)
        print("💾 best_conf 저장 완료: best_conf.json")
    elif args.mode == "predict":
        predict_folder(conf_threshold=0.5)
    elif args.mode == "live-camera":
        # 저장된 best_conf 불러오기
        try:
            with open("best_conf.json", "r") as f:
                best_conf = json.load(f).get("best_conf", 0.5)
            print(f"🎯 저장된 best_conf 불러옴: {best_conf:.3f}")
        except Exception:
            best_conf = 0.5
            print("⚠️ best_conf.json 없음, 기본값 0.5 사용")

        live_detect(video_path=VIDEO_PATH, session_id="session_001", conf_threshold=best_conf,
                    dynamic_update=args.dynamic_update, update_interval_frames=args.update_interval_frames,
                    val_images_dir="./data/road_defects/images/val", val_labels_dir="./data/road_defects/labels/val")
