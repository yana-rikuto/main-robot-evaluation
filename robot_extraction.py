import torch
import cv2
import numpy as np

# YOLOv5の事前学習モデルをロード
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# 動画を読み込む
cap = cv2.VideoCapture('dancemovie3.mp4')

# 初期設定
prev_center = None
speeds = []
rotations = []
frame_count = 0

# スピードを計算する関数（前フレームと現フレームの中心点の距離）
def calculate_speed(center, prev_center, fps):
    if prev_center is None:
        return 0.0
    distance = np.sqrt((center[0] - prev_center[0]) ** 2 + (center[1] - prev_center[1]) ** 2)
    speed = distance / fps  # 1ピクセル = 1メートルと仮定（調整可能）
    return speed

# バウンディングボックスの回転角度を計算（簡易的な例）
def calculate_rotation(bbox):
    x1, y1, x2, y2 = bbox[:4]  # 最初の4つの値のみ取得
    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi  # 角度を度で計算
    return angle

# 動画のFPSを取得
fps = cap.get(cv2.CAP_PROP_FPS)

# 5秒ごとのフレーム数を計算
seconds_per_segment = 5
frames_per_segment = int(fps * seconds_per_segment)

# 区間ごとの結果を保存するためのリスト
all_speeds = []
all_rotations = []

# 区間カウンター
segment_count = 1

# 各フレームを処理
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOモデルを使って物体を検出
    results = model(frame)

    # 検出された物体のうち、最も大きなバウンディングボックス（ロボットと仮定）
    detections = results.xyxy[0].cpu().numpy()
    if len(detections) > 0:
        bbox = max(detections, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
        x1, y1, x2, y2, confidence, class_id = bbox

        # バウンディングボックスの中心を計算
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        center = (cx, cy)

        # スピードと回転を計算
        speed = calculate_speed(center, prev_center, fps)
        rotation = calculate_rotation(bbox)

        speeds.append(speed)
        rotations.append(rotation)

        # 前のフレームの中心を更新
        prev_center = center

        # バウンディングボックスと中心をフレームに描画
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.circle(frame, (int(cx), int(cy)), 5, (255, 0, 0), -1)

        # フレームを表示
        cv2.imshow('YOLO Robot Detection', frame)
        
    # 'q'キーを押すと終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

    # 5秒ごとに区間を切り替え
    if frame_count % frames_per_segment == 0:
        # 平均スピードと回転角度を計算
        average_speed = np.mean(speeds) if speeds else 0.0
        average_rotation = np.mean(rotations) if rotations else 0.0

        # 結果を保存
        all_speeds.append(average_speed)
        all_rotations.append(average_rotation)

        print(f"区間 {segment_count}: 平均スピード: {average_speed:.4f} メートル/秒, 平均回転角度: {average_rotation:.2f} 度/フレーム")
        
        # 次の区間の準備
        speeds = []
        rotations = []
        segment_count += 1

# 動画のキャプチャを解放し、ウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()

# 最終結果の表示
print("\n--- 各区間の分析結果 ---")
for i, (speed, rotation) in enumerate(zip(all_speeds, all_rotations), 1):
    print(f"区間 {i}: 平均スピード: {speed:.4f} メートル/秒, 平均回転角度: {rotation:.2f} 度/フレーム")
