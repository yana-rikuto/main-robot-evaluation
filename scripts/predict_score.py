import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# 動画をフレームに分割する関数（preprocess.pyと同じ関数を使用）
def load_video_frames(video_path, frame_size=(64, 64), num_frames=30):
    """
    動画ファイルを読み込み、指定した数のフレームをリサイズして取得する関数
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while cap.isOpened() and count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)
        frames.append(frame)
        count += 1

    cap.release()
    frames = np.array(frames)
    return frames

# カスタム損失関数の指定
custom_objects = {'mse': tf.keras.losses.MeanSquaredError()}

# モデルの読み込み
print("モデルを読み込んでいます...")
model = load_model('./models/robot_dance_model.h5', custom_objects=custom_objects)
print("モデルの読み込みが完了しました。")

# 新しい動画ファイルのパス
new_video_path = './data/test/new_video.mp4'

# 動画をフレームに分割して前処理
new_video_frames = load_video_frames(new_video_path)
if new_video_frames.shape[0] != 30:
    print("エラー: 動画のフレーム数が不足しています。")
    exit()

# モデル入力の形式に合わせて次元を追加
new_video_frames = np.expand_dims(new_video_frames, axis=0)

# スコアの予測
predicted_scores = model.predict(new_video_frames)
print(f"予測されたスコア: {predicted_scores[0]}")
