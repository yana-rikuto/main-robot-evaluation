# scripts/predict_score.py

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# 動画をフレームに分割する関数（preprocess.pyと同じ関数を使用）
def load_video_frames(video_path, frame_size=(64, 64), num_frames=30):
    """
    動画ファイルを読み込み、指定した数のフレームをリサイズして取得する関数
    """
    cap = cv2.VideoCapture(video_path)  # 動画ファイルを読み込み
    frames = []  # フレームを格納するリスト
    count = 0  # フレームカウンタ

    while cap.isOpened() and count < num_frames:
        ret, frame = cap.read()  # フレームを1つずつ読み込む
        if not ret:
            break  # フレームの読み込みに失敗した場合は終了
        frame = cv2.resize(frame, frame_size)  # フレームを指定サイズにリサイズ
        frames.append(frame)  # フレームをリストに追加
        count += 1  # フレームカウントを増加

    cap.release()  # 動画ファイルを閉じる
    frames = np.array(frames)  # フレームリストをnumpy配列に変換
    return frames

# モデルの読み込み
model = load_model('../models/robot_dance_model.h5')  # 学習済みモデルの読み込み

# 新しい動画ファイルのパス
new_video_path = '../data/new_video.mp4'

# 動画をフレームに分割して前処理
new_video_frames = load_video_frames(new_video_path)
if new_video_frames.shape[0] != 30:
    print("エラー: 動画のフレーム数が不足しています。")
    exit()

# モデル入力の形式に合わせて次元を追加
new_video_frames = np.expand_dims(new_video_frames, axis=0)  # (1, 30, 64, 64, 3)

# スコアの予測
predicted_scores = model.predict(new_video_frames)
print(f"予測されたスコア: {predicted_scores[0]}")
