import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# 動画をフレームに分割する関数
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

def predict_score(video_path='./data/test/new_video.mp4'):
    """ 学習済みモデルを使って動画に対するスコアを予測する関数 """
    
    # カスタム損失関数の指定
    custom_objects = {'mse': tf.keras.losses.MeanSquaredError()}
    
    # モデルの読み込み
    print("モデルを読み込んでいます...")
    model = load_model('./models/robot_dance_model.h5', custom_objects=custom_objects)
    print("モデルの読み込みが完了しました。")

    # 動画をフレームに分割して前処理
    new_video_frames = load_video_frames(video_path)
    if new_video_frames.shape[0] != 30:
        print("エラー: 動画のフレーム数が不足しています。")
        return 0.0

    # モデル入力の形式に合わせて次元を追加
    new_video_frames = np.expand_dims(new_video_frames, axis=0)

    score_names = ['リズム', '創造性', '感情表現']

    # スコアの予測
    predicted_scores = model.predict(new_video_frames)

    # スコアをフォーマットして表示
    print("予測されたスコア:")
    print("")
    for name, score in zip(score_names, predicted_scores[0]):
        print(f"{name}: {score:.2f}")

    # スコアの合計を返す
    total_predicted_score = sum(predicted_scores[0])
    print("")
    print(f"合計スコア: {total_predicted_score:.2f}")
    
    return total_predicted_score

if __name__ == "__main__":
    predict_score()