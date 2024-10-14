# scripts/preprocess.py

import cv2
import os
import numpy as np
import pandas as pd

# 動画データとラベルデータのディレクトリ設定
TRAIN_DIR = '../data/train'
TEST_DIR = '../data/test'
LABELS_FILE = '../data/labels.csv'
FRAME_SIZE = (64, 64)  # フレームのサイズを64x64にリサイズ
NUM_FRAMES = 30  # 各動画から抽出するフレーム数

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

def preprocess_data():
    """
    トレーニングデータとテストデータを前処理し、ラベルデータを読み込む関数
    """
    # ラベルデータの読み込み
    labels_df = pd.read_csv(LABELS_FILE)

    # トレーニングデータの前処理
    train_videos = []
    train_labels = []
    for index, row in labels_df.iterrows():
        video_path = os.path.join(TRAIN_DIR, row['filename'])
        frames = load_video_frames(video_path, FRAME_SIZE, NUM_FRAMES)
        if frames.shape[0] == NUM_FRAMES:
            train_videos.append(frames)
            train_labels.append(row[['score1', 'score2', 'score3']].values)  # 3種類のスコアを取得

    train_videos = np.array(train_videos)
    train_labels = np.array(train_labels)

    # テストデータの前処理
    test_videos = []
    test_labels = []
    for index, row in labels_df.iterrows():
        video_path = os.path.join(TEST_DIR, row['filename'])
        frames = load_video_frames(video_path, FRAME_SIZE, NUM_FRAMES)
        if frames.shape[0] == NUM_FRAMES:
            test_videos.append(frames)
            test_labels.append(row[['score1', 'score2', 'score3']].values)

    test_videos = np.array(test_videos)
    test_labels = np.array(test_labels)

    # データの保存
    np.save('../data/train_videos.npy', train_videos)
    np.save('../data/train_labels.npy', train_labels)
    np.save('../data/test_videos.npy', test_videos)
    np.save('../data/test_labels.npy', test_labels)

if __name__ == "__main__":
    preprocess_data()
    print("データの前処理が完了しました。")
