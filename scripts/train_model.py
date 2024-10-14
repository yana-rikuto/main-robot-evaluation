# scripts/train_model.py

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# データの読み込み
train_videos = np.load('../data/train_videos.npy')  # トレーニング用動画データ
train_labels = np.load('../data/train_labels.npy')  # トレーニング用ラベルデータ
test_videos = np.load('../data/test_videos.npy')    # テスト用動画データ
test_labels = np.load('../data/test_labels.npy')    # テスト用ラベルデータ

def build_model(input_shape, num_scores=3):
    """
    3D CNN + LSTMを用いたモデルを構築する関数
    """
    model = models.Sequential()
    
    # 3D畳み込み層1: 動画の空間的特徴を抽出
    model.add(layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape))
    # 3D最大プーリング層1: 特徴マップの次元を縮小
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    
    # 3D畳み込み層2: より複雑な特徴を抽出
    model.add(layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
    # 3D最大プーリング層2: 特徴マップの次元を縮小
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    
    # 3D畳み込み層3: 高次の特徴を抽出
    model.add(layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu'))
    # 3D最大プーリング層3: 特徴マップの次元を縮小
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    
    # フラット化: 3Dデータを1次元に変換
    model.add(layers.Flatten())
    
    # リピートベクター: LSTM層への入力用にデータを複製
    model.add(layers.RepeatVector(10))  # 10タイムステップに繰り返す
    
    # LSTM層1: 時間的な依存関係を学習
    model.add(layers.LSTM(64, return_sequences=True))
    
    # LSTM層2: 高次の時間的な依存関係を学習
    model.add(layers.LSTM(64))
    
    # 全結合層: 高次の特徴を集約
    model.add(layers.Dense(64, activation='relu'))
    
    # 出力層: 3つのスコアを予測（回帰問題）
    model.add(layers.Dense(num_scores, activation='linear'))
    
    # モデルのコンパイル: 損失関数は平均二乗誤差（MSE）、最適化手法はAdam
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

# モデルの入力形状（例: 30フレーム、64x64ピクセル、3チャンネル）
input_shape = (30, 64, 64, 3)

# モデルの構築
model = build_model(input_shape, num_scores=3)
model.summary()

# モデルのトレーニング
history = model.fit(
    train_videos, train_labels,
    epochs=50,               # エポック数
    batch_size=16,           # バッチサイズ
    validation_data=(test_videos, test_labels)  # 検証データ
)

# トレーニング結果のプロット
plt.figure(figsize=(12, 4))

# 損失関数のプロット
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='訓練損失')
plt.plot(history.history['val_loss'], label='検証損失')
plt.title('損失関数の推移')
plt.xlabel('エポック')
plt.ylabel('損失')
plt.legend()

# 平均絶対誤差（MAE）のプロット
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='訓練MAE')
plt.plot(history.history['val_mae'], label='検証MAE')
plt.title('平均絶対誤差の推移')
plt.xlabel('エポック')
plt.ylabel('MAE')
plt.legend()

plt.show()

# モデルの保存
model.save('../models/robot_dance_model.h5')
print("モデルのトレーニングと保存が完了しました。")
