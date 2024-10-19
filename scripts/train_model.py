import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# データの読み込み
print("トレーニングデータを読み込んでいます...")
train_videos = np.load('./data/train_videos.npy', allow_pickle=True)  # トレーニング用動画データ
train_labels = np.load('./data/train_labels.npy', allow_pickle=True)  # トレーニング用ラベルデータ

# ラベルデータをfloat32型に変換
train_labels = train_labels.astype(np.float32)

print(f"トレーニングデータの形状: {train_videos.shape}")
print(f"トレーニングラベルの形状: {train_labels.shape}")

def build_model(input_shape, num_scores=3):
    """
    3D CNN + LSTMを用いたモデルを構築する関数
    """
    model = models.Sequential()

    # 3D畳み込み層1: 動画の空間的特徴を抽出
    model.add(layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

    # 3D畳み込み層2: より複雑な特徴を抽出
    model.add(layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

    # 3D畳み込み層3: 高次の特徴を抽出
    model.add(layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

    # フラット化: 3Dデータを1次元に変換
    model.add(layers.Flatten())

    # リピートベクター: LSTM層への入力用にデータを複製
    model.add(layers.RepeatVector(10))

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
input_shape = (30, 480, 640, 3)

# モデルの構築
model = build_model(input_shape, num_scores=3)
model.summary()

# モデルのトレーニング
print("モデルのトレーニングを開始します...")
history = model.fit(
    train_videos, train_labels,
    epochs=50,               # エポック数
    batch_size=16            # バッチサイズ
)

# トレーニング結果のプロット
plt.figure(figsize=(12, 4))

# 損失関数のプロット
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='訓練損失')
plt.title('損失関数の推移')
plt.xlabel('エポック')
plt.ylabel('損失')
plt.legend()

# 平均絶対誤差（MAE）のプロット
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='訓練MAE')
plt.title('平均絶対誤差の推移')
plt.xlabel('エポック')
plt.ylabel('MAE')
plt.legend()

plt.show()

# モデルの保存
model.save('./models/robot_dance_model.h5')
print("モデルのトレーニングと保存が完了しました。")
