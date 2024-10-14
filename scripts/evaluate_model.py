# scripts/evaluate_model.py

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# モデルとデータの読み込み
model = load_model('../models/robot_dance_model.h5')  # 学習済みモデルの読み込み
test_videos = np.load('../data/test_videos.npy')      # テスト用動画データ
test_labels = np.load('../data/test_labels.npy')      # テスト用ラベルデータ

# モデルによる予測
predictions = model.predict(test_videos)

# 評価指標の計算
mae = mean_absolute_error(test_labels, predictions)
mse = mean_squared_error(test_labels, predictions)
print(f"平均絶対誤差 (MAE): {mae}")
print(f"平均二乗誤差 (MSE): {mse}")

# 各スコアの散布図
scores = ['スコア1', 'スコア2', 'スコア3']
for i in range(3):
    plt.figure()
    plt.scatter(test_labels[:, i], predictions[:, i], alpha=0.5)
    plt.xlabel(f"実際の{scores[i]}")
    plt.ylabel(f"予測された{scores[i]}")
    plt.title(f"{scores[i]}の実際値と予測値の比較")
    plt.plot([0, 10], [0, 10], 'r--')  # 理想的な予測ライン
    plt.show()
