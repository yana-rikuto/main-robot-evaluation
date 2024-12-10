import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_features_from_json(file_path):
    """
    JSONファイルから特徴量を読み込む
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df

def preprocess_features(features_df):
    """
    特徴量データの前処理
    """
    features = features_df[["speed", "rotationSpeed", "duration"]].values
    feature_names = ["Speed", "Rotation Speed", "Duration"]
    return features, feature_names

def create_feature_based_model(input_dim):
    """
    特徴量ベースの簡易モデルを作成
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(3, activation='linear')  # スコア（リズム、創造性、感情表現）を出力
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def analyze_with_pdp(model, features, feature_names):
    """
    Partial Dependence Plot を用いた寄与度解析と可視化
    """
    print("PDP（部分依存プロット）を生成中...")
    fig, axes = plt.subplots(1, len(feature_names), figsize=(15, 5), sharey=True)
    
    for i, feature_name in enumerate(feature_names):
        ax = axes[i]
        display = PartialDependenceDisplay.from_estimator(
            model,
            X=features,
            features=[i],
            feature_names=feature_names,
            ax=ax
        )
        ax.set_title(feature_name)

    plt.tight_layout()
    output_path = "./data/pdp_results/pdp_plot.png"
    plt.savefig(output_path)
    print(f"PDPプロットを保存しました: {output_path}")
    plt.show()

def main():
    # データファイルパス
    json_file_path = "./objective_evaluate/robotData.json"

    # 特徴量データのロードと前処理
    features_df = load_features_from_json(json_file_path)
    features, feature_names = preprocess_features(features_df)
    print("特徴量をロードしました。")
    print(features_df.head())
    print(f"特徴量の形状: {features.shape}")

    # ラベル（スコア）の作成
    scores = np.random.rand(features.shape[0], 3)  # 仮のスコアデータ（実際は別途取得したものを利用する）
    print(f"スコアの形状: {scores.shape}")

    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(features, scores, test_size=0.2, random_state=42)

    # 特徴量ベースのモデル作成と訓練
    model = create_feature_based_model(input_dim=features.shape[1])
    print("特徴量ベースのモデルを作成しました。")
    model.fit(X_train, y_train, epochs=10, batch_size=4, verbose=1, validation_split=0.1)

    # PDP解析
    analyze_with_pdp(model, X_test, feature_names)

if __name__ == "__main__":
    main()
