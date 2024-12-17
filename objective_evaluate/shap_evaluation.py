import json
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def load_features_from_json(file_path):
    """ JSONファイルから特徴量を読み込む """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def preprocess_features(features_df):
    """ 特徴量データの前処理 """
    features = features_df[["speed", "rotationSpeed", "duration"]].values
    feature_names = ["Speed", "Rotation Speed", "Duration"]
    return features, feature_names

def create_feature_based_model(input_dim):
    """ 特徴量ベースの簡易モデルを作成 """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(3, activation='linear')  # スコアを出力
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def analyze_with_shap_and_plot(model, features, feature_names, score_names):
    """
    SHAPを用いた寄与度解析と可視化
    """
    explainer = shap.Explainer(model, features)

    # SHAP値の計算
    shap_values = explainer(features)
    print("SHAP解析が完了しました。")

    # SHAP値の可視化
    print("\nSHAP値の可視化:")
    for score_idx, score_name in enumerate(score_names):
        print(f"--- {score_name} のSHAP解析 ---")
        score_shap_values = shap_values.values[:, :, score_idx]

        # SHAP値のプロット
        shap.summary_plot(
            score_shap_values,
            features,
            feature_names=feature_names,
            plot_type="bar",
            show=True
        )
        output_path = f"./output/shap_score_{score_idx + 1}_{score_name.lower()}.png"
        plt.savefig(output_path)
        print(f"SHAPグラフを保存しました: {output_path}")
        plt.clf()

    return shap_values

def suggest_improvement(shap_values, features, feature_names, score_names, scores):
    """
    最もスコアが低い項目に対して改善提案を行う
    """
    min_score_idx = np.argmin(scores)
    print(f"\n--- 改善対象: {score_names[min_score_idx]} ---")

    mean_shap_values = np.mean(shap_values[:, :, min_score_idx], axis=0)
    min_feature_idx = np.argmin(mean_shap_values)
    suggestion = (
        f"{score_names[min_score_idx]} を改善するには、"
        f"{feature_names[min_feature_idx]} を調整することが有効です。\n"
    )
    print(suggestion)

    # 改善提案を保存
    output_path = "./output/improvement_suggestions.txt"
    with open(output_path, "w") as f:
        f.write(suggestion)
    print(f"改善提案を {output_path} に保存しました。")

def main():
    # ファイルパス
    json_file_path = "./objective_evaluate/robotData.json"
    scores_csv_path = "./output/predict_scores.csv"

    # 特徴量データの読み込み
    features_df = load_features_from_json(json_file_path)
    features, feature_names = preprocess_features(features_df)
    print("特徴量をロードしました。")
    print(f"特徴量の形状: {features.shape}")

    # スコアの読み込み
    with open(scores_csv_path, 'r') as f:
        reader = csv.reader(f)
        score_names = next(reader)  # ヘッダー行
        scores = np.array(next(reader), dtype=float)  # スコア行
    print(f"スコア: {scores}")

    # モデルを作成
    model = create_feature_based_model(features.shape[1])
    print("特徴量ベースのモデルを作成しました。")

    # SHAP解析とグラフ出力
    shap_values = analyze_with_shap_and_plot(model, features, feature_names, score_names)

    # 最も低いスコアに対する改善提案
    suggest_improvement(shap_values.values, features, feature_names, score_names, scores)

if __name__ == "__main__":
    main()
