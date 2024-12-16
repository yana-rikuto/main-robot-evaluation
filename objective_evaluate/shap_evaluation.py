import json
import numpy as np
import pandas as pd
import shap
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt

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

def analyze_with_shap(model, features, feature_names, score_names=["Rhythm", "Creativity", "Emotional Expression"]):
    """
    SHAPを用いた寄与度解析と可視化
    """
    explainer = shap.Explainer(model, features)
    shap_values = explainer(features)
    print("SHAP 値を計算しました。")

    for score_idx, score_name in enumerate(score_names):
        print(f"--- {score_name} の SHAP値 ---")
        score_shap_values = shap_values.values[:, :, score_idx]
        shap.summary_plot(
            score_shap_values,
            features,
            feature_names=feature_names,
            plot_type="bar",
            show=True
        )
    return shap_values

def suggest_improvements_with_actions(shap_values, feature_names, score_names):
    """
    SHAP値に基づいて改善提案を生成（符号を考慮）
    """
    suggestions = []

    for score_idx, score_name in enumerate(score_names):
        print(f"--- {score_name} の改善提案 ---")
        # スコアに対応するSHAP値を取得
        score_shap_values = shap_values.values[:, :, score_idx]
        mean_shap_values = np.mean(score_shap_values, axis=0)  # 特徴量ごとの平均寄与度

        # ネガティブ寄与が大きい特徴量を特定
        min_contrib_idx = np.argmin(mean_shap_values)
        min_feature = feature_names[min_contrib_idx]
        min_value = mean_shap_values[min_contrib_idx]

        # 改善提案の生成
        if min_value < 0:
            suggestion = (
                f"{score_name} を改善するには、{min_feature} を調整することが有効です。\n"
            )
        else:
            suggestion = (
                f"{score_name} のスコアは現状改善の必要が低いようです。\n"
            )

        print(suggestion)
        suggestions.append(suggestion)

    return suggestions


def main():
    json_file_path = "./objective_evaluate/robotData.json"
    features_df = load_features_from_json(json_file_path)
    print("特徴量をロードしました。")
    print(features_df.head())

    features, feature_names = preprocess_features(features_df)
    print(f"特徴量の形状: {features.shape}")

    model = create_feature_based_model(input_dim=features.shape[1])
    print("特徴量ベースのモデルを作成しました。")

    score_names = ["Rhythm", "Creativity", "Emotional Expression"]
    shap_values = analyze_with_shap(model, features, feature_names, score_names)

    suggestions = suggest_improvements_with_actions(shap_values, feature_names, score_names)

    with open("./data/improvement_suggestions_with_actions.txt", "w") as f:
        for suggestion in suggestions:
            f.write(suggestion + "\n")
    print("改善提案を保存しました。")

if __name__ == "__main__":
    main()
