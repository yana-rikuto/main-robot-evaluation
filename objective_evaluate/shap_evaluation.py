import json
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 特徴量データの読み込み（JSONファイル）
def load_features_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df

# スコアデータの読み込み（CSVファイル）
def load_scores_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df

# 特徴量の前処理
def preprocess_features(features_df):
    features = features_df[["speed", "rotationSpeed", "duration"]].values
    feature_names = ["Speed", "Rotation Speed", "Duration"]
    return features, feature_names

# SHAP解析を行う
def analyze_with_shap(features, scores, feature_names, score_names):
    # スコア関数を作成
    def score_model(X):
        # `features` に基づき、対応するスコアを返す
        return scores * np.ones((X.shape[0], scores.shape[1]))  # 特徴量のサンプル数にスコアを繰り返す

    # SHAP explainer の初期化
    explainer = shap.KernelExplainer(score_model, features)

    # SHAP値の計算
    shap_values = explainer.shap_values(features)
    print("SHAP 値を計算しました。")

    # SHAP値をスコアごとにプロット
    for score_idx, score_name in enumerate(score_names):
        print(f"スコア {score_idx + 1}（{score_name}）の SHAP値:")
        score_shap_values = shap_values[score_idx]

        # SHAPプロット
        try:
            shap.summary_plot(score_shap_values, features, feature_names=feature_names, plot_type="bar", show=False)
            output_path = f"./data/shap_results/shap_score_{score_idx + 1}_{score_name.lower()}.png"
            plt.savefig(output_path)
            print(f"SHAPグラフを保存しました: {output_path}")
            plt.clf()  # グラフをクリアして次のプロットへ
        except Exception as e:
            print(f"SHAPのプロット中にエラーが発生しました: {e}")
            print("SHAP値の形状:", score_shap_values.shape)
            print("特徴量の形状:", features.shape)
            print("特徴量名:", feature_names)

# メイン処理
def main():
    # ファイルパス
    json_file_path = "./objective_evaluate/robotData.json"
    csv_file_path = "./objective_evaluate/scores.csv"

    # JSONから特徴量をロード
    features_df = load_features_from_json(json_file_path)
    features, feature_names = preprocess_features(features_df)

    # CSVからスコアをロード
    scores_df = load_scores_from_csv(csv_file_path)
    scores = scores_df.values  # スコアデータ（Rhythm, Creativity, Emotional Expression）
    score_names = list(scores_df.columns)

    print("特徴量をロードしました。")
    print(features_df.head())
    print(f"特徴量の形状: {features.shape}")
    print(f"スコアの形状: {scores.shape}")

    # SHAP解析
    analyze_with_shap(features, scores, feature_names, score_names)

if __name__ == "__main__":
    main()
