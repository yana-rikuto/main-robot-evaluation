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
    # DataFrameに変換
    df = pd.DataFrame(data)
    return df

def preprocess_features(features_df):
    """
    特徴量データの前処理
    """
    # 数値データのみを抽出
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
    # SHAP explainer の初期化
    explainer = shap.Explainer(model, features)

    # SHAP値の計算
    shap_values = explainer(features)
    print("SHAP 値を計算しました。")

    # デバッグ: SHAP値と特徴量の形状を出力
    print(f"SHAP 値の形状: {shap_values.values.shape}")
    print(f"特徴量の形状: {features.shape}")

    # SHAP値をスコアごとにプロット
    try:
        for score_idx, score_name in enumerate(score_names):
            print(f"スコア {score_idx + 1}（{score_name}）の SHAP値:")
            # スコアに対応するSHAP値を抽出
            score_shap_values = shap_values.values[:, :, score_idx]
            
            # SHAPプロット
            shap.summary_plot(
                score_shap_values, 
                features, 
                feature_names=feature_names, 
                plot_type="bar", 
                show=True
            )
            output_path = f"./data/shap_results/shap_score_{score_idx + 1}_{score_name.lower()}.png"
            plt.savefig(output_path)  # グラフを保存
            print(f"SHAPグラフを保存しました: {output_path}")
            plt.clf()  # グラフをクリアして次のプロットへ
    except Exception as e:
        print("SHAPのプロット中にエラーが発生しました:", e)
        print("SHAP値の形状:", shap_values.values.shape)
        print("特徴量の形状:", features.shape)
        print("特徴量名:", feature_names)

    return shap_values



def suggest_improvements_with_sign(shap_values, feature_names, score_names):
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
                f"{score_name} を改善するには、{min_feature} を調整することが有効です。"
                f"この特徴量は現在スコアに対して負の寄与を持っています ({min_value:.4f})。"
            )
        else:
            suggestion = (
                f"{score_name} のスコアは現状改善の必要が低いようです。"
                f"主要な特徴量 {min_feature} はスコアに正の寄与を持っています ({min_value:.4f})。"
            )

        print(suggestion)
        suggestions.append(suggestion)

    return suggestions


def main():
    # ファイルパス
    json_file_path = "./objective_evaluate/robotData.json"

    # JSONから特徴量をロード
    features_df = load_features_from_json(json_file_path)
    print("特徴量をロードしました。")
    print(features_df.head())  # デバッグ用

    # 特徴量の前処理
    features, feature_names = preprocess_features(features_df)
    print(f"特徴量の形状: {features.shape}")

    # 特徴量ベースの簡易モデルを作成
    model = create_feature_based_model(input_dim=features.shape[1])
    print("特徴量ベースのモデルを作成しました。")

    # SHAP解析
    score_names = ["Rhythm", "Creativity", "Emotional Expression"]
    shap_values = analyze_with_shap(model, features, feature_names, score_names)

    # 改善提案の生成
    suggestions = suggest_improvements_with_sign(shap_values, feature_names, score_names)

    # 改善提案を保存
    with open("./data/improvement_suggestions_with_sign.txt", "w") as f:
        for suggestion in suggestions:
            f.write(suggestion + "\n")
    print("改善提案を保存しました。")

if __name__ == "__main__":
    main()