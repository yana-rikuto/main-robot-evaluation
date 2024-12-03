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
    analyze_with_shap(model, features, feature_names)

if __name__ == "__main__":
    main()
