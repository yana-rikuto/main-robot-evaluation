import subprocess
import os

def run_preprocess():
    """ 動画の前処理を実行 """
    print("=== 前処理を実行中 ===")
    subprocess.run(['python', 'train_scripts/preprocess.py'])
    print("=== 前処理が完了しました ===\n")

def run_train_model():
    """ モデルのトレーニングを実行 """
    print("=== モデルのトレーニングを実行中 ===")
    subprocess.run(['python', 'train_scripts/train_model.py'])
    print("=== モデルのトレーニングが完了しました ===\n")

def run_predict():
    """ スコア予測を実行 """
    print("=== スコア予測を実行中 ===")
    subprocess.run(['python', 'train_scripts/predict_score.py'])
    print("=== スコア予測が完了しました ===\n")

def run_robot_evaluation():
    """ ロボットの動作評価を実行 """
    print("=== ロボット動作の評価を実行中 ===")
    subprocess.run(['python', 'objective_evaluate/tecnique_evaluate.py'])
    print("=== ロボット動作の評価が完了しました ===\n")

if __name__ == "__main__":
    print("=== 全体のプロセスを開始します ===\n")
    
    # 1. 動画の前処理を実行
    run_preprocess()

    # 2. モデルのトレーニングを実行
    run_train_model()

    # 3. スコア予測を実行
    run_predict()

    # 4. ロボット動作の評価を実行
    run_robot_evaluation()

    print("=== 全てのプロセスが完了しました ===")
