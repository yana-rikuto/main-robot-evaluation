import json

success_number = 0

# 動作のパターンに対する評価ルールを定義する関数
def rule_1(movements):
    """
    例: 前進 -> 回転 -> 後退 の順で動作があったら5点加点
    """
    global success_number  # グローバル変数として指定

    for i in range(len(movements) - 2):
        if (
            movements[i]["action"] == "move forward" and
            movements[i]["speed"] == 1 and
            movements[i + 1]["action"] == "rotate" and
            movements[i + 2]["action"] == "move backward"
        ):
            success_number += 1
            return 5
    return 0

def rule_2(movements):
    """
    例: 前進 -> 回転 -> 後退 の順で動作があったら5点加点
    """

    global success_number  # グローバル変数として指定

    for i in range(len(movements) - 2):
        if (
            movements[i]["action"] == "move forward" and
            movements[i]["speed"] == 1 and
            movements[i + 1]["action"] == "rotate" and
            movements[i + 2]["action"] == "move backward"
        ):
            success_number += 1
            return 5
    return 0

def rule_time_penalty(movements, max_time=30, penalty_interval=3, max_penalty=10):
    """
    30秒以上の演技の場合、3秒ごとに1点を減点する
    減点は最大10点まで
    """
    # 全体の演技時間を計算
    total_duration = sum(movement["duration"] for movement in movements)

    # 30秒を超えた分の時間を計算
    overtime = max(0, total_duration - max_time)

    # 減点計算
    penalty_points = min(int(overtime // penalty_interval), max_penalty)

    print(f"演技時間: {total_duration}秒 - 減点: {penalty_points}点")
    
    return -penalty_points

def rule_smoothness_bonus(movements, speed_threshold=1.0, rotation_speed_threshold=30.0, max_bonus=10):
    """
    滑らかさの評価: 速度や回転速度の変化が閾値以下であれば1点加点
    最大10点の加点
    """
    bonus_points = 0

    for i in range(1, len(movements)):
        # 前の動作と現在の動作の速度と回転速度の差を計算
        speed_diff = abs(movements[i]["speed"] - movements[i - 1]["speed"])
        rotation_speed_diff = abs(movements[i]["rotationSpeed"] - movements[i - 1]["rotationSpeed"])

        # 速度変化と回転速度変化が閾値以下であれば加点
        if speed_diff <= speed_threshold and rotation_speed_diff <= rotation_speed_threshold:
            bonus_points += 1
            if bonus_points >= max_bonus:  # 最大加点に達したら終了
                bonus_points = max_bonus
                break

    print(f"滑らかさ評価による加点: {bonus_points}点")
    return bonus_points

# 評価関数
def evaluate_movements(movements, rules):
    global success_number
    success_number = 0  # カウントをリセット
    total_score = 0
    for rule in rules:
        total_score += rule(movements)  # 各ルールに従って加点
    return total_score

# JSONファイルから動きを読み込む
def load_movements_from_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# メイン処理
def main():
    # 動きのデータを読み込む
    movements = load_movements_from_json("./objective_evaluate/robotData.json")
    # 評価ルールのリストを定義
    evaluation_rules = [
        rule_1,
        rule_2,
        rule_time_penalty, # 演技時間に基づく減点ルール
        rule_smoothness_bonus
    ]
    

    # 動きの組み合わせを評価
    technique_score = evaluate_movements(movements, evaluation_rules) 

    # 成功数に応じて加点（最大10点まで）
    additional_score = min(success_number, 10)

    # 総合スコアに成功数の加点を追加
    total_score = additional_score + technique_score

    print(f"成功した技の数: {success_number} - 加点: {additional_score}点")

    print(f"技術得点: {technique_score}点")
    
    # 結果を表示
    print(f"技術的総合スコア: {total_score}")

if __name__ == "__main__":
    main()