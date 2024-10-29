import json

# 動作のパターンに対する評価ルールを定義する関数
def rule_1(movements):
    """
    例: 前進 -> 回転 -> 後退 の順で動作があったら5点加点
    """
    for i in range(len(movements) - 2):
        if (
            movements[i]["action"] == "move forward" and
            movements[i]["speed"] == 1 and
            movements[i + 1]["action"] == "rotate" and
            movements[i + 2]["action"] == "move backward"
        ):
            return 5
    return 0

def rule_2(movements):
    """
    例: 前進 -> 回転 -> 後退 の順で動作があったら5点加点
    """
    for i in range(len(movements) - 2):
        if (
            movements[i]["action"] == "move forward" and
            movements[i]["speed"] == 1 and
            movements[i + 1]["action"] == "rotate" and
            movements[i + 2]["action"] == "move backward"
        ):
            return 5
    return 0

# 評価関数
def evaluate_movements(movements, rules):
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
        rule_2
    ]

    # 動きの組み合わせを評価
    total_score = evaluate_movements(movements, evaluation_rules)

    # 結果を表示
    print(f"総合評価スコア: {total_score}")

if __name__ == "__main__":
    main()