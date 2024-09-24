import json

# JSONファイルを読み込む関数
def read_robot_data(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

# ロボットの動作を解析して出力する関数
def analyze_robot_data(data):
    for i, action in enumerate(data):
        speed = action['speed']
        rotation_speed = action['rotationSpeed']
        action_type = action['action']
        time = action['time']
        
        print("\n--- ロボットの特徴量分析結果 ---")
        if speed > 0 and rotation_speed == 0:
            print(f"動作 {i+1}: 前進 - 速度: {speed}m/s、時間: {time}秒")
        elif speed == 0 and rotation_speed > 0:
            print(f"動作 {i+1}: 回転 - 回転速度: {rotation_speed}度/秒、時間: {time}秒")
        else:
            print(f"動作 {i+1}: 未知の動作 - 速度: {speed}, 回転速度: {rotation_speed}, 時間: {time}秒")

# メイン関数
def main():
    # 読み込むJSONファイルのパス
    json_file = 'robotData.json'
    
    # ロボットデータを読み込み
    robot_data = read_robot_data(json_file)
    
    # ロボットの動きを解析して出力
    analyze_robot_data(robot_data)

# 実行
if __name__ == "__main__":
    main()
