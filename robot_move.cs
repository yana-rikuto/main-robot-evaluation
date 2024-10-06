using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;  // Newtonsoft.Jsonライブラリを使用してJSON操作を実装
public class RobotCombination : MonoBehaviour
{
    public float speed1 = 5.0f;           // ロボットの移動速度
    public float speed2 = 1.0f;           // ロボットの移動速度
    public float speed3 = 3.0f;           // ロボットの移動速度
    public float speed4 = 2.0f;           // ロボットの移動速度
    public float rotationSpeed1 = 90.0f;  // ロボットの回転速度
    public float rotationSpeed2 = 180.0f; // ロボットの回転速度
    public float rotationSpeed3 = 45.0f;  // ロボットの回転速度
    public float rotationSpeed4 = 120.0f; // ロボットの回転速度

    // ロボットの動作データを格納するクラス
    [System.Serializable]
    public class RobotData
    {
        public float speed;
        public float rotationSpeed;
        public string action;
        public float duration;  // 動作の持続時間
    }

    // 複数のロボットデータを格納するリスト
    private List<RobotData> robotDataList = new List<RobotData>();

    // 現在の動作を記録する変数
    private float currentSpeed = 0.0f;
    private float currentRotationSpeed = 0.0f;
    private string currentAction = "";
    private float actionStartTime = 0.0f;

    void Start()
    {
        StartCoroutine(MoveRoutine());
    }

    IEnumerator MoveRoutine()
    {
        yield return new WaitForSeconds(1.5f);

        // 前進
        StartAction(speed2, 0, "move forward");
        yield return MoveForDuration(1.0f);
        EndAction();

        // 左右回転
        StartAction(0, rotationSpeed1, "rotate");
        yield return RotateForDuration(1.0f);
        EndAction();

        // 前進
        StartAction(speed2, 0, "move forward");
        yield return MoveForDuration(1.0f);
        EndAction();

        // ロボットの動作データをJSONファイルに保存
        SaveToJson();
    }

    // 前進動作
    IEnumerator MoveForDuration(float duration)
    {
        float moveTime = 0.0f;
        while (moveTime < duration)
        {
            transform.Translate(Vector3.forward * currentSpeed * Time.deltaTime);
            moveTime += Time.deltaTime;
            yield return null;
        }
    }

    // 回転動作
    IEnumerator RotateForDuration(float duration)
    {
        float rotationTime = 0.0f;
        while (rotationTime < duration)
        {
            transform.Rotate(Vector3.up * currentRotationSpeed * Time.deltaTime);
            rotationTime += Time.deltaTime;
            yield return null;
        }
    }

    // 新しい動作を開始するメソッド
    void StartAction(float speed, float rotationSpeed, string action)
    {
        currentSpeed = speed;
        currentRotationSpeed = rotationSpeed;
        currentAction = action;
        actionStartTime = Time.time;
    }

    // 動作を終了してデータを記録するメソッド
    void EndAction()
    {
        float actionDuration = Time.time - actionStartTime;
        RecordRobotData(currentSpeed, currentRotationSpeed, currentAction, actionDuration);
    }

    // ロボットデータをリストに記録するメソッド
    void RecordRobotData(float speed, float rotationSpeed, string action, float duration)
    {
        RobotData data = new RobotData
        {
            speed = speed,
            rotationSpeed = rotationSpeed,
            action = action,
            duration = duration
        };
        robotDataList.Add(data);
    }

    // ロボットデータをJSON形式で保存するメソッド
    void SaveToJson()
    {
        string json = JsonConvert.SerializeObject(robotDataList, Formatting.Indented);
        File.WriteAllText("robotData.json", json);
        Debug.Log("JSONファイルに保存されました: robotData.json");
    }
}