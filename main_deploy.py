import cv2
import time
import math
from ultralytics import YOLO

# === ⚙️ 参数配置区域 ===
MODEL_PATH = 'best.pt'       # 你的模型路径
CONF_THRES = 0.5             # 置信度阈值 (建议0.4-0.6)
CAMERA_ID = 0                # USB摄像头通常是0，MIPI可能是8
CUBE_WIDTH_REAL = 5.0        # 方块实际宽度 5cm
FOCAL_LENGTH = 600           # 焦距 (需要根据你的摄像头微调)

# 颜色定义 (BGR格式)
COLORS = {
    'red_cube': (0, 0, 255),
    'blue_cube': (255, 0, 0),
    'green_cube': (0, 255, 0),
    'pink_cube': (180, 105, 255)
}

def main():
    # 1. 加载模型
    print(f"🚀 正在加载模型 {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 2. 打开摄像头
    cap = cv2.VideoCapture(CAMERA_ID)
    # 降低分辨率以提速 (320x240 或 640x480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ 摄像头打开失败")
        return

    print("✅ 开始运行！按 'q' 退出")

    while True:
        t_start = time.time()
        
        ret, frame = cap.read()
        if not ret: break

        # === 3. 模型推理 (核心代码) ===
        # verbose=False 防止终端刷屏
        results = model(frame, conf=CONF_THRES, verbose=False)

        # === 4. 解析数据 (转换代码的核心) ===
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # 获取坐标 (中心x, 中心y, 宽, 高)
                x, y, w, h = box.xywh[0].tolist()
                
                # 获取类别
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])

                # --- 🎯 距离计算 (单目测距) ---
                # 距离 = (实际宽 * 焦距) / 像素宽
                distance = (CUBE_WIDTH_REAL * FOCAL_LENGTH) / w

                # --- 🛠️ 逻辑修正 (可选) ---
                # 如果发现红粉混淆严重，可以在这里加补丁，例如：
                # if label == 'pink_cube' and conf < 0.6: label = 'red_cube'

                # --- 🎨 绘图 ---
                color = COLORS.get(label, (255, 255, 255))
                
                # 画框 (xywh 转 xyxy 用于画图)
                x1 = int(x - w/2)
                y1 = int(y - h/2)
                x2 = int(x + w/2)
                y2 = int(y + h/2)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # 显示信息
                info = f"{label} {distance:.1f}cm"
                cv2.putText(frame, info, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # === 🤖 这里可以添加控制代码 ===
                # if label == 'red_cube' and distance < 10:
                #     serial.write(b'STOP') # 发送停车指令给下位机

        # 计算 FPS
        fps = 1.0 / (time.time() - t_start)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 显示画面 (如果接了屏幕)
        cv2.imshow("RDK X5 Deploy", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()