import torch
import sys
from ultralytics import YOLO

def check_gpu():
    """
    严谨性检查：在训练开始前，确保 GPU 环境完全就绪。
    """
    print("🔍 正在检查 GPU 环境...")
    
    # 1. 检查 CUDA 是否可用
    if not torch.cuda.is_available():
        print("❌ 【致命错误】未检测到 NVIDIA GPU 或 CUDA 环境！")
        print("   原因可能是：")
        print("   1. 您安装的是 CPU 版 PyTorch（请去官网重新下载 CUDA 版）。")
        print("   2. 显卡驱动未安装。")
        print("   程序已强制终止，以免使用 CPU 龟速训练。")
        sys.exit(1) # 直接退出程序

    # 2. 获取显卡信息
    gpu_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ 检测到 {gpu_count} 张显卡")
    print(f"🚀 当前使用显卡: {gpu_name}")
    print("------------------------------------------------")

def train_model():
    # 先执行环境检查
    check_gpu()

    # 1. 加载模型
    # 会自动下载 yolov8n.pt，如果网络不好，请手动下载放旁边
    model = YOLO('yolov8n.pt') 

    # 2. 开始训练
    print("🚀 开始 GPU 极速训练...")
    
    model.train(
        data='data.yaml',   # 数据集配置文件
        epochs=150,          # 训练轮数
        imgsz=640,          # 图像尺寸
        
        # === ⚡ GPU 核心配置 ===
        device=0,           # 【强制】指定使用第 0 号显卡
        batch=16,           # 批次大小 (显存如果只有 4G，建议改为 8 或 4)
        workers=2,          # Windows 下建议设为 2，设太高容易报错
        # === 👇 重点：加这一行解决卡死 ===
        amp=False,          # 【强制关闭混合精度】，解决 4090 卡 AMP 问题
        # =================================
        # === 🎨 解决红粉撞色的严谨参数 ===
        hsv_h=0.0,          # 【核心】关闭色相抖动
        hsv_s=0.2,          # 饱和度微调
        hsv_v=0.3,          # 亮度微调
        
        # === 🏗️ 增强策略 ===
        mosaic=1.0,         # 开启马赛克增强
        degrees=0.0,        # 关闭旋转
        scale=0.5,          # 缩放增强
        
        # === 📁 保存设置 ===
        project='Logistics_Project',
        name='gpu_run_v1'
    )
    
    print("\n✅ 训练结束！")
    print(f"最佳模型保存在: Logistics_Project/gpu_run_v1/weights/best.pt")

if __name__ == '__main__':
    # Windows 下必须放在 if __name__ == '__main__': 下运行，否则多进程会报错
    train_model()