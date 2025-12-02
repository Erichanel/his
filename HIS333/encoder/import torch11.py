import torch

print("===== PyTorch 验证结果 =====")
print(f"PyTorch版本：{torch.__version__}")
print(f"CUDA是否可用：{torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"可用GPU数量：{torch.cuda.device_count()}")
    print(f"当前GPU名称：{torch.cuda.get_device_name(0)}")
else:
    print("提示：仅安装了CPU版本PyTorch，或GPU/CUDA配置异常")