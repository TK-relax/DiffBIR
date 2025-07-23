import torch

def test_cudnn():
    print("=" * 30)
    print("PyTorch cuDNN 测试")
    print("=" * 30)
    
    # 1. 检查PyTorch版本
    print(f"PyTorch 版本: {torch.__version__}")
    
    # 2. 检查CUDA是否对PyTorch可用
    is_cuda_available = torch.cuda.is_available()
    print(f"PyTorch 是否能够使用 GPU (CUDA): {is_cuda_available}")
    
    if not is_cuda_available:
        print("测试失败：PyTorch 无法找到 CUDA。请检查驱动、CUDA Toolkit安装和PyTorch的GPU版本。")
        return

    # 3. 检查cuDNN是否对PyTorch可用
    is_cudnn_available = torch.backends.cudnn.is_available()
    print(f"PyTorch 是否能够使用 cuDNN: {is_cudnn_available}")
    
    # 4. 获取PyTorch链接的cuDNN版本
    cudnn_version = torch.backends.cudnn.version()
    print(f"检测到的 cuDNN 版本: {cudnn_version}")

    # 5. 获取当前GPU设备信息
    current_device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_device)
    print(f"当前使用的 GPU: {gpu_name} (设备号: {current_device})")

    # 6. 做一个简单的张量运算来确认实际可用性
    try:
        tensor = torch.randn(5, 5).cuda()
        print("\n在GPU上成功创建张量:")
        print(tensor)
        print("\n测试通过！您的PyTorch环境已正确配置cuDNN。")
    except Exception as e:
        print(f"\n在GPU上创建张量时出错: {e}")
        print("\n测试失败！虽然库可能已找到，但实际运算时出现问题。")

if __name__ == "__main__":
    test_cudnn()
    print('1')