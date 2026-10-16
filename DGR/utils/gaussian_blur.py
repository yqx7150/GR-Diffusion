import torch
import torch.nn as nn

def gaussian_blur_2d(
    x: torch.Tensor, 
    kernel_size: int = 5, 
    sigma: float = 1.0,
    device: torch.device = None
) -> torch.Tensor:
    """
    对GPU上的2D张量（图像）添加高斯模糊，支持梯度反向传播
    
    Args:
        x: 输入张量，shape=[B, C, H, W]（批量数, 通道数, 高度, 宽度），需在GPU上
        kernel_size: 高斯核大小（奇数，如3/5/7）
        sigma: 高斯核标准差
        device: 设备（自动识别为x的设备，无需手动指定）
    
    Returns:
        模糊后的张量，shape与输入一致，在同一GPU上，梯度可回传
    """
    # 自动获取输入张量的设备（GPU/CPU）
    if device is None:
        device = x.device
    
    # 1. 校验参数
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size必须为奇数！")
    if not x.is_cuda and device.type == 'cuda':
        raise RuntimeError("输入张量未在GPU上！请先将张量移至GPU（x = x.cuda()）")
    
    # 2. 生成高斯核（1D -> 2D）
    # 生成1D高斯分布
    kernel_1d = torch.arange(kernel_size, dtype=torch.float32, device=device)
    kernel_1d -= (kernel_size - 1) / 2  # 中心置0
    kernel_1d = torch.exp(-0.5 * (kernel_1d / sigma) ** 2)  # 高斯公式
    kernel_1d /= kernel_1d.sum()  # 归一化
    
    # 生成2D高斯核（外积）
    kernel_2d = kernel_1d.unsqueeze(1) @ kernel_1d.unsqueeze(0)  # [K, K]
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
    
    # 3. 扩展核以匹配输入通道数（每个通道用相同的核）
    num_channels = x.shape[1]
    kernel = kernel_2d.repeat(num_channels, 1, 1, 1)  # [C, 1, K, K]
    
    # 4. 高斯模糊（卷积操作，padding保证输出尺寸不变）
    # 使用groups=C实现逐通道卷积，保证多通道图像每个通道独立模糊
    blur = nn.functional.conv2d(
        input=x,
        weight=kernel,  # 核张量（可微分）
        bias=None,
        stride=1,
        padding=(kernel_size - 1) // 2,  # 保持输出尺寸与输入一致
        groups=num_channels
    )
    
    return blur

def gaussian_blur_3d(
    x: torch.Tensor, 
    kernel_size: int = 5, 
    sigma: float = 1.0,
    device: torch.device = None
) -> torch.Tensor:
    """
    对GPU上的3D张量（如[B, C, D, H, W]）添加高斯模糊，支持梯度反向传播
    
    Args:
        x: 输入张量，shape=[B, C, D, H, W]（批量数, 通道数, 深度, 高度, 宽度），需在GPU上
        kernel_size: 高斯核大小（必须为奇数，如3/5/7）
        sigma: 高斯核标准差
        device: 设备（自动识别为x的设备，无需手动指定）
    
    Returns:
        模糊后的3D张量，shape与输入一致，在同一GPU上，梯度可正常回传
    """
    # 自动获取输入张量的设备（GPU/CPU）
    if device is None:
        device = x.device
    
    # 1. 参数校验
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size必须为奇数！")
    if len(x.shape) != 5:
        raise ValueError("输入张量必须是5维[B, C, D, H, W]！")
    if not x.is_cuda and device.type == 'cuda':
        raise RuntimeError("输入张量未在GPU上！请先将张量移至GPU（x = x.cuda()）")
    
    # 2. 生成3D高斯核（1D -> 3D）
    # 生成1D高斯分布（在GPU上生成，避免数据拷贝）
    kernel_1d = torch.arange(kernel_size, dtype=torch.float32, device=device)
    kernel_1d -= (kernel_size - 1) / 2  # 核中心置0
    kernel_1d = torch.exp(-0.5 * (kernel_1d / sigma) ** 2)  # 高斯公式
    kernel_1d /= kernel_1d.sum()  # 归一化，保证核的权重和为1
    
    # 生成3D高斯核（三重外积：1D ⊗ 1D ⊗ 1D）
    kernel_3d = kernel_1d.unsqueeze(1).unsqueeze(2) @ \
                kernel_1d.unsqueeze(0).unsqueeze(2) @ \
                kernel_1d.unsqueeze(0).unsqueeze(1)
    kernel_3d = kernel_3d.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K, K]
    
    # 3. 扩展核以匹配输入通道数（每个通道用相同的3D高斯核）
    num_channels = x.shape[1]
    kernel = kernel_3d.repeat(num_channels, 1, 1, 1, 1)  # [C, 1, K, K, K]
    
    # 4. 3D高斯模糊（卷积操作，padding保证输出尺寸不变）
    # groups=C实现逐通道3D卷积，多通道独立模糊
    blur = nn.functional.conv3d(
        input=x,
        weight=kernel,  # 核张量（可微分，支持梯度回传）
        bias=None,
        stride=1,
        padding=(kernel_size - 1) // 2,  # 3个维度同时padding，保证尺寸不变
        groups=num_channels
    )
    
    return blur