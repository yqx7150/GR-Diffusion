import torch
import torch.nn.functional as F

def apply_3d_gaussian_smooth(
    x: torch.Tensor,
    k: int = 3,          # 高斯核大小（必须为奇数）
    sigma: float = 1.0,  # 标准差，控制平滑程度（可选）
    padding_mode: str = "replicate"  # 边缘填充模式，避免边界失真
) -> torch.Tensor:
    """
    对3D张量执行3D高斯平滑（适配shape: [1, C, D, H, W]）
    :param x: 输入张量，shape=(1, C, D, H, W)
    :param k: 高斯核大小（必须为奇数，如3/5/7）
    :param sigma: 高斯核标准差，σ越大平滑越强（默认1.0）
    :param padding_mode: 边缘填充模式（replicate/zeros/reflection，默认replicate避免边界失真）
    :return: 平滑后的张量，shape与输入一致：(1, C, D, H, W)
    """
    # ========== 输入校验 ==========
    # 校验张量维度（必须是5维：1, C, D, H, W）
    assert x.dim() == 5, f"输入张量必须是5维，当前维度：{x.dim()}"
    assert x.shape[0] == 1, f"输入张量batch维度必须为1，当前：{x.shape[0]}"
    # 校验核大小（必须为奇数）
    assert k % 2 == 1, f"高斯核大小k必须为奇数，当前：{k}"
    
    # ========== 提取参数 ==========
    device = x.device
    dtype = x.dtype
    C = x.shape[1]  # 输入通道数
    
    # ========== 生成3D高斯核 ==========
    half_k = k // 2
    # 1. 生成3D坐标网格
    coords = torch.linspace(-half_k, half_k, k, device=device, dtype=dtype)
    z, y, x_axis = torch.meshgrid(coords, coords, coords, indexing="ij")
    # 2. 计算3D高斯分布
    gaussian = torch.exp(-(z**2 + y**2 + x_axis**2) / (2 * sigma**2))
    # 3. 归一化（核内总和=1，避免改变张量强度）
    gaussian = gaussian / gaussian.sum()
    # 4. 调整核形状：适配多通道卷积（out_ch=C, in_ch=1, k, k, k）
    #    每个通道使用相同的高斯核（各通道独立平滑）
    gaussian_kernel = gaussian.unsqueeze(0).unsqueeze(0)  # (1,1,k,k,k)
    gaussian_kernel = gaussian_kernel.repeat(C, 1, 1, 1, 1)  # (C,1,k,k,k)
    
    # ========== 执行3D高斯平滑 ==========
    # 1. 边缘填充（避免边界像素失真）
    padding = (half_k, half_k, half_k, half_k, half_k, half_k)  # (padD_left, padD_right, padH_left, padH_right, padW_left, padW_right)
    x_padded = F.pad(x, padding, mode=padding_mode)
    
    # 2. 分组卷积（保证每个通道独立用高斯核平滑）
    #    groups=C：每个通道对应一个卷积核，避免通道间干扰
    x_smoothed = F.conv3d(
        input=x_padded,
        weight=gaussian_kernel,
        bias=None,
        stride=1,
        padding=0,  # 已手动pad，这里设0
        groups=C    # 多通道独立卷积
    )
    
    return x_smoothed

def gaussian_mse_cond_fn(x, t, low_res, kernels, sigmas, weights=[0.2, 0.2, 0.6], guided_scale=1.0):
    assert len(kernels) == len(sigmas) == len(weights)
    with torch.enable_grad():
        x_in = x.clone().requires_grad_(True)
        loss = 0.
        for k, s, w in zip(kernels, sigmas, weights):
            x_g = apply_3d_gaussian_smooth(x_in, k, s)
            y_g = apply_3d_gaussian_smooth(low_res, k, s)
            score = F.mse_loss(x_g, y_g)
            loss += score * w
        loss = loss / len(kernels)
        grad = torch.autograd.grad(
            outputs=loss,
            inputs=x_in,
            grad_outputs=torch.ones_like(score),  # 必须加，否则梯度维度不匹配
            create_graph=True,                    # 扩散采样必需（二阶梯度）
            retain_graph=True,                    # 防止计算图释放
            allow_unused=True                     # 兼容边界情况
        )[0]

    return grad * guided_scale