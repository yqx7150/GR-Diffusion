import torch.nn.functional as F
import torch
from icecream import ic

def low_res_mae_cond_fn(x, t, low_res, guided_scale=1.0):
    with torch.enable_grad():
        x_in = x.clone().requires_grad_(True)
        score = F.l1_loss(x_in, low_res)
        grad = torch.autograd.grad(
            outputs=score,
            inputs=x_in,
            grad_outputs=torch.ones_like(score),  # 必须加，否则梯度维度不匹配
            create_graph=True,                    # 扩散采样必需（二阶梯度）
            retain_graph=True,                    # 防止计算图释放
            allow_unused=True                     # 兼容边界情况
        )[0]

    return grad * guided_scale
    