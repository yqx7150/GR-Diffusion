
def range_interval(lower_bound, higher_bound, num_samples):
    # 参数合法性校验：样本数需为正整数，且不超过999（避免间隔为0）
    assert isinstance(num_samples, int), "样本数量必须是整数"
    assert num_samples <= higher_bound - lower_bound, "样本数量最多为1000（覆盖0-999所有数）"
    
    # 计算间隔：总范围（999-0） / (样本数-1)，保证首尾都能取到
    if num_samples == 1:
        # 特殊情况：只取1个样本时，默认返回[0]（也可改为[999]，按需调整）
        return [0]
    
    interval = (higher_bound - lower_bound) / (num_samples - 1)
    
    # 生成等间隔数字列表
    sample_list = []
    for i in range(num_samples):
        # 计算第i个样本的数值，四舍五入为整数（避免小数）
        num = round(i * interval)
        # 确保数值不超出0-999范围（极端情况浮点精度问题）
        num = max(lower_bound, min(higher_bound, num))
        sample_list.append(num)
    
    return sample_list
