"""
测试 build_chat_inputs 的返回值格式。
"""

from transformers import AutoTokenizer

def test_build_chat_inputs():
    model_path = "./model/sdar"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 测试 build_chat_inputs
    messages = [
        {"role": "user", "content": "Hello"}
    ]
    
    print("=" * 80)
    print("测试 build_chat_inputs")
    print("=" * 80)
    
    # 方法 1: 使用 apply_chat_template (tokenize=True)
    inputs1 = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
    )
    print(f"\n1. apply_chat_template(tokenize=True):")
    print(f"   类型: {type(inputs1)}")
    print(f"   值: {inputs1}")
    if isinstance(inputs1, list):
        print(f"   长度: {len(inputs1)}")
        if len(inputs1) > 0:
            print(f"   第一个元素类型: {type(inputs1[0])}")
            if isinstance(inputs1[0], list):
                print(f"   第一个元素（列表）: {inputs1[0][:20]}...")
            else:
                print(f"   第一个元素（非列表）: {inputs1[0]}")
    
    # 方法 2: 使用 apply_chat_template (tokenize=False) 然后手动 tokenize
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs2 = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    print(f"\n2. apply_chat_template(tokenize=False) + tokenizer():")
    print(f"   类型: {type(inputs2)}")
    print(f"   值: {inputs2}")
    if hasattr(inputs2, 'shape'):
        print(f"   Shape: {inputs2.shape}")
        print(f"   值: {inputs2[0].tolist()[:20]}...")
    
    # 方法 3: 直接 tokenize 文本
    inputs3 = tokenizer([prompt], add_special_tokens=False)["input_ids"]
    print(f"\n3. tokenizer([prompt], add_special_tokens=False):")
    print(f"   类型: {type(inputs3)}")
    print(f"   值: {inputs3}")
    if hasattr(inputs3, 'shape'):
        print(f"   Shape: {inputs3.shape}")
        print(f"   值: {inputs3[0].tolist()[:20]}...")
    
    # 检查 sampler.sample 期望的格式
    print(f"\n4. sampler.sample 期望的格式:")
    print(f"   - inputs 应该是一个列表")
    print(f"   - inputs[0] 可以是 list 或 tensor")
    print(f"   - 如果 inputs[0] 是 list，会被转换为 tensor")
    
    # 测试不同的格式
    print(f"\n5. 测试不同格式:")
    
    # 格式 A: 列表的列表
    format_a = [inputs1] if isinstance(inputs1, list) else [[inputs1]]
    print(f"   格式 A (列表的列表): {type(format_a)}")
    if len(format_a) > 0:
        print(f"   format_a[0] 类型: {type(format_a[0])}")
    
    # 格式 B: tensor 的列表
    import torch
    if isinstance(inputs2, torch.Tensor):
        format_b = [inputs2[0]]
    else:
        format_b = [torch.tensor(inputs2[0])]
    print(f"   格式 B (tensor 的列表): {type(format_b)}")
    if len(format_b) > 0:
        print(f"   format_b[0] 类型: {type(format_b[0])}")
        print(f"   format_b[0] shape: {format_b[0].shape}")
    
    # 格式 C: 列表的列表（从 tokenizer 返回）
    format_c = inputs3
    print(f"   格式 C (tokenizer 返回): {type(format_c)}")
    if hasattr(format_c, '__len__') and len(format_c) > 0:
        print(f"   format_c[0] 类型: {type(format_c[0])}")
        if hasattr(format_c[0], 'shape'):
            print(f"   format_c[0] shape: {format_c[0].shape}")
    
    print(f"\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == "__main__":
    test_build_chat_inputs()

