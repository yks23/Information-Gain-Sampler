"""
最简单的 SDAR 模型测试脚本。

直接调用模型，检查基本功能。
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_simple():
    model_path = "./model/sdar"
    
    print("=" * 80)
    print("最简单的 SDAR 模型测试")
    print("=" * 80)
    
    # 加载模型和 tokenizer
    print(f"\n1. 加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"   ✓ Tokenizer 加载成功")
    print(f"   - EOS: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    print(f"   - PAD: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    print(f"   - MASK: {getattr(tokenizer, 'mask_token', 'N/A')} (id: {getattr(tokenizer, 'mask_token_id', 'N/A')})")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    print(f"   ✓ Model 加载成功")
    print(f"   - Device: {next(model.parameters()).device}")
    
    # 测试输入
    print(f"\n2. 准备输入")
    
    # 检查是否有 chat template
    has_chat_template = tokenizer.chat_template is not None
    print(f"   Chat template 可用: {has_chat_template}")
    
    if has_chat_template:
        # 使用对话格式
        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]
        print(f"   使用对话格式:")
        print(f"   Messages: {messages}")
        
        # 应用 chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(f"   应用 chat template 后的 prompt:")
        print(f"   {prompt[:200]}...")
    else:
        # 直接使用文本
        prompt = "Hello, how are you?"
        print(f"   使用直接文本: {prompt}")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    
    print(f"\n   Tokenized:")
    print(f"   Input IDs: {input_ids[0].tolist()[:20]}... (共 {input_ids.shape[1]} tokens)")
    print(f"   Input shape: {input_ids.shape}")
    
    # 显示 tokenized 的文本（用于调试）
    decoded_input = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    print(f"   解码后的输入（包含特殊 token）:")
    print(f"   {decoded_input[:200]}...")
    
    # 测试 forward
    print(f"\n3. 测试 forward（单步）")
    with torch.no_grad():
        # 生成 position_ids
        position_ids = torch.arange(input_ids.shape[1], device=model.device).unsqueeze(0)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
    
    print(f"   ✓ Forward 成功")
    print(f"   Logits shape: {outputs.logits.shape}")
    
    # 检查最后一个位置的 logits
    last_logits = outputs.logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)
    
    print(f"\n4. 最后一个位置的 top 10 tokens:")
    top_probs, top_indices = torch.topk(probs, 10)
    for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
        token = tokenizer.decode([idx.item()])
        print(f"   {i+1:2d}. {token!r:20s} (id: {idx.item():6d}, prob: {prob.item():.4f})")
    
    # 应用 EOS penalty
    print(f"\n5. 应用 EOS penalty (4.0) 后:")
    last_logits_penalized = last_logits.clone()
    last_logits_penalized[tokenizer.eos_token_id] -= 4.0
    if tokenizer.pad_token_id is not None:
        last_logits_penalized[tokenizer.pad_token_id] -= 4.0
    probs_penalized = torch.softmax(last_logits_penalized, dim=-1)
    
    top_probs_penalized, top_indices_penalized = torch.topk(probs_penalized, 10)
    for i, (idx, prob) in enumerate(zip(top_indices_penalized, top_probs_penalized)):
        token = tokenizer.decode([idx.item()])
        print(f"   {i+1:2d}. {token!r:20s} (id: {idx.item():6d}, prob: {prob.item():.4f})")
    
    # 手动生成几个 token
    print(f"\n6. 手动生成 20 个 token:")
    current_ids = input_ids.clone()
    current_attn = attention_mask.clone() if attention_mask is not None else None
    generated = []
    
    for step in range(20):
        pos_ids = torch.arange(current_ids.shape[1], device=model.device).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(
                input_ids=current_ids,
                attention_mask=current_attn,
                position_ids=pos_ids,
                use_cache=False,
            )
        
        next_logits = outputs.logits[0, -1, :]
        
        # 应用 EOS penalty
        next_logits[tokenizer.eos_token_id] -= 4.0
        if tokenizer.pad_token_id is not None:
            next_logits[tokenizer.pad_token_id] -= 4.0
        
        # 采样
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        
        generated.append(next_token)
        token_str = tokenizer.decode([next_token])
        
        print(f"   Step {step+1:2d}: {token_str!r:20s} (id: {next_token:6d})")
        
        if next_token == tokenizer.eos_token_id:
            print(f"   → 遇到 EOS，停止")
            break
        
        # 更新
        current_ids = torch.cat([current_ids, torch.tensor([[next_token]], device=model.device)], dim=1)
        if current_attn is not None:
            current_attn = torch.cat([current_attn, torch.ones((1, 1), device=model.device, dtype=current_attn.dtype)], dim=1)
    
    # 解码完整结果
    print(f"\n7. 完整生成结果:")
    full_ids = torch.cat([input_ids[0], torch.tensor(generated, device=model.device)])
    full_text = tokenizer.decode(full_ids, skip_special_tokens=False)
    print(f"   完整序列（包含特殊 token）:")
    print(f"   {full_text[:300]}...")
    
    full_text_clean = tokenizer.decode(full_ids, skip_special_tokens=True)
    print(f"\n8. 清理后的结果:")
    print(f"   {full_text_clean}")
    
    # 如果使用了 chat template，尝试提取 assistant 的回复
    if has_chat_template:
        print(f"\n9. 提取 assistant 回复:")
        # 找到 assistant 开始的位置
        assistant_start_marker = "<|im_start|>assistant\n"
        if assistant_start_marker in full_text_clean:
            assistant_reply = full_text_clean.split(assistant_start_marker, 1)[1]
            # 移除可能的结束标记
            if "<|im_end|>" in assistant_reply:
                assistant_reply = assistant_reply.split("<|im_end|>")[0]
            print(f"   {assistant_reply}")
        else:
            # 尝试其他可能的标记
            if "assistant" in full_text_clean.lower():
                print(f"   找到 'assistant' 关键字，但格式可能不同")
                print(f"   显示完整结果")
            else:
                print(f"   未找到 assistant 标记，显示完整结果")
    
    print(f"\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == "__main__":
    test_simple()

