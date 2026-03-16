"""
直接测试 SDAR 模型的 generate 方法。

用于检查模型的基本生成能力，排除 Info-Gain sampler 的问题。
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

def test_sdar_generate():
    model_path = "./model/sdar"
    
    print("=" * 80)
    print("测试 SDAR 模型的基本 generate 方法")
    print("=" * 80)
    
    # 加载模型和 tokenizer
    print(f"\n1. 加载模型: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print(f"   ✓ Tokenizer 加载成功")
        print(f"   - EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
        print(f"   - PAD token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
        print(f"   - Mask token: {getattr(tokenizer, 'mask_token', 'N/A')} (id: {getattr(tokenizer, 'mask_token_id', 'N/A')})")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model.eval()
        print(f"   ✓ Model 加载成功")
        print(f"   - Device: {next(model.parameters()).device}")
        print(f"   - Dtype: {next(model.parameters()).dtype}")
    except Exception as e:
        print(f"   ✗ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试输入
    print(f"\n2. 准备测试输入")
    test_prompt = "Hello, how are you?"
    print(f"   Prompt: {test_prompt}")
    
    # Tokenize
    inputs = tokenizer(test_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    
    print(f"   Input IDs shape: {input_ids.shape}")
    print(f"   Input IDs: {input_ids[0].tolist()[:20]}...")  # 只显示前20个
    
    # 测试 forward（单步生成）
    print(f"\n3. 测试 forward 方法（单步生成）")
    try:
        # 生成 position_ids
        position_ids = torch.arange(input_ids.shape[1], device=model.device).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
        
        print(f"   ✓ Forward 成功")
        print(f"   Logits shape: {outputs.logits.shape}")
        
        # 检查 logits 中 EOS 的概率
        last_logits = outputs.logits[0, -1, :]
        eos_logit = last_logits[tokenizer.eos_token_id].item()
        probs = torch.softmax(last_logits, dim=-1)
        eos_prob = probs[tokenizer.eos_token_id].item()
        
        print(f"   - EOS logit: {eos_logit:.4f}")
        print(f"   - EOS probability: {eos_prob:.4f}")
        print(f"   - Top 10 tokens:")
        top_probs, top_indices = torch.topk(probs, 10)
        for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
            token = tokenizer.decode([idx.item()])
            print(f"     {i+1}. {token!r} (id: {idx.item()}, prob: {prob.item():.4f})")
        
        # 应用 EOS penalty 后检查
        print(f"\n4. 应用 EOS penalty (4.0) 后:")
        last_logits_penalized = last_logits.clone()
        last_logits_penalized[tokenizer.eos_token_id] -= 4.0
        probs_penalized = torch.softmax(last_logits_penalized, dim=-1)
        eos_prob_penalized = probs_penalized[tokenizer.eos_token_id].item()
        print(f"   - EOS probability (after penalty): {eos_prob_penalized:.4f}")
        print(f"   - Top 10 tokens (after penalty):")
        top_probs_penalized, top_indices_penalized = torch.topk(probs_penalized, 10)
        for i, (idx, prob) in enumerate(zip(top_indices_penalized, top_probs_penalized)):
            token = tokenizer.decode([idx.item()])
            print(f"     {i+1}. {token!r} (id: {idx.item()}, prob: {prob.item():.4f})")
        
    except Exception as e:
        print(f"   ✗ Forward 失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试多步生成（手动循环）
    print(f"\n5. 测试多步生成（手动循环，模拟 generate）")
    try:
        current_ids = input_ids.clone()
        current_attn = attention_mask.clone() if attention_mask is not None else None
        generated_tokens = []
        
        for step in range(20):  # 生成 20 个 token
            # 生成 position_ids
            pos_ids = torch.arange(current_ids.shape[1], device=model.device).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(
                    input_ids=current_ids,
                    attention_mask=current_attn,
                    position_ids=pos_ids,
                    use_cache=False,
                )
            
            # 获取最后一个位置的 logits
            next_logits = outputs.logits[0, -1, :]
            
            # 应用 EOS penalty
            next_logits[tokenizer.eos_token_id] -= 4.0
            if tokenizer.pad_token_id is not None:
                next_logits[tokenizer.pad_token_id] -= 4.0
            
            # 采样
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            generated_tokens.append(next_token)
            
            # 检查是否遇到 EOS
            if next_token == tokenizer.eos_token_id:
                print(f"   - Step {step+1}: 遇到 EOS，停止生成")
                break
            
            # 更新 input_ids
            current_ids = torch.cat([current_ids, torch.tensor([[next_token]], device=model.device)], dim=1)
            if current_attn is not None:
                current_attn = torch.cat([current_attn, torch.ones((1, 1), device=model.device, dtype=current_attn.dtype)], dim=1)
            
            if step < 5:  # 只显示前5步
                token_str = tokenizer.decode([next_token])
                prob = probs[next_token].item()
                print(f"   - Step {step+1}: {token_str!r} (id: {next_token}, prob: {prob:.4f})")
        
        # Decode 生成的结果
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
        print(f"\n6. 生成结果（包含特殊 token）:")
        print(f"   {generated_text}")
        
        generated_text_clean = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"\n7. 生成结果（清理后）:")
        print(f"   {generated_text_clean}")
        
        # 统计
        eos_count = sum(1 for t in generated_tokens if t == tokenizer.eos_token_id)
        print(f"\n8. 统计:")
        print(f"   - 总生成 token 数: {len(generated_tokens)}")
        print(f"   - EOS token 数: {eos_count}")
        
    except Exception as e:
        print(f"   ✗ 多步生成失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    
    print(f"\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == "__main__":
    test_sdar_generate()

