"""
测试 SDAR 模型的 Info-Gain sampler。

用于检查 Info-Gain sampler 是否能正常生成，排除问题。
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from dllm.dllm.pipelines.info_gain.sdar.sampler import InfoGainSDARSampler, InfoGainSDARSamplerConfig

def test_sdar_infogain():
    model_path = "./model/sdar"
    
    print("=" * 80)
    print("测试 SDAR 模型的 Info-Gain sampler")
    print("=" * 80)
    
    # 加载模型和 tokenizer
    print(f"\n1. 加载模型: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"   ✓ Tokenizer 加载成功")
        print(f"   - EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
        print(f"   - PAD token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
        print(f"   - Mask token: {getattr(tokenizer, 'mask_token', 'N/A')} (id: {getattr(tokenizer, 'mask_token_id', 'N/A')})")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
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
    
    # 创建 sampler
    print(f"\n2. 创建 Info-Gain sampler")
    try:
        from dllm.core.schedulers import LinearAlphaScheduler
        
        config = InfoGainSDARSamplerConfig(
            candidate_number=8,
            position_temperature=0.1,
            temperature=0.0,
            threshold=0.9,
            variant="info_gain",
            eos_penalty=4.0,
            pad_penalty=4.0,
        )
        sampler = InfoGainSDARSampler(
            model=model,
            tokenizer=tokenizer,
            scheduler=LinearAlphaScheduler(),  # 显式提供 scheduler
        )
        print(f"   ✓ Sampler 创建成功")
        print(f"   - Candidate number: {config.candidate_number}")
        print(f"   - Position temperature: {config.position_temperature}")
        print(f"   - EOS penalty: {config.eos_penalty}")
        print(f"   - Pad penalty: {config.pad_penalty}")
        print(f"   - Scheduler: {type(sampler.scheduler).__name__}")
    except Exception as e:
        print(f"   ✗ Sampler 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试输入
    print(f"\n3. 准备测试输入")
    
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
        test_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(f"   应用 chat template 后的 prompt:")
        print(f"   {test_prompt[:200]}...")
    else:
        # 直接使用文本
        test_prompt = "Hello, how are you?"
        print(f"   使用直接文本: {test_prompt}")
    
    # Tokenize
    inputs = tokenizer(test_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    
    print(f"\n   Tokenized:")
    print(f"   Input IDs shape: {input_ids.shape}")
    print(f"   Input IDs: {input_ids[0].tolist()[:20]}... (共 {input_ids.shape[1]} tokens)")
    
    # 显示 tokenized 的文本（用于调试）
    decoded_input = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    print(f"   解码后的输入（包含特殊 token）:")
    print(f"   {decoded_input[:200]}...")
    
    # 测试采样
    print(f"\n4. 测试 Info-Gain 采样")
    try:
        # 设置 max_new_tokens
        max_new_tokens = 50
        
        with torch.no_grad():
            outputs = sampler.sample(
                inputs=[input_ids[0]],
                config=config,  # 传递 config
                max_new_tokens=max_new_tokens,
                use_cache=None,  # no-cache mode
            )
        
        print(f"   ✓ 采样成功")
        print(f"   Output shape: {outputs.shape}")
        print(f"   Output IDs: {outputs[0].tolist()[:30]}...")  # 只显示前30个
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"\n5. 生成结果（包含特殊 token）:")
        print(f"   {generated_text[:400]}...")
        
        generated_text_clean = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n6. 生成结果（清理后）:")
        print(f"   {generated_text_clean}")
        
        # 如果使用了 chat template，尝试提取 assistant 的回复
        if has_chat_template:
            print(f"\n7. 提取 assistant 回复:")
            # 找到 assistant 开始的位置
            assistant_start_marker = "<|im_start|>assistant\n"
            if assistant_start_marker in generated_text_clean:
                assistant_reply = generated_text_clean.split(assistant_start_marker, 1)[1]
                # 移除可能的结束标记
                if "<|im_end|>" in assistant_reply:
                    assistant_reply = assistant_reply.split("<|im_end|>")[0]
                print(f"   {assistant_reply}")
            else:
                # 尝试其他可能的标记
                if "assistant" in generated_text_clean.lower():
                    print(f"   找到 'assistant' 关键字，但格式可能不同")
                else:
                    print(f"   未找到 assistant 标记")
        
        # 检查是否有大量 EOS
        eos_count = (outputs[0] == tokenizer.eos_token_id).sum().item()
        print(f"\n8. 统计:")
        print(f"   - 总 token 数: {outputs.shape[1]}")
        print(f"   - EOS token 数: {eos_count}")
        if outputs.shape[1] > 0:
            print(f"   - EOS 比例: {eos_count / outputs.shape[1] * 100:.2f}%")
        
        # 检查是否有 mask token
        mask_token_id = getattr(tokenizer, 'mask_token_id', None)
        if mask_token_id is not None:
            mask_count = (outputs[0] == mask_token_id).sum().item()
            print(f"   - Mask token 数: {mask_count}")
            if mask_count > 0:
                print(f"   ⚠️  警告：输出中仍有 {mask_count} 个 mask token，可能生成未完成")
        
        # 检查输出是否为空或只有 prompt
        if outputs.shape[1] <= input_ids.shape[1]:
            print(f"   ⚠️  警告：输出长度 ({outputs.shape[1]}) 小于等于输入长度 ({input_ids.shape[1]})")
            print(f"   可能没有生成新内容")
        
    except Exception as e:
        print(f"   ✗ 采样失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == "__main__":
    test_sdar_infogain()

