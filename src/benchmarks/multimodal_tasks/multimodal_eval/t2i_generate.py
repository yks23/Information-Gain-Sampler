# coding=utf-8
# Copyright 2025 MMaDA Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MMaDA Text-to-Image Generation Script (Info-Gain 版)

功能：
  1. 从文本提示生成图像（支持 GenEval 格式 / 普通格式输出）
  2. 支持多 GPU 并行生成
  3. 使用 Info-Gain (瞬时熵 + 下一步平均熵) 采样策略

用法：
  python t2i_generate.py \
      config=configs/mmada_t2i.yaml \
      mmada_model_path=../mmada-mix \
      vq_model_path=../magvitv2 \
      validation_prompts_file=prompts/generation_prompts.txt \
      output_dir=./output_images \
      batch_size=1 \
      text_to_image.samples_per_prompt=4 \
      text_to_image.guidance_scale=3.5 \
      text_to_image.generation_timesteps=50 \
      text_to_image.generation_temperature=1.0 \
      candidate_number=4
"""

import os
import sys

# 将当前脚本所在目录加入路径，以便导入本地 mmada_utils 模块
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# 获取项目根目录（向上3级：multimodal_eval -> multimodal_tasks -> benchmarks -> src -> project_root）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))

os.environ["TOKENIZERS_PARALLELISM"] = "true"

from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import json
import multiprocessing

# 从本地 mmada_utils 导入（不再依赖外部 MMaDA 项目）
from mmada_utils.models import MAGVITv2, MMadaModelLM, get_mask_schedule
from mmada_utils.training.prompting_utils import UniversalPrompting
from mmada_utils.training.utils import get_config
from transformers import AutoTokenizer


# ============================================================
# 辅助函数
# ============================================================

def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")


# ============================================================
# Info-Gain T2I 生成
# ============================================================

@torch.no_grad()
def t2i_info_gain_generate(
    model,
    input_ids,
    uncond_input_ids=None,
    attention_mask=None,
    uncond_attention_mask=None,
    guidance_scale=3.5,
    temperature=1.0,
    timesteps=50,
    seq_len=1024,
    mask_token_id=None,
    candidate_number=4,
    codebook_size=8192,
    llm_vocab_size=126464,
):
    """
    T2I masked-diffusion generation with Info-Gain position selection.

    算法：每一步对候选 unmask 动作打分，选择得分最低的候选。
        score = 瞬时熵（被选位置的熵之和）
              + 下一步平均熵（剩余 mask 位置的平均熵）

    Args:
        model:                MMaDA 模型 (HuggingFace 格式, model(input_ids).logits).
        input_ids:            [B, L] 含 mask_token_id 占位的图像区域.
        uncond_input_ids:     [B, L] CFG 无条件输入 (可选).
        attention_mask:       [B, L] attention mask.
        uncond_attention_mask:[B, L] 无条件 attention mask.
        guidance_scale:       CFG 引导尺度.
        temperature:          Gumbel 噪声温度.
        timesteps:            去噪步数.
        seq_len:              图像 token 数量 (num_vq_tokens).
        mask_token_id:        Mask token ID.
        candidate_number:     Info-Gain 候选数量.
        codebook_size:        VQ codebook 大小.
        llm_vocab_size:       LLM 词表大小 (VQ offset).

    Returns:
        gen_token_ids: [B, seq_len] VQ codes (0-indexed).
    """
    device = input_ids.device
    batch_size = input_ids.shape[0]
    all_gen_ids = []

    for b in range(batch_size):
        x = input_ids[b : b + 1].clone()
        ux = uncond_input_ids[b : b + 1].clone() if uncond_input_ids is not None else None
        attn = attention_mask[b : b + 1] if attention_mask is not None else None
        u_attn = uncond_attention_mask[b : b + 1] if uncond_attention_mask is not None else None

        # 找到图像token的位置（在input_ids的末尾，长度为seq_len）
        # input_ids结构: [text_tokens, image_tokens (mask_token_id), ...]
        seq_length = x.shape[1]
        # 图像tokens在序列末尾，从后往前找seq_len个mask_token_id的位置
        # 更准确的方法：找到最后一个非mask位置，之后的所有mask都是图像tokens
        # 或者直接使用序列末尾的seq_len个位置
        image_start_idx = seq_length - seq_len
        image_pos = torch.arange(image_start_idx, seq_length, device=device)
        n_img = seq_len

        # 每步 unmask 的 token 数（均匀分配）
        base_k = n_img // timesteps
        rem = n_img % timesteps
        k_schedule = [base_k + (1 if i < rem else 0) for i in range(timesteps)]

        for step in range(timesteps):
            masked_pos = (x[0, image_pos] == mask_token_id).nonzero(as_tuple=True)[0]
            n_masked = len(masked_pos)
            if n_masked == 0:
                break

            k = min(k_schedule[step], n_masked)
            if k == 0:
                continue

            # 转换为完整序列中的位置
            masked_pos_abs = image_pos[masked_pos]

            # ---------- 前向推理 (with CFG) ----------
            logits = model(x, attention_mask=attn).logits
            if guidance_scale > 0 and ux is not None:
                u_logits = model(ux, attention_mask=u_attn).logits
                logits = u_logits + (1 + guidance_scale) * (logits - u_logits)

            # ---------- 在 mask 位置预测 token ----------
            m_logits = logits[0, masked_pos_abs].clone()
            # 限制预测范围到 VQ 词表
            m_logits[:, :llm_vocab_size] = float("-inf")
            if llm_vocab_size + codebook_size < m_logits.shape[-1]:
                m_logits[:, llm_vocab_size + codebook_size :] = float("-inf")

            if temperature > 0:
                m_f64 = m_logits.to(torch.float64)
                noise = torch.rand_like(m_f64)
                gumbel = (-torch.log(noise)) ** temperature
                x0 = torch.argmax(m_f64.exp() / gumbel, dim=-1)
            else:
                x0 = torch.argmax(m_logits, dim=-1)

            # ---------- 各 mask 位置的熵 ----------
            probs = F.softmax(m_logits.float(), dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)  # [n_masked]

            # ---------- Info-Gain 选择 ----------
            if k >= n_masked or candidate_number <= 1:
                # 无需 lookahead：直接选熵最低的位置
                _, si = torch.sort(entropy)
                sel = si[:k]
            else:
                cands_x, cands_ux, cands_sel = [], [], []
                for c in range(candidate_number):
                    if c == 0:
                        # 贪心：选熵最低的 k 个位置
                        _, si = torch.sort(entropy)
                        cs = si[:k]
                    else:
                        # 随机：Gumbel 扰动采样
                        gn = -torch.log(-torch.log(
                            torch.rand(n_masked, device=device) + 1e-10
                        ) + 1e-10)
                        _, cs = torch.topk(-entropy + gn, k=k)

                    cx = x.clone()
                    cx[0, masked_pos_abs[cs]] = x0[cs]
                    cands_x.append(cx)
                    cands_sel.append(cs)

                    if ux is not None:
                        cux = ux.clone()
                        cux[0, masked_pos_abs[cs]] = x0[cs]
                        cands_ux.append(cux)

                # 批量前向：lookahead
                nc = len(cands_x)
                x_batch = torch.cat(cands_x, dim=0)
                if attn is not None:
                    batch_attn = attn.expand(nc, -1)
                else:
                    batch_attn = None
                next_logits = model(x_batch, attention_mask=batch_attn).logits

                if guidance_scale > 0 and ux is not None and len(cands_ux) > 0:
                    ux_batch = torch.cat(cands_ux, dim=0)
                    if u_attn is not None:
                        batch_u_attn = u_attn.expand(nc, -1)
                    else:
                        batch_u_attn = None
                    u_next_logits = model(ux_batch, attention_mask=batch_u_attn).logits
                    next_logits = u_next_logits + (1 + guidance_scale) * (next_logits - u_next_logits)

                # 评估候选
                best_score = float("inf")
                sel = cands_sel[0]
                for c in range(nc):
                    # 瞬时熵：被选位置的熵之和
                    ie = entropy[cands_sel[c]].sum().item()
                    # 下一步平均熵：剩余 mask 位置的平均熵
                    rem_pos_rel = (cands_x[c][0, image_pos] == mask_token_id).nonzero(as_tuple=True)[0]
                    if len(rem_pos_rel) > 0:
                        rem_pos_abs = image_pos[rem_pos_rel]
                        rp = F.softmax(next_logits[c, rem_pos_abs].float(), dim=-1)
                        re = -torch.sum(rp * torch.log(rp + 1e-10), dim=-1)
                        ae = re.mean().item()
                    else:
                        ae = 0.0

                    score = ie + ae
                    if score < best_score:
                        best_score = score
                        sel = cands_sel[c]

            # ---------- 应用 unmask ----------
            x[0, masked_pos_abs[sel]] = x0[sel]
            if ux is not None:
                ux[0, masked_pos_abs[sel]] = x0[sel]

        # 提取 VQ codes（vocab_id → VQ code）
        # 从图像位置提取生成的token，减去llm_vocab_size得到VQ code
        gen = x[0, image_pos] - llm_vocab_size
        all_gen_ids.append(gen.unsqueeze(0))

    return torch.cat(all_gen_ids, dim=0)


# ============================================================
# 单 GPU 生成
# ============================================================

def generate_on_gpu(
    gpu_id,
    prompts_subset,
    prompt_start_idx,
    config,
    mmada_model_path,
    vq_model_path,
    output_dir,
    batch_size,
    samples_per_prompt,
    guidance_scale,
    generation_timesteps,
    generation_temperature,
    use_geneval_format,
    geneval_metadata,
    total_prompts_count,
    candidate_number=4,
):
    """在指定 GPU 上使用 Info-Gain 策略生成图像"""
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(mmada_model_path, padding_side="left")
    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=config.dataset.preprocessing.max_seq_length,
        special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>",
                        "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
        ignore_id=-100,
        cond_dropout_prob=config.training.cond_dropout_prob,
        use_reserved_token=True,
    )

    # 加载 VQ 模型
    vq_model = get_vq_model_class(config.model.vq_model.type)
    try:
        vq_model = vq_model.from_pretrained(vq_model_path, use_safetensors=None).to(device)
    except Exception:
        try:
            vq_model = vq_model.from_pretrained(vq_model_path, use_safetensors=True).to(device)
        except Exception:
            vq_model = vq_model.from_pretrained(vq_model_path, use_safetensors=False).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    # 加载 MMaDA 模型
    model = MMadaModelLM.from_pretrained(
        mmada_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    model.to(device)
    model.eval()

    mask_token_id = model.config.mask_token_id

    # 扩展提示列表（samples_per_prompt）
    expanded_prompts = []
    for prompt in prompts_subset:
        expanded_prompts.extend([prompt] * samples_per_prompt)

    total_images = len(expanded_prompts)
    print(f"[GPU {gpu_id}] 开始处理 {len(prompts_subset)} 个提示，共 {total_images} 张图像")

    image_count = 0
    num_batches = (len(expanded_prompts) + batch_size - 1) // batch_size

    # 创建进度条（多 GPU 时使用不同位置，单 GPU 时使用位置 0）
    pbar = tqdm(
        total=total_images,
        desc=f"[GPU {gpu_id}] 生成图像",
        unit="张",
        leave=True,
        position=gpu_id,
        ncols=100,
        mininterval=1.0,  # 至少 1 秒更新一次
    )

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(expanded_prompts))
            current_batch_size = end_idx - start_idx
            batch_prompts = expanded_prompts[start_idx:end_idx]

            image_tokens = torch.ones(
                (current_batch_size, config.model.mmada.num_vq_tokens),
                dtype=torch.long, device=device,
            ) * mask_token_id

            input_ids, attention_mask = uni_prompting((batch_prompts, image_tokens), 't2i_gen')

            if guidance_scale > 0:
                uncond_input_ids, uncond_attention_mask = uni_prompting(
                    ([''] * current_batch_size, image_tokens), 't2i_gen'
                )
            else:
                uncond_input_ids = None
                uncond_attention_mask = None

            # 获取 mask schedule
            if config.get("mask_schedule", None) is not None:
                schedule = config.mask_schedule.schedule
                args = config.mask_schedule.get("params", {})
                mask_schedule = get_mask_schedule(schedule, **args)
            else:
                mask_schedule = get_mask_schedule(config.training.get("mask_schedule", "cosine"))
            
            # 更新进度条描述（显示当前批次信息）
            pbar.set_description(f"[GPU {gpu_id}] 批次 {batch_idx+1}/{num_batches}")
            
            # 使用 model.t2i_generate() 方法（支持 IGP/Info-Gain）
            result = model.t2i_generate(
                input_ids=input_ids,
                uncond_input_ids=uncond_input_ids,
                attention_mask=attention_mask,
                uncond_attention_mask=uncond_attention_mask,
                guidance_scale=guidance_scale,
                temperature=generation_temperature,
                timesteps=generation_timesteps,
                noise_schedule=mask_schedule,
                noise_type=config.training.get("noise_type", "mask"),
                seq_len=config.model.mmada.num_vq_tokens,
                uni_prompting=uni_prompting,
                config=config,
                use_igp=True,  # 启用 Info-Gain Planner
                igp_num_candidates=candidate_number,
                igp_position_tau=config.get("position_temperature", 0.1),
                igp_heuristic=config.get("heuristic", "confidence"),
                igp_similarity_threshold=config.get("similarity_threshold", 0.5),
                igp_max_resample_attempts=config.get("max_resample_attempts", 3),
            )
            
            # 解析返回结果
            if isinstance(result, tuple):
                gen_token_ids = result[0]
            else:
                gen_token_ids = result

            # 解码图像
            gen_token_ids = torch.clamp(
                gen_token_ids, max=config.model.mmada.codebook_size - 1, min=0
            )
            images = vq_model.decode_code(gen_token_ids)

            images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
            images *= 255.0
            images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            pil_images = [Image.fromarray(image) for image in images]

            # 保存图像
            for i, pil_image in enumerate(pil_images):
                prompt = batch_prompts[i]
                global_prompt_idx = prompt_start_idx + (image_count // samples_per_prompt)
                sample_idx = image_count % samples_per_prompt

                if use_geneval_format and geneval_metadata:
                    prompt_dir = os.path.join(output_dir, f"{global_prompt_idx:05d}")
                    samples_dir = os.path.join(prompt_dir, "samples")
                    os.makedirs(samples_dir, exist_ok=True)

                    if sample_idx == 0:
                        metadata_path = os.path.join(prompt_dir, "metadata.jsonl")
                        with open(metadata_path, "w", encoding="utf-8") as f:
                            json.dump(geneval_metadata[global_prompt_idx], f, ensure_ascii=False)

                    image_filename = f"{sample_idx:04d}.png"
                    image_path = os.path.join(samples_dir, image_filename)
                    pil_image.save(image_path)
                else:
                    safe_prompt = "".join(
                        c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')
                    ).strip().replace(' ', '_')
                    if not safe_prompt:
                        safe_prompt = "prompt"

                    if samples_per_prompt > 1:
                        image_filename = f"image_{global_prompt_idx * samples_per_prompt + sample_idx:06d}_{safe_prompt}_sample{sample_idx:02d}.png"
                    else:
                        image_filename = f"image_{global_prompt_idx * samples_per_prompt + sample_idx:06d}_{safe_prompt}.png"

                    image_path = os.path.join(output_dir, image_filename)
                    pil_image.save(image_path)

                    txt_filename = image_filename.replace('.png', '.txt')
                    txt_path = os.path.join(output_dir, txt_filename)
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(prompt)

                image_count += 1
                # 更新进度条
                pbar.update(1)
                # 更新进度条详细信息
                pbar.set_postfix({
                    '批次': f'{batch_idx+1}/{num_batches}',
                    '提示': f'{global_prompt_idx+1}/{len(prompts_subset)}',
                    '样本': f'{sample_idx+1}/{samples_per_prompt}'
                })

    pbar.close()
    print(f"[GPU {gpu_id}] 完成！共生成 {image_count} 张图像")
    return image_count


# ============================================================
# 主函数
# ============================================================

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    config = get_config()

    # ---------- 读取参数 ----------
    prompts_file = config.get("validation_prompts_file", None)
    single_prompt = config.get("prompt", None)
    batch_size = config.get("batch_size", 1)

    text_to_image_config = config.get("text_to_image", {})
    # 优先从顶层获取，然后从 text_to_image 子配置获取
    samples_per_prompt = config.get("samples_per_prompt",
                                     text_to_image_config.get("samples_per_prompt", 1))
    output_dir = config.get("output_dir", "output_images")
    # guidance_scale: 顶层 > text_to_image > training
    guidance_scale = config.get("guidance_scale",
                                 text_to_image_config.get("guidance_scale",
                                                          config.training.get("guidance_scale", 3.5)))
    # generation_timesteps: 顶层 > text_to_image > training (默认15)
    generation_timesteps = config.get("generation_timesteps",
                                       text_to_image_config.get("generation_timesteps",
                                                                config.training.get("generation_timesteps", 15)))
    generation_temperature = config.get(
        "generation_temperature",
        text_to_image_config.get("generation_temperature",
                                  config.training.get("generation_temperature", 1.0)))

    # Info-Gain 配置
    candidate_number = config.get("candidate_number", 4)

    # 多 GPU
    num_gpus = config.get("num_gpus", None)
    use_multi_gpu = config.get("use_multi_gpu", True)
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        if num_gpus is None:
            num_gpus = available_gpus if use_multi_gpu else 1
        else:
            num_gpus = min(num_gpus, available_gpus)
        print(f"检测到 {available_gpus} 个 GPU，将使用 {num_gpus} 个")
    else:
        num_gpus = 1
        use_multi_gpu = False

    # GenEval 格式
    geneval_metadata_file = config.get("geneval_metadata_file", None)
    use_geneval_format = config.get("use_geneval_format", False)

    # 模型路径
    mmada_model_path_override = config.get("mmada_model_path", None)
    vq_model_path_override = config.get("vq_model_path", None)

    # 读取 GenEval metadata
    geneval_metadata = None
    if use_geneval_format or geneval_metadata_file:
        if geneval_metadata_file and os.path.exists(geneval_metadata_file):
            with open(geneval_metadata_file, "r", encoding="utf-8") as f:
                geneval_metadata = [json.loads(line.strip()) for line in f if line.strip()]
            use_geneval_format = True
            print(f"加载 GenEval metadata: {len(geneval_metadata)} 条")

    os.makedirs(output_dir, exist_ok=True)

    # 读取提示
    if prompts_file and os.path.exists(prompts_file):
        with open(prompts_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
        print(f"从文件读取 {len(prompts)} 个提示: {prompts_file}")
        if use_geneval_format and geneval_metadata:
            min_len = min(len(prompts), len(geneval_metadata))
            prompts = prompts[:min_len]
            geneval_metadata = geneval_metadata[:min_len]
    elif single_prompt:
        prompts = [single_prompt]
        use_geneval_format = False
    else:
        prompts = ["a beautiful landscape with mountains and lakes"]
        use_geneval_format = False

    # 模型路径 - 优先从model目录加载
    if mmada_model_path_override:
        mmada_model_path = mmada_model_path_override
    else:
        # 检查顺序：1. model/mmada 2. PROJECT_ROOT/mmada-mix 3. config中的路径
        model_dir_mmada = os.path.join(PROJECT_ROOT, "model", "mmada")
        local_mmada_path = os.path.join(PROJECT_ROOT, "mmada-mix")
        if os.path.exists(model_dir_mmada):
            mmada_model_path = model_dir_mmada
        elif os.path.exists(local_mmada_path):
            mmada_model_path = local_mmada_path
        else:
            mmada_model_path = config.model.mmada.pretrained_model_path

    if vq_model_path_override:
        vq_model_path = vq_model_path_override
    else:
        # 检查顺序：1. model/magvitv2 2. PROJECT_ROOT/magvitv2 3. config中的路径
        model_dir_vq = os.path.join(PROJECT_ROOT, "model", "magvitv2")
        local_vq_path = os.path.join(PROJECT_ROOT, "magvitv2")
        if os.path.exists(model_dir_vq):
            vq_model_path = model_dir_vq
        elif os.path.exists(local_vq_path):
            vq_model_path = local_vq_path
        else:
            vq_model_path = config.model.vq_model.vq_model_name

    total_images = len(prompts) * samples_per_prompt
    print(f"\n=== 生成配置 ===")
    print(f"提示数: {len(prompts)}, 每提示: {samples_per_prompt} 张, 总计: {total_images} 张")
    print(f"引导尺度: {guidance_scale}, 步数: {generation_timesteps}, 温度: {generation_temperature}")
    print(f"Info-Gain 候选数: {candidate_number}")
    print(f"输出目录: {output_dir}, GenEval 格式: {use_geneval_format}")
    print(f"MMaDA: {mmada_model_path}")
    print(f"VQ: {vq_model_path}")
    print()

    # ---------- 生成 ----------
    if num_gpus > 1 and use_multi_gpu and len(prompts) > 1:
        prompts_per_gpu = len(prompts) // num_gpus
        remainder = len(prompts) % num_gpus
        processes = []
        for gpu_id in range(num_gpus):
            start_idx = gpu_id * prompts_per_gpu + min(gpu_id, remainder)
            end_idx = start_idx + prompts_per_gpu + (1 if gpu_id < remainder else 0)
            prompts_subset = prompts[start_idx:end_idx]
            if not prompts_subset:
                continue
            p = multiprocessing.Process(
                target=generate_on_gpu,
                args=(gpu_id, prompts_subset, start_idx, config,
                      mmada_model_path, vq_model_path, output_dir,
                      batch_size, samples_per_prompt, guidance_scale,
                      generation_timesteps, generation_temperature,
                      use_geneval_format, geneval_metadata,
                      len(prompts), candidate_number))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        generate_on_gpu(
            0, prompts, 0, config,
            mmada_model_path, vq_model_path, output_dir,
            batch_size, samples_per_prompt, guidance_scale,
            generation_timesteps, generation_temperature,
            use_geneval_format, geneval_metadata,
            len(prompts), candidate_number)

    print(f"\n✅ 生成完成！共 {total_images} 张图像，保存于 {output_dir}")
