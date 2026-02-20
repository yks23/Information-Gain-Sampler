#!/usr/bin/env python
"""
CLIP Score 评测脚本

使用 open_clip 计算图像与文本提示之间的语义相似度。
支持 GenEval 目录格式（从 metadata.jsonl 读取 prompt）和普通目录格式。

用法：
  # GenEval 格式目录
  python eval_clip_score.py --image-dir ./output_geneval --output results/clip_scores.json

  # 普通目录 + 自定义提示文件
  python eval_clip_score.py --image-dir ./output_images --prompts-file prompts.txt --output results/clip_scores.json

  # 指定 CLIP 模型
  python eval_clip_score.py --image-dir ./output_geneval --clip-arch ViT-L-14 --clip-pretrained openai

依赖：
  pip install open_clip_torch torch Pillow numpy tqdm
"""

import argparse
import os
import json
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    import open_clip
    HAS_OPEN_CLIP = True
except ImportError:
    HAS_OPEN_CLIP = False


def load_geneval_data(directory):
    """从 GenEval 格式目录加载图像路径和文本"""
    texts = []
    image_paths = []
    for subdir in sorted(os.listdir(directory)):
        subdir_path = os.path.join(directory, subdir)
        if not os.path.isdir(subdir_path):
            continue
        metadata_path = os.path.join(subdir_path, "metadata.jsonl")
        samples_dir = os.path.join(subdir_path, "samples")
        if os.path.exists(metadata_path) and os.path.exists(samples_dir):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                prompt = metadata.get('prompt', '')
            for img_file in sorted(os.listdir(samples_dir)):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(samples_dir, img_file))
                    texts.append(prompt)
    return image_paths, texts


def load_plain_images(directory):
    """从普通目录加载图像路径"""
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in sorted(files):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))
    return sorted(image_paths)


def calculate_clip_score(image_paths, texts, batch_size=32, device='cuda',
                         clip_arch="ViT-L-14", pretrained="openai"):
    """
    计算 CLIP Score（使用 open_clip）

    Returns:
        mean_score: 平均 CLIP score
        scores: 每个图像-文本对的分数列表
    """
    if not HAS_OPEN_CLIP:
        raise ImportError("需要安装 open_clip_torch: pip install open_clip_torch")

    if len(image_paths) != len(texts):
        raise ValueError(f"图像数量 ({len(image_paths)}) 与文本数量 ({len(texts)}) 不匹配")

    print(f"加载 CLIP 模型: {clip_arch} (pretrained={pretrained})")
    model, _, transform = open_clip.create_model_and_transforms(
        clip_arch, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(clip_arch)
    model.eval()

    # 预处理
    image_tensors = []
    text_tokens = []
    valid_indices = []

    for idx, (img_path, text) in enumerate(tqdm(zip(image_paths, texts), desc="加载数据", total=len(image_paths))):
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            image_tensors.append(img_tensor)
            text_token = tokenizer([text]).to(device)
            text_tokens.append(text_token)
            valid_indices.append(idx)
        except Exception as e:
            print(f"警告: 处理 {img_path} 失败: {e}")

    # 批量计算
    scores = []
    with torch.no_grad():
        for i in tqdm(range(0, len(image_tensors), batch_size), desc="计算 CLIP score"):
            batch_images = torch.cat(image_tensors[i:i + batch_size], dim=0)
            batch_texts = torch.cat(text_tokens[i:i + batch_size], dim=0)

            image_features = model.encode_image(batch_images)
            text_features = model.encode_text(batch_texts)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T).diag()
            scores.extend(similarity.cpu().tolist())

    scores = np.array(scores)
    return scores.mean(), scores.tolist()


def main():
    parser = argparse.ArgumentParser(description="计算 CLIP Score")
    parser.add_argument("--image-dir", type=str, required=True, help="图像目录路径")
    parser.add_argument("--prompts-file", type=str, default=None,
                        help="提示文件（每行一个），仅在普通目录格式时需要")
    parser.add_argument("--output", "-o", type=str, default="results/clip_scores.json",
                        help="输出 JSON 文件路径")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--clip-arch", type=str, default="ViT-L-14",
                        help="CLIP 模型架构 (默认 ViT-L-14)")
    parser.add_argument("--clip-pretrained", type=str, default="openai")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 检测格式
    first_subdir = None
    for name in sorted(os.listdir(args.image_dir)):
        if os.path.isdir(os.path.join(args.image_dir, name)):
            first_subdir = name
            break

    is_geneval = (first_subdir and first_subdir.isdigit() and
                  os.path.exists(os.path.join(args.image_dir, first_subdir, "metadata.jsonl")))

    if is_geneval:
        print("检测到 GenEval 格式目录")
        image_paths, texts = load_geneval_data(args.image_dir)
    else:
        print("检测到普通目录格式")
        image_paths = load_plain_images(args.image_dir)
        if args.prompts_file and os.path.exists(args.prompts_file):
            with open(args.prompts_file, 'r') as f:
                texts = [line.strip() for line in f if line.strip()]
            # 扩展 texts 匹配图像数量
            if len(texts) < len(image_paths):
                repeat = len(image_paths) // len(texts)
                texts = texts * repeat + texts[:len(image_paths) - len(texts) * repeat]
        else:
            print("警告: 未提供 prompts-file，使用空文本")
            texts = [""] * len(image_paths)

    print(f"图像数量: {len(image_paths)}, 文本数量: {len(texts)}")
    if len(image_paths) == 0:
        print("错误: 未找到图像")
        return

    mean_score, all_scores = calculate_clip_score(
        image_paths, texts, batch_size=args.batch_size, device=device,
        clip_arch=args.clip_arch, pretrained=args.clip_pretrained
    )

    results = {
        "clip_score_mean": float(mean_score),
        "clip_arch": args.clip_arch,
        "clip_pretrained": args.clip_pretrained,
        "num_images": len(image_paths),
        "scores": [float(s) for s in all_scores],
    }

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"  CLIP Score (mean): {mean_score:.4f}")
    print(f"  模型: {args.clip_arch} ({args.clip_pretrained})")
    print(f"  图像数: {len(image_paths)}")
    print(f"{'='*50}")
    print(f"结果已保存到: {args.output}")


if __name__ == "__main__":
    main()

