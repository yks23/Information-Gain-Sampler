#!/usr/bin/env python3
"""
图像格式转换工具

将 MMaDA 生成的普通格式图像转换为 GenEval 评估所需的目录结构。

GenEval 格式：
    <IMAGE_FOLDER>/
        00000/
            metadata.jsonl   (JSON，包含 prompt / tag / include / exclude 等)
            samples/
                0000.png
                0001.png
        00001/
            metadata.jsonl
            samples/
                0000.png

用法：
  python convert_format.py \
      --input-dir ./output_images \
      --metadata-file ../geneval/prompts/evaluation_metadata.jsonl \
      --output-dir ./output_geneval \
      --images-per-prompt 4
"""

import os
import json
import shutil
import argparse
from pathlib import Path


def convert_to_geneval_format(input_dir, metadata_file, output_dir, images_per_prompt=1):
    """
    将普通格式图像转换为 GenEval 格式

    Args:
        input_dir: 输入图像目录（包含 *.png 文件）
        metadata_file: GenEval metadata 文件 (evaluation_metadata.jsonl)
        output_dir: 输出目录（GenEval 格式）
        images_per_prompt: 每个提示对应的图像数量
    """
    print(f"读取 metadata: {metadata_file}")
    with open(metadata_file, 'r', encoding='utf-8') as f:
        prompts_metadata = [json.loads(line.strip()) for line in f if line.strip()]
    print(f"共 {len(prompts_metadata)} 个提示")

    # 收集输入图像
    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.endswith('_grid.png')
    ])
    print(f"找到 {len(image_files)} 张图像")

    expected = len(prompts_metadata) * images_per_prompt
    if len(image_files) < expected:
        print(f"警告: 图像数量 ({len(image_files)}) < 预期 ({expected})")
        print(f"将只处理前 {len(image_files) // images_per_prompt} 个提示")

    os.makedirs(output_dir, exist_ok=True)

    image_idx = 0
    for prompt_idx, metadata in enumerate(prompts_metadata):
        if image_idx >= len(image_files):
            break

        prompt_dir = os.path.join(output_dir, f"{prompt_idx:05d}")
        samples_dir = os.path.join(prompt_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)

        # 保存 metadata
        with open(os.path.join(prompt_dir, "metadata.jsonl"), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False)

        # 复制图像
        for sample_idx in range(images_per_prompt):
            if image_idx >= len(image_files):
                break
            src = os.path.join(input_dir, image_files[image_idx])
            dst = os.path.join(samples_dir, f"{sample_idx:04d}.png")
            shutil.copy2(src, dst)
            image_idx += 1

        if (prompt_idx + 1) % 100 == 0:
            print(f"  已处理 {prompt_idx + 1}/{len(prompts_metadata)} 个提示")

    print(f"\n✅ 转换完成！共处理 {prompt_idx + 1} 个提示，{image_idx} 张图像")
    print(f"输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="将图像转换为 GenEval 格式")
    parser.add_argument("--input-dir", required=True, help="输入图像目录")
    parser.add_argument("--metadata-file", required=True,
                        help="GenEval metadata 文件 (evaluation_metadata.jsonl)")
    parser.add_argument("--output-dir", required=True, help="输出目录 (GenEval 格式)")
    parser.add_argument("--images-per-prompt", type=int, default=1,
                        help="每个提示对应的图像数量 (默认 1)")
    args = parser.parse_args()

    convert_to_geneval_format(
        args.input_dir,
        args.metadata_file,
        args.output_dir,
        args.images_per_prompt,
    )


if __name__ == '__main__':
    main()

