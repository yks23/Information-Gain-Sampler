"""
Evaluation script for multimodal tasks (text-to-image generation).

This script provides a unified interface to multimodal evaluation tasks.
For detailed usage, see src/benchmarks/multimodal_tasks/multimodal_eval/README.md

Usage:
    # Run full evaluation pipeline
    python eval_multimodal.py --pipeline all

    # Run only generation
    python eval_multimodal.py --pipeline generate

    # Run only GenEval evaluation
    python eval_multimodal.py --pipeline geneval --image_dir ./output_geneval
"""

import sys
import os
import subprocess

# Add project root to path
current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(scripts_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse


def main_multimodal():
    parser = argparse.ArgumentParser(
        description="Evaluation for multimodal tasks (text-to-image generation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: generation + all evaluations
  python eval_multimodal.py --pipeline all

  # Only generate images
  python eval_multimodal.py --pipeline generate

  # Only evaluate existing images
  python eval_multimodal.py --pipeline geneval --image_dir ./output_geneval
  python eval_multimodal.py --pipeline fid --image_dir ./output_geneval
  python eval_multimodal.py --pipeline clip --image_dir ./output_geneval
        """
    )
    
    parser.add_argument('--pipeline', type=str, required=True,
                        choices=['all', 'generate', 'geneval', 'fid', 'clip', 'imagenet_fid'],
                        help='Pipeline to run: all|generate|geneval|fid|clip|imagenet_fid')
    
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Image directory (for eval-only pipelines)')
    
    parser.add_argument('--mmada_model_path', type=str, default=None,
                        help='MMaDA model path (default: auto-detect from model/ directory)')
    
    parser.add_argument('--vq_model_path', type=str, default=None,
                        help='VQ model path (default: auto-detect from model/ directory)')
    
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode (smaller scale)')
    
    args = parser.parse_args()
    
    # Auto-detect model paths from model/ directory if not provided
    if args.mmada_model_path is None:
        model_dir_mmada = os.path.join(project_root, 'model', 'mmada')
        if os.path.exists(model_dir_mmada):
            args.mmada_model_path = model_dir_mmada
            print(f"Auto-detected MMaDA model: {args.mmada_model_path}")
        else:
            args.mmada_model_path = os.path.join(project_root, 'mmada-mix')
            if not os.path.exists(args.mmada_model_path):
                print(f"Warning: MMaDA model not found at {args.mmada_model_path}")
    
    if args.vq_model_path is None:
        model_dir_vq = os.path.join(project_root, 'model', 'magvitv2')
        if os.path.exists(model_dir_vq):
            args.vq_model_path = model_dir_vq
            print(f"Auto-detected VQ model: {args.vq_model_path}")
        else:
            args.vq_model_path = os.path.join(project_root, 'magvitv2')
            if not os.path.exists(args.vq_model_path):
                print(f"Warning: VQ model not found at {args.vq_model_path}")
    
    # Get multimodal_eval directory
    multimodal_dir = os.path.join(project_root, 'src', 'benchmarks', 'multimodal_tasks', 'multimodal_eval')
    
    if not os.path.exists(multimodal_dir):
        print(f"Error: Multimodal evaluation directory not found: {multimodal_dir}")
        sys.exit(1)
    
    # Change to multimodal directory
    os.chdir(multimodal_dir)
    
    # Run appropriate script
    if args.pipeline == 'all':
        script = 'scripts/run_all.sh'
        cmd = ['bash', script]
    elif args.pipeline == 'generate':
        script = 'scripts/run_generate.sh'
        cmd = ['bash', script]
    elif args.pipeline == 'geneval':
        if not args.image_dir:
            parser.error("--image_dir is required for geneval pipeline")
        script = 'scripts/run_eval_geneval.sh'
        cmd = ['bash', script, args.image_dir]
    elif args.pipeline == 'fid':
        if not args.image_dir:
            parser.error("--image_dir is required for fid pipeline")
        script = 'scripts/run_eval_fid.sh'
        cmd = ['bash', script, './VIRTUAL_imagenet512.npz', args.image_dir]
    elif args.pipeline == 'clip':
        if not args.image_dir:
            parser.error("--image_dir is required for clip pipeline")
        script = 'scripts/run_eval_clip.sh'
        cmd = ['bash', script, args.image_dir]
    elif args.pipeline == 'imagenet_fid':
        script = 'scripts/run_imagenet_fid.sh'
        cmd = ['bash', script]
        if args.test:
            cmd.append('--test')
    
    # Execute command
    print(f"Running: {' '.join(cmd)}")
    print(f"Working directory: {multimodal_dir}")
    result = subprocess.run(cmd, cwd=multimodal_dir)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main_multimodal()
