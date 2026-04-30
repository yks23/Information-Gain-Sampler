# Models

## Supported Models

| Model | HuggingFace Path | Local alias |
|-------|-----------------|-------------|
| **LLaDA** | `GSAI-ML/LLaDA-8B-Instruct` | `llada` |
| **Dream** | `Dream-org/Dream-v0-Instruct-7B` | `dream` |
| **SDAR** | `JetLM/SDAR-8B-Chat` | `sdar` |
| **TraDo** | `Gen-Verse/TraDo-8B-Instruct` | `trado` |
| **MMaDA** | `Gen-Verse/MMaDA-8B-MixCoT` | `mmada` |

## Download Commands

```bash
huggingface-cli download GSAI-ML/LLaDA-8B-Instruct --local-dir ./model/llada
huggingface-cli download Dream-org/Dream-v0-Instruct-7B --local-dir ./model/dream
huggingface-cli download JetLM/SDAR-8B-Chat --local-dir ./model/sdar
huggingface-cli download Gen-Verse/TraDo-8B-Instruct --local-dir ./model/trado
huggingface-cli download Gen-Verse/MMaDA-8B-MixCoT --local-dir ./model/mmada
```

Models should be placed under `model/` in the repository root:

```
model/
├── llada/       # LLaDA-8B-Instruct
├── dream/       # Dream-v0-Instruct-7B
├── sdar/        # SDAR-8B-Chat
├── trado/       # TraDo-8B-Instruct
├── mmada/       # MMaDA-8B-MixCoT  (multimodal)
└── magvitv2/    # MAGVITv2 VQ model (multimodal)
```

## Notes

- SDAR and TraDo share the same adapter (`SDARAdapter`) internally.
- MMaDA requires a separate conda environment (`mmada`, Python 3.11) — see [installation.md](installation.md).
- You can also pass a HuggingFace path directly instead of the local alias: `--model GSAI-ML/LLaDA-8B-Instruct`.
