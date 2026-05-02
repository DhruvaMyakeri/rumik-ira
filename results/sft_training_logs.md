# SFT Training Logs - Ira 12B

## Run Details

- Model: unsloth/gemma-3-12b-it
- Hardware: NVIDIA A100-SXM4-40GB (39.494 GB)
- Framework: Unsloth 2025.11.1, Transformers 4.57.2, Torch 2.10.0+cu128
- Date: April 30, 2026

## Config

- Train examples: 615
- Val examples: 62
- Epochs: 2
- Total steps: 78
- Batch size per device: 1
- Gradient accumulation: 16
- Effective batch size: 16
- Trainable parameters: 68,456,448 / 12,255,781,488 (0.56%)
- Runtime: 1664.4 seconds (~27 minutes)

## Loss Progression

| Step | Epoch | Loss   | Grad Norm | LR      |
| ---- | ----- | ------ | --------- | ------- |
| ~10  | 0.26  | 0.182  | 2.817     | 1.97e-4 |
| ~20  | 0.52  | 0.0505 | 0.488     | 1.78e-4 |
| ~30  | 0.78  | 0.0412 | 0.475     | 1.46e-4 |
| ~40  | 1.03  | 0.0422 | 0.328     | 1.06e-4 |
| ~50  | 1.29  | 0.0320 | 0.355     | 6.51e-5 |
| ~60  | 1.55  | 0.0321 | 0.321     | 3.00e-5 |
| ~70  | 1.81  | 0.0308 | 0.334     | 7.02e-6 |
| 78   | 2.00  | -      | -         | -       |

## Final Metrics

- Train loss: 0.0563
- Train samples/sec: 0.739
- Train steps/sec: 0.047
- Total runtime: 27m 44s
