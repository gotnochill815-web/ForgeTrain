# ForgeTrain 

ForgeTrain is a lightweight PyTorch training framework built to explore deep learning engineering workflows such as distributed training, benchmarking, profiling, checkpointing, and reproducible experimentation.

This project was built to understand how training pipelines scale from a single process to multi-worker distributed setups.

---

## Features

- PyTorch CNN training on CIFAR-10
- Config-driven experiments using YAML
- Model checkpoint saving
- Distributed Data Parallel (DDP) training
- Performance benchmarking (time, throughput, workers)
- PyTorch Profiler integration
- Clean modular project structure

---

## Project Structure

```bash
ForgeTrain/
├── configs/
│   └── base.yaml
├── src/
│   ├── train.py
│   ├── ddp_train.py
│   ├── profile_train.py
│   ├── models.py
│   ├── data.py
│   ├── metrics.py
│   ├── engine.py
│   └── checkpoint.py
├── requirements.txt
└── README.md