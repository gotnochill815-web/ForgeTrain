 # ForgeTrain

ForgeTrain is a lightweight PyTorch experimentation framework built to explore real-world deep learning engineering workflows such as benchmarking, profiling, distributed training, checkpointing, and reproducible experiments.

Rather than being just another training script, ForgeTrain focuses on how different neural network architectures behave under system constraints such as CPU-only environments, throughput limits, and training-time tradeoffs.

---

## Why ForgeTrain?

Most beginner ML projects stop at:

- loading a dataset
- training one model
- printing loss
- ending the script

ForgeTrain goes further by studying **systems behavior**:

- Which model is fastest?
- Which model is most accurate?
- How expensive is deeper architecture on CPU?
- Where are runtime bottlenecks?
- How do we track experiments cleanly?

---

## Core Features

- Multi-model training support:
  - MLP
  - SimpleCNN
  - ResNet18
- YAML config-driven experiments
- Automatic checkpoint saving
- PyTorch Profiler integration
- CSV metrics logging
- Auto-generated benchmark graphs
- Multi-model comparison charts
- Clean modular project structure
- Ready for future GPU / DDP scaling

---

## Project Structure

```bash
ForgeTrain/
│── configs/
│   └── base.yaml
│
│── logs/
│   ├── train_metrics.csv
│   ├── accuracy.png
│   ├── loss.png
│   ├── throughput.png
│   ├── compare_accuracy.png
│   ├── compare_speed.png
│   └── compare_time.png
│
│── src/
│   ├── train.py
│   ├── ddp_train.py
│   ├── profile_train.py
│   ├── plot_metrics.py
│   ├── models.py
│   ├── data.py
│   ├── engine.py
│   ├── metrics.py
│   └── checkpoint.py
│
│── requirements.txt
│── README.md