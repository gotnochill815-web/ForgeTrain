import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("logs/train_metrics.csv")

# -------- Accuracy Comparison --------
plt.figure(figsize=(8,5))
for model in df["model"].unique():
    sub = df[df["model"] == model]
    plt.plot(sub["epoch"], sub["accuracy"], marker="o", label=model)

plt.title("Model Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)
plt.savefig("logs/compare_accuracy.png")
plt.close()

# -------- Throughput Comparison --------
plt.figure(figsize=(8,5))
models = df["model"]
speed = df["samples_per_sec"]

plt.bar(models, speed)
plt.title("Samples/sec Comparison")
plt.xlabel("Model")
plt.ylabel("Samples/sec")
plt.grid(axis="y")
plt.savefig("logs/compare_speed.png")
plt.close()

# -------- Train Time Comparison --------
plt.figure(figsize=(8,5))
times = df["train_time"]

plt.bar(models, times)
plt.title("Train Time Comparison")
plt.xlabel("Model")
plt.ylabel("Seconds")
plt.grid(axis="y")
plt.savefig("logs/compare_time.png")
plt.close()

print("Saved comparison charts in logs/")