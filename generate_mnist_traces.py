import csv
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from train_mnist import SimpleCNN

import matplotlib.pyplot as plt

def log_base(x: float, base: float) -> float:
    if x <= 0:
        raise ValueError(f"log_base: x must be > 0, got {x}")
    if base <= 0 or base == 1.0:
        raise ValueError(f"log_base: base must be > 0 and != 1, got {base}")
    return math.log(x) / math.log(base)


def generate_drift_trace(
    model_path="experimental-results/MLmodels/mnist_cnn.pt",
    out_csv="mnist_drift_trace.csv",
    T=10_000,
    batch_size=32,
    max_noise_std=2,     # controls distribution shift strength
    drift_type="sigmoid"   # "linear" or "sigmoid"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=base_transform)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    def noise_schedule(t):
        x = t / T
        if drift_type == "linear":
            return max_noise_std * x
        elif drift_type == "sigmoid":
            return max_noise_std / (1 + np.exp(-10*(x-0.5)))
        else:
            raise ValueError("Unknown drift_type")

    trace = []

    t = 0
    with torch.no_grad():
        while t < T:
            for x, y in loader:
                if t >= T:
                    break

                # distribution drift
                sigma = noise_schedule(t)
                noise = torch.randn_like(x) * sigma
                x_noisy = x + noise
                x_noisy = torch.clamp(x_noisy, -2.0, 2.0)

                x_noisy = x_noisy.to(device)
                y = y.to(device)

                logits = model(x_noisy)
                probs = F.softmax(logits, dim=1)

                max_softmax, preds = probs.max(dim=1)
                acc = (preds == y).float()

                for i in range(x.size(0)):
                    trace.append([
                        t,
                        int(y[i].cpu().item()),              # input label id
                        float(max_softmax[i].cpu().item()),  # confidence
                        float(acc[i].cpu().item())           # 0 or 1
                    ])
                    t += 1
                    if t >= T:
                        break



    # write csv
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "input_label", "softmax_value", "prediction_accuracy"])
        for row in trace:
            writer.writerow(row)

    print(f"Trace saved to {out_csv}")


def main():
    max_noise_std_vals = np.linspace(0.1, 2, 8)

    outcsv_base = "mnist-traces/mnist_drift_trace"

    params = {}
    params["infR"] = 0
    params["supR"] = 1
    params["r"] = 0.95
    params["s"] = 0.95
    params["eps"] = 0.001
    use_averages = True
    params["use_averages"] = use_averages
    avg_normalization_value = 1+(params["r"]/(1-params["r"]))+(params["s"]/(1-params["s"])) if use_averages else 1
    params["T"] = np.ceil(log_base( (1-params["r"])*params["eps"]/(2*avg_normalization_value) ,params["r"]))
    params["L"] = 0.96
    params["U"] = 1
    params["save_csv"] = False    


    dicts_vec = []



    for max_noise_std in max_noise_std_vals:
        out_csv = f"{outcsv_base}_sigmoid_noise_{max_noise_std:.2f}.csv"
        generate_drift_trace(max_noise_std=max_noise_std, out_csv = out_csv)
        params["data_source"] = out_csv
        params["data_column_name"] = "prediction_accuracy"
        params["data_results"] = f"experimental-results/mnist_noise_{max_noise_std:.2f}.csv"
        dicts_vec.append(params.copy())
    

    with open("experimental-setups/RQ1-mnist-traces-increase-noise.json", "w") as fp:
        json.dump(dicts_vec, fp, indent=2)


if __name__ =="__main__":
    main()