"""
Benchmark: Inductor (torch.compile) vs eager CPU training.

Usage (single run):
    python benchmarks/inductor_vs_cpu_train.py
    python benchmarks/inductor_vs_cpu_train.py --epochs 5 --warmup 2 --batch-size 64 --mode eager
    python benchmarks/inductor_vs_cpu_train.py --epochs 5 --warmup 2 --batch-size 64 --mode inductor
"""

import argparse
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_block(64, 64, 2)
        self.layer2 = self._make_block(64, 128, 2, stride=2)
        self.layer3 = self._make_block(128, 256, 2, stride=2)
        self.layer4 = self._make_block(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_block(self, in_c, out_c, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_c, out_c, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class TransformerModel(nn.Module):
    def __init__(self, vocab_size=10000, d_model=256, nhead=4, num_layers=4, num_classes=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 128, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, : x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    start = time.perf_counter()
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    elapsed = time.perf_counter() - start
    return elapsed, total_loss / len(loader)


def benchmark_model(model, loader, criterion, optimizer, device, epochs, warmup):
    for _ in range(warmup):
        train_epoch(model, loader, criterion, optimizer, device)

    times = []
    for _ in range(epochs):
        elapsed, avg_loss = train_epoch(model, loader, criterion, optimizer, device)
        times.append(elapsed)

    return sum(times) / len(times)


def main():
    parser = argparse.ArgumentParser(description="Inductor vs CPU Training Benchmark")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mode", choices=["eager", "inductor"], default="inductor",
                        help="Training mode: eager or inductor (torch.compile)")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = "cpu"
    n_samples = 1024

    conv_dataset = TensorDataset(
        torch.randn(n_samples, 3, 32, 32),
        torch.randint(0, 10, (n_samples,)),
    )
    conv_loader = DataLoader(conv_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    seq_dataset = TensorDataset(
        torch.randint(0, 10000, (n_samples, 64)),
        torch.randint(0, 10, (n_samples,)),
    )
    seq_loader = DataLoader(seq_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    results = {}
    models = [
        ("ResNet18", lambda: ResNet18(), conv_loader),
        ("Transformer", lambda: TransformerModel(), seq_loader),
    ]

    for name, model_fn, loader in models:
        torch.manual_seed(42)
        model = model_fn().to(device)
        if args.mode == "inductor":
            model = torch.compile(model)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        avg_time = benchmark_model(model, loader, criterion, optimizer, device, args.epochs, args.warmup)
        results[name] = avg_time
        print(f"{name}: {avg_time:.3f}s avg epoch ({args.mode})")

    if args.json:
        print(json.dumps({
            "torch_version": torch.__version__,
            "mode": args.mode,
            "device": device,
            "epochs": args.epochs,
            "warmup": args.warmup,
            "batch_size": args.batch_size,
            "results": results,
        }))


if __name__ == "__main__":
    main()
