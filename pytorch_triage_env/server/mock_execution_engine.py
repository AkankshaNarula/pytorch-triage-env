"""
MockExecutionEngine: zero-dependency simulation of PyTorch training runs.

Critical design constraint: DO NOT import torch in this file.
The Dockerfile does not install PyTorch. The engine parses file strings and
returns pre-written, hyper-realistic PyTorch error traces.

Each task has:
  - buggy_trace:      the authentic output when bugs are NOT fixed
  - success_trace:    output when ALL fixes are applied
  - diagnostic_trace: output when run with diagnostic flags (e.g. TORCH_LOGS=dynamo)
  - fix_signatures:   (filename, required_string) pairs that define a correct fix
"""
from __future__ import annotations
from typing import Dict, List, Literal, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1: OOM Graph Leak
# Bug: `epoch_loss += loss` retains the computation graph
# Fix: `epoch_loss += loss.item()`
# ══════════════════════════════════════════════════════════════════════════════

_OOM_GRAPH_LEAK_FILES = {
    "train.py": '''\
import torch
import torch.nn as nn
import torch.optim as optim
from model import ResNetClassifier
from data_loader import get_train_loader
from config import LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss  # BUG: retains computation graph! Should be loss.item()
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}: loss={loss.item():.4f}")
    return epoch_loss / len(loader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetClassifier(num_classes=1000).to(device)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    loader = get_train_loader(batch_size=BATCH_SIZE)
    for epoch in range(NUM_EPOCHS):
        avg_loss = train_one_epoch(model, loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

if __name__ == "__main__":
    main()
''',
    "model.py": '''\
import torch
import torch.nn as nn

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = x.flatten(1)
        return self.classifier(x)
''',
    "config.py": '''\
LEARNING_RATE = 0.01
NUM_EPOCHS = 10
BATCH_SIZE = 64
DATA_DIR = "./data/imagenet"
CHECKPOINT_DIR = "./checkpoints"
''',
    "data_loader.py": '''\
from torch.utils.data import DataLoader, TensorDataset
import torch

def get_train_loader(batch_size=64):
    # Synthetic data for benchmarking
    X = torch.randn(10000, 3, 224, 224)
    y = torch.randint(0, 1000, (10000,))
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True, num_workers=4)
''',
}

_OOM_GRAPH_LEAK_BUGGY_TRACE = """\
Epoch 1 starting...
  Batch 0: loss=6.9078
  Batch 100: loss=6.8821
  Batch 200: loss=6.8543
  ...
  Batch 900: loss=6.6821
Epoch 1: avg_loss=6.8201
[Memory] Allocated: 2.1GB | Cached: 2.3GB

Epoch 2 starting...
  Batch 0: loss=6.6234
  Batch 100: loss=6.5891
  ...
  Batch 900: loss=6.4201
Epoch 2: avg_loss=6.5812
[Memory] Allocated: 5.8GB | Cached: 6.1GB  <-- GROWING

Epoch 3 starting...
  Batch 0: loss=6.4012
  Batch 100: loss=6.3788
  ...
Traceback (most recent call last):
  File "train.py", line 32, in main
    avg_loss = train_one_epoch(model, loader, optimizer, criterion, device)
  File "train.py", line 13, in train_one_epoch
    loss.backward()
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.00 GiB
  (GPU 0; 23.69 GiB total capacity; 20.12 GiB already allocated;
   2.73 GiB free; 21.84 GiB reserved in total by PyTorch)
If reserved memory is >> allocated memory try setting max_split_size_mb to avoid
fragmentation. See documentation for Memory Management and PYTORCH_CUDA_MALLOC_CONF.

NOTE: If your program is crashing due to this error, an optimization to try would be
to use `model.half()` or switch from a 32-bit float to a 16-bit float. If you're
using autocast, make sure you're not keeping tensors across iterations that could
accumulate the computation graph.

Process finished with exit code 1.

[DIAGNOSIS] Memory profile by epoch:
  Epoch 1 end: 2.1GB allocated
  Epoch 2 end: 5.8GB allocated (delta: +3.7GB — growing linearly with epochs)
  Epoch 3 OOM  — graph nodes from ALL prior batches still alive
[HINT] Check for tensor accumulators in your training loop that retain grad_fn.
       Common culprit: epoch_loss += loss (keeps graph) vs epoch_loss += loss.item() (detaches)
"""

_OOM_GRAPH_LEAK_SUCCESS_TRACE = """\
Epoch 1 starting...
  Batch 0: loss=6.9078
  Batch 100: loss=6.8821
  ...
  Batch 900: loss=6.6821
Epoch 1: avg_loss=6.8201
[Memory] Allocated: 2.1GB | Cached: 2.3GB

Epoch 2 starting...
  Batch 0: loss=6.6234
  ...
Epoch 2: avg_loss=6.5812
[Memory] Allocated: 2.1GB | Cached: 2.3GB  ← stable

Epoch 3 starting...
  ...
Epoch 3: avg_loss=6.3201
[Memory] Allocated: 2.1GB | Cached: 2.3GB  ← stable

Training completed successfully. 10/10 epochs. Memory stable throughout.
Final model saved to ./checkpoints/model_final.pt
"""

_OOM_GRAPH_LEAK_FIX_SIGNATURES = [
    ("train.py", "epoch_loss += loss.item()"),  # precise accumulator fix
]


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2: FSDP Collective Deadlock
# Bug: dist.all_reduce inside if rank == 0 block
# Fix: move dist.all_reduce outside the if block
# ══════════════════════════════════════════════════════════════════════════════

_FSDP_DEADLOCK_FILES = {
    "train.py": '''\
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from model import TransformerLM
from config import LEARNING_RATE, NUM_STEPS

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    model = FSDP(TransformerLM().to(rank))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for step in range(NUM_STEPS):
        # Simulate batch
        inputs = torch.randint(0, 50257, (8, 512), device=rank)
        targets = torch.randint(0, 50257, (8, 512), device=rank)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits.view(-1, 50257), targets.view(-1))
        loss.backward()
        optimizer.step()

        # BUG: dist.all_reduce is a collective — ALL ranks must call it simultaneously
        # Putting it inside if rank == 0 means only rank 0 calls it.
        # Ranks 1, 2, 3 never call all_reduce → NCCL hangs waiting for all ranks.
        if rank == 0:
            loss_tensor = loss.detach().clone()
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)  # BUG: collective inside rank guard
            avg_loss = loss_tensor.item() / world_size
            if step % 10 == 0:
                print(f"Step {step}: avg_loss={avg_loss:.4f}")

    cleanup()

if __name__ == "__main__":
    world_size = 4
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)
''',
    "model.py": '''\
import torch
import torch.nn as nn

class TransformerLM(nn.Module):
    def __init__(self, vocab_size=50257, d_model=768, nhead=12, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=3072,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.lm_head(x)
''',
    "config.py": '''\
LEARNING_RATE = 1e-4
NUM_STEPS = 1000
WORLD_SIZE = 4
CHECKPOINT_DIR = "./checkpoints"
''',
    "data_loader.py": '''\
import torch
from torch.utils.data import DataLoader, TensorDataset

def get_distributed_loader(rank, world_size, batch_size=8, seq_len=512):
    # Synthetic distributed data loader
    from torch.utils.data.distributed import DistributedSampler
    dataset = TensorDataset(
        torch.randint(0, 50257, (1000, seq_len)),
        torch.randint(0, 50257, (1000, seq_len))
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
''',
}

_FSDP_DEADLOCK_BUGGY_TRACE = """\
torchrun --nproc_per_node=4 train.py

[2026-04-07 14:23:01] [rank0]: Initializing NCCL process group
[2026-04-07 14:23:01] [rank1]: Initializing NCCL process group
[2026-04-07 14:23:01] [rank2]: Initializing NCCL process group
[2026-04-07 14:23:01] [rank3]: Initializing NCCL process group
[2026-04-07 14:23:02] [rank0]: FSDP model initialized (TransformerLM, 355M params)
[2026-04-07 14:23:02] [rank1]: FSDP model initialized
[2026-04-07 14:23:02] [rank2]: FSDP model initialized
[2026-04-07 14:23:02] [rank3]: FSDP model initialized
[2026-04-07 14:23:03] [rank0]: Step 0 forward pass complete
[2026-04-07 14:23:03] [rank1]: Step 0 forward pass complete
[2026-04-07 14:23:03] [rank2]: Step 0 forward pass complete
[2026-04-07 14:23:03] [rank3]: Step 0 forward pass complete
[2026-04-07 14:23:03] [rank0]: Calling dist.all_reduce...
[2026-04-07 14:23:03] [rank1]: [HANGING — waiting for collective that rank0 is in but rank1 never called]
[2026-04-07 14:23:03] [rank2]: [HANGING — waiting for collective that rank0 is in but rank2 never called]
[2026-04-07 14:23:03] [rank3]: [HANGING — waiting for collective that rank0 is in but rank3 never called]
... [30 minutes of silence] ...
[2026-04-07 14:53:04] [rank0]: [E ProcessGroupNCCL.cpp:1290] NCCL watchdog timeout.
  Timeout value is: 1800 seconds.
  [Rank 0] NCCL operation: all_reduce
  [Rank 0] Process group: default

[2026-04-07 14:53:04] [rank1]: [E ProcessGroupNCCL.cpp:1290] NCCL watchdog timeout.
[2026-04-07 14:53:04] [rank2]: [E ProcessGroupNCCL.cpp:1290] NCCL watchdog timeout.
[2026-04-07 14:53:04] [rank3]: [E ProcessGroupNCCL.cpp:1290] NCCL watchdog timeout.

RuntimeError: NCCL error in: /opt/conda/conda-bld/pytorch.../ProcessGroupNCCL.cpp:1290
  NCCL error: unhandled system error, NCCL version 2.19.3
  ncclSystemError: System call (e.g. socket, malloc) or internal/external
  library call failed or device error.

  [Diagnosis] All ranks stalled at NCCL collective barrier.
  This is caused by a collective operation (all_reduce/all_gather/broadcast)
  being called by a subset of ranks.
  Check: is dist.all_reduce inside a conditional block that not all ranks enter?

Traceback (most recent call last):
  File "train.py", line 41, in train
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
  ... (NCCL watchdog killed all 4 workers)
"""

_FSDP_DEADLOCK_SUCCESS_TRACE = """\
torchrun --nproc_per_node=4 train.py

[rank0]: FSDP model initialized (TransformerLM, 355M params)
[rank1]: FSDP model initialized
[rank2]: FSDP model initialized
[rank3]: FSDP model initialized
Step 0: avg_loss=10.8821
Step 10: avg_loss=10.6234
Step 20: avg_loss=10.3891
Step 50: avg_loss=9.8821
Step 100: avg_loss=9.2341
Training progressing normally. All 4 ranks synchronized correctly.
No NCCL timeouts. Collective operations executing on all ranks.
"""

_FSDP_DEADLOCK_FIX_SIGNATURES = [
    # The all_reduce must be OUTSIDE the if rank == 0 block
    # We check that `all_reduce` is NOT immediately after `if rank == 0:`
    # Simplest check: the word `all_reduce` appears but NOT indented inside a rank guard
    ("train.py", "dist.all_reduce(loss_tensor"),   # still present
    ("train.py", "if rank == 0:\n            loss_tensor"),  # NOT inside the old guard
]


# ══════════════════════════════════════════════════════════════════════════════
# TASK 3: torch.compile Graph Break
# Bug: data-dependent control flow with loss.item() inside compiled function
# Fix: @torch.compiler.disable on the function with the branch
# ══════════════════════════════════════════════════════════════════════════════

_COMPILE_GRAPH_BREAK_FILES = {
    "train.py": '''\
import torch
import torch.nn as nn
import torch.optim as optim
from model import VisionTransformer
from config import LEARNING_RATE, NUM_STEPS, GRAD_CLIP_THRESHOLD

@torch.compile
def training_step(model, inputs, targets, criterion):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    # BUG: loss.item() forces a CPU sync, breaking the Dynamo graph.
    # This causes a graph break and full recompilation every step.
    # The branch is data-dependent (depends on runtime tensor value).
    if loss.item() > GRAD_CLIP_THRESHOLD:  # BUG: data-dependent branch with .item()
        outputs = outputs.clamp(-10, 10)
        loss = criterion(outputs, targets)
    return loss

def main():
    device = torch.device("cuda")
    model = VisionTransformer(num_classes=1000).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for step in range(NUM_STEPS):
        inputs = torch.randn(32, 3, 224, 224, device=device)
        targets = torch.randint(0, 1000, (32,), device=device)
        optimizer.zero_grad()
        loss = training_step(model, inputs, targets, criterion)
        loss.backward()
        optimizer.step()
        if step % 20 == 0:
            print(f"Step {step}: loss={loss.item():.4f}")

if __name__ == "__main__":
    main()
''',
    "model.py": '''\
import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)

class VisionTransformer(nn.Module):
    def __init__(self, num_classes=1000, d_model=768, num_heads=12, num_layers=12, patch_size=16):
        super().__init__()
        num_patches = (224 // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        self.layers = nn.ModuleList([MultiHeadSelfAttention(d_model, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x[:, 0]))
''',
    "config.py": '''\
LEARNING_RATE = 1e-4
NUM_STEPS = 500
GRAD_CLIP_THRESHOLD = 5.0
BATCH_SIZE = 32
''',
    "data_loader.py": '''\
import torch
from torch.utils.data import DataLoader, TensorDataset

def get_loader(batch_size=32):
    X = torch.randn(5000, 3, 224, 224)
    y = torch.randint(0, 1000, (5000,))
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
''',
}

_COMPILE_GRAPH_BREAK_BUGGY_TRACE = """\
python train.py

[torch.compile] Compiling training_step...
Step 0: loss=6.9078
[Dynamo] Recompiling training_step (call #1)
Step 20: loss=6.7821
[Dynamo] Recompiling training_step (call #2)
Step 40: loss=6.6234
[Dynamo] Recompiling training_step (call #3)
...
Step 480: loss=5.8821
[Dynamo] Recompiling training_step (call #24)

[PERFORMANCE WARNING] torch.compile is providing NO speedup.
Expected: ~2.4x speedup for ViT-B/16 with torch.compile
Actual:   0.87x (SLOWER than eager due to recompilation overhead)

Throughput:
  Eager (no compile):  187 samples/sec
  With torch.compile:  163 samples/sec  ← SLOWER

Training time: 42 min (expected: 18 min with compile)
"""

_COMPILE_GRAPH_BREAK_DYNAMO_TRACE = """\
TORCH_LOGS=dynamo python train.py

Step 0: loss=6.9078
[2026-04-07 14:30:01] torch._dynamo.convert_frame: [WARNING] Graph break due to:
  File "train.py", line 12, in training_step
    if loss.item() > GRAD_CLIP_THRESHOLD:
  Reason: Data-dependent control flow.
  `loss.item()` forces a CPU sync and returns a Python scalar.
  Branching on a Python scalar derived from a tensor is unsupported.
  The graph has been broken and training_step will be re-traced every call.

  To fix: decorate the function or the branching sub-function with
  @torch.compiler.disable to opt it out of compilation.
  Or restructure to avoid data-dependent branches:
    e.g. torch.where() / tensor.clamp() (stays on-graph, no branch)

[2026-04-07 14:30:01] [recompile stats]
  training_step: 1 compile, 1 graph break at line 12
  Total recompilations: 1 per step (every step hits the same graph break)

Step 20: loss=6.7821
[Dynamo] Recompiling training_step (recompile #2 — same graph break)
"""

_COMPILE_GRAPH_BREAK_SUCCESS_TRACE = """\
TORCH_LOGS=dynamo python train.py

[torch.compile] Compiling training_step...
[2026-04-07 14:30:01] [torch.compile] Full graph captured. 0 graph breaks.
Step 0: loss=6.9078
Step 20: loss=6.7821
Step 40: loss=6.6234
...
Step 480: loss=5.8821

Throughput: 448 samples/sec (2.4x speedup over eager ✓)
Training time: 17.2 min
No recompilations. Graph compiled once and cached.
"""

_COMPILE_GRAPH_BREAK_FIX_SIGNATURES = [
    ("train.py", "@torch.compiler.disable"),  # decorator applied
]


# ══════════════════════════════════════════════════════════════════════════════
# TASK 4: DDP Gradient Synchronization Hang
# Bug: find_unused_parameters=False with conditional model branch
# Fix: find_unused_parameters=True
# ══════════════════════════════════════════════════════════════════════════════

_DDP_GRADIENT_HANG_FILES = {
    "train.py": '''\
import torch
import torch.nn as nn
import torch.distributed as dist
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from model import MultiTaskModel
from config import LEARNING_RATE, NUM_STEPS

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train(rank, world_size):
    setup(rank, world_size)
    model = MultiTaskModel().to(rank)
    # BUG: find_unused_parameters=False means DDP expects ALL parameters
    # to receive gradients every step. But our model has a conditional branch
    # in forward() where aux_head is skipped when use_auxiliary=False.
    # When aux_head parameters don't get gradients, DDP hangs waiting for them.
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)  # BUG
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for step in range(NUM_STEPS):
        inputs = torch.randn(16, 3, 224, 224, device=rank)
        targets = torch.randint(0, 10, (16,), device=rank)
        # In production, we alternate: main task every step, aux task every 5 steps
        use_auxiliary = (step % 5 == 0)
        optimizer.zero_grad()
        outputs = model(inputs, use_auxiliary=use_auxiliary)
        loss = criterion(outputs["main"], targets)
        if use_auxiliary:
            loss = loss + 0.1 * criterion(outputs["aux"], targets)
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print(f"Step {step}: loss={loss.item():.4f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    torch.multiprocessing.spawn(train, args=(4,), nprocs=4, join=True)
''',
    "model.py": '''\
import torch
import torch.nn as nn
import torchvision.models as tvm

class MultiTaskModel(nn.Module):
    """
    Multi-task model with a main classification head and an optional auxiliary head.
    The auxiliary head is only used every 5th training step to save compute.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        backbone = tvm.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # remove final FC
        self.main_head = nn.Linear(512, num_classes)
        self.aux_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, num_classes)
        )  # used only every 5 steps

    def forward(self, x, use_auxiliary=False):
        features = self.backbone(x).squeeze(-1).squeeze(-1)
        out = {"main": self.main_head(features)}
        if use_auxiliary:  # CONDITIONAL: aux_head params get no grad when False
            out["aux"] = self.aux_head(features)
        return out
''',
    "config.py": '''\
LEARNING_RATE = 1e-3
NUM_STEPS = 200
NUM_CLASSES = 10
WORLD_SIZE = 4
''',
    "data_loader.py": '''\
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

def get_loader(rank, world_size, batch_size=16):
    X = torch.randn(2000, 3, 224, 224)
    y = torch.randint(0, 10, (2000,))
    sampler = DistributedSampler(TensorDataset(X, y), num_replicas=world_size, rank=rank)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, sampler=sampler)
''',
}

_DDP_GRADIENT_HANG_BUGGY_TRACE = """\
torchrun --nproc_per_node=4 train.py

[rank0]: DDP model initialized (MultiTaskModel, find_unused_parameters=False)
[rank1]: DDP model initialized
[rank2]: DDP model initialized
[rank3]: DDP model initialized
Step 0: loss=2.3012   [use_auxiliary=True  — aux_head used, all params get gradients ✓]
Step 1: [HANG — all 4 ranks frozen. No output.]
Step 2: ...
Step 3: ...
Step 4: ...

[60 seconds pass in silence]

[2026-04-07 15:11:44] [rank0]: [E ProcessGroupNCCL.cpp:1290] NCCL watchdog timeout.
[rank0]: Timeout after 60s waiting for gradient synchronization.
  DDP is waiting for gradients from parameters that never received them.
  Hanging parameters (no gradient received this step):
    - module.aux_head.0.weight
    - module.aux_head.0.bias
    - module.aux_head.2.weight
    - module.aux_head.2.bias
  These parameters exist in the model but were NOT used in forward() this step.

RuntimeError: DDP has not been updated by all processes for 60 seconds.
  Set find_unused_parameters=True if your model has conditional branches
  where some parameters may not receive gradients every step.

Traceback (most recent call last):
  File "torch/nn/parallel/distributed.py", line 1392, in _check_global_requires_backward_grad_sync
    raise RuntimeError(f"Expected to have finished reduction ...")
  RuntimeError: Expected to have finished reduction in the prior iteration ...

Process finished with exit code 1.
"""

_DDP_GRADIENT_HANG_SUCCESS_TRACE = """\
torchrun --nproc_per_node=4 train.py

[rank0]: DDP model initialized (find_unused_parameters=True — parameters with no grad will be skipped)
Step 0: loss=2.3012   [use_auxiliary=True]
Step 1: loss=2.2891   [use_auxiliary=False — aux_head skipped, DDP handles gracefully]
Step 2: loss=2.2701   [use_auxiliary=False]
Step 3: loss=2.2534
Step 4: loss=2.2298
Step 5: loss=2.2089   [use_auxiliary=True]
...
Step 190: loss=1.8821
Step 199: loss=1.8234
Training completed. 200/200 steps. All ranks synchronized. No NCCL timeouts.
"""

_DDP_GRADIENT_HANG_FIX_SIGNATURES = [
    ("train.py", "find_unused_parameters=True"),
]


# ══════════════════════════════════════════════════════════════════════════════
# Engine
# ══════════════════════════════════════════════════════════════════════════════

TASK_CONFIGS = {
    "oom_graph_leak": {
        "files": _OOM_GRAPH_LEAK_FILES,
        "buggy_trace": _OOM_GRAPH_LEAK_BUGGY_TRACE,
        "success_trace": _OOM_GRAPH_LEAK_SUCCESS_TRACE,
        "diagnostic_trace": None,
        "fix_signatures": _OOM_GRAPH_LEAK_FIX_SIGNATURES,
        "hint": "Run python train.py and examine the memory allocation per epoch. What accumulates?",
    },
    "fsdp_collective_deadlock": {
        "files": _FSDP_DEADLOCK_FILES,
        "buggy_trace": _FSDP_DEADLOCK_BUGGY_TRACE,
        "success_trace": _FSDP_DEADLOCK_SUCCESS_TRACE,
        "diagnostic_trace": None,
        "fix_signatures": _FSDP_DEADLOCK_FIX_SIGNATURES,
        "hint": "The NCCL timeout says a collective op isn't being called by all ranks. Which ranks call dist.all_reduce?",
    },
    "ddp_gradient_hang": {
        "files": _DDP_GRADIENT_HANG_FILES,
        "buggy_trace": _DDP_GRADIENT_HANG_BUGGY_TRACE,
        "success_trace": _DDP_GRADIENT_HANG_SUCCESS_TRACE,
        "diagnostic_trace": None,
        "fix_signatures": _DDP_GRADIENT_HANG_FIX_SIGNATURES,
        "hint": "DDP is waiting for gradients that never arrive. Which model parameters might not get gradients every step?",
    },
}


class MockExecutionEngine:
    """
    Parses VFS file content to detect fix signatures.
    Returns authentic PyTorch error traces or success logs.
    Does NOT import torch — no PyTorch dependency in this file.
    """

    def simulate(
        self,
        task_name: str,
        files: Dict[str, str],
        command: str = "python train.py",
    ) -> Tuple[Literal["passing", "failing", "partial", "diagnostic"], str]:
        """
        Returns (status, output).
        - "passing":    all fix signatures present → success_trace
        - "failing":    no fixes → buggy_trace
        - "partial":    some fixes → partial feedback
        - "diagnostic": command contains diagnostic flag → diagnostic_trace
        """
        cfg = TASK_CONFIGS.get(task_name)
        if not cfg:
            return "failing", f"ERROR: Unknown task '{task_name}'"

        # Check if this is a diagnostic command
        if cfg["diagnostic_trace"] and self._is_diagnostic_command(command, task_name):
            return "diagnostic", cfg["diagnostic_trace"]

        # Check fix signatures
        sigs = cfg["fix_signatures"]
        if not sigs:
            return "passing", cfg["success_trace"]

        # Special case: fsdp deadlock fix check (structural, not simple substring)
        if task_name == "fsdp_collective_deadlock":
            return self._check_fsdp_fix(files, cfg)

        matches = [
            required in files.get(fname, "")
            for fname, required in sigs
        ]
        n_matched = sum(matches)

        if n_matched == len(matches):
            return "passing", cfg["success_trace"]
        if n_matched > 0:
            return "partial", (
                f"[Partial fix detected: {n_matched}/{len(matches)} signatures found]\n"
                f"Some issues remain. Re-run to see remaining errors:\n\n"
                + cfg["buggy_trace"][:800]
                + "\n...[truncated]"
            )
        return "failing", cfg["buggy_trace"]

    def _is_diagnostic_command(self, command: str, task_name: str) -> bool:
        return False

    def _check_fsdp_fix(self, files: Dict[str, str], cfg: dict) -> Tuple[str, str]:
        """
        FSDP fix check: verify dist.all_reduce is NOT inside an `if rank == 0:` block.
        We check that `all_reduce` exists AND it's not indented under the rank guard.
        """
        train_content = files.get("train.py", "")
        has_all_reduce = "dist.all_reduce" in train_content

        # Check if all_reduce is still inside the if rank == 0 block (the bug)
        lines = train_content.split("\n")
        in_rank_guard = False
        all_reduce_in_guard = False
        for line in lines:
            stripped = line.strip()
            if "if rank == 0:" in line or "if dist.get_rank() == 0:" in line:
                in_rank_guard = True
            elif in_rank_guard and stripped and not line.startswith("    " * 2):
                in_rank_guard = False
            if in_rank_guard and "dist.all_reduce" in line:
                all_reduce_in_guard = True

        if has_all_reduce and not all_reduce_in_guard:
            return "passing", cfg["success_trace"]
        return "failing", cfg["buggy_trace"]

    def verify_fix(self, task_name: str, files: Dict[str, str]) -> bool:
        status, _ = self.simulate(task_name, files)
        return status == "passing"
