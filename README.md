# 🔁 Collatz GPU Hunter

A CUDA-based brute-force search for counterexamples to the **Collatz Conjecture**, starting from **2^68 + 1** — one step past the current world record.

> "Take any positive integer. If even, divide by 2. If odd, multiply by 3 and add 1. Repeat. You will always reach 1."  
> — The Collatz Conjecture (unproven since 1937)

This program searches for numbers that **don't** reach 1 — either a non-trivial cycle or a diverging sequence. Finding one would be one of the biggest results in modern mathematics.

---

## Features

- 🚀 **Starts at 2^68 + 1** — beyond all previously verified territory
- ⚡ **Barina early-stop optimization** — stops the moment a sequence drops below its start, skipping ~70% of work
- 🧮 **8-bit lookup table** — processes 8 Collatz steps in a single GPU multiply
- 🔢 **128-bit arithmetic** — handles numbers past 2^64 using `unsigned __int128`
- 🧵 **Branchless ctz128** — trailing zero strip compiles to a SELP instruction, zero warp divergence
- 💾 **Checkpointing** — saves progress every 50 batches, resume anytime with `./collatz`
- 📊 **Live dashboard** — real-time stats including GPU temp, utilization, peak steps, ETA, and progress bar
- 🔬 **Floyd's Tortoise & Hare** — CPU-side cycle detection on any GPU suspect

---

## Dashboard

```
+-------------------------------------------------+
|       COLLATZ BARINA HUNTER  (2^68 -> 2^69)    |
+-------------------------------------------------+
|  Batch           : 439417                       |
|  Numbers checked : 14.382T                      |
|  Speed           : 1.689T              /sec     |
|  Suspects found  : 0                            |
+-------------------------------------------------+
|  PEAK STEPS (hardest number so far)             |
|  Steps           : 595                          |
|  Number (hi/lo)  : 16    / 7371584685851        |
+-------------------------------------------------+
|  GPU                                            |
|  Utilization     :  72%                         |
|  Temperature     :  70 C                        |
+-------------------------------------------------+
|  PROGRESS TO 2^69                               |
|  [##----------------------------]   0.15%       |
|  ETA             : 245.2 days                   |
|  Elapsed         : 9m 32s                       |
+-------------------------------------------------+
```

---

## Requirements

- NVIDIA GPU with CUDA support (tested on RTX 3050 6GB Laptop)
- CUDA Toolkit
- NVML (comes with CUDA)
- GCC / build-essential
- Linux or WSL2 on Windows

---

## Setup

### On Linux / WSL2 (Ubuntu)

```bash
# Install dependencies
sudo apt update
sudo apt install build-essential nvidia-cuda-toolkit -y

# Clone the repo
git clone https://github.com/YOUR_USERNAME/collatz-gpu-hunter
cd collatz-gpu-hunter

# Compile
nvcc -O3 -arch=sm_86 -o collatz collatz_barina.cu -lnvidia-ml

# Run
./collatz
```

> Replace `sm_86` with your GPU's compute capability:  
> RTX 3050/3060/3070/3080 → `sm_86`  
> RTX 2060/2070/2080 → `sm_75`  
> GTX 1060/1070/1080 → `sm_61`  
> Check yours: `nvidia-smi --query-gpu=compute_cap --format=csv`

### On Windows

Use **WSL2** (recommended):
```powershell
wsl --install -d Ubuntu
```
Then follow the Linux steps above inside Ubuntu.

---

## How It Works

### The Conjecture
For any positive integer `n`:
- If `n` is even → `n = n / 2`
- If `n` is odd  → `n = 3n + 1`
- Repeat until `n = 1`

The conjecture says this always terminates. No counterexample has ever been found for positive integers, verified up to **2^68 ≈ 2.95 × 10^20** by David Barina (2020).

### Barina Early Stop
Instead of running each sequence all the way to 1, we exploit induction:

> If all numbers below `N` are already verified, then the moment a sequence drops below its starting value, it's safe — it will eventually reach 1 through already-verified territory.

This skips the vast majority of computation for large numbers near 2^68.

### 8-bit Lookup Table
For an odd number `n`, its bottom 8 bits fully determine the next 8 Collatz steps (since odd/even decisions only depend on lower bits). We precompute a table of `(pow3, addend)` pairs so one iteration becomes:

```
n = (pow3 * n + addend) >> 8
```

One multiply instead of 8 sequential branches.

### Branchless ctz128
Stripping trailing zeros (making `n` odd) is done with a hardware `__ffsll` instruction instead of a loop. The 128-bit version uses a predicated select (SELP) — no branch, no warp divergence:

```cuda
__device__ __forceinline__ int ctz128(u128 n) {
    u64 lo = (u64)n, hi = (u64)(n >> 64);
    return (lo != 0) ? __ffsll(lo) - 1 : 63 + __ffsll(hi);
}
```

### Floyd's Cycle Detection
Any GPU suspect (number that doesn't converge within the iteration limit) is verified on CPU using Floyd's Tortoise & Hare algorithm — O(1) memory cycle detection. If a non-trivial cycle is confirmed, it's written to `suspects.txt`.

---

## Output Files

| File | Contents |
|------|----------|
| `checkpoint.txt` | Current batch number and total numbers checked — **don't delete** |
| `suspects.txt` | Any numbers that didn't converge or triggered a cycle — check these |
| `stats_log.txt` | Per-checkpoint log of speed, temp, suspects |

To resume after stopping:
```bash
./collatz   # automatically reads checkpoint.txt
```

---

## Performance

Tested on **RTX 3050 6GB Laptop GPU** (20 SMs):

| Metric | Value |
|--------|-------|
| Speed | ~1.5–2T numbers/sec |
| GPU Utilization | 55–75% (limited by thread imbalance) |
| GPU Temp | 65–85°C under load |
| Time to cover 2^68 → 2^69 | ~200–250 days |

GPU utilization isn't 100% because Barina early-stop means some threads finish much faster than others in the same warp — fast threads wait for slow ones. This is an inherent tradeoff of the algorithm.

---

## Context

| Milestone | Who | When |
|-----------|-----|------|
| Conjecture proposed | Lothar Collatz | 1937 |
| Verified to 2^60 | Various | ~2000s |
| Verified to 2^68 | David Barina | 2020 |
| Tao's "almost proof" | Terence Tao | 2019 |
| This project starts | You | 2025 |

Terence Tao proved in 2019 that almost all Collatz sequences reach arbitrarily small values — the closest anyone has gotten to a full proof. The conjecture remains open.

---

## Contributing

If you have a stronger GPU or a cluster, you can parallelize across machines by splitting the search range. Each instance just needs a different `START_LO / START_HI` in `main()`.

Pull requests welcome for:
- Better warp utilization strategies
- Multi-GPU support
- Network-distributed search

---

## License

MIT — do whatever you want with it. If you find a counterexample, you're legally obligated to share the Fields Medal.
