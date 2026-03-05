// collatz_barina.cu  —  Barina hunter | Dashboard: Peak steps, GPU temp/util, ETA
//
// Compile: nvcc -O3 -arch=sm_86 -o collatz collatz_barina.cu -lnvidia-ml
// Run:     ./collatz

#include <cuda_runtime.h>
#include <nvml.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>

typedef unsigned __int128  u128;
typedef unsigned long long u64;

#define K               8
#define TSIZE           256
#define CHECKPOINT_FILE "checkpoint.txt"
#define SUSPECT_FILE    "suspects.txt"
#define LOG_FILE        "stats_log.txt"
#define SAVE_EVERY      50

// Goal: search up to 2^69 (one full power past the record)
// 2^69 - 2^68 = 2^68 numbers to check
// In odd numbers that's 2^67
#define GOAL_ODD  9223372036854775808ULL   // 2^63 odd numbers to check

struct Entry { uint32_t pow3, addend; };
__constant__ Entry d_table[TSIZE];

// ── Branchless ctz128 — SELP, zero warp divergence ───────────────────────────
__device__ __forceinline__ int ctz128(u128 n) {
    u64 lo    = (u64)n;
    u64 hi    = (u64)(n >> 64);
    int tz_lo = __ffsll(lo) - 1;
    int tz_hi = 63 + __ffsll(hi);
    return (lo != 0) ? tz_lo : tz_hi;
}

// ── Lookup table ──────────────────────────────────────────────────────────────
void build_table(Entry *t) {
    for (int v = 1; v < TSIZE; v += 2) {
        u64 cur = v;
        u64 p3 = 1, add = 0;
        int shifts = 0;
        while (shifts < K) {
            if (cur & 1) {
                add = 3*add + (1ULL << shifts);
                p3 *= 3;
                cur = 3*cur + 1;
            } else {
                shifts++;
                cur >>= 1;
            }
        }
        t[v] = { (uint32_t)p3, (uint32_t)add };
    }
}

// ── GPU Kernel ────────────────────────────────────────────────────────────────
__global__ void hunt(
    u64 range_lo,   u64 range_hi,
    u64 batch_size,
    u64 *suspects,  u64 *nsusp,
    u64 *peak_steps_out,
    u64 *peak_n_lo, u64 *peak_n_hi
) {
    u64 idx = (u64)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= batch_size) return;

    u128 start = ((u128)range_hi << 64) | range_lo;
    start += 2ULL * idx;
    if (!(start & 1)) start += 1;

    u128 n     = start;
    u64  steps = 0;

    for (int iter = 0; iter < 1000000; iter++) {
        n >>= ctz128(n);
        steps++;

        if (n < start) goto done;
        if (n <= 4)    goto done;

        {
            int v   = (int)(n & (TSIZE - 1));
            Entry e = d_table[v];
            n = ((u128)e.pow3 * n + (u128)e.addend) >> K;
            steps += K;
        }
    }

    // Suspect
    {
        u64 pos = atomicAdd(nsusp, 1ULL);
        if (pos < 4096) {
            suspects[2*pos]   = (u64)(start);
            suspects[2*pos+1] = (u64)(start >> 64);
        }
    }

done:
    // Track peak steps — atomicMax + race to store which n
    u64 old = atomicMax(peak_steps_out, steps);
    if (steps > old) {
        atomicExch(peak_n_lo, (u64)(start));
        atomicExch(peak_n_hi, (u64)(start >> 64));
    }
}

// ── CPU: Floyd cycle detection ────────────────────────────────────────────────
void cpu_verify(u128 start) {
    printf("\n[VERIFY] hi=%llu lo=%llu\n",
           (u64)(start >> 64), (u64)start);
    auto step = [](u128 x) -> u128 {
        return (x & 1) ? 3*x + 1 : x >> 1;
    };
    u128 slow = start, fast = start;
    for (int i = 0; i < 100000000; i++) {
        slow = step(slow);
        fast = step(step(fast));
        if (slow == 1 || fast == 1) { printf("   -> Reaches 1. False alarm.\n"); return; }
        if (slow == fast) {
            printf("\n");
            printf("   =============================================\n");
            printf("   ===   NON-TRIVIAL CYCLE FOUND!!!         ===\n");
            printf("   ===   COLLATZ CONJECTURE BROKEN!         ===\n");
            printf("   ===       FIELDS MEDAL TIME!             ===\n");
            printf("   =============================================\n");
            printf("   Cycle: hi=%llu lo=%llu\n\n", (u64)(slow>>64),(u64)slow);
            FILE *f = fopen(SUSPECT_FILE, "a");
            if (f) {
                fprintf(f, "CYCLE: start_hi=%llu start_lo=%llu cycle_hi=%llu cycle_lo=%llu\n",
                        (u64)(start>>64),(u64)start,(u64)(slow>>64),(u64)slow);
                fclose(f);
            }
            return;
        }
    }
    printf("   -> 100M steps, no convergence. Saved.\n");
    FILE *f = fopen(SUSPECT_FILE, "a");
    if (f) { fprintf(f, "SUSPECT: hi=%llu lo=%llu\n",(u64)(start>>64),(u64)start); fclose(f); }
}

// ── Checkpoint ────────────────────────────────────────────────────────────────
int load_checkpoint(u64 *total_odd_done) {
    FILE *f = fopen(CHECKPOINT_FILE, "r");
    if (!f) { *total_odd_done = 0; return 0; }
    int batch = 0;
    fscanf(f, "%d\n%llu\n", &batch, total_odd_done);
    fclose(f);
    return batch;
}
void save_checkpoint(int batch, u64 total_odd_done) {
    FILE *f = fopen(CHECKPOINT_FILE, "w");
    if (!f) return;
    fprintf(f, "%d\n%llu\n", batch, total_odd_done);
    fclose(f);
}

// ── Format helpers ────────────────────────────────────────────────────────────
void fmt_big(char *buf, u64 n) {
    if      (n >= 1000000000000ULL) sprintf(buf, "%.3fT", n/1e12);
    else if (n >= 1000000000ULL)    sprintf(buf, "%.3fB", n/1e9);
    else if (n >= 1000000ULL)       sprintf(buf, "%.3fM", n/1e6);
    else if (n >= 1000ULL)          sprintf(buf, "%.2fK", n/1e3);
    else                            sprintf(buf, "%llu",  n);
}

void fmt_time(char *buf, double sec) {
    if (sec < 0)              { sprintf(buf, "---"); return; }
    if (sec < 60)             { sprintf(buf, "%.0fs", sec); return; }
    if (sec < 3600)           { sprintf(buf, "%.0fm %.0fs", sec/60, fmod(sec,60)); return; }
    if (sec < 86400)          { sprintf(buf, "%.0fh %.0fm", sec/3600, fmod(sec,3600)/60); return; }
    if (sec < 86400*365)      { sprintf(buf, "%.1f days", sec/86400); return; }
    sprintf(buf, "%.1f years", sec/86400/365);
}

// ── Draw dashboard ────────────────────────────────────────────────────────────
void draw_dashboard(
    int   batch,
    u64   total_odd_done,
    u64   peak_steps_ever,
    u64   peak_n_lo, u64 peak_n_hi,
    u64   total_suspects,
    double elapsed,
    unsigned int gpu_util,   // %
    unsigned int gpu_temp    // celsius
) {
    // Move cursor to top-left every time — no flicker, true static dashboard
    printf("\033[H");

    // ETA: based on current speed vs remaining odd numbers to 2^69
    double odd_per_sec = elapsed > 0 ? (double)total_odd_done / elapsed : 0;
    double goal_d      = 9223372036854775808.0;   // 2^63 as double
    double done_d      = (double)total_odd_done;
    double pct         = done_d / goal_d * 100.0;
    if (pct > 100.0) pct = 100.0;
    double remaining   = (done_d < goal_d) ? (goal_d - done_d) : 0.0;
    double eta_sec     = (odd_per_sec > 0 && remaining > 0) ? remaining / odd_per_sec : -1;

    char buf_checked[32], buf_speed[32], buf_peak[32];
    char buf_eta[32],     buf_elapsed[32];
    fmt_big(buf_checked, total_odd_done * 2);
    fmt_big(buf_speed,   (u64)odd_per_sec * 2);
    fmt_big(buf_peak,    peak_steps_ever);
    fmt_time(buf_eta,    eta_sec);
    fmt_time(buf_elapsed, elapsed);

    // Progress bar (30 chars)
    char bar[31]; memset(bar, '-', 30); bar[30] = '\0';
    int filled = (int)(pct / 100.0 * 30);
    if (filled > 30) filled = 30;
    for (int i = 0; i < filled; i++) bar[i] = '#';

    printf("+-------------------------------------------------+\n");
    printf("|       COLLATZ BARINA HUNTER  (2^68 -> 2^69)    |\n");
    printf("+-------------------------------------------------+\n");
    printf("|  Batch           : %-28d |\n", batch);
    printf("|  Numbers checked : %-28s |\n", buf_checked);
    printf("|  Speed           : %-21s /sec |\n", buf_speed);
    printf("|  Suspects found  : %-28llu |\n", total_suspects);
    printf("+-------------------------------------------------+\n");
    printf("|  PEAK STEPS (hardest number so far)            |\n");
    printf("|  Steps           : %-28s |\n", buf_peak);
    printf("|  Number (hi/lo)  : %-5llu / %-20llu |\n", peak_n_hi, peak_n_lo);
    printf("+-------------------------------------------------+\n");
    printf("|  GPU                                           |\n");
    printf("|  Utilization     : %3u%%                        |\n", gpu_util);
    printf("|  Temperature     : %3u C                        |\n", gpu_temp);
    printf("+-------------------------------------------------+\n");
    printf("|  PROGRESS TO 2^69                              |\n");
    printf("|  [%-30s] %5.2f%%   |\n", bar, pct);
    printf("|  ETA             : %-28s |\n", buf_eta);
    printf("|  Elapsed         : %-28s |\n", buf_elapsed);
    printf("+-------------------------------------------------+\n");
    fflush(stdout);
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main() {
    // Init NVML for GPU temp + utilization
    nvmlInit();
    nvmlDevice_t nvml_dev;
    nvmlDeviceGetHandleByIndex(0, &nvml_dev);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("=================================================\n");
    printf("  Collatz Barina Hunter — Dashboard Mode\n");
    printf("=================================================\n");
    printf("GPU  : %s\n", prop.name);
    printf("SMs  : %d  |  Warp: %d\n\n",
           prop.multiProcessorCount, prop.warpSize);

    Entry h_table[TSIZE] = {};
    build_table(h_table);
    cudaMemcpyToSymbol(d_table, h_table, sizeof(h_table));
    printf("Lookup table ready.\n");

    // 2^68 + 1 → hi=16, lo=1
    const u64 START_HI = 16ULL;
    const u64 START_LO = 1ULL;

    u64 total_odd_done = 0;
    int start_batch    = load_checkpoint(&total_odd_done);
    if (start_batch > 0)
        printf("Resuming from batch %d (%llu odd numbers done)\n\n",
               start_batch, total_odd_done);
    else
        printf("Starting fresh from 2^68 + 1\n\n");

    // GPU buffers
    u64 *d_suspects, *d_nsusp;
    u64 *d_peak_steps, *d_peak_n_lo, *d_peak_n_hi;
    cudaMalloc(&d_suspects,   4096 * 2 * sizeof(u64));
    cudaMalloc(&d_nsusp,      sizeof(u64));
    cudaMalloc(&d_peak_steps, sizeof(u64));
    cudaMalloc(&d_peak_n_lo,  sizeof(u64));
    cudaMalloc(&d_peak_n_hi,  sizeof(u64));

    const u64 BATCH    = 1ULL << 25;   // 32M odd per batch — tuned for RTX 3050 6GB
    const int BLOCK_SZ = 256;
    const int BLOCKS   = (int)((BATCH + BLOCK_SZ - 1) / BLOCK_SZ);

    u64 h_suspects[4096 * 2];
    u64 h_nsusp, h_peak_steps, h_peak_n_lo, h_peak_n_hi;
    u64 total_suspects      = 0;
    u64 peak_steps_ever     = 0;
    u64 peak_ever_lo        = 0, peak_ever_hi = 0;

    clock_t t0 = clock();

    printf("\033[2J");  // clear screen once at startup

    for (int b = start_batch; ; b++) {
        u64 delta    = 2ULL * (u64)b * BATCH;
        u64 batch_lo = START_LO + delta;
        u64 carry    = (batch_lo < START_LO) ? 1ULL : 0ULL;
        u64 batch_hi = START_HI + carry;

        cudaMemset(d_nsusp,      0, sizeof(u64));
        cudaMemset(d_peak_steps, 0, sizeof(u64));
        cudaMemset(d_peak_n_lo,  0, sizeof(u64));
        cudaMemset(d_peak_n_hi,  0, sizeof(u64));

        hunt<<<BLOCKS, BLOCK_SZ>>>(
            batch_lo, batch_hi, BATCH,
            d_suspects, d_nsusp,
            d_peak_steps, d_peak_n_lo, d_peak_n_hi
        );
        cudaDeviceSynchronize();

        cudaMemcpy(&h_nsusp,      d_nsusp,      sizeof(u64), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_peak_steps, d_peak_steps,  sizeof(u64), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_peak_n_lo,  d_peak_n_lo,   sizeof(u64), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_peak_n_hi,  d_peak_n_hi,   sizeof(u64), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_suspects,    d_suspects, 4096*2*sizeof(u64), cudaMemcpyDeviceToHost);

        total_odd_done  += BATCH;
        total_suspects  += h_nsusp;

        if (h_peak_steps > peak_steps_ever) {
            peak_steps_ever = h_peak_steps;
            peak_ever_lo    = h_peak_n_lo;
            peak_ever_hi    = h_peak_n_hi;
        }

        // GPU stats via NVML
        nvmlUtilization_t util;
        unsigned int temp = 0;
        nvmlDeviceGetUtilizationRates(nvml_dev, &util);
        nvmlDeviceGetTemperature(nvml_dev, NVML_TEMPERATURE_GPU, &temp);

        double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;

        draw_dashboard(
            b+1,
            total_odd_done,
            peak_steps_ever,
            peak_ever_lo, peak_ever_hi,
            total_suspects,
            elapsed,
            util.gpu, temp
        );

        // Verify suspects
        if (h_nsusp > 0) {
            printf("\n[!] %llu suspect(s) in batch %d!\n", h_nsusp, b+1);
            u64 n = h_nsusp < 4096 ? h_nsusp : 4096;
            for (u64 i = 0; i < n; i++) {
                u128 s = ((u128)h_suspects[2*i+1] << 64) | h_suspects[2*i];
                cpu_verify(s);
            }
        }

        // Save
        if (b % SAVE_EVERY == 0) {
            save_checkpoint(b, total_odd_done);
            FILE *log = fopen(LOG_FILE, "a");
            if (log) {
                fprintf(log,
                    "batch=%d checked=%llu peak_steps=%llu suspects=%llu gpu=%u%% temp=%uC elapsed=%.1f\n",
                    b+1, total_odd_done*2, peak_steps_ever,
                    total_suspects, util.gpu, temp, elapsed);
                fclose(log);
            }
        }
    }

    cudaFree(d_suspects);  cudaFree(d_nsusp);
    cudaFree(d_peak_steps);cudaFree(d_peak_n_lo);cudaFree(d_peak_n_hi);
    nvmlShutdown();
    return 0;
}
