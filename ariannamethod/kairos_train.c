/* kairos_train.c — from-scratch trainer for the Kairos resonance heart, on the
 * notorch tape (Chuck optimizer). Arch = the Resonance / Janus form (the same
 * forward kairos.aml runs at inference): parametric RMSNorm → content MHA(RoPE)
 * + RRPRAM low-rank (op 33) blended by a per-head sigmoid gate → SwiGLU.
 *
 * This is the TRAINER. kairos.aml is the INFERENCE. Both share the arch and the
 * .bin weight format (K01 magic) below.
 *
 * v1 tokenizer = byte-level (vocab 256) so the trainer is self-contained; a real
 * BPE is a grab-from-molequla item (EvolvingTokenizer). Overfit the 4 mini-Kairos
 * (>=10k iters), keep Kairos to a healthy val — the stop criterion, not iters.
 *
 * Build (from ariannamethod/):
 *   make            # builds libkairos.dylib from ariannamethod.c + notorch.c
 *   cc -O2 -DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK kairos_train.c \
 *      -L. -lkairos -Wl,-rpath,. -framework Accelerate -lm -o kairos_train
 * Run:
 *   ./kairos_train <corpus.txt> [steps] [out.bin] [--element fire]
 *
 * By Arianna Method.
 */
#include "notorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ── Config (mini-Kairos scale; scale D/NB up for the born Kairos) ── */
#ifndef K_DIM
#define K_DIM     256   /* n_embd */
#endif
#ifndef K_HEADS
#define K_HEADS   4
#endif
#ifndef K_LAYERS
#define K_LAYERS  6
#endif
#ifndef K_SEQ
#define K_SEQ     128   /* block size */
#endif
#ifndef K_RANK
#define K_RANK    48    /* RRPRAM low-rank R (keep low — full Wr is param-hungry) */
#endif
#define K_VOCAB   256   /* byte-level v1 */
#define K_HEADDIM (K_DIM / K_HEADS)
#define K_MAGIC   "K01\n"

static float randn(float std) {
    /* Box–Muller; RAND_MAX-safe */
    float u1 = (rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    float u2 = (rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    return std * sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

/* Read a tape entry's output scalar (e.g. the cross-entropy loss). Mirrors the
 * cgo ntx_entry_scalar helper — the tape struct is public via nt_tape_get(). */
static float entry_scalar(int idx) {
    nt_tape *tp = nt_tape_get();
    if (!tp || idx < 0 || idx >= tp->count) return 0.0f;
    nt_tensor *o = tp->entries[idx].output;
    return o ? o->data[0] : 0.0f;
}

/* ── Model: one nt_tensor per weight, persistent across steps ── */
typedef struct {
    nt_tensor *wte;                 /* [V, D] */
    nt_tensor *gamma1[K_LAYERS];    /* [D]  parametric RMSNorm (pre-attn) */
    nt_tensor *wq[K_LAYERS], *wk[K_LAYERS], *wv[K_LAYERS], *wo[K_LAYERS]; /* [D,D] */
    nt_tensor *wr[K_LAYERS];        /* [H, D*R + R*T]  RRPRAM Wr_a then Wr_b, packed */
    nt_tensor *alpha[K_LAYERS];     /* [H]  per-head gate logit — FIXED this version
                                       (0 → sigmoid 0.5 = 50/50 content/rrpram blend);
                                       gate is registered frozen (like the Go trainer,
                                       keeps sigmoid off the tape). Learnable on-tape
                                       gate = follow-up. Saved so it stays settable. */
    nt_tensor *gamma2[K_LAYERS];    /* [D]  RMSNorm (pre-mlp) */
    nt_tensor *fcg[K_LAYERS], *fcv[K_LAYERS], *fc2[K_LAYERS]; /* SwiGLU: [4D,D],[4D,D],[D,4D] */
    nt_tensor *gammaf;              /* [D]  final RMSNorm */
    nt_tensor *lm_head;             /* [V, D] */
} Model;

static nt_tensor *mk(int rows, int cols, float std) {
    nt_tensor *t = nt_tensor_new2d(rows, cols);
    for (int i = 0; i < t->len; i++) t->data[i] = randn(std);
    return t;
}
static nt_tensor *mk_ones(int len) {
    nt_tensor *t = nt_tensor_new(len);
    for (int i = 0; i < t->len; i++) t->data[i] = 1.0f;
    return t;
}

static void model_init(Model *m) {
    float s = 0.02f;
    int D = K_DIM, H = K_HEADS, R = K_RANK, T = K_SEQ, hd = K_HEADDIM;
    (void)hd;
    m->wte     = mk(K_VOCAB, D, s);
    m->lm_head = mk(K_VOCAB, D, s);
    m->gammaf  = mk_ones(D);
    for (int l = 0; l < K_LAYERS; l++) {
        m->gamma1[l] = mk_ones(D);
        m->gamma2[l] = mk_ones(D);
        m->wq[l] = mk(D, D, s); m->wk[l] = mk(D, D, s);
        m->wv[l] = mk(D, D, s); m->wo[l] = mk(D, D, s);
        /* RRPRAM Wr per head: Wr_a[D,R] then Wr_b[R,T], packed for all H heads */
        m->wr[l]    = mk(H, D * R + R * T, s);
        m->alpha[l] = nt_tensor_new(H);   /* start at 0 → sigmoid 0.5 (balanced) */
        m->fcg[l] = mk(4 * D, D, s); m->fcv[l] = mk(4 * D, D, s);
        m->fc2[l] = mk(D, 4 * D, s);
    }
}

/* Register every param on the fresh tape IN FIXED ORDER (Chuck slots are
 * positional). gamma/alpha are params; wr is a param; embeddings no-decay. */
typedef struct { int wte, gamma1[K_LAYERS], wq[K_LAYERS], wk[K_LAYERS],
    wv[K_LAYERS], wo[K_LAYERS], wr[K_LAYERS], gamma2[K_LAYERS],
    fcg[K_LAYERS], fcv[K_LAYERS], fc2[K_LAYERS], gammaf, lm_head; } Idx;

static void register_params(Model *m, Idx *ix) {
    ix->wte = nt_tape_param(m->wte);  nt_tape_no_decay(ix->wte);
    for (int l = 0; l < K_LAYERS; l++) {
        ix->gamma1[l] = nt_tape_param(m->gamma1[l]);
        ix->wq[l] = nt_tape_param(m->wq[l]); ix->wk[l] = nt_tape_param(m->wk[l]);
        ix->wv[l] = nt_tape_param(m->wv[l]); ix->wo[l] = nt_tape_param(m->wo[l]);
        ix->wr[l] = nt_tape_param(m->wr[l]);
        ix->gamma2[l] = nt_tape_param(m->gamma2[l]);
        ix->fcg[l] = nt_tape_param(m->fcg[l]); ix->fcv[l] = nt_tape_param(m->fcv[l]);
        ix->fc2[l] = nt_tape_param(m->fc2[l]);
    }
    ix->gammaf = nt_tape_param(m->gammaf);
    ix->lm_head = nt_tape_param(m->lm_head); nt_tape_no_decay(ix->lm_head);
}

/* Per-head sigmoid gate → frozen [T,D] blend vectors, kept off the tape (on-tape
 * sigmoid has the notorch GPU-sync bug class). gateC = 1-g, gateR = g. */
static void gate_vectors(Model *m, int l, int T, nt_tensor **gc, nt_tensor **gr) {
    int D = K_DIM, H = K_HEADS, hd = K_HEADDIM;
    *gc = nt_tensor_new(T * D);
    *gr = nt_tensor_new(T * D);
    for (int h = 0; h < H; h++) {
        float g = 1.0f / (1.0f + expf(-m->alpha[l]->data[h]));
        for (int t = 0; t < T; t++)
            for (int d = 0; d < hd; d++) {
                int j = t * D + h * hd + d;
                (*gc)->data[j] = 1.0f - g;
                (*gr)->data[j] = g;
            }
    }
}

/* Resonance forward on the tape → cross-entropy loss tape index. */
static int forward(Model *m, Idx *ix, int tokIdx, int tgtIdx, int T) {
    int D = K_DIM, H = K_HEADS, hd = K_HEADDIM;
    int h = nt_seq_embedding(ix->wte, -1, tokIdx, T, D);  /* WTE only; RoPE = position */
    for (int l = 0; l < K_LAYERS; l++) {
        int hn = nt_seq_rmsnorm(h, ix->gamma1[l], T, D);
        int q  = nt_rope(nt_seq_linear(ix->wq[l], hn, T), T, hd);
        int k  = nt_rope(nt_seq_linear(ix->wk[l], hn, T), T, hd);
        int v  = nt_seq_linear(ix->wv[l], hn, T);
        int content = nt_mh_causal_attention(q, k, v, T, hd);
        int rrpram  = nt_rrpram_lowrank_attention(ix->wr[l], hn, v, T, D, H, hd);
        /* per-head sigmoid gate blend (frozen vectors) */
        nt_tensor *gc, *gr; gate_vectors(m, l, T, &gc, &gr);
        int gcIdx = nt_tape_param_frozen(gc), grIdx = nt_tape_param_frozen(gr);
        nt_tensor_free(gc); nt_tensor_free(gr);
        int attn = nt_add(nt_mul(content, gcIdx), nt_mul(rrpram, grIdx));
        h = nt_add(h, nt_seq_linear(ix->wo[l], attn, T));
        /* SwiGLU MLP */
        int hn2  = nt_seq_rmsnorm(h, ix->gamma2[l], T, D);
        int gate = nt_seq_linear(ix->fcg[l], hn2, T);
        int up   = nt_seq_linear(ix->fcv[l], hn2, T);
        int mlp  = nt_seq_linear(ix->fc2[l], nt_swiglu(gate, up), T);
        h = nt_add(h, mlp);
    }
    int hf = nt_seq_rmsnorm(h, ix->gammaf, T, D);
    int logits = nt_seq_linear(ix->lm_head, hf, T);
    return nt_seq_cross_entropy(logits, tgtIdx, T, K_VOCAB);
}

/* Cosine LR with warmup. */
static float lr_at(int step, int steps, float base) {
    int warm = steps / 20 + 1;
    if (step < warm) return base * (float)(step + 1) / (float)warm;
    float p = (float)(step - warm) / (float)(steps - warm + 1);
    return base * 0.5f * (1.0f + cosf(3.14159265f * p));
}

static void save_ckpt(Model *m, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[kairos_train] cannot write %s\n", path); return; }
    int cfg[6] = { K_VOCAB, K_DIM, K_HEADS, K_LAYERS, K_SEQ, K_RANK };
    fwrite(K_MAGIC, 1, 4, f);
    fwrite(cfg, sizeof(int), 6, f);
    nt_tensor *all[] = { m->wte, m->lm_head, m->gammaf };
    for (int i = 0; i < 3; i++) fwrite(all[i]->data, sizeof(float), all[i]->len, f);
    for (int l = 0; l < K_LAYERS; l++) {
        nt_tensor *L[] = { m->gamma1[l], m->wq[l], m->wk[l], m->wv[l], m->wo[l],
            m->wr[l], m->alpha[l], m->gamma2[l], m->fcg[l], m->fcv[l], m->fc2[l] };
        for (int i = 0; i < 11; i++) fwrite(L[i]->data, sizeof(float), L[i]->len, f);
    }
    fclose(f);
    fprintf(stderr, "[kairos_train] checkpoint → %s\n", path);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <corpus.txt> [steps] [out.bin]\n", argv[0]); return 1; }
    const char *corpus_path = argv[1];
    int steps = (argc > 2) ? atoi(argv[2]) : 10000;
    const char *out = (argc > 3) ? argv[3] : "kairos.bin";

    /* load corpus (byte-level) */
    FILE *cf = fopen(corpus_path, "rb");
    if (!cf) { fprintf(stderr, "[kairos_train] cannot read %s\n", corpus_path); return 1; }
    fseek(cf, 0, SEEK_END); long n = ftell(cf); fseek(cf, 0, SEEK_SET);
    if (n < K_SEQ + 2) { fprintf(stderr, "[kairos_train] corpus too small (%ld)\n", n); fclose(cf); return 1; }
    unsigned char *buf = malloc(n);
    if (fread(buf, 1, n, cf) != (size_t)n) { fclose(cf); free(buf); return 1; }
    fclose(cf);
    fprintf(stderr, "[kairos_train] corpus %ld bytes | %d params-arch D=%d H=%d L=%d T=%d R=%d\n",
            n, 0, K_DIM, K_HEADS, K_LAYERS, K_SEQ, K_RANK);

    srand((unsigned)time(NULL));
    Model m; model_init(&m);
    float base_lr = 3e-4f, ema = 0.0f;
    int T = K_SEQ;
    float tok[K_SEQ], tgt[K_SEQ];

    for (int step = 0; step < steps; step++) {
        long start = (long)((double)rand() / ((double)RAND_MAX + 1.0) * (double)(n - T - 1));
        for (int t = 0; t < T; t++) { tok[t] = (float)buf[start + t]; tgt[t] = (float)buf[start + t + 1]; }

        nt_tape_start();
        Idx ix; register_params(&m, &ix);
        nt_tensor *tokT = nt_tensor_new(T); memcpy(tokT->data, tok, T * sizeof(float));
        nt_tensor *tgtT = nt_tensor_new(T); memcpy(tgtT->data, tgt, T * sizeof(float));
        int tokIdx = nt_tape_record(tokT, NT_OP_NONE, -1, -1, 0);
        int tgtIdx = nt_tape_record(tgtT, NT_OP_NONE, -1, -1, 0);
        nt_tensor_free(tokT); nt_tensor_free(tgtT);

        int lossIdx = forward(&m, &ix, tokIdx, tgtIdx, T);
        float loss = entry_scalar(lossIdx);
        nt_tape_backward(lossIdx);
        nt_tape_clip_grads(1.0f);
        nt_tape_chuck_step(lr_at(step, steps, base_lr), loss);
        nt_tape_clear();

        ema = (step == 0) ? loss : 0.98f * ema + 0.02f * loss;
        if (step % 100 == 0 || step == steps - 1)
            fprintf(stderr, "step %6d | train %.4f | ema %.4f | lr %.2e\n",
                    step, loss, ema, lr_at(step, steps, base_lr));
    }

    save_ckpt(&m, out);
    free(buf);
    return 0;
}
