/* kairos_train.c — from-scratch trainer for the Kairos resonance heart, on the
 * notorch tape (Chuck optimizer). Arch = the Resonance / Janus form (the same
 * forward kairos.aml runs at inference): parametric RMSNorm → content MHA(RoPE)
 * + RRPRAM low-rank (op 33) blended by a per-head sigmoid gate → SwiGLU.
 *
 * This is the TRAINER. kairos.aml is the INFERENCE. Both share the arch and the
 * K02 weight format (K02 magic + fixed-BPE merge table + tensors) below.
 *
 * Tokenizer = fixed BPE (learn-once, K_MERGES merges → vocab 256+K_MERGES; wall #2,
 * no vocab shift across stages). Overfit the 4 mini-Kairos (>=10k iters), keep the
 * born Kairos to a healthy val — the stop criterion, not iters.
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
#ifndef K_MERGES
#define K_MERGES  1024  /* fixed BPE merges → vocab = 256 + K_MERGES (wall #2) */
#endif
#define K_MAX_MERGES 4096
#define K_HEADDIM (K_DIM / K_HEADS)
#define K_MAGIC   "K02\n"
static int g_vocab = 256;   /* runtime = 256 + BPE.n_merges */

/* ── Fixed BPE (learn-once; arch from RRPRAM/resonance-bpe.c, fast O(n)/merge
 *    pair-count via a hash instead of the O(n^2) reference scan). ── */
typedef struct { int a, b, result; } MergeRule;
static struct { MergeRule merges[K_MAX_MERGES]; int n_merges, vocab_size; } BPE = { .n_merges = 0, .vocab_size = 256 };

static int bpe_encode(const unsigned char *text, int len, int *out, int max) {
    int n = 0;
    for (int i = 0; i < len && n < max; i++) out[n++] = text[i];
    for (int m = 0; m < BPE.n_merges; m++) {
        MergeRule *mr = &BPE.merges[m]; int j = 0;
        for (int i = 0; i < n; i++) {
            if (i + 1 < n && out[i] == mr->a && out[i + 1] == mr->b) { out[j++] = mr->result; i++; }
            else out[j++] = out[i];
        }
        n = j;
    }
    return n;
}

static int bpe_decode_token(int tok, char *buf, int max) {
    if (tok < 256) { if (max > 0) buf[0] = (char)tok; return 1; }
    for (int m = BPE.n_merges - 1; m >= 0; m--)
        if (BPE.merges[m].result == tok) {
            int n1 = bpe_decode_token(BPE.merges[m].a, buf, max);
            return n1 + bpe_decode_token(BPE.merges[m].b, buf + n1, max - n1);
        }
    if (max > 0) buf[0] = '?';
    return 1;
}

static void bpe_learn_merges(const unsigned char *data, int len, int nm) {
    int n = len;
    long HS = 1L << 22;  /* headroom >> distinct adjacent pairs (< vocab^2) */
    int  *tok  = malloc((size_t)len * sizeof(int));
    long *hkey = malloc((size_t)HS * sizeof(long));
    int  *hcnt = malloc((size_t)HS * sizeof(int));
    if (!tok || !hkey || !hcnt) {
        fprintf(stderr, "[kairos_train] BPE OOM\n");
        free(tok); free(hkey); free(hcnt); return;
    }
    for (int i = 0; i < n; i++) tok[i] = data[i];
    int max_m = nm < K_MAX_MERGES ? nm : K_MAX_MERGES;
    for (int m = 0; m < max_m; m++) {
        memset(hkey, 0xff, (size_t)HS * sizeof(long));  /* -1 = empty */
        memset(hcnt, 0, (size_t)HS * sizeof(int));
        long bestKey = -1; int bestCnt = 0;
        for (int i = 0; i + 1 < n; i++) {
            long key = (long)tok[i] * 100000L + tok[i + 1];
            long h = ((key * 2654435761UL) >> 11) & (HS - 1);
            /* probe-cap: distinct adjacent pairs (<= n-1) can never fill HS = 2^22 at
               our vocab scale, but bound the linear probe so a full table can never
               spin forever or read past the slots. If somehow full, skip this pair. */
            long probes = 0;
            while (hkey[h] != -1 && hkey[h] != key && probes < HS) { h = (h + 1) & (HS - 1); probes++; }
            if (probes >= HS) continue;
            hkey[h] = key; hcnt[h]++;
            if (hcnt[h] > bestCnt) { bestCnt = hcnt[h]; bestKey = key; }
        }
        if (bestCnt < 2) break;
        int ba = (int)(bestKey / 100000L), bb = (int)(bestKey % 100000L), nid = 256 + m;
        BPE.merges[m] = (MergeRule){ ba, bb, nid };
        BPE.n_merges = m + 1; BPE.vocab_size = 256 + m + 1;
        int j = 0;
        for (int i = 0; i < n; i++) {
            if (i + 1 < n && tok[i] == ba && tok[i + 1] == bb) { tok[j++] = nid; i++; }
            else tok[j++] = tok[i];
        }
        n = j;
        if ((m + 1) % 256 == 0) fprintf(stderr, "  bpe merge %d/%d vocab=%d tokens=%d\n", m + 1, max_m, nid + 1, n);
    }
    free(hkey); free(hcnt); free(tok);
}

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
    m->wte     = mk(g_vocab, D, s);
    m->lm_head = mk(g_vocab, D, s);
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
    return nt_seq_cross_entropy(logits, tgtIdx, T, g_vocab);
}

/* Cosine LR with warmup. */
static float lr_at(int step, int steps, float base) {
    int warm = steps / 20 + 1;
    if (step < warm) return base * (float)(step + 1) / (float)warm;
    float p = (float)(step - warm) / (float)(steps - warm + 1);
    return base * 0.5f * (1.0f + cosf(3.14159265f * p));
}

/* Held-out validation loss: forward-only over nwin evenly-spaced windows in the
 * val region [lo,hi). No backward, no chuck_step — Chuck state is untouched (the
 * positional param slots are re-registered identically but never stepped). The
 * born-Kairos stop criterion is a healthy val (not iters); minis may overfit on
 * train. Returns -1 if the val region is too small for one window. */
static float eval_val(Model *m, const int *toks, long lo, long hi, int T, int nwin) {
    long span = hi - lo - T - 1;
    if (span < 1) return -1.0f;
    double acc = 0.0; int cnt = 0;
    float tok[K_SEQ], tgt[K_SEQ];
    for (int w = 0; w < nwin; w++) {
        long start = lo + (long)((double)w / (double)nwin * (double)span);
        for (int t = 0; t < T; t++) { tok[t] = (float)toks[start + t]; tgt[t] = (float)toks[start + t + 1]; }
        nt_tape_start();
        Idx ix; register_params(m, &ix);
        nt_tensor *tokT = nt_tensor_new(T); memcpy(tokT->data, tok, T * sizeof(float));
        nt_tensor *tgtT = nt_tensor_new(T); memcpy(tgtT->data, tgt, T * sizeof(float));
        int tokIdx = nt_tape_record(tokT, NT_OP_NONE, -1, -1, 0);
        int tgtIdx = nt_tape_record(tgtT, NT_OP_NONE, -1, -1, 0);
        nt_tensor_free(tokT); nt_tensor_free(tgtT);
        int lossIdx = forward(m, &ix, tokIdx, tgtIdx, T);
        acc += entry_scalar(lossIdx); cnt++;
        nt_tape_clear();
    }
    return cnt ? (float)(acc / cnt) : -1.0f;
}

static void save_ckpt(Model *m, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[kairos_train] cannot write %s\n", path); return; }
    int cfg[6] = { g_vocab, K_DIM, K_HEADS, K_LAYERS, K_SEQ, K_RANK };
    fwrite(K_MAGIC, 1, 4, f);
    fwrite(cfg, sizeof(int), 6, f);
    fwrite(&BPE.n_merges, sizeof(int), 1, f);
    fwrite(BPE.merges, sizeof(MergeRule), BPE.n_merges, f);
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

    /* Fixed BPE: learn merges once, encode the whole corpus into tokens (wall #2). */
    fprintf(stderr, "[kairos_train] learning BPE (%d merges) on %ld bytes...\n", K_MERGES, n);
    bpe_learn_merges(buf, (int)n, K_MERGES);
    g_vocab = BPE.vocab_size;
    int *toks = malloc((size_t)n * sizeof(int));
    long ntok = bpe_encode(buf, (int)n, toks, (int)n);
    free(buf);
    if (ntok < K_SEQ + 2) { fprintf(stderr, "[kairos_train] too few tokens (%ld)\n", ntok); free(toks); return 1; }
    fprintf(stderr, "[kairos_train] BPE vocab=%d | %ld bytes -> %ld tokens | D=%d H=%d L=%d T=%d R=%d\n",
            g_vocab, n, ntok, K_DIM, K_HEADS, K_LAYERS, K_SEQ, K_RANK);

    /* Held-out val split: last 10% of tokens. The born Kairos stops on a healthy
     * val (not iters); minis may overfit train. If either the train region or the
     * val slice is too small for one window, fall back to training on the FULL
     * corpus (no val) — never silently withhold data. */
    long train_end = (long)((double)ntok * 0.9);
    int have_val = (train_end >= (long)(K_SEQ + 2)) && ((ntok - train_end) >= (long)(K_SEQ + 2));
    if (!have_val) train_end = ntok;                 /* no val → train on everything */
    long val_lo = train_end, val_hi = ntok;          /* val_lo == val_hi when no val */
    fprintf(stderr, "[kairos_train] split: train[0,%ld) val[%ld,%ld) %s\n",
            train_end, val_lo, val_hi,
            have_val ? "" : "(val region too small — training on full corpus)");

    srand((unsigned)time(NULL));
    Model m; model_init(&m);
    float base_lr = 3e-4f, ema = 0.0f;
    int T = K_SEQ;
    float tok[K_SEQ], tgt[K_SEQ];

    for (int step = 0; step < steps; step++) {
        /* upper bound train_end - T (exclusive rand < 1.0) → starts [0, train_end-T-1],
           so the last legal window (target reads train_end-1) is sampled; no OOB, no
           1-token gap before the val region. */
        long start = (long)((double)rand() / ((double)RAND_MAX + 1.0) * (double)(train_end - T));
        for (int t = 0; t < T; t++) { tok[t] = (float)toks[start + t]; tgt[t] = (float)toks[start + t + 1]; }

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
        if (step % 500 == 0 || step == steps - 1) {
            float val = have_val ? eval_val(&m, toks, val_lo, val_hi, T, 16) : -1.0f;
            if (val >= 0.0f)
                fprintf(stderr, "step %6d | train %.4f | val %.4f | ema %.4f | lr %.2e\n",
                        step, loss, val, ema, lr_at(step, steps, base_lr));
            else
                fprintf(stderr, "step %6d | train %.4f | ema %.4f | lr %.2e\n",
                        step, loss, ema, lr_at(step, steps, base_lr));
        } else if (step % 100 == 0) {
            fprintf(stderr, "step %6d | train %.4f | ema %.4f | lr %.2e\n",
                    step, loss, ema, lr_at(step, steps, base_lr));
        }
    }

    save_ckpt(&m, out);
    free(toks);
    return 0;
}
