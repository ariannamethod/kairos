package gotools

/*
#cgo CFLAGS: -I${SRCDIR}/../ariannamethod -O2 -DUSE_BLAS
#cgo darwin CFLAGS: -DACCELERATE -DACCELERATE_NEW_LAPACK
#cgo LDFLAGS: -lm
#cgo darwin LDFLAGS: -L${SRCDIR}/../ariannamethod -lkairos -framework Accelerate -Wl,-rpath,${SRCDIR}/../ariannamethod
#cgo linux CFLAGS: -DUSE_BLAS -I/usr/include/x86_64-linux-gnu/openblas-pthread/
#include "notorch.h"
#include <string.h>

// Copy a Go float32 slice into a tensor's data buffer.
static void ntx_set(nt_tensor* t, float* src, int n) {
    if (t && src && n > 0 && n <= t->len) memcpy(t->data, src, (size_t)n * sizeof(float));
}
// Copy a tensor's data buffer out into a caller buffer.
static void ntx_get(nt_tensor* t, float* dst, int n) {
    if (t && dst && n > 0 && n <= t->len) memcpy(dst, t->data, (size_t)n * sizeof(float));
}
// Read the scalar at a tape entry's output (loss lives at output->data[0]).
static float ntx_entry_scalar(int idx) {
    nt_tape* tp = nt_tape_get();
    if (!tp || idx < 0 || idx >= tp->count) return 0.0f;
    nt_tensor* o = tp->entries[idx].output;
    return (o && o->len > 0) ? o->data[0] : 0.0f;
}
// Copy a tape entry's full output tensor out (for op-output parity checks).
static void ntx_entry_data(int idx, float* dst, int n) {
    nt_tape* tp = nt_tape_get();
    if (!tp || idx < 0 || idx >= tp->count || !dst) return;
    nt_tensor* o = tp->entries[idx].output;
    if (!o) return;
    int m = (n < o->len) ? n : o->len;
    for (int i = 0; i < m; i++) dst[i] = o->data[i];
}
*/
import "C"
import "unsafe"

// ═══════════════════════════════════════════════════════════════════════════════
// CGO bridge to notorch — kairos's training path.
//
// AML stays the inference / field-physics language (cgo_aml.go); notorch is how
// the organism *learns* — fast C tape autograd, BLAS, optional CUDA. This bridge
// is training-only. Op semantics mirror notorch/examples/train_llama3_bpe.c.
//
// GPU mode (nt_set_gpu_mode / gpu_init) is compiled out of libnotorch without
// USE_CUDA, so it is NOT bound here — it lives in a `cuda`-tagged file.
// ═══════════════════════════════════════════════════════════════════════════════

// NtTensor is an opaque handle to a notorch nt_tensor.
type NtTensor = *C.nt_tensor

func NtTensorNew(length int) NtTensor       { return C.nt_tensor_new(C.int(length)) }
func NtTensorNew2D(rows, cols int) NtTensor { return C.nt_tensor_new2d(C.int(rows), C.int(cols)) }
func NtTensorFree(t NtTensor)               { C.nt_tensor_free(t) }

// NtTensorSet copies a Go float32 slice into the tensor (len ≤ tensor len).
func NtTensorSet(t NtTensor, data []float32) {
	if t == nil || len(data) == 0 {
		return
	}
	C.ntx_set(t, (*C.float)(unsafe.Pointer(&data[0])), C.int(len(data)))
}

// NtTensorGet copies the tensor's first n floats out into a fresh Go slice.
func NtTensorGet(t NtTensor, n int) []float32 {
	if t == nil || n <= 0 {
		return nil
	}
	out := make([]float32, n)
	C.ntx_get(t, (*C.float)(unsafe.Pointer(&out[0])), C.int(n))
	return out
}

// ── Tape lifecycle ──
// nt_tape_clear keeps Chuck m/v state (positional, keyed by registration order);
// nt_tape_destroy wipes it — call destroy+start only after a growth event.

func NtTapeStart()   { C.nt_tape_start() }
func NtTapeClear()   { C.nt_tape_clear() }
func NtTapeDestroy() { C.nt_tape_destroy() }

// NtTapeParam registers a trainable tensor, returns its tape index. The caller
// MUST register params in a byte-identical order every burst (an explicitly
// ordered slice — never a Go map range) so Chuck's positional m/v slots stay
// bound to the same tensor.
func NtTapeParam(t NtTensor) int { return int(C.nt_tape_param(t)) }
func NtTapeNoDecay(idx int)      { C.nt_tape_no_decay(C.int(idx)) }

// NtTapeInput records a non-trainable input tensor (tokens / targets) on the tape.
func NtTapeInput(t NtTensor) int {
	return int(C.nt_tape_record(t, C.NT_OP_NONE, -1, -1, 0))
}

func NtTapeBackward(lossIdx int)              { C.nt_tape_backward(C.int(lossIdx)) }
func NtTapeClipGrads(maxNorm float64) float64 { return float64(C.nt_tape_clip_grads(C.float(maxNorm))) }
func NtTapeChuckStep(lr, lossVal float64)     { C.nt_tape_chuck_step(C.float(lr), C.float(lossVal)) }

// NtEntryScalar reads output->data[0] of a tape entry (e.g. the loss).
func NtEntryScalar(idx int) float64 { return float64(C.ntx_entry_scalar(C.int(idx))) }

// ntEntryData copies a tape entry's full output tensor into a Go slice (first n
// floats). Used to compare an op's output against a Go reimplementation.
func ntEntryData(idx, n int) []float32 {
	if n <= 0 {
		return nil
	}
	out := make([]float32, n)
	C.ntx_entry_data(C.int(idx), (*C.float)(unsafe.Pointer(&out[0])), C.int(n))
	return out
}

// ── NaN/Inf guard ──

type ntNanGuard struct{ g C.nt_nan_guard }

func NewNTNanGuard() ntNanGuard { return ntNanGuard{g: C.nt_nan_guard_new()} }

// check returns true if grads are clean, false if NaN/Inf was detected (grads zeroed).
func (n *ntNanGuard) Check() bool { return C.nt_nan_guard_check(&n.g) != 0 }

// ── Forward ops — each records on the tape and returns a tape entry index ──

func NtSeqEmbedding(wteIdx, wpeIdx, tokensIdx, T, D int) int {
	return int(C.nt_seq_embedding(C.int(wteIdx), C.int(wpeIdx), C.int(tokensIdx), C.int(T), C.int(D)))
}
func NtRope(xIdx, T, headDim int) int {
	return int(C.nt_rope(C.int(xIdx), C.int(T), C.int(headDim)))
}
func NtSeqRMSNorm(xIdx, gammaIdx, T, D int) int {
	return int(C.nt_seq_rmsnorm(C.int(xIdx), C.int(gammaIdx), C.int(T), C.int(D)))
}
func NtSeqLinear(wIdx, xIdx, T int) int {
	return int(C.nt_seq_linear(C.int(wIdx), C.int(xIdx), C.int(T)))
}
func NtMHCausalAttention(qIdx, kIdx, vIdx, T, headDim int) int {
	return int(C.nt_mh_causal_attention(C.int(qIdx), C.int(kIdx), C.int(vIdx), C.int(T), C.int(headDim)))
}
func NtAdd(aIdx, bIdx int) int { return int(C.nt_add(C.int(aIdx), C.int(bIdx))) }
func NtMul(aIdx, bIdx int) int { return int(C.nt_mul(C.int(aIdx), C.int(bIdx))) }
func NtSilu(xIdx int) int      { return int(C.nt_silu(C.int(xIdx))) }
func NtSeqCrossEntropy(logitsIdx, targetsIdx, T, V int) int {
	return int(C.nt_seq_cross_entropy(C.int(logitsIdx), C.int(targetsIdx), C.int(T), C.int(V)))
}

// ── Mode ──

func ntTrainMode(on bool) {
	if on {
		C.nt_train_mode(1)
	} else {
		C.nt_train_mode(0)
	}
}

func ntSeed(s uint64) { C.nt_seed(C.uint64_t(s)) }

// ── Increment 2: low-rank RRPRAM (Resonance form, op 33) ──
// kairos's never-trained position-bias w_pattern is replaced by notorch's
// proven low-rank attention. Reference: notorch examples/train_resonance_lora.c.

// NtRrpramLowrankAttention — op 33, packed low-rank RRPRAM attention.
// wrCombined packs Wr_a[H,E,R] then Wr_b[H,R,T_r] in one tensor (T_r == T);
// rank is derived as len/(H·(E+T_r)). xIdx is the post-RMSNorm hidden (full E),
// vIdx the same value tensor the content head uses. Returns the post-softmax
// attention-weighted V, shape [T × (nrHeads·headDim)] — same layout as content.
func NtRrpramLowrankAttention(wrCombinedIdx, xIdx, vIdx, T, nEmbd, nrHeads, headDim int) int {
	return int(C.nt_rrpram_lowrank_attention(C.int(wrCombinedIdx), C.int(xIdx), C.int(vIdx),
		C.int(T), C.int(nEmbd), C.int(nrHeads), C.int(headDim)))
}

// NtTapeParamFrozen registers a tensor as a FROZEN tape param: it takes part in
// the forward and gradient flows through it, but the optimizer step skips it.
// Used for the precomputed per-head gate vectors (g_sig / g_one) — the gate is
// frozen this increment, which keeps sigmoid/scale_by_t off the tape and so
// sidesteps the notorch GPU-sync bug class on those ops.
func NtTapeParamFrozen(t NtTensor) int { return int(C.nt_tape_param_frozen(t)) }
