//go:build !linux || !cuda

package gotools

import "unsafe"

// ═══════════════════════════════════════════════════════════════════════════════
// GPU stubs — non-linux builds (darwin / windows / etc).
//
// CUDA toolchain is only wired through cgo_aml.go on Linux. Other platforms
// get pure-Go stubs so the rest of kairos compiles unchanged. GpuReady()
// always returns false → ForwardStep dispatcher routes to the existing CPU
// path everywhere. Signatures mirror gpu_bindings_linux.go exactly.
// ═══════════════════════════════════════════════════════════════════════════════

func GpuInit() int                                         { return -1 }
func gpuShutdown()                                         {}
func GpuReady() bool                                       { return false }
func gpuAlloc(n int) unsafe.Pointer                        { return nil }
func gpuFree(p unsafe.Pointer)                             {}
func GpuUpload(dst unsafe.Pointer, src []float32)          {}
func GpuDownload(dst []float32, src unsafe.Pointer, n int) {}
func gpuZero(p unsafe.Pointer, n int)                      {}
func GpuSgemmNT(M, N, K int, dA, dB, dC unsafe.Pointer)    {}
func gpuSgemmNN(M, N, K int, dA, dB, dC unsafe.Pointer)    {}
func gpuSgemmTN(M, N, K int, dA, dB, dC unsafe.Pointer)    {}
func gpuAdd(dOut, dA, dB unsafe.Pointer, n int)            {}
func gpuMul(dOut, dA, dB unsafe.Pointer, n int)            {}
func gpuSiLU(dOut, dIn unsafe.Pointer, n int)              {}
func gpuRMSNorm(dOut, dIn unsafe.Pointer, T, D int)        {}
func GpuCacheWeight(name string, h []float32) int          { return -1 }
func GpuGetWeight(name string) (unsafe.Pointer, int)       { return nil, 0 }
func gpuMarkAllDirty()                                     {}
func GpuScratch(slot, n int) unsafe.Pointer                { return nil }
func gpuMultiHeadAttention(dQ, dK, dV, dOut, dScores unsafe.Pointer, T, D, nHeads int) {
}
