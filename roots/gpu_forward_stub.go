//go:build !linux || !cuda

package main

// ═══════════════════════════════════════════════════════════════════════════════
// GPU forward stubs — non-linux builds. CUDA toolchain is unavailable, so
// MatvecGPU returns nil (signalling the Matvec dispatcher to fall back to the
// existing CPU/BLAS path) and gpuRefreshWeights is a no-op. The dispatcher in
// Matvec already gates on `CFG.UseGPU && gotools.GpuReady()` and gotools.GpuReady() returns
// false on this build (see gpu_bindings_stub.go), so MatvecGPU is never
// actually reached at runtime — this body exists purely so the rest of
// kairos compiles cleanly on macOS / windows / etc.
// ═══════════════════════════════════════════════════════════════════════════════

func (m *MatrixParam) MatvecGPU(x *Vec) *Vec { return nil }

func gpuRefreshWeights(gpt *GPT) {}
