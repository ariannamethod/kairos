//go:build !cuda

// CPU-build linkage for the notorch bridge — links the plain libnotorch.a
// (CPU + BLAS, no CUDA). The cuda build links libnotorch_gpu.a instead
// (cgo_notorch_cuda.go). Split out of cgo_notorch.go so the LDFLAGS differ
// by build tag without touching the shared bridge code.

package gotools

/*
#cgo linux LDFLAGS: -L${SRCDIR}/../ariannamethod -lkairos -Wl,-rpath,${SRCDIR}/../ariannamethod -L/usr/lib/x86_64-linux-gnu/openblas-pthread/ -lopenblas -lm
*/
import "C"
