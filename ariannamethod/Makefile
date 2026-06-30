# Kairos — vendored field-physics engine (AML + notorch)
# builds libkairos.so (Linux) or libkairos.dylib (macOS)
# with optional BLAS acceleration for notorch

UNAME := $(shell uname)

CC = gcc
CFLAGS = -O2 -fPIC -Wall

ifeq ($(UNAME), Darwin)
	# macOS: Apple Accelerate (zero deps, AMX/Neural Engine)
	TARGET = libkairos.dylib
	LDFLAGS = -shared -framework Accelerate
	CFLAGS += -DUSE_BLAS -DACCELERATE
else
	# Linux: OpenBLAS (apt install libopenblas-dev)
	TARGET = libkairos.so
	BLAS_CHECK := $(shell pkg-config --exists openblas 2>/dev/null && echo yes)
	ifeq ($(BLAS_CHECK), yes)
		LDFLAGS = -shared $(shell pkg-config --libs openblas)
		CFLAGS += -DUSE_BLAS $(shell pkg-config --cflags openblas)
	else
		# fallback: try -lopenblas directly
		LDFLAGS_BLAS := $(shell echo 'int main(){}' | gcc -xc - -lopenblas -o /dev/null 2>/dev/null && echo "-lopenblas")
		ifneq ($(LDFLAGS_BLAS),)
			LDFLAGS = -shared -lopenblas
			CFLAGS += -DUSE_BLAS
		else
			LDFLAGS = -shared -lm
		endif
	endif
endif

SRCS = ariannamethod.c notorch.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

ariannamethod.o: ariannamethod.c ariannamethod.h
	$(CC) $(CFLAGS) -c $< -o $@

notorch.o: notorch.c notorch.h
	$(CC) $(CFLAGS) -c $< -o $@

# Opt-in SIMD path: hand-rolled cblas_* shim in notorch_simd.h (AVX2 + FMA + pthread).
# Mutually exclusive with USE_BLAS. Replaces external BLAS dep with zero-deps AVX2
# kernels. **x86_64 only.** Apple Silicon / arm64 — no AVX2; use default `make`
# (Accelerate). For pure-CPU scalar debug fallback on ARM, see
# notorch_simd_scalar.h and override `SIMD_CFLAGS` manually.
# Usage: `make -f Makefile clean && make -f Makefile simd`
SIMD_ARCH := $(shell uname -m)
SIMD_CFLAGS = -O2 -fPIC -Wall -DUSE_SIMD -mavx2 -mfma
SIMD_LDFLAGS_DARWIN = -shared -lpthread
SIMD_LDFLAGS_LINUX  = -shared -lpthread -lm

simd:
	@if [ "$(SIMD_ARCH)" != "x86_64" ] && [ "$(SIMD_ARCH)" != "amd64" ]; then \
		echo "ERROR: 'make simd' requires x86_64 with AVX2 (Intel/Linux)."; \
		echo "       Detected arch: $(SIMD_ARCH)."; \
		echo "       On Apple Silicon / arm64 use default 'make' (Accelerate)."; \
		echo "       For scalar debug fallback override SIMD_CFLAGS manually."; \
		exit 1; \
	fi
ifeq ($(UNAME), Darwin)
	$(CC) $(SIMD_CFLAGS) -c ariannamethod.c -o ariannamethod.o
	$(CC) $(SIMD_CFLAGS) -c notorch.c        -o notorch.o
	$(CC) $(SIMD_CFLAGS) -o $(TARGET) ariannamethod.o notorch.o $(SIMD_LDFLAGS_DARWIN)
else
	$(CC) $(SIMD_CFLAGS) -c ariannamethod.c -o ariannamethod.o
	$(CC) $(SIMD_CFLAGS) -c notorch.c        -o notorch.o
	$(CC) $(SIMD_CFLAGS) -o $(TARGET) ariannamethod.o notorch.o $(SIMD_LDFLAGS_LINUX)
endif

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean simd
