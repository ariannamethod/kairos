package main

import "math"

// ═══════════════════════════════════════════════════════════════════════════════
// MetaWeight embedding seeding — postgpt's «tokenizer IS training» trick.
//
// Ported from github.com/ariannamethod/postgpt postgpt.c:545-570. The load-bearing mechanism
// that lets postgpt produce coherent text at zero training: token embeddings
// are biased by Hebbian co-occurrence before the first forward pass, and the
// LM head is biased by unigram × wte. Carries corpus structure into the model
// without gradient descent.
//
// In kairos this runs once after `field.BuildFromCorpus(tok, docs)` and
// before any warmup training. Idempotent within a single call site; if invoked
// multiple times it stacks (so callers should guard or scale `seedScale`
// appropriately).
// ═══════════════════════════════════════════════════════════════════════════════

// SeedEmbeddingsFromMetaweights biases the model's `wte` and `lm_head` rows
// from the populated CooccurField. Mirror of postgpt.c:545-570.
//
//   - For each token `a`: signal = Σ_{b in bigram[a]} bigram_prob(a→b) * wte[b].
//     wte[a] += seedScale * signal / neighbours.
//   - For each token t: lm_head[t] += seedScale * unigram_prob(t) * wte[t].
//
// `seedScale` defaults to 0.1 (postgpt value). Pass 0 to disable.
func SeedEmbeddingsFromMetaweights(model *GPT, field *CooccurField, seedScale float64) {
	if model == nil || field == nil || seedScale == 0 {
		return
	}
	wte := model.Base["wte"]
	if wte == nil {
		return
	}
	V := wte.Nout
	D := wte.Nin
	if V <= 0 || D <= 0 {
		return
	}

	field.mu.RLock()
	defer field.mu.RUnlock()

	// --- Hebbian → wte seeding (postgpt.c:545-562).
	// signal[d] = Σ_b bigramProb(a→b) * wte[b][d].
	signal := make([]float64, D)
	for a := 0; a < V; a++ {
		ctx, ok := field.BigramByFirst[a]
		if !ok || len(ctx) == 0 {
			continue
		}
		var total float64
		for _, v := range ctx {
			total += v
		}
		if total <= 0 {
			continue
		}
		// Reset signal accumulator.
		for d := 0; d < D; d++ {
			signal[d] = 0
		}
		neighbours := 0
		for b, cnt := range ctx {
			if b < 0 || b >= V {
				continue
			}
			prob := cnt / total
			if prob < 0.01 {
				continue // postgpt threshold (postgpt.c:550)
			}
			row := wte.Rows[b].Data
			for d := 0; d < D && d < len(row); d++ {
				signal[d] += prob * row[d]
			}
			neighbours++
		}
		if neighbours == 0 {
			continue
		}
		aRow := wte.Rows[a].Data
		inv := seedScale / float64(neighbours)
		for d := 0; d < D && d < len(aRow); d++ {
			aRow[d] += inv * signal[d]
		}
	}

	// --- Unigram → lm_head seeding (postgpt.c:566-570).
	// Compute unigram probabilities first.
	var uniTotal float64
	for _, v := range field.Unigram {
		uniTotal += v
	}
	if uniTotal <= 0 {
		return
	}
	lmHead := model.Base["lm_head"]
	if lmHead == nil {
		// Some kairos configs tie lm_head to wte; if it's not a separate
		// matrix, the wte seeding above already carries unigram structure
		// indirectly via Hebbian signal. Quiet no-op.
		return
	}
	if lmHead.Nout != V || lmHead.Nin != D {
		return
	}
	for t := 0; t < V; t++ {
		cnt, ok := field.Unigram[t]
		if !ok || cnt <= 0 {
			continue
		}
		uniProb := cnt / uniTotal
		if uniProb <= 0 || math.IsNaN(uniProb) {
			continue
		}
		wteRow := wte.Rows[t].Data
		lmRow := lmHead.Rows[t].Data
		scale := seedScale * uniProb
		for d := 0; d < D && d < len(wteRow) && d < len(lmRow); d++ {
			lmRow[d] += scale * wteRow[d]
		}
	}
}
