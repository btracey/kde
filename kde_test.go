package kde

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/gonum/stat/distmv"
)

func TestSilvermanScott(t *testing.T) {
	src := rand.New(rand.NewSource(1))
	// Generate uniform random numbers.
	d := 4
	n := 1000
	x := mat.NewDense(n, d, nil)
	dist := distmv.NewUnitUniform(d, src)
	for i := 0; i < n; i++ {
		dist.Rand(x.RawRowView(i))
	}

	// Compute Silverman and Scott method.
	silverman := Silverman(x, nil)
	scott := Scott(x, nil)

	if !scottSilvermanConsistent(scott, silverman, n) {
		t.Errorf("scott and silverman not consistent")
	}

	if !scottMatches(scott, x, nil) {
		t.Errorf("scott doesn't match computed variance")
	}

	// Generate some random weights and test again.
	weights := make([]float64, n)
	for i := 0; i < n; i++ {
		weights[i] = src.Float64()
	}
	silverman = Silverman(x, weights)
	scott = Scott(x, weights)

	if !scottSilvermanConsistent(scott, silverman, n) {
		t.Errorf("scott and silverman not consistent weighted")
	}

	if !scottMatches(scott, x, weights) {
		t.Errorf("scott doesn't match computed variance weighted")
	}
}

func scottMatches(scott *mat.SymBandDense, xs *mat.Dense, weights []float64) bool {
	n, d := xs.Dims()
	// Scott should be variance * n^(-1/(d+4)).
	for j := 0; j < d; j++ {
		col := mat.Col(nil, j, xs)
		std := stat.StdDev(col, weights)
		v := std * math.Pow(float64(n), -1.0/(float64(d)+4))
		v *= v
		if math.Abs(scott.At(j, j)-v) > 1e-14 {
			fmt.Println(v, scott.At(j, j))
			return false
		}
	}
	return true
}

func scottSilvermanConsistent(scott, silverman *mat.SymBandDense, n int) bool {
	d := scott.Symmetric()
	if d != silverman.Symmetric() {
		return false
	}
	// The two should differ by a factor of  (4/(d+2))^(1/(d+4)) squared
	diff := 4 / (float64(d) + 2)
	diff = math.Pow(diff, 1/(float64(d)+4))
	for i := 0; i < d; i++ {
		if math.Abs(diff*diff*scott.At(i, i)-silverman.At(i, i)) > 1e-14 {
			return false
		}
	}
	return true
}

func TestGaussianRandLogProb(t *testing.T) {
	// Test LogProb using importance sampling
	// int p dx = 1, so int p/q q dx = 1
	src := rand.New(rand.NewSource(1))
	d := 4
	n := 10
	x := mat.NewDense(n, d, nil)
	cov := mat.NewDiagonal(d, nil)
	for i := 0; i < d; i++ {
		cov.SetSymBand(i, i, 1)
	}
	dist, _ := distmv.NewNormal(make([]float64, d), cov, src)
	for i := 0; i < n; i++ {
		dist.Rand(x.RawRowView(i))
	}

	sym := Scott(x, nil)
	var chol mat.Cholesky
	chol.Factorize(sym)

	// Add weights
	weights := make([]float64, n)
	for i := range weights {
		weights[i] = src.Float64()
	}
	gauss := Gaussian{
		X:    x,
		Chol: &chol,
		Src:  src,
	}

	cov2 := mat.NewDiagonal(d, nil)
	for i := 0; i < d; i++ {
		cov2.SetSymBand(i, i, 4)
	}
	dist2, _ := distmv.NewNormal(make([]float64, d), cov2, src)
	for i := 0; i < n; i++ {
		dist.Rand(x.RawRowView(i))
	}

	cov3 := mat.NewDiagonal(d, nil)
	for i := 0; i < d; i++ {
		cov3.SetSymBand(i, i, 0.5)
	}
	dist3, _ := distmv.NewNormal(make([]float64, d), cov3, src)
	for i := 0; i < n; i++ {
		dist.Rand(x.RawRowView(i))
	}

	// Test weighted.
	gauss.Weights = weights
	// Test logProb by generating samples from dist2.
	if !matchImportanceSampling(gauss, dist2, 100000, d, 1.5e-2) {
		t.Errorf("importance sampling mismatch")
	}
	// Test Rand and LogProb by generating samples from gauss.
	if !matchImportanceSampling(dist3, gauss, 100000, d, 1.5e-2) {
		t.Errorf("importance sampling mismatch")
	}

	// Test unweighted.
	gauss.Weights = nil
	// Test LogProb by generating samples from dist2.
	if !matchImportanceSampling(gauss, dist2, 100000, d, 1.5e-2) {
		t.Errorf("importance sampling mismatch")
	}
	// Test Rand and LogProb by generating samples from gauss.
	if !matchImportanceSampling(dist3, gauss, 100000, d, 1.5e-2) {
		t.Errorf("importance sampling mismatch")
	}
}

// match importance sampling tests if p and q work with importance sampling.
//  \int p dx = 1
// thus
//  \int p/q q dx = 1
// so log of that should be 0.
func matchImportanceSampling(p distmv.LogProber, q distmv.RandLogProber, nSamples, dim int, tol float64) bool {
	xtest := mat.NewDense(nSamples, dim, nil)
	for i := 0; i < nSamples; i++ {
		q.Rand(xtest.RawRowView(i))
	}

	// The expectation of p/q should be 1
	lps := make([]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		lp := p.LogProb(xtest.RawRowView(i))
		lq := q.LogProb(xtest.RawRowView(i))
		lps[i] = lp - lq
	}
	logmc := floats.LogSumExp(lps) - math.Log(float64(nSamples))
	if math.Abs(logmc) > tol {
		return false
	}
	return true
}
