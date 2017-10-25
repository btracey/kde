// package kde implements useful routines for performing kernel density estimation.
package kde

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/stat/distuv"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/gonum/stat/distmv"
)

var sizeMismatch string = "kde: input size mismatch"

// Silverman approximates the kernel bandwidth for the set of samples using
// Silverman's rule of thumb. See https://en.wikipedia.org/wiki/Multivariate_kernel_density_estimation#Rule_of_thumb
// for more information.
//
// Note that there is no correction for the value of the weights in computing
// the effective number of samples.
func Silverman(x mat.Matrix, weights []float64) *mat.SymBandDense {
	n, d := x.Dims()
	if weights != nil && len(weights) != n {
		panic(sizeMismatch)
	}
	out := mat.NewDiagonal(d, nil)
	col := make([]float64, n)
	nf, df := float64(n), float64(d)
	for j := 0; j < d; j++ {
		mat.Col(col, j, x)
		std := stat.StdDev(col, weights)
		hii := math.Pow(4/(df+2), 1/(df+4)) * math.Pow(nf, -1/(df+4)) * std
		hii *= hii
		out.SetSymBand(j, j, hii)
	}
	return out
}

// Scott approximates the kernel bandwidth for the set of samples using
// Scott's rule of thumb. See https://en.wikipedia.org/wiki/Multivariate_kernel_density_estimation#Rule_of_thumb
// for more information.
//
// Note that there is no correction for the value of the weights in computing
// the effective number of samples.
func Scott(x mat.Matrix, weights []float64) *mat.SymBandDense {
	n, d := x.Dims()
	if weights != nil && len(weights) != n {
		panic(sizeMismatch)
	}
	out := mat.NewDiagonal(d, nil)
	col := make([]float64, n)
	nf, df := float64(n), float64(d)
	for j := 0; j < d; j++ {
		mat.Col(col, j, x)
		std := stat.StdDev(col, weights)
		hii := math.Pow(nf, -1/(df+4)) * std
		hii *= hii
		out.SetSymBand(j, j, hii)
	}
	return out
}

// Gaussian represents a KDE where each component has the same covariance matrix
// (represented by the Cholesky decomposition) and each center is a row of x.
type Gaussian struct {
	X       mat.Matrix
	Chol    *mat.Cholesky
	Weights []float64
	Src     *rand.Rand
}

// LogProb computes the log of the probability for the location x.
func (gauss Gaussian) LogProb(x []float64) float64 {
	n, d := gauss.X.Dims()
	if gauss.Chol.Size() != d {
		panic(sizeMismatch)
	}
	if gauss.Weights != nil && len(gauss.Weights) != n {
		panic(sizeMismatch)
	}
	lps := make([]float64, n)
	row := make([]float64, d)
	if gauss.Weights == nil {
		for i := 0; i < n; i++ {
			mat.Row(row, i, gauss.X)
			lps[i] = distmv.NormalLogProb(x, row, gauss.Chol)
		}
		lp := floats.LogSumExp(lps)
		return lp - math.Log(float64(n))
	}
	logSumWeights := math.Log(floats.Sum(gauss.Weights))
	for i := 0; i < n; i++ {
		mat.Row(row, i, gauss.X)
		lps[i] = distmv.NormalLogProb(x, row, gauss.Chol) + math.Log(gauss.Weights[i]) - logSumWeights
	}
	return floats.LogSumExp(lps)
}

// Rand generates a random number according to the distributon.
// If the input slice is nil, new memory is allocated, otherwise the result is stored
// in place.
func (gauss Gaussian) Rand(x []float64) []float64 {
	n, d := gauss.X.Dims()

	var idx int
	if gauss.Weights == nil {
		if gauss.Src == nil {
			idx = rand.Intn(n)
		} else {
			idx = gauss.Src.Intn(n)
		}
	} else {
		c := distuv.NewCategorical(gauss.Weights, gauss.Src)
		idx = int(c.Rand())
	}
	row := make([]float64, d)
	mat.Row(row, idx, gauss.X)
	return distmv.NormalRand(x, row, gauss.Chol, gauss.Src)
}

// EntropyUpper returns an upper bound on the mixture entropy, as computed by
//  H(X) <= d/2 - \sum_i w_i ln \sum_j w_j p_j(μ_i)
//  H(X) <= d/2 - \sum_i w_i ln p(μ_i)
// See section IV of
//  Estimating Mixture Entropy with Pairwise Distances
//  A. Kolchinsky and B. Tracey
// for more information.
func (gauss Gaussian) EntropyUpper() float64 {
	n, d := gauss.X.Dims()
	row := make([]float64, d)

	var entropy float64
	if gauss.Weights == nil {
		for i := 0; i < n; i++ {
			mat.Row(row, i, gauss.X)
			entropy -= gauss.LogProb(row)
		}
		entropy /= float64(n)
	} else {
		for i := 0; i < n; i++ {
			mat.Row(row, i, gauss.X)
			entropy -= gauss.Weights[i] * gauss.LogProb(row)
		}
		entropy /= floats.Sum(gauss.Weights)
	}
	entropy += float64(d) / 2
	return entropy
}