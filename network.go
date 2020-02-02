package neural_go

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type Network struct {
	sizes   []int
	biases  []*mat.Dense
	weights []*mat.Dense
	sigmoid func(v float64) float64
}

var defaultSigmoid = func(v float64) float64 {
	return 1.0 / (1.0 + math.Exp(v))
}

func generateMatrix(r, c int, generator func() float64) *mat.Dense {
	data := make([]float64, r*c)
	for ic := 0; ic < c; ic++ {
		for ir := 0; ir < r; ir++ {
			data[ic+c*ir] = generator()
		}
	}
	m := mat.NewDense(r, c, data)
	return m
}

func randomMatrix(r, c int) *mat.Dense {
	rr := rand.New(rand.NewSource(99))
	return generateMatrix(r, c, rr.Float64)
}

func GeneratedNetwork(sizes []int, weightGenerator, biasGenerator func() float64, sigmoid func(v float64) float64) Network {
	biases := make([]*mat.Dense, len(sizes)-1)
	weights := make([]*mat.Dense, len(sizes)-1)

	for ix, _ := range sizes {
		if ix < len(sizes)-1 {
			weights[ix] = generateMatrix(sizes[ix+1], sizes[ix], weightGenerator)
		}
		if ix > 0 {
			fmt.Println(ix)
			biases[ix-1] = generateMatrix(sizes[ix], 1, biasGenerator)
		}
	}

	return Network{sizes, biases, weights, sigmoid}

}

func RandomNetwork(sizes []int) Network {
	rr := rand.New(rand.NewSource(99))
	return GeneratedNetwork(sizes, rr.Float64, rr.Float64, defaultSigmoid)
}

func (n *Network) maxSize() int {
	m := 0
	for _, s := range n.sizes {
		if s > m {
			m = s
		}
	}
	return m
}

func (n *Network) FeedForward(a *mat.Dense) *mat.Dense {
	max := n.maxSize()
	// Larger matrix to hold products of weights and a's
	var m mat.Dense
	m.ReuseAs(max, max)

	// Larger vector to hold a's
	var aa mat.Dense
	aa.ReuseAs(max, 1)
	aaa := a

	sig := func(_, _ int, v float64) float64 {
		return n.sigmoid(v)
	}
	for ix, weight := range n.weights {
		ar, _ := weight.Dims()
		_, bc := aaa.Dims()
		resizeMatrix(&m, ar, bc)
		m.Mul(weight, aaa) // weight * a

		resizeMatrix(&aa, ar, 1)
		bias := n.biases[ix]
		aa.Add(&m, bias) // (weight * a) + bias

		aa.Apply(sig, &aa) // sigmoid((weight * a) + bias)
		aaa = &aa
	}
	return &aa
}

func resizeMatrix(m *mat.Dense, r, c int) {
	raw := m.RawMatrix()
	raw.Cols = c
	raw.Rows = r
	m.SetRawMatrix(raw)
}
