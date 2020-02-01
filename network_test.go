package neural_go

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestNeuralNetworkSizes(t *testing.T) {
	biasGenerator := func() float64 {
		return 1.0
	}
	weightGenerator := func() float64 {
		return 2.0
	}
	sig := func(v float64) float64 {
		return v + 1.0
	}
	network := GeneratedNetwork([]int{2, 3, 4}, weightGenerator, biasGenerator, sig)
	weight := network.weights[1]
	bias := network.biases[0]

	fmt.Println(weight, bias)
	var receiver mat.Dense
	receiver.Mul(weight, bias)
	resultGenerator := func() float64 {
		return 6.0
	}
	mat.Equal(receiver.T(), generateMatrix(4, 1, resultGenerator))
	fmt.Println("product: ", receiver)
}

func TestNeuralNetworkFeedForward(t *testing.T) {
	biasGenerator := func() float64 {
		return 1.0
	}
	weightGenerator := func() float64 {
		return 2.0
	}
	sig := func(v float64) float64 {
		return v + 1.0
	}
	aGenerator := func() float64 {
		return 1.0
	}
	network := GeneratedNetwork([]int{2, 3, 4}, weightGenerator, biasGenerator, sig)
	a := generateMatrix(2, 1, aGenerator)

	aa := network.FeedForward(a)
	resultGenerator := func() float64 {
		return 38.0
	}
	mat.Equal(aa, generateMatrix(4, 1, resultGenerator))
}
