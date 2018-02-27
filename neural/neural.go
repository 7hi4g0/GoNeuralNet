package neural

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type NetworkType int

const (
	Classification = iota
)

type Network struct {
	netType NetworkType

	in     uint
	out    uint
	hidden []uint

	theta []*mat.Dense

	alpha  float64
	lambda float64
}

func NewClassificationNetwork(in uint, hidden []uint, out uint) *Network {
	newNet := &Network{
		in:      in,
		out:     out,
		hidden:  hidden,
		netType: Classification,
	}

	var theta []*mat.Dense

	previous := in

	for _, nodes := range hidden {
		theta = append(theta, initTheta(nodes, previous+1))

		previous = nodes
	}

	newNet.theta = append(theta, initTheta(out, previous+1))

	return newNet
}

func initTheta(out, in uint) *mat.Dense {
	total := out * in
	data := make([]float64, total)

	for i := range data {
		data[i] = rand.NormFloat64()
	}

	return mat.NewDense(int(out), int(in), data)
}

func (net *Network) SetAlpha(alpha float64) {
	net.alpha = alpha
}

func (net *Network) SetLambda(lambda float64) {
	net.lambda = lambda
}

func (net *Network) Cost(X, y *mat.Dense) (float64, []float64) {
	previous := X

	for _, theta := range net.theta {
		rows, _ := previous.Dims()

		data := make([]float64, rows)
		for i := range data {
			data[i] = 1
		}

		var input mat.Dense
		input.Augment(mat.NewVecDense(rows, data), previous)

		var output mat.Dense
		output.Mul(&input, theta.T())

		output.Apply(sigmoid, &output)

		previous = &output
	}

	return 0, nil
}

func sigmoid(i, j int, v float64) float64 {
	return 1 / (1 + math.Exp(-v))
}
