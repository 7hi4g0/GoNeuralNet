package neural

import (
	"fmt"
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
	m, _ := X.Dims()

	previous := X

	for _, theta := range net.theta {
		rows, _ := previous.Dims()

		data := make([]float64, rows)
		for i := range data {
			data[i] = 1
		}

		var input mat.Dense
		input.Augment(mat.NewVecDense(rows, data), previous)

		xi, yi := input.Dims()
		xt, yt := theta.Dims()

		fmt.Println("Input", xi, yi, "Theta", xt, yt)

		var output mat.Dense
		output.Mul(&input, theta.T())

		output.Apply(sigmoid, &output)

		previous = &output
	}

	hTheta := previous

	// (-y'*log(hTheta) - (1 - y)'*log(1 - hTheta)) / m
	var yTemp mat.Dense
	var logTemp mat.Dense
	var costTemp mat.Dense

	yTemp.Scale(-1, y)
	logTemp.Apply(log, hTheta)

	costTemp.Mul(yTemp.T(), &logTemp)

	cost := mat.Sum(&costTemp)

	yTemp.Apply(add1, &yTemp)
	logTemp.Scale(-1, hTheta)
	logTemp.Apply(add1, hTheta)

	costTemp.Mul(yTemp.T(), &logTemp)

	cost = (cost - mat.Sum(&costTemp)) / float64(m)

	return cost, nil
}

func ConvertLabels(y []uint8) *mat.Dense {
	dataLen := len(y)
	max := y[0]

	for _, e := range y {
		if e > max {
			max = e
		}
	}

	yMat := mat.NewDense(dataLen, int(max)+1, nil)

	for idx, label := range y {
		yMat.Set(idx, int(label), 1)
	}

	return yMat
}

func sigmoid(i, j int, v float64) float64 {
	return 1 / (1 + math.Exp(-v))
}

func log(i, j int, v float64) float64 {
	return math.Log(v)
}

func add1(i, j int, v float64) float64 {
	return v + 1
}
