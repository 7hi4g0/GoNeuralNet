package neural

import (
	"fmt"
	"log"
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

	epsilon := 0.12

	for i := range data {
		data[i] = (float64(rand.Int63())/math.MaxInt64)*2*epsilon - epsilon
	}

	return mat.NewDense(int(out), int(in), data)
}

func (net *Network) SetAlpha(alpha float64) {
	net.alpha = alpha
}

func (net *Network) SetLambda(lambda float64) {
	net.lambda = lambda
}

func (net *Network) Cost(X, y *mat.Dense) (float64, []*mat.Dense) {
	m, _ := X.Dims()

	thetaLen := len(net.theta)

	activation := make([]mat.Matrix, thetaLen+1)
	preActivation := make([]mat.Dense, thetaLen+1)

	activation[0] = X

	for layer, theta := range net.theta {
		input := prependBias(activation[layer])

		// xi, yi := input.Dims()
		// xt, yt := theta.Dims()

		// fmt.Println("Input", xi, yi, "Theta", xt, yt)

		var output mat.Dense
		output.Mul(input, theta.T())

		preActivation[layer+1].Clone(&output)

		output.Apply(sigmoid, &output)

		activation[layer] = input
		activation[layer+1] = &output
	}

	hTheta := activation[thetaLen]

	// (-y'*log(hTheta) - (1 - y)'*log(1 - hTheta)) / m
	var yTemp mat.Dense
	var logTemp mat.Dense
	var costTemp mat.Dense

	yTemp.Scale(-1, y)
	logTemp.Apply(logarithm, hTheta)

	costTemp.MulElem(&yTemp, &logTemp)

	cost := mat.Sum(&costTemp)

	yTemp.Apply(add1, &yTemp)
	logTemp.Scale(-1, hTheta)
	logTemp.Apply(add1, &logTemp)
	logTemp.Apply(logarithm, &logTemp)

	costTemp.MulElem(&yTemp, &logTemp)

	cost = (cost - mat.Sum(&costTemp)) / float64(m)

	delta := make([]mat.Dense, thetaLen+1)
	grad := make([]*mat.Dense, thetaLen)

	delta[thetaLen].Sub(hTheta, y)

	grad[thetaLen-1] = mat.NewDense(0, 0, nil)
	grad[thetaLen-1].Mul(delta[thetaLen].T(), activation[thetaLen-1])
	grad[thetaLen-1].Scale(1/float64(m), grad[thetaLen-1])

	for layer := thetaLen - 1; layer > 0; layer-- {
		theta := net.theta[layer]
		dNext := delta[layer+1]

		var dTemp mat.Dense
		dTemp.Mul(&dNext, theta)

		var sigGrad mat.Dense
		sigGrad.Apply(sigmoidGradient, &preActivation[layer])

		rows, cols := dTemp.Dims()

		// printDims(dTemp.Slice(0, rows, 1, cols), &sigGrad)

		delta[layer].MulElem(dTemp.Slice(0, rows, 1, cols), &sigGrad)

		// printDims(delta[layer].T(), activation[layer-1])

		grad[layer-1] = mat.NewDense(0, 0, nil)
		grad[layer-1].Mul(delta[layer].T(), activation[layer-1])
		grad[layer-1].Scale(1/float64(m), grad[layer-1])
	}

	return cost, grad
}

func (net *Network) Train(X, y *mat.Dense, cycles int) {
	for cycle := 0; cycle < cycles; cycle++ {
		cost, grad := net.Cost(X, y)

		log.Println("Iter", cycle, "Cost", cost)

		for idx, theta := range net.theta {
			g := grad[idx]

			g.Scale(net.alpha, g)

			// printDims(theta, g)

			theta.Sub(theta, g)
		}
	}

	cost, _ := net.Cost(X, y)

	log.Println("Iter", cycles, "Cost", cost)
}

func (net *Network) Predict(X, y *mat.Dense) mat.Matrix {
	thetaLen := len(net.theta)

	activation := make([]mat.Matrix, thetaLen+1)
	preActivation := make([]mat.Dense, thetaLen+1)

	activation[0] = X

	for layer, theta := range net.theta {
		input := prependBias(activation[layer])

		// xi, yi := input.Dims()
		// xt, yt := theta.Dims()

		// fmt.Println("Input", xi, yi, "Theta", xt, yt)

		var output mat.Dense
		output.Mul(input, theta.T())

		preActivation[layer+1].Clone(&output)

		output.Apply(sigmoid, &output)

		activation[layer] = input
		activation[layer+1] = &output
	}

	return activation[thetaLen]
}

func (net *Network) Accuracy(X *mat.Dense, y []uint8) float64 {
	prediction := net.Predict(X, ConvertLabels(y))

	rows, cols := prediction.Dims()

	predictionLabels := make([]uint8, rows)

	for row := range predictionLabels {
		max := prediction.At(row, 0)
		maxLabel := 0
		for col := 1; col < cols; col++ {
			value := prediction.At(row, col)

			if value > max {
				max = value
				maxLabel = col
			}
		}

		predictionLabels[row] = uint8(maxLabel)
	}

	tests := len(y)
	var accuracy float64

	for test, label := range y {
		if label == predictionLabels[test] {
			accuracy++
		}
	}

	return accuracy * 100 / float64(tests)
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

func prependBias(input mat.Matrix) mat.Matrix {
	rows, _ := input.Dims()

	data := make([]float64, rows)
	for i := range data {
		data[i] = 1
	}

	var activation mat.Dense
	activation.Augment(mat.NewVecDense(rows, data), input)

	return &activation
}

func printDims(a, b mat.Matrix) {
	xa, ya := a.Dims()
	xb, yb := b.Dims()

	fmt.Println("a", xa, ya, "b", xb, yb)
}

func sigmoidGradient(i, j int, v float64) float64 {
	sig := sigmoid(i, j, v)
	return sig * (1 - sig)
}

func sigmoid(i, j int, v float64) float64 {
	return 1 / (1 + math.Exp(-v))
}

func logarithm(i, j int, v float64) float64 {
	return math.Log(v)
}

func add1(i, j int, v float64) float64 {
	return v + 1
}
