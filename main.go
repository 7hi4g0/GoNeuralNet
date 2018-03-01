package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"

	"./data"
	"./neural"
)

var (
	TrainImages []*mat.Dense
	TrainLabels []uint8

	TestImages []*mat.Dense
	TestLabels []uint8
)

func main() {
	fmt.Println("Loading Training data...")
	TrainImages, _ = data.LoadIDXData("data/train-images-idx3-ubyte")
	_, TrainLabels = data.LoadIDXData("data/train-labels-idx1-ubyte")
	fmt.Println("Loaded Training data")

	fmt.Println("Loading Test data...")
	TestImages, _ = data.LoadIDXData("data/t10k-images-idx3-ubyte")
	_, TestLabels = data.LoadIDXData("data/t10k-labels-idx1-ubyte")
	fmt.Println("Loaded Test data")

	nn := neural.NewClassificationNetwork(784, []uint{25}, 10)
	nn.SetAlpha(1)
	nn.SetLambda(1)

	rows, columns := TrainImages[0].Dims()

	columns = rows * columns
	rows = len(TrainImages)

	UnrolledTrainImages := mat.NewDense(rows, columns, nil)

	for idx, image := range TrainImages {
		UnrolledTrainImages.SetRow(idx, image.RawMatrix().Data)
	}

	nn.Train(UnrolledTrainImages, neural.ConvertLabels(TrainLabels), 100)

	rows, columns = TestImages[0].Dims()

	columns = rows * columns
	rows = len(TestImages)

	UnrolledTestImages := mat.NewDense(rows, columns, nil)

	for idx, image := range TestImages {
		UnrolledTestImages.SetRow(idx, image.RawMatrix().Data)
	}

	accuracy := nn.Accuracy(UnrolledTestImages, TestLabels)

	fmt.Println("Accuracy", accuracy)
}
