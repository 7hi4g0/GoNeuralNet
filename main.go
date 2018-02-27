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

	nn := neural.NewClassificationNetwork(400, []uint{25}, 10)
	nn.SetAlpha(0.001)
	nn.SetLambda(1)

	nn.Train(TrainImages, TrainLabels, 50)

	accuracy := nn.Accuracy(TestImages, TestLabels)

	fmt.Println(accuracy)
}
