package data

import (
	"encoding/binary"
	"os"

	"gonum.org/v1/gonum/mat"
)

func LoadIDXData(filename string) ([]*mat.Dense, []uint8) {
	file, err := os.Open(filename)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	var magic uint32
	err = binary.Read(file, binary.BigEndian, &magic)
	if err != nil {
		panic(err)
	}

	switch magic {
	case 2051:
		var count uint32

		err = binary.Read(file, binary.BigEndian, &count)
		if err != nil {
			panic(err)
		}

		var rows, columns uint32
		err = binary.Read(file, binary.BigEndian, &rows)
		if err != nil {
			panic(err)
		}
		err = binary.Read(file, binary.BigEndian, &columns)
		if err != nil {
			panic(err)
		}

		dataSize := rows * columns
		matrices := make([]*mat.Dense, count)

		for matrix := uint32(0); matrix < count; matrix++ {
			data := make([]float64, dataSize)

			for dataPoint := uint32(0); dataPoint < dataSize; dataPoint++ {
				var pixel uint8
				err = binary.Read(file, binary.BigEndian, &pixel)
				if err != nil {
					panic(err)
				}

				data[dataPoint] = float64(pixel) / 255
			}

			matrices[matrix] = mat.NewDense(int(rows), int(columns), data)
		}

		return matrices, nil

	case 2049:
		var count uint32

		err = binary.Read(file, binary.BigEndian, &count)
		if err != nil {
			panic(err)
		}

		labels := make([]uint8, count)

		for lbl := 0; lbl < int(count); lbl++ {
			var label uint8

			err = binary.Read(file, binary.BigEndian, &label)
			if err != nil {
				panic(err)
			}

			labels[lbl] = label
		}

		return nil, labels
	default:
		panic("Unknown magic number")
	}
}
