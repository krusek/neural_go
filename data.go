package neural_go

import (
	"encoding/binary"
	"os"
)

type ImageStream struct {
	file      *os.File
	count     int
	imageRows int
	imageCols int
}

type LabelStream struct {
	file  *os.File
	count int
}

func check(e error) {
	if e != nil {
		panic(e)
	}
}

func readInt(dat *os.File) uint32 {
	b1 := make([]byte, 4)
	_, err := dat.Read(b1)
	check(err)
	return binary.BigEndian.Uint32(b1)
}

func LoadLabelStream(name string) LabelStream {
	dat, err := os.Open(name)
	check(err)

	_ = readInt(dat)

	count := int(readInt(dat))

	return LabelStream{dat, count}
}

func (l LabelStream) ReadLabels() []byte {
	defer l.file.Close()
	labels := make([]byte, l.count)
	n, err := l.file.Read(labels)
	check(err)
	if n != l.count {
		panic("not enough stuff read")
	}
	return labels
}

func LoadImageStream(name string) ImageStream {
	dat, err := os.Open(name)
	check(err)

	_ = readInt(dat)

	count := int(readInt(dat))
	rows, cols := int(readInt(dat)), int(readInt(dat))
	return ImageStream{dat, count, rows, cols}
}

func (i ImageStream) ReadImage() []byte {
	image := make([]byte, i.imageRows*i.imageCols)
	_, err := i.file.Read(image)
	check(err)
	return image
}

func (i ImageStream) ReadImages() [][]byte {
	defer i.file.Close()
	images := make([][]byte, i.count)
	for ix := 0; ix < i.count; ix++ {
		images[ix] = i.ReadImage()
	}
	return images
}
