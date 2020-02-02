package neural_go

import (
	"bytes"
	"strings"
	"testing"

	"gotest.tools/assert"
)

func TestReadImageData(t *testing.T) {
	stream := LoadImageStream("data/t10k-images-idx3-ubyte")
	image := stream.ReadImage()

	var b bytes.Buffer

	rows, cols := stream.imageRows, stream.imageCols
	for ix := 0; ix < rows; ix++ {
		for iy := 0; iy < cols; iy++ {
			if image[ix*cols+iy] > 0 {
				b.WriteString("1")
			} else {
				b.WriteString("0")
			}
		}
		b.WriteString("\n")
	}

	expectedLines := []string{
		"0000000000000000000000000000",
		"0000000000000000000000000000",
		"0000000000000000000000000000",
		"0000000000000000000000000000",
		"0000000000000000000000000000",
		"0000000000000000000000000000",
		"0000000000000000000000000000",
		"0000001111110000000000000000",
		"0000001111111111111111000000",
		"0000001111111111111111000000",
		"0000000000011111111111000000",
		"0000000000000000001111000000",
		"0000000000000000011110000000",
		"0000000000000000011110000000",
		"0000000000000000111100000000",
		"0000000000000000111100000000",
		"0000000000000001111000000000",
		"0000000000000001110000000000",
		"0000000000000011110000000000",
		"0000000000000111100000000000",
		"0000000000001111100000000000",
		"0000000000001111000000000000",
		"0000000000011111000000000000",
		"0000000000011110000000000000",
		"0000000000111110000000000000",
		"0000000000111110000000000000",
		"0000000000111100000000000000",
		"0000000000000000000000000000",
		"",
	}
	expected := strings.Join(expectedLines, "\n")
	assert.Equal(t, expected, b.String())
}

func TestReadImagesData(t *testing.T) {
	stream := LoadImageStream("data/t10k-images-idx3-ubyte")
	images := stream.ReadImages()

	assert.Equal(t, len(images), stream.count)

	for _, image := range images {
		assert.Equal(t, stream.imageCols*stream.imageRows, len(image))
	}
}

func TestReadLabel(t *testing.T) {
	stream := LoadLabelStream("data/t10k-labels-idx1-ubyte")
	labels := stream.ReadLabels()
	assert.Equal(t, stream.count, 10000)
	assert.Equal(t, int(labels[0]), 7)
}
