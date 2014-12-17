#!/usr/bin/env sh
# This script converts the mnist data into leveldb format.

EXAMPLES=../../examples/mnist
#EXAMPLES=./
DATA=../../data/mnist

echo "Creating leveldb..."

rm -rf mnist-train-leveldb
rm -rf mnist-test-leveldb

$EXAMPLES/convert_mnist_data.exe $DATA/train-images-idx3-ubyte $DATA/train-labels-idx1-ubyte mnist-train-leveldb
$EXAMPLES/convert_mnist_data.exe $DATA/t10k-images-idx3-ubyte $DATA/t10k-labels-idx1-ubyte mnist-test-leveldb

echo "Done."
