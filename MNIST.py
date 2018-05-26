#!/usr/bin/perl3
# -*- coding: utf8 -*-

import numpy as np
from struct import unpack

def loader_image(imagefile, normalize, shape):
	with open(imagefile, "rb") as f:
		f.read(4)	# ignore magic number
		N, W, H = unpack(">III", f.read(4 * 3))	# total images
		M = W * H
		image = []
		dty = ">" + str(M) + "B"
		for i in range(0, N):
			chunk = unpack(dty, f.read(M))
			image.append(chunk)
		data = np.array(image)
		if normalize == True:
			data = data / 255.0
		if shape == True:
			data = np.reshape(data, (W, H, N))
		return data

def expand_digit(x, c):
	image = []
	for i in range(c):
		if x == i:
			image.append(1)
		else:
			image.append(0)
	return image

def loader_label(labelfile, decode = False):
	with open(labelfile, "rb") as f:
		f.read(4)	# ignore magic number
		N = unpack(">I", f.read(4))	# total labels
		N = N[0]
		dty = ">" + str(N) + "B"
		chunk = unpack(dty, f.read(N))
		data = np.array(chunk)
		if decode == False:
			return data
		image = []
		for i in range(0, N):
			image.append(expand_digit(chunk[i], 10))
		return np.array(image)

def loader_training(normalize, shape, decode):
	imagefile = "MNIST/" + "train-images-idx3-ubyte"
	labelfile = "MNIST/" + "train-labels-idx1-ubyte"
	return (loader_image(imagefile, normalize, shape), loader_label(labelfile, decode))

def loader_evaluate(normalize, shape, decode):
	imagefile = "MNIST/" + "t10k-images-idx3-ubyte"
	labelfile = "MNIST/" + "t10k-labels-idx1-ubyte"
	return (loader_image(imagefile, normalize, shape), loader_label(labelfile, decode))

def loader(normalize = False, shape = False, decode = False):
	return loader_training(normalize, shape, decode), loader_evaluate(normalize, shape, decode)

def test_main():
	(x_train, t_train) = loader_training(True, False, True)

if __name__ == "__main__":
	test_main()

