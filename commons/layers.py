#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os, sys
sys.path.append(os.pardir)
from commons.functions import *
import numpy as np

class ReLU():
	def __init__(self):
		self.mask = None

	def forward(self, x):
		self.mask = (x <= 0)
		out = x.copy()
		out[self.mask] = 0.0
		return out

	def backward(self, dout):
		dx = dout.copy()
		dx[self.mask] = 0.0
		return dx

class Sigmoid():
	def __init__(self):
		self.out = 0.0

	def forward(self, x):
		out = 1.0 / (1.0 + np.exp(-x))
		self.out = out
		return self.out

	def backward(self, dout):
		dx = dout * (1.0 - self.out) * self.out
		return dx

class Affine:
	def __init__(self, W, b):
		self.W = W
		self.b = b
		self.x = None
		self.dW = None
		self.db = None
		self.o = None
		self.dout = None

	def forward(self, x):
		self.x = x.copy()
		out = np.dot(x, self.W) + self.b
		self.o = out
		return out

	def backward(self, dout):
		self.dout = dout.copy()
		dx = np.dot(dout, self.W.T)
		self.dW = np.dot(self.x.T, dout)
		self.db = np.sum(dout, axis = 0)
		return dx

class SoftmaxWithLoss:
	def __init__(self):
		self.loss = None
		self.y = None	# output of the softmax
		self.t = None	# label

	def forward(self, x, t):
		self.t = t
		self.y = softmax(x)
		self.loss = cross_entropy_error(self.y, self.t)
		return self.loss

	def backward(self, dout = 1):
		batchsize = self.t.shape[0]
		dx = (self.y - self.t) / batchsize
		return dx
