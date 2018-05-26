# -*- coding: utf-8 -*-
import numpy as np
from commons.functions import *
from commons.layers import *
from collections import OrderedDict

class MultiLayerNet():
	def __init__(self, netlist, logistic = True, params = None):
		self.params = []
		self.layer = []
		for i in range(len(netlist) - 1):
			if params != None:
				w = params[i][0]
				b = params[i][1]
			else:
				w = np.random.randn(netlist[i], netlist[i + 1])
				b = np.random.randn(netlist[i + 1])
			pair = []
			pair.append(w)
			pair.append(b)
			self.params.append(pair)
			self.layer.append(Affine(pair[0], pair[1]))
			if i != len(netlist) - 2:
				self.layer.append(Sigmoid())
		if logistic:
			self.lastLayer = SoftmaxWithLoss()
		else:
			self.lastLayer = Sigmoid()
		self.logistic = logistic
			
	def toLogistic(self):
		self.logistic = True
		self.lastLayer = SoftmaxWithLoss()

	def append(self, params):
		param_size = len(self.params)
		self.params.append(params)
		if param_size > 0:
			self.layer.append(Sigmoid())
		self.layer.append(Affine(params[0], params[1]))

	def predict(self, x):
		for layer in self.layer:
			x = layer.forward(x)
		return x

	def loss(self, x, t):
		# logistic == True の時のみ意味がある
		if self.logistic == False:
			return None

		x = self.predict(x)
		return self.lastLayer.forward(x, t)

	def accuracy(self, x, t):
		# logistic == True の時のみ意味がある
		if self.logistic == False:
			return None

		y = self.predict(x)
		y = np.argmax(y, axis = 1)
		tt = np.argmax(t, axis = 1)
		return np.sum(y == tt) / y.shape[0]

	def gradient(self, x, t):
		# logistic == True の時のみ意味がある
		if self.logistic == False:
			return None

		# 順方向の計算
		self.loss(x, t)
		
		# 誤差逆伝播法
		dout = 1
		dout = self.lastLayer.backward(dout)
		layers = list(self.layer)
		layers.reverse()
		for layer in layers:
			dout = layer.backward(dout)

		# delta を収集する
		grad = []
		for layer in self.layer:
			delta = []
			if type(layer) == Affine:
				delta.append(layer.dW)
				delta.append(layer.db)
				grad.append(delta)

		return grad

	def update(self, x, t, eta):
		# 微分系数を求める
		grads = self.gradient(x, t)

		# 誤差逆伝播法
		for i, grad in enumerate(grads):
			for j, g in enumerate(grad):
				self.params[i][j] -= eta * g

