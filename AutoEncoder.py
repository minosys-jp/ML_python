# -*- coding: utf-8 -*-
import numpy as np
from commons.functions import sigmoid

class AutoEncoder():
	def __init__(self, input_size, hidden_size, noise_threshold = 0.0):
		self.w = np.random.randn(input_size, hidden_size)
		self.b1 = np.random.randn(hidden_size)
		self.b2 = np.random.randn(input_size)
		self.noise = noise_threshold
		self.x = None
		self.y = None
		self.dw = None
		self.db1 = None
		self.db2 = None

	def forward(self, x):
		xx = x
		if self.noise > 0.0:
			nn = np.random.rand(x.size)
			nn = nn.reshape(x.shape)
			cc = (nn >= self.noise)
			xx = x * cc
		self.x = xx
		self.y = sigmoid(np.dot(xx, self.w) + self.b1)
		z = sigmoid(np.dot(self.y, self.w.T) + self.b2)
		return z

	def backward(self, z, t):
		# Sigmoid と二次ロス関数の合成関数の微分
		dout = (z - t) / z.shape[0]	# (sample_size, input_size)
		self.db2 = np.sum(dout, axis = 0)
		self.db1 = np.sum(np.dot(dout, self.w) * self.y * (1.0 - self.y), axis = 0)
		tmp1 = np.dot(self.y.T, dout).T	# (input_size, hidden_size)
		tmp2 = np.dot(dout, self.w)	# (sample_size, hidden_size)
		tmp2 = tmp2 * self.y * (1.0 - self.y)
		tmp2 = np.dot(self.x.T, tmp2)	# (input_size, hidden_size)
		self.dw = tmp1 + tmp2

	def update(self, x, eta):
		z = self.forward(x)	
		self.backward(z, x)

		# 誤差逆伝播法
		self.w -= eta * self.dw
		self.b1 -= eta * self.db1
		self.b2 -= eta * self.db2

	def loss(self, x, t):
		z = self.forward(x) - x	# (sample_size, input_size)
		return np.sum(np.dot(z.T, z)) / z.shape[0]
