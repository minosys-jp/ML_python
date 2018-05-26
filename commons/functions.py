#!/usr/bin/python3
# -*- coding: utf8 -*-

import numpy as np

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def softmax(x):
	mx = np.max(x, axis = 1)
	r = np.zeros_like(x)
	for i in range(mx.shape[0]):
		r[i] = np.exp(x[i] - mx[i]) / np.sum(np.exp(x[i] - mx[i]))
	return r

# x must be softmax format (appearing probability)
# t must be in decode format (unit vector)
def cross_entropy_error(x, t):
	if x.ndim == 1:
		t = t.reshape(1, t.size)
		x = x.reshape(1, x.size)

	d = 1e-7
	batch_size = x.shape[0]
	if t.ndim != x.ndim:
		return -np.sum(np.log(x[np.arange(batch_size), t])) / batch_size
	else:
		return -np.sum(t * np.log(x + d)) / batch_size
