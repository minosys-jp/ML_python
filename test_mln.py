#!/usr/bin/python3
# -*- coding: utf8 -*-

import numpy as np
from MNIST import loader_evaluate
from TwoLayerNet import TwoLayerNet
from MultiLayerNet import MultiLayerNet
import json
from collections import OrderedDict

x_test, t_test = \
	loader_evaluate(normalize = True, shape = False, decode = True)

# パラメータ
network = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)

# 保存済みニューラルネットパラメータをロード
network.load()

# NLN を TwuLayerNet から構築
netlist = [ 784, 50, 10 ]
para = [ \
	[network.params['W1'], network.params['b1']], \
	[network.params['W2'], network.params['b2']]  \
	]
mln = MultiLayerNet(logistic = True, netlist = netlist, params = para)

test_acc = network.accuracy(x_test, t_test)
mln_acc = mln.accuracy(x_test, t_test)

print("test(%): ", test_acc * 100, ", mln(%): ", mln_acc * 100)

# gradient を計算して結果を比較する
batch_size = 100
batch_mask = np.random.choice(x_test.shape[0], batch_size)
x_batch = x_test[batch_mask]
t_batch = t_test[batch_mask]

g_net = network.gradient(x_batch, t_batch)
g_mln = mln.gradient(x_batch, t_batch)
print("diff-W1: ", np.sum(g_net['W1'] != g_mln[0][0]))
print("diff-b1: ", np.sum(g_net['b1'] != g_mln[0][1]))
print("diff-W2: ", np.sum(g_net['W2'] != g_mln[1][0]))
print("diff-b2: ", np.sum(g_net['b2'] != g_mln[1][1]))
