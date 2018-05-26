#!/usr/bin/python3
# -*- coding: utf8 -*-

import numpy as np
from MNIST import loader
from AutoEncoder import AutoEncoder
from MultiLayerNet import MultiLayerNet
import pickle
import json

(x_train, t_train), (x_test, t_test) = \
	loader(normalize = True, shape = False, decode = True)

train_loss_list = []
train_acc_list = []
test_acc_list = []

# ハイパーパラメータ
#netlist = [ 784, 400, 400, 10 ]
netlist = [ 784, 400, 400, 10 ]
iters_num = 4000
train_size = x_train.shape[0]
test_size = x_test.shape[0]
batch_size = 100
learning_rate = 0.1
#iter_per_epoch = int(max(train_size / batch_size, 1))
iter_per_epoch = 200

mln = MultiLayerNet(logistic = False, netlist = [])

def pretraining(mln, x, t, eta, iters, bar = lambda c: 0):
	for i in range(len(netlist) - 2):
		if i == len(netlist) - 3:
			nl = [ netlist[i], netlist[i + 1], netlist[i + 2] ]
			nn = MultiLayerNet(logistic = True, netlist = nl)
		else:
			nn = AutoEncoder(input_size = netlist[i], hidden_size = netlist[i + 1], noise_threshold = 0.3)
		for j in range(iters):
			# ミニバッチの取得
			batch_mask = np.random.choice(train_size, batch_size)
			x_batch = x[batch_mask]

			# 中間結果の MultiLayerNet に通す
			if i > 0:
				x_batch = mln.predict(x_batch)
				x_batch = mln.lastLayer.forward(x_batch)

			# 誤差逆伝播法でパラメータを更新
			if i == len(netlist) - 3:
				t_batch = t[batch_mask]
				nn.update(x_batch, t_batch, eta)
			else:
				t_batch = x_batch
				nn.update(x_batch, eta)	
			if j % iter_per_epoch == 0:
				print("loss=", nn.loss(x_batch, t_batch))
				bar('*')

		# AE, mini-MLN の parameter を mln に追加する
		if i == len(netlist) - 3:
			mln.append([ nn.params[0][0], nn.params[0][1] ])
			mln.append([ nn.params[1][0], nn.params[1][1] ])
		else:
			mln.append([ nn.w, nn.b1 ])
		bar('/')
	# mln の logistic を有効にする
	mln.toLogistic()

def finetuning(mln, x, t, eta, iters, bar = lambda acc: 0):
	for i in range(iters_num):
		# ミニバッチの取得
		batch_mask = np.random.choice(train_size, batch_size)
		x_batch = x[batch_mask]
		t_batch = t[batch_mask]

		# 勾配の計算
		mln.update(x_batch, t_batch, learning_rate)
		if i % iter_per_epoch == 0:
			acc = mln.accuracy(x_batch, t_batch)
			bar(acc)

def dumper(mln, f):
	# 注釈
	f.write("# MLN dumper version 1.0\n")

	# MLN の段数
	f.write("PIPELINES")
	f.write(" ")
	f.write(str(mln.params[0][0].shape[0]))
	for u in mln.params:
		f.write(" ");
		f.write(str(len(u[1])))
	f.write("\n")

	# パラメータの実際の値を出力する
	for p in mln.params:
		f.write("W ")
		f.write(str(p[0].shape[0]))
		f.write(" ")
		f.write(str(p[0].shape[1]))
		f.write("\n")
		spacing = False
		for v in range(p[0].shape[1]):
			spacing = False
			for u in range(p[0].shape[0]):
				if spacing:
					f.write(" ")
				f.write(str(p[0][u][v]))
				spacing = True
			f.write("\n")
		f.write("b ")
		f.write(str(p[1].shape[0]))
		f.write("\n")
		spacing = False
		for v in range(p[1].shape[0]):
			if spacing:
				f.write(" ")
			f.write(str(p[1][v]))
			spacing = True
		f.write("\n")

# 大まかな training
pretraining(mln, x_train, t_train, learning_rate, iters_num, bar = lambda c: print(c, end = '', flush = True))
print()

# fine tuning
finetuning(mln, x_train, t_train, learning_rate, iters_num, bar = lambda acc: print("acc=", acc))

print("...certify by test data")

# テストデータで検証
for i in range(10):
	# ミニバッチの取得
	batch_mask = np.random.choice(test_size, batch_size)
	x_batch = x_test[batch_mask]
	t_batch = t_test[batch_mask]

	print("acc=", mln.accuracy(x_batch, t_batch))

# MLN をダンプ
with open("./mlndump.pickle", "wb") as f:
	pickle.dump(mln, f)
with open("./mlndump.json", "w") as f:
	js = [ [p[0].tolist(), p[1].tolist()] for p in  mln.params]
	json.dump(js, f)
with open("./mlndump.txt", "w") as f:
	dumper(mln, f)

