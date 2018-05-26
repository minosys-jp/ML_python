#!/usr/bin/python3
# -*- coding: utf8 -*-

from PIL import Image
import numpy as np
from MNIST import loader
from AutoEncoder import AutoEncoder

(x_train, t_train), (x_test, t_test) = \
	loader(normalize = True, shape = False, decode = True)

train_loss_list = []
train_acc_list = []
test_acc_list = []

# ハイパーパラメータ
iters_num = 1000
train_size = x_train.shape[0]
test_size = x_test.shape[0]
batch_size = 100
learning_rate = 0.1
#iter_per_epoch = int(max(train_size / batch_size, 1))
iter_per_epoch = 100

ae = AutoEncoder(input_size = 784, hidden_size = 784, noise_threshold = 0.3)

for i in range(iters_num):
	# ミニバッチの取得
	batch_mask = np.random.choice(train_size, batch_size)
	x_batch = x_train[batch_mask]

	# 勾配の計算
	#grad = network.numerical_gradient(x_batch, t_batch)
	ae.update(x_batch, learning_rate)

	# １エポックごとに loss を表示
	if i % iter_per_epoch == 0:
		print("loss=", ae.loss(x_batch, x_batch))

batch_mask = np.random.choice(test_size, batch_size)
x_batch = x_test[batch_mask]
z = ae.forward(x_batch).reshape((batch_size, 28, 28))
bk = Image.new('RGB', (28 * 10, 28 * 10), color = (128, 0, 0))
for i in range(100):
	x = i % 10
	y = i // 10
	im = Image.fromarray(np.uint8(z[i] * 255))
	bk.paste(im, (x * 28, y * 28))
bk.show()
