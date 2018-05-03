# DCGAN in Pytorch

## Overview
This code is the Pytorch implementation of Deep Convolutional Generative Adversarial Networks on MNIST and Anime dataset.

## Dataset
Anime dataset can be downloaded from https://zhuanlan.zhihu.com/p/24767059 or https://drive.google.com/drive/folders/1qDhrjL0iz9rKb5t90Iy81pJKKCy9zNXx?usp=sharing

## Prerequisites
* python(3.6)
* pytorch(0.3.1)


## Results on MNIST dataset
The network was trained for 50 epochs in MNIST dataset, 
<table align='center'>
<tr align='center'>
	<td> epoch:10</td>
	<td> epoch:20</td>
	<td> epoch:50</td>
</tr>
<tr>
	<td><img src = 'https://github.com/limingwu8/DCGAN/blob/master/DCGAN-mnist/images/img9.png'>
	<td><img src = 'https://github.com/limingwu8/DCGAN/blob/master/DCGAN-mnist/images/img19.png'>
	<td><img src = 'https://github.com/limingwu8/DCGAN/blob/master/DCGAN-mnist/images/img49.png'>
</tr>
</table>

## Results on Anime dataset
The network was trained for 200 epochs in Anime dataset.
<table align='center'>
<tr align='center'>
	<td> epoch:10</td>
	<td> epoch:50</td>
	<td> epoch:200</td>
</tr>
<tr>
	<td><img src = 'https://github.com/limingwu8/DCGAN/blob/master/DCGAN-anime/images/img9.png'>
	<td><img src = 'https://github.com/limingwu8/DCGAN/blob/master/DCGAN-anime/images/img49.png'>
	<td><img src = 'https://github.com/limingwu8/DCGAN/blob/master/DCGAN-anime/images/img199.png'>
</tr>
</table>

## References
* https://github.com/togheppi/DCGAN

* https://zhuanlan.zhihu.com/p/24767059

* https://github.com/pytorch/examples/tree/master/dcgan
