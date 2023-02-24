# Watermarking for Out-of-distribution Detection

This repository is the official implementation of the NeurIPS'22 paper [Watermarking for Out-of-distribution Detection](), authored by Qizhou Wang *, Feng Liu *, Yonggang Zhang, Jing Zhang, Chen Gong, Tongliang Liu, and Bo Han. Our method utilizes the reprogramming properties of deep models, boosting the capablity of fixed models in OOD detection without changing their parameters. 

**Key Words**: Out-of-distribution Detection, Reliable Machine Learning

**Abstract**: Out-of-distribution (OOD) detection aims to identify OOD data based on representations extracted from well-trained deep models. However, existing methods largely ignore the reprogramming property of deep models and thus may not fully unleash their intrinsic strength: without modifying parameters of a well-trained deep model, we can reprogram this model for a new purpose via data-level manipulation (e.g., adding a specific feature perturbation to the data). This property motivates us to reprogram a classification model to excel at OOD detection (a new task), and thus we propose a general methodology named watermarking in this paper. Specifically, we learn a unified pattern that is superimposed onto features of original data, and the model's detection capability is largely boosted after watermarking. Extensive experiments verify the effectiveness of watermarking, demonstrating the significance of the reprogramming property of deep models in OOD detection.

@inproceedings{
wang2022watermarking,
title={Watermarking for Out-of-distribution Detection},
author={Qizhou Wang and Feng Liu and Yonggang Zhang and Jing Zhang and Chen Gong and Tongliang Liu and Bo Han},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=6rhl2k1SUGs}
}


## Get Started

### Environment
- Python (3.7.10)
- Pytorch (1.7.1)
- torchvision (0.8.2)
- CUDA
- Numpy

### Pretrained Models and Datasets

Pretrained models are provided in folder

```
./ckpt/
```

Please download the datasets in folder

```
../data/
```

- [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/)

- [Places365](http://places2.csail.mit.edu/download.html)

- [LSUN-C](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz)
 
- [LSUN-R](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz)

- [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz)




## Training

To train the watermarking on CIFAR benckmarks, simply run:

- CIFAR-10
```train cifar10
python train.py cifar10 
```


- CIFAR-100
```train cifar100
python train.py cifar100
```
