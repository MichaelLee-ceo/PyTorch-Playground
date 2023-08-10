# PyTorch implementation of some deep learning algorithms
- Knowledge Distillation
- Domain Adaptation


## Domain Adaptation
### Domain Adversarial Training of Neural Networks
- Source Dataset (MNIST)

![MNIST](./Domain%20Adaptation/figures/mnist.png)

- Target Dataset (MNIST-M)

![MNIST-M](./Domain%20Adaptation/figures/mnist_m.png)

| | w/o Domain Adaptation | Domain Adaptation |
| :-: | :-: | :-: |
| Accuracy | 45% | 77% |

</br>

## Knowledge Distillation
### Baseline
CIFAR10
|  | 3-layer CNN (Student) | Resnet18 (Teacher) |
|:-:|:-:|:-:|
| Accuracy | 83.46% | 93.27% |

FashionMNIST
|  | 3-layer CNN (Student) | Resnet18 (Teacher) |
|:-:|:-:|:-:|
| Accuracy | 92.43% | 94.66% |

### With knowledge distillation
*T: temperature, R: alpha rate*

CIFAR10
|  | T=4, R=0.9 | T=4, R=0.95 | T=4, R=1.0 | T=10, R=0.9 |
|--|:-:|:-:|:-:|:-:|
| Accuracy | 84.47% | 84.86% | 84.14% | -      |

FashionMNIST
|  | T=4, R=0.9 | T=4, R=0.95 | T=4, R=1.0 | T=10, R=0.9 |
|--|:-:|:-:|:-:|:-:|
| Accuracy | 92.94% | 92.97% | 92.93% | 92.63% |