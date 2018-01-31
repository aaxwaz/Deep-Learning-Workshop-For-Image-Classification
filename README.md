# NUS Deep Learning workshop in Computer Vision
Building am image classifier using Convolutional Neural Networks in TensorFlow as well as Keras. 

For tutorials on TensorFlow, please refer to: 

```TensorFlow_basic_CNN.ipynb```

1. For tutorials on Keras (including use of pre-trained models of VGG and Inception-V3): 

```keras_VGG_InceptionV3.ipynb```

Files structure: 

2. To train a customerized CNN from scratch, run command line below 

```python train.py --trainDir /dir/to/download/and/load/data/from/ --savedSessionDir /dir/to/save/your/model/session/```

Dependencies: 

```
configuration - model architecture configuration 
build_graph - main graph building for CNN model 
data_utils - utils functions to download, save and load data 
vis_utils - util functions to visualize hidden / activation layers 
```






