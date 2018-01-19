# Google's Deep_Dream
Keras 2.1.2 with model: VGG 19, implementation of Google's Deep Dream, based on the Google [blog](https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html), which is based on [Caffe](https://github.com/google/deepdream) and model : GoogLeNet ( aka inception Net ). #deepdream

INPUT ( [IM1](https://drive.google.com/open?id=1Lijvb-LTS6uliaRFM3YVioml7nnjNfEe), [IM2](https://drive.google.com/open?id=1vJTAC61hFudwazMlO1udg1BTCEukHIr6) ) : OUTPUT
---------------
<b>Output 1:</b> VGG19
<img src="https://github.com/SKKSaikia/Deep_Dream_/blob/master/cod.jpg">
<img src="https://github.com/SKKSaikia/Deep_Dream_/blob/master/cod_dream.jpg">

<b>Output 2:</b> with #geekodour in BITs Pilani, dreamed using py script from [here](https://github.com/keras-team/keras/blob/master/examples/deep_dream.py), which uses Inception_V3
<img src="https://github.com/SKKSaikia/Deep_Dream_/blob/master/res/de.jpg">
Tweaking the hyperparameters, we have :

        1step = 0.02
        num_octave = 5 
        octave_scale = 1.4  
        iterations = 4  
        max_loss = 10.
        
<img src="https://github.com/SKKSaikia/Deep_Dream_/blob/master/res/cod_inc.jpg">

Weights ( VGG19 )
---------------------
Before running this .py script, download the weights for the VGG19 ( ImageNet trained ) model at:
1. [VGG19_weights_tf_dim_ordering_tf_kernels](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5)
2. [VGG19_weights_tf_dim_ordering_tf_kernels_notop](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5)

You can also try it on ResNet50, download the weights for ResNet50 model at:
1. [resnet50_weights_tf_dim_ordering_tf_kernels](https://drive.google.com/open?id=1TXWSlWjrrDYW5D5bYJ94Q0spg3nGEEHx)
2. [resnet50_weights_tf_dim_ordering_tf_kernels_notop](https://drive.google.com/open?id=18pj_hzTDIFmYiCumpAVS_QYkLDPJv04E)

and make sure the variable weights_path in this script matches the location of the file.

        default_dir = /Users/User/.keras/models/
        
Also, download the Inception_V3 weights here : [tf_top](https://drive.google.com/open?id=1jZUnu32vAjiYWVVH8R-25ta8zv4we416), [tf_no_top](https://drive.google.com/open?id=1ILwxc67ZwYWqOjJH8u9DtD79OggtkiAg)

Dependencies:
-------------
    1. keras 2.1.2
    2. scipy 0.19.1
    3. CUDA & cuDNN ( GPU ), 8/6 & GTX 960m- my system

Run .py (Start Dreaming):
-------------
    > python dream.py

<b>Motivation:</b> [giuseppebonaccorso](https://github.com/giuseppebonaccorso/keras_deepdream).

Why are there so many dog heads, Chalices, Japanese-style buildings and eyes being imagined by these neural networks?
-
Nearly all of these images are being created by 'reading the mind' of neural networks that were trained on the ImageNet dataset. This dataset has lots of different types of images within it, but there happen to be a ton of dogs, chalices, etc.

If you were to train your own neural network with lots of images of hands then you could generate your own deepdream images from this net and see everything be created from hands.

Here is using MIT's places dataset [implementation](https://www.youtube.com/watch?v=6IgbMiEaFRY).

We must go deeper: Iterations
-
If we apply the algorithm iteratively on its own outputs and apply some zooming after each iteration, we get an endless stream of new impressions, exploring the set of things the network knows about. Here's my [implemetation]().
