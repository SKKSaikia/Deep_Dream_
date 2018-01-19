# Google's Deep_Dream
Keras 2.1.2 & tf 1.4 with model: ResNet 50, implementation of Google's Deep Dream, based on the Google [blog](https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html), which is based on [Caffe](https://github.com/google/deepdream) and model : GoogLeNet ( aka inception Net ). #deepdream

INPUT ( [IM1](http://www.pixoto.com/images-photography/nature-up-close/flowers---2011-2013/floral-beauty-104552406), [IM2](http://www.pixoto.com/images-photography/abstract/patterns/water-jaws-104547646) ) : OUTPUT
---------------
Output 1:

Trial ([code](https://github.com/SKKSaikia/Deep_Dream_/blob/master/try.py))
<img src="https://github.com/SKKSaikia/Deep_Dream_/blob/master/res/google.jpg">
<b>Output 2:</b> My Friend #geekodour and me in BITs Pilani, dreamed using py script from [here](https://github.com/keras-team/keras/blob/master/examples/deep_dream.py), which uses Inception_V3
<img src="https://github.com/SKKSaikia/Deep_Dream_/blob/master/res/de.jpg">
Output 3: Inception_V3, You can see how Pig like features are formed in the tree and bushes.
<img src="https://github.com/SKKSaikia/Deep_Dream_/blob/master/res/a.jpg">

Weights ( ResNet 50 )
---------------------
Before running this .py script, download the weights for the ResNet50 ( ImageNet trained ) model at:
1. [resnet50_weights_tf_dim_ordering_tf_kernels](https://drive.google.com/open?id=1TXWSlWjrrDYW5D5bYJ94Q0spg3nGEEHx)
2. [resnet50_weights_tf_dim_ordering_tf_kernels_notop](https://drive.google.com/open?id=18pj_hzTDIFmYiCumpAVS_QYkLDPJv04E)

and make sure the variable weights_path in this script matches the location of the file.

        default_dir = /Users/User/.keras/models/
        
Also download the Inception_V3 weights here : [tf_top](https://drive.google.com/open?id=1jZUnu32vAjiYWVVH8R-25ta8zv4we416), [tf_no_top](https://drive.google.com/open?id=1ILwxc67ZwYWqOjJH8u9DtD79OggtkiAg)

Dependencies:
-------------
    1. tensorflow 1.4
    2. keras 2.1.2
    3. cv2 ( opencv 3.4+contrib )
    4. scipy 0.19.1
    4. CUDA & cuDNN ( GPU ), 8/6 & GTX 960m- my system
    
Motivation:
-----------
1. Keras 1.0.6 implementation from [titu1994](https://github.com/titu1994/Deep-Dream).
2. Keras based on VGG19 weights from [giuseppebonaccorso](https://github.com/giuseppebonaccorso/keras_deepdream).
3. Deep Dream with Tensorflow from [tensorflow](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/deepdream).

Why are there so many dog heads, Chalices, Japanese-style buildings and eyes being imagined by these neural networks?
-
Nearly all of these images are being created by 'reading the mind' of neural networks that were trained on the ImageNet dataset. This dataset has lots of different types of images within it, but there happen to be a ton of dogs, chalices, etc.

If you were to train your own neural network with lots of images of hands then you could generate your own deepdream images from this net and see everything be created from hands.

Here is using MIT's places dataset [implementation](https://www.youtube.com/watch?v=6IgbMiEaFRY).

We must go deeper: Iterations
-
If we apply the algorithm iteratively on its own outputs and apply some zooming after each iteration, we get an endless stream of new impressions, exploring the set of things the network knows about. Here's my [implemetation]().
