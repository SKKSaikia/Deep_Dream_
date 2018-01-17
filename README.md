# Google's Deep_Dream
Keras 2.1.2 & tf 1.4 with model: ResNet 50, implementation of Google's Deep Dream, based on the Google [blog](https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html), which is based on [Caffe](https://github.com/google/deepdream) and model : GoogLeNet ( aka inception Net ). #deepdream

INPUT : OUTPUT
---------------

Weights ( ResNet 50 )
---------------------
Before running this .py script, download the weights for the ResNet50 ( image-net trained ) model at:
1. [resnet50_weights_tf_dim_ordering_tf_kernels](https://drive.google.com/open?id=1TXWSlWjrrDYW5D5bYJ94Q0spg3nGEEHx)
2. [resnet50_weights_tf_dim_ordering_tf_kernels_notop](https://drive.google.com/open?id=18pj_hzTDIFmYiCumpAVS_QYkLDPJv04E)

and make sure the variable weights_path in this script matches the location of the file.

        default_dir = /Users/User/.keras/models/
        
Dependencies:
-------------
    1. tensorflow 1.4
    2. keras 2.1.2
    3. cv2
    4. CUDA & cuDNN ( GPU )
    
Motivation:
-----------
1. Keras 1.0.6 implementation from [titu1994](https://github.com/titu1994/Deep-Dream).
2. Keras based on VGG19 weights from [giuseppebonaccorso](https://github.com/giuseppebonaccorso/keras_deepdream).
3. Deep Dream with Tensorflow from [tensorflow](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/deepdream).
