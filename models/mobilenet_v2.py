from keras.applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from keras.layers import Input, Convolution2D, \
    GlobalAveragePooling2D, Dense, BatchNormalization, Activation, ZeroPadding2D
from keras.models import Model
from keras.engine.topology import get_source_inputs
from depthwise_conv2d import DepthwiseConvolution2D
import keras
'''Google MobileNet model for Keras.
# Reference:
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)
'''

def MobileNet(input_tensor=None,layers = True, input_shape=None, alpha=1, shallow=False):
    """Instantiates the MobileNet.Network has two hyper-parameters
        which are the width of network (controlled by alpha)
        and input size.
        
        # Arguments
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(224, 224, 3)` (with `channels_last` data format)
                or `(3, 224, 244)` (with `channels_first` data format).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 96.
                E.g. `(200, 200, 3)` would be one valid value.
            alpha: optional parameter of the network to change the 
                width of model.
            shallow: optional parameter for making network smaller.
            classes: optional number of classes to classify images
                into.
        # Returns
            A Keras model instance.

        """

    # input_shape = _obtain_input_shape(input_shape,
    #                                   default_size=224,
    #                                   min_size=96,
    #                                   data_format=K.image_data_format(),
    #                                   include_top=True)

    # if input_tensor is None:
    #     img_input = Input(shape=input_shape)
    # else:
    #     if not K.is_keras_tensor(input_tensor):
    #         img_input = Input(tensor=input_tensor, shape=input_shape)
    #     else:
    #         img_input = input_tensor

    if not K.is_keras_tensor(input_tensor):
        print 'input tensor is not valid'


    print 'mobilenet is on !!!!!!!!!!!!!!!!!!'


    training_flag = True

    if(layers == "base_network" or layers == "all"):
        training_flag = True
        print "Training the base network also"
    else:
        training_flag = False
        print "Training off for the base_network"







    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv1_padding')(input_tensor)
    x = Convolution2D(int(32 * alpha), (3, 3), strides=(2, 2), padding='valid', use_bias=False,trainable=training_flag)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)





  

    x = keras.layers.DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depth_multiplier=1,use_bias=False,trainable=training_flag)(x)
    # x = DepthwiseConvolution2D(int(32 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False,trainable=training_flag)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(64 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False,trainable=training_flag)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv2_padding')(x)
    x = keras.layers.DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid', depth_multiplier=1,use_bias=False,trainable=training_flag)(x)
    # x = DepthwiseConvolution2D(int(64 * alpha), (3, 3), strides=(2, 2), padding='valid', use_bias=False,trainable=training_flag)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(128 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False,trainable=training_flag)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = keras.layers.DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depth_multiplier=1,use_bias=False,trainable=training_flag)(x)
    # x = DepthwiseConvolution2D(int(128 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False,trainable=training_flag)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(128 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False,trainable=training_flag)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv3_padding')(x)
    x = keras.layers.DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid', depth_multiplier=1,use_bias=False,trainable=training_flag)(x)
    # x = DepthwiseConvolution2D(int(128 * alpha), (3, 3), strides=(2, 2), padding='valid', use_bias=False,trainable=training_flag)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(256 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False,trainable=training_flag)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = keras.layers.DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depth_multiplier=1,use_bias=False,trainable=training_flag)(x)
    # x = DepthwiseConvolution2D(int(256 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False,trainable=training_flag)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(256 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False,trainable=training_flag)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    conv_4  =x
    
    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv4_padding')(x)
    # x = DepthwiseConvolution2D(int(256 * alpha), (3, 3), strides=(2, 2), padding='valid', use_bias=False,trainable=training_flag)(x)
    x = keras.layers.DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid', depth_multiplier=1,use_bias=False,trainable=training_flag)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False,trainable=training_flag)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    if not shallow:
        for _ in range(5):
            x = keras.layers.DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depth_multiplier=1,use_bias=False,trainable=training_flag)(x)
            # x = DepthwiseConvolution2D(int(512 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False,trainable=training_flag)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Convolution2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False,trainable=training_flag)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)



    conv_11 = x

    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv5_padding')(x)
    x = keras.layers.DepthwiseConv2D((3, 3), strides=(1, 1), padding='valid', depth_multiplier=1,use_bias=False,trainable=training_flag)(x)
    # x = DepthwiseConvolution2D(int(512 * alpha), (3, 3), strides=(2, 2), padding='valid', use_bias=False,trainable=training_flag)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(1024 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False,trainable=training_flag)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = keras.layers.DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', depth_multiplier=1,use_bias=False,trainable=training_flag)(x)
    # x = DepthwiseConvolution2D(int(1024 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False,trainable=training_flag)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(1024 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False,trainable=training_flag)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)


    return [conv_4,conv_11,x]


if __name__ == '__main__':
    m = MobileNet()
    print "model ready"