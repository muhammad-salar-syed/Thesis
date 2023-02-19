from keras.models import Model
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,Concatenate,Activation,Conv2DTranspose,BatchNormalization,Dropout,Lambda
from tensorflow.keras.optimizers import Adam
from keras.layers import Activation,MaxPool2D,Concatenate
from tensorflow.keras.applications import VGG16


def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def Unet_VGG16(input_shape):

    inputs = Input(input_shape)

    """ Pre-trained VGG19 Model """
    vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=inputs)
    #vgg16.summary()
    """ Encoder """
    s1 = vgg16.get_layer("block1_conv2").output         ## (512 x 512)
    s2 = vgg16.get_layer("block2_conv2").output         ## (256 x 256)
    s3 = vgg16.get_layer("block3_conv3").output         ## (128 x 128)
    s4 = vgg16.get_layer("block4_conv3").output         ## (64 x 64)
    """ Bridge """
    b1 = vgg16.get_layer("block5_conv3").output 
    """ Decoder """
    d1 = decoder_block(b1, s4, 512)                     ## (64 x 64)
    d2 = decoder_block(d1, s3, 256)                     ## (128 x 128)
    d3 = decoder_block(d2, s2, 128)                     ## (256 x 256)
    d4 = decoder_block(d3, s1, 64)                      ## (512 x 512)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="UNet_VGG16")
    return model

