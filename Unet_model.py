
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Activation,Conv2DTranspose, BatchNormalization,Dropout, Lambda
from tensorflow.keras.optimizers import Adam


def Unet(input_size):
    inputs=Input(input_size,name='Input')
    
    B1_conv1=Conv2D(64,3,padding='same',kernel_initializer='he_normal',name='B1_Conv1')(inputs)
    B1_conv1=BatchNormalization(name='B1_Conv1_BatchNormalization')(B1_conv1)
    B1_conv1=Activation('relu',name='B1_Conv1_Activation')(B1_conv1)
    
    B1_conv2=Conv2D(64,3,padding='same',kernel_initializer='he_normal',name='B1_Conv2')(B1_conv1)
    B1_conv2=BatchNormalization(name='B1_Conv2_BatchNormalization')(B1_conv2)
    B1_conv2=Activation('relu',name='B1_Conv2_Activation')(B1_conv2)
    
    B1_pool=MaxPooling2D(pool_size=(2,2),name='B1_Pooling')(B1_conv2)
    
    B2_conv1=Conv2D(128,3,padding='same',kernel_initializer='he_normal',name='B2_Conv1')(B1_pool)
    B2_conv1=BatchNormalization(name='B2_Conv1_BatchNormalization')(B2_conv1)
    B2_conv1=Activation('relu',name='B2_Conv1_Activation')(B2_conv1)
    
    B2_conv2=Conv2D(128,3,padding='same',kernel_initializer='he_normal',name='B2_Conv2')(B2_conv1)
    B2_conv2=BatchNormalization(name='B2_Conv2_BatchNormalization')(B2_conv2)
    B2_conv2=Activation('relu',name='B2_Conv2_Activation')(B2_conv2)
    
    B2_pool=MaxPooling2D(pool_size=(2,2),name='B2_Pooling')(B2_conv2)
    
    B3_conv1=Conv2D(256,3,padding='same',kernel_initializer='he_normal',name='B3_Conv1')(B2_pool)
    B3_conv1=BatchNormalization(name='B3_Conv1_BatchNormalization')(B3_conv1)
    B3_conv1=Activation('relu',name='B3_Conv1_Activation')(B3_conv1)
    
    B3_conv2=Conv2D(256,3,padding='same',kernel_initializer='he_normal',name='B3_Conv2')(B3_conv1)
    B3_conv2=BatchNormalization(name='B3_Conv2_BatchNormalization')(B3_conv2)
    B3_conv2=Activation('relu',name='B3_Conv2_Activation')(B3_conv2)
    
    B3_pool=MaxPooling2D(pool_size=(2,2),name='B3_Pooling')(B3_conv2)
    
    B4_conv1=Conv2D(512,3,padding='same',kernel_initializer='he_normal',name='B4_Conv1')(B3_pool)
    B4_conv1=BatchNormalization(name='B4_Conv1_BatchNormalization')(B4_conv1)
    B4_conv1=Activation('relu',name='B4_Conv1_Activation')(B4_conv1)
    
    B4_conv2=Conv2D(512,3,padding='same',kernel_initializer='he_normal',name='B4_Conv2')(B4_conv1)
    B4_conv2=BatchNormalization(name='B4_Conv2_BatchNormalization')(B4_conv2)
    B4_conv2=Activation('relu',name='B4_Conv2_Activation')(B4_conv2)
    
    B4_pool=MaxPooling2D(pool_size=(2,2),name='B4_Pooling')(B4_conv2)
    
    BN_conv1=Conv2D(1024,3,padding='same',kernel_initializer='he_normal',name='BN_Conv1')(B4_pool)
    BN_conv1=BatchNormalization(name='BN_Conv1_BatchNormalization')(BN_conv1)
    BN_conv1=Activation('relu',name='BN_Conv1_Activation')(BN_conv1)
    
    BN_conv2=Conv2D(1024,3,padding='same',kernel_initializer='he_normal',name='BN_Conv2')(BN_conv1)
    BN_conv2=BatchNormalization(name='BN_Conv2_BatchNormalization')(BN_conv2)
    BN_conv2=Activation('relu',name='BN_Conv2_Activation')(BN_conv2)
    
    up_B4=Conv2DTranspose(512,(2,2),strides=2,padding='same',name='UP_B4')(BN_conv2)
    merge_B4=concatenate([B4_conv2,up_B4],axis=3,name='Concatenate_B4')
    conv1_B4=Conv2D(512,3,padding='same',kernel_initializer='he_normal',name='UP_Conv1_B4')(merge_B4)
    conv1_B4=BatchNormalization(name='UP_Conv1_B4_BatchNormalization')(conv1_B4)
    conv1_B4=Activation('relu',name='UP_Conv1_B4_Activation')(conv1_B4)
    conv2_B4=Conv2D(512,3,padding='same',kernel_initializer='he_normal',name='UP_Conv2_B4')(conv1_B4)
    conv2_B4=BatchNormalization(name='UP_Conv2_B4_BatchNormalization')(conv2_B4)
    conv2_B4=Activation('relu',name='UP_Conv2_B4_Activation')(conv2_B4)
    
    up_B3=Conv2DTranspose(256,(2,2),strides=2,padding='same',name='UP_B3')(conv2_B4)
    merge_B3=concatenate([B3_conv2,up_B3],axis=3,name='Concatenate_B3')
    conv1_B3=Conv2D(256,3,padding='same',kernel_initializer='he_normal',name='UP_Conv1_B3')(merge_B3)
    conv1_B3=BatchNormalization(name='UP_Conv1_B3_BatchNormalization')(conv1_B3)
    conv1_B3=Activation('relu',name='UP_Conv1_B3_Activation')(conv1_B3)
    conv2_B3=Conv2D(256,3,padding='same',kernel_initializer='he_normal',name='UP_Conv2_B3')(conv1_B3)
    conv2_B3=BatchNormalization(name='UP_Conv2_B3_BatchNormalization')(conv2_B3)
    conv2_B3=Activation('relu',name='UP_Conv2_B3_Activation')(conv2_B3)
    
    up_B2=Conv2DTranspose(128,(2,2),strides=2,padding='same',name='UP_B2')(conv2_B3)
    merge_B2=concatenate([B2_conv2,up_B2],axis=3,name='Concatenate_B2')
    conv1_B2=Conv2D(128,3,padding='same',kernel_initializer='he_normal',name='UP_Conv1_B2')(merge_B2)
    conv1_B2=BatchNormalization(name='UP_Conv1_B2_BatchNormalization')(conv1_B2)
    conv1_B2=Activation('relu',name='UP_Conv1_B2_Activation')(conv1_B2)
    conv2_B2=Conv2D(128,3,padding='same',kernel_initializer='he_normal',name='UP_Conv2_B2')(conv1_B2)
    conv2_B2=BatchNormalization(name='UP_Conv2_B2_BatchNormalization')(conv2_B2)
    conv2_B2=Activation('relu',name='UP_Conv2_B2_Activation')(conv2_B2)
    
    up_B1=Conv2DTranspose(64,(2,2),strides=2,padding='same',name='UP_B1')(conv2_B2)
    merge_B1=concatenate([B1_conv2,up_B1],axis=3,name='Concatenate_B1')
    conv1_B1=Conv2D(64,3,padding='same',kernel_initializer='he_normal',name='UP_Conv1_B1')(merge_B1)
    conv1_B1=BatchNormalization(name='UP_Conv1_B1_BatchNormalization')(conv1_B1)
    conv1_B1=Activation('relu',name='UP_Conv1_B1_Activation')(conv1_B1)
    conv2_B1=Conv2D(64,3,padding='same',kernel_initializer='he_normal',name='UP_Conv2_B1')(conv1_B1)
    conv2_B1=BatchNormalization(name='UP_Conv2_B1_BatchNormalization')(conv2_B1)
    conv2_B1=Activation('relu',name='UP_Conv2_B1_Activation')(conv2_B1)
    
    outputs=Conv2D(1,1,activation='sigmoid',padding='same',name='Output')(conv2_B1)
    
    model=Model(inputs,outputs,name='UNet')
    return model



