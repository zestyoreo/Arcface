import tensorflow as tf
bn_axis = -1
initializer = 'glorot_normal'


def residual_unit(inputs, num_filter, stride, dim_match, name):
    bn_axis = -1
    initializer = 'glorot_normal'
    x = tf.keras.layers.BatchNormalization(axis = bn_axis, 
                                           scale = True, 
                                           momentum = 0.9, 
                                           epsilon = 2e-5, 
                                           gamma_regularizer=tf.keras.regularizers.l2(l=5e-4), 
                                           name=name + '_bn1')(inputs)
    x = tf.keras.layers.ZeroPadding2D(padding=(1, 1), 
                                      name=name + '_conv1_pad')(x)
    x = tf.keras.layers.Conv2D(num_filter, 
                               (3, 3),
                               strides=(1, 1),
                               padding='valid',
                               kernel_initializer=initializer,
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(l=5e-4),
                               name=name + '_conv1')(x)
    x = tf.keras.layers.BatchNormalization(axis = bn_axis, 
                                           scale = True, 
                                           momentum = 0.9, 
                                           epsilon = 2e-5, 
                                           gamma_regularizer=tf.keras.regularizers.l2(l=5e-4), 
                                           name=name + '_bn2')(x)
    x = tf.keras.layers.PReLU(name=name + '_relu1',
                             alpha_regularizer=tf.keras.regularizers.l2(l = 5e-4))(x)
    
    x = tf.keras.layers.ZeroPadding2D(padding=(1, 1), 
                                      name=name + '_conv2_pad')(x)
    
    x = tf.keras.layers.Conv2D(num_filter, 
                               (3, 3),
                               strides=stride,
                               padding='valid',
                               kernel_initializer=initializer,
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(l=5e-4),
                               name=name + '_conv2')(x)
    
    x = tf.keras.layers.BatchNormalization(axis = bn_axis, 
                                           scale = True, 
                                           momentum = 0.9, 
                                           epsilon = 2e-5, 
                                           gamma_regularizer=tf.keras.regularizers.l2(l=5e-4), 
                                           name=name + '_bn3')(x)
    
    if(dim_match):
        shortcut = inputs
    else:
        shortcut = tf.keras.layers.Conv2D(num_filter,
                                         (1,1),
                                         strides=stride,
                                         padding='valid',
                                         kernel_initializer=initializer,
                                         use_bias=False,
                                         kernel_regularizer=tf.keras.regularizers.l2(l=5e-4),
                                         name=name + '_conv1sc')(inputs)
        shortcut = tf.keras.layers.BatchNormalization(axis = bn_axis,
                                                     scale = True,
                                                     momentum = 0.9,
                                                     epsilon=2e-5,
                                                     gamma_regularizer=tf.keras.regularizers.l2(l=5e-4),
                                                     name=name + '_sc')(shortcut)
    return x + shortcut

def head(input_shape = [112, 112, 3]):
    img_input = tf.keras.layers.Input(shape=input_shape)
    
    x = tf.keras.layers.ZeroPadding2D(padding=(1, 1), 
                                      name='conv0_pad')(img_input)
    
    x = tf.keras.layers.Conv2D(64, 
                               (3, 3),
                               strides=(1, 1),
                               padding='valid',
                               kernel_initializer=initializer,
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(l=5e-4),
                               name='conv0')(x)
    
    x = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                           scale=True,
                                           momentum=0.9,
                                           epsilon=2e-5,
                                           gamma_regularizer=tf.keras.regularizers.l2(l=5e-4),
                                           name='bn0')(x)
    
    
    x = tf.keras.layers.PReLU(name = 'prelu0',
                             alpha_regularizer = tf.keras.regularizers.l2(l = 5e-4))(x)
    return img_input, x

def out_layer(inputs, out_size = 512):
    x = tf.keras.layers.BatchNormalization(axis = bn_axis,
                                          scale = True,
                                          momentum = 0.9,
                                          epsilon = 2e-5,
                                          gamma_regularizer = tf.keras.regularizers.l2(l=5e-4),
                                          name = 'bn1')(inputs)
    x = tf.keras.layers.Dropout(0.4)(x)
    resnet_shape = inputs.shape
    x = tf.keras.layers.Reshape([resnet_shape[1]*resnet_shape[2]*resnet_shape[3]], name = 'reshapelayer')(x)
    x = tf.keras.layers.Dense(out_size,
                             name = 'E_denseLayer',
                             kernel_initializer = initializer,
                             kernel_regularizer = tf.keras.regularizers.l2(l = 5e-4),
                             bias_regularizer = tf.keras.regularizers.l2(l=5e-4))(x)
    x = tf.keras.layers.BatchNormalization(axis = bn_axis,
                                          scale = False,
                                          momentum = 0.9,
                                          epsilon = 2e-5,
                                          #gamma_regularizer = tf.keras.regularizers.l2(l=5e-4),
                                          name = 'fc')(x)
    x = tf.keras.layers.Softmax(axis = -1)(x)
    return x


def body_resnet(x, no_layers = 34):
    num_stage = 4
    if no_layers == 10:
        units = [1, 1, 1, 1]
    elif no_layers == 18:
        units = [2, 2, 2, 2]
    elif no_layers == 34:
        units = [3, 4, 6, 3]
    elif no_layers == 50:
        units = [3, 4, 6, 3]
    elif no_layers == 101:
        units = [3, 4, 23, 3]
    elif no_layers == 152:
        units = [3, 8, 36, 3]
    elif no_layers == 200 :
        units = [3, 24, 36, 3]
        
    filter_list = [64, 64, 128, 256, 512]
    for i in range(num_stage):
        x = residual_unit(x, filter_list[i+1], (2,2), False, name = 'stage%d_unit%d' %(i+1, 1))
        for j in range(units[i] - 1):
            x = residual_unit(x, filter_list[i + 1], (1,1), True, name= 'stage%d_unit%d' %(i + 1, j +2))
    return x


def ResNet(no_layers = 34):
    inputs, x = head()
    x = body_resnet(x, no_layers = no_layers)
    out =  out_layer(x)
    model = tf.keras.models.Model(inputs, out, name = 'resnet50')
    model.trainable = True
    for i in range(len(model.layers)):
        model.layers[i].trainable = True
    
    return model