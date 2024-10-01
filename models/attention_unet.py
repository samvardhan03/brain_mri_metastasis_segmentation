from tensorflow.keras import layers, models

def attention_block(x, g, inter_shape):
    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2))(x)
    phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(g)
    f = layers.Activation('relu')(layers.add([theta_x, phi_g]))
    psi_f = layers.Conv2D(1, (1, 1), padding='same')(f)
    return layers.multiply([x, layers.Activation('sigmoid')(psi_f)])

def attention_unet(input_shape):
    inputs = layers.Input(input_shape)
    
    # Contracting Path
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    
    # Bottleneck
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    
    # Attention Block
    g = layers.Conv2D(128, (1, 1), padding='same')(conv3)
    att_block = attention_block(conv2, g, 128)
    
    # Expanding Path
    up4 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv3)
    up4 = layers.concatenate([up4, att_block])
    conv4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up4)
    conv4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
    
    up5 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv4)
    up5 = layers.concatenate([up5, conv1])
    conv5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up5)
    conv5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv5)
    
    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model
