from __future__ import print_function
import argparse
import cv2
import numpy as np
import os
import tensorflow as tf

def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8): 
    try: 
        n = np.fromfile(filename, dtype) 
        img = cv2.imdecode(n, flags) 
        return img 
    except Exception as e: 
        print(e) 
        return None

def imwrite(filename, img, params=None): 
    try: 
        ext = os.path.splitext(filename)[1] 
        result, n = cv2.imencode(ext, img, params) 
        if result: 
            with open(filename, mode='w+b') as f: 
                n.tofile(f) 
            return True 
        else: 
            return False 
    except Exception as e: 
        print(e) 
        return False

def one_hot_encoding(word, word2index):
       one_hot_vector = [0]*(len(word2index))
       index=word2index[word]
       one_hot_vector[index]=1
       return one_hot_vector

###################

def conv1_layer(x):    
    x = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(x)
    x = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.ZeroPadding2D(padding=(1,1))(x)
 
    return x   
 
    
 
def conv2_layer(x):         
    x = tf.keras.layers.MaxPooling2D((3, 3), 2)(x)     
 
    shortcut = x
 
    for i in range(3):
        if (i == 0):
            x = tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            
            x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
 
            x = tf.keras.layers.Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = tf.keras.layers.Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(shortcut)            
            x = tf.keras.layers.BatchNormalization()(x)
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)
 
            x = tf.keras.layers.Add()([x, shortcut])
            x = tf.keras.layers.Activation('relu')(x)
            
            shortcut = x
 
        else:
            x = tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            
            x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
 
            x = tf.keras.layers.Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            x = tf.keras.layers.BatchNormalization()(x)            
 
            x = tf.keras.layers.Add()([x, shortcut])   
            x = tf.keras.layers.Activation('relu')(x)  
 
            shortcut = x        
    
    return x
 
 
 
def conv3_layer(x):        
    shortcut = x    
    
    for i in range(4):     
        if(i == 0):            
            x = tf.keras.layers.Conv2D(128, (1, 1), strides=(2, 2), padding='valid')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)        
            
            x = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)  
 
            x = tf.keras.layers.Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = tf.keras.layers.Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = tf.keras.layers.BatchNormalization()(x)
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)            
 
            x = tf.keras.layers.Add()([x, shortcut])    
            x = tf.keras.layers.Activation('relu')(x)    
 
            shortcut = x              
        
        else:
            x = tf.keras.layers.Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            
            x = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
 
            x = tf.keras.layers.Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            x = tf.keras.layers.BatchNormalization()(x)            
 
            x = tf.keras.layers.Add()([x, shortcut])     
            x = tf.keras.layers.Activation('relu')(x)
 
            shortcut = x      
            
    return x


######################


if __name__ == "__main__" :

    # ap = argparse.ArgumentParser()
    # ap.add_argument('-i', '--image', required = True, help = 'Path to the input image')
    # args = vars(ap.parse_args())
    # filename = args['image']
    
    TRAIN_PATH = 'C:\\Users\\손현철\\Desktop\\hc\\사업\\태양광\\solar\\컬러\\'

    x_train = []
    y_train = []

    folder_list = os.listdir(TRAIN_PATH)
    folder_list.sort()

    word2index={}
    for voca in folder_list:
        if voca not in word2index.keys():
            word2index[voca]=len(word2index)
    print(word2index)

    for folder in folder_list:
        file_list = os.listdir(TRAIN_PATH + folder)
        for filename in file_list:
            filepath = TRAIN_PATH + folder + '\\' + filename
            image = imread(filepath)
            image.tolist()
            x_train.append(image)
            y_train.append(one_hot_encoding(folder, word2index))

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    inputs = tf.keras.Input(shape = (128,128,3))

    x = conv1_layer(inputs)
    x = conv2_layer(x)
    x = conv3_layer(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(len(folder_list), activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['poisson', 'accuracy'])

    hist = model.fit(x_train, y_train, epochs=150)

    model.save('solar_model')