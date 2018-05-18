# -*- coding: utf-8 -*-
"""
@author: Aravind Project
SYDE 675 CK Subset to Extract Features
"""
import numpy as np
import os
from keras import applications
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from keras import applications
from keras.models import Model
import matplotlib.pyplot as plt
    
base_model = applications.vgg19.VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

import glob
import imageio

curr_cwd = os.getcwd()
layers_to_extract = ["block1_pool","block2_pool","block3_pool","block4_pool","block5_pool","fc1"]
#To extract the features from the selected layer of VGG19 Net
for layer_num in range(0,6):
    model = Model(input=base_model.input, output=base_model.get_layer(layers_to_extract[layer_num]).output)
        
    for cls in range (1,9):
        img_count=0
        # Different Class labels
        classes = ["Anger","Dislike","Fear","Happy","Neutral","Sad","Surprised","Test"]
        #To store the Features
        feats=[]
        
        for image_path in glob.glob(curr_cwd+"\\"+classes[cls-1]+"\\"+"*.tiff"):
            img_count=img_count+1
            print(img_count)
            
            #Pre-processing
            img = image.load_img(image_path, target_size=(224, 224))
            x_in = image.img_to_array(img)
            x_in = np.expand_dims(x_in, axis=0)
            x_in = preprocess_input(x_in)
            
            #Feature Extraction
            features = model.predict(x_in)
            features = features.flatten()
            feats.append(features)
            features_arr = np.char.mod('%f', features)
        
        feature_list = np.squeeze(np.asarray(feats))
        labels = np.ones(len(feature_list))*cls
        feature_list = np.column_stack((feature_list,labels))
        #Save the features as numpy array for further processing
        np.save("class_jaffe"+layers_to_extract[layer_num]+str(cls)+"vgg19_data.npy",feature_list)
     

    #To Plot the block3_pool images
    
    '''
    w=20
    h=20
    fig=plt.figure(figsize=(20, 20))
    columns = np.round(np.sqrt(features.shape[3]))+1
    rows = np.round(np.sqrt(features.shape[3]))
    for i in range(1, features.shape[3]):
        img = features.squeeze()
        fig.add_subplot(rows, columns, i)
        fig.tight_layout()
        plt.imshow(img[:,:,i])
    plt.show()
    fig.savefig(layers_to_extract[layer_num]+'other.png')
'''