"""It is highly desirable to use image tiles that contain .prj and .tfw (if it's tiff images), and to overwrite the existing image tiles so that the following files can be used for georeferencing. Make sure that a backup of the original image tiles is made. Another variant would be to rename the accompanying files in accordance with the suffix of the tested image tiles (eg. *_segmented.tiff)"""

import cv2
import numpy as np

from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
from PIL import Image
import segmentation_models as sm

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

from smooth_tiled_predictions import predict_img_with_smooth_windowing
import glob


#Navigate to folder where your image tiles are located
files_images=glob.glob("./New folder/*.tif")

for file in files_images:
    
    BACKBONE = 'resnet34'
    preprocess_input = sm.get_preprocessing(BACKBONE)
    
    img = cv2.imread(file)  #N-34-66-C-c-4-3.tif, N-34-97-D-c-2-4.tif
    input_img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
    input_img = preprocess_input(input_img)
    
    from keras.models import load_model

    #Following line defines the deep learning model which should be located in the same directory as this code
    model = load_model("mini_inria_60_epochs_RESNET_backbone_batch16.hdf5", compile=False)
                      
    # size of patches
    patch_size = 256
    
    # Number of classes 
    n_classes = 2
    
             
    ###################################################################################
    #Predict using smooth blending
    
    # Use the algorithm. The `pred_func` is passed and will process all the image 8-fold by tiling small patches with overlap, called once with all those image as a batch outer dimension.
    # Note that model.predict(...) accepts a 4D tensor of shape (batch, x, y, nb_channels), such as a Keras model.
    predictions_smooth = predict_img_with_smooth_windowing(
        input_img,
        window_size=patch_size,
        subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
        nb_classes=n_classes,
        pred_func=(
            lambda img_batch_subdiv: model.predict((img_batch_subdiv))
        )
    )
    
    
    final_prediction = np.argmax(predictions_smooth, axis=2)
    
    
    #Navigate to folder where your want to save image prediction tiles 
    cv2.imwrite("D:/Struka/KUCE/New_folder/"+file[-15:-4]+"_segmented.tif",final_prediction)
    ###################
    
    plt.figure(figsize=(12, 12))
    plt.subplot(221)
    plt.title('Testing Image')
    plt.imshow(img)
    plt.subplot(222)
    plt.title('Testing Label')
    #plt.imshow(original_mask)
    plt.subplot(223)
    plt.title('Prediction with smooth blending')
    plt.imshow(final_prediction)
    plt.show()
    