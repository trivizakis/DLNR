#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Eleftherios Trivizakis
@github: https://github.com/trivizakis
"""
# import argparse
import os
import numpy as np
import SimpleITK as sitk

from model_factory import DnCNN
from hypes_handler import HypesHandler as hh

import tensorflow as tf

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
#from skimage.io import imsave

import mlflow

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

os.environ['MLFLOW_TRACKING_URI'] = 'https://pcai-flow.duckdns.org/tracking/'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://139.91.210.36:9000/forth.ioan/'
os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'admin8888'
mlflow.set_tracking_uri("https://pcai-flow.duckdns.org/tracking/")

name="DL_Denoising"
mlflow.set_experiment(name)

def hypes_generator(input_shape):
    #DEEP LEARNING HYPE-PARAMS
    params={}
    params["batch_size"] = 1
    params["kernels"] = 64
    params["layers"] = 15
    params["lr"] = 0.005
    params["residual"] = True
    params["input_shape"]=input_shape
    params["regularizer"]= "l1_l2"
    params["reg_value"]= [1e-2,1e-2]
    params["kernel_initializer"]= "Orthogonal"
    params["loss"] = "SDI"
    params["optimizer"] = "Adam"

    params = hh.add_loss_hypes(params)
    params = hh.add_optimizer_hypes(params)
    params = hh.add_regularizer_hypes(params)
    return params
    
def denoiser():  
    input_dir = './input_data'
    output_dir = './output_data'

    for root, dirs, files in os.walk(input_dir+"/"):
        for file in files:
            #report to mlflow
            with mlflow.start_run(run_name=name+" - Volume ID: "+file,nested=True):
                
                reader = sitk.ImageFileReader()
                reader.SetImageIO("NiftiImageIO")
                reader.SetFileName(input_dir+"/"+file)
                reader.LoadPrivateTagsOn()
                reader.ReadImageInformation()  
                nifti_volume = reader.Execute()
                
                #get metadata          
                metadata={}
                for key in nifti_volume.GetMetaDataKeys():
                    metadata[key] = nifti_volume.GetMetaData(key)
                
                noisy_volume = sitk.GetArrayFromImage(nifti_volume)
                
                var_type = noisy_volume.dtype
                shape = noisy_volume.shape
                maximum = noisy_volume.max()
                minimum = noisy_volume.min()
                
                if shape[1] == shape[2] and shape[0]<shape[1]:
                    print("Preparing denoising")
                elif shape[0] == shape[1] and shape[2]<shape[1]:
                    noisy_volume = np.transpose(noisy_volume, (2, 0, 1))
                    print("Preparing denoising.")
                
                #initiate parameters for the deep learning architecture
                params = hypes_generator(input_shape=tuple((shape[1],shape[2],1)))
                
                #initiate the model's arcitecture
                dnconv = DnCNN(params)
                
                #compile the model
                dnconv.compile(loss=params["loss"],
                               optimizer=params["optimizer"])
                
                #load weights
                dnconv.load_weights("denoiser.h5")
                
                #freeze model
                for layer in dnconv.layers:
                    layer.trainable = False
                
                
                print("Initiating denoising for volume: "+file)
                #inference for nifti volume
                denoised_volume = []
                scalar_loss = []
                for pos, noisy_image in enumerate(noisy_volume):        
                    img = np.expand_dims(np.expand_dims(((noisy_image-minimum)/(maximum-minimum)), axis=0), axis=3)
                    clean_prd = dnconv.predict(img, batch_size=1)
                    scalar_loss.append(dnconv.test_on_batch(img))
                    clean_prd = np.squeeze(clean_prd,axis=3)
                    clean_img = (np.squeeze(clean_prd,axis=0)*maximum).astype(var_type)
                    denoised_volume.append(clean_img)
                    
                    #print-to-png
                    # clean_img_print = (np.squeeze(clean_prd,axis=0)*255.0).astype(np.uint8)
                    # imsave(output_dir+"/"+file[:13]+"_"+str(pos)+"_cleaned.png", clean_img_print)                        
                    
                tf.keras.backend.clear_session()
                
                #list to numpy array
                denoised_volume = np.stack(denoised_volume, axis=0)
                
                PSNR = psnr(denoised_volume, noisy_volume)
                SSIM = ssim(denoised_volume, noisy_volume)
                MSE = mse(denoised_volume, noisy_volume)
                                
                mlflow.log_metric("Scalar loss", np.array(scalar_loss).mean())
                mlflow.log_metric("Differences Index", 1-SSIM)
                mlflow.log_metric("PSNR", PSNR)
                mlflow.log_metric("MSE", MSE)      
                
                print("Completed volume denoising: "+file)
                
                final_nifti = sitk.GetImageFromArray(denoised_volume)
            
                
                #set metadata to denoised image (nifti)
                for key in list(metadata.keys()):
                    final_nifti.SetMetaData(key, metadata[key])
                    
                writer = sitk.ImageFileWriter()
                writer.SetImageIO("NiftiImageIO")
                writer.SetFileName(output_dir+"/"+file)
                writer.Execute(final_nifti)

if __name__ == '__main__':

    with mlflow.start_run(run_name=name):
        tags = {"Author":"FORTH","Release Version": "1.0","Repository":"Not publicly available"}
        mlflow.set_tags(tags)
        
        denoiser() 