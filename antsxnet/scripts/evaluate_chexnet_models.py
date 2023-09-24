import ants
import antspynet
import deepsimlr
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import sklearn.metrics as sklm
import glob

base_directory = '/Users/ntustison/Data/Public/Chexnet/reproduce-chexnet/'
image_directory = base_directory + "data/nifti/"

output_directory = base_directory + "antsxnet/scripts/testing_data_predictions/"

disease_categories = ['Atelectasis',
                        'Cardiomegaly',
                        'Effusion',
                        'Infiltration',
                        'Mass',
                        'Nodule',
                        'Pneumonia',
                        'Pneumothorax',
                        'Consolidation',
                        'Edema',
                        'Emphysema',
                        'Fibrosis',
                        'Pleural_Thickening',
                        'Hernia']
number_of_dx = len(disease_categories)

pytorch_output_file = output_directory + "chexnet_pytorch_predictions.npy"
keras_output_file = output_directory + "chexnet_keras_predictions.npy"
antsxnet_output_file = output_directory + "chexnet_antsxnet_predictions.npy"
true_diagnoses_file = output_directory + "chexnet_true_diagnoses.npy"

if not os.path.exists(pytorch_output_file):
    print("Pytorch file does not exist.")
if not os.path.exists(keras_output_file):
    print("Keras file does not exist.")
if not os.path.exists(antsxnet_output_file):
    print("ANTsXNet file does not exist.")
if not os.path.exists(true_diagnoses_file):
    print("True dx file does not exist.")

if not os.path.exists(pytorch_output_file) and \
   not os.path.exists(keras_output_file) and \
   not os.path.exists(antsxnet_output_file) and \
   not os.path.exists(true_diagnoses_file):
    
    demo = pd.read_csv(base_directory + "nih_labels.csv", index_col=0)
    test_demo = demo.loc[demo['fold'] == 'test']

    pytorch_chexnet_predictions = np.zeros((test_demo.shape[0], number_of_dx))
    keras_chexnet_predictions = np.zeros((test_demo.shape[0], number_of_dx))
    antsxnet_chexnet_predictions = np.zeros((test_demo.shape[0], number_of_dx))
    true_diagnoses = np.zeros((test_demo.shape[0], number_of_dx))

    for i in range(test_demo.shape[0]):
        print(i, " out of ", test_demo.shape[0])    
        subject_row = demo.iloc[[i]]
        base_image_file = subject_row.index.values[0]
        base_image_file = base_image_file.replace(".png", ".nii.gz")
        image_file = glob.glob(base_directory + "/../Nifti/*/" + base_image_file)[0]    
        mask_file = image_file.replace("Nifti", "Masks")
        if not os.path.exists(image_file) or not os.path.exists(mask_file):
            print(image_file, " does not exist.")
            continue
        image = ants.image_read(image_file)
        mask = ants.image_read(mask_file)

        cxr_pytorch = deepsimlr.chexnet(image, verbose=False)      
        cxr_keras = antspynet.chexnet(image, use_antsxnet_variant=False, verbose=False)      
        cxr_antsxnet = antspynet.chexnet(image, lung_mask=mask, use_antsxnet_variant=True, verbose=False)      

        for d in range(number_of_dx):
            pytorch_chexnet_predictions[i,d] = cxr_pytorch[disease_categories[d]]
            keras_chexnet_predictions[i,d] = cxr_keras[disease_categories[d]]
            antsxnet_chexnet_predictions[i,d] = cxr_antsxnet[disease_categories[d]]
            true_diagnoses[i,d] = subject_row[disease_categories[d]]

        if i > 0 and i % 1000 == 0:        
            pytorch_auc = np.zeros((number_of_dx,))
            keras_auc = np.zeros((number_of_dx,))
            antsxnet_auc = np.zeros((number_of_dx,))
            for j in range(number_of_dx):
                pytorch_auc[j] = sklm.roc_auc_score(true_diagnoses[:i,j], pytorch_chexnet_predictions[:i,j])            
                keras_auc[j] = sklm.roc_auc_score(true_diagnoses[:i,j], keras_chexnet_predictions[:i,j])            
                antsxnet_auc[j] = sklm.roc_auc_score(true_diagnoses[:i,j], antsxnet_chexnet_predictions[:i,j])            
            print("Pytorch:  ", pytorch_auc)
            print("Keras:  ", keras_auc)
            print("ANTsXNet:  ", antsxnet_auc)

    np.save(pytorch_output_file, pytorch_chexnet_predictions)
    np.save(keras_output_file, keras_chexnet_predictions)
    np.save(antsxnet_output_file, antsxnet_chexnet_predictions)
    np.save(true_diagnoses_file, true_diagnoses)

else:
    pytorch_chexnet_predictions = pd.DataFrame(np.load(pytorch_output_file),
                                               columns=disease_categories)  
    keras_chexnet_predictions = pd.DataFrame(np.load(keras_output_file),
                                             columns=disease_categories)
    antsxnet_chexnet_predictions = pd.DataFrame(np.load(antsxnet_output_file),
                                                columns=disease_categories)
    true_diagnoses = pd.DataFrame(np.load(true_diagnoses_file),
                                          columns=disease_categories)
    
    auc_scores = np.zeros((3, number_of_dx))
    for j in range(number_of_dx):
        auc_scores[0,j] = sklm.roc_auc_score(true_diagnoses[disease_categories[j]], 
                                             pytorch_chexnet_predictions[disease_categories[j]])            
        auc_scores[1,j] = sklm.roc_auc_score(true_diagnoses[disease_categories[j]], 
                                             keras_chexnet_predictions[disease_categories[j]])            
        auc_scores[2,j] = sklm.roc_auc_score(true_diagnoses[disease_categories[j]], 
                                             antsxnet_chexnet_predictions[disease_categories[j]]) 
    auc_scores = pd.DataFrame(auc_scores, columns=disease_categories, index=['Pytorch', 'Keras', 'ANTsXNet'])               
    auc_scores = auc_scores.reindex(sorted(auc_scores.columns), axis=1)  
    print(auc_scores.transpose())

