import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import os
import imageio.v2 as imageio

sns.set_style('darkgrid')
plt.style.use('seaborn-notebook')

import fastbook
fastbook.setup_book()
import fastai

import random
random.seed(42)

from fastbook import *
from fastai.vision.all import *
import torch
from pathlib import Path
from PIL import Image

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import *
import statistics

#set device and ensure GPU is running
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


def First_step:
  #1. Load metadata .csv files

  #read in train csv
  train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

  #read in test csv
  test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv'


def Second_step:
  #Step 2: select all the malignant images, then downsample the benign category in remaining training data so n_malignant = n_benign for training

  #split train_ims into 5 sets using StratifiedKFold from sklearn
  #create a df with ALL of the malignant images
  mal_ims = train[train['target']==1]
  n_mal_ims = len(mal_ims)
  print("Number of malignant images: {}".format(n_mal_ims))
  mal_ims.head()

  #create a df of a subset of benevolent images
  ben_ims_subset = train[train['target']==0].sample(n=n_mal_ims, random_state=42)
  n_ben_ims = len(ben_ims_subset)
  print("Number of benign images in subset: {}".format(n_ben_ims))

  #concatenate the two together and check the result
  train_ims = pd.concat([mal_ims, ben_ims_subset])
  n_training_items = len(train_ims)
  print("Number of training items: {}".format(n_training_items))

  #add /train prefix and /test prefix to the respective dfs
  train_ims['image_name'] = 'train/' + train_ims['image_name'].astype(str)
  test['image_name'] = 'test/' + test['image_name'].astype(str)

  train_ims.reset_index(drop=True, inplace=True)


def Third_step:
  #Step 3: split off 20% of training data as a validation set,
  
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  X = train_ims['image_name'].copy()
  y = train_ims['target'].copy()
  fold = 0
  for train_index, test_index in skf.split(X, y):
      fold+= 1
      print('In fold',fold)
      print("TRAIN LENGTH:", len(train_index), "VALIDATION LENGTH:", len(test_index))
      train_ims[f'fold_{fold}_valid']=False
      train_ims.loc[test_index,f'fold_{fold}_valid']=True
      
  #Pull out 20% of entries to be held out as unseen test data for validation.
  valid_ims = train_ims[train_ims['fold_5_valid'] == True]

  valid_ims.reset_index(drop=True, inplace=True)


def Fourth_step:
  #Step 4: create a 5-fold split of what remains for cross-validation in model training

  #drop the test set from train_ims, then split the remaining images into 5 folds for cross-validation during model training
  train_ims = train_ims[train_ims['fold_5_valid'] == False]
  train_ims = train_ims[['image_name','patient_id','sex','age_approx','anatom_site_general_challenge','diagnosis','benign_malignant','target']]
  train_ims.reset_index(drop=True, inplace=True)

  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  X = train_ims['image_name'].copy()
  y = train_ims['target'].copy()
  fold = 0
  for train_index, test_index in skf.split(X, y):
      fold+= 1
      print('In fold',fold)
      print("TRAIN LENGTH:", len(train_index), "VALIDATION LENGTH:", len(test_index))
      train_ims[f'fold_{fold}_valid']=False
      train_ims.loc[test_index,f'fold_{fold}_valid']=True

def dataloader(fold, bs=8, b_tfms=Normalize.from_stats(*imagenet_stats)):
  dls = ImageDataLoaders.from_df(df = train_ims, #specify df holding image names
                                 path = '../input/siic-isic-224x224-images',    #set path for where to find images
                                 suff = '.png',      #add the .png suffix to file names from df
                                 label_col = 'target',      
                                 bs = bs,            #set batch size
                                 device=device,      #set device
                                 batch_tfms = b_tfms,
                                 valid_col = f'fold_{fold}_valid')
  return dls
          
def Learner
    #instantiate metrics
  rocAucBinary = RocAucBinary()
  recall = Recall()
  precision = Precision()

  #instantiate arrays to hold probability results
  valid_preds = np.zeros((valid_ims.shape[0],2))
  kaggle_preds = np.zeros((test.shape[0],2))

  #choose batch transforms, batch size, and the name of 
  b_tfms = [#*aug_transforms(do_flip=True, flip_vert=True, max_rotate=45.0, max_zoom=1.1, size=224, max_lighting=0.2, max_warp=0.4, p_affine=0.75, p_lighting=0.75, xtra_tfms=None, mode='bilinear'),
            Normalize.from_stats(*imagenet_stats)]
  batch_size=8

  for fold in range(1,6):
      dls=dataloader(fold, batch_size, b_tfms)
      print(f'Fold {fold}:')
      learn = vision_learner(dls,                  #specify dataloader object
                         models.resnet34,             #specify a pre-trained model we want to build off of
                         metrics=[rocAucBinary, error_rate, precision, recall], #specify metrics we want to see
                         model_dir = '/kaggle/working')   #specify output location to store model

      learn.fine_tune(15,                               #set number of epochs 
                   #base_lr=valley,                  #set the initial learning rate
                   cbs=[SaveModelCallback(            #use the SaveModelCallback to save the best model
                       monitor='roc_auc_score',      #set the roc_auc_score as the montitored metric
                       fname = f'resnet34_rocauc_fold{fold}',       #choose the name the best model will be saved under
                       comp = np.greater,            #specify that when the roc_auc_score increases, that's considered better
                       with_opt=True),               #saves optimizer state, if available, when saving model
                      ReduceLROnPlateau(monitor = 'roc_auc_score', comp = np.greater, patience=2)])
                     # EarlyStoppingCallback(monitor = 'roc_auc_score', comp=np.greater, patience=6)])

      learn.load(f'resnet34_rocauc_fold{fold}', 
                 device=device,     #ensure the loaded model uses the active cuda:0 device
                 with_opt=True,     #load optimizer state
                 strict=True)

      test_dl = learn.dls.test_dl(valid_ims)
      preds, _ = learn.tta(dl=test_dl) 
      valid_preds += preds.numpy()

      kag_dl = learn.dls.test_dl(test)
      preds, _ = learn.get_preds(dl=kag_dl) 
      kaggle_preds += preds.numpy()

      print(f'Prediction completed in fold: {fold}')

      fold+=1

def saving_preds
  ss = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv'
  #Save kaggle predictions to csv for submission
  sub_file_name="submission_resnet34_Oct172022_rocauc_opt.csv"
  submission = pd.DataFrame({'image_name':ss['image_name'], 'target':list(kaggle_preds[:,1])})
  submission.to_csv(sub_file_name, index=False)
                   
def exporting_model
  for fold in range(1,6):
    learn.load(f'resnet34_rocauc_fold{fold}', 
               device=device,     #ensure the loaded model uses the active cuda:0 device
               with_opt=True,     #load optimizer state
               strict=True)
    learn.export(f'/kaggle/working/melanoma_detector_fold{fold}.pkl')
                     
int main{
  print("Hello! Choose command for starting:\n Data preparing, 1 step\n Data preparing, 2 step\n Data preparing, 3 step\n
        Data preparing, 4 step\n Learning model\n Exporting model\n Saving predictions\n")
  commands = {
            'Data preparing, 1 step':  First_step,
            'Data preparing, 2 step':  Second_step,
            'Data preparing, 3 step':  Third_step,
            'Data preparing, 4 step':  Fourth_step,
            'Learning model':     Learner,
            'Exporting model':  exporting_model,
            'Saving predictions':  saving_preds,
            }
  #check for any overlap between the train_ims df and valid_ims df

  idx1 = pd.Index(train_ims['image_name'])
  idx2 = pd.Index(valid_ims['image_name'])
  idx1.intersection(idx2)
}
