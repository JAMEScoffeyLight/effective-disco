# -*- coding: utf-8 -*-

#!pip install imageio #Возможно понадобится скачать

import pandas as pd
import numpy as np
import os

import fastbook
fastbook.setup_book()
import fastai

from fastbook import *
from fastai.vision.all import *
import torch
from pathlib import Path
from PIL import Image

def prediction_func(uploaded_image, fpath):
    
    #Uploaded image - имя файла c типом .png, директория только /input/siic-isic-224x224-images
    learneR = load_learner(fpath,cpu=True) 
    # fpath=f'/content/melanoma_detector_fold5.pkl', типа этого

    line = ''.join(list(uploaded_image)) #Загрузка названия изображения в таблицу
    line = line.rstrip(".png")    #Удаление типа файла из названия
    d={'image_name':[line],'patient_id':['IP_7425436'] 	,'sex':['n/mentioned'], 	'age_approx':['unknown'], 	'anatom_site_general_challenge':['body']}
    train_ims = pd.DataFrame(data=d) #Создания таблицы из подготовленного шаблона

    valid_preds = np.zeros((train_ims.shape[0],2))#массив переменных предсказания: первая переменная содержит вероятность меланомы на изображении, вторую не разобрал
    learn = learneR
    #Далее идут параметры загрузчика:
    b_tfms = [Normalize.from_stats(*imagenet_stats)]#приведение батча к нормализованному виду
    batch_size=8
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #Инициализация загрузчика
    dls = ImageDataLoaders.from_df(df = train_ims, #датафрейм, содержащий название изображения
                                   path = '/input/siic-isic-224x224-images',    #путь до изображения. файл должен загружаться в эту директорию
                                   suff = '.png',      #тип распознаваемого файла (также неизменяем)
                                   bs = batch_size,            #размер батча
                                   device=device,      #устройство выполнения предсказания
                                   batch_tfms =b_tfms) #функция преобразования батча (к нормализованному виду, например)

    #Процесс предсказания
    kag_dl = learn.dls.test_dl(train_ims)#Загрузка данных изображения (модель понимает только PNG файлы)
    preds, _ = learn.get_preds(dl=kag_dl, inner=False) #Предсказание 
    valid_preds += preds.numpy()#output данных
    return valid_preds
