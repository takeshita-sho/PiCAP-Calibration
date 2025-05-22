from sklearn.isotonic import IsotonicRegression
import pickle
import os
import numpy as np
import pandas as pd
from utils import *
from picap import *
from egnn.egnn import *
import matplotlib.pyplot as plt
from torchvision.models.feature_extraction import create_feature_extractor
import re
import json


model_name = '/home/stakesh1/scr16-jgra21/scanner1/capsif2_repo/sho/models_DL/model-CAPSIF2_RES2_128_nlayer-3-3_knn10-20BCELoss_fin_2.pt'
TEST_PDB = "/home/stakesh1/scr16-jgra21/scanner1/capsif2_repo/dataset/final0_test_pdb.csv"
TEST_CLUSTER = "/home/stakesh1/scr16-jgra21/scanner1/capsif2_repo/dataset/final0_test_cluster_prune.csv"
print("loading Picap predictions")
#Load predictions
names, bce_probs = run_picap(TEST_PDB,TEST_CLUSTER,model_name)
names2, focal_probs = run_picap(TEST_PDB,TEST_CLUSTER,"/home/stakesh1/scr16-jgra21/scanner1/capsif2_repo/sho/models_DL/model-CAPSIF2_RES2_128_nlayer-3-3_knn10-20focal_loss_fin_0.pt")
names3, dice_pp_probs = run_picap(TEST_PDB,TEST_CLUSTER,"/home/stakesh1/scr16-jgra21/scanner1/capsif2_repo/sho/models_DL/model-CAPSIF2_RES2_128_nlayer-3-3_knn10-20dice_pp_lossBCELoss_fin_0.pt")
with open(f'./models_DL/isotonic-BCELoss.pkl', 'rb') as f:
    iso_bce = pickle.load(f)

iso_bce_p = iso_bce.predict(bce_probs)

with open(f'./models_DL/isotonic-focal_loss.pkl', 'rb') as f:
    iso_focal = pickle.load(f)

iso_focal_p = iso_focal.predict(focal_probs)
df = pd.DataFrame({'name': np.squeeze(names), 'BCELoss': np.squeeze(bce_probs),
                    'IsotonicBCE': np.squeeze(iso_bce_p),'Focal Loss': np.squeeze(focal_probs),'IsotonicFocal': np.squeeze(iso_focal_p),'Dice++Loss': np.squeeze(dice_pp_probs)})
df.dropna(inplace=True)


df.to_csv("./results/isotonic-BCELoss.csv",index=False)
