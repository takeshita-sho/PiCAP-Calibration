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

SPECIES = 'human_pre_high'
TEST_PDB =   '/home/stakesh1/scr16-jgra21/scanner1/capsif2_repo/af2_datasets/' + SPECIES + '/dataset_pdb.csv'
TEST_CLUSTER = '/home/stakesh1/scr16-jgra21/scanner1/capsif2_repo/af2_datasets/' + SPECIES + '/dataset_clust.csv'
model_name = '/home/stakesh1/scr16-jgra21/scanner1/capsif2_repo/sho/models_DL/model-CAPSIF2_RES2_128_nlayer-3-3_knn10-20BCELoss_fin_2.pt'
print("loading Picap predictions")
#Load predictions
names, bce_probs = run_picap(TEST_PDB,TEST_CLUSTER,model_name)
with open(f'./models_DL/isotonic-BCELoss.pkl', 'rb') as f:
    iso_bce = pickle.load(f)

iso_bce_p = iso_bce.predict(bce_probs)

df = pd.DataFrame({'name': np.squeeze(names), 'BCELoss': np.squeeze(bce_probs),
                    'IsotonicBCE': np.squeeze(iso_bce_p)})
df.dropna(inplace=True)


df.to_csv("./results/isotonic-proteome.csv",index=False)