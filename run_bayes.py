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
import sys

n = int(sys.argv[1])
print(n)
model_name = '/home/stakesh1/scr16-jgra21/scanner1/capsif2_repo/sho/models_DL/model-CAPSIF2_RES2_128_nlayer-3-3_knn10-20elbo_loss_2elbo_loss_fin_1.pt'
TEST_PDB = "/home/stakesh1/scr16-jgra21/scanner1/capsif2_repo/dataset/final0_test_pdb.csv"
TEST_CLUSTER = "/home/stakesh1/scr16-jgra21/scanner1/capsif2_repo/dataset/final0_test_cluster_prune.csv"
print("loading Picap predictions")
#Load predictions
output_mc = []
for i in range(n):
    names, probs = run_picapb(TEST_PDB,TEST_CLUSTER,model_name)
    output_mc.append(probs)
output = np.stack(output_mc)  
pred_mean = output.mean(axis=0)
#y_pred = np.argmax(pred_mean, axis=-1)

df = pd.DataFrame({'name': np.squeeze(names), 'BNN': np.squeeze(pred_mean)})
df.dropna(inplace=True)

df.to_csv(f"./results/BNN-{n}.csv",index=False)
