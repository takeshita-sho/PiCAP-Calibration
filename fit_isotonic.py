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

loss_fn = 'focal_loss'
model_name = '/home/stakesh1/scr16-jgra21/scanner1/capsif2_repo/sho/models_DL/model-CAPSIF2_RES2_128_nlayer-3-3_knn10-20focal_loss_fin_0.pt'
TEST_PDB = "/home/stakesh1/scr16-jgra21/scanner1/capsif2_repo/dataset/final0_val_pdb.csv"
TEST_CLUSTER = "/home/stakesh1/scr16-jgra21/scanner1/capsif2_repo/dataset/final0_val_cluster_prune.csv"
print("loading Picap predictions")
#Load predictions
names, pred_probs = run_picap(TEST_PDB,TEST_CLUSTER,model_name)
print(names[:10])
print(names[-10:])
print("loading lables")
names = np.squeeze(names)
pred_probs = np.squeeze(pred_probs)
#Load labels
df = pd.read_csv(TEST_PDB,header=None)
for i in range(len(df[0].values)):
    if names[i] not in df.iloc[i,0]:
        print(df.iloc[i,0],names[i])
        df.drop([i],inplace=True)
        break
true_labels = df[5].values
print(df.tail())
        
print(len(true_labels))
print(len(pred_probs))

iso_reg = IsotonicRegression() 
print("fitting isotonic regression")
#Fit isotonic regression
iso_reg.fit(pred_probs, true_labels)
plt.plot(pred_probs, true_labels,'r|',markersize=1,label='True Labels vs PBind')
pred_probs = np.sort(pred_probs)
plt.plot(pred_probs, iso_reg.predict(pred_probs), 'g.-', markersize=12,label='Isotonic regression')
plt.title('Isotonic Regression of Uncalibrated Probabilities of Binding')
plt.xlabel('Uncalibrated Binding Probabilities')
plt.ylabel('True/Calibrated Probabilities')
plt.legend()
plt.tight_layout()
plt.savefig(f'./results/isotonic-regression.png')
print("saving isotonic regression")
with open(f'./models_DL/isotonic-{loss_fn}.pkl','wb') as f:
    pickle.dump(iso_reg,f)
print("done")
