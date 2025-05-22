import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

df = pd.read_excel('/home/stakesh1/scr16-jgra21/scanner1/capsif2_repo/sho/results/proteome_results.xlsx', sheet_name='hsapiens')
plt.figure(figsize=(6, 3))
sns.histplot(data=df,x='PiCAP_Probability',binrange=(0, 1),bins=10,edgecolor=None,color='orange')
plt.title('Human Proteome')
f = 13
plt.xlabel('Mean predicted probabilities', fontsize=f)
plt.ylabel('Count',fontsize=f)
plt.xticks(fontsize=f)
plt.yticks(fontsize=f)
plt.tight_layout()
plt.savefig('./results/proteome.svg')
plt.clf()
df = pd.read_csv('/home/stakesh1/scr16-jgra21/scanner1/capsif2_repo/sho/results/isotonic-proteome.csv')
plt.figure(figsize=(6, 3))
sns.histplot(data=df,x='IsotonicBCE',binrange=(0, 1),bins=10,edgecolor=None,color='lightcoral')
plt.title('Calibrated Human Proteome')
f = 13
plt.xlabel('Mean calibrated probabilities', fontsize=f)
plt.ylabel('Count',fontsize=f)
plt.xticks(fontsize=f)
plt.yticks(fontsize=f)
plt.tight_layout()
plt.savefig('./results/calibrated-proteome.svg')