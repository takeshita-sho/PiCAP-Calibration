import pandas as pd
from metrics import *

df_pred = pd.read_csv('/home/stakesh1/scr16-jgra21/scanner1/capsif2_repo/sho/results/isotonic-BCELoss.csv')
df_label = pd.read_csv('/home/stakesh1/scr16-jgra21/scanner1/capsif2_repo/dataset/final0_test_pdb.csv', header=None)
df_bayes = pd.read_csv('/home/stakesh1/scr16-jgra21/scanner1/capsif2_repo/sho/results/BNN-2.csv')
df_bayes_10 = pd.read_csv('/home/stakesh1/scr16-jgra21/scanner1/capsif2_repo/sho/results/BNN-10.csv')
df_bayes_50 = pd.read_csv('/home/stakesh1/scr16-jgra21/scanner1/capsif2_repo/sho/results/BNN-50.csv')
df_bayes_100 = pd.read_csv('/home/stakesh1/scr16-jgra21/scanner1/capsif2_repo/sho/results/BNN-100.csv')
# Truncate column 1 to first 3 characters
df_label[1] = df_label[1].astype(str).str[:3]

# Keep only columns 1 and 5
df_label = df_label[[1, 5]]
df_label.columns = ['name', 'label']  # rename for easier merge
print(df_label)
# 1) Merge on matching “name”
#df_merged = pd.merge(df_pred, df_label, on='name', how='inner')
df_label = df_label[df_label.name.isin(df_pred.name)]
df_pred['label'] = df_label['label']


labels = df_pred['label'].values
acc = pd.DataFrame(pd.DataFrame(index=[0]))
#print(df_pred['BCELoss'].values)
acc['BCELoss'] = accuracy(df_pred['BCELoss'].values, labels)
acc['FocalLoss'] = accuracy(df_pred['Focal Loss'].values, labels)
acc['Dice++Loss'] = accuracy(df_pred['Dice++Loss'].values, labels)
acc['IsotonicBCE'] = accuracy(df_pred['IsotonicBCE'].values, labels)
acc['IsotonicFocal'] = accuracy(df_pred['IsotonicFocal'].values, labels)
#print(df_bayes['BNN'].values)
acc['BNN_2'] = accuracy(df_bayes['BNN'].values, labels)
acc['BNN_10'] = accuracy(df_bayes_10['BNN'].values, labels)
acc['BNN_50'] = accuracy(df_bayes_50['BNN'].values, labels)
acc['BNN_100'] = accuracy(df_bayes_100['BNN'].values, labels)

bacc_df = pd.DataFrame(pd.DataFrame(index=[0]))
bacc_df['BCELoss'] = bacc(labels, df_pred['BCELoss'].values)
bacc_df['FocalLoss'] = bacc(labels, df_pred['Focal Loss'].values)
bacc_df['Dice++Loss'] = bacc(labels, df_pred['Dice++Loss'].values)
bacc_df['IsotonicBCE'] = bacc(labels, df_pred['IsotonicBCE'].values)
bacc_df['IsotonicFocal'] = bacc(labels, df_pred['IsotonicFocal'].values)
bacc_df['BNN_2'] = bacc(labels, df_bayes['BNN'].values)
bacc_df['BNN_10'] = bacc(labels, df_bayes_10['BNN'].values)
bacc_df['BNN_50'] = bacc(labels, df_bayes_50['BNN'].values)
bacc_df['BNN_100'] = bacc(labels, df_bayes_100['BNN'].values)
# 4) Now “df_pred” and “df_label” have exactly the same rows in the same order
draw_reliability_graph_marginal(df_pred['BCELoss'].values, labels, 'Vanilla Loss Binding Reliability')
draw_reliability_graph_marginal(df_pred['Focal Loss'].values, labels, 'Focal Loss Binding Reliability')
draw_reliability_graph_marginal(df_pred['Dice++Loss'].values, labels, 'Dice++ Loss Binding Reliability')
draw_reliability_graph_marginal(df_pred['IsotonicBCE'].values, labels, 'Isotonic Vanilla Binding Reliability')
draw_reliability_graph_marginal(df_pred['IsotonicFocal'].values, labels, 'Isotonic Focal Binding Reliability')
draw_reliability_graph_marginal(df_bayes['BNN'].values, labels, 'BNN n=2 Binding Reliability')
draw_reliability_graph_marginal(df_bayes_10['BNN'].values, labels, 'BNN n=10 Binding Reliability')
draw_reliability_graph_marginal(df_bayes_50['BNN'].values, labels, 'BNN n=50 Binding Reliability')
draw_reliability_graph_marginal(df_bayes_100['BNN'].values, labels, 'BNN n=100 Binding Reliability')

bacc_df.to_csv('./results/balanced_accuracy.csv', index=False)
acc.to_csv('./results/accuracy.csv', index=False)
print("done")
