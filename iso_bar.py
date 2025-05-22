import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('/home/stakesh1/scr16-jgra21/scanner1/capsif2_repo/sho/results/isotonic-BCELoss.csv')
bayes_df = pd.read_csv('/home/stakesh1/scr16-jgra21/scanner1/capsif2_repo/sho/results/BNN-100.csv')
# Create 4 subplots (2 rows x 2 columns)
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
b = 5
# Plot BCELoss distribution
sns.histplot(data=df, x='BCELoss', binrange=(0, 1), bins=b, color='red', ax=axes[0, 0])
axes[0, 0].set_title('Vanilla Distribution',size=25)
axes[0, 0].set_xlabel(None)
axes[0, 0].set_ylabel(None)

# Plot Isotonic distribution
sns.histplot(data=df, x='IsotonicBCE', binrange=(0, 1), bins=b, color='blue', ax=axes[0, 1])
axes[0, 1].set_title('Isotonic Vanilla Distribution',size=25)
axes[0, 1].set_xlabel(None)
axes[0, 1].set_ylabel(None)

# Plot Focal Loss distribution
sns.histplot(data=df, x='Focal Loss', binrange=(0, 1), bins=b, color='green', ax=axes[0, 2])
axes[0, 2].set_title('Focal Loss Distribution',size=25)
axes[0, 2].set_xlabel(None)
axes[0, 2].set_ylabel(None)

# Plot Dice++Loss distribution
sns.histplot(data=df, x='Dice++Loss', binrange=(0, 1), bins=b, color='purple', ax=axes[1, 0])
axes[1, 0].set_title('Dice++ Loss Distribution',size=25)
axes[1, 0].set_xlabel(None)
axes[1, 0].set_ylabel(None)


sns.histplot(data=bayes_df, x='BNN', binrange=(0, 1), bins=b, color='yellow', ax=axes[1, 1])
axes[1, 1].set_title('BNN n=100 Distribution',size=25)
axes[1, 1].set_xlabel(None)
axes[1, 1].set_ylabel(None)
fig.supxlabel('Probability',size=30)
fig.supylabel('Count',size=30)

#plt.legend()
plt.tight_layout()
plt.gcf().savefig('./results/isotonic-BCELoss-bind-notbind.svg')
