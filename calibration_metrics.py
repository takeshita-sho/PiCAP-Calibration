import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
barWidth = 0.15
fig = plt.subplots(figsize =(14, 8)) 

Vanilla = [5.84,57.91,32.79] 
dicepp = [3.74,49.80,25.65] 
focal = [7.21,32.74,21.36] 
isotonic = [3.24,19.21,10.38]
bnn = [2.37,19.19,14.19]

br1 = np.arange(len(Vanilla)) 
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2] 
br4 = [x + barWidth for x in br3]
br5 = [x + barWidth for x in br4]
plt.bar(br1, Vanilla, color ='skyblue', width = barWidth, 
        edgecolor ='black', label ='Vanilla',linewidth=5) 
plt.bar(br2, dicepp, color ='lightcoral', width = barWidth, 
        edgecolor ='black', label ='Dice++',linewidth=5) 
plt.bar(br3, focal, color ='lightgreen', width = barWidth, 
        edgecolor ='black', label ='Focal',linewidth=5)
plt.bar(br4, isotonic, color ='gold', width = barWidth,
        edgecolor ='black', label ='Isotonic',linewidth=5) 
plt.bar(br5, bnn, color ='#E5E2E0', width = barWidth,
        edgecolor ='black', label ='BNN',linewidth=5)

plt.xlabel('') 
plt.ylabel('% Error', fontsize = 30) 
plt.yticks(fontsize=20)
plt.xticks([r + barWidth*1.5 for r in range(len(Vanilla))], 
        ['ECE', 'MCE', 'CECE'],fontsize=30)
plt.legend(fontsize=30)
sns.despine()
plt.tight_layout()
plt.savefig('./results/error.svg')
plt.clf()
fig = plt.figure(figsize =(14, 8)) 
Vanilla = [87,86] 
dicepp = [89,88] 
focal = [90,90] 
isotonic = [89,89]
bnn = [89,89]

br1 = np.arange(len(Vanilla)) 
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2] 
br4 = [x + barWidth for x in br3]
br5 = [x + barWidth for x in br4]

plt.bar(br1, Vanilla, color ='skyblue', width = barWidth, 
        edgecolor ='black', label ='Vanilla',linewidth=5) 
plt.bar(br2, dicepp, color ='lightcoral', width = barWidth, 
        edgecolor ='black', label ='Dice++',linewidth=5) 
plt.bar(br3, focal, color ='lightgreen', width = barWidth, 
        edgecolor ='black', label ='Focal',linewidth=5)
plt.bar(br4, isotonic, color ='gold', width = barWidth,
        edgecolor ='black', label ='Isotonic',linewidth=5)
plt.bar(br5, bnn, color ='#E5E2E0', width = barWidth,
        edgecolor ='black', label ='BNN',linewidth=5)

plt.xlabel('') 
plt.ylabel('% Accuracy', fontsize = 30) 
plt.yticks(fontsize=20)
plt.xticks([r + barWidth*1.5 for r in range(len(Vanilla))], 
        ['Accuracy', 'Balanced Accuracy'],fontsize=30)
#fig.legend(fontsize=30,bbox_to_anchor=(1.2, 1), loc="upper right")
sns.despine()
plt.legend(fontsize=30,loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig('./results/accuracies.svg')