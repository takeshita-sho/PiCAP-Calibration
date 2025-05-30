#init(" ".join(options.split('\n')))
import os
import numpy as np
import pandas as pd
from utils import *
from egnn.egnn import *
from utils_model import *
#import torch
import matplotlib.pyplot as plt
#from torchsummary import summary
#from torchvision.models.feature_extraction import create_feature_extractor
#from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import create_feature_extractor

import os

#SPECIES = 'dekosky'
#SPECIES = 'test_set'
SPECIES = 'yeast_pre_high'
TEST_PDB =   '../af2_datasets/' + SPECIES + '/dataset_pdb.csv'
TEST_CLUST = '../af2_datasets/' + SPECIES + '/dataset_clust.csv'

#TEST_CLUST = '../dataset/final0_train_cluster_prune.csv'
#TEST_PDB = '../dataset/final0_train_pdb.csv'
#TEST_CLUST = './test_datasets/final0_test_cluster_prune.csv'


BATCH_SIZE = 1;
FAKE_BATCH_SIZE = 1;
NUM_WORKERS = 0;
NUM_EPOCHS = 1000;

#Hyper parameters!
#LOSS_FN = dice_ent_loss
#loss_str = "_loss-dbce"
LOSS_FN = nn.BCELoss()
KNN = [16,16,16,16]
N_LAYERS = [3,3,3,3]
HIDDEN_NF = 128
PAD_SIZE = 2500
LEARNING_RATE = 2e-5
W_DECAY = 1e-6
ADAPOOL_SIZE = (150,HIDDEN_NF)
loss_str = ''
NUM = '_0'
CUTOFF = 0.001


DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
    NUM_WORKERS = 8;
print("Using: " + DEVICE)
DEVICE = torch.device(DEVICE)
print(DEVICE)

#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#Load Test Dataset
test_loader = get_test_loader(TEST_CLUST,TEST_PDB,root_dir="../af2_datasets/",train=0,
#test_loader = get_test_loader(TEST_CLUST,TEST_PDB,root_dir="../",train=0,
                                batch_size=1,num_workers=NUM_WORKERS,knn=KNN, return_pdb_ref=True)


#model = CAPSIF2_TWO(hidden_nf = HIDDEN_NF, n_layers=N_LAYERS,
#                device=DEVICE).to(DEVICE)
#my_model_name = model.get_string_name() + "_knn" + str(KNN[0]) + "_fakeBatch" + str(FAKE_BATCH_SIZE) + "_2CAP_[0.75, 0.5]-100"


model = CAPSIF2_RES2(hidden_nf = HIDDEN_NF, n_layers=N_LAYERS,normalize=False,
                device=DEVICE).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
scaler = torch.cuda.amp.GradScaler()
torch.autograd.set_detect_anomaly(True)
model.train()
print("Model loaded")
#my_model_name = model.get_string_name() + "_knn" + str(KNN[0]) #+ '_coef_' + str(MY_COEF[0][0]) + "-" + str(MY_LOSS_EPOCHS) + "_all"

my_model_name = model.get_string_name() + "_knn" + str(KNN[0]) + '-' + str(KNN[1]) + "_normFalse_" + loss_str + "_carb_fin" + NUM

print(my_model_name)

if DEVICE == 'cuda':
    checkpoint = torch.load("./models_DL/model-" + my_model_name + ".pt")
else:
    checkpoint = torch.load("./models_DL/model-" + my_model_name + ".pt",
        map_location=torch.device('cpu') )
model.load_state_dict(checkpoint['model_state_dict'])
print(checkpoint['info'])

model.eval()
print("Model loaded")


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

#model.ada_pool.register_forward_hook(get_activation('ada_pool'))
#model.conv_prot1[3].register_forward_hook(get_activation('GELU'))
#model.mlp_prot1[0].register_forward_hook(get_activation('Linear'))
#model.mlp_prot2[3].register_forward_hook(get_activation('GELU'))


#get the outputs for this guy to compute accuracy and everything
def model_test_prot_env(loader, model, DEVICE='cpu'):
    loop = tqdm(loader)
    #print("HIIIIIIIII")
    pred_res, res_label = [], [];
    res_pred, res_label = [],[];
    names = []


    mlp_leak2 = []

    n_stuff = 0

    model.eval()
    for batch_idx, (node_feat, coor, edges, edge_feat, carb_binder, sm_binder, label_res, n_res, n_edge, name, ref_pdb) in enumerate(loop):
        #print(node_feat,coor)

        with torch.no_grad():

            #nodes = torch.ones(len(coor),1)
            #nodes = nodes.to(device=DEVICE,dtype=torch.int)
            coor = coor.to(device=DEVICE,dtype=torch.float).squeeze()

            #print(coor.shape,nodes.shape)
            #exit the fail_state
            if len(coor.shape) < 2:
                #print('skip')
                continue;

            node_feat = node_feat.to(device=DEVICE,dtype=torch.float).squeeze()
            #label_res = label_res.to(device=DEVICE,dtype=torch.float).squeeze()
            #label_prot = carb_binder.to(device=DEVICE,dtype=torch.float).squeeze()
            #edges = edges.to(device=DEVICE,dtype=torch.int)
            #exit the fail_state
            if len(coor.shape) < 2:
                continue;

            #print(label_res.size()[0],node_feat.size()[0])

            #DOESNT WORK: summary(model,[node_feat, coor, edges, edge_feat])
            #summary(model)
            pred = model(node_feat, coor, edges, edge_feat,
                            is_batch=False, n_res=n_res, n_edge=n_edge)
            #print("PREDRES SIZE:",pred_res.size())
            #print(activation)

            
            pred_res.append(pred.detach().cpu().numpy())
            c_p = pred.detach().cpu().numpy().reshape(-1)
            #print(c_p)
            #print(pred.detach().cpu().numpy() > CUTOFF)
            #print(ref_pdb)
            c_res = []
            for kk in range(len(c_p)):
                #print(c_p[kk])
                if c_p[kk] > CUTOFF:
                    c_res.append(ref_pdb[kk])
                
            res_label.append( c_res )
            #prot_label.append(label_prot.detach().cpu().numpy())
            names.append(name)

            #ada.append(activation['ada_pool'].cpu().detach().numpy())
            #final_conv_batch.append(activation['GELU'].cpu().detach().numpy())
            #mlp_lin1.append(activation['Linear'].detach().cpu().numpy())
            #mlp_leak2.append(activation['GELU'].detach().cpu().numpy())
            n_stuff += 1

            #print(name,pred_prot.item())
            #if n_stuff > 5:
            #    break
            

    return pred_res, names, res_label



train_loss = [];
val_loss = [];
epochs = [];
print("EPOCH,TRAIN,VAL,VAL_CLUST")

if __name__ == "__main__":

    model.eval()
    pred_res, names, res_label  = model_test_prot_env(test_loader, model, DEVICE=DEVICE)

    #pred_res = np.array(pred_res)
    #res_label = np.array(res_label)

    #prot_label = np.array(prot_label)
    #print(np.shape(res_label))
    #for ii in res_label:
    #    print(np.shape(ii))
    names = np.array(names)

    file = './models_DL/af2_' + SPECIES + '-' + my_model_name + '_res.csv'
    f = open(file,'w+')
    for ii in range(len(names)):
        #print(names[ii][0])
        f.write(names[ii][0] + '|')
        for jj in range(len(res_label[ii])):
            #print(res_label[ii][jj][0] )
            f.write(res_label[ii][jj][0] + ',')
        f.write('\n')
    f.close()

    #mlp_leak2 = np.array(mlp_leak2)


    file = "./models_DL/af2_" + SPECIES + "-" + my_model_name + "_res.npz"
    #print(file)
    #np.savez(file,pred_res=pred_res, 
    #            names = names, res_label=res_label)



    #print(epochs,train_loss,val_loss)

print('FIN');
