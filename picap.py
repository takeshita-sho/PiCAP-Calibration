
import os
import numpy as np
import pandas as pd
from utils import *
from egnn.egnn import *
import matplotlib.pyplot as plt
from torchvision.models.feature_extraction import create_feature_extractor
import re
import json
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
def run_picap(TEST_PDB,TEST_CLUST,my_model_name,JSON=False):
    """
    Runs Capsif2 and predicts all residues on given input pdb/cluster files
    Arguments:
        TEST_PDB (string): Path to input PDB csv file
        TEST_CLUST (string): Path to the input CLUSTER csv file (not used)
    Returns:
        names (arr, string): all the input pdb names
        prot_pred (arr, float): predicted probability of carb binding
    Outputs:
        "./output_data/predictions_prot.tsv" - tsv of picap predictions
    """

    print("\n\n\nloading PiCAP")
    NUM_WORKERS = 0;

    #Hyper parameters!
    KNN = [10,20,40,60]
    N_LAYERS = [3,3,3,3]
    HIDDEN_NF = 128
    ADAPOOL_SIZE = (150,HIDDEN_NF)

    DEVICE = 'cpu'
    if torch.cuda.is_available():
        DEVICE = 'cuda'
        NUM_WORKERS = 8;
    print("Using: " + DEVICE)
    DEVICE = torch.device(DEVICE)
    print(DEVICE)

    test_loader = get_test_loader(TEST_CLUST,TEST_PDB,root_dir="./",train=0,
                                    batch_size=1,num_workers=NUM_WORKERS,knn=KNN)

    model = CAPSIF2_PROT(hidden_nf = HIDDEN_NF, n_layers=N_LAYERS,attention=True,
                    device=DEVICE).to(DEVICE)
    
    ###
    #For bayes
    """
    const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",  # Flipout or Reparameterization
        "moped_enable": True,  # True to initialize mu/sigma from the pretrained dnn weights
        "moped_delta": 0.5,
    }
    dnn_to_bnn(model,const_bnn_prior_parameters)
    model = model.to(DEVICE)
    """
    ###
    
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scaler = torch.cuda.amp.GradScaler()
    torch.autograd.set_detect_anomaly(True)
    model.train()

    #my_model_name = 'picap'

    if DEVICE == 'cuda':
        checkpoint = torch.load(my_model_name)
    else:
        checkpoint = torch.load(my_model_name,
            map_location=torch.device('cpu') )
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    print("PiCAP loaded")

    print("Running Predictions...")

    model.eval()
    prot_pred, names  = model_test_prot_env(test_loader, model, DEVICE=DEVICE)

    prot_pred = np.array(prot_pred)
    prot_pred = prot_pred.reshape((-1))

    names = np.array(names)
    #fix the names to not include "_0"
    for ii in range(len(names)):
        names[ii][0] = names[ii][0][ :names[ii][0].rfind('_') ]

    file = "./output_data/predictions_prot.tsv"
    print('\n\t------PiCAP results-------')
    
    out = ''
    for ii in range(len(names)):
        #if OUTPUT_INT_TO_CMD:
        print(names[ii][0],',', str(prot_pred[ii]))
        #out += str(names[ii][0]) + '\t' + str(round(prot_pred[ii],4)) + '\n'
    '''
    if not JSON:
        if not os.path.exists(file):
            out = 'PDB_NAME\tpred\n' + out
        f = open(file,'a+')
        f.write(out)
        f.close()
    '''
    return names, prot_pred
def run_picapb(TEST_PDB,TEST_CLUST,my_model_name,JSON=False):
    """
    Runs Capsif2 and predicts all residues on given input pdb/cluster files
    Arguments:
        TEST_PDB (string): Path to input PDB csv file
        TEST_CLUST (string): Path to the input CLUSTER csv file (not used)
    Returns:
        names (arr, string): all the input pdb names
        prot_pred (arr, float): predicted probability of carb binding
    Outputs:
        "./output_data/predictions_prot.tsv" - tsv of picap predictions
    """

    print("\n\n\nloading PiCAP")
    NUM_WORKERS = 0;

    #Hyper parameters!
    KNN = [10,20,40,60]
    N_LAYERS = [3,3,3,3]
    HIDDEN_NF = 128
    ADAPOOL_SIZE = (150,HIDDEN_NF)

    DEVICE = 'cpu'
    if torch.cuda.is_available():
        DEVICE = 'cuda'
        NUM_WORKERS = 8;
    print("Using: " + DEVICE)
    DEVICE = torch.device(DEVICE)
    print(DEVICE)

    test_loader = get_test_loader(TEST_CLUST,TEST_PDB,root_dir="./",train=0,
                                    batch_size=1,num_workers=NUM_WORKERS,knn=KNN)

    model = CAPSIF2_PROT(hidden_nf = HIDDEN_NF, n_layers=N_LAYERS,attention=True,
                    device=DEVICE).to(DEVICE)
    
    ###
    #For bayes
    
    const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",  # Flipout or Reparameterization
        "moped_enable": True,  # True to initialize mu/sigma from the pretrained dnn weights
        "moped_delta": 0.5,
    }
    dnn_to_bnn(model,const_bnn_prior_parameters)
    model = model.to(DEVICE)
    
    ###
    
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scaler = torch.cuda.amp.GradScaler()
    torch.autograd.set_detect_anomaly(True)
    model.train()

    #my_model_name = 'picap'

    if DEVICE == 'cuda':
        checkpoint = torch.load(my_model_name)
    else:
        checkpoint = torch.load(my_model_name,
            map_location=torch.device('cpu') )
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    print("PiCAP loaded")

    print("Running Predictions...")

    model.eval()
    prot_pred, names  = model_test_prot_env(test_loader, model, DEVICE=DEVICE)

    prot_pred = np.array(prot_pred)
    prot_pred = prot_pred.reshape((-1))

    names = np.array(names)
    #fix the names to not include "_0"
    for ii in range(len(names)):
        names[ii][0] = names[ii][0][ :names[ii][0].rfind('_') ]

    file = "./output_data/predictions_prot.tsv"
    print('\n\t------PiCAP results-------')
    
    out = ''
    for ii in range(len(names)):
        #if OUTPUT_INT_TO_CMD:
        print(names[ii][0],',', str(prot_pred[ii]))
        #out += str(names[ii][0]) + '\t' + str(round(prot_pred[ii],4)) + '\n'
    '''
    if not JSON:
        if not os.path.exists(file):
            out = 'PDB_NAME\tpred\n' + out
        f = open(file,'a+')
        f.write(out)
        f.close()
    '''
    return names, prot_pred