import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Experimenter: performs feature extraction followed by linear training/classification with KNN, LDA and SVM")
    #data, paths, and other settings of general setup
    parser.add_argument('--data_path', type=str, default= '/ssd/DATA/datasets', help='Path to load/save torchvision or custom datasets')
    parser.add_argument('--output_path', type=str, default= '/home/DATA/results/ViT_features/Experimental', help='Path for saving each models performance')    
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID or list of ids ID1,ID2,... (see nvidia-smi), wrong ID will make the script use the CPU')
    parser.add_argument('--multigpu',  action='store_true', default=False, help='Either to use parallel GPU processing or not')

    #methods
    parser.add_argument('--model', type=str, default='vit_base_patch16_224', help='Name of an architecture to experiment with (see models.py')
    parser.add_argument('--pooling', type=str, default='VORTEX', help='Pooling technique (pytorch module/layer name, eg. AvgPool2d)')
    parser.add_argument('--M', type=int, default=4, help='M parameter, only used by the REnATA model')

    parser.add_argument('--dataset', type=str, default='KTH-TIPS2-b', help='dataset name, same as the dataloader name')
    parser.add_argument('--grayscale',  action='store_true', default=False, help='Converts images to grayscale')

    parser.add_argument('--input_dimm', type=int, default=224, help='Image input size (single value, square). The standard is a forced resize to 224 (square)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size, increase for better speed, if you have enough VRAM')

    parser.add_argument('--seed', type=int, default=666999, help='Base random seed for weight initialization and data splits/shuffle')
   
    return parser.parse_args()


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import os
import numpy as np
import random
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

import multiprocessing
total_cores=multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"]='16'
args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
import torch
import torchvision
from models import extract_features
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, model_selection

import datasets

import pickle
import time
from joblib import parallel_backend

if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()       
    
    DATASETS_ = {'DTD' : datasets.DTD,
                 'FMD' : datasets.FMD,
                 'Vistex': datasets.Vistex,
                 'USPtex': datasets.USPtex,
                 'CURet': datasets.CURet,
                 'LeavesTex1200': datasets.LeavesTex1200,
                 'MBT': datasets.MBT,
                 'KTH-TIPS2-b': datasets.KTH_TIPS2_b,
                 'Outex' : datasets.Outex,
                 'MINC': datasets.MINC,
                 'GTOS': datasets.GTOS,
                 'GTOS-Mobile': datasets.GTOS_mobile,
                }

    ##########
    path = os.path.join(args.data_path, args.dataset)
    os.makedirs(os.path.join(args.output_path, 'feature_matrix', args.dataset), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'classification', args.dataset), exist_ok=True)
     
    if 'REnATA' not in args.pooling and 'VORTEX' not in args.pooling:
        args.M=''
        
    
 
    print(args.model,   args.pooling, 'M=', args.M, args.dataset, args.input_dimm, 'gray?', args.grayscale)
    # print('Evaluating fold (...)= ', sep=' ', end='', flush=True)    
    
    base_seed = args.seed
    
    # ImageNet normalization parameters and other img transformations
    averages =  (0.485, 0.456, 0.406)
    variances = (0.229, 0.224, 0.225)  
    #Data loader
    if args.input_dimm == 'original':
        _transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(averages, variances),
            # torchvision.transforms.Resize((args.input_dimm ,args.input_dimm ))
            # torchvision.transforms.CenterCrop(args.input_dimm)
        ])   
    else:
        _transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(averages, variances),
            torchvision.transforms.Resize((args.input_dimm ,args.input_dimm ))
            # torchvision.transforms.CenterCrop(args.input_dimm)
        ])        
    
    ### kfold
    args.iterations = 1
    if args.dataset == 'Outex13' or args.dataset == 'Outex14' or args.dataset == 'Outex10' or args.dataset == 'Outex11' or args.dataset == 'GTOS-Mobile':
        args.K = 1 #these Outexes and GTOS have a single train/test split
    elif args.dataset == 'Outex12':
        args.K = 2
    elif args.dataset == 'DTD':
        args.K = 10 #DTD have a fixed set of 10 splits
    elif 'KTH' in args.dataset:
        args.K = 4 #KTH2 have a fixed set of 4 splits
    elif args.dataset == 'MINC'  or args.dataset == 'GTOS':
        args.K = 5 #MINC and GTOS have a fixed set of 5 splits
    elif args.dataset == 'FMD' or args.dataset == 'LeavesTex1200':
        args.K = 10
        args.iterations = 10
    else: #10fOutex also falls here
        args.K = 10
        args.iterations = 10
     
    splits = args.K #datasets with no defined sets get randomly K-splited
    
    gtruth_= []
    preds_KNN, preds_LDA, preds_SVM = [], [], []
    accs_KNN, accs_LDA, accs_SVM = [], [], []
    #file name for saving feature matrices (we are not saving right now...)
    file = [args.output_path + '/feature_matrix/' + args.dataset + '/' + args.model +  '_' + args.pooling + str(args.M)
            + '_' + args.dataset + '_' + str(args.input_dimm) + '_gray' + str(args.grayscale) + '_AllSamples.pkl'][0]
    
    file2 = [args.output_path +  '/classification/' + args.dataset + '/' + args.model +  '_' + args.pooling + str(args.M)
             + '_' + args.dataset + '_' + str(args.input_dimm) +  '_gray' + str(args.grayscale) + '_K' + str(args.K) +'_EVALUATION.pkl'][0]
    #print(file2)
    for it_ in range(args.iterations):
        seed = base_seed*(it_+1)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
    
        if 'KTH' in args.dataset:
            crossval = model_selection.PredefinedSplit   
        elif '10f' in args.dataset:
            crossval = model_selection.StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed+1)
        # elif 'Midrib' in args.dataset:
        #     crossval = model_selection.KFold(n_splits=splits, shuffle=True, random_state=seed+1)
        elif args.dataset != 'DTD' and args.dataset != 'MINC' and 'Outex' not in args.dataset and 'GTOS-Mobile' not in args.dataset:
            crossval = model_selection.StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed+1)

        for partition in range(splits):
            if not os.path.isfile(file2):         
                if os.path.isfile(file) :
                    # I DISABLED THIS PART, but you can enable feature saving if wanted 
                    # # To enable it, uncomment the lines below and the ones after the 'else' statatement. This may use a lot of disk space if your dataset is big like GTOS
                    TODO=None 
                    # if 'Outex' in args.dataset or args.dataset == 'GTOS-Mobile':
                    #     with open(file, 'rb') as f:
                    #         X_train,Y_train,X_test,Y_test = pickle.load(f) 
                    # else:
                    #     if partition == 0:
                    #         with open(file, 'rb') as f:
                    #             X_, Y_, files = pickle.load(f)                              
                            
                    #         if 'KTH' in args.dataset:
                    #             crossval = crossval(DATASETS_[args.dataset](root=path, load_all=False).splits) 
                                
                    #         elif 'DTD' in args.dataset or 'MINC' in args.dataset:
                    #             dataset = DATASETS_[args.dataset](root=path)
                    #         else:                                
                    #             crossval.get_n_splits(X_, Y_)                    
                    #             crossval=crossval.split(X_, Y_)                            
                          
                    #     if 'DTD' in args.dataset or 'MINC' in args.dataset:
                    #         train_index = dataset.get_indexes(split="train", partition=partition+1)
                    #         if args.dataset == 'DTD':
                    #             val_index = dataset.get_indexes(split="val", partition=partition+1)
                    #         else:
                    #             val_index = dataset.get_indexes(split="validate", partition=partition+1)
                    #         train_index = train_index + val_index
                    #         test_index = dataset.get_indexes(split="test", partition=partition+1)
                    #     else:
                    #         train_index, test_index = next(crossval) 
                            
                    #     X_test,Y_test = X_[test_index], Y_[test_index]
                    #     X_train, Y_train = X_[train_index], Y_[train_index]
                else:            
                    if args.dataset == 'DTD' or args.dataset == 'MINC' or args.dataset == 'GTOS':
                        if partition == 0: 
                            dataset= DATASETS_[args.dataset](root=path, download=True,
                                                                transform=_transform) 
                            
                            X_, Y_ = extract_features(args.model, dataset, pooling=args.pooling, M=args.M,
                                                      batch_size=args.batch_size, multigpu=args.multigpu, seed=seed)       
                                        
                            # with open(file, 'wb') as f:
                            #     pickle.dump([X_, Y_, dataset._image_files], f)
                                
                        train_index = dataset.get_indexes(split="train", partition=partition+1)
                        if args.dataset == 'DTD':
                            val_index = dataset.get_indexes(split="val", partition=partition+1)
                            train_index = train_index + val_index
                        elif args.dataset == "MINC":
                            val_index = dataset.get_indexes(split="validate", partition=partition+1)                            
                            train_index = train_index + val_index
                            
                        test_index = dataset.get_indexes(split="test", partition=partition+1)
                        
                        X_train, Y_train = X_[train_index], Y_[train_index]
                        X_test, Y_test = X_[test_index], Y_[test_index]                        
                        
                    elif 'Outex' in args.dataset:
                        if partition == 0: 
                            if it_ == 0:
                                outex_path = os.path.join(args.data_path, 'Outex')
                                train_dataset= DATASETS_['Outex'](root=outex_path, split='train',
                                                            suite=args.dataset.split('Outex')[1],                                                      
                                                            transform=_transform) 
                                
                                X_train,Y_train = extract_features(args.model, train_dataset, pooling=args.pooling, M=args.M,
                                                                   batch_size=args.batch_size, multigpu=args.multigpu, seed=seed)        
                                
                                test_dataset= DATASETS_['Outex'](root=outex_path, split='test',
                                                            suite=args.dataset.split('Outex')[1],
                                                            transform=_transform)
                                
                                X_test,Y_test = extract_features(args.model, test_dataset, pooling=args.pooling, M=args.M,
                                                                 batch_size=args.batch_size, multigpu=args.multigpu, seed=seed)
                        # with open(file, 'wb') as f:
                        #     pickle.dump([X_train, Y_train, X_test, Y_test, train_dataset._image_files, test_dataset._image_files], f)
                        
                        if '10f' in args.dataset:
                            if partition == 0:
                                if it_ == 0:
                                    X_ = np.vstack((X_train, X_test))
                                    Y_ = np.hstack((Y_train, Y_test))
                                
                                crossval.get_n_splits(X_, Y_)
                                crossval=crossval.split(X_, Y_)  
                                
                            train_index, test_index = next(crossval)  
                                
                            X_test,Y_test = X_[test_index], Y_[test_index]
                            X_train, Y_train = X_[train_index], Y_[train_index]
                    elif args.dataset == 'GTOS-Mobile':
                        train_dataset= DATASETS_['GTOS-Mobile'](root=path, split='train',                                                                                                       
                                                    transform=_transform) 
                        
                        X_train,Y_train = extract_features(args.model, train_dataset, pooling=args.pooling, M=args.M,
                                                           batch_size=args.batch_size, multigpu=args.multigpu, seed=seed)        
                        
                        test_dataset= DATASETS_['GTOS-Mobile'](root=path, split='test',                                                                                                       
                                                    transform=_transform) 
                        
                        X_test,Y_test = extract_features(args.model, test_dataset, pooling=args.pooling, M=args.M,
                                                         batch_size=args.batch_size, multigpu=args.multigpu, seed=seed)
                        
                    else:
                        if partition == 0: 
                            if it_ == 0:
                                dataset= DATASETS_[args.dataset](root=path,
                                                                 transform= _transform, grayscale=args.grayscale)
                                
                                X_,Y_ = extract_features(args.model, dataset, pooling=args.pooling, M=args.M,
                                                         batch_size=args.batch_size, multigpu=args.multigpu, seed=seed) 
                            
                                # with open(file, 'wb') as f:
                                #     pickle.dump([X_, Y_, dataset._image_files], f)
                                
                            
                            if 'KTH' in args.dataset:
                                crossval = crossval(DATASETS_[args.dataset](root=path, load_all=False).splits) 
                                
                            crossval.get_n_splits(X_, Y_)
                            crossval=crossval.split(X_, Y_)  
                            
                        train_index, test_index = next(crossval)  
                            
                        X_test,Y_test = X_[test_index], Y_[test_index]
                        X_train, Y_train = X_[train_index], Y_[train_index]
             
    
            if not os.path.isfile(file2):
                if partition == 0: 
                    print(f"{time.time()-start_time:.4f}",                          
                          's so far, now classifying...', (X_train.shape, X_test.shape))
                gtruth_.append(Y_test)
                with parallel_backend('threading', n_jobs=16):
                    
                    KNN = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='auto', 
                                                leaf_size=30, p=2, metric='minkowski', metric_params=None)
                    
                    KNN.fit(X_train,Y_train)
                    preds=KNN.predict(X_test)              
                    preds_KNN.append(preds)            
                    acc= sklearn.metrics.accuracy_score(Y_test, preds)        
                    accs_KNN.append(acc*100)     
                    
                    if X_train.shape[1] > 4300:
                    # A little tuning tha helps when a lot of features are used
                    # Note that this was not used for the VORTEX paper, since we used a limit if 1280 features
                        LDA= LinearDiscriminantAnalysis(solver='svd',
                                                        priors=None,
                                                        n_components=None,
                                                        store_covariance=False,
                                                        tol=0.0001, covariance_estimator=None)
                    else:
                        LDA= LinearDiscriminantAnalysis(solver='lsqr',
                                                        shrinkage='auto', priors=None,
                                                        n_components=None,
                                                        store_covariance=False,
                                                        tol=0.0001, covariance_estimator=None)
                    
                    LDA.fit(X_train,Y_train)
                    preds=LDA.predict(X_test)              
                    preds_LDA.append(preds)            
                    acc= sklearn.metrics.accuracy_score(Y_test, preds)
                    accs_LDA.append(acc*100)  
                    
              
                    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='scale', 
                                  coef0=0.0, shrinking=True, probability=False, tol=0.001,
                                  cache_size=200, class_weight=None, verbose=False, 
                                  max_iter=100000, decision_function_shape='ovr', 
                                  break_ties=False, random_state=seed)
                    
  
                    start_time2 = time.time()                    
                    SVM.fit(X_train,Y_train)

                    
                    preds=SVM.predict(X_test)            
                    preds_SVM.append(preds)            
                    acc= sklearn.metrics.accuracy_score(Y_test, preds)
                    accs_SVM.append(acc*100)       

    if os.path.isfile(file2):           
        with open(file2, 'rb') as f:
            results = pickle.load(f) 
            
    else:
        results = {'gtruth_':gtruth_,
                    'preds_KNN':preds_KNN,
                    'accs_KNN':accs_KNN,
                    'preds_LDA':preds_LDA,
                    'accs_LDA':accs_LDA,
                    'preds_SVM':preds_SVM,
                    'accs_SVM':accs_SVM}  
         
        with open(file2, 'wb') as f:
            pickle.dump(results, f)
    
    
     
    if args.iterations > 1: #in this case avg acc is computed over iterations, and std over these averages
        knn, lda, svm = [], [], []
        for it_ in range(args.iterations):
            knn.append(np.mean(results['accs_KNN'][it_*args.K: it_*args.K + args.K]))
            lda.append(np.mean(results['accs_LDA'][it_*args.K: it_*args.K + args.K]))
            svm.append(np.mean(results['accs_SVM'][it_*args.K: it_*args.K + args.K]))        
        results['accs_KNN'] = knn
        results['accs_LDA'] = lda
        results['accs_SVM'] = svm
    
    print('LDA:', f"{np.mean(results['accs_LDA']):.1f} (+-{np.std(results['accs_LDA']):.1f})", 
          'SVM:', f"{np.mean(results['accs_SVM']):.1f} (+-{np.std(results['accs_SVM']):.1f})", 
          'KNN:', f"{np.mean(results['accs_KNN']):.1f} (+-{np.std(results['accs_KNN']):.1f})"
          )
