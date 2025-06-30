import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from gnn import GNN

from tqdm import tqdm
import argparse
import time
import numpy as np

from dataset import datasets, PeptideDataset
from sklearn.metrics import roc_auc_score

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

def train(model, device, loader, optimizer, task_type):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type: 
                loss = cls_criterion(pred.to(torch.float32)[is_labeled].squeeze(), batch.y.to(torch.float32)[is_labeled].squeeze())
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()

def eval(model, device, loader):
    model.eval()
    y_true = []
    y_pred = []
    results = {}

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    acc_list = []
    pre_list = []
    rec_list = []
    mi_rec_list = []
    rocauc_list = []
    for i in range(y_true.shape[1]):
        is_labeled = y_true[:,i] == y_true[:,i]
        tp = np.sum((y_true[is_labeled,i] == 1) & (y_pred[is_labeled,i] >= 0.5))
        fp = np.sum((y_true[is_labeled,i] == 0) & (y_pred[is_labeled,i] >= 0.5))
        tn = np.sum((y_true[is_labeled,i] == 0) & (y_pred[is_labeled,i] < 0.5))
        fn = np.sum((y_true[is_labeled,i] == 1) & (y_pred[is_labeled,i] < 0.5))
        acc_list.append((tp + tn) / (tp + tn + fp + fn))
        if tp + fp > 0:
            pre_list.append(tp / (tp + fp))
        else:
            pre_list.append(0.0)
        if tp + fn > 0:
            rec_list.append(tp / (tp + fn))
        else:
            rec_list.append(0.0)
        mi_rec_list.append(tp / (tp + fp + fn))  # Micro precision
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:,i] == y_true[:,i]
            rocauc_list.append(roc_auc_score(y_true[is_labeled,i], y_pred[is_labeled,i]))
    results['acc']=sum(acc_list)/len(acc_list)
    results['precision'] = sum(pre_list) / len(pre_list)
    results['recall'] = sum(rec_list) / len(rec_list)
    results['f1'] = 2 * results['precision'] * results['recall'] / (results['precision'] + results['recall'])
    results['roc_auc'] = sum(rocauc_list) / len(rocauc_list)
    # return evaluator.eval(input_dict)
    return results



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="Aox_APML",
                        help='dataset name (default: ogbg-molhiv)')
    parser.add_argument('--activity', type=str, default='Theraputic-Other',
                        help='activity type (default: ADME)')
    parser.add_argument('--model_list', type=list, default=['gcn','gin','gat','transformer'],
                        help='choose detection model from gcn, gin, gat, transformer (default: [gcn, gin, gat, transformer])')
    parser.add_argument('--is_smiles', type=bool, default=True,
                        help='if the dataset is in smiles format (default: False)')

    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    for i in ('random1','random2','random3','random4','random5','mmseqs21','mmseqs22','mmseqs23','mmseqs24','mmseqs25'):

        ### automatic dataloading and splitting
        root_path = "/data0/data_share/peptide_data_2025.6.27v/"+args.activity+"/"+args.dataset+"/"  # Replace with your actual file path
        smiles, labels = datasets(root_path, args.is_smiles)  # Replace with your actual file path
        train_dataset = PeptideDataset(smiles, labels, root_path+i[:-1]+'_split.json', 'seed_'+i[-1], 'train','./graph_data/'+args.activity+"/"+args.dataset+"/")
        valid_dataset = PeptideDataset(smiles, labels, root_path+i[:-1]+'_split.json', 'seed_'+i[-1], 'valid','./graph_data/'+args.activity+"/"+args.dataset+"/")
        test_dataset = PeptideDataset(smiles, labels, root_path+i[:-1]+'_split.json', 'seed_'+i[-1], 'test','./graph_data/'+args.activity+"/"+args.dataset+"/")

        for j in args.model_list:
            temp_best_train = None
            temp_best_val = None
            temp_best_test = None
            args.gnn = j
            for k in [0.01,0.001,0.0001,1e-5,1e-6]:
                for l in [0.01,0.001,0.0001,1e-5,1e-6,0]:
                    ### automatic evaluator. takes dataset name as input
                    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
                    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
                    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

                    if args.gnn == 'gin':
                        model = GNN(gnn_type = 'gin', num_tasks = 1, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
                    elif args.gnn == 'gin-virtual':
                        model = GNN(gnn_type = 'gin', num_tasks = 1, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
                    elif args.gnn == 'gcn':
                        model = GNN(gnn_type = 'gcn', num_tasks = 1, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
                    elif args.gnn == 'gcn-virtual':
                        model = GNN(gnn_type = 'gcn', num_tasks = 1, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
                    elif args.gnn == 'gat':
                        model = GNN(gnn_type = 'gat', num_tasks = 1, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
                    elif args.gnn == 'transformer':
                        model = GNN(gnn_type = 'transformer', num_tasks = 1, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
                    else:
                        raise ValueError('Invalid GNN type')

                    optimizer = optim.Adam(model.parameters(), lr=k, weight_decay=l)

                    valid_curve = dict()
                    test_curve = dict()
                    train_curve = dict()
                    count = 0
                    temp_val_f1 = 0

                    for epoch in range(1, args.epochs + 1):
                        print("=====Epoch {}".format(epoch))
                        print('Training...')
                        train(model, device, train_loader, optimizer, 'binary classification')

                        print('Evaluating...')
                        train_perf = eval(model, device, train_loader)
                        valid_perf = eval(model, device, valid_loader)
                        test_perf = eval(model, device, test_loader)

                        print({'Train': train_perf})
                        print({'Validation': valid_perf})
                        print({'Test': test_perf})

                        for key in train_perf:
                            if key not in train_curve:
                                train_curve[key] = []
                                valid_curve[key] = []
                                test_curve[key] = []
                            train_curve[key].append(train_perf[key])
                            valid_curve[key].append(valid_perf[key])
                            test_curve[key].append(test_perf[key])

                        if temp_val_f1 < valid_perf['roc_auc']:
                            temp_val_f1 = valid_perf['roc_auc']
                            count = 0
                        else:
                            count += 1

                        if count > 9 and epoch > 50:
                            print('Early stopping at epoch {}'.format(epoch))
                            break

                        with open('./epoch_results/'+args.dataset+'_'+i+'_'+args.gnn+'_lr'+str(k)+'_wd'+str(l)+'.txt', 'a') as f:  
                            f.write('Epoch: {}, Train Acc: {:.4f}, Train Pre: {:.4f}, Train Rec: {:.4f}, Train F1: {:.4f}, Train ROC AUC: {:.4f}'.format(epoch, train_perf['acc'], train_perf['precision'], train_perf['recall'], train_perf['f1'], train_perf['roc_auc']) + '\n')
                            f.write('Epoch: {}, Valid Acc: {:.4f}, Valid Pre: {:.4f}, Valid Rec: {:.4f}, Valid F1: {:.4f}, Valid ROC AUC: {:.4f}'.format(epoch, valid_perf['acc'], valid_perf['precision'], valid_perf['recall'], valid_perf['f1'], valid_perf['roc_auc']) + '\n')
                            f.write('Epoch: {}, Test Acc: {:.4f}, Test Pre: {:.4f}, Test Rec: {:.4f}, Test F1: {:.4f}, Test ROC AUC: {:.4f}'.format(epoch, test_perf['acc'], test_perf['precision'], test_perf['recall'], test_perf['f1'], test_perf['roc_auc']) + '\n')
                    
                    if temp_best_train is not None:
                        for key in train_curve:
                            train_curve[key].append(temp_best_train[key])
                            valid_curve[key].append(temp_best_val[key])
                            test_curve[key].append(temp_best_test[key])
                    
                    best_train_epoch = np.argmax(np.array(train_curve['roc_auc']))
                    best_val_epoch = np.argmax(np.array(valid_curve['roc_auc']))

                    print('Finished training!')
                    print('Best validation score: {}'.format(valid_curve['roc_auc'][best_val_epoch]))
                    print('Test score: {}'.format(test_curve['roc_auc'][best_val_epoch]))

                    temp_best_train = {'acc': train_curve['acc'][best_train_epoch], 'precision': train_curve['precision'][best_train_epoch], 'recall': train_curve['recall'][best_train_epoch], 'f1': train_curve['f1'][best_train_epoch], 'roc_auc': train_curve['roc_auc'][best_train_epoch]}
                    temp_best_val = {'acc': valid_curve['acc'][best_val_epoch], 'precision': valid_curve['precision'][best_val_epoch], 'recall': valid_curve['recall'][best_val_epoch], 'f1': valid_curve['f1'][best_val_epoch], 'roc_auc': valid_curve['roc_auc'][best_val_epoch]}
                    temp_best_test = {'acc': test_curve['acc'][best_val_epoch], 'precision': test_curve['precision'][best_val_epoch], 'recall': test_curve['recall'][best_val_epoch], 'f1': test_curve['f1'][best_val_epoch], 'roc_auc': test_curve['roc_auc'][best_val_epoch]}

            with open('./results/'+args.dataset+'.txt', 'a') as fi:
                fi.write('Model: {}, Split: {}, Train Acc: {:.4f}, Train Pre: {:.4f}, Train Rec: {:.4f}, Train F1: {:.4f}, Train ROC AUC: {:.4f}'.format(args.gnn, i, temp_best_train['acc'], temp_best_train['precision'], temp_best_train['recall'], temp_best_train['f1'], temp_best_train['roc_auc']) + '\n')
                fi.write('Model: {}, Split: {}, Valid Acc: {:.4f}, Valid Pre: {:.4f}, Valid Rec: {:.4f}, Valid F1: {:.4f}, Valid ROC AUC: {:.4f}'.format(args.gnn, i, temp_best_val['acc'], temp_best_val['precision'], temp_best_val['recall'], temp_best_val['f1'], temp_best_val['roc_auc']) + '\n')
                fi.write('Model: {}, Split: {}, Test Acc: {:.4f}, Test Pre: {:.4f}, Test Rec: {:.4f}, Test F1: {:.4f}, Test ROC AUC: {:.4f}'.format(args.gnn, i, temp_best_test['acc'], temp_best_test['precision'], temp_best_test['recall'], temp_best_test['f1'], temp_best_test['roc_auc']) + '\n')

if __name__ == "__main__":
    main()
