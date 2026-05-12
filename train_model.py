import os
import json
import argparse
import glob

import torch
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

import sys
sys.path.append('../')

from datasets import H5Dataset, bag_collate_fn
from models import ModelFactory
from training import Callback, train_classification_model

parser = argparse.ArgumentParser()

# Loading and saving
parser.add_argument('--aggregation-model', type=str, required=True, choices=['transmil', 'attention_mil'])
parser.add_argument('--path-features', type=str, required=True)
parser.add_argument('--path-df', type=str, required=True)
parser.add_argument('--patch-encoder', type=str, default='virchow2', choices=['uni_v2', 'virchow2', 'resnet50'])
parser.add_argument('--path-checkpoint', type=str, required=True)
parser.add_argument('--path-splits', type=str, default='splits')
parser.add_argument('--batch-size', type=int, required=True)
parser.add_argument('--features-dim', type=int, default='256')
parser.add_argument('--epochs', type=int, default='20')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])


# Set deterministic behavior
SEED = 1234
np.random.seed(SEED)
#random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args = parser.parse_args()
aggregation_model = args.aggregation_model
batch_size = args.batch_size
path_checkpoint = args.path_checkpoint
device = args.device
model_name = args.patch_encoder
features_dim = args.features_dim
epochs = args.epochs
path_features = args.path_features
path_df = args.path_df
path_splits = args.path_splits

if model_name == 'virchow2':
    input_dim = 2560
elif model_name == 'uni_v2':
    input_dim = 1536
elif model_name == 'resnet50':
    input_dim = 1024

args_dict = {'aggregation_model': aggregation_model,
             'input_dim': input_dim,
             'num_classes': 3,
             'features_dim': features_dim,
             'inner_attention_dim': 128,
             'dropout': 0.2,
             'num_layers': 2,
             'dropout_strategy': 'all',
             'learning_rate': 2e-4,
             'weight_decay': 1e-3,
             'optimizer': 'Adam',
             'objective': 'cross-entropy',
             'grad_clip': None,
             'n_epochs': epochs,
             'batch_size': batch_size,
             'val_interval': 1,
             'schedule_lr': False,
             'early_stopping': True,
             'patience': 5,
             'targets': ['label'],
             'train_bag_size': 512,
             'min_bag_size': 0,
             'max_bag_size': None,
             "path_checkpoints": path_checkpoint,
             "path_features": path_features,
             "path_df": path_df,
             "path_splits": path_splits,
             #TransMIL params
             'num_features': features_dim,
             'n_layers': 2,
             'dropout_att': 0.2,
             'dropout_class': 0.2,
             'dropout_feat': 0.2,
             'pool-method': 'cls_token',
             'no_attn_residual': True
             }

df = pd.read_csv(path_df)
path_splits = sorted(glob.glob(path_splits + '/*'))

for i, path in enumerate(path_splits):
    path_checkpoint_i = os.path.join(path_checkpoint, f'split_{i}')
    if os.path.exists(os.path.join(path_checkpoint_i, 'last_model.pt')):
        print(f'Skipping split {i}')
        continue
    if not(os.path.exists(path_checkpoint_i)):
        os.mkdir(path_checkpoint_i)
    args_dict['path_checkpoints'] = path_checkpoint_i
    df_split = pd.read_csv(path)
    df.loc[df['fold_0'].isin(['val', 'train']), 'fold_0'] = 'train'
    val_set = df_split['val'].to_list()
    df.loc[df['slide_id'].isin(val_set), 'fold_0'] = 'val'
    # Saving arguments
    with open(os.path.join(args_dict['path_checkpoints'], 'args.json'), 'w') as f:
        json.dump(args_dict, f)

    # Set up model and classifier
    model, classifier = ModelFactory.build(args_dict, device)

    # Set up callback
    callback = Callback(
        schedule_lr=args_dict['schedule_lr'], checkpoint_epoch=args_dict['val_interval'], path_checkpoints=args_dict['path_checkpoints'],
        stop_criterion='loss', early_stop=args_dict['early_stopping'], patience=args_dict['patience'])

    if args_dict['aggregation_model'] == 'transmil':
        collate_fn = None
    else:
        collate_fn = bag_collate_fn

    train_loader = DataLoader(H5Dataset(path_features, df, "train", num_features=args_dict['train_bag_size'], max_bag_size=args_dict['max_bag_size'], min_bag_size=args_dict['min_bag_size']),
                              batch_size=args_dict['batch_size'], shuffle=True, worker_init_fn=lambda _: np.random.seed(SEED), collate_fn=collate_fn)
    val_loader = DataLoader(H5Dataset(path_features, df, "val"), batch_size=1, shuffle=True, worker_init_fn=lambda _: np.random.seed(SEED), collate_fn=collate_fn)

    train_classification_model(model, classifier, classifier.optimizer, args_dict['n_epochs'], args_dict['learning_rate'], train_loader, val_loader, callback, args_dict['targets'])