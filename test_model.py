import sys
sys.path.append("../")

import os
import json
import glob
from pathlib import Path
import argparse

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import h5py

from training.callback import Callback
from models.model_factory import ModelFactory
from datasets import H5Dataset, bag_collate_fn

from tqdm import tqdm

# Set deterministic behavior
SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()

# Loading and saving
parser.add_argument('--path-args', type=str, required=True)
parser.add_argument('--path-df', type=str, required=True)
parser.add_argument('--path-features', type=str, required=True)
parser.add_argument('--path-test-features', type=str)
parser.add_argument('--path-test-df', type=str)
parser.add_argument('--patch-encoder', type=str, default='virchow2', choices=['uni_v2', 'virchow2', 'resnet50'])

args = parser.parse_args()

model_name = args.patch_encoder
path_df = args.path_df
path_features = args.path_features
path_test_features = args.path_test_features
path_test_df = args.path_test_df

if args.path_test_features is None:
    path_test_features = path_features
    
if args.path_test_df is None:
    path_test_df = path_df

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
paths_args = sorted(glob.glob(args.path_args + '/*/args*'))

with open(paths_args[0]) as f:
    args_dict_0 = json.load(f)

path_splits = sorted(glob.glob(args_dict_0['path_splits'] + '/*'))
test_df = pd.read_csv(path_test_df)

for i, path_args in enumerate(paths_args):

    with open(path_args) as f:
        args_dict = json.load(f)
        
    args_dict['path_checkpoints'] = Path(path_args).parent
    
    # define callback, model, classifier, xmodel
    sel_checkpoint = 'best'

    callback = Callback(
            schedule_lr=args_dict['schedule_lr'], checkpoint_epoch=1, path_checkpoints=args_dict['path_checkpoints'],
            early_stop=args_dict['early_stopping'], device=device)
    model, classifier = ModelFactory.build(args_dict, device)
    model = callback.load_checkpoint(model, checkpoint=sel_checkpoint)

    if args_dict['aggregation_model'] == 'transmil':
        collate_fn = None
    else:
        collate_fn = bag_collate_fn

    path_split = path_splits[i]
    df_split = pd.read_csv(path_split)
    
    # df reloaded as it is modified below
    df = pd.read_csv(path_df)
    
    df.loc[df['fold_0'].isin(['val', 'train']), 'fold_0'] = 'train'
    val_set = df_split['val'].to_list()
    df.loc[df['slide_id'].isin(val_set), 'fold_0'] = 'val'

    train_loader = DataLoader(H5Dataset(path_features, df, "train"), batch_size=1, shuffle=True, worker_init_fn=lambda _: np.random.seed(SEED), collate_fn=collate_fn)
    val_loader = DataLoader(H5Dataset(path_features, df, "val"), batch_size=1, shuffle=True, worker_init_fn=lambda _: np.random.seed(SEED), collate_fn=collate_fn)
    test_loader = DataLoader(H5Dataset(path_test_features, test_df, "test"), batch_size=1, shuffle=True, worker_init_fn=lambda _: np.random.seed(SEED), collate_fn=collate_fn)
    
    print(f"Test set evaluation with checkpoint: {sel_checkpoint}")
    
    for loader, setx in zip([train_loader, val_loader, test_loader], ['train', 'val', 'test']):
        if os.path.exists(os.path.join(args_dict['path_checkpoints'], f'results_{setx}.csv')):
            print(f'Skipping test set: {setx}')
            continue
        
        all_preds = []
        all_labels = []
        all_slides = []

        model.eval()
        with torch.no_grad():
            for batch in tqdm(loader):
                preds, label, _, slide_info = classifier.validation_step(batch)
                label = label.item()
                pred = torch.argmax(preds).item()
                slide_name = slide_info['slide_id'][0]
                all_slides.append(slide_name)
                all_labels.append(label)
                all_preds.append(pred)    
                
        results_df = pd.DataFrame({'slide_id': all_slides, 'pred': all_preds, 'label': all_labels})
        results_df.to_csv(os.path.join(args_dict['path_checkpoints'], f'results_{setx}.csv'))