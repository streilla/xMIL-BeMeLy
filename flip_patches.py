import os
import json
import argparse

import pandas as pd
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

from training.callback import Callback
from models.model_factory import ModelFactory, xModelFactory
from datasets import H5Dataset, bag_collate_fn
from xai.evaluation import xMILEval

parser = argparse.ArgumentParser()

# Loading and saving
parser.add_argument('--aggregation-model', type=str, required=True, choices=['transmil', 'attention_mil'])
parser.add_argument('--path-features', type=str, required=True)
parser.add_argument('--path-df', type=str, required=True)
parser.add_argument('--patch-encoder', type=str, default='virchow2', choices=['uni_v2', 'virchow2', 'resnet50'])
parser.add_argument('--path-checkpoint', type=str, required=True)
parser.add_argument('--results-dir', type=str, default='results/BeMeLy')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])

args = parser.parse_args()

path_checkpoint = args.path_checkpoint
results_dir = args.results_dir
path_features = args.path_features
explanation_folder = f'{path_checkpoint}/explanations'
explanation_path = f'{explanation_folder}/test_predictions_local.csv'
flip_perc = 1
strategy=f'{flip_perc}%-of-all'

# Set deterministic behavior
SEED = 1234
np.random.seed(SEED)
#random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
    
args_flip = {
    'model_path': path_checkpoint,
    'results_dir': results_dir,
    'sel_checkpoint': 'best',
    'explanation_types': ['lrp', 'perturbation_keep', 'attention', 'gi', 'ig'],
    'explain_scores_path': None,
    'precomputed_heatmap_types': ['lrp', 'perturbation_keep', 'attention', 'gi', 'ig'],
    'max_bag_size': None,
    'min_bag_size': None,
    'strategy': strategy,
    'explained_rel': 'logit',
    'lrp-params': {"gamma": 0, "eps": 1e-8, "no_bias": 1},
    'approach': 'drop',
    'device': args.device,
    'dataset': 'test',
    'flipping': True,
    'baseline': True,
    'morl_abs': False,
    'preload_data': False,
}


def main():
    df = pd.read_csv(args.path_df)
    with open(os.path.join(args_flip['model_path'], 'args.json')) as f:
        args_model = json.load(f)
        args_model['preload_data'] = args_flip['preload_data']

    print(json.dumps(args_model, indent=4))

    device = torch.device(args_flip['device'])

    # load the data_loader of interest based on the user argument args_user.dataset
    none_datasets = [f'{set_name}_subsets' for set_name in ['train', 'val', 'test'] if set_name != args_flip['dataset']]
    for set_name in none_datasets:
        args_model[set_name] = None
    
    if args_model['aggregation_model'] == 'transmil':
        collate_fn = None
    else:
        collate_fn = bag_collate_fn
    
    test_loader = DataLoader(H5Dataset(path_features, df, "test"), batch_size=1, shuffle=True, worker_init_fn=lambda _: np.random.seed(SEED), collate_fn=collate_fn)
    data_loader = [loader for loader in [test_loader] if loader is not None][0]

    # define callback, model, classifier, xmodel, and xmodel_eval
    callback = Callback(
        schedule_lr=args_model['schedule_lr'], checkpoint_epoch=1, path_checkpoints=args_flip['model_path'],
        early_stop=args_model['early_stopping'], device=device)
    model, classifier = ModelFactory.build(args_model, device)
    model = callback.load_checkpoint(model, checkpoint=args_flip['sel_checkpoint'])
    xmodel = xModelFactory.build(model, args_flip)

    if args_flip['explain_scores_path'] is not None:
        df_predictions = pd.read_csv(args_flip['explain_scores_path'])

    # the loop over the heatmap types of interest
    for heatmap_type in args_flip['explanation_types']:
        print(heatmap_type)

        if args_flip['explain_scores_path'] is not None and heatmap_type in args_flip['precomputed_heatmap_types']:
            df_patch_scores = df_predictions
        else:
            df_patch_scores = None

        xmodel_eval = xMILEval(xmodel, classifier, heatmap_type=heatmap_type, scores_df=df_patch_scores)
        if args_flip['flipping']:
            torch.cuda.empty_cache()
            print(f'{args_flip["approach"]} most relevant first ...')
            df_results_flipping = xmodel_eval.patch_drop_or_add(data_loader, attribution_strategy='original',
                                                                order='morf', approach=args_flip['approach'],
                                                                strategy=args_flip['strategy'],
                                                                max_bag_size=args_flip['max_bag_size'],
                                                                min_bag_size=args_flip['min_bag_size'],
                                                                verbose=False)

            df_results_flipping.to_csv(os.path.join(args_flip['results_dir'],
                                                    f'{heatmap_type}_{args_flip["approach"]}_patch_flipping_results.csv'))

        if args_flip['morl_abs']:
            torch.cuda.empty_cache()
            print(f'{args_flip["approach"]} most relevant last from absolute values ...')
            df_results_morl_abs = xmodel_eval.patch_drop_or_add(data_loader, attribution_strategy='abs',
                                                                order='morl', approach=args_flip['approach'],
                                                                strategy=args_flip['strategy'],
                                                                max_bag_size=args_flip['max_bag_size'],
                                                                min_bag_size=args_flip['min_bag_size'],
                                                                verbose=False)

            df_results_morl_abs.to_csv(os.path.join(args_flip['results_dir'],
                                                    f'{heatmap_type}_{args_flip["approach"]}_morl_abs_results.csv'))

    if args_flip['baseline']:
        torch.cuda.empty_cache()
        print(f'random baseline ... ')
        xmodel_eval = xMILEval(xmodel, classifier, heatmap_type=None, scores_df=None)
        df_results_random = xmodel_eval.patch_drop_or_add(data_loader, attribution_strategy='random',
                                                          order='morf', approach=args_flip['approach'],
                                                          strategy=args_flip['strategy'],
                                                          max_bag_size=args_flip['max_bag_size'],
                                                          min_bag_size=args_flip['min_bag_size'],
                                                          verbose=False)

        df_results_random.to_csv(os.path.join(args_flip['results_dir'],
                                              f'random_{args_flip["approach"]}_patch_flipping_results.csv'))