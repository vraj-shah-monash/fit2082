#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark runner: trains MS4N across datasets."""

import os
import sys
import gc
from types import SimpleNamespace
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'monster')))
from utils.utils import *          # Initialization
from utils.data_loader import load
from utils.trainer import trainer

DATASETS = ['Skoda',]#'WhaleSounds', 'Skoda']
MODELS = ['MS4N','GRU']

# the setting is for Monster 
# For MONSTER EPOCHS is 200, but for EEG , HAR , and ImageSatellite is 100
# For UEA is 300
EPOCHS = 200         
BATCH_SIZE = 256
FOLD = 0            # single-fold runs; wrap run_all() in a fold loop if you need more
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUTPUT_PATH, exist_ok=True)


def make_config(dataset):
    """Build one dataset's config: args -> Initialization -> extras, in one place."""
    args = SimpleNamespace(
        dataset=dataset,
        output_path=OUTPUT_PATH,
        val_ratio=0.1,
        print_interval=10,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        normalization='per_channel',
        lr=1e-3,
        dropout=0.01,
        val_interval=2,
        key_metric='loss',
        gpu=0,
        seed=1234,
        fold=FOLD,
    )
    config = Initialization(args)
    return config


def run_all():
    results = []

    for dataset in DATASETS:
        print(f"\n===== Dataset: {dataset} =====")
        gc.collect()
        torch.cuda.empty_cache()

        config = make_config(dataset)
        data = load(config, fold=FOLD)

        for model in MODELS:
            print(f"Training {model} on {dataset}")
            config['Model_Type'] = model
            metrics = trainer(config, data)
            results.append({
                'Dataset': dataset,
                'Model': model,
                'Accuracy': metrics.get('accuracy', 'N/A'),
            })
            print(f">>> Finished {model}")

        print(f"\U0001F4CC Completed dataset: {dataset}")
        pd.DataFrame(results).to_csv(
            os.path.join(OUTPUT_PATH, 'results_FINAL.csv'), index=False, float_format='%.4f'
        )



if __name__ == '__main__':
    run_all()
