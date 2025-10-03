import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
import argparse
import os
import glob
from datetime import datetime


def cumulative_max_per_round(sequences):
    num_rounds = sequences['round'].max() + 1
    max_per_round = [sequences['true_score'][sequences['round'] == r].max()
                     for r in range(num_rounds)]

    return np.maximum.accumulate(max_per_round)


def get_latest_csv(directory, name_pattern):
    """Return the newest CSV file(s) under ``directory`` matching ``name_pattern``.

    Files whose names contain a timestamp take precedence; if none exist, fall back to
    the most recently modified file."""
    # Gather all CSV files in the directory
    pattern = os.path.join(directory, "*.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {directory}")
    
    # Attempt to identify files containing a timestamp in their names
    timestamped_files = []
    other_files = []
    
    for file in csv_files:
        try:
            filename = os.path.basename(file)
            timestamp_str = filename.split('*')[-1].split('.')[0]
            datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            if name_pattern in file:
                timestamped_files.append(file)
        except (IndexError, ValueError):
            if name_pattern in file:
                other_files.append(file)
    
    if timestamped_files:
        # If timestamped files exist, pick the most recent timestamp
        latest_file = max(timestamped_files, key=lambda x: datetime.strptime(
            os.path.basename(x).split('*')[-1].split('.')[0], 
            "%Y%m%d_%H%M%S"
        ))
        print("Using newest timestamped file")
        print(latest_file)
    else:
        # Otherwise fall back to the file with the latest modification time
        try:
            latest_file = max(other_files, key=os.path.getmtime)
        except:
            print(directory)
        print("Using most recently modified file")
        print(latest_file)
    
    return latest_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn', action='store_true')
    args = parser.parse_args()
    # for landscape in ['muformer']:
    if args.cnn:
        landscapes = ['aav', 'rna', 'gfp', 'tf', 'rosetta']
    else:
        landscapes = ['muformer']
    for landscape in landscapes:
        for sequences_batch_size, model_queries_per_batch in [(100, 5000), (1000, 5000)]:
            plt.figure(dpi=300)
            plt.title(f'Performance with respect to {sequences_batch_size} and {model_queries_per_batch} on {landscape}')
            if args.cnn:
                # method_list = ['adalead', 'cbas', 'cmaes', 'dynappo', 'BO', 'dirichlet_ppo_cnn', 'dirichlet_ppo_update_starting_sequence_cnn', 'test_cnn']
                # dirichlet_ppo_list = ['dirichlet_ppo_cnn', 'dirichlet_ppo_update_starting_sequence_cnn', 'test_cnn']
                method_list = ['adalead', 'cmaes', 'dynappo', 'BO', 'musearch']
                # method_list = ['adalead', 'cbas', 'cmaes', 'dynappo', 'BO', 'dirichlet_ppo_cnn']
                dirichlet_ppo_list = ['dirichlet_ppo_cnn']
            else:
                method_list = ['adalead_gt', 'cmaes_gt', 'dynappo_gt', 'BO_gt', 'dirichlet_ppo']
                # method_list = ['adalead_gt', 'cbas_gt', 'cmaes_gt', 'dynappo_gt', 'BO_gt', 'dirichlet_ppo']
                dirichlet_ppo_list = ['dirichlet_ppo']
            for method in method_list:
            # for method in ['adalead_gt', 'cbas_gt', 'cmaes_gt', 'dynappo_gt', 'BO_gt', 'dirichlet_ppo', 'dirichlet_ppo_update_starting_sequence', 'test']:
                if method in dirichlet_ppo_list:
                # if method in ['dirichlet_ppo', 'dirichlet_ppo_update_starting_sequence', 'test']:
                    if args.cnn:
                        file_name = get_latest_csv(f'efficiency/{method}/{landscape}', str(sequences_batch_size)+'_'+str(model_queries_per_batch))
                        # file_name = f'/home/v-zhaozhiyu/code/FLEXS/efficiency/{method}/{landscape}/{sequences_batch_size}_{model_queries_per_batch}.csv'
                    else:
                        # if method == 'dirichlet_ppo' and landscape == 'rna':
                        #     file_name = f'/home/v-zhaozhiyu/code/FLEXS/efficiency/dirichlet_ppo/rna/100_5000_20250118_094302.csv'
                        file_name = get_latest_csv(f'efficiency/{method}/{landscape}', str(sequences_batch_size)+'_'+str(model_queries_per_batch))
                        # file_name = f'/home/v-zhaozhiyu/code/FLEXS/efficiency/{method}/{landscape}/{sequences_batch_size}_{model_queries_per_batch}_new.csv'
                else:
                    file_name = get_latest_csv(f'efficiency/{method}/{landscape}', str(sequences_batch_size)+'_'+str(model_queries_per_batch))
                with open(file_name) as f:
                    metadata = json.loads(next(f))
                    data = pd.read_csv(f)

                    rounds = data['round'].unique()
                    max_per_round = cumulative_max_per_round(data)
                    # plt.plot(rounds, max_per_round, '-o', label=f'{method}')
                try:
                    plt.plot(rounds, max_per_round, '-o', label=f'{method}')
                except Exception as e:
                    print(file_name)
                    print(f"rounds: {rounds}, max_per_round: {max_per_round}")
                    print(e)                
                    pass
                plt.ylabel("Cumulative max")
                plt.xlabel("Round")

            plt.legend()
            # plt.savefig(f'examples/figs/efficiency_{landscape}_{sequences_batch_size}_{model_queries_per_batch}.png')
            if args.cnn:
                plt.savefig(f'examples/figs/cnn/efficiency_{landscape}_{sequences_batch_size}_{model_queries_per_batch}_cnn.png')
            else:
                plt.savefig(f'examples/figs/gt/efficiency_{landscape}_{sequences_batch_size}_{model_queries_per_batch}_gt_updated.png')