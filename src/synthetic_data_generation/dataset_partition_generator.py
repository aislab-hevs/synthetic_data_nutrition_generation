from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from enum import Enum
import os 
from pathlib import Path
import argparse


class SplitsModes(str, Enum):
    random = 'random'
    user_oriented = 'user_oriented'
    mixed = 'mixed'
    

def join_datasets(tracking_dataset: pd.DataFrame, 
                  user_dataset: pd.DataFrame, 
                  recipes_dataset: pd.DataFrame)-> pd.DataFrame:
    # Join Datasets and return a joined dataset. 
    df_join = pd.merge(tracking_dataset, 
                       user_dataset, 
                       left_on='userId', 
                       right_on='userId')
    df_join = pd.merge(df_join, 
                       recipes_dataset, 
                       left_on='foodId', 
                       right_on='recipeId')
    return df_join

def split_datasets(full_dataset: pd.DataFrame, 
                   mode: SplitsModes = SplitsModes.random,
                   test_size: float = 0.2, 
                   random_state: int = 41) -> List[pd.DataFrame]: 
    # Split Datasets based on random or user oriented or both. 
    if mode == SplitsModes.random:
        train, test_val = train_test_split(full_dataset, 
                                           test_size=test_size, 
                                           random_state=random_state)
        val, test = train_test_split(test_val, 
                                     test_size=test_size, 
                                     random_state=random_state)
        return [train, val, test]
    elif mode == SplitsModes.user_oriented:
        unique_users = full_dataset['userId'].unique()
        users_val_test = unique_users.sample(frac=test_size, 
                                             random_state=random_state)
        val_users = users_val_test.sample(frac=test_size,
                                          random_state=random_state)
        test_users = set(users_val_test.to_list()) - set(val_users.to_list())
        train = full_dataset[~full_dataset['userId'].isin(users_val_test)]
        val = full_dataset[full_dataset['userId'].isin(val_users)]
        test = full_dataset[full_dataset['userId'].isin(test_users)]
        return [train, val, test]
    elif mode == SplitsModes.mixed:
        # Random split
        train, test_val = train_test_split(full_dataset, 
                                           test_size=test_size, 
                                           random_state=random_state)
        val, test = train_test_split(test_val, 
                                     test_size=test_size, 
                                     random_state=random_state)
        # User oriented split
        unique_users = full_dataset['userId'].unique()
        users_val_test = unique_users.sample(frac=test_size, 
                                             random_state=random_state)
        val_users = users_val_test.sample(frac=test_size,
                                          random_state=random_state)
        test_users = set(users_val_test.to_list()) - set(val_users.to_list())
        train = train[~train['userId'].isin(users_val_test)]
        val = val[val['userId'].isin(val_users)]
        return [train, val, test]
    else:
        raise ValueError(f'Invalid mode: {mode}')


def full_dataset_preprocess(path_to_track_df: pd.DataFrame, 
                            path_to_user_df: pd.DataFrame, 
                            path_to_recipes_df: pd.DataFrame, 
                            destination_directory: str, 
                            track_sep: str =',',
                            user_sep: str =',',
                            recipes_sep: str ='|',
                            mode: SplitsModes = SplitsModes.random,
                            test_size: float = 0.2, 
                            random_state: int = 41) -> None: 
    # Load datasets join and split and write in the target directory. 
    # load datasets 
    df_track = pd.read_csv(path_to_track_df, sep=track_sep, index_col=0)
    print(f"Loaded tracking dataset with shape: {df_track.shape}")
    df_user = pd.read_csv(path_to_user_df, sep=user_sep, index_col=0)
    print(f"Loaded users dataset with shape: {df_user.shape}")
    df_recipes = pd.read_csv(path_to_recipes_df, sep=recipes_sep, index_col=0)
    print(f"Loaded recipes dataset with shape: {df_recipes.shape}")
    # Join datasets 
    full_dataset = join_datasets(df_track, df_user, df_recipes)
    print(f"Join dataset with shape: {full_dataset.shape}")
    # Split datasets 
    dataset_list = split_datasets(full_dataset=full_dataset,
                                  mode=mode,
                                  test_size=test_size,
                                  random_state=random_state)
    # Save dataset in target directory
    dataset_order = ['train', 'val', 'test']
    for i, dataset in enumerate(dataset_list):
        dataset.to_csv(os.path.join(destination_directory, f'dataset_{dataset_order[i]}.csv'), 
                       sep='|')
    print(f"Saved datasets in {destination_directory}")


if __name__ == '__main__':
    # receive parameters and execute full pipeline.
    parser = argparse.ArgumentParser(description="Dataset generation pipeline")
    # Add arguments 
    parser.add_argument('-t', '--path_tracking_file', type=str, required=True)
    parser.add_argument('-u', '--path_user_file', type=str, required=True)
    parser.add_argument('-r', '--path_recipes_file', type=str, required=True)
    parser.add_argument('-o', '--path_output_directory', type=str, required=True)
    parser.add_argument('-m', '--mode', type=SplitsModes, default=SplitsModes.random)
    parser.add_argument('-ts', '--test_size', type=float, default=0.2)
    parser.add_argument('-rs', '--random_state', type=int, default=41)

    # Parse arguments
    args = parser.parse_args()
    full_dataset_preprocess(path_to_track_df=args.path_tracking_file,
                            path_to_user_df=args.path_user_file,
                            path_to_recipes_df=args.path_recipes_file,
                            destination_directory=args.path_output_directory,
                            mode=args.mode,
                            test_size=args.test_size,
                            random_state=args.random_state)
    