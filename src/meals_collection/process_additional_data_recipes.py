import pandas as pd
import numpy as np
import json
import re
import os
from typing import List, Dict, Callable
import glob

# processing functions


def extract_numeric_values_grams(raw_text: str):
    answer = []
    lower_text = raw_text.lower()
    if "per 100g" in lower_text:
        lower_text = lower_text.replace("100g", "")
    list_values = re.findall(r"\d+.?\d*g", lower_text)
    answer = list_values
    return answer


def extract_prep_time(raw_text: str):
    answer = ''
    lower_text = raw_text.lower()
    if "hour" in lower_text or "hours" in lower_text:
        # procedure for hours
        find_answers = re.findall(
            r'(\d+\.?\d*)[\s]*(?=hours|hour)', lower_text, re.IGNORECASE)
        find_answers = [float(i)*60 for i in find_answers]
    else:
        find_answers = re.findall(r'\d+', lower_text, re.IGNORECASE)
    if len(find_answers) == 1:
        answer = find_answers[0]
    elif len(find_answers) > 1:
        answer = sum([float(i) for i in find_answers])
    else:
        answer = ''
    return answer


def extract_taste_profile(raw_text: str):
    answer = ""
    lower_text = raw_text.lower()
    splitted_text = lower_text.split(' ')
    if len(splitted_text) == 1:
        answer = splitted_text[0]
    elif len(splitted_text) > 1 and len(splitted_text) < 6:
        answer = ";".join(splitted_text)
    else:
        answer = ''
    return answer


def get_cooking_style(raw_text: str):
    lower_text = raw_text.lower()
    return lower_text


def get_meal_type(raw_text: str):
    lower_text = raw_text.lower()
    return lower_text


def get_price_estimation(raw_text: str):
    answer = ''
    lower_text = raw_text.lower()
    numbers = re.findall(r'[1-3]', lower_text, re.IGNORECASE)
    if len(numbers) == 1:
        answer = numbers[0]
    elif len(numbers) > 1:
        answer = max(numbers)
    else:
        answer = ''
    return answer


def check_name(file_name: str):
    if "failed.json" in file_name:
        return False
    else:
        return True

# load the information files


def load_files_and_extract_information(pattern: str,
                                       extraction_func: Callable[[str], str] = None):
    list_files_to_process = glob.glob(pattern)
    # filter filenames
    list_files_to_process = list(
        filter(lambda x: check_name(x), list_files_to_process))
    print(f"Files to process: {list_files_to_process}")
    raw_text = {}
    processed_dict = {}
    # load the information
    for file_path in list_files_to_process:
        print(f"Processing file {file_path}")
        try:
            # extract information
            data_raw = None
            with open(file_path, 'r') as fp:
                data_raw = json.load(fp)
            if data_raw is not None:
                for k in data_raw.keys():
                    raw_text[k] = data_raw[k]["message"]["content"]
                    if extraction_func is not None:
                        processed_dict[k] = extraction_func(
                            data_raw[k]["message"]["content"])
        except Exception as e:
            print(f"Error {e} with file: {file_path}")
            continue
    return processed_dict, raw_text

# process the information files


def processing_info_in_files(recipe_df: pd.DataFrame):
    patterns = ["full_taste_*.json",
                "full_cooking_*.json",
                "full_meal_type_*.json",
                "full_prep_*.json",
                "full_price_*.json",
                ]
    recipes_df = recipe_df.copy()
    for idx, p in enumerate(patterns):
        if idx == 0:
            dict_result, dict_raw = load_files_and_extract_information(p,
                                                                       extraction_func=extract_taste_profile)
            print(f"dict results: {dict_result}")
            data = [tuple for tuple in dict_result.items()]
            df_taste = pd.DataFrame(data, columns=["title", "taste"])
        elif idx == 1:
            dict_result, dict_raw = load_files_and_extract_information(p,
                                                                       extraction_func=get_cooking_style)
            data = [tuple for tuple in dict_result.items()]
            df_cooking_style = pd.DataFrame(
                data, columns=["title", "cooking_style"])
        elif idx == 2:
            dict_result, dict_raw = load_files_and_extract_information(p,
                                                                       extraction_func=get_meal_type)
            data = [tuple for tuple in dict_result.items()]
            df_meal_type = pd.DataFrame(data, columns=["title", "meal_type_1"])
        elif idx == 3:
            dict_result, dict_raw = load_files_and_extract_information(p,
                                                                       extraction_func=extract_prep_time)
            data = [tuple for tuple in dict_result.items()]
            df_preparation_time = pd.DataFrame(
                data, columns=["title", "prep_time"])
        else:
            dict_result, dict_raw = load_files_and_extract_information(p,
                                                                       extraction_func=get_price_estimation)
            data = [tuple for tuple in dict_result.items()]
            df_precio = pd.DataFrame(data, columns=["title", "price"])
    # merge the final result and save
    df_list = [recipes_df,
               df_taste,
               df_cooking_style,
               df_meal_type,
               df_preparation_time,
               df_precio]
    # set title as index
    for i in range(1, len(df_list)):
        print(f"processing data frame: {df_list[i].shape}")
        df_list[i].set_index("title", inplace=True)
    # join data frames
    final_df = df_list[0]
    for i in range(1, len(df_list)):
        final_df = final_df.join(df_list[i])
    # save the final df
    final_df.to_csv("full_reyhan_format_old_recipes.csv", sep="|")
    print("success")


def process_nutritional_information(recipe_df: pd.DataFrame,
                                    pattern: str):
    recipes_df = recipe_df.copy()
    dict_results, dict_raw = load_files_and_extract_information(pattern=pattern,
                                                                extraction_func=extract_numeric_values_grams)
    # transform to df
    data = [tuple for tuple in dict_raw.items()]
    df_raw = pd.DataFrame(data, columns=["title", "raw_text"])
    data = [tuple for tuple in dict_results.items()]
    return df_raw


if __name__ == "__main__":
    # load recipes dataset
    recipe_df = pd.read_csv("/home/victor/Documents/Expectation_data_generation/src/meals_collection/df_unique_recipes_6865.csv",
                            sep="|",
                            index_col=0)
    print(recipe_df.head(4))
    base_path = "/home/victor/Documents/Expectation_data_generation/"
    processing_info_in_files(recipe_df)
    patterns = [
        # "full_carbs_fixed_new_rec*.json",
        # "full_fat_fixed_new_rec*.json",
        # 'full_carbs_fixed_new_rec*.json'
        'full_ingredients_187_fixed*.json'
        # 'full_fat_193_final*.json'
        # 'full_carbs_80_final*.json'
        # 'full_allergies_fixed_latest*.json'
        # 'full_calories_fixed_last*.json'
        # 'full_allergens_700_rec*.json'
        # 'full_meal_type_700_rec*.json'
        # "full_prep_700_rec*.json",
        # "full_fiber_fixed_new_rec*.json",
        # "full_protein_fixed_new_rec*.json"
        # "full_calories_700_rec*.json"
        # "full_carbs_700_rec*.json",
        # "full_fat_700_rec*.json",
        # "full_fiber_700_rec*.json",
        # "full_protein_700_rec*.json",
        # "full_taste_700_rec*.json",
        # "full_price_700_rec*.json",
        # "full_prep_time_700_rec*.json"
        # 'full_cooking_style_700_rec*.json',
        # "full_cuisine_700_rec*.json"
    ]
    for pattern in patterns:
        print(f"Pattern: {pattern}")

        df_temp = process_nutritional_information(recipe_df,
                                                  pattern=os.path.join(base_path,
                                                                       pattern
                                                                       ))
        name = "_".join(pattern.split("_")[1:3])
        print(f"Possible name: {name}")
        df_temp.to_csv(f"df_{name}_raw_new.csv", sep="|")
