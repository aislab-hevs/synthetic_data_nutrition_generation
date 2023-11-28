import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import openai
import json
from typing import List
import argparse


def generate_nutrition_plans(user_text: str,
                             samples_to_generated: int = 10,
                             model: str = "gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        n=samples_to_generated,
        messages=[
            {"role": "system", "content": "Generating recipes"},
            {"role": "user", "content": user_text}
        ]
    )
    return response


def generate_ingredients_for_recipes(query_template: str,
                                     recipes: List[str],
                                     model: str = "gpt-4-1106-preview"):
    answer = {}
    for recipe in recipes:
        try:
            print(f"Processing recipe: {recipe}")
            current_recipe = recipe.strip()
            processed_query = query_template.format(recipe_name=current_recipe)
            response = generate_nutrition_plans(processed_query,
                                                samples_to_generated=1,
                                                model=model)
            dict_text = dict(response["choices"][0])
            answer[recipe] = dict_text["message"]["content"]
        except Exception as e:
            print(f"Exception: {e} for recipe: {recipe}")
            answer[recipe] = ""
            continue
    return answer


def generate_numeric_data_for_recipes(query_template: str,
                                      recipes: List[str],
                                      ingredients: List[str],
                                      model: str = "gpt-4-1106-preview"):
    answer = {}
    for i in range(0, len(recipes)):
        try:
            print(f"Processing recipe: {recipes[i]}")
            current_recipe = recipes[i].strip()
            current_ingredients = ingredients[i].strip()
            processed_query = query_template.format(recipe_name=current_recipe,
                                                    recipe_ingredients=current_ingredients
                                                    )
            response = generate_nutrition_plans(processed_query,
                                                samples_to_generated=1,
                                                model=model)
            dict_text = dict(response["choices"][0])
            answer[recipes[i]] = dict_text["message"]["content"]
        except Exception as e:
            print(f"Exception: {e} for recipe: {recipes[i]}")
            answer[recipes[i]] = ""
            continue
    return answer


def transform_title(text: str):
    new_text = text.strip()
    if "-" in text:
        new_text = text.split("-")[0]
    return new_text


def enrich_recipes(model: str = "gpt-4-1106-preview",
                   start_index: int = 0,
                   end_index: int = 1000,
                   chunk_size: int = 50,
                   additional_data: bool = True):
    # Load environment variables
    load_dotenv()
    # get api key from environment variable
    api_key = os.getenv("API_KEY")
    # create a openai object
    openai.api_key = api_key
    # get the list of available models
    models = openai.Model.list()
    # print available models
    print(f"Available models: {models}")
    # define the query template
    query_ingredients = """Please give me the ingredients and preparation steps, 
    separated by ';' for the following recipe {recipe_name}"""
    query_additional_data = """
    Please give the carbohydrates, protein, fat, fiber and calories content per 
    portion of the following recipe: {recipe_name}
    """
    # load recipes file
    recipes_ds = pd.read_csv("/home/victor/Documents/Expectation_data_generation/src/recipes/processed_recipes_dataset_id.csv",
                             index_col=0,
                             sep="|")
    # process titles
    new_titles = recipes_ds["title"].apply(lambda x: transform_title(x))
    if additional_data:
        filtered_titles = list(
            filter(lambda x: len(x) > 2, new_titles.tolist()))
        recipes_list = list(np.unique(filtered_titles))
        print(f"Number of unique recipes: {len(recipes_list)}")
        ingredients_dict = {}
        for i in range(start_index, end_index, chunk_size):
            print(f"Process chunk: {i}")
            ingredients_dict = generate_ingredients_for_recipes(query_additional_data,
                                                                recipes=recipes_list[i:i +
                                                                                     chunk_size],
                                                                model=model)
            with open(f"additional_info_recipes_{start_index}_{end_index}_{i}.json", 'w') as fp:
                json.dump(ingredients_dict, fp)
    else:
        # generate additional data
        mask = recipes_ds["calories"] > 0
        recipes_ds = recipes_ds.loc[mask, :]
        unique_recipes = np.unique(recipes_ds["title"])
        print(f"Number of unique recipes: {len(unique_recipes)}")
        mask = [
            True if r in unique_recipes else False for r in recipes_ds["title"].tolist()]
        unique_ds = recipes_ds.loc[mask, :]
        print(unique_ds.shape)
        print(unique_ds.head(4))

        recipes_list = unique_ds["title"].tolist()
        chunk_size = 50
        ingredients_dict = {}
        for i in range(start_index, end_index, chunk_size):
            print(f"processing chunk {i}")
            ingredients_dict = generate_ingredients_for_recipes(query_ingredients,
                                                                recipes=recipes_list[i:i +
                                                                                     chunk_size],
                                                                model=model)
            with open(f"Complementary_recipes_{start_index}_{end_index}_{i}.json", 'w') as fp:
                json.dump(ingredients_dict, fp)


def enrich_recipes_numeric_data(model: str = "gpt-4-1106-preview",
                                start_index: int = 0,
                                end_index: int = 1000,
                                chunk_size: int = 50,
                                additional_data: bool = True):
    # Load environment variables
    load_dotenv()
    # get api key from environment variable
    api_key = os.getenv("API_KEY")
    # create a openai object
    openai.api_key = api_key
    # define the query template
    query_additional_data = """
    Please give the carbohydrates, protein, fat, fiber and calories content per 
    portion of 100g of the following recipe: {recipe_name} which contains the following 
    ingredients: {recipe_ingredients}
    """
    # load recipes file
    recipes_ds = pd.read_csv("/home/victor/Documents/Expectation_data_generation/src/meals_collection/extended_recipes_partial.csv",
                             sep="|")
    # recipes successfully loaded
    print(f"loaded recipes: {recipes_ds.shape}")
    # process titles
    recipes_ds["title"] = recipes_ds["title"].apply(
        lambda x: transform_title(x))
    if additional_data:
        filtered_titles = list(
            filter(lambda x: len(x) > 2, recipes_ds["title"].tolist()))
        recipes_list = list(np.unique(filtered_titles))

        ingredients_ds = recipes_ds[recipes_ds["title"].isin(recipes_list)]
        ingredients_ds = ingredients_ds.drop_duplicates(subset="title")
        print(f"Number of unique recipes: {len(recipes_list)}")
        print(f"Selected data frame size: {ingredients_ds.shape}")
        recipes_list = ingredients_ds["title"].tolist()
        ingredient_list = ingredients_ds["ingredients"].tolist()
        ingredients_dict = {}
        for i in range(start_index, end_index, chunk_size):
            print(f"Process chunk: {i}")
            ingredients_dict = generate_numeric_data_for_recipes(query_additional_data,
                                                                 recipes=recipes_list[i:i +
                                                                                      chunk_size],
                                                                 ingredients=ingredient_list[i:i +
                                                                                             chunk_size],
                                                                 model=model
                                                                 )
            with open(f"Additional_info_recipes_{start_index}_{end_index}_{i}.json", 'w') as fp:
                json.dump(ingredients_dict, fp)


if __name__ == '__main__':
    # Create a parser object
    print("Starting...")
    parser = argparse.ArgumentParser(
        description="Enrich recipes with command-line arguments.")

    # Define command-line arguments
    parser.add_argument('--start_index', type=int, default=0,
                        help='Start index (default: 0)')
    parser.add_argument('--end_index', type=int, default=9000,
                        help='End index (default: 9000)')
    parser.add_argument('--chunk_size', type=int, default=500,
                        help='Chunk size (default: 500)')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-16k',
                        help='Model name (default: gpt-3.5-turbo-16k)')
    parser.add_argument('--additional_data', action='store_true',
                        help='Include additional data (default: True)')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parsed arguments
    start_index = args.start_index
    end_index = args.end_index
    chunk_size = args.chunk_size
    model = args.model
    additional_data = args.additional_data

    # Call the enrich_recipes function with the captured parameters
    enrich_recipes_numeric_data(start_index=start_index,
                                end_index=end_index,
                                chunk_size=chunk_size,
                                model=model,
                                additional_data=additional_data)

    # parser = argparse.ArgumentParser(
    #     description="Generate additional data for recipes.")
    # # Define command-line arguments
    # parser.add_argument('start_index', type=int, help='start_index')
    # parser.add_argument('end_index', type=int, help='End index')
    # parser.add_argument('chunk_size', type=int, help='Chunk size')
    # parser.add_argument('model', type=str, help='model_to_use')
    # # parse arguments
    # args = parser.parse_args()
    # enrich_recipes(start_index=args.start_index,
    #                end_index=args.start_index,
    #                chunk_size=args.chunk_size,
    #                model=args.model,
    #                additional_data=True)
    print("Finished")
