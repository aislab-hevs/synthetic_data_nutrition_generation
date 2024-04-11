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
    # Load environment variables
    load_dotenv()
    # get api key from environment variable
    api_key = os.getenv("API_KEY")
    # create a openai object
    openai.api_key = api_key
    # get the list of available models
    models = openai.Model.list()
    # print available models
    # print(f"Available models: {models}")
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
                                additional_data: bool = True,
                                list_of_recipes: List[str] = None):
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
        if list_of_recipes is None:
            recipes_list = list(np.unique(filtered_titles))
        else:
            recipes_list = list_of_recipes

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
            with open(f"Additional_info_recipes_missing_{start_index}_{end_index}_{i}.json", 'w') as fp:
                json.dump(ingredients_dict, fp)


def get_taste_profile(recipe_df: pd.DataFrame,
                      title_colum: str,
                      ingredients_col: str,
                      file_name: str,
                      type_query: str,
                      model: str = "gpt-3.5-turbo-16k",
                      max_recipes: int = -1):
    # Load environment variables
    load_dotenv()
    # get api key from environment variable
    api_key = os.getenv("API_KEY")
    # create a openai object
    openai.api_key = api_key
    # data useful for recipes
    fail_recipes = []
    processed_recipes = {}
    sub_df = recipe_df.loc[:, [title_colum, ingredients_col]]
    if type_query == "taste":
        query_template = """
        Indicate the key taste profile between sweet, bitter, salty, sour and umami, in the following recipe. Answer with only one word.

        {ingredients_list}
        """
    elif type_query == "price":
        query_template = """
        Indicate from 1 to 3 (with 3 being expensive) the estimated cost of the following recipe.
        Answer only with one number, given the following ingredients:

        {ingredients_list}
        """
    elif type_query == "ingredients":
        query_template = """
        Please give me the list of ingredients with quantities for 2 portions for
        the following recipes: 
        
        Recipe name: {rec_title}
        """
    elif type_query == "prep_time":
        query_template = """
        Indicate the preparation time in minutes of the following recipe. Answer with only the estimate time 
        in minutes in one number:
        
        Recipe name: {rec_title}
        
        Ingredients: {ingredients_list}
        """
    elif type_query == "cooking_style":
        query_template = """
        Indicate the cooking style (Sauteed, Slow-cooked, backed, fired, etc) of 
        the following recipe. Answer with only one word:
        Recipe name: {rec_title}

        Ingredients: {ingredients_list}
        """
    # cuisine type
    elif type_query == "cuisine":
        query_template = """
        Indicate the cuisine type of 
        the following recipe with name: {rec_title} and ingredient 
        list: {ingredients_list}. Please answer with only one word.
      
        """
    # carbs
    elif type_query == "carbs":
        query_template = """
        Please give me an estimation of the total carbohydrates for a 
        recipe with these ingredients and quantities, please answer only with 
        the total number: 

        {ingredients_list}
        """
    # get preparation instructions
    elif type_query == "preparation":
        query_template = """
        Please give me the preparation or cooking instructions for the recipe entitled: 
        {rec_title}, and with the following ingredients:
        {ingredients_list}
        """
    # Calories
    elif type_query == "calories":
        query_template = """
        Indicate the total estimate calories 
        from the following recipe given the ingredients and quantities bellow.  
        Answer with only one total number. 

        {ingredients_list}
        """
    # fiber
    elif type_query == "fiber":
        query_template = """
        Please give me an estimation of the total fiber for a recipe with these 
        ingredients and quantities, please answer only with the total number: 

        {ingredients_list}
        """
    # fat
    elif type_query == "fat":
        query_template = """
        Please give me an estimation of the total fat for a recipe with these ingredients and quantities, 
        please answer only with the total number: 

        {ingredients_list}
        """
    # protein
    elif type_query == "protein":
        query_template = """
        Please give me an estimation of the total protein for a recipe with these ingredients and quantities, 
        please answer only with the total number: 

        {ingredients_list}
        """
    # Allergens
    elif type_query == "allergens" or type_query == "allergies":
        query_template = """
        Classify the following recipe in one of the following allergy categories: 
        Milk, Eggs, Fish (e.g., bass, flounder, cod), 
        Crustacean shellfish (e.g., crab, lobster, shrimp),
        Tree nuts (e.g., almonds, walnuts, pecans), Peanuts,
        Wheat, Soybeans, Sesame, None. Please answer with one word. 

        {ingredients_list}
        """
    # Cultural factor
    elif type_query == "cultural_restriction":
        query_template = """
        Classify the following recipe in one of the following categories:
        veggie, vegetarian, halal, kosher, meat-based, 
        grain-based, etc. Please answer with only one word.  
        Recipe title: {rec_title} with ingredients: {ingredients_list}
        """
    else:
        raise Exception(f"Not valid query: {type_query}")
    # process the sub array
    row_idx = 0
    for row in sub_df.index:
        print(f"processing row {row_idx} of {len(sub_df)}...")
        try:
            # make query
            recipe_title = sub_df.loc[row, title_colum]
            ingredients = sub_df.loc[row, ingredients_col]
            if max_recipes > 0 and row > max_recipes:
                break
            raw_response = generate_nutrition_plans(
                user_text=query_template.format(ingredients_list=ingredients,
                                                rec_title=recipe_title),
                samples_to_generated=1,
                model=model)
            processed_recipes[recipe_title] = dict(raw_response["choices"][0])
        except Exception as e:
            print(f"Error during processing row{row}")
            print(f"error: {e}")
            fail_recipes.append(row)
            continue
        row_idx += 1
    # save data
    with open(f"{file_name}.json", 'w') as fp:
        json.dump(processed_recipes, fp)
    with open(f"{file_name}_failed.json", 'w') as fp:
        json.dump(fail_recipes, fp)


if __name__ == '__main__':
    # Create a parser object
    print("Starting...")
    # get ingredients and preparation steps for missing recipes
    # query = """Hi! Please can you give me the ingredient list and the preparation steps
    # for the following recipe: {recipe_name}
    # """
    # # load the list
    # recipe_list = None
    # with open("/home/victor/Documents/Expectation_data_generation/src/meals_collection/recipes_to_fix_2.json", "r") as fp:
    #     recipe_list = json.load(fp)
    # if recipe_list is not None:
    #     print(f"recipes to process: {len(recipe_list)}")
    #     # query and get results
    #     answer_dict = generate_ingredients_for_recipes(query,
    #                                                    recipe_list,
    #                                                    model="gpt-4-1106-preview")
    #     # save dict
    #     with open("missing_recipes_ingredients_raw_2.json", "w") as fa:
    #         json.dump(answer_dict, fa)

    # get additional data on recipes

    parser = argparse.ArgumentParser(
        description="Enrich recipes with command-line arguments.")

    # Define command-line arguments
    parser.add_argument('--start_index', type=int, default=0,
                        help='Start index (default: 0)')
    parser.add_argument('--end_index', type=int, default=9000,
                        help='End index (default: 9000)')
    parser.add_argument('--chunk_size', type=int, default=500,
                        help='Chunk size (default: 500)')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-1106',
                        help='Model name (default: gpt-3.5-turbo-1106)')
    parser.add_argument('--title_column', type=str, default='Title',
                        help='Title column name in the dataset')
    parser.add_argument('--ingredients_column', type=str, default='ingredients',
                        help='Ingredients column name in the dataset')
    parser.add_argument('--file_name', type=str, default='meal_type_new_recipes',
                        help='Target file name.')
    parser.add_argument('--query', type=str, default='',
                        help='Query to execute.')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parsed arguments
    start_index = args.start_index
    end_index = args.end_index
    chunk_size = args.chunk_size
    model = args.model
    title_col = args.title_column
    ingredients_col = args.ingredients_column
    target_file_name = args.file_name
    query_type = args.query

    # load dataset
    # run for column fixing
    # data_loading = pd.read_csv(f"/home/victor/Documents/Expectation_data_generation/src/meals_collection/df_fix_final_col_{query_type}.csv",
    #                            sep="|",
    #                            index_col=0)
    data_loading = pd.read_csv("/home/victor/Documents/Expectation_data_generation/src/meals_collection/df_ingredients_187_to_fix.csv",
                               sep="|",
                               index_col=0)
    print(f"size: {data_loading.shape}")
    # load index dictionary
    # index_dict = None
    # with open("/home/victor/Documents/Expectation_data_generation/src/meals_collection/dict_indexes_to_fix.json",
    #           'r') as fp:
    #     index_dict = json.load(fp)
    # if index_dict is None:
    #     print(f"Not data load")
    # else:
    #     print(f"dict_size: {len(index_dict)}")
    #     # choose columns to fix
    #     index_list = index_dict.get(query_type, None)
    #     if index_list is not None:
    #         data_loading = data_loading.loc[index_list, :]
    #         data_loading.reset_index(drop=True, inplace=True)
    #         print(f"Data shape: {data_loading.shape}")
    # call the function
    for i in range(start_index, end_index, chunk_size):
        print(f"Processing batch: {i}... query: {query_type}")
        sub_df = data_loading.iloc[i:i+chunk_size, :]
        get_taste_profile(recipe_df=sub_df,
                          title_colum=title_col,
                          ingredients_col=ingredients_col,
                          type_query=query_type,
                          file_name=target_file_name+f"{i}_{i+chunk_size}",
                          model=model)
    # # Define command-line arguments
    # parser.add_argument('--start_index', type=int, default=0,
    #                     help='Start index (default: 0)')
    # parser.add_argument('--end_index', type=int, default=9000,
    #                     help='End index (default: 9000)')
    # parser.add_argument('--chunk_size', type=int, default=500,
    #                     help='Chunk size (default: 500)')
    # parser.add_argument('--model', type=str, default='gpt-3.5-turbo-16k',
    #                     help='Model name (default: gpt-3.5-turbo-16k)')
    # parser.add_argument('--additional_data', action='store_true',
    #                     help='Include additional data (default: True)')
    # parser.add_argument('--load_file', action='store_true',
    #                     help='Include list data (default: False)')

    # # Parse the command-line arguments
    # args = parser.parse_args()

    # # Access the parsed arguments
    # start_index = args.start_index
    # end_index = args.end_index
    # chunk_size = args.chunk_size
    # model = args.model
    # additional_data = args.additional_data
    # missing_data = args.load_file

    # data_loading = None
    # if missing_data:
    #     with open("/home/victor/Documents/Expectation_data_generation/src/meals_collection/missing_recipes.txt") as fp:
    #         data_loading = json.load(fp)
    #     print(f"loaded files: {len(data_loading)}")

    # # Call the enrich_recipes function with the captured parameters
    # enrich_recipes_numeric_data(start_index=start_index,
    #                             end_index=end_index,
    #                             chunk_size=chunk_size,
    #                             model=model,
    #                             additional_data=additional_data,
    #                             list_of_recipes=data_loading)

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

    # get new recipes
    # user_text = """
    # Give me {n_recipes} different recipes from {place} with ingredients,
    # preparation steps, allergens, and carbohydrates, protein, fat, fiber and
    # calories per 100g portion and also tell me if it is appropriate for
    # breakfast, lunch or dinner.
    # """
    # new_recipes = generate_nutrition_plans(user_text.format(n_recipes=15,
    #                                                         place="China"),
    #                                        samples_to_generated=10,
    #                                        model="gpt-4-1106-preview")
    # with open("new_china_recipes_1.json", "w") as fp:
    #     json.dump(new_recipes, fp)
    print("Finished")
