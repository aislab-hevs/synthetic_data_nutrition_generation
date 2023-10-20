import pytest
from faker import Faker
import numpy as np
import configparser
import os
from unittest.mock import Mock
import pandas as pd
import pathlib
from synthetic_data_generation.generators import (create_name_surname,
                                                  generate_country,
                                                  generate_email_from_name,
                                                  password_generation,
                                                  generate_age_range,
                                                  generate_localization,
                                                  generate_personal_data,
                                                  choose_one_from_list,
                                                  calculate_weight_from_height,
                                                  generate_user_life_style_data,
                                                  generate_health_condition_data,
                                                  define_user_goal_according_BMI,
                                                  generate_user_goals,
                                                  assign_probabilities,
                                                  generate_cultural_data,
                                                  generate_preferences_data,
                                                  calculate_basal_metabolic_rate,
                                                  calculate_daily_calorie_needs,
                                                  define_daily_calorie_plan,
                                                  generate_diet_plan,
                                                  generate_therapy_data,
                                                  generate_daily_calories_requirement_according_next_BMI,
                                                  distribute_calories_in_meal,
                                                  generate_meals_plan_per_user,
                                                  generate_user_simulation,
                                                  generate_allergy_oriented_food_dataset,
                                                  generate_cultural_factor_oriented_dataset,
                                                  generate_meal_type_oriented_dataset,
                                                  generate_delta_values,
                                                  generate_recommendations,
                                                  generate_food_day,
                                                  create_a_summary_table,
                                                  run_full_simulation,
                                                  Gender,
                                                  BMI_constants,
                                                  NutritionGoals,
                                                  ActivityLevel
                                                  )

from synthetic_data_generation.default_inputs import (person_entity,
                                                      user_entity,
                                                      BMI_probabilities_dict,
                                                      meal_time_distribution,
                                                      age_probabilities_dict,
                                                      meals_proba_dict,
                                                      meals_calorie_dict,
                                                      gender_probabilities_dict,
                                                      allergies_probability_dict,
                                                      flexi_probabilities_dict,
                                                      food_restriction_probability_dict,
                                                      place_proba_dict,
                                                      social_situation_proba_dict,
                                                      delta_distribution_dict,
                                                      cultural_query_text
                                                      )


@pytest.fixture
def faker():
    fake = Faker()
    return fake


@pytest.fixture
def personal_data():
    user_data = generate_personal_data(gender_probabilities={"M": 0.2, "F": 0.8},
                                       age_probabilities=age_probabilities_dict,
                                       num_users=20,
                                       person_entity=person_entity,
                                       )
    return user_data


@pytest.fixture
def BMI_proba():
    BMI_values = ["underweight", "healthy", "overweight", "obesity"]
    BMI_prob = [0.1, 0.3, 0.3, 0.3]
    BMI_probabilities = dict(zip(BMI_values, BMI_prob))
    return BMI_probabilities


@pytest.fixture
def allergies_proba():
    allergies = ["cow's milk", "eggs", "peanut", "soy",
                 "fish", "tree nuts", "shellfish", "wheat", "None"]
    allergies_probas = [0.075, 0.075, 0.075,
                        0.075, 0.075, 0.075, 0.075, 0.075, 0.4]
    allergies_probability_dict = dict(zip(allergies, allergies_probas))
    return allergies_probability_dict


@pytest.fixture
def user_life_style_data():
    p_data = generate_personal_data(gender_probabilities={"M": 0.2, "F": 0.8},
                                    age_probabilities=age_probabilities_dict,
                                    num_users=20,
                                    person_entity=person_entity,
                                    )
    bmi_proba = BMI_probabilities_dict
    df_user_entity = generate_user_life_style_data(p_data["userId"].tolist(),
                                                   user_entity=user_entity,
                                                   df_personal_data=p_data,
                                                   BMI_probabilities_dict=bmi_proba)
    return df_user_entity, p_data


@pytest.fixture
def flexi_probas():
    food_restrictions = ["vegan_observant",
                         "vegetarian_observant",
                         "halal_observant",
                         "kosher_observant",
                         "None"]
    flexi_probabilities = {
        "flexi_vegie": dict(zip(food_restrictions, [0.6, 0.2, 0.05, 0.05, 0.1])),
        # "flexi_vegetarian" : dict(zip(food_restrictions,[NN, 0.6, 0.05, 0.05, 0.1])),
        "flexi_vegetarian": dict(zip(food_restrictions, [0.0, 0.6, 0.05, 0.05, 0.3])),
        "flexi_halal": dict(zip(food_restrictions, [0.1, 0.2, 0.6, 0.0, 0.1])),
        "flexi_kosher": dict(zip(food_restrictions, [0.1, 0.1, 0.1, 0.6, 0.1]))
    }
    return flexi_probabilities


@pytest.fixture
def food_probas():
    food_restrictions = ["vegan_observant", "vegetarian_observant",
                         "halal_observant", "kosher_observant", "flexi_observant", "None"]
    food_restriction_probs = [0.2, 0.3, 0.05, 0.05, 0.1, 0.3]
    food_restriction_probability_dict = dict(
        zip(food_restrictions, food_restriction_probs))
    return food_restriction_probability_dict


@pytest.fixture
def load_recipes():
    # config = configparser.ConfigParser()
    # # get file location
    file_location = os.getenv(
        "TEST_CONFIG_FILE", default="configs/test_config.ini")
    print(f"file location: {file_location}")
    # config.read(file_location)
    # print(config.sections())
    # # read location
    # recipes_location = config['DEFAULT']['df_recipes']
    # df_recipes = pd.read_csv(recipes_location)
    # return df_recipes
    pass


@pytest.fixture
def get_food_db():
    file_path = pathlib.Path(
        "/home/victor/Documents/Expectation_data_generation/test/test_data/processed_recipes_dataset_id.csv")
    df_food = pd.read_csv(file_path, sep='|', index_col=0)
    return df_food


@pytest.fixture
def full_user_pipeline():
    print("Starting user generation")
    # Generate user data
    df_personal_data = generate_personal_data(num_users=5,
                                              age_probabilities=age_probabilities_dict,
                                              person_entity=person_entity,
                                              gender_probabilities=gender_probabilities_dict)
    # Generate user status
    df_user_entity = generate_user_life_style_data(df_personal_data["userId"].tolist(),
                                                   user_entity=user_entity,
                                                   df_personal_data=df_personal_data,
                                                   BMI_probabilities_dict=BMI_probabilities_dict)
    # Generate health conditions
    df_health_conditions = generate_health_condition_data(df_personal_data["userId"].tolist(),
                                                          allergies_probability_dict=allergies_probability_dict)

    # Generate user goals
    df_user_goals = generate_user_goals(df_personal_data["userId"].tolist(),
                                        df_user_entity=df_user_entity)

    # Generate cultural factors
    df_cultural_factors = generate_cultural_data(df_personal_data["userId"].tolist(),
                                                 food_restriction_probability_dict=food_restriction_probability_dict,
                                                 flexi_probability_dict=flexi_probabilities_dict)
    # Generate therapy
    df_treatment, df_user_data = generate_therapy_data(df_personal_data["userId"].tolist(),
                                                       df_personal_data=df_personal_data,
                                                       df_user_entity=df_user_entity,
                                                       df_user_goals=df_user_goals)
    # unify all the DataFrames
    df_user_join = df_user_data.merge(df_treatment, on="userId")
    df_user_join = df_user_join.merge(df_personal_data[["userId",
                                                        "country_of_origin",
                                                        "living_country",
                                                        "current_location"
                                                        ]], on="userId")
    df_user_join = df_user_join.merge(df_cultural_factors,  on="userId")
    df_user_join = df_user_join.merge(df_health_conditions,  on="userId")
    df_user_join = df_user_join.merge(
        df_user_entity[["userId",
                        "current_working_status",
                        "marital_status",
                        "ethnicity",
                        "BMI"]],  on="userId")
    # Generates meals plan
    meals_plan = generate_meals_plan_per_user(
        df_user_join["userId"].tolist(), meals_proba_dict)
    return df_user_join, meals_plan


def test_create_name_surname(faker):
    names = create_name_surname(Gender.male, faker)
    assert len(names) == 2


def test_generate_country(faker):
    countries = generate_country(samples=2,
                                 fake=faker)
    assert len(countries) == 2


def test_generate_email_from_name():
    email = generate_email_from_name("john", "doe")
    assert email == "john.doe@fake.com"


def test_password_generation():
    password = password_generation(8)
    assert len(password) == 8


def test_generate_age_range():
    age_range = generate_age_range([0.8, 0.2], ["20-30", "30-40"])
    assert age_range != None


def test_generate_localization(faker):
    localization = generate_localization(3, faker)
    assert len(localization) == 3


def test_generate_personal_data():
    user_data = generate_personal_data(gender_probabilities={"M": 0.2, "F": 0.8},
                                       age_probabilities=age_probabilities_dict,
                                       num_users=10,
                                       person_entity=person_entity,
                                       )
    assert user_data.shape[0] == 10


def test_choose_one_from_list():
    chosen = choose_one_from_list(list_values=["a", "b", "c", "d", "e", "f"],
                                  samples=3,
                                  probabilities=[0.4, 0.05,
                                                 0.05, 0.1, 0.2, 0.2],
                                  size=1)
    assert len(chosen) == 3


def test_calculate_weight_from_height():
    weight = calculate_weight_from_height(1.65,
                                          BMI_constants.overweight)
    assert np.round(weight, 2) == 76.23


def test_generate_user_life_style_data(personal_data, BMI_proba):
    list_users = personal_data['userId'].tolist()
    user_life_style = generate_user_life_style_data(list_user_id=list_users,
                                                    user_entity=user_entity,
                                                    BMI_probabilities_dict=BMI_proba,
                                                    df_personal_data=personal_data)
    assert user_life_style.shape[0] == 20


def test_generate_health_condition_data(personal_data, allergies_proba):
    list_users = personal_data['userId'].tolist()
    health_conditions = generate_health_condition_data(list_user_id=list_users,
                                                       allergies_probability_dict=allergies_proba)
    assert health_conditions.shape[0] == 20


def test_define_user_goal_according_BMI():
    user_goal = define_user_goal_according_BMI(BMI_constants.overweight)
    assert NutritionGoals.lose_weight == user_goal


def test_generate_user_goals(user_life_style_data):
    life_style, p_data = user_life_style_data
    list_users = life_style['userId'].tolist()
    user_goals = generate_user_goals(list_user_id=list_users,
                                     df_user_entity=life_style)
    assert user_goals.shape[0] == len(list_users)


def test_assign_probabilities(flexi_probas):
    flexi_factor = assign_probabilities(cultural_factor="flexi_observant",
                                        flexi_probability_dict=flexi_probas)
    assert flexi_factor != None


def test_generate_cultural_data(personal_data, flexi_probas, food_probas):
    list_users = personal_data['userId'].tolist()
    cultural_data = generate_cultural_data(list_user_id=list_users,
                                           food_restriction_probability_dict=food_probas,
                                           flexi_probability_dict=flexi_probas)
    assert cultural_data.shape[0] == len(list_users)


def test_generate_preferences_data(personal_data):
    list_users = personal_data["userId"].tolist()
    preferences = generate_preferences_data(list_user_id=list_users,
                                            df_personal_data=personal_data,
                                            meals_time_distribution=meal_time_distribution)
    assert preferences.shape[0] == len(list_users)


def test_calculate_basal_metabolic_rate():
    metabolic_basal_rate = calculate_basal_metabolic_rate(weight=82,
                                                          height=165,
                                                          age=32,
                                                          clinical_gender="M")
    assert np.round(1797.087, 2) == np.round(metabolic_basal_rate, 2)


def test_calculate_daily_calorie_needs():
    calory_needs = calculate_daily_calorie_needs(1696.0,
                                                 activity_level=ActivityLevel.moderate_active)
    assert calory_needs > np.round(1797.087, 2)


def test_define_daily_calorie_plan():
    daily_calorie_needs = 1797.087
    calorie_plan = define_daily_calorie_plan(nutrition_goal=NutritionGoals.lose_weight,
                                             daily_calorie_need=daily_calorie_needs)
    assert np.round(calorie_plan, 2) == np.round(daily_calorie_needs - 500, 2)


def test_generate_diet_plan():
    daily_calorie_needs = 1218.34
    diet_plan, _ = generate_diet_plan(weight=82,
                                      height=1.65,
                                      age_range="30-40",
                                      clinical_gender="M",
                                      activity_level=ActivityLevel.moderate_active,
                                      nutrition_goal=NutritionGoals.lose_weight
                                      )
    assert np.round(diet_plan, 2) == np.round(daily_calorie_needs, 2)


def test_generate_therapy_data(user_life_style_data):
    life_style, p_data = user_life_style_data
    list_users = life_style['userId'].tolist()
    user_goals = generate_user_goals(list_user_id=list_users,
                                     df_user_entity=life_style)
    df_therapy, df_user = generate_therapy_data(list_user_id=list_users,
                                                df_personal_data=p_data,
                                                df_user_goals=user_goals,
                                                df_user_entity=life_style
                                                )
    assert df_therapy.shape[0] == len(list_users)


def test_generate_daily_calories_requirement_according_next_BMI(user_life_style_data):
    life_style, p_data = user_life_style_data
    list_users = life_style['userId'].tolist()
    user_goals = generate_user_goals(list_user_id=list_users,
                                     df_user_entity=life_style)
    days_to_simulate = 20
    df_therapy, df_user = generate_therapy_data(list_user_id=list_users,
                                                df_personal_data=p_data,
                                                df_user_goals=user_goals,
                                                df_user_entity=life_style
                                                )
    # user dataset updated with next state
    bmi_conditions = [BMI_constants.underweight,
                      BMI_constants.healthy,
                      BMI_constants.overweight,
                      BMI_constants.obesity]
    results = \
        generate_daily_calories_requirement_according_next_BMI(
            df_therapy["current_daily_calories"],
            bmi_conditions,
            current_state=BMI_constants.overweight.value,
            next_state=BMI_constants.healthy.value,
            days_to_simulated=days_to_simulate)
    # print(results)
    assert len(results) == days_to_simulate


def test_generate_meals_plan_per_user():
    meals_plan = generate_meals_plan_per_user(users=['test_user'],
                                              probability_dict=meals_proba_dict)
    # print(meals_plan)
    assert len(meals_plan.keys()) == 5


def test_distribute_calories_in_meal():
    meals_plan = generate_meals_plan_per_user(users=['test_user'],
                                              probability_dict=meals_proba_dict)
    result = distribute_calories_in_meal(meals_plan=meals_plan,
                                         meals_calorie_distribution=meals_calorie_dict
                                         )

    # print(result)
    assert abs(sum(result.values()) - 1.0) <= 0.0001


def test_load_food_db(get_food_db):
    df_food = get_food_db
    # print(df_food.head(4))
    assert df_food is not None


def test_generate_cultural_factor_oriented_dataset(get_food_db):
    food_db = get_food_db
    cultural_ids = generate_cultural_factor_oriented_dataset(
        food_db=food_db
    )
    # print({k: len(cultural_ids[k]) for k in cultural_ids.keys()})
    assert len(cultural_ids.keys()) > 0


def test_generate_allergy_oriented_food_dataset(get_food_db):
    food_db = get_food_db
    allergy_ids = generate_allergy_oriented_food_dataset(
        food_db=food_db
    )
    # print({k: len(allergy_ids[k]) for k in allergy_ids.keys()})
    assert len(allergy_ids.keys()) > 0


def test_generate_meal_type_oriented_dataset(get_food_db):
    food_db = get_food_db
    meal_type_ids = generate_meal_type_oriented_dataset(
        food_db=food_db
    )
    # print({k: len(meal_type_ids[k]) for k in meal_type_ids.keys()})
    assert len(meal_type_ids.keys()) > 0


@pytest.mark.parametrize("chosen_dist, days_to_simulate", [
    ("Normal", 10),
    ("Normal", 20),
    ("Bimodal", 10),
    ("Bimodal", 20)
])
def test_generate_delta_values(chosen_dist: str,
                               days_to_simulate: int):
    result = generate_delta_values(chose_dist=chosen_dist,
                                   parameters=delta_distribution_dict[chosen_dist],
                                   size=days_to_simulate)
    # print(result)
    assert len(result) == days_to_simulate

# @pytest.mark.parametrize("full_user_pipeline", [1, 2], indirect=True)


def test_generate_food_day(get_food_db):
    selected_food_df = get_food_db
    result = generate_food_day(
        selected_food_df=selected_food_df,
        calories=600
    )
    # print(f"Dataframe size: {result.shape}")
    print(result)
    assert result is not None


def test_generate_user_simulation(full_user_pipeline, get_food_db):
    # test the new function to generate data.
    df_user_join, meals_plan = full_user_pipeline
    food_db = get_food_db
    bmi_conditions = [BMI_constants.underweight,
                      BMI_constants.healthy,
                      BMI_constants.overweight,
                      BMI_constants.obesity]
    cultural_ids = generate_cultural_factor_oriented_dataset(
        food_db=food_db
    )
    allergy_ids = generate_allergy_oriented_food_dataset(
        food_db=food_db
    )
    meal_type_ids = generate_meal_type_oriented_dataset(
        food_db=food_db
    )
    chose_dist = "Bimodal"
    df_user_join["next_BMI"] = df_user_join["BMI"]
    result = generate_user_simulation(
        user_id=df_user_join["userId"][0],
        df_user=df_user_join,
        food_db=food_db,
        meals_probability_dict=meals_proba_dict,
        meals_calorie_distribution=meals_calorie_dict,
        allergies_food_ids=allergy_ids,
        cultural_food_ids=cultural_ids,
        meal_type_food_ids=meal_type_ids,
        place_probabilities=place_proba_dict,
        social_situation_probabilities=social_situation_proba_dict,
        chose_dist=chose_dist,
        meals_time_dict=meal_time_distribution,
        delta_dist_params=delta_distribution_dict[chose_dist],
        dict_flexi_probas=flexi_probabilities_dict,
        days_to_simulated=10,
        bmi_conditions=bmi_conditions)
    # print(result)
    assert result is not None


def test_generate_recommendations(full_user_pipeline, get_food_db):
    df_user_join, meals_plan = full_user_pipeline
    food_db = get_food_db
    meals_plan = generate_meals_plan_per_user(
        df_user_join["userId"].tolist(),
        meals_proba_dict)
    # Execute the function
    chosen_dist = "Bimodal"
    tracking_result = generate_recommendations(
        df_user=df_user_join,
        transition_matrix=None,
        df_recipes_db=food_db,
        place_probabilities=place_proba_dict,
        social_situation_probabilities=social_situation_proba_dict,
        meals_plan=meals_plan,
        chose_dist=chosen_dist,
        delta_dist_params=delta_distribution_dict[chosen_dist],
        flexi_probabilities_dict=flexi_probabilities_dict,
        meals_calorie_dict=meals_calorie_dict,
        meals_time_dict=meal_time_distribution,
        days_to_simulated=10,
        progress_bar=None
    )
    print(tracking_result)
    assert tracking_result is not None
