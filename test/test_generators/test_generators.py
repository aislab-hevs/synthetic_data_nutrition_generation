import pytest
from faker import Faker
import numpy as np
import configparser
import os
from unittest.mock import Mock
import pandas as pd
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
                                                  generate_probabilities_for_flexi,
                                                  assign_probabilities,
                                                  generate_cultural_data,
                                                  generate_preferences_data,
                                                  calculate_basal_metabolic_rate,
                                                  calculate_daily_calorie_needs,
                                                  define_daily_calorie_plan,
                                                  generate_diet_plan,
                                                  generate_therapy_data,
                                                  simulate_final_result,
                                                  allergy_searcher,
                                                  generate_meals_plan_per_user,
                                                  generate_recommendations,
                                                  create_a_summary_table,
                                                  run_full_simulation,
                                                  Gender,
                                                  BMI_constants,
                                                  NutritionGoals,
                                                  ActivityLevel
                                                  )

from synthetic_data_generation.constants import (person_entity,
                                                 user_entity,
                                                 BMI_probabilities_dict)


@pytest.fixture
def faker():
    fake = Faker()
    return fake


@pytest.fixture
def personal_data():
    user_data = generate_personal_data(gender_probabilities={"M": 0.2, "F": 0.8},
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


def test_generate_probabilities_for_flexi():
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
    probabilities = generate_probabilities_for_flexi(flexi_probabilities)
    assert flexi_probabilities == probabilities


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
                                            df_personal_data=personal_data)
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
    diet_plan = generate_diet_plan(weight=82,
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
