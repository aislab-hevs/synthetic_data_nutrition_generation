import pandas as pd
import numpy as np
from faker import Faker
from enum import Enum
from typing import List, Any, Tuple, Dict
import string
import uuid
import seaborn as sns
import matplotlib.pyplot as plt
from prettytable import PrettyTable, ALL, FRAME
from html import escape
from scipy.stats import bernoulli


# Constants
person_entity = {
    "userId": str,
    "username": str,
    "password": str,
    "email": str,
    "name": str,
    "surname": str,
    "clinical_gender": ["M", "F"],
    "current_location": [],
    "age_range": ["20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-100"],
    "living_country": [],
    "country_of_origin": []
}

user_entity = {
    "current_working_status": ["Half-time-worker", "Full-time-worker", "Self-employee", "Unemployed"],
    "marital_status": ["Single", "Married"],
    "life_style": ["Sedentary", "Lightly active", "Moderately active", "Very active"],
    "weight": [],
    "ethnicity": ["White", "Black", "Latino", "Asian"],
    "height": []
}

cultural_factors = {
    "vegan_observant": [True, False],
    "vegetarian_observant": [True, False],
    "halal_observant": [True, False],
    "kosher_observant": [True, False],
    "religion_observant": [True, False],
    "drink_limitation": [True, False],
    "pescatarian_observant": [True, False],
    "religion": [],
    "food_limitation": []
}

sustainability = {
    "environmental_harm": [],
    "eco_score": [],
    "co2_food_print": [],
    "recyclable_packaging": []
}

actions = {
    "action_type": [],
    "location": [],
    "action_date": []
}

preferences = {
    "breakfast_time": [],
    "lunch_time": [],
    "dinner_time": []
}

health_conditions = {
    "food_allergies": []
}

user_goals = {
    "user_goals": ["loss_weight", "fit", "food_restrictions"]
}

cultural_factors = {
    "cultural_factors": []
}

diet = {
    "diet_daily_calories": [],
    "calorie_deficit": []
}

# Classes


class Gender(str, Enum):
    male = "M"
    female = "F"


class BMI_constants(str, Enum):
    underweight = "underweight"
    healthy = "healthy"
    overweight = "overweight"
    obesity = "obesity"


class NutritionGoals(str, Enum):
    lose_weight = "lose_weight"
    maintain_fit = "maintain_fit"
    gain_weight = "gain_weight"


class ActivityLevel(str, Enum):
    sedentary = "Sedentary"
    light_active = "Lightly active"
    moderate_active = "Moderately active"
    very_active = "Very active"

# Functions


def create_name_surname(gender: str, fake: Faker) -> str:
    if gender == Gender.male:
        names = fake.name_male()
    else:
        names = fake.name_female()
    return names.split(" ")


def generate_country(samples, fake: Faker) -> List:
    return list(map(lambda x: fake.country(), range(samples)))


def generate_email_from_name(name: str, surname: str, domain: str = "fake.com"):
    return f"{name.lower()}.{surname.lower()}@{domain.lower()}"


def password_generation(length):
    chars = string.ascii_letters + string.digits
    list_chars = list(chars)
    password = np.random.choice(list_chars, length)
    return ''.join(password)


def generate_age_range(probabilities=None, list_age_range: List = person_entity.get("age_range")):
    return np.random.choice(list_age_range, size=1, replace=True, p=probabilities)[0]


def generate_localization(samples, fake: Faker):
    return list(map(lambda x: fake.locale(), range(samples)))


def generate_personal_data(num_users: int = 500, person_entity: Dict[str, Any] = None) -> pd.DataFrame:
    # Create Personal data frame
    df_personal_data = pd.DataFrame(
        data=[], columns=list(person_entity.keys()))
    # Generate gender and number of users
    df_personal_data["clinical_gender"] = np.random.choice(np.array(person_entity.get("clinical_gender")),
                                                           size=num_users,
                                                           replace=True,
                                                           p=[0.5, 0.5])
    # Initialize Faker
    fake = Faker()
    # Generate names and last names
    names = df_personal_data["clinical_gender"].apply(
        create_name_surname, fake=fake)
    names_list = list(zip(*names))
    df_personal_data["name"] = names_list[0]
    df_personal_data["surname"] = names_list[1]
    # Generate countries
    df_personal_data["country_of_origin"] = generate_country(
        num_users, fake=fake)
    df_personal_data["living_country"] = generate_country(num_users, fake=fake)
    df_personal_data["current_location"] = generate_country(
        num_users, fake=fake)
    df_personal_data["current_location"] = generate_localization(
        num_users, fake=fake)
    list_names = list(
        zip(*df_personal_data[["name", "surname"]].values.tolist()))
    # Generate email
    df_personal_data["email"] = list(
        map(lambda x, y: generate_email_from_name(x, y), list_names[0], list_names[1]))
    # Generate password
    df_personal_data["password"] = list(
        map(lambda x: password_generation(8), range(num_users)))
    # Generate user id
    df_personal_data["username"] = df_personal_data["name"].apply(
        lambda x: x.lower()+str(uuid.uuid4()).split("-")[-2])
    df_personal_data["userId"] = df_personal_data["name"].apply(
        lambda x: x.lower()+str(uuid.uuid4()).split("-")[-2])
    # Generate age range
    df_personal_data["age_range"] = list(
        map(lambda x: generate_age_range(), range(num_users)))
    return df_personal_data


def choose_one_from_list(list_values: List,
                         samples: int,
                         probabilities: List = None,
                         size: int = 1,
                         replace: bool = True):
    return list(map(lambda x: np.random.choice(list_values, size=size, replace=replace, p=probabilities), range(samples)))


# set the weight
def calculate_weight_from_height(height: float, bmi: string):
    bmi_numeric = 0.0
    if bmi == BMI_constants.underweight:
        bmi_numeric = 18.0
    elif bmi == BMI_constants.healthy:
        bmi_numeric = 21.0
    elif bmi == BMI_constants.overweight:
        bmi_numeric = 28.0
    else:
        bmi_numeric = 32.0
    return (height**2)*bmi_numeric


def generate_user_life_style_data(list_user_id: List[str],
                                  user_entity: Dict[str, Any],
                                  df_columns: List[str] = ["userId",
                                                           "current_working_status",
                                                           "marital_status",
                                                           "life_style",
                                                           "weight",
                                                           "ethnicity",
                                                           "height"]) -> pd.DataFrame:
    df_user_entity = pd.DataFrame(data=[], columns=df_columns)
    df_user_entity["userId"] = list_user_id
    num_users = len(list_user_id)
    df_user_entity["current_working_status"] = choose_one_from_list(
        user_entity.get("current_working_status"), samples=num_users)
    df_user_entity["marital_status"] = choose_one_from_list(
        user_entity.get("marital_status"), samples=num_users)
    df_user_entity["life_style"] = choose_one_from_list(
        user_entity.get("life_style"), samples=num_users)
    df_user_entity["ethnicity"] = choose_one_from_list(
        user_entity.get("ethnicity"), samples=num_users)
    # Generate BMI values
    BMI_values = ["underweight", "healthy", "overweight", "obesity"]
    BMI_prob = [0.1, 0.3, 0.3, 0.3]
    bmis = np.random.choice(BMI_values, size=500, replace=True, p=BMI_prob)
    df_user_entity["BMI"] = bmis
    # Generate height
    male_height = np.random.normal(170, 10, 500)
    female_height = np.random.normal(160, 10, 500)
    female_number = df_personal_data[df_personal_data["clinical_gender"] == 'F'].shape[0]
    male_number = num_users - female_number
    df_user_entity.loc[df_personal_data["clinical_gender"] == 'F',
                       "height"] = np.random.choice(female_height, size=female_number)
    df_user_entity.loc[df_personal_data["clinical_gender"] == 'M',
                       "height"] = np.random.choice(male_height, size=male_number)
    df_user_entity["height"] = df_user_entity["height"].astype(int)
    # Generate weight
    df_user_entity["weight"] = np.round(df_user_entity.apply(
        lambda row: calculate_weight_from_height(row["height"]/100.0, row["BMI"]), axis=1), 2)
    #
    df_user_entity["current_working_status"] = df_user_entity["current_working_status"].apply(
        lambda x: x[0])
    df_user_entity["marital_status"] = df_user_entity["marital_status"].apply(
        lambda x: x[0])
    df_user_entity["life_style"] = df_user_entity["life_style"].apply(
        lambda x: x[0])
    df_user_entity["ethnicity"] = df_user_entity["ethnicity"].apply(
        lambda x: x[0])
    return df_user_entity


def generate_health_condition_data(list_user_id: List[str]):
    df_health_conditions = pd.DataFrame(data=[], columns=["userId", "allergy"])
    df_health_conditions["userId"] = list_user_id
    # Allergy array and probabilities
    allergies = ["cow's milk", "eggs", "peanut", "soy",
                 "fish", "tree nuts", "shellfish", "wheat", "None"]
    allergies_prob = [0.075, 0.075, 0.075,
                      0.075, 0.075, 0.075, 0.075, 0.075, 0.4]
    user_allergies = np.random.choice(
        allergies, size=500, replace=True, p=allergies_prob)
    df_health_conditions["allergy"] = user_allergies
    return df_health_conditions


def define_user_goal_according_BMI(bmi: str):
    if bmi == BMI_constants.underweight:
        # goal gain muscle
        return f"{NutritionGoals.gain_weight}"
    elif bmi == BMI_constants.healthy:
        # Maintain fit and increase activity if required
        return f"{NutritionGoals.maintain_fit}"
    else:
        # nutritional goal loss weight
        return f"{NutritionGoals.lose_weight}"


def generate_user_goals(list_user_id: List[str]) -> pd.DataFrame:
    df_user_goals = pd.DataFrame(columns=["userId", "nutrition_goal"], data=[])
    df_user_goals["userId"] = list_user_id
    df_user_goals["nutrition_goal"] = df_user_entity["BMI"].apply(
        lambda x: define_user_goal_according_BMI(x))
    return df_user_goals


def generate_probabilities_for_flexi(food_restrictions=["vegan_observant",
                                                        "vegetarian_observant",
                                                        "halal_observant",
                                                        "kosher_observant",
                                                        "None"]):
    # generate different probabilities for the flexible
    flexi_probabilities = {
        "flexi_vegie": dict(zip(food_restrictions, [0.6, 0.2, 0.05, 0.05, 0.1])),
        # "flexi_vegetarian" : dict(zip(food_restrictions,[NN, 0.6, 0.05, 0.05, 0.1])),
        "flexi_vegetarian": dict(zip(food_restrictions, [0.0, 0.6, 0.05, 0.05, 0.3])),
        "flexi_halal": dict(zip(food_restrictions, [0.1, 0.1, 0.6, 0.1, 0.1])),
        "flexi_kosher": dict(zip(food_restrictions, [0.1, 0.1, 0.1, 0.6, 0.1]))
    }
    return flexi_probabilities

# assign probabilities


def assign_probabilities(cultural_factor):
    if cultural_factor == "flexi_observant":
        flexi_proba = generate_probabilities_for_flexi()
        value = np.random.choice(list(flexi_proba.keys()))
        return value
    pass


def generate_cultural_data(list_user_id: List[str]):
    df_cultural_factors = pd.DataFrame(
        data=[], columns=["userId", "cultural_factor"])
    df_cultural_factors["userId"] = list_user_id
    users_number = len(list_user_id)
    # Food restrictions probabilities
    food_restrictions = ["vegan_observant", "vegetarian_observant",
                         "halal_observant", "kosher_observant", "flexi_observant", "None"]
    food_restriction_probs = [0.2, 0.3, 0.05, 0.05, 0.1, 0.3]
    # generate cultural restrictions
    food_restrictions_user = np.random.choice(
        food_restrictions, size=users_number, replace=True, p=food_restriction_probs)
    df_cultural_factors["cultural_factor"] = food_restrictions_user
    df_cultural_factors["probabilities"] = None
    # Generate flexi probabilities
    dict_queries_cultural_factors = {
        "vegan_observant": "",
        "vegetarian_observant": "",
        "halal_observant": "",
        "kosher_observant": "",
        "flexi_observant": "None"
    }
    flexi_probabilities = generate_probabilities_for_flexi()
    df_cultural_factors["probabilities"] = df_cultural_factors["cultural_factor"].apply(
        lambda x: assign_probabilities(x))
    return df_cultural_factors


def generate_preferences_data(list_user_id: List[str]) -> pd.DataFrame:
    df_preferences = pd.DataFrame(
        data=[], columns=["userId", "breakfast_time", "lunch_time", "dinner_time"])
    df_preferences["userId"] = df_personal_data["userId"]
    users_number = len(list_user_id)
    # Normal time distribution
    breakfast_time = np.random.normal(7, 1, size=users_number)
    lunch_time = np.random.normal(13, 1, size=users_number)
    dinner_time = np.random.normal(20, 1, size=users_number)
    # generate probabilities
    df_preferences["breakfast_time"] = np.round(breakfast_time, 2)
    df_preferences["lunch_time"] = np.round(lunch_time, 2)
    df_preferences["dinner_time"] = np.round(dinner_time, 2)
    return df_preferences


def calculate_basal_metabolic_rate(weight: float, height: float, age: int, clinical_gender: str):
    BMR = 0
    if Gender.male == clinical_gender:
        BMR = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        BMR = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
    return BMR
