import pandas as pd
import numpy as np
import datetime as dt
import time
from faker import Faker
from enum import Enum
from typing import List, Any, Tuple, Dict, Union, Set
from functools import partial
import string
import os
import json
import uuid
from parfor import parfor
import seaborn as sns
import matplotlib.pyplot as plt
from html import escape
from scipy.stats import bernoulli
import traceback
from string import Formatter
from .default_inputs import (person_entity,
                             user_entity,
                             meals_calorie_dict,
                             height_distribution,
                             meal_time_distribution,
                             allergies_queries_dict,
                             cultural_query_text,
                             columns_tracking,
                             meals_queries_dict,
                             meals_proba_dict)

from .html_utilities import HTML_Table
from .json_utilities import JsonSerializationHelper

class Gender(str, Enum):
    """Enumeration to maintain clinical genders M for male and F for female.

    :param str: str class to save string constants representing the clinical gender.
    :type str: str
    :param Enum: Enumeration interface
    :type Enum: Enum
    """
    male = "M"
    female = "F"


class BMI_constants(str, Enum):
    """Enumeration to maintain the Body Index Mass (BMI) conditions:
    * underweight
    * healthy
    * overweight
    * obesity.

    :param str: str class to save string constants representing the BMI conditions.
    :type str: str
    :param Enum: Enumeration interface
    :type Enum: Enum
    """
    underweight = "underweight"
    healthy = "healthy"
    overweight = "overweight"
    obesity = "obesity"


class NutritionGoals(str, Enum):
    """Enumeration to maintain the nutrition goals:
    * lose_weight
    * maintain_fit
    * gain_weight.

    :param str: str class to save string constants representing the nutrition goals.
    :type str: str
    :param Enum: Enumeration interface
    :type Enum: Enum
    """
    lose_weight = "lose_weight"
    maintain_fit = "maintain_fit"
    gain_weight = "gain_weight"


class ActivityLevel(str, Enum):
    """Enumeration to maintain the activity levels:
    * Sedentary
    * Lightly active
    * Moderately active
    * Very active.

    :param str: str class to save string constants representing the activity level.
    :type str: str
    :param Enum: Enumeration interface
    :type Enum: Enum
    """
    sedentary = "Sedentary"
    light_active = "Lightly active"
    moderate_active = "Moderately active"
    very_active = "Very active"

# Functions


def create_name_surname(gender: Gender,
                        fake: Faker) -> List[str]:
    """Returns a list of string with a simulated name and surname based on clinical gender parameter.

    :param gender: Clinical gender M for male F for female.
    :type gender: Gender
    :param fake: Faker object
    :type fake: Faker
    :return: List of string in first position the name and in second position the surname.
    :rtype: List[str]
    """
    if gender == Gender.male:
        names = fake.name_male()
    else:
        names = fake.name_female()
    return names.split(" ")


def generate_country(samples: int,
                     fake: Faker) -> List[str]:
    """Generates a country name.

    :param samples: Number of country samples to generate.
    :type samples: int
    :param fake: Faker object
    :type fake: Faker
    :return: List of country names with len == samples
    :rtype: List[str]
    """
    return list(map(lambda x: fake.country(), range(samples)))


def generate_email_from_name(name: str,
                             surname: str,
                             domain: str = "fake.com") -> str:
    """Returns an email address from a given name and surname.

    :param name: user's name
    :type name: str
    :param surname: user's surname
    :type surname: str
    :param domain: email domain, defaults to "fake.com"
    :type domain: str, optional
    :return: fake email address with the form name.surname@domain
    :rtype: str
    """
    return f"{name.lower()}.{surname.lower()}@{domain.lower()}"


def password_generation(length: int) -> str:
    """Generates a password from a given length.

    :param length: Password's length
    :type length: int
    :return: Generated password with len length
    :rtype: str
    """
    chars = string.ascii_letters + string.digits
    list_chars = list(chars)
    password = np.random.choice(list_chars, length)
    return ''.join(password)


def generate_age_range(probabilities: List[float] = None,
                       list_age_range: List[str] = person_entity.get("age_range")) -> str:
    """Randomly chose an age range from a given age range's list and probabilities' list.

    :param probabilities: List of probabilities to chose one age range, probabilities should sum up 1, defaults to None
    :type probabilities: List[float], optional
    :param list_age_range: List of age ranges, defaults to person_entity.get("age_range")
    :type list_age_range: List[str], optional
    :return: age range randomly chosen based on probabilities.
    :rtype: str
    """
    return np.random.choice(list_age_range, size=1, replace=True, p=probabilities)[0]


def generate_localization(samples: int, fake: Faker) -> List[str]:
    """Returns a list of localizations.

    :param samples: number of samples to generate
    :type samples: int
    :param fake: Faker object
    :type fake: Faker
    :return: List of localizations with len samples
    :rtype: List[str]
    """
    return list(map(lambda x: fake.locale(), range(samples)))


def generate_personal_data(gender_probabilities: Dict[str, Any],
                           age_probabilities: Dict[str, Any],
                           num_users: int = 500,
                           person_entity: Dict[str, Any] = None) -> pd.DataFrame:
    """Generates a pandas Dataframe with personal user data.

    :param gender_probabilities: Dictionary that contains probability for clinical genders male M and female F
    :type gender_probabilities: Dict[str, Any]
    :param num_users: number of users to generate, defaults to 500
    :type num_users: int, optional
    :param person_entity: person entity data with the required fields, defaults to None
    :type person_entity: Dict[str, Any], optional
    :return: Pandas dataframe with users' personal data
    :rtype: pd.DataFrame
    """
    # Create Personal data frame
    df_personal_data = pd.DataFrame(
        data=[], columns=list(person_entity.keys()))
    # Generate gender and number of users
    gender_list = []
    gender_proba = []
    for k, v in gender_probabilities.items():
        gender_list.append(k)
        gender_proba.append(v)
    df_personal_data["clinical_gender"] = np.random.choice(gender_list,
                                                           size=num_users,
                                                           replace=True,
                                                           p=gender_proba)
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
    # Generate gender and number of users
    age_list = []
    age_proba = []
    for k, v in age_probabilities.items():
        age_list.append(k)
        age_proba.append(v)
    df_personal_data["age_range"] = np.random.choice(age_list,
                                                     size=num_users,
                                                     replace=True,
                                                     p=age_proba)
    return df_personal_data


def choose_one_from_list(list_values: List[Any],
                         samples: int,
                         probabilities: List[float] = None,
                         size: int = 1,
                         replace: bool = True):
    """Choose one item from a list given a probability

    :param list_values: List of item to be selected
    :type list_values: List[Any]
    :param samples: number of samples to generate
    :type samples: int
    :param probabilities: probability array should sum up 1, defaults to None
    :type probabilities: List[float], optional
    :param size: number of element to select at once, defaults to 1
    :type size: int, optional
    :param replace: True for sampling with replacement or not, defaults to True
    :type replace: bool, optional
    :return: List of choices
    :rtype: _type_
    """
    return list(map(lambda x: np.random.choice(list_values,
                                               size=size,
                                               replace=replace,
                                               p=probabilities),
                    range(samples)))


# set the weight
def calculate_weight_from_height(height: float, bmi: BMI_constants) -> float:
    """Returns the user's weight based on their height and Body Mass Index (BMI) value.

    :param height: user's height in meters
    :type height: float
    :param bmi: user's BMI condition
    :type bmi: BMI_constants
    :return: user's weight in kg
    :rtype: float
    """
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
                                  BMI_probabilities_dict: Dict[str, Any],
                                  df_personal_data: pd.DataFrame,
                                  height_distribution: Dict[str,
                                                            Any] = height_distribution,
                                  df_columns: List[str] = ["userId",
                                                           "current_working_status",
                                                           "marital_status",
                                                           "life_style",
                                                           "weight",
                                                           "ethnicity",
                                                           "height"]) -> pd.DataFrame:
    """Generate user life style data including marital status, current working status, ethnicity and exercise frequency.

    :param list_user_id: List of users' IDs
    :type list_user_id: List[str]
    :param user_entity: Probability dictionary for ethnicity values.
    :type user_entity: Dict[str, Any]
    :param BMI_probabilities_dict: Probability dictionary for BMI conditions
    :type BMI_probabilities_dict: Dict[str, Any]
    :param df_personal_data: Pandas Dataframe with users' personal data (e.g, name, surname, etc)
    :type df_personal_data: pd.DataFrame
    :param height_distribution: Probability dictionary for generate height distribution, defaults to height_distribution
    :type height_distribution: Dict[str, Any], optional
    :param df_columns: List of columns names for the new Dataframe, defaults to ["userId", "current_working_status", "marital_status", "life_style", "weight", "ethnicity", "height"]
    :type df_columns: List[str], optional
    :return: users' life style Dataframe
    :rtype: pd.DataFrame
    """
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
    BMI_values = []
    BMI_prob = []
    for k, v in BMI_probabilities_dict.items():
        BMI_values.append(k)
        BMI_prob.append(v)
    bmis = np.random.choice(BMI_values, size=num_users,
                            replace=True, p=BMI_prob)
    df_user_entity["BMI"] = bmis
    # Generate height
    male_height = np.random.normal(height_distribution["male"]["mean"],
                                   height_distribution["male"]["std"],
                                   num_users)
    female_height = np.random.normal(height_distribution["female"]["mean"],
                                     height_distribution["female"]["std"],
                                     num_users)
    female_number = df_personal_data[df_personal_data["clinical_gender"]
                                     == Gender.female.value].shape[0]
    male_number = num_users - female_number
    df_user_entity.loc[df_personal_data["clinical_gender"] == Gender.female.value,
                       "height"] = np.random.choice(female_height, size=female_number)
    df_user_entity.loc[df_personal_data["clinical_gender"] == Gender.male.value,
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


def generate_health_condition_data(list_user_id: List[str],
                                   allergies_probability_dict: Dict[str, Any],
                                   multiple_allergies_number: int = 2):
    """Generate users health conditions (allergies) based on probability dictionary.

    :param list_user_id: users' IDs
    :type list_user_id: List[str]
    :param allergies_probability_dict: Probability dictionary where keys are allergy conditions and values their probabilities, total probabilities should sum up 1
    :type allergies_probability_dict: Dict[str, Any]
    :return: Pandas Dataframe with users IDs and their assigned health condition.
    :rtype: _type_
    """
    df_health_conditions = pd.DataFrame(
        data=[], columns=["userId", "allergy", "Multi-allergy"])
    df_health_conditions["userId"] = list_user_id
    num_users = len(list_user_id)
    # Allergy array and probabilities
    allergies = []
    allergies_prob = []
    for k, v in allergies_probability_dict.items():
        allergies.append(k)
        allergies_prob.append(v)
    user_allergies = np.random.choice(
        allergies, size=num_users, replace=True, p=allergies_prob)
    df_health_conditions["allergy"] = user_allergies
    # generate multiple allergies users
    allergies_list = list(allergies_queries_dict.keys())
    # sampling
    multi_allergies = np.random.choice(a=allergies_list,
                                       size=multiple_allergies_number
                                       )
    df_health_conditions["Multi-allergy"] = df_health_conditions["allergy"].apply(
        lambda x: " ".join(np.random.choice(a=allergies_list,
                                            size=multiple_allergies_number,
                                            replace=False
                                            ).tolist()) if x == "Multiple" else "N/A"
    )
    return df_health_conditions


def define_user_goal_according_BMI(bmi: BMI_constants) -> NutritionGoals:
    """Assigns to the user a nutrition goal (e.g., lose weight, maintain weight, gain weight) based on their current BMI.

    :param bmi: Current BMI condition
    :type bmi: BMI_constants
    :return: A proposed nutrition goal to improve their health condition.
    :rtype: NutritionGoals
    """
    if bmi == BMI_constants.underweight:
        # goal gain muscle
        return f"{NutritionGoals.gain_weight}"
    elif bmi == BMI_constants.healthy:
        # Maintain fit and increase activity if required
        return f"{NutritionGoals.maintain_fit}"
    else:
        # nutritional goal loss weight
        return f"{NutritionGoals.lose_weight}"


def generate_user_goals(list_user_id: List[str], df_user_entity: pd.DataFrame) -> pd.DataFrame:
    """Generate a Dataframe with users' goals.

    :param list_user_id: users' IDs
    :type list_user_id: List[str]
    :param df_user_entity: Dataframe with users' BMI
    :type df_user_entity: pd.DataFrame
    :return: Dataframe with users' IDs and users' nutrition goals
    :rtype: pd.DataFrame
    """
    df_user_goals = pd.DataFrame(columns=["userId", "nutrition_goal"], data=[])
    df_user_goals["userId"] = list_user_id
    df_user_goals["nutrition_goal"] = df_user_entity["BMI"].apply(
        lambda x: define_user_goal_according_BMI(x))
    return df_user_goals


def assign_probabilities(cultural_factor: str,
                         flexi_probability_dict: Dict[str, Any]) -> Any:
    """Choose a cultural factor for divergent flexible (flexi) user.

    :param cultural_factor: user's cultural factor (e.g., None, Kosher observant, Flexi_vegan)
    :type cultural_factor: str
    :param flexi_probability_dict: flexible probability dictionary
    :type flexi_probability_dict: Dict[str, Any]
    :return: A chosen cultural factor based on probability dictionary
    :rtype: Any
    """
    if cultural_factor == "flexi_observant":
        flexi_proba = flexi_probability_dict
        value = np.random.choice(list(flexi_proba.keys()))
        return value
    pass


def generate_cultural_data(list_user_id: List[str],
                           food_restriction_probability_dict: Dict[str, Any],
                           flexi_probability_dict: Dict[str, Any]) -> pd.DataFrame:
    """Returns a Pandas Dataframe with users' IDs and assigned cultural factors.

    :param list_user_id: users' IDs list
    :type list_user_id: List[str]
    :param food_restriction_probability_dict: probability dictionary with cultural food restrictions
    :type food_restriction_probability_dict: Dict[str, Any]
    :param flexi_probability_dict: probability dictionary with flexible probabilities
    :type flexi_probability_dict: Dict[str, Any]
    :return: Pandas Dataframe with the chosen cultural food restrictions
    :rtype: pd.DataFrame
    """
    df_cultural_factors = pd.DataFrame(
        data=[], columns=["userId", "cultural_factor"])
    df_cultural_factors["userId"] = list_user_id
    users_number = len(list_user_id)
    # Food restrictions probabilities
    food_restrictions = []
    food_restriction_probs = []
    for k, v in food_restriction_probability_dict.items():
        food_restrictions.append(k)
        food_restriction_probs.append(v)
    # generate cultural restrictions
    food_restrictions_user = np.random.choice(
        food_restrictions, size=users_number, replace=True, p=food_restriction_probs)
    df_cultural_factors["cultural_factor"] = food_restrictions_user
    df_cultural_factors["probabilities"] = None
    # Generate flexi probabilities
    df_cultural_factors["probabilities"] = df_cultural_factors["cultural_factor"].apply(
        lambda x, flexi_probability_dict: assign_probabilities(x, flexi_probability_dict), flexi_probability_dict=flexi_probability_dict
    )
    return df_cultural_factors


def generate_preferences_data(list_user_id: List[str],
                              df_personal_data: pd.DataFrame,
                              meals_time_distribution: Dict[str, Any]
                              ) -> pd.DataFrame:
    """Returns a Dataframe with user's meals consumption time.

    :param list_user_id: users' IDs list
    :type list_user_id: List[str]
    :param df_personal_data: Pandas Dataframe with users data
    :type df_personal_data: pd.DataFrame
    :param meals_time_distribution: probability dictionary with mean and std for meals time distribution
    :type meals_time_distribution: Dict[str, Any]
    :return: Pandas Dataframe with users' preferences meal time
    :rtype: pd.DataFrame
    """
    df_preferences = pd.DataFrame(
        data=[], columns=["userId", "breakfast_time", "lunch_time", "dinner_time"])
    df_preferences["userId"] = df_personal_data["userId"]
    users_number = len(list_user_id)
    # Normal time distribution
    breakfast_time = np.random.normal(meals_time_distribution["breakfast"]["mean"],
                                      meals_time_distribution["breakfast"]["std"],
                                      size=users_number)
    lunch_time = np.random.normal(meals_time_distribution["lunch"]["mean"],
                                  meals_time_distribution["lunch"]["std"],
                                  size=users_number)
    dinner_time = np.random.normal(meals_time_distribution["dinner"]["mean"],
                                   meals_time_distribution["dinner"]["std"],
                                   size=users_number)
    # generate probabilities
    df_preferences["breakfast_time"] = np.round(breakfast_time, 2)
    df_preferences["lunch_time"] = np.round(lunch_time, 2)
    df_preferences["dinner_time"] = np.round(dinner_time, 2)
    return df_preferences


def calculate_basal_metabolic_rate(weight: float, height: float, age: int, clinical_gender: str) -> float:
    """Calculate basal metabolic rate (BMR) based on user's weight, height, age and clinical gender.

    :param weight: user's weight
    :type weight: float
    :param height: user's height
    :type height: float
    :param age: user's age
    :type age: int
    :param clinical_gender: user's clinical gender
    :type clinical_gender: str
    :return: user's BMR
    :rtype: float
    """
    BMR = 0
    if Gender.male == clinical_gender:
        # Numbers here are part of the Basal metabolic rate (BMR) formula.
        BMR = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        BMR = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
    return BMR


def calculate_daily_calorie_needs(BMR: float, activity_level: ActivityLevel) -> float:
    """Calculate the daily calorie needs given the BMR and the activity level.

    :param BMR: Basal Metabolic Rate (BMR)
    :type BMR: float
    :param activity_level: user's activity level (e.g., sedentary, light active, moderate active)
    :type activity_level: ActivityLevel
    :return: Daily calorie needs
    :rtype: float
    """
    calories_daily = 1200
    if activity_level == ActivityLevel.sedentary:
        calories_daily = 1.2 * BMR
    elif activity_level == ActivityLevel.light_active:
        calories_daily = 1.375 * BMR
    elif activity_level == ActivityLevel.moderate_active:
        calories_daily = 1.725 * BMR
    else:
        calories_daily = 1.9 * BMR
    return np.max([calories_daily, 1200])


def define_daily_calorie_plan(nutrition_goal: NutritionGoals, daily_calorie_need: float) -> float:
    """Calculate the user's projected calorie needs to reach the nutrition goal according daily calorie needs.

    :param nutrition_goal:user's nutrition goals (e.g., loss weight, maintain, gain weight)
    :type nutrition_goal: NutritionGoals
    :param daily_calorie_need: user's daily calorie needs
    :type daily_calorie_need: float
    :return: projected daily user's calories needs to reach the nutrition goal
    :rtype: float
    """
    projected_calories_need = 1200
    if nutrition_goal == NutritionGoals.gain_weight:
        # Add or remove calories to create metabolic deficit
        projected_calories_need = daily_calorie_need + 500
    elif nutrition_goal == NutritionGoals.maintain_fit:
        projected_calories_need = daily_calorie_need
    else:
        projected_calories_need = daily_calorie_need - 500
    return np.max(np.array([projected_calories_need, 1200]))


def generate_diet_plan(weight: float,
                       height: float,
                       age_range: str,
                       clinical_gender: Gender,
                       activity_level: ActivityLevel,
                       nutrition_goal: NutritionGoals) -> float:
    """Generate a full diet plan from an user given their current age range, weight, height, clinical gender, activity level and nutrition goal.

    :param weight: user's weight
    :type weight: float
    :param height: user's height
    :type height: float
    :param age_range: user's age range
    :type age_range: str
    :param clinical_gender: user's clinical gender
    :type clinical_gender: Gender
    :param activity_level: user's activity level
    :type activity_level: ActivityLevel
    :param nutrition_goal: user's nutrition goal
    :type nutrition_goal: NutritionGoals
    :return: user's projected daily calorie needs
    :rtype: float
    """
    # transform age
    age_list = age_range.split("-")
    age = np.ceil((int(age_list[-1]) - int(age_list[0]))/2 + int(age_list[0]))
    bmr = calculate_basal_metabolic_rate(weight, height, age, clinical_gender)
    calorie_needs = calculate_daily_calorie_needs(bmr, activity_level)
    projected_calorie_needs = define_daily_calorie_plan(
        nutrition_goal, calorie_needs)
    return projected_calorie_needs, calorie_needs


def generate_therapy_data(list_user_id: List[str],
                          df_personal_data: pd.DataFrame,
                          df_user_goals: pd.DataFrame,
                          df_user_entity) -> pd.DataFrame:
    """Returns a Dataframe with planned therapy for the user given.

    :param list_user_id: User's IDs list.
    :type list_user_id: List[str]
    :param df_personal_data: Dataframe with users' features.
    :type df_personal_data: pd.DataFrame
    :param df_user_goals: Dataframe with users' nutritional goals.
    :type df_user_goals: pd.DataFrame
    :param df_user_entity: Dataframe with additional users' features.
    :type df_user_entity: pd.DataFrame
    :return: Dataframe with the therapies for each user.
    :rtype: pd.DataFrame
    """
    # generate treatment for the users
    # prepare data
    df_user_data = df_user_goals.merge(df_personal_data[["userId", "clinical_gender", "age_range"]],
                                       on="userId")
    df_user_data = df_user_data.merge(df_user_entity[["userId", "life_style", "weight", "height"]],
                                      on="userId")
    # temp df

    df_total = pd.concat(
        (df_user_data['userId'],
         pd.DataFrame(np.ceil(list(df_user_data.apply(lambda row: generate_diet_plan(weight=row["weight"],
                                                                                     height=row["height"],
                                                                                     age_range=row["age_range"],
                                                                                     clinical_gender=row["clinical_gender"],
                                                                                     activity_level=row["life_style"],
                                                                                     nutrition_goal=row["nutrition_goal"]), axis=1))),
                      columns=["projected_daily_calories", "current_daily_calories"])),
        axis=1
    )
    return df_total, df_user_data


def generate_meals_plan_per_user(users: List[str], probability_dict: Dict[str, float]) -> Dict[str, Any]:
    """Generate meals plan per user with meals to consume or not based on meals probabilities.

    :param users: users' IDs list.
    :type users: List[str]
    :param probability_dict: Meals' probability dict.
    :type probability_dict: Dict[str, float]
    :return: Dictionary with meals presence or not per user.
    :rtype: Dict[str, Any]
    """
    total_users = len(users)
    meal_presence = {}
    for key, proba in probability_dict.items():
        meal_presence[key] = bernoulli.rvs(proba, size=total_users)
    return meal_presence


def generate_delta_values(chose_dist: str, parameters: Dict[str, Any], size=1):
    # print(f"received parameter: {parameters}")
    if chose_dist == 'Normal':
        result = np.random.normal(
            loc=parameters['mean'], scale=parameters['std'], size=size)
        result[result < 0.0] = 0.0
        result[result > 1] = 1.0
        return result
    else:
        # choose distribution list
        dist_list = np.random.choice([1, 2],
                                     p=[0.4, 0.6],
                                     size=size,
                                     replace=True)
        output_list = []
        for i in dist_list:
            output_list.append(np.random.normal(
                loc=parameters[f"mean_{i}"], scale=parameters[f"std_{i}"]))
        result = np.array(output_list)
        result[result < 0.0] = 0.0
        result[result > 1] = 1.0
        return result


def generate_allergy_oriented_food_dataset(food_db: pd.DataFrame,
                                           allergies_queries: Dict[str,
                                                                   List[str]] =
                                           allergies_queries_dict
                                           ):
    allergy_food_ids = {}
    for allergy in allergies_queries.keys():
        allergy_food_ids[allergy] = []
        for w in allergies_queries[allergy]:
            # print(w)
            # choose ids that contain the world
            recipes_id = food_db.loc[food_db.allergies.str.contains(
                w, case=False, na=False), 'recipeId'].tolist()
            allergy_food_ids[allergy].extend(recipes_id)
        # eliminate duplicates
        allergy_food_ids[allergy] = list(set(allergy_food_ids[allergy]))
    return allergy_food_ids


def generate_cultural_factor_oriented_dataset(food_db: pd.DataFrame,
                                              cultural_factor_query:
                                                  Dict[str, Any] = cultural_query_text,
                                              ):
    cultural_food_ids = {}
    for cultural_factor in cultural_factor_query.keys():
        filtered_food_db = food_db.query(
            cultural_factor_query[cultural_factor])
        cultural_food_ids[cultural_factor] = list(
            set(filtered_food_db['recipeId'].tolist())
        )
    return cultural_food_ids


def generate_meal_type_oriented_dataset(food_db: pd.DataFrame,
                                        meal_type_query: Dict[str, Any] = meals_queries_dict):
    # choose meals types
    meal_ids = {}
    for meal_tp in meal_type_query.keys():
        meal_ids[meal_tp] = food_db.loc[food_db["meal_type"]
                                        == meal_tp, "recipeId"].tolist()
    return meal_ids


def generate_next_BMI_based_on_transition_matrix(bmi_conditions: List[BMI_constants],
                                                 df_user: pd.DataFrame,
                                                 transition_matrix: np.array = None):
    # Generate users probabilities to success or fail the process
    df_user_copy = df_user
    df_user_copy["next_BMI"] = ""
    if transition_matrix is not None:
        bmi_list_str = [bmi.value for bmi in bmi_conditions]
        for idx, bmi in enumerate(bmi_conditions):
            # choose users from that condition
            users_bmi_condition = df_user.query(f"BMI == '{bmi.value}'")
            if len(users_bmi_condition) > 0:
                # choose probabilities to this user
                next_states = np.random.choice(bmi_list_str,
                                               size=len(users_bmi_condition),
                                               p=transition_matrix[idx, :]
                                               )
                # set the next value
                df_user_copy.loc[users_bmi_condition.index,
                                 "next_BMI"] = next_states
    else:
        df_user_copy["next_BMI"] = df_user_copy["BMI"]
    return df_user_copy


def generate_daily_calories_requirement_according_next_BMI(
        current_daily_calories: float,
        bmi_conditions: List[BMI_constants],
        current_state: str,
        next_state: str,
        days_to_simulated: int) -> Union[List, np.array]:
    # choose daily calories
    daily_calories_list = []
    # print(f"current state: {current_state} and next state: {next_state}")
    # print(
    #    f"current state type: {type(current_state)} and next state type: {type(next_state)}")
    if current_state == next_state:
        # maintain state
        daily_calories_list = np.random.normal(loc=current_daily_calories,
                                               scale=20,
                                               size=days_to_simulated)
    # gain weight
    elif bmi_conditions.index(current_state) < bmi_conditions.index(next_state):
        daily_calories_list = np.full(days_to_simulated, current_daily_calories) + np.random.normal(
            loc=500,
            scale=100,
            size=days_to_simulated
        )
        # print(f"daily calories list: {daily_calories_list}")
        # print(f"daily calories list len: {len(daily_calories_list)}")
    # lose weight
    else:
        daily_calories_list = np.full(days_to_simulated, current_daily_calories) - np.random.normal(
            loc=500,
            scale=100,
            size=days_to_simulated
        )
    return daily_calories_list


def distribute_calories_in_meal(meals_plan: Dict[str, float],
                                meals_calorie_distribution: Dict[str, float]
                                ) -> Dict[str, float]:
    # Generate calorie distribution per meal
    meals_calorie_dict = {}
    for key in meals_calorie_distribution.keys():
        if meals_plan[key] == 1:
            meals_calorie_dict[key] = meals_calorie_distribution[key]
    current_percentages = sum(meals_calorie_dict.values())
    difference = 1.0 - current_percentages
    # print(f"difference: {difference}")
    if difference > 0:
        # redistribute percentages
        meals_count = len(meals_calorie_dict.keys())
        difference_meal = difference/meals_count
        for k in meals_calorie_dict.keys():
            meals_calorie_dict[k] += difference_meal
    elif difference < 0:
        # redistribute percentages reducing them
        meals_count = len(meals_calorie_dict.keys())
        difference_meal = difference/meals_count
        for k in meals_calorie_dict.keys():
            meals_calorie_dict[k] -= difference_meal
    else:
        pass
    return meals_calorie_dict


def choose_meal(all_food_ids_set: Set,
                allergies: Union[str, List[str]],
                allergy_dataset: Dict[str, List[str]],
                ):
    all_food_ids = set()
    allergy_set = set()
    if isinstance(allergies, list):
        # process allergies one by one
        for allergy in allergies:
            # print(f"Multiple fail")
            allergy_set = allergy_set.union(
                set(
                    allergy_dataset.get(allergy, [])
                )
            )
    else:
        # print(f"I arrive here")
        allergy_set = allergy_set.union(
            set(
                allergy_dataset.get(allergies, [])
            )
        )
    # remove allergies
    all_food_ids = all_food_ids_set - allergy_set
    return all_food_ids


def generate_food_day(selected_food_df: pd.DataFrame,
                      calories: float,
                      appreciation_feedback: float,
                      cultural_factor: str, 
                      allergies_food_ids: Dict[str, List[str]] = None,
                      allergy_factors: List[str] = [],
                      threshold: float = 0.5):
    # positive feedback
    if appreciation_feedback >= threshold:   
        # Filter by allergy factor
        if allergies_food_ids is not None:
            candidate_set = choose_meal(all_food_ids_set=set(selected_food_df["recipeId"].tolist()),
                                allergies=allergy_factors,
                                allergy_dataset=allergies_food_ids
                                )
            candidates_ids = candidate_set
            selected_food_df_safe = selected_food_df.query("recipeId in @candidates_ids").copy()
        else:
            selected_food_df_safe = selected_food_df.copy()
        if len(selected_food_df_safe) == 0:
            selected_food_df_safe = selected_food_df
        # filter by cultural factor
        cultural_mask = selected_food_df_safe["cultural_restriction"] == cultural_factor
        selected_food_df_safe.loc[cultural_mask, "calorie_distance"] =\
            selected_food_df_safe.loc[cultural_mask, "calories"].apply(
                lambda x: np.abs(calories - x)
        )
        # sort by nearest
        sorted_candidates = selected_food_df_safe.sort_values(
            by="calorie_distance",
            ascending=True
        )
        # choose top
        top_candidates = sorted_candidates.iloc[:20, :]
        # print(f"Top candidates len: {len(top_candidates)}")

        choose_recipe = top_candidates["recipeId"].sample(n=1).tolist()[0]
    else: 
        # Negative feedback
        top_candidates = selected_food_df.sample(frac=1.)
        choose_recipe = top_candidates["recipeId"].sample(n=1).tolist()[0]
    return choose_recipe


def generate_user_simulation(
        user_id: str,
        df_user: pd.DataFrame,
        food_db: pd.DataFrame,
        meals_probability_dict: Dict[str, Any],
        meals_calorie_distribution: Dict[str, Any],
        allergies_food_ids: Dict[str, Any],
        cultural_food_ids: Dict[str, Any],
        meal_type_food_ids: Dict[str, Any],
        place_probabilities: Dict[str, Any],
        social_situation_probabilities: Dict[str, Any],
        chose_dist: str,
        meals_time_dict: Dict[str, Any],
        delta_dist_params: Dict[str, Any],
        dict_flexi_probas: Dict[str, Any],
        days_to_simulated: int,
        bmi_conditions: List[BMI_constants]):
    # get basic dataset from the user
    user_db = df_user.loc[df_user.userId == user_id, :]
    # print(f"user db: {user_db}")
    # check that user is found
    if user_db.empty:
        raise Exception(f"User: {user_id} not found in users database")
    # calculate daily calories requirement per day
    daily_calories_per_day_list = \
        generate_daily_calories_requirement_according_next_BMI(
            current_daily_calories=user_db["current_daily_calories"],
            bmi_conditions=bmi_conditions,
            current_state=user_db['BMI'].tolist()[0],
            next_state=user_db["next_BMI"].tolist()[0],
            days_to_simulated=days_to_simulated
        )
    # Generate flexi probabilities
    # distribute daily calories need into meals plan
    meals_plan = generate_meals_plan_per_user(
        users=[user_db.userId],
        probability_dict=meals_probability_dict
    )
    meals_calorie_dict = distribute_calories_in_meal(
        meals_plan=meals_plan,
        meals_calorie_distribution=meals_calorie_distribution
    )
    # distribute the calories in the daily calorie need
    temp_df = pd.DataFrame(columns=columns_tracking)
    # iterate over the meals plan and cross meal type, allergies, cultural and filter by calories
    allergy_list = []
    user_db_series = user_db.squeeze()
    if user_db_series.get(key='allergy') == "Multiple":
        allergy_list = user_db_series.get(key="Multi-allergy").split(" ")
    else:
        allergy_list = user_db_series.get(key='allergy')
    daily_food_cultural = None
    if user_db.cultural_factor.tolist()[0] == "flexi_observant":
        probas = user_db.probabilities.tolist()[0]
        if probas is not None:
            chosen_probas = dict_flexi_probas.get(probas, None)
            if chosen_probas is not None:
                daily_food_cultural = np.random.choice(
                    a=list(chosen_probas.keys()),
                    p=list(chosen_probas.values()),
                    replace=True,
                    size=days_to_simulated
                )
    else:
        daily_food_cultural = np.full(days_to_simulated,
                                      user_db.cultural_factor.tolist()[0])
    meals_dfs = []
    for meal_tp in meals_calorie_dict.keys():
        temp_meal_df = pd.DataFrame(columns=columns_tracking)
        # Generate appreciation feedback
        appreciation_feedback = generate_delta_values(chose_dist=chose_dist,
                                                      parameters=delta_dist_params,
                                                      size=days_to_simulated)
        #print(f"generated feedback shape: {appreciation_feedback.shape}")
        temp_meal_df["appreciation_feedback"] = appreciation_feedback
        # raise Exception(
        # "No candidates found to this user, meal type and restriction")
        # calculate calories distance
        # daily calorie
        calories_available_per_day =\
            daily_calories_per_day_list *\
            meals_calorie_dict.get(meal_tp, 0)
        # print(f"Len available calories per dar: {len(calories_available_per_day)}")
        # print(f"selected food candidates: {selected_food_df.columns}")
        calories_df = pd.DataFrame(columns=["days", "calories"])
        calories_df["days"] = np.arange(0, days_to_simulated)
        calories_df["calories"] = calories_available_per_day
        calories_df["cultural_factor"] = daily_food_cultural
        calories_df['appreciation_feedback'] = appreciation_feedback
        partial_daily_calorie_recipe = partial(generate_food_day,
                                               selected_food_df=food_db,
                                               allergy_factors=allergy_list,
                                               allergies_food_ids=allergies_food_ids
                                               )
        # print(f"calorie df: {calories_df.columns}")
        # print(f"Calories df size: {calories_df.shape}")
        calories_df["recipeId"] = calories_df.apply(lambda row:
                                                    partial_daily_calorie_recipe(
                                                        calories=row["calories"],
                                                        cultural_factor=row["cultural_factor"],
                                                        appreciation_feedback=row['appreciation_feedback']),
                                                    axis=1)
        # temp_df = temp_df.append(food_dfs, ignore_index=True)
        # fill  the data frame
        temp_meal_df["day_number"] = np.arange(0, days_to_simulated)
        temp_meal_df["meal_type"] = meal_tp
        temp_meal_df["userId"] = user_id
        temp_meal_df["foodId"] = calories_df["recipeId"]
        temp_meal_df["time_of_meal_consumption"] = np.random.normal(
            loc=meals_time_dict[meal_tp]['mean'],
            scale=meals_time_dict[meal_tp]['std'],
            size=len(temp_meal_df)
        )
        temp_meal_df["place_of_meal_consumption"] = np.random.choice(
            a=list(place_probabilities.keys()),
            p=list(place_probabilities.values()),
            size=len(temp_meal_df)
        )
        temp_meal_df["social_situation_of_meal_consumption"] = np.random.choice(
            a=list(social_situation_probabilities.keys()),
            p=list(social_situation_probabilities.values()),
            size=len(temp_meal_df)
        )
        meals_dfs.append(temp_meal_df)

    temp_df = pd.concat(meals_dfs, ignore_index=True).reset_index(drop=True)
    return temp_df


def generate_recommendations(df_user: pd.DataFrame,
                             transition_matrix: np.array,
                             df_recipes_db: pd.DataFrame,
                             place_probabilities: Dict[str, Any],
                             social_situation_probabilities: Dict[str, Any],
                             meals_plan: Any,
                             chose_dist: str,
                             delta_dist_params: Dict[str, Any],
                             flexi_probabilities_dict: dict[str, Any],
                             meals_calorie_dict: Dict[str,
                                                      float] = meals_calorie_dict,
                             meals_time_dict: Dict[str,
                                                   Dict] = meal_time_distribution,
                             days_to_simulated: int = 365,
                             progress_bar: Any = None) -> Dict[str, pd.DataFrame]:
    """Generate the simulated tracking data  for users during a given time. 

    :param df_user: Users' data Dataframe.
    :type df_user: pd.DataFrame
    :param transition_matrix: probability transition matrix between user states. 
    :type transition_matrix: np.array
    :param df_recipes_db: Recipe's Dataframe contains recipes data and features.
    :type df_recipes_db: pd.DataFrame
    :param meals_plan: Meals plan per user.
    :type meals_plan: Any
    :param flexi_probabilities_dict: Probabilities for flexible users
    :type flexi_probabilities_dict: dict[str, Any]
    :param meals_calorie_dict: calorie distribution per meal and user, defaults to meals_calorie_dict
    :type meals_calorie_dict: Dict[str, float], optional
    :param days_to_simulated: day top generate simulation, defaults to 365
    :type days_to_simulated: int, optional
    :param progress_bar: Progress bar object to be update as the simulation progress, defaults to None
    :type progress_bar: Any, optional
    :return: Simulated meals per user during num_days_to_simulate each key is an user and each value is the tracking Dataframe
    :rtype: Dict[str, pd.DataFrame]
    """
    update_amount = 90.0/len(df_user)
    # Prepare food database
    df_recipes_db = df_recipes_db.copy()
    df_recipes_db["allergies"] = df_recipes_db["allergies"].fillna("")
    # generate filtered datasets
    # user dataset updated with next state
    bmi_conditions = [BMI_constants.underweight,
                      BMI_constants.healthy,
                      BMI_constants.overweight,
                      BMI_constants.obesity]
    df_user_db = generate_next_BMI_based_on_transition_matrix(
        bmi_conditions=bmi_conditions,
        df_user=df_user,
        transition_matrix=transition_matrix
    )
    # create a functools partial user function
    cultural_ids = generate_cultural_factor_oriented_dataset(
        food_db=df_recipes_db
    )
    allergy_ids = generate_allergy_oriented_food_dataset(
        food_db=df_recipes_db
    )
    meal_type_ids = generate_meal_type_oriented_dataset(
        food_db=df_recipes_db
    )
    partial_user_generator = partial(
        generate_user_simulation,
        df_user=df_user_db,
        food_db=df_recipes_db,
        meals_probability_dict=meals_proba_dict,
        meals_calorie_distribution=meals_calorie_dict,
        allergies_food_ids=allergy_ids,
        cultural_food_ids=cultural_ids,
        meal_type_food_ids=meal_type_ids,
        place_probabilities=place_probabilities,
        social_situation_probabilities=social_situation_probabilities,
        chose_dist=chose_dist,
        meals_time_dict=meals_time_dict,
        delta_dist_params=delta_dist_params,
        dict_flexi_probas=flexi_probabilities_dict,
        days_to_simulated=days_to_simulated,
        bmi_conditions=bmi_conditions
    )
    # Generate user simulation
    # handle progress bar

    def handle_progress_bar(iteration):
        # print(f"Iteration: {iteration}")
        # print(f"update amount: {update_amount}")
        if progress_bar is not None:
            progress_bar.update(update_amount)

    # collect results
    track_df_list = []
    if len(df_user_db['userId'].unique()) > 60:
        @parfor(df_user_db['userId'].tolist(), bar=handle_progress_bar)
        def execute_parallel(user_id):
            try:
                return partial_user_generator(user_id)
            except Exception as e:
                print(f"Error processing user: {user_id}")
                # print(traceback.print_exc())
        track_df_list = execute_parallel
    else:
        for user in df_user_db['userId'].tolist():
            try:
                # print(f"processing user: {user}")
                track_df_list.append(partial_user_generator(user))
                if progress_bar is not None:
                    progress_bar.update(update_amount)
            except Exception as e:
                print(f"Error: {e}")
                # print(traceback.print_exc())
                continue
    # process output
    if len(track_df_list) > 1:
        final_tracking_df = pd.concat(track_df_list,
                                      axis=0,
                                      ignore_index=True)
        final_tracking_df.reset_index(inplace=True, drop=True)
    elif len(track_df_list) == 1:
        final_tracking_df = track_df_list[0]
    else:
        final_tracking_df = pd.DataFrame(columns=columns_tracking)
    # update bar in case is not updated
    if progress_bar is not None:
        pbar = progress_bar.get_progress_bar()
        if pbar.value < 100.0:
            update_value = 100.0 - pbar.value
            progress_bar.update(update_value,
                                bar_status='success' if len(final_tracking_df) > 0 else 'danger')
    return final_tracking_df


def generate_table_template(max_cols: int,
                            health_conditions: List[BMI_constants]
                            ) -> HTML_Table:
    # Create table
    table = HTML_Table(cols=max_cols)
    # Add rows to the table
    table.add_row(
        "<tr><th style=\"text-align: center;\" colspan=\"{span_cols}\"><strong>Tracking simulation: {days} days</strong></th></tr>")
    table.add_row("<tr><td style=\"text-align: left;\" colspan=\"{span_cols}\">\
        Total users: {total_users}</td></tr>")
    # add column gender
    table.add_row("""<tr>
                  <td style=\"text-align: left;\" colspan=\"{span_cols_half}\">{female_stats}</td>
                  <td style=\"text-align: left;\" colspan=\"{span_cols_half}\">{male_stats}</td>
                  </tr>""")
    # add health conditions
    table.add_row("""<tr>
                  <td style=\"text-align: left;\" colspan=\"{colspan}\">{underweight_stats}</td>
                  <td style=\"text-align: left;\" colspan=\"{colspan}\">{healthy_stats}</td>
                  <td style=\"text-align: left;\" colspan=\"{colspan}\">{overweight_stats}</td>
                  <td style=\"text-align: left;\" colspan=\"{colspan}\">{obesity_stats}</td>
                  </tr>""")
    # add allergies
    table.add_row(
        "<tr><th style=\"text-align: center;\" colspan=\"{span_cols}\"><strong>Allergies</strong></th></tr>")
    table.add_row("""<tr>
                <td style=\"text-align: left;\" colspan=\"{colspan}\">{underweight_allergy}</td>
                <td style=\"text-align: left;\" colspan=\"{colspan}\">{healthy_allergy}</td>
                <td style=\"text-align: left;\" colspan=\"{colspan}\">{overweight_allergy}</td>
                <td style=\"text-align: left;\" colspan=\"{colspan}\">{obesity_allergy}</td>
                </tr>""")
    # add Cultural factors
    table.add_row(
        "<tr><th style=\"text-align: center;\" colspan=\"{span_cols}\"><strong>Cultural factors</strong></th></tr>")
    table.add_row("""<tr>
                <td style=\"text-align: left;\" colspan=\"{colspan}\">{underweight_cultural}</td>
                <td style=\"text-align: left;\" colspan=\"{colspan}\">{healthy_cultural}</td>
                <td style=\"text-align: left;\" colspan=\"{colspan}\">{overweight_cultural}</td>
                <td style=\"text-align: left;\" colspan=\"{colspan}\">{obesity_cultural}</td>
                </tr>""")
    # add Contextual information
    table.add_row(
        "<tr><th style=\"text-align: center;\" colspan=\"{span_cols}\"><strong>Context Information</strong></th></tr>")
    table.add_row("""<tr>
                  <td style=\"text-align: left;\" colspan=\"{colspan}\">{underweight_context}</td>
                <td style=\"text-align: left;\" colspan=\"{colspan}\">{healthy_context}</td>
                <td style=\"text-align: left;\" colspan=\"{colspan}\">{overweight_context}</td>
                <td style=\"text-align: left;\" colspan=\"{colspan}\">{obesity_context}</td>
                  </tr>
                  """)
    # add food summary
    table.add_row(
        "<tr><th style=\"text-align: center;\" colspan=\"{span_cols}\"><strong>Food Summary</strong></th></tr>")
    table.add_row("""<tr>
            <td style=\"text-align: left;\" colspan=\"{colspan}\">{underweight_totals}</td>
            <td style=\"text-align: left;\" colspan=\"{colspan}\">{healthy_totals}</td>
            <td style=\"text-align: left;\" colspan=\"{colspan}\">{overweight_totals}</td>
            <td style=\"text-align: left;\" colspan=\"{colspan}\">{obesity_totals}</td>
            </tr>""")
    # Totals
    table.add_row(
        "<tr><th style=\"text-align: center;\" colspan=\"{span_cols}\"><strong>Totals</strong></th></tr>")
    table.add_row(
        """<tr><td style=\"text-align: left;\" colspan=\"{colspan}\">{Totals}</td></tr>""")
    return table


def create_a_summary_table(df_total_user: pd.DataFrame,
                           tracking_df: pd.DataFrame,
                           food_df: pd.DataFrame,
                           max_cols: int = 4,
                           round_digits: int = 0,
                           meals_calorie_dict: Dict[str, float] = meals_calorie_dict) -> HTML_Table:
    """Return an HTML_Table object that summarizes the generated synthetic dataset. 

    :param df_total_user: Join Dataframe that contains users' features.
    :type df_total_user: pd.DataFrame
    :param dict_recommendations: Dictionary with the simulated tracking data where keys are users' IDs and values tracking DataFrames. 
    :type dict_recommendations: Dict[str, pd.DataFrame]
    :param max_cols: Maximum number of columns for the table, defaults to 4
    :type max_cols: int, optional
    :param round_digits: number of digits to round float results in the table, defaults to 0
    :type round_digits: int, optional
    :param meals_calorie_dict: Dictionary with calorie distribution per meal, defaults to meals_calorie_dict
    :type meals_calorie_dict: Dict[str, float], optional
    :return: A HTML_table object that can be rendered into an HTML page. 
    :rtype: HTML_Table
    """
    # health conditions
    try:
        conditions = [BMI_constants.underweight, BMI_constants.healthy,
                      BMI_constants.overweight, BMI_constants.obesity]

        simulation_days = len(np.unique(tracking_df.loc[:, "day_number"]))
        total_users = df_total_user.shape[0]
        # Create table
        table = generate_table_template(max_cols=4,
                                        health_conditions=conditions)
        # fill the values
        table.set_value(0, {"span_cols": max_cols, "days": simulation_days})
        table.set_value(1, {"span_cols": max_cols, "total_users": total_users})
        # fill gender stats
        clinical_gender_count = df_total_user["clinical_gender"].value_counts()
        gender_text_template = "Medical gender {gender}: {num_users} users ({percentage} %)"
        male_text = ""
        female_text = ""
        for idx, item in clinical_gender_count.items():
            if idx == 'M':
                male_text = gender_text_template.format(gender="male",
                                                        num_users=item,
                                                        percentage=np.round((item/total_users)*100, round_digits))
            else:
                female_text = gender_text_template.format(gender="female",
                                                          num_users=item,
                                                          percentage=np.round((item/total_users)*100, round_digits))
        table.set_value(2, {"span_cols_half": max_cols/2,
                            "female_stats": female_text,
                            "male_stats": male_text})
        # fill health conditions
        condition_template_text = """Health condition {cond}: {user_cond} users ({cond_percent} %)
        <p>Male: {male_count} ({male_percent}%), Female: {female_count} ({female_percent}%) </p>
        """
        weight_condition = df_total_user["BMI"].value_counts()
        weight_condition_gender = df_total_user.groupby(
            by=["BMI", "clinical_gender"]).count()
        fill_dict = {"colspan": 1,
                     "underweight_stats": "",
                     "healthy_stats": "",
                     "overweight_stats": "",
                     "obesity_stats": ""}
        for condition in conditions:
            users_condition = 0
            percent_user = 0.0
            male_count = 0.0
            female_count = 0.0
            male_percent = 0.0
            female_percent = 0.0
            if condition in list(
                    weight_condition_gender.index.get_level_values(0)):
                # Check if M exist
                weight_gender_count = weight_condition_gender.xs(condition, level=0)[
                    "userId"]
                if 'M' in weight_gender_count.index:
                    male_count = weight_gender_count.M
                if 'F' in weight_gender_count.index:
                    female_count = weight_gender_count.F
                users_condition = weight_condition[condition]
                percent_user = np.round(
                    (users_condition/total_users)*100, round_digits)
                male_percent = np.round(
                    (male_count/users_condition)*100, round_digits)
                female_percent = np.round(
                    (female_count/users_condition)*100, round_digits)
            fill_dict[f"{condition}_stats"] = condition_template_text.format(cond=condition,
                                                                             user_cond=users_condition,
                                                                             cond_percent=percent_user,
                                                                             male_percent=male_percent,
                                                                             female_percent=female_percent,
                                                                             male_count=male_count,
                                                                             female_count=female_count
                                                                             )
        table.set_value(3, fill_dict)
        # fill allergies
        table.set_value(4, {"span_cols": max_cols})
        fill_dict = {"underweight_allergy": "",
                     "healthy_allergy": "",
                     "overweight_allergy": "",
                     "obesity_allergy": ""}
        df_groups = df_total_user.groupby(by=["BMI", "allergy"])
        df_counts = df_groups.count()
        first_index = df_counts.index.get_level_values(0)
        template_text = "<ul style=\"list-style-type: none;margin: 0;padding: 0;\" >{list_items}</ul>"
        for key in fill_dict.keys():
            if key.split("_")[0] in conditions:
                condition = key.split("_")[0]
                if condition in first_index:
                    per_condition_patient = 1
                    if condition in weight_condition.index:
                        per_condition_patient = weight_condition[condition]
                    temp_list = []
                    for allergy in df_counts.xs(condition, level=0).index:
                        users_count = df_counts.loc[(
                            condition, allergy), 'userId']
                        temp_list.append(f"""<li>{allergy.capitalize()}: {users_count}  
                                         <font color=\"red\">({np.round((users_count/total_users)*100, 2)} %)</font>
                                         <font color=\"green\">({np.round((users_count/per_condition_patient)*100, 2)} %)</font>
                                         </li>""")
                    fill_dict[key] = template_text.format(
                        list_items='\n'.join(temp_list))
                else:
                    fill_dict[key] = "N/A"
        fill_dict["colspan"] = 1
        table.set_value(5, fill_dict)
        # fill cultural factors
        table.set_value(6, {"span_cols": max_cols})
        fill_dict = {"underweight_cultural": "",
                     "healthy_cultural": "",
                     "overweight_cultural": "",
                     "obesity_cultural": ""}
        df_groups = df_total_user.groupby(by=["BMI", "cultural_factor"])
        df_counts = df_groups.count()
        first_index = df_counts.index.get_level_values(0)
        template_text = "<ul style=\"list-style-type: none;margin: 0;padding: 0;\" >{list_items}</ul>"
        for key in fill_dict.keys():
            if key.split("_")[0] in conditions:
                condition = key.split("_")[0]
                if condition in first_index:
                    per_condition_patient = 1
                    if condition in weight_condition.index:
                        per_condition_patient = weight_condition[condition]
                    temp_list = []
                    for cultural_fact in df_counts.xs(condition, level=0).index:
                        users_count = df_counts.loc[(
                            condition, cultural_fact), 'userId']
                        temp_list.append(f"""<li>{cultural_fact.capitalize()}: {users_count}  
                                         <font color=\"red\">({np.round((users_count/total_users)*100, 2)} %)</font> 
                                         <font color=\"green\">({np.round((users_count/per_condition_patient)*100, 2)} %)</font>
                                         </li>""")
                    fill_dict[key] = template_text.format(
                        list_items='\n'.join(temp_list))
                else:
                    fill_dict[key] = "N/A"
        fill_dict["colspan"] = 1
        table.set_value(7, fill_dict)
        # fill contextual data
        fill_dict = {"underweight_context": "",
                     "healthy_context": "",
                     "overweight_context": "",
                     "obesity_context": ""}
        df_counts = df_total_user["BMI"].value_counts()
        for condition in conditions:
            if condition in df_counts.index:
                # summarize track food
                selected_users = df_total_user[df_total_user["BMI"]
                                               == condition]["userId"].tolist()
                selected_tracking = tracking_df.query(
                    "userId in @selected_users").copy()
                temp_list = []
                for meal in list(meals_calorie_dict.keys()):
                    social_situation_count = 0.0
                    places_count = 0.0
                    delta_count = 0.0
                    meal_tracking = selected_tracking.query(
                        "meal_type == @meal")
                    if not meal_tracking.empty:
                        social_situation_count =\
                            meal_tracking["social_situation_of_meal_consumption"].value_counts(
                            )
                        places_count = \
                            meal_tracking["place_of_meal_consumption"].value_counts(
                            )
                        delta_count = {
                            "mean": meal_tracking["appreciation_feedback"].mean(),
                            "std": meal_tracking["appreciation_feedback"].std()
                        }
                        # Visualization
                        if len(places_count) == 0 and len(social_situation_count) == 0:
                            temp_list.append(f"""<li><strong>{meal.capitalize()}:</strong>
                                        <p>social situations consume meal: N/A</p> 
                                        <p>places consume meal: N/A</p>
                                        <p>appreciation: N/A</p>
                                        </li>""")
                        else:
                            temp_list.append(f"""<li><strong>{meal.capitalize()}:</strong>
                                            <p>social situations consume meal: {', '.join([ind+':'+str(social_situation_count[ind]) if ind != "N/A" else "N/A" for ind in social_situation_count.index])}</p> 
                                            <p>places consume meal: {', '.join([ind+':'+str(places_count[ind]) if ind != "N/A" else "N/A" for ind in places_count.index])}</p>
                                            <p>appreciation: {delta_count['mean']} &plusmn; {delta_count['std']}</p>
                                            </li>""")
                    else:
                        social_situation_count = 0.0
                        places_count = 0.0
                        delta_count = 0.0
                        temp_list.append(f"""<li>{meal.capitalize()}: 
                                        social situations consume meal: N/A, 
                                        places consume meal:  N/A, 
                                        appreciation: N/A
                                        </li>""")
                    fill_dict[f"{condition}_context"] = template_text.format(
                        list_items='\n'.join(temp_list))
            else:
                fill_dict[f"{condition}_context"] = "N/A"
        table.set_value(8, {"span_cols": max_cols})
        fill_dict["colspan"] = 1
        # print(fill_dict)
        table.set_value(9, fill_dict)
        # fill food summary
        table.set_value(10, {"span_cols": max_cols})
        fill_dict = {"underweight_totals": "",
                     "healthy_totals": "",
                     "overweight_totals": "",
                     "obesity_totals": ""}
        df_counts = df_total_user["BMI"].value_counts()
        for condition in conditions:
            if condition in df_counts.index:
                # summarize track food
                selected_users = df_total_user[df_total_user["BMI"] == condition]["userId"].tolist(
                )
                selected_tracking = tracking_df.query(
                    "userId in @selected_users").copy()
                temp_list = []
                for meal in list(meals_calorie_dict.keys()):
                    meal_tracking = selected_tracking.query(
                        "meal_type == @meal")
                    meal_tracking_food = meal_tracking.merge(
                        food_df, right_on="recipeId", left_on="foodId")
                    if not meal_tracking_food.empty:
                        mean = meal_tracking_food["calories"].mean()
                        std = meal_tracking_food["calories"].std()
                        recipes = len(meal_tracking_food["recipeId"])
                        unique_recipes = len(
                            meal_tracking_food["recipeId"].unique())
                    else:
                        mean = 0.0
                        std = 0.0
                        recipes = 0.0
                        unique_recipes = 0.0
                    temp_list.append(f"""<li>{meal.capitalize()}: 
                                     {recipes} recipes ({unique_recipes} unique recipes),
                                     # calories: {np.round(mean, 1)} Kcals 
                                     # &plusmn; {np.round(std, 1)} Kcals
                                     </li>""")
                fill_dict[f"{condition}_totals"] = template_text.format(
                    list_items='\n'.join(temp_list))
            else:
                fill_dict[f"{condition}_totals"] = "N/A"
        fill_dict["colspan"] = 1
        table.set_value(11, fill_dict)
        # fill totals
        table.set_value(12, {"span_cols": max_cols})
        fill_dict["Totals"] = ""
        temp_list = []
        for meal in meals_calorie_dict.keys():
            meal_tracking = tracking_df.query(
                "meal_type == @meal")
            meal_tracking_food = meal_tracking.merge(
                food_df, right_on="recipeId", left_on="foodId")
            if not meal_tracking_food.empty:
                mean = meal_tracking_food["calories"].mean()
                std = meal_tracking_food["calories"].std()
                recipes = len(meal_tracking_food)
                unique_recipes = len(meal_tracking_food["recipeId"].unique())
                if unique_recipes == 1:
                    recipes = 0.0
                    unique_recipes = 0.0
            else:
                mean = 0.0
                std = 0.0
                recipes = 0.0
                unique_recipes = 0.0
            temp_list.append(f"""<li>{meal.capitalize()}: 
                             {recipes} recipes ({unique_recipes} unique recipes),
                             # calories: {np.round(mean, 2)} Kcals &plusmn; 
                             # {np.round(std, 2)} Kcals
                             </li>""")
        fill_dict["Totals"] = template_text.format(
            list_items='\n'.join(temp_list))
        fill_dict["colspan"] = max_cols
        table.set_value(13, fill_dict)
    except Exception as e:
        # print(f"table error: {e}")
        raise Exception(f"Error generating table: {e}")
        # print(traceback.format_exc())
    return table


def save_outputs(base_path: str, output_folder: str, files: Dict[str, Any]):
    if os.path.exists(base_path):
        target_path = os.path.join(base_path, output_folder)
        if not os.path.exists(target_path):
            os.mkdir(target_path)
        # save files
        for k in files.keys():
            # print(f"key: {k}, extension: {k.split('.')[-1]}")
            if k.split(".")[-1] == "csv":
                files[k].to_csv(os.path.join(target_path, k))
            elif k.split(".")[-1] == "json":
                # save parameters in json format
                raw_parameters = files[k]
                with open(os.path.join(target_path, k), 'w') as fs:
                    json.dump(raw_parameters, fs, cls=JsonSerializationHelper)
            elif k.split(".")[-1] == "npy":
                # save parameters 
                save_path = os.path.join(target_path, k)
                np.save(save_path, files[k])
            elif k.split(".")[-1] == "png":
                files[k].render(filename=os.path.join(
                    target_path, k), format='png')
            else:
                with open(os.path.join(target_path, k), 'w') as fs:
                    fs.write(files[k])
    else:
        raise Exception(f"Output folder: {base_path} not found")


# Full pipeline to simulation
def run_full_simulation(num_users: int,
                        delta_dist_dict: Dict[str, Any],
                        chose_dist: str,
                        place_probabilities: Dict[str, Any],
                        social_situation_probabilities: Dict[str, Any],
                        age_probabilities: Dict[str, Any],
                        gender_probabilities: Dict[str, Any],
                        BMI_probabilities: Dict[str, Any],
                        allergies_probability_dict: Dict[str, Any],
                        food_restriction_probability_dict: Dict[str, Any],
                        flexi_probabilities: Dict[str, Any],
                        probability_transition_matrix: np.ndarray,
                        df_recipes: pd.DataFrame,
                        meals_proba: Dict[str, Any],
                        meals_time_dict: Dict[str,
                                              Any] = meal_time_distribution,
                        progress_bar=None,
                        num_days: int = 365,
                        multiple_allergies_number: int = 2
                        ) -> Tuple[pd.DataFrame, pd.DataFrame, HTML_Table]:
    """

    :param num_users: Executes the full simulation pipeline and produces a results Dataframe and a HTML summary table. 
    :type num_users: int
    :param gender_probabilities: Probability dictionary with the probability to generate male or female users. 
    :type gender_probabilities: Dict[str, Any]
    :param BMI_probabilities: Probability dictionary to generate distribution of BMI values (e.g., underweight, healthy, overweight, obesity).
    :type BMI_probabilities: Dict[str, Any]
    :param allergies_probability_dict: Probability dictionary with allergies and their occurrence probability.
    :type allergies_probability_dict: Dict[str, Any]
    :param food_restriction_probability_dict: Probability dictionary with cultural food restrictions like vegetarian, kosher, etc.
    :type food_restriction_probability_dict: Dict[str, Any]
    :param flexi_probabilities: Probability dictionary with divergence probability food. 
    :type flexi_probabilities: Dict[str, Any]
    :param probability_transition_matrix: probability of change from one BMI state to another. 
    :type probability_transition_matrix: np.ndarray
    :param df_recipes: Recipes' Dataframe.
    :type df_recipes: pd.DataFrame
    :param meals_proba: Probability of meal occurrences.
    :type meals_proba: Dict[str, Any]
    :param progress_bar: Progress bar widget to be update as the simulation progress, defaults to None
    :type progress_bar: Any, optional
    :param num_days: number days to simulate, defaults to 365
    :type num_days: int, optional
    :return: Tuple with Tracking DataFrame, Users join DataFrame, and Summary HTML table.
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, HTML_Table]
    """
    # Save simulation parameters 
    simulation_parameters = {
                'simulation_date': dt.datetime.now().strftime('%d-%m-%Y_%H-%M-%S'),
                "total_users": num_users,
                "simulation_days": num_days,
                'age_ranges': age_probabilities,
                'gender': gender_probabilities,
                'bmi': BMI_probabilities,
                'allergies': allergies_probability_dict,
                'cultural_restriction': food_restriction_probability_dict, 
                'meal_probabilities': meals_proba, 
                'flexible_probabilities': flexi_probabilities,
                'bmi_transition_probabilities': probability_transition_matrix,
                'meal_time': meals_time_dict, 
                'meals_probability': meals_proba,
                'place_of_meal_consumption': place_probabilities,
                'social_situation_of_meal_consumption': social_situation_probabilities,
                'appreciation_feedback': delta_dist_dict,
                "num_simultaneous_allergies": multiple_allergies_number,
                "delta_dist_chose": chose_dist,
    }
    # count time 
    start_time = time.time()
    # Generate user data
    df_personal_data = generate_personal_data(num_users=num_users,
                                              age_probabilities=age_probabilities,
                                              person_entity=person_entity,
                                              gender_probabilities=gender_probabilities)
    # Generate user status
    df_user_entity = generate_user_life_style_data(df_personal_data["userId"].tolist(),
                                                   user_entity=user_entity,
                                                   df_personal_data=df_personal_data,
                                                   BMI_probabilities_dict=BMI_probabilities)
    # Generate health conditions
    df_health_conditions = generate_health_condition_data(df_personal_data["userId"].tolist(),
                                                          allergies_probability_dict=allergies_probability_dict,
                                                          multiple_allergies_number=multiple_allergies_number)
    # Generate user goals
    df_user_goals = generate_user_goals(df_personal_data["userId"].tolist(),
                                        df_user_entity=df_user_entity)
    # Generate cultural factors
    df_cultural_factors = generate_cultural_data(df_personal_data["userId"].tolist(),
                                                 food_restriction_probability_dict=food_restriction_probability_dict,
                                                 flexi_probability_dict=flexi_probabilities)
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
    # Load recipes database
    df_recipes_filter = df_recipes[df_recipes["calories"] >= 0.0]
    # Generates meals plan
    meals_plan = generate_meals_plan_per_user(
        df_user_join["userId"].tolist(), meals_proba)
    # update bar
    if progress_bar is not None:
        progress_bar.update(10.0)
    # Execute simulation
    new_tracking_df = generate_recommendations(df_user_join,
                                               transition_matrix=probability_transition_matrix,
                                               chose_dist=chose_dist,
                                               social_situation_probabilities=social_situation_probabilities,
                                               place_probabilities=place_probabilities,
                                               delta_dist_params=delta_dist_dict,
                                               df_recipes_db=df_recipes_filter,
                                               meals_plan=meals_plan,
                                               flexi_probabilities_dict=flexi_probabilities,
                                               days_to_simulated=num_days,
                                               progress_bar=progress_bar,
                                               meals_time_dict=meals_time_dict)
    # Create a summary table
    table = create_a_summary_table(df_user_join, new_tracking_df, df_recipes)
    # return the files
    end_time = time.time() - start_time
    # print(f"Simulation finished in {end_time} seconds")
    if progress_bar:
        progress_bar.update_tail_text(f"Simulation finished in {np.round(end_time, 3)} seconds")
    return df_user_join, table, new_tracking_df, simulation_parameters
