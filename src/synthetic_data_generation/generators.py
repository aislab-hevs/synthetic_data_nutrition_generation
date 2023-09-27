import pandas as pd
import numpy as np
from faker import Faker
from enum import Enum
from typing import List, Any, Tuple, Dict
import string
import uuid
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
                             meal_time_distribution)


# Classes
class HTML_Table:
    """This classes produces and static html page that visualizes the summary table after data generation.
    """

    def __init__(self,
                 cols: int = 4,
                 rows: List[str] = None) -> None:
        """Constructor method to create HTML_table object able to render an static HTML page with the summary table. 

        :param cols: maximum number of columns in the table, defaults to 4
        :type cols: int, optional
        :param rows: List of rows to incorporate in the table, each row is a string containing HTML tags <tr> <th>, defaults to None
        :type rows: List[str], optional
        """
        self.cols = cols
        if rows is not None:
            self.rows = rows
        else:
            self.rows = []

    def add_rows(self, new_rows: List[str]) -> None:
        """Add a list of rows to the HTML table.

        :param new_rows:  List of rows to be added to the HTML table
        :type new_rows: List[str]
        """
        self.rows.extend(new_rows)

    def add_row(self, row: str) -> None:
        """Add one row to the HTML table. 

        :param row: row to be added to the table.
        :type row: str
        """
        self.rows.append(row)

    def get_row(self, row_index: int):
        if row_index >= 0 and row_index < len(self.rows):
            return self.rows[row_index]

    def get_row_count(self):
        return len(self.rows)

    def get_rows(self):
        return self.rows

    def set_value(self, row_number: int, dict_parameters: Dict[str, Any]):
        # check index
        if row_number >= 0 and row_number < len(self.rows):
            self.rows[row_number] = self.rows[row_number].format(
                **dict_parameters)

    def _repr_html_(self) -> str:
        return """
    <!DOCTYPE HTML PUBLIC
	 	"-//W3 Organization//DTD W3 HTML 2.0//EN">
    <html>
    <head>
    <title>
    Summary Table
    </title>
    </head>
    <body>
    <table border=\"1\">
        {row}
    </table>
    <p>
    <ul>
    <li>
    Percentage in <font color="red">red color</font> represent the total percentage respect to all users.
    </li>
    <li>
    Percentage in <font color="green">green color</font> represent the percentage respect to health conditions.
    </li>
    </ul>
    </p>
    </body>
    </html>""".format(row="\n".join(self.rows))

    def render(self) -> str:
        """Returns the HTML str that represents the summary table. 

        :return: HTML string representing the HTML summary table. 
        :rtype: str
        """
        return self._repr_html_()


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
    df_personal_data["age_range"] = list(
        map(lambda x: generate_age_range(), range(num_users)))
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
    return list(map(lambda x: np.random.choice(list_values, size=size, replace=replace, p=probabilities), range(samples)))


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
                                   allergies_probability_dict: Dict[str, Any]):
    """Generate users health conditions (allergies) based on probability dictionary. 

    :param list_user_id: users' IDs
    :type list_user_id: List[str]
    :param allergies_probability_dict: Probability dictionary where keys are allergy conditions and values their probabilities, total probabilities should sum up 1
    :type allergies_probability_dict: Dict[str, Any]
    :return: Pandas Dataframe with users IDs and their assigned health condition. 
    :rtype: _type_
    """
    df_health_conditions = pd.DataFrame(data=[], columns=["userId", "allergy"])
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
    calories_daily = 0
    if activity_level == ActivityLevel.sedentary:
        calories_daily = 1.2 * BMR
    elif activity_level == ActivityLevel.light_active:
        calories_daily = 1.375 * BMR
    elif activity_level == ActivityLevel.moderate_active:
        calories_daily = 1.725 * BMR
    else:
        calories_daily = 1.9 * BMR
    return calories_daily


def define_daily_calorie_plan(nutrition_goal: NutritionGoals, daily_calorie_need: float) -> float:
    """Calculate the user's projected calorie needs to reach the nutrition goal according daily calorie needs.

    :param nutrition_goal:user's nutrition goals (e.g., loss weight, maintain, gain weight)
    :type nutrition_goal: NutritionGoals
    :param daily_calorie_need: user's daily calorie needs
    :type daily_calorie_need: float
    :return: projected daily user's calories needs to reach the nutrition goal
    :rtype: float
    """
    projected_calories_need = 0
    if nutrition_goal == NutritionGoals.gain_weight:
        # Add or remove calories to create metabolic deficit
        projected_calories_need = daily_calorie_need + 500
    elif nutrition_goal == NutritionGoals.maintain_fit:
        projected_calories_need = daily_calorie_need
    else:
        projected_calories_need = daily_calorie_need - 500
    return projected_calories_need


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
    return projected_calorie_needs


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
    df_treatment = pd.DataFrame(
        data=[], columns=["userId", "projected_daily_calories"])
    df_treatment["userId"] = list_user_id
    # prepare data
    df_user_data = df_user_goals.merge(df_personal_data[["userId", "clinical_gender", "age_range"]],
                                       on="userId")
    df_user_data = df_user_data.merge(df_user_entity[["userId", "life_style", "weight", "height"]],
                                      on="userId")
    df_treatment["projected_daily_calories"] = np.ceil(df_user_data.apply(lambda row: generate_diet_plan(weight=row["weight"],
                                                                                                         height=row["height"],
                                                                                                         age_range=row["age_range"],
                                                                                                         clinical_gender=row[
                                                                                                             "clinical_gender"],
                                                                                                         activity_level=row[
                                                                                                             "life_style"],
                                                                                                         nutrition_goal=row[
                                                                                                             "nutrition_goal"]
                                                                                                         ), axis=1))
    return df_treatment, df_user_data


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


def generate_recommendations(df_user: pd.DataFrame,
                             transition_matrix: np.array,
                             df_recipes_db: pd.DataFrame,
                             meals_plan: Any,
                             flexi_probabilities_dict: dict[str, Any],
                             meals_calorie_dict: Dict[str,
                                                      float] = meals_calorie_dict,
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

    query_text = {
        "vegan_observant": "cultural_restriction =='vegan'",
        "vegetarian_observant": "cultural_restriction =='vegan' |  cultural_restriction =='vegetarian'",
        "halal_observant": "cultural_restriction =='halal'",
        "kosher_observant": "cultural_restriction =='kosher'"
    }
    dict_flexi_probas = flexi_probabilities_dict
    simulation_results = {}
    df_recipes_db = df_recipes_db.copy()
    df_recipes_db["allergies"] = df_recipes_db["allergies"].fillna("")
    update_amount = 90.0/len(df_user)
    bmi_conditions = [BMI_constants.underweight,
                      BMI_constants.healthy,
                      BMI_constants.overweight,
                      BMI_constants.obesity]
    if transition_matrix is not None:
        # Generate users probabilities to success or fail the process
        for idx, bmi in enumerate(bmi_conditions):
            # choose users from that condition
            users_bmi_condition = df_user.query(f"BMI == '{bmi.value}'")
            if len(users_bmi_condition) > 0:
                # choose probabilities to this user
                next_state = np.random.choice([BMI_constants.underweight.value,
                                               BMI_constants.healthy.value,
                                               BMI_constants.overweight.value,
                                               BMI_constants.obesity.value],
                                              size=len(users_bmi_condition),
                                              p=transition_matrix[idx, :]
                                              )
                # set the next value
                df_user.loc[users_bmi_condition.index, "next_BMI"] = next_state
        # check df_integrity
        mask_nan = df_user["next_BMI"].isna()
        df_user.loc[mask_nan, "next_BMI"] = ""
    # Start processing
    for i in range(len(df_user)):
        # Generate recommendations for each user
        try:
            if progress_bar is not None:
                progress_bar.update(update_amount)
            user_db = df_user.iloc[i, :]
            daily_calories = user_db.projected_daily_calories
            current_state = user_db.BMI
            next_state = user_db.next_BMI
            if next_state == "":
                next_state = current_state
            # re-asses daily calories
            # maintain weight
            # daily required
            if current_state == BMI_constants.overweight or current_state != BMI_constants.obesity:
                daily_required_calories = daily_calories+500
            elif current_state == BMI_constants.healthy:
                daily_required_calories = daily_calories
            else:
                daily_required_calories = daily_calories-500
            # choose daily calories
            if current_state == next_state:
                daily_calories_list = [
                    daily_required_calories for i in range(days_to_simulated)]
            # gain weight
            elif bmi_conditions.index(current_state) < bmi_conditions.index(next_state):
                daily_calories_list = [
                    daily_required_calories+500 for i in range(days_to_simulated)]
            # lose weight
            else:
                daily_calories_list = [
                    daily_required_calories-500 for i in range(days_to_simulated)]
            flexi_probas = None
            df_recommendations = pd.DataFrame(columns=[f"{k}_calories" for k in meals_calorie_dict.keys()]+list(meals_calorie_dict.keys()),
                                              index=np.arange(0, days_to_simulated))
            # filter cultural factor and allergies
            # allergy restrictions filter
            allergies_factor = user_db.allergy
            if allergies_factor != "None":
                filtered_recipe_db = df_recipes_db[~df_recipes_db["allergies"].str.contains(
                    allergies_factor)]
            else:
                filtered_recipe_db = df_recipes_db
            if filtered_recipe_db.shape[0] == 0:
                # Remove filter if it is empty
                filtered_recipe_db = df_recipes_db
            # cultural restrictions filter
            cultural_factor = user_db.cultural_factor
            if cultural_factor is not None or cultural_factor != "None":
                if cultural_factor != "None" and cultural_factor != "flexi_observant":
                    # new method
                    # print(f"Cultural factor: {cultural_factor}")
                    filtered_recipe_db = filtered_recipe_db.query(query_text.get(cultural_factor,
                                                                                 ""
                                                                                 ))
                elif cultural_factor == "flexi_observant":
                    # get flexi_proba
                    flexi_class = df_user.loc[i, "probabilities"]
                    flexi_probas = dict_flexi_probas[flexi_class]
                else:
                    filtered_recipe_db = df_recipes_db
            else:
                filtered_recipe_db = df_recipes_db
            if filtered_recipe_db.shape[0] == 0:
                # remove filter if it is empty
                filtered_recipe_db = df_recipes_db
            for meal_tp in ["lunch", "dinner", "breakfast", "morning snacks", "afternoon snacks"]:
                # generate recommendations
                # generate meals according to meals plan
                meal = meals_plan[meal_tp]
                if meal[i] != 0:
                    meal_db = filtered_recipe_db[filtered_recipe_db["meal_type"] == meal_tp]
                    if meal_db.shape[0] == 0:
                        meal_db = filtered_recipe_db
                    meal_chosen = []
                    for j in range(days_to_simulated):
                        # flexi
                        if flexi_probas is not None:
                            # print("Flexi proba...")
                            list_flexi_classes = [
                                k for k in flexi_probas.keys()]
                            list_flexi_probas = [flexi_probas[k]
                                                 for k in list_flexi_classes]
                            flexi_meal = np.random.choice(
                                list_flexi_classes, p=list_flexi_probas)
                            if flexi_meal != "None":
                                flexi_meal = flexi_meal.split("_")[0]
                                meal_db = meal_db[meal_db["cultural_restriction"]
                                                  == flexi_meal]
                            if meal_db.shape[0] == 0:
                                meal_db = filtered_recipe_db[filtered_recipe_db["meal_type"] == meal_tp]
                        max_calories_meal = daily_calories_list[j] * \
                            meals_calorie_dict[meal_tp]
                        possible_recipes = meal_db[meal_db["calories"]
                                                   <= max_calories_meal+np.random.normal(0, 50)]
                        if possible_recipes.shape[0] == 0:
                            possible_recipes = meal_db
                        choose_recipes = possible_recipes.sample(
                            1, replace=True)
                        meal_chosen.append(
                            choose_recipes[["title", "calories"]])
                        # update counter
                        daily_calories_list[j] = daily_calories_list[j] - \
                            choose_recipes['calories'].values[0]
                    total_simulations = pd.concat(meal_chosen)
                    total_simulations.reset_index(inplace=True)
                    df_recommendations[f"{meal_tp}_calories"] = total_simulations['calories']
                    df_recommendations[meal_tp] = total_simulations['title']
                else:
                    df_recommendations[meal_tp] = [
                        "N/A" for i in range(days_to_simulated)]
                    df_recommendations[f"{meal_tp}_calories"] = [
                        0 for i in range(days_to_simulated)]
            simulation_results[f"{user_db.userId}"] = df_recommendations
        except Exception as e:
            # print(f"Error processing user: {df_user.iloc[i, 0]}, {e}")
            # print(traceback.print_exc())
            continue
    # print(f"Simulation len: {len(simulation_results)}")
    return simulation_results


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
                           dict_recommendations: Dict[str, pd.DataFrame],
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
        random_user = np.random.choice(list(dict_recommendations.keys()))
        simulation_days = dict_recommendations.get(random_user).shape[0]
        total_users = df_total_user.shape[0]
        # Create table
        table = generate_table_template(max_cols=4,
                                        health_conditions=conditions)
        # fill the values
        table.set_value(0, {"span_cols": max_cols, "days": simulation_days})
        table.set_value(1, {"span_cols": max_cols, "total_users": total_users})
        # fill gender stats
        clinical_gender_count = df_total_user["clinical_gender"].value_counts()
        gender_text_template = "Clinical gender {gender}: {num_users} users ({percentage} %)"
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
        <p>(Male: {male_percent}%), (Female: {female_percent}%) </p>
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
            male_percent = 0.0
            female_percent = 0.0
            if condition in list(
                    weight_condition_gender.index.get_level_values(0)):
                # Check if M exist
                weight_gender_count = weight_condition_gender.xs(condition, level=0)[
                    "userId"]
                male_count = 0
                female_count = 0
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
                                                                             female_percent=female_percent
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
                                         <font color=\"red\">({np.round((users_count/total_users)*100, 2)} % total)</font>
                                         <font color=\"green\">({np.round((users_count/per_condition_patient)*100, 2)} % relative)</font>
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
                                         <font color=\"red\">({np.round((users_count/total_users)*100, 2)} % total)</font> 
                                         <font color=\"green\">({np.round((users_count/per_condition_patient)*100, 2)} % relative)</font>
                                         </li>""")
                    fill_dict[key] = template_text.format(
                        list_items='\n'.join(temp_list))
                else:
                    fill_dict[key] = "N/A"
        fill_dict["colspan"] = 1
        table.set_value(7, fill_dict)
        # fill food summary
        table.set_value(8, {"span_cols": max_cols})
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
                # print(len(selected_users))
                df_users_list = []
                for u in selected_users:
                    if u in dict_recommendations.keys():
                        df_users_list.append(dict_recommendations[u])
                # print(len(df_users_list))
                # check objects to concat
                if len(df_users_list) > 1:
                    temp_df = pd.concat(df_users_list, axis=0)
                elif len(df_users_list) == 1:
                    # one user
                    # print("One user detected")
                    temp_df = df_users_list[0]
                else:
                    temp_df = pd.DataFrame()
                # print(len(temp_df))
                # summarize
                temp_list = []
                for meal in list(meals_calorie_dict.keys()):
                    if not temp_df.empty:
                        mean = temp_df[f"{meal}_calories"].mean()
                        std = temp_df[f"{meal}_calories"].std()
                        recipes = len(temp_df[meal])
                        unique_recipes = len(temp_df[meal].unique())
                        # print(
                        # f"calculated values condition {condition}: {meal} mean {mean} std {std}")
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
        table.set_value(9, fill_dict)
        # fill totals
        table.set_value(10, {"span_cols": max_cols})
        fill_dict["Totals"] = ""
        temp_list = []
        for u in dict_recommendations.keys():
            df_users_list.append(dict_recommendations[u])
        # calculate totals
        if len(df_users_list) > 1:
            temp_df = pd.concat(df_users_list, axis=0)
        elif len(df_users_list) == 1:
            temp_df = df_users_list[0]
        else:
            temp_df = pd.DataFrame()
        for meal in meals_calorie_dict.keys():
            if not temp_df.empty:
                mean = temp_df[f"{meal}_calories"].mean()
                std = temp_df[f"{meal}_calories"].std()
                recipes = len(temp_df[meal])
                unique_recipes = len(temp_df[meal].unique())
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
        table.set_value(11, fill_dict)
    except Exception as e:
        # print(f"table error: {e}")
        raise Exception(f"Error generating table: {e}")
        # print(traceback.format_exc())
    return table

# Full pipeline to simulation


def run_full_simulation(num_users: int,
                        gender_probabilities: Dict[str, Any],
                        BMI_probabilities: Dict[str, Any],
                        allergies_probability_dict: Dict[str, Any],
                        food_restriction_probability_dict: Dict[str, Any],
                        flexi_probabilities: Dict[str, Any],
                        probability_transition_matrix: np.ndarray,
                        df_recipes: pd.DataFrame,
                        meals_proba: Dict[str, Any],
                        progress_bar=None,
                        num_days: int = 365
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
    # Generate user data
    df_personal_data = generate_personal_data(num_users=num_users,
                                              person_entity=person_entity,
                                              gender_probabilities=gender_probabilities)
    # Generate user status
    df_user_entity = generate_user_life_style_data(df_personal_data["userId"].tolist(),
                                                   user_entity=user_entity,
                                                   df_personal_data=df_personal_data,
                                                   BMI_probabilities_dict=BMI_probabilities)
    # Generate health conditions
    df_health_conditions = generate_health_condition_data(df_personal_data["userId"].tolist(),
                                                          allergies_probability_dict=allergies_probability_dict)
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
    # Simulate the transition matrix
    counts = df_user_entity.groupby(by="BMI").count()["userId"]
    # unify all the dataframes
    df_user_join = df_user_data.merge(df_treatment, on="userId")
    df_user_join = df_user_join.merge(df_cultural_factors,  on="userId")
    df_user_join = df_user_join.merge(df_health_conditions,  on="userId")
    df_user_join = df_user_join.merge(
        df_user_entity[["userId", "BMI"]],  on="userId")
    # Load recipes database
    df_recipes_filter = df_recipes[df_recipes["calories"] >= 0.0]
    # Generates meals plan
    meals_plan = generate_meals_plan_per_user(
        df_user_join["userId"].tolist(), meals_proba)
    # update bar
    if progress_bar is not None:
        progress_bar.update(10.0)
    # Execute simulation
    simulation_results = generate_recommendations(df_user_join,
                                                  transition_matrix=probability_transition_matrix,
                                                  df_recipes_db=df_recipes_filter,
                                                  meals_plan=meals_plan,
                                                  flexi_probabilities_dict=flexi_probabilities,
                                                  days_to_simulated=num_days,
                                                  progress_bar=progress_bar)
    # Create a summary table
    table = create_a_summary_table(df_user_join, simulation_results)
    return simulation_results, df_user_join, table
