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
from .default_inputs import (person_entity,
                             user_entity,
                             meals_calorie_dict)


# Classes
class HTML_Table:
    def __init__(self, cols: int = 4, rows: List[str] = None):
        self.cols = cols
        if rows is not None:
            self.rows = rows
        else:
            self.rows = []

    def add_rows(self, new_rows=List[str]):
        self.rows.extend(new_rows)

    def add_row(self, row: str):
        self.rows.append(row)

    def _repr_html_(self):
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
    Percentage in red color represent the total percentage respect to all users.
    </li>
    <li>
    Percentage in green color represent the percentage respect to health conditions.
    </li>
    </ul>
    </p>
    </body>
    </html>""".format(row="\n".join(self.rows))

    def render(self):
        return self._repr_html_()


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


def generate_personal_data(gender_probabilities: Dict[str, Any], num_users: int = 500, person_entity: Dict[str, Any] = None) -> pd.DataFrame:
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
                                  BMI_probabilities_dict: Dict[str, Any],
                                  df_personal_data: pd.DataFrame,
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
    BMI_values = []
    BMI_prob = []
    for k, v in BMI_probabilities_dict.items():
        BMI_values.append(k)
        BMI_prob.append(v)
    bmis = np.random.choice(BMI_values, size=num_users,
                            replace=True, p=BMI_prob)
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


def generate_health_condition_data(list_user_id: List[str], allergies_probability_dict: Dict[str, Any]):
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


def generate_user_goals(list_user_id: List[str], df_user_entity: pd.DataFrame) -> pd.DataFrame:
    df_user_goals = pd.DataFrame(columns=["userId", "nutrition_goal"], data=[])
    df_user_goals["userId"] = list_user_id
    df_user_goals["nutrition_goal"] = df_user_entity["BMI"].apply(
        lambda x: define_user_goal_according_BMI(x))
    return df_user_goals


def assign_probabilities(cultural_factor: str,
                         flexi_probability_dict: Dict[str, Any]):
    if cultural_factor == "flexi_observant":
        flexi_proba = flexi_probability_dict
        value = np.random.choice(list(flexi_proba.keys()))
        return value
    pass


def generate_cultural_data(list_user_id: List[str], food_restriction_probability_dict: Dict[str, Any],
                           flexi_probability_dict: Dict[str, Any]):
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


def generate_preferences_data(list_user_id: List[str], df_personal_data: pd.DataFrame) -> pd.DataFrame:
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


def calculate_daily_calorie_needs(BMR: float, activity_level: str):
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


def define_daily_calorie_plan(nutrition_goal: str, daily_calorie_need: float):
    projected_calories_need = 0
    if nutrition_goal == NutritionGoals.gain_weight:
        projected_calories_need = daily_calorie_need + 500
    elif nutrition_goal == NutritionGoals.maintain_fit:
        projected_calories_need = daily_calorie_need
    else:
        projected_calories_need = daily_calorie_need - 500
    return projected_calories_need


def generate_diet_plan(weight: float,
                       height: float,
                       age_range: str,
                       clinical_gender: str,
                       activity_level: str,
                       nutrition_goal: str):
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


def allergy_searcher(recipes_db_allergy_col, allergy: str):
    res = []
    allergy_low = allergy.lower()
    for item in recipes_db_allergy_col.items():
        text = str(item[1]).lower()
        if allergy_low in text:
            print(text)
            print(allergy_low)
            res.append(False)
        else:
            res.append(True)
    return res


def generate_meals_plan_per_user(users: List[str], probability_dict: Dict[str, float]):
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
                             progress_bar: Any = None):

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
                                              index=np.arange(1, days_to_simulated+1))
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
            if cultural_factor != "None" and cultural_factor != "flexi_observant":
                # new method
                filtered_recipe_db = filtered_recipe_db.query(query_text.get(cultural_factor,
                                                                             default=""
                                                                             ))
            elif cultural_factor == "flexi_observant":
                # get flexi_proba
                flexi_class = df_user.loc[i, "probabilities"]
                flexi_probas = dict_flexi_probas[flexi_class]
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
            continue
    return simulation_results


def create_a_summary_table(df_total_user, dict_recommendations, max_cols=4, round_digits=0,
                           meals_calorie_dict: Dict[str, float] = meals_calorie_dict):
    # Create table
    table = HTML_Table(cols=max_cols)
    total_users = df_total_user.shape[0]
    random_user = np.random.choice(list(dict_recommendations.keys()))
    simulation_days = dict_recommendations.get(random_user).shape[0]
    # Add rows to the table
    table.add_row(
        "<tr><th style=\"text-align: center;\" colspan=\"{span_cols}\"><strong>Tracking simulation: {days} days</strong></th></tr>".format(
            span_cols=max_cols,
            days=simulation_days))
    table.add_row("<tr><td style=\"text-align: left;\" colspan=\"{span_cols}\">\
        Total users: {total_users}</td></tr>".format(
        span_cols=max_cols,
        total_users=total_users))
    # Clinical gender
    clinical_gender_count = df_total_user["clinical_gender"].value_counts()
    # Show clinical gender
    temp_row = []
    for idx, item in clinical_gender_count.items():
        temp_row.append(f"<td style=\"text-align: left;\" colspan=\"{max_cols/2}\">Clinical gender\
            {'male' if idx == 'M' else 'female'}: {item}\
            users ({np.round((item/total_users)*100, round_digits)} %)</td>")
    table.add_row("<tr>{row_data}</tr>".format(row_data="".join(temp_row)))
    # Health condition
    weight_condition = df_total_user["BMI"].value_counts()
    weight_condition_gender = df_total_user.groupby(
        by=["BMI", "clinical_gender"]).count()
    temp_row = []
    for idx, item in weight_condition.items():
        weight_gender_count = weight_condition_gender.xs(idx, level=0)[
            "userId"]
        temp_row.append(f"<td style=\"text-align: left;\">Health condition {idx}:\
            {item} users ({np.round((item/total_users)*100, round_digits)} %) \
                <font color=\"lightblue\">(Male: {np.round((weight_gender_count.M/item)*100, round_digits)}%</font>,\
                <font color=\"pink\">Female: {np.round((weight_gender_count.F/item)*100, round_digits)}%</font>)</td>")
    table.add_row("<tr>{row_data}</tr>".format(row_data="".join(temp_row)))
    # Allergies
    table.add_row(
        "<tr><th style=\"text-align: center;\" colspan=\"{span_cols}\"><strong>Allergies</strong></th></tr>".format(span_cols=max_cols))
    df_groups = df_total_user.groupby(by=["BMI", "allergy"])
    temp_row = []
    temp_cols = []
    df_counts = df_groups.count()
    allergy_index = list(df_counts.xs("healthy", level=0).index)
    for allergy in allergy_index:
        temp_cols = []
        for hl, per_condition_patient in weight_condition.items():
            # check number
            # print(f"hl: {hl}, allergy: {allergy}")
            row_number = df_counts.query(
                f'allergy.str.contains("{allergy.strip()}") and BMI.str.contains("{hl.strip()}")').shape[0]
            # print(f"row number: {row_number}")
            # check index
            if row_number > 0 and any(df_counts.index.isin([(hl, allergy)])):
                users_count = df_counts.loc[(hl, allergy), 'userId']
            else:
                users_count = 0
            temp_cols.append(f"<td style=\"text-align: left;\">{allergy}: {users_count}\
                <font color=\"red\">({np.round((users_count/total_users)*100, 2)} % total)</font> \
                <font color=\"green\">({np.round((users_count/per_condition_patient)*100, 2)} % relative)</font> </td>")
        table.add_row("<tr>{cells}</tr>".format(cells="".join(temp_cols)))
    # Cultural factors
    table.add_row(
        "<tr><th style=\"text-align: center;\" colspan=\"{span_cols}\"><strong>Cultural factors</strong></th></tr>".format(span_cols=max_cols))
    df_groups = df_total_user.groupby(by=["BMI", "cultural_factor"])
    temp_row = []
    temp_cols = []
    df_counts = df_groups.count()
    allergy_index = list(df_counts.xs("healthy", level=0).index)
    for allergy in allergy_index:
        temp_cols = []
        for hl, per_condition_patient in weight_condition.items():
            # check number
            row_number = df_counts.query(
                f'cultural_factor.str.contains("{allergy.strip()}") and BMI.str.contains("{hl.strip()}")'
            ).shape[0]
            if row_number > 0:
                users_count = df_counts.loc[(hl, allergy), 'userId']
            else:
                users_count = 0
            temp_cols.append(f"<td style=\"text-align: left;\">{allergy}: {users_count}\
                <font color=\"red\">({np.round((users_count/total_users)*100, 2)} % total)</font> \
                <font color=\"green\">({np.round((users_count/per_condition_patient)*100, 2)} % relative)</font> </td>")
        table.add_row("<tr>{cells}</tr>".format(cells="".join(temp_cols)))
    # Food summary
    table.add_row(
        "<tr><th style=\"text-align: center;\" colspan=\"{span_cols}\"><strong>Food Summary</strong></th></tr>".format(span_cols=max_cols))
    temp_dict = {}
    for key, _ in weight_condition.items():
        df_users_list = []
        users = df_total_user[df_total_user["BMI"] == key]["userId"].tolist()
        for u in users:
            if u in dict_recommendations.keys():
                df_users_list.append(dict_recommendations[u])
        temp_dict[key] = pd.concat(df_users_list, axis=0)
    # visualize
    total_recipes = []
    total_recipes_unique = []
    total_recipes_per_meal = {}
    temp_row = []
    temp_cols = []
    df_counts = df_groups.count()
    meals_index = list(meals_calorie_dict.keys())
    for meal in meals_index:
        temp_cols = []
        for hl, _ in weight_condition.items():
            mean = temp_dict[hl][f"{meal}_calories"].mean()
            std = temp_dict[hl][f"{meal}_calories"].std()
            recipes = len(temp_dict[hl][meal])
            unique_recipes = len(temp_dict[hl][meal].unique())
            temp_cols.append(f"<td style=\"text-align: left;\">{meal}: {recipes} recipes ({unique_recipes} unique recipes),\
                calories: {np.round(mean, 1)} Kcals &plusmn; {np.round(std, 1)} Kcals </td>")
        table.add_row("<tr>{cells}</tr>".format(cells="".join(temp_cols)))
    # Total recipes
    table.add_row(
        "<tr><th style=\"text-align: center;\" colspan=\"{span_cols}\"><strong>Totals</strong></th></tr>".format(span_cols=max_cols))
    for key, _ in weight_condition.items():
        total_recipes.append(len(temp_dict[key]))
        list_vals = [len(temp_dict[key][x].unique())
                     for x in meals_calorie_dict.keys()]
        # print(list_vals)
        flat_list = []
        total_recipes_unique.append(sum(list_vals))
    table.add_row("<tr>{values}</tr>".format(
        values="".join([f"<td style=\"text-align: left;\">Total recommend meals: {total_recipes[i]} ({total_recipes_unique[i]} unique)</td>" for i in range(len(total_recipes))])))
    # total recipes per meal
    total_df_list = [temp_dict[k] for k in temp_dict.keys()]
    total_df = pd.concat(total_df_list, axis=0)
    for meal in meals_calorie_dict.keys():
        total_recommendations = len(total_df[meal])
        total_meal = len(total_df[meal].unique())
        table.add_row(f"<tr><td style=\"text-align: left;\", colspan={max_cols}>Total {meal}:\
            {total_recommendations} ({total_meal} uniques)</td></tr>")
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
                        ):
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
