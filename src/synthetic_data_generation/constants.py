
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

meals_calorie_dict = {"breakfast": 0.3,
                      "morning snacks": 0.05,
                      "afternoon snacks": 0.4,
                      "lunch": 0.05,
                      "dinner": 0.2}

# Initializers
# Dictionary initialization
# User number


def init_age_dict():
    # Generate age range
    age_range = person_entity.get("age_range")
    age_probabilities = dict(
        zip(age_range, [0 for i in range(len(age_range))]))
    return age_probabilities


def init_gender_dict():
    # Male and female distribution
    gender_probabilities = dict(
        zip(person_entity.get("clinical_gender"), [0.5, 0.5]))
    return gender_probabilities


def init_bmi_dict():
    # Generate BMI values
    BMI_values = ["underweight", "healthy", "overweight", "obesity"]
    BMI_prob = [0.1, 0.3, 0.3, 0.3]
    BMI_probabilities = dict(zip(BMI_values, BMI_prob))
    return BMI_probabilities


def init_allergies_dict():
    # Allergy array and probabilities
    allergies = ["cow's milk", "eggs", "peanut", "soy",
                 "fish", "tree nuts", "shellfish", "wheat", "None"]
    allergies_prob = [0.1, 0.1, 0.1,
                      0.1, 0.1, 0.1, 0.1, 0.1, 0.2]
    allergies_probability_dict = dict(zip(allergies, allergies_prob))
    return allergies_probability_dict


def init_food_restrictions():
    # Food restrictions probabilities
    food_restrictions = ["vegan_observant", "vegetarian_observant",
                         "halal_observant", "kosher_observant", "flexi_observant", "None"]
    food_restriction_probs = [0.2, 0.3, 0.05, 0.05, 0.1, 0.3]
    food_restriction_probability_dict = dict(
        zip(food_restrictions, food_restriction_probs))
    return food_restriction_probability_dict


def init_flexi_probabilities():
    # generate different probabilities for the flexible
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


# meals probabilities
meals_proba = {
    "breakfast": 0.80,
    "morning snacks": 0.45,
    "afternoon snacks": 0.40,
    "lunch": 0.95,
    "dinner": 0.85
}
