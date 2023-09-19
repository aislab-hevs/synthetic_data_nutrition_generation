
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
age_probabilities_dict = {
    "20-29": 0.30,
    "30-39": 0.20,
    "40-49": 0.10,
    "50-59": 0.10,
    "60-69": 0.10,
    "70-79": 0.10,
    "80-89": 0.10,
    "90-100": 0.0
}

gender_probabilities_dict = {
    "M": 0.5,
    "F": 0.5
}

BMI_probabilities_dict = {
    "underweight": 0.10,
    "healthy": 0.30,
    "overweight": 0.30,
    "obesity": 0.30
}

allergies_probability_dict = {
    "cow's milk": 0.1,
    "eggs":  0.1,
    "peanut": 0.1,
    "soy": 0.1,
    "fish":  0.1,
    "tree nuts": 0.1,
    "shellfish": 0.1,
    "wheat": 0.1,
    "None": 0.2
}

food_restriction_probability_dict = {
    "vegan_observant": 0.2,
    "vegetarian_observant": 0.3,
    "halal_observant": 0.05,
    "kosher_observant": 0.05,
    "flexi_observant": 0.1,
    "None": 0.3
}

flexi_probabilities_dict = {
    "flexi_vegie": {
        "vegan_observant": 0.6,
        "vegetarian_observant": 0.2,
        "halal_observant": 0.05,
        "kosher_observant": 0.05,
        "None": 0.1
    },
    "flexi_vegetarian": {
        "vegan_observant": 0.0,
        "vegetarian_observant": 0.6,
        "halal_observant": 0.05,
        "kosher_observant": 0.05,
        "None": 0.3
    },
    "flexi_halal": {
        "vegan_observant": 0.1,
        "vegetarian_observant":  0.2,
        "halal_observant": 0.6,
        "kosher_observant": 0.0,
        "None": 0.1
    },
    "flexi_kosher": {
        "vegan_observant": 0.1,
        "vegetarian_observant": 0.1,
        "halal_observant": 0.1,
        "kosher_observant": 0.6,
        "None": 0.1
    }
}


# meals probabilities
meals_proba_dict = {
    "breakfast": 0.80,
    "morning snacks": 0.45,
    "afternoon snacks": 0.40,
    "lunch": 0.95,
    "dinner": 0.85
}
