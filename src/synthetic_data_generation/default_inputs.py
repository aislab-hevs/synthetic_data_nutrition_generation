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

height_distribution = {
    'male': {
        'mean': 170,
        'std': 10
    },
    'female': {
        'mean': 160,
        'std': 10
    }
}

meal_time_distribution = {
    "breakfast": {
        "mean": 8,
        "std": 1
    },
    "morning snacks": {
        "mean": 10,
        "std": 1
    },
    "afternoon snacks": {
        "mean": 16,
        "std": 1
    },
    "lunch": {
        "mean": 13,
        "std": 1
    },
    "dinner": {
        "mean": 20,
        "std": 1
    }
}

meals_calorie_dict = {"breakfast": 0.3,
                      "morning snacks": 0.05,
                      "afternoon snacks": 0.4,
                      "lunch": 0.05,
                      "dinner": 0.2}

# Initializers
age_probabilities_dict = {
    "20-29": 0.20,
    "30-39": 0.20,
    "40-49": 0.10,
    "50-59": 0.10,
    "60-69": 0.10,
    "70-79": 0.10,
    "80-89": 0.10,
    "90-100": 0.10
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
    "vegetarian_observant": 0.2,
    "halal_observant": 0.10,
    "kosher_observant": 0.10,
    "flexi_observant": 0.10,
    "None": 0.30
}

flexi_probabilities_dict = {
    "flexi_vegie": {
        "vegan_observant": 0.60,
        "vegetarian_observant": 0.20,
        "halal_observant": 0.10,
        "kosher_observant": 0.1,
        "None": 0.0
    },
    "flexi_vegetarian": {
        #no meanginful flexi for this class:
        # vegetarian -> vegan
        "vegan_observant": 0.00,
        "vegetarian_observant": 0.60,
        "halal_observant": 0.0,
        "kosher_observant": 0.1,
        "None": 0.30
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

DEFAULT_NUM_USERS = 500

DEFAULT_NUM_DAYS = 365

bmi_probability_transition_dict = {
    "underweight": {"underweight": 0.60,
                    "healthy": 0.40,
                    "overweight": 0.0,
                    "obese": 0.0
                    },
    "healthy": {"underweight": 0.10,
                "healthy": 0.80,
                "overweight": 0.10,
                "obese": 0.00
                },
    "overweight": {"underweight": 0.00,
                   "healthy": 0.30,
                   "overweight": 0.60,
                   "obese": 0.10
                   },
    "obese": {"underweight": 0.0,
              "healthy": 0.0,
              "overweight": 0.30,
              "obese": 0.70
              }
}

# legend
legend_text = """digraph {
  rankdir=LR
  node [shape=plaintext]
  subgraph cluster_01 { 
    label = "Legend";
    key [label=<<table border="0" cellpadding="2" cellspacing="0" cellborder="0">
      <tr><td align="right" port="i1">Health improvement</td></tr>
      <tr><td align="right" port="i2">Health regression</td></tr>
      </table>>]
    key2 [label=<<table border="0" cellpadding="2" cellspacing="0" cellborder="0">
      <tr><td port="i1">&nbsp;</td></tr>
      <tr><td port="i2">&nbsp;</td></tr>
      </table>>]
    key:i1:e -> key2:i1:w [color=green]
    key:i2:e -> key2:i2:w [color=red]
  }
  }"""
