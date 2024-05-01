import pymc as pm 
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List
import numpy as np
import pandas as pd


def generate_user_data(user_parameters: Dict):
    total_users = user_parameters.get('total_users', 0)
    age_parameters = user_parameters.get('age_parameters',None)
    gender_parameters = user_parameters.get('gender_parameters',None)
    bmi_parameters = user_parameters.get('bmi_parameters',None)
    allergies_parameters = user_parameters.get('allergies_parameters',None)
    cultural_parameters = user_parameters.get('cultural_parameters',None)
    flexible_parameters = user_parameters.get('flexible_parameters',None)
    bmi_transition_parameters = user_parameters.get('bmi_trans', None)
    model = pm.Model()
    with model:
        age_index = pm.Categorical('age_index', p=age_parameters['probabilities'])
        gender_index = pm.Categorical('gender_index', p=gender_parameters['probabilities'])
        bmi_index = pm.Categorical('bmi_index', p=bmi_parameters['probabilities'])
        allergies_index = pm.Categorical('allergies_index', p=allergies_parameters['probabilities'])
        cultural_index = pm.Categorical('cultural_parameters', p=cultural_parameters['probabilities'])
        flexible_index = pm.Categorical('flexible_parameters', p=flexible_parameters['probabilities'])
    
    # sample from the model 
    with model: 
        trace = pm.sample(draws=total_users, tune=200)
        
    # extract values from the trace 
    age_values = [age_parameters['values'][i] for i in trace['age_index']]
    gender_values = [gender_parameters['values'][i] for i in trace['gender_index']]
    bmi_values = [bmi_parameters['values'][i] for i in trace['bmi_index']]
    allergies_values = [allergies_parameters['values'][i] for i in trace['allergies_index']]
    cultural_values = [cultural_parameters['values'][i] for i in trace['cultural_parameters']]
    flexible_index = [i for i in trace['flexible_index']]
    # create a dataframe
    df_user_data = pd.DataFrame({'age': age_values,
                                  'gender': gender_values,
                                  'bmi': bmi_values,
                                  'allergies': allergies_values,
                                  'cultural_factor': cultural_values,
                                  'flexible_factor': flexible_index})
    return df_user_data
    
    
    
    
def simulate_meal_context(context_parameters: Dict):
    # Capture parameters and return the context sampled dataframe for an user. 
    pass

def simulate_feedback(feedback_parameters: Dict, df_recipes, df_user_data, df_context):
    pass