# Food data generator

**Description**
This simulation tool aims to generate synthetic data representing the interactions between a group of users and a food recommendation system during a parameterized number of days. The library also has an interface for Jupyter notebooks that facilitates its use and the variation of parameters to obtain different results.

**Table of contents:**

* Requirements
* Parameters
* Code Organization
* Documentation
* Execution
* Develop

## Requirements

The required python version is **Python version**: 3.9. And the following libraries are required to execute the generation script:

* numpy
* pandas
* faker
* openai
* json
* graphviz
* ipywidgets

## Parameters

To run the simulation several parameters can be setter trough the GUI in the main notebook or directly in the default_inputs.py file. The parameters that can be set are described bellow:

* **Age**: This parameter sets the probability distribution for different age ranges. This probability define the user age range distribution. The values in this parameter should sum up 1.
* **Gender**: This parameter define the distribution for users between the clinical genders male (M) and female (F). The values in this parameter should sum up 1.
* **BMI**: This parameter defines the probability distributions for users within the 4 Body Mass Index (BMI) conditions (underweight, healthy, overweight, obesity). The values in this parameter should sum up 1.
* **Allergies**: This parameter defines the probability distribution to assign users under different allergy conditions. The values in this parameter should sum up 1.
* **Food restrictions**: This parameter defines the probability distribution to assign users under different cultural restrictions (e.g., vegan, vegetarian, halal, kosher). The values in this parameter should sum up 1.
* **Meal probabilities**: This parameter determine the percentages of users that could take a meal or not (e.g., breakfast with 0.8 probability will be present in the 80% of the users). The values in this parameter are independent and each one can vary between 0 and 1.
* **Flexible probability**: This parameter defines the user's soft-preferences (e.g., a person could have a preference for vegan food but also can take vegetarian food also).
* **BMI transition probability**: This probabilities define the transition probability between BMI states, this probabilities define if the user change or maintain they current BMI.

## Code Organization

Code is organized in the following setup:
* In the src folder 

## Documentation

Todo

## Execution

Todo

## Develop

Todo