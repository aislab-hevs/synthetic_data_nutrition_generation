from IPython.display import display
from IPython.display import HTML
import base64
import ipywidgets as widgets
import io
from collections import OrderedDict
from typing import Any
import numpy as np
import pandas as pd
from typing import Any, Dict, List

from synthetic_data_generation.generators import (person_entity,
                                                  run_full_simulation)


def values_from_dictionary(dictionary: Dict[str, Any]):
    new_dict = {}
    for k, v in dictionary.items():
        new_dict[k] = v.value
    return new_dict


def execute_simulation(num_users, num_days, dictionaries, probability_transition_matrix, df_recipes, progress_bar=None):
    # Todo check dictionaries probabilities
    # Todo: Get values from dictionaries and send to the simulation function
    simulation_results, df_user_join, table = run_full_simulation(
        num_users=num_users,
        gender_probabilities=values_from_dictionary(
            dictionaries['gender_probabilities']),
        BMI_probabilities=values_from_dictionary(
            dictionaries['BMI_probabilities']),
        allergies_probability_dict=values_from_dictionary(
            dictionaries['allergies_probability_dict']),
        food_restriction_probability_dict=values_from_dictionary(
            dictionaries['food_restriction_probability_dict']),
        flexi_probabilities={k: values_from_dictionary(
            dictionaries['flexi_probabilities'][k]) for k in dictionaries['flexi_probabilities'].keys()},
        probability_transition_matrix=probability_transition_matrix,
        df_recipes=df_recipes,
        meals_proba=values_from_dictionary(dictionaries['meals_proba']),
        progress_bar=progress_bar,
        num_days=num_days
    )
    return simulation_results, df_user_join, table


class FloatProgressBar:
    def __init__(self, initial_value: float = 0.0,
                 min_value=0.0,
                 max_values=100.0,
                 step=1.0,
                 description='generating...',
                 bar_style='info',
                 orientation='horizontal') -> None:
        self.min_val = min_value
        self.max_val = max_values
        self.description = description
        self.progress_bar = widgets.FloatProgress(
            value=initial_value,
            min=min_value,
            max=max_values,
            step=step,
            description=description,
            bar_style=bar_style,
            orientation=orientation
        )

    def update(self, value: float, bar_status: str = 'info'):
        if value <= self.max_val and value >= self.min_val:
            self.progress_bar.value += value
            self.progress_bar.bar_style = bar_status
            self._respond_calculus()

    def display(self):
        display(self.progress_bar)

    def reset_progress_bar(self):
        self.progress_bar.value = 0.0
        self.progress_bar.description = self.description
        self.progress_bar.bar_style = 'info'

    def hide(self):
        self.progress_bar.layout.display = "none"

    def _respond_calculus(self, default_description='success'):
        value = self.progress_bar.value
        if value >= self.max_val:
            self.progress_bar.bar_style = default_description
            self.progress_bar.description = "success"


def check_sum_proba(dict_proba):
    total_probability = sum([v.value for v in dict_proba.values()])
    if total_probability == 1.0:
        return True
    else:
        return False


class DictValidator:
    def __init__(self, dict_proba, description="Validity status:"):
        self.dict_proba = dict_proba
        style = {'description_width': 'initial'}
        self.valid_widget = widgets.Valid(
            value=True,
            description=description,
            style=style
        )
        self.total_text = "Total probability: {total}"
        self.label = widgets.Label()

    def sum_values(self):
        total_probability = sum([v.value for v in self.dict_proba.values()])
        return total_probability

    def check_sum_proba(self):
        total_probability = self.sum_values()
        if total_probability == 1.0:
            return True
        else:
            return False

    def get_validator_widget(self):
        self.label.value = f"Total probability: {self.sum_values()}"
        box = widgets.VBox([self.label, self.valid_widget])
        return box

    def validator_event(self, change):
        valid_value = self.check_sum_proba()
        self.valid_widget.value = valid_value
        self.label.value = f"Total probability: {self.sum_values()}"


def form_probability_dict(proba_dict, widget_class, exclude_validator=False, **kwargs):
    # layout = widgets.Layout(width='auto', height='40px')
    style = {'description_width': 'initial'}
    validator = DictValidator(proba_dict)
    for k, v in proba_dict.items():
        if kwargs is not None:
            proba_dict[k] = widget_class(
                value=v,
                description=k.replace("_", " "),
                disable=False,
                style=style,
                layout=widgets.Layout(
                    display="flex", justify_content="flex-start"),
                **kwargs
            )
        else:
            proba_dict[k] = widget_class(
                value=v,
                description=k.replace("_", " "),
                disable=False,
                style=style,
            )
        # Add observer
        if not exclude_validator:
            proba_dict[k].observe(validator.validator_event)
    if len(proba_dict.keys()) > 3:
        vbox = widgets.VBox(list(proba_dict.values()) +
                            [validator.get_validator_widget()])
        return vbox
    return widgets.VBox([widgets.Box(list(proba_dict.values())), validator.get_validator_widget()])


class DownloadButton:
    def __init__(self, resource, filename, extension, description=''):
        self.resource = resource
        self.payload = None
        self.filename = filename
        self.extension = extension
        self.description = description
        self.html_buttons = '''<html>
                                <head>
                                <meta name="viewport" content="width=device-width, initial-scale=1">
                                </head>
                                <body>
                                <a download="{filename}" href="data:text/{ext};base64,{payload}" download>
                                <button class="p-Widget jupyter-widgets jupyter-button widget-button mod-warning">Download {description}</button>
                                </a>
                                </body>
                                </html>
                            '''

    def _transform_to_object(self):
        # FILE
        b64 = base64.b64encode(self.resource.encode())
        self.payload = b64.decode()

    def get_html_button(self):
        self._transform_to_object()
        html_button = self.html_buttons.format(payload=self.payload,
                                               filename=self.filename,
                                               ext=self.extension,
                                               description=self.description)
        return html_button


class NotebookUIBuilder:
    def __init__(self, probability_dictionary: OrderedDict) -> None:
        self.proba_dict = probability_dictionary
        self.main_accordion = None

    def build_ui(self):
        widgets_list = []
        widgets_names = []
        for k in self.proba_dict.keys():
            widgets_list.append(self.proba_dict[k]["widget_list"])
            widgets_names.append(self.proba_dict[k]["titles"])
        self.main_accordion = widgets.Accordion(
            children=widgets_list,
            titles=widgets_names
        )

    def display(self):
        if self.main_accordion is None:
            self.build_ui()
        display(self.main_accordion)

    def get_widget(self):
        if self.main_accordion is None:
            self.build_ui()
        return self.main_accordion


def process_simulation_results(simulation_results_dict):
    list_dataframes = []
    for k in simulation_results_dict.keys():
        temp_df = simulation_results_dict.get(k)
        if temp_df is not None:
            temp_df["userId"] = k
            list_dataframes.append(temp_df)
    # concatenate dataframes
    final_df = pd.concat(list_dataframes, axis=0)
    return final_df


class ExecuteButton:
    def __init__(self, progress_bar: FloatProgressBar, num_users, num_days, dictionaries) -> None:
        self.progress_bar = progress_bar
        self.num_users = num_users
        self.num_days = num_days
        self.dictionaries = dictionaries
        pass

    def execute_simulation(self):
        try:
            print("simulation starting")
            # self.progress_bar.hide()
            self.progress_bar.reset_progress_bar()
            self.progress_bar.display()
            # execute simulation
            probability_transition_matrix = np.array([[0.65, 0.35, 0.0, 0.0],
                                                      [0.05, 0.80, 0.15, 0.0],
                                                      [0.0, 0.28, 0.67, 0.05],
                                                      [0.0, 0.0, 0.35, 0.65]
                                                      ])
            # validate before execute
            if self.num_users.value < 30:
                raise Exception(
                    "The minimum number of users to simulate is 30.")
            if self.num_days.value < 30:
                raise Exception("The minimum number of days should be over 30")
            if not check_sum_proba(self.dictionaries['gender_probabilities']):
                raise ("Gender probabilities should sum up 1.0")
            # load recipes data
            df_recipes = pd.read_csv("processed_recipes_dataset.csv", sep="|")
            simulation_results, df_user_join, table = execute_simulation(num_users=self.num_users.value,
                                                                         dictionaries=self.dictionaries,
                                                                         probability_transition_matrix=probability_transition_matrix,
                                                                         df_recipes=df_recipes,
                                                                         progress_bar=self.progress_bar,
                                                                         num_days=self.num_days.value)
            self.simulation_results = simulation_results
            self.df_user_join = df_user_join
            self.table = table
            # Show download buttons
            df_tracking = process_simulation_results(
                simulation_results_dict=simulation_results)
            csv_buffer = df_user_join.to_csv()
            tracking_csv = df_tracking.to_csv()
            button_1 = DownloadButton(
                resource=table.render(),
                filename="summary.html",
                extension="html",
                description="Summary Table"
            )

            button_2 = DownloadButton(
                resource=csv_buffer,
                filename="user_data.csv",
                extension="csv",
                description="User's data"
            )

            button_3 = DownloadButton(
                resource=tracking_csv,
                filename="tracking.csv",
                extension="csv",
                description="User's tracking data"
            )

            display(HTML(button_1.get_html_button()), HTML(
                button_2.get_html_button()), HTML(button_3.get_html_button()))
        except Exception as e:
            out = widgets.Output(layout={'border': '1px solid red'})
            with out:
                print(f"Error processing inputs: {e}")
            display(out)

    def is_finished(self):
        pass

    def generate_buttons(self):
        pass

    def button_callback(self, b):
        self.execute_simulation()


def build_full_ui():
    # Dictionary initialization
    # User number
    NUM_USERS = 100
    # Generate age range
    age_range = person_entity.get("age_range")
    age_probabilities = dict(
        zip(age_range, [1/len(age_range) for i in range(len(age_range))]))
    # Male and female distribution
    gender_probabilities = dict(
        zip(person_entity.get("clinical_gender"), [0.5, 0.5]))
    # Generate BMI values
    BMI_values = ["underweight", "healthy", "overweight", "obesity"]
    BMI_prob = [0.1, 0.3, 0.3, 0.3]
    BMI_probabilities = dict(zip(BMI_values, BMI_prob))
    # Allergy array and probabilities
    allergies = ["cow's milk", "eggs", "peanut", "soy",
                 "fish", "tree nuts", "shellfish", "wheat", "None"]
    allergies_prob = [0.1, 0.1, 0.1,
                      0.1, 0.1, 0.1, 0.1, 0.1, 0.2]
    allergies_probability_dict = dict(zip(allergies, allergies_prob))
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

    # Food restrictions probabilities
    food_restrictions = ["vegan_observant", "vegetarian_observant",
                         "halal_observant", "kosher_observant", "flexi_observant", "None"]
    food_restriction_probs = [0.2, 0.3, 0.05, 0.05, 0.1, 0.3]
    food_restriction_probability_dict = dict(
        zip(food_restrictions, food_restriction_probs))
    # meals probabilities
    meals_proba = {
        "breakfast": 0.80,
        "morning snacks": 0.45,
        "afternoon snacks": 0.40,
        "lunch": 0.95,
        "dinner": 0.85
    }
    # UI building
    # Starting
    # Prepare dictionaries
    dict_widgets = OrderedDict()
    dict_widgets['age'] = {"widget_list": form_probability_dict(age_probabilities, widgets.FloatSlider, min=0, max=1.0),
                           "titles": "Age"}
    dict_widgets['gender'] = {"widget_list": form_probability_dict(gender_probabilities, widgets.FloatSlider, min=0, max=1.0),
                              "titles": "Gender"}
    dict_widgets['bmi'] = {"widget_list": form_probability_dict(BMI_probabilities, widgets.FloatSlider, min=0, max=1.0),
                           "titles": "BMI"}
    dict_widgets['allergies'] = {"widget_list": form_probability_dict(allergies_probability_dict, widgets.FloatSlider, min=0, max=1.0),
                                 "titles": "Allergies"}
    dict_widgets['food_restrictions'] = {"widget_list": form_probability_dict(food_restriction_probability_dict, widgets.FloatSlider, min=0, max=1.0),
                                         "titles": "Food restrictions"}
    dict_widgets['meal_probabilities'] = {"widget_list": form_probability_dict(meals_proba, widgets.FloatSlider, exclude_validator=True, min=0, max=1.0),
                                          "titles": "Meal probabilities"}
    dict_widgets['flexible_probabilities'] = {"widget_list": widgets.Accordion(children=[form_probability_dict(flexi_probabilities[k], widgets.FloatSlider, min=0, max=1.0) for k in flexi_probabilities.keys()],
                                                                               titles=[k.replace("_", " ") for k in flexi_probabilities.keys()]),
                                              "titles": "Flexible probability"}
    # UI displaying
    style = {'description_width': 'initial'}
    NUM_USERS = widgets.IntText(500, description="Total users:")
    NUM_DAYS = widgets.IntText(
        365, description="Days to generate:", style=style)
    top_box = widgets.Box([NUM_USERS, NUM_DAYS], style=style)
    main_widget = NotebookUIBuilder(dict_widgets)
    # Create button to execute the simulation
    execution_button = widgets.Button(description="Start Generation",
                                      icon="play",
                                      tooltip="Click to start the data generation")

    p_bar = FloatProgressBar()
    # parameters
    # Test the function general
    general_dict = {
        'gender_probabilities': gender_probabilities,
        'BMI_probabilities': BMI_probabilities,
        'allergies_probability_dict': allergies_probability_dict,
        'food_restriction_probability_dict': food_restriction_probability_dict,
        'flexi_probabilities': flexi_probabilities,
        'meals_proba': meals_proba
    }
    button_control = ExecuteButton(p_bar,
                                   num_users=NUM_USERS,
                                   num_days=NUM_DAYS,
                                   dictionaries=general_dict)
    execution_button.on_click(button_control.button_callback)
    display(top_box)
    main_widget.display()
    display(execution_button)
