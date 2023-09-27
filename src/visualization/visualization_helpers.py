from IPython.display import display, clear_output
from IPython.display import HTML
import base64
import ipywidgets as widgets
from collections import OrderedDict
from typing import Any
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple
import copy
import graphviz as graphv

from synthetic_data_generation.generators import (person_entity,
                                                  BMI_constants,
                                                  run_full_simulation)

import synthetic_data_generation.default_inputs as defaultValues
# import warnings
# warnings.filterwarnings("ignore")


def render_graph_from_text(dot_text: str):
    """Render a graph based in dot text. 

    :param dot_text: dot text to be render. 
    :type dot_text: str
    :return: Graph constructed from dot text. 
    :rtype: graphv.Graph
    """
    return graphv.Source(dot_text)


def render_transition_graph(states: List[str], probability_matrix: np.array) -> graphv.Graph:
    """Render a transition graph given the list of states and the probability transition matrix. 

    :param states: List of transition states. 
    :type states: List[str]
    :param probability_matrix: Probability transition matrix between states. 
    :type probability_matrix: np.array
    :return: Transition graph between states with probability given by probability_matrix
    :rtype: graphv.Graph
    """
    dot = graphv.Digraph()
    for state in states:
        dot.node(name=state, label=state)
    for i in range(probability_matrix.shape[0]):
        for j in range(probability_matrix.shape[1]):
            if (i == 0 and j == 1) or (i == 1 and i == j) or (i > 1 and i > j):
                dot.edge(states[i], states[j],
                         f"p={probability_matrix[i][j]}", color="green")
            else:
                dot.edge(states[i], states[j],
                         f"p={probability_matrix[i][j]}", color="red")
    return dot


def values_from_dictionary(dictionary: Dict[str, Any], round_digits: int = 1) -> Dict[str, float]:
    """Extract numerical values from widget dictionaries.
    :param dictionary: Dictionary that contains the key and a widget with numeric attributes.
    :type dictionary: Dict[str, Any]
    :param round_digits: round decimal to positions, defaults to 1
    :type round_digits: int, optional
    :return: Value dictionary.
    :rtype: Dict[str, float]
    """
    new_dict = {}
    for k, v in dictionary.items():
        new_dict[k] = np.round(v.value, round_digits)
    return new_dict


class FloatProgressBar:
    """Float progress bar shows the progress in a process.
    """

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


def execute_simulation(num_users: int,
                       num_days: int,
                       dictionaries: Dict[str, Any],
                       probability_transition_matrix: np.array,
                       df_recipes: pd.DataFrame,
                       progress_bar: FloatProgressBar = None) -> Tuple[Any, Any, Any]:
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


def check_sum_proba(dict_proba, round_digits=1):
    total_probability = sum([np.round(v.value, round_digits)
                            for v in dict_proba.values()])
    # print(f"total proba dict: {total_probability}")
    if np.round(total_probability, round_digits) == 1.0:
        return True
    else:
        return False


class DictValidator:
    def __init__(self, dict_proba, description="Validity status:", round_digits: int = 1):
        self.dict_proba = dict_proba
        self.round_digits = round_digits
        style = {'description_width': 'initial'}
        self.valid_widget = widgets.Valid(
            value=True,
            description=description,
            style=style
        )
        self.label = widgets.Label()

    def sum_values(self):
        total_probability = np.round(sum([np.round(v.value, self.round_digits)
                                          for v in self.dict_proba.values()]), self.round_digits)
        return total_probability

    def check_sum_proba(self):
        total_probability = self.sum_values()
        if total_probability == 1.0:
            return True
        else:
            return False

    def get_validator_widget(self):
        # self.validator_event({})
        self.label.value = f"Total probability: {self.sum_values()}"
        # self.label.value = f"Total probability: {self.sum_values()}"
        box = widgets.VBox([self.label, self.valid_widget])
        return box

    def validator_event(self, change):
        # print("validator event")
        # update dictionary values according to round digits
        for k in self.dict_proba.keys():
            self.dict_proba[k].value = np.round(
                self.dict_proba[k].value, self.round_digits)
        valid_value = self.check_sum_proba()
        self.valid_widget.value = valid_value
        self.label.value = f"Total probability: {self.sum_values()}"


def form_probability_dict(proba_dict, widget_class, exclude_validator=False, **kwargs):
    # layout = widgets.Layout(width='auto', height='40px')
    style = {'description_width': 'initial'}
    list_widgets = []
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
    # create visualization
    if not exclude_validator:
        list_widgets = list(proba_dict.values()) + \
            [validator.get_validator_widget()]
    else:
        label = widgets.Label(
            "The probabilities are independent and can sum more than 1.")
        list_widgets = list(proba_dict.values()) + [label]
    if len(proba_dict.keys()) > 3:
        vbox = widgets.VBox(list_widgets)
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
                                <meta name="viewport" content="width=device-width, initial-scale=1.2">
                                </head>
                                <body>
                                <a download="{filename}" href="data:text/{ext};base64,{payload}" download>
                                <button class="p-Widget jupyter-widgets jupyter-button widget-button mod-warning" style="width: initial;">Download {description}</button>
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
    def __init__(self, progress_bar: FloatProgressBar,
                 num_users, num_days, dictionaries,
                 out: widgets.Output = None,
                 round_digits=2) -> None:
        self.round_digits = round_digits
        self.progress_bar = progress_bar
        self.num_users = num_users
        self.num_days = num_days
        self.dictionaries = dictionaries
        self.out = out

    def execute_simulation(self):
        try:
            # self.progress_bar.hide()
            self.progress_bar.reset_progress_bar()
            if self.out is not None:
                with self.out:
                    clear_output()
            # execute simulation
            # validate before execute
            if self.num_users.value < 1:
                raise Exception(
                    f"The minimum number of users to simulate is 1. Current value: {self.num_users.value}")
            if self.num_days.value < 1:
                raise Exception(
                    f"The minimum number of days should be over 1. Current value: {self.num_days.value}")
            if not check_sum_proba(self.dictionaries['gender_probabilities']):
                raise Exception(f"Gender probabilities should sum up 1.0. \
                                Current value: {check_sum_proba(self.dictionaries['gender_probabilities'])}")
            if not check_sum_proba(self.dictionaries['age']):
                raise Exception("Age probabilities should sum up 1.0."
                                f"Current value: {[v.value for v in self.dictionaries['age'].values()]}")
            if not check_sum_proba(self.dictionaries['BMI_probabilities']):
                raise Exception("BMI probabilities should sum up 1.0")
            if not check_sum_proba(self.dictionaries['allergies_probability_dict']):
                raise Exception("Allergies probabilities should sum up 1.0")
            if not check_sum_proba(self.dictionaries['food_restriction_probability_dict']):
                raise Exception(
                    "Food restrictions probabilities should sum up 1.0")
            for k in self.dictionaries['flexi_probabilities'].keys():
                if not check_sum_proba(self.dictionaries['flexi_probabilities'][k]):
                    raise Exception(
                        f"{k.replace('_', ' ')} probabilities should sum up 1.0")
            # check probabilities for BMI transition
            for k in self.dictionaries["bmi_transition_proba"].keys():
                if not check_sum_proba(self.dictionaries["bmi_transition_proba"][k]):
                    raise Exception(
                        f"{k.replace('_', ' ')} probabilities should sum up 1.0"
                    )
            # Process probability values
            probability_transition_matrix = np.array([[self.dictionaries["bmi_transition_proba"]["underweight"]["underweight"].value,
                                                       self.dictionaries["bmi_transition_proba"]["underweight"]["healthy"].value,
                                                       self.dictionaries["bmi_transition_proba"]["underweight"]["overweight"].value,
                                                       self.dictionaries["bmi_transition_proba"]["underweight"]["obese"].value,
                                                       ],
                                                      [self.dictionaries["bmi_transition_proba"]["healthy"]["underweight"].value,
                                                       self.dictionaries["bmi_transition_proba"]["healthy"]["healthy"].value,
                                                       self.dictionaries["bmi_transition_proba"]["healthy"]["overweight"].value,
                                                       self.dictionaries["bmi_transition_proba"]["healthy"]["obese"].value,
                                                       ],
                                                      [self.dictionaries["bmi_transition_proba"]["overweight"]["underweight"].value,
                                                       self.dictionaries["bmi_transition_proba"]["overweight"]["healthy"].value,
                                                       self.dictionaries["bmi_transition_proba"]["overweight"]["overweight"].value,
                                                       self.dictionaries["bmi_transition_proba"]["overweight"]["obese"].value,
                                                       ],
                                                      [self.dictionaries["bmi_transition_proba"]["obese"]["underweight"].value,
                                                       self.dictionaries["bmi_transition_proba"]["obese"]["healthy"].value,
                                                       self.dictionaries["bmi_transition_proba"]["obese"]["overweight"].value,
                                                       self.dictionaries["bmi_transition_proba"]["obese"]["obese"].value,
                                                       ]
                                                      ])
            # show transition graph
            if self.out is not None:
                with self.out:
                    print("simulation starting")
                    self.progress_bar.display()
                    transition_graph = render_transition_graph([BMI_constants.underweight.value,
                                                                BMI_constants.healthy.value,
                                                                BMI_constants.overweight.value,
                                                                BMI_constants.obesity.value],
                                                               probability_matrix=probability_transition_matrix)
                    legend_text = render_graph_from_text(
                        defaultValues.legend_text)
                    box_style = {"align-content": "center"}
                    box = widgets.VBox(children=[widgets.Image(value=transition_graph.pipe(format="png"), format="png"),
                                                 widgets.Image(value=legend_text.pipe(format="png"), format="png")],
                                       style=box_style
                                       )
                    display(box)
                    # display(render_transition_graph([BMI_constants.underweight.value,
                    #                                  BMI_constants.healthy.value,
                    #                                  BMI_constants.overweight.value,
                    #                                  BMI_constants.obesity.value],
                    #                                 probability_matrix=probability_transition_matrix))
                    # display(render_graph_from_text(defaultValues.legend_text))
            else:
                print("simulation starting")
                self.progress_bar.display()
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
            if self.out is not None:
                with self.out:
                    display(HTML(button_1.get_html_button()), HTML(
                        button_2.get_html_button()), HTML(button_3.get_html_button()))
            else:
                display(HTML(button_1.get_html_button()), HTML(
                        button_2.get_html_button()), HTML(button_3.get_html_button()))
        except Exception as e:
            exceptionOut = widgets.Output(layout={'border': '1px solid red'})
            if self.out is not None:
                with self.out:
                    with exceptionOut:
                        clear_output()
                        print(f"Error processing inputs: {e}")
                    display(exceptionOut)
            else:
                with exceptionOut:
                    print(f"Error processing inputs: {e}")
                display(exceptionOut)

    def button_callback(self, b):
        self.execute_simulation()


def build_full_ui():
    # Get initializers
    age_probabilities = copy.deepcopy(defaultValues.age_probabilities_dict)
    gender_probabilities = copy.deepcopy(
        defaultValues.gender_probabilities_dict)
    BMI_probabilities = copy.deepcopy(defaultValues.BMI_probabilities_dict)
    allergies_probability = copy.deepcopy(
        defaultValues.allergies_probability_dict)
    food_restriction_probability = copy.deepcopy(
        defaultValues.food_restriction_probability_dict)
    flexi_probabilities = copy.deepcopy(defaultValues.flexi_probabilities_dict)
    meals_proba = copy.deepcopy(defaultValues.meals_proba_dict)
    bmi_transition_probabilities = copy.deepcopy(
        defaultValues.bmi_probability_transition_dict)
    # UI building
    # Starting
    # Prepare dictionaries
    dict_widgets = OrderedDict()
    dict_widgets['age'] = {"widget_list":
                           form_probability_dict(
                               age_probabilities, widgets.SelectionSlider, options=np.round(np.arange(0.0, 1.1, 0.1), 1).tolist()),

                           "titles": "Age"}
    dict_widgets['gender'] = {"widget_list":
                              form_probability_dict(
                                  gender_probabilities, widgets.FloatSlider, min=0, max=1.0, step=0.1),
                              "titles": "Gender"}
    dict_widgets['bmi'] = {"widget_list":
                           form_probability_dict(
                               BMI_probabilities, widgets.FloatSlider, min=0, max=1.0, step=0.1),
                           "titles": "BMI"}
    dict_widgets['allergies'] = {"widget_list":
                                 form_probability_dict(
                                     allergies_probability, widgets.FloatSlider, min=0, max=1.0, step=0.1),
                                 "titles": "Allergies"}
    dict_widgets['food_restrictions'] = {"widget_list":
                                         form_probability_dict(
                                             food_restriction_probability, widgets.FloatSlider, min=0, max=1.0, step=0.1),
                                         "titles": "Food restrictions"}
    dict_widgets['meal_probabilities'] = {"widget_list":
                                          form_probability_dict(
                                              meals_proba, widgets.FloatSlider, exclude_validator=True, min=0, max=1.0, step=0.1),
                                          "titles": "Meal probabilities"}
    dict_widgets['flexible_probabilities'] = {"widget_list":
                                              widgets.Accordion(children=[form_probability_dict(flexi_probabilities[k],
                                                                                                widgets.FloatSlider,
                                                                                                min=0,
                                                                                                max=1.0,
                                                                                                step=0.1) for k in flexi_probabilities.keys()],
                                                                titles=[k.replace("_", " ") for k in flexi_probabilities.keys()]),
                                              "titles": "Flexible probability"}
    dict_widgets['transition_probabilities'] = {"widget_list": widgets.Accordion(children=[form_probability_dict(bmi_transition_probabilities[k],
                                                                                                                 widgets.FloatSlider,
                                                                                                                 min=0,
                                                                                                                 max=1.0,
                                                                                                                 step=0.1) for k in bmi_transition_probabilities.keys()],
                                                                                 titles=[k.replace("_", " ") for k in bmi_transition_probabilities.keys()]),
                                                "titles": "BMI transition probability"
                                                }
    # UI displaying
    style = {'description_width': 'initial'}
    NUM_USERS = widgets.IntText(
        defaultValues.DEFAULT_NUM_USERS, description="Total users:")
    NUM_DAYS = widgets.IntText(
        defaultValues.DEFAULT_NUM_DAYS, description="Days to generate:", style=style)
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
        'age': age_probabilities,
        'gender_probabilities': gender_probabilities,
        'BMI_probabilities': BMI_probabilities,
        'allergies_probability_dict': allergies_probability,
        'food_restriction_probability_dict': food_restriction_probability,
        'flexi_probabilities': flexi_probabilities,
        'meals_proba': meals_proba,
        "bmi_transition_proba": bmi_transition_probabilities
    }

    out = widgets.Output()
    button_control = ExecuteButton(p_bar,
                                   num_users=NUM_USERS,
                                   num_days=NUM_DAYS,
                                   dictionaries=general_dict,
                                   out=out)

    execution_button.on_click(button_control.button_callback)
    display(top_box)
    main_widget.display()
    display(execution_button)
    display(out)
