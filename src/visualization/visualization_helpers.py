from IPython.display import display, clear_output
from IPython.display import HTML
import base64
import codecs
import ipywidgets as widgets
from collections import OrderedDict
from typing import Any
import os
import datetime as dt
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple
import copy
import graphviz as graphv
import traceback
from functools import partial
import io

from synthetic_data_generation.generators import (person_entity,
                                                  BMI_constants,
                                                  save_outputs,
                                                  run_full_simulation)
from synthetic_data_generation.html_utilities import HTML_Table

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
                 orientation='horizontal',
                 tail_text: str = '') -> None:
        self.min_val = min_value
        self.max_val = max_values
        self.description = description
        self.tail_text = tail_text
        self.progress_bar = widgets.FloatProgress(
            value=initial_value,
            min=min_value,
            max=max_values,
            step=step,
            description=description,
            bar_style=bar_style,
            orientation=orientation
        )
        self.tail_label = widgets.Label(self.tail_text)
        self.progress_widget = widgets.Box(
            [
            self.progress_bar,
            self.tail_label
            ]
        )
        
    def update_tail_text(self, new_text: str):
        self.tail_text = new_text
        self.tail_label.value = new_text

    def update(self, value: float, bar_status: str = 'info'):
        if value <= self.max_val and value >= self.min_val:
            self.progress_bar.value += value
            self.progress_bar.bar_style = bar_status
            self._respond_calculus()

    def display(self):
        display(self.progress_widget)

    def get_progress_bar(self) -> widgets.FloatProgress:
        return self.progress_bar

    def reset_progress_bar(self):
        self.progress_bar.value = 0.0
        self.progress_bar.description = self.description
        self.progress_bar.bar_style = 'info'
        self.tail_text = ''

    def hide(self):
        self.progress_widget.layout.display = "none"

    def _respond_calculus(self, default_description='success'):
        value = self.progress_bar.value
        if value >= self.max_val:
            self.progress_bar.bar_style = default_description
            self.progress_bar.description = "success"


def execute_simulation(num_users: int,
                       num_days: int,
                       chose_dist: str,
                       dictionaries: Dict[str, Any],
                       probability_transition_matrix: np.array,
                       df_recipes: pd.DataFrame,
                       progress_bar: FloatProgressBar = None,
                       num_simultaneous_allergies: int = 2) -> Tuple[Any, Any, Any]:
    # Execute simulation 
    df_user_join, table, new_tracking, simulation_parameters = run_full_simulation(
        num_users=num_users,
        chose_dist=chose_dist,
        delta_dist_dict=values_from_dictionary(
            dictionaries['delta_probabilities'][chose_dist]),
        place_probabilities=values_from_dictionary(
            dictionaries["places_meal"]
        ),
        social_situation_probabilities=values_from_dictionary(
            dictionaries["social_situation_meal"]
        ),
        age_probabilities=values_from_dictionary(
            dictionaries['age']),
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
        meals_time_dict={
            k: values_from_dictionary(dictionaries["meal_time"][k]) for k in dictionaries["meal_time"].keys()
        },
        probability_transition_matrix=probability_transition_matrix,
        df_recipes=df_recipes,
        meals_proba=values_from_dictionary(dictionaries['meals_proba']),
        progress_bar=progress_bar,
        num_days=num_days,
        multiple_allergies_number=num_simultaneous_allergies
    )
    return df_user_join, table, new_tracking, simulation_parameters


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


def form_probability_dict(proba_dict, widget_class, exclude_validator=False, round_digits=1, **kwargs):
    # layout = widgets.Layout(width='auto', height='40px')
    style = {'description_width': 'initial'}
    validator = DictValidator(proba_dict, round_digits=round_digits)
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
    list_widgets = []
    if not exclude_validator:
        if len(proba_dict.keys()) > 2:
            list_widgets = widgets.VBox(
                list(proba_dict.values()) +
                [validator.get_validator_widget()]
            )
        else:
            list_widgets = widgets.VBox(
                [
                    widgets.Box(list(proba_dict.values())),
                    validator.get_validator_widget()
                ]
            )
        # list_widgets = list(proba_dict.values()) + \
        #     [validator.get_validator_widget()]
    else:
        # print(f"I should be here not validator {exclude_validator}")
        label = widgets.Label(
            "The probabilities are independent and can sum more than 1.")
        if len(proba_dict.keys()) > 2:
            list_widgets = widgets.VBox(
                list(proba_dict.values()) +
                [label]
            )
        else:
            list_widgets = widgets.VBox(
                [
                    widgets.Box(list(proba_dict.values())),
                    label
                ]
            )
    #     list_widgets = list(proba_dict.values()) + [label]
    # if len(proba_dict.keys()) > 3:
    #     vbox = widgets.VBox(list_widgets)
    # else:
    #     vbox = widgets.Box(list_widgets)
    return list_widgets


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


class DistributionSelector:
    def __init__(self, distribution_options_dict: Dict[str, Any],
                 chose_dist: str,
                 widget_control: Any,
                 **kwargs) -> None:
        self.kwargs = kwargs
        self.chose_dist = chose_dist
        self.dict = distribution_options_dict
        self.widget_control = widget_control
        self.out = widgets.Output()
        self.combo = widgets.Dropdown(description="Distribution: ")
        self._fill_the_drop_down()
        self.make_visible()

    def get_output(self):
        return self.out

    def display(self):
        display(self.out)

    def make_visible(self):
        with self.out:
            display(self.combo)
            widget_dict = self.dict.get(self.combo.value)
            if widget_dict is not None:
                if len(widget_dict) > 2:
                    container = widgets.VBox(
                        [widget_dict[k] for k in widget_dict.keys()])
                else:
                    container = widgets.Box([widget_dict[k]
                                            for k in widget_dict.keys()])
                display(container)

    def _fill_the_drop_down(self):
        options = list(self.dict.keys())
        self.combo.options = options
        self.combo.value = options[0]
        self.chose_dist = self.combo.value
        self.combo.observe(self.on_option_change, names='value')
        # create the graph part
        for k in self.dict.keys():
            for sk in self.dict[k].keys():
                self.dict[k][sk] = self.widget_control(value=self.dict[k][sk],
                                                       description=sk,
                                                       **self.kwargs)

    def on_option_change(self, change):
        new_value = change.new
        self.chose_dist = self.combo.value
        self.out.clear_output()
        widget_dict = self.dict.get(new_value)
        if widget_dict is not None:
            with self.out:
                display(self.combo)
                if len(widget_dict) > 2:
                    container = widgets.VBox(
                        [widget_dict[k] for k in widget_dict.keys()])
                else:
                    container = widgets.Box([widget_dict[k]
                                            for k in widget_dict.keys()])
                display(container)
        else:
            pass

    def get_current_values(self):
        current_selection = self.combo.value
        return current_selection, self.dict


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


class UpdateDropdown:
    def __init__(self, value_dict, presets_dict) -> None:
        self.value_dict = value_dict
        self.presets_dict = presets_dict

    def dropDownChange(self, change):
        new_value = change.new
        # print(f"received value: {new_value}")
        # get preset and update the controls
        current_preset_dict = self.presets_dict.get(new_value, None)
        if current_preset_dict is not None:
            for k in current_preset_dict.keys():
                self.value_dict[k].value = current_preset_dict[k]


class ExecuteButton:
    def __init__(self, progress_bar: FloatProgressBar,
                 num_users, num_days, num_simultaneous_allergies, dictionaries,
                 delta_dist_chose: str,
                 out: widgets.Output = None,
                 round_digits=2) -> None:
        self.delta_dist_chose = delta_dist_chose
        self.num_simultaneous_allergies = num_simultaneous_allergies
        self.round_digits = round_digits
        self.progress_bar = progress_bar
        self.num_users = num_users
        self.num_days = num_days
        self.dictionaries = dictionaries
        self.out = out

    def show_table_preview(self, change: Any,  table: HTML_Table):
        # print(f"got: {change}")
        if self.out is not None:
            with self.out:
                display(HTML(table.render()))
        else:
            display(HTML(table.render()))

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
            if not check_sum_proba(self.dictionaries['allergies_probability_dict'], round_digits=2):
                raise Exception("Allergies probabilities should sum up 1.0")
            if not check_sum_proba(self.dictionaries["places_meal"]):
                raise Exception(
                    "Place of meal consumption probabilities should sum up 1.0")
            if not check_sum_proba(self.dictionaries["social_situation_meal"]):
                raise Exception(
                    "Social situation of meal consumption probabilities should sum up 1.0")
            if not check_sum_proba(self.dictionaries['food_restriction_probability_dict']):
                raise Exception(
                    "Cultural restrictions probabilities should sum up 1.0")
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
            else:
                print("simulation starting")
                self.progress_bar.display()
            # load recipes data
            #TODO: make the recipes file reading parametrize
            current_dir = os.getcwd()
            # default_path_recipes = 'recipes/recipes_sampling_1000.csv'
            default_path_recipes = 'recipes/recipes_dataset_final.csv'
            # default_path_recipes = "recipes/extended_processed_recipes_dataset_id.csv"
            df_recipes = pd.read_csv(os.path.join(current_dir, default_path_recipes),
                                     sep="|", index_col=0)
            df_user_join, table, new_tracking_df, sim_parameters = execute_simulation(num_users=self.num_users.value,
                                                                      chose_dist=self.delta_dist_chose,
                                                                      dictionaries=self.dictionaries,
                                                                      probability_transition_matrix=probability_transition_matrix,
                                                                      df_recipes=df_recipes,
                                                                      progress_bar=self.progress_bar,
                                                                      num_days=self.num_days.value,
                                                                      num_simultaneous_allergies=self.num_simultaneous_allergies.value)
            self.df_user_join = df_user_join
            self.table = table
            # save result to a temporal  directory
            base_output_path = os.path.join(os.getcwd(), "outputs")
            folder_output_name = dt.datetime.now().strftime('%d-%m-%Y_%H-%M-%S') 
            files_dict = {
                "simulation_parameters.json": sim_parameters,
                "simulation_parameters.npy": sim_parameters,
                "users_dataset.csv": df_user_join,
                "tracking.csv": new_tracking_df,
                "summary_table.html": table.render(),
                "transition_graph.png": transition_graph
            }
            save_outputs(
                base_output_path,
                folder_output_name,
                files_dict
            )
            # Show download buttons
            csv_buffer = df_user_join.to_csv()
            tracking_csv = new_tracking_df.to_csv()
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
            button_4 = widgets.Button(description="Preview table",
                                      tooltip="Click to show a preview of the table",
                                      icon='html5')

            button_4.add_class("p-Widget")
            button_4.add_class("jupyter-widgets")
            button_4.add_class("jupyter-button")
            button_4.add_class("widget-button")
            button_4.add_class("mod-warning")

            partial_table = partial(self.show_table_preview, table=table)
            button_4.on_click(partial_table)

            button_box = widgets.Box(children=[widgets.HTML(button_1.get_html_button()),
                                               widgets.HTML(
                                                   button_2.get_html_button()),
                                               widgets.HTML(
                                                   button_3.get_html_button()),
                                               button_4])

            if self.out is not None:
                with self.out:
                    display(button_box)
            else:
                display(button_box)
        except Exception as e:
            print(traceback.format_exc())
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
        
def load_parameters_from_file(button: Any, file_object: widgets.FileUpload):
    #TODO: Load parameters from uploaded file
    print(f'file_object: {file_object.value}')
    f_object = file_object.value[0]
    data = np.load(io.BytesIO(f_object.content), allow_pickle=True)
    dict_parameters = data.item()
    print(dict_parameters)
        
def process_uploaded_file(uploaded_file):
    content = io.StringIO(uploaded_file.decode('utf-8'))
    df = pd.read_csv(content, sep="|", index_col=0)
    print(df.head(3))
    return df

def on_file_change(change):
    print(change)
    uploaded_filename = next(iter(change.new))
    print(f"File uploaded: {uploaded_filename}")
    uploaded_file = change.new[uploaded_filename]['content']
    process_uploaded_file(uploaded_file)

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
    meals_time_distribution = copy.deepcopy(
        defaultValues.meal_time_distribution)
    meals_proba = copy.deepcopy(defaultValues.meals_proba_dict)
    bmi_transition_probabilities = copy.deepcopy(
        defaultValues.bmi_probability_transition_dict)
    places_dict = copy.deepcopy(defaultValues.place_proba_dict)
    social_situation_dict = copy.deepcopy(
        defaultValues.social_situation_proba_dict)
    delta_dict = copy.deepcopy(defaultValues.delta_distribution_dict)
    chose_dist = list(delta_dict.keys())[0]
    # distribute selector
    distribution_selector = DistributionSelector(delta_dict,
                                                 chose_dist=chose_dist,
                                                 widget_control=widgets.FloatSlider,
                                                 min=0,
                                                 max=1,
                                                 step=0.05)
    # UI building
    # Starting
    # Prepare dictionaries
    # add combo for presets
    age_preset_combo = widgets.Dropdown(
        options=[k for k in defaultValues.age_presets_dict.keys()],
        value="Flat",
        description="Age preset: ",
        disable=False
    )
    bmi_preset_combo = widgets.Dropdown(
        options=[k for k in defaultValues.bmi_presets_dict.keys()],
        value="Flat",
        description="BMI preset: ",
        disable=False
    )
    cultural_preset_combo = widgets.Dropdown(
        options=[k for k in defaultValues.cultural_restriction_presets_dict.keys()],
        value="Flat",
        description="Cultural factor preset: ",
        disable=False
    )
    # layout = widgets.Layout(width='auto', height='40px')
    allergies_preset_combo = widgets.Dropdown(
        options=[k for k in defaultValues.allergies_presets_dict.keys()],
        value="Europe",
        description="Allergies presets: ",
        disable=False,
        style={'description_width': 'initial'}
    )
    multiple_allergies_number = widgets.IntSlider(
        value=2,
        min=2,
        max=7,
        step=1,
        description="Multiple allergies number: ",
        style={'description_width': 'initial'}
    )
    # Created ordered dict
    dict_widgets = OrderedDict()
    dict_widgets['age'] = {"widget_list": widgets.VBox(
        [age_preset_combo,
         form_probability_dict(
             age_probabilities, widgets.SelectionSlider, options=np.round(np.arange(0.0, 1.1, 0.1), 1).tolist())
         ]),

        "titles": "Age"}
    dict_widgets['gender'] = {"widget_list":
                              form_probability_dict(
                                  gender_probabilities, widgets.FloatSlider, min=0, max=1.0, step=0.1),
                              "titles": "Gender"}
    dict_widgets['bmi'] = {"widget_list":
                           widgets.VBox([
                               bmi_preset_combo,
                               form_probability_dict(
                                   BMI_probabilities, widgets.FloatSlider, round_digits=2, min=0, max=1.0, step=0.05)
                           ]),
                           "titles": "BMI"}
    dict_widgets['allergies'] = {"widget_list": widgets.Box(
                                 [widgets.VBox(
                                     [allergies_preset_combo,
                                      form_probability_dict(allergies_probability,
                                                            widgets.FloatSlider,
                                                            round_digits=2,
                                                            min=0, max=1.0, step=0.05)
                                      ]
                                 ), multiple_allergies_number
                                 ]),
                                 "titles": "Allergies"}
    dict_widgets['food_restrictions'] = {"widget_list":
                                         widgets.VBox([
                                             cultural_preset_combo,
                                             form_probability_dict(
                                                 food_restriction_probability, widgets.FloatSlider, 
                                                 round_digits=2, 
                                                 min=0, 
                                                 max=1.0, 
                                                 step=0.05)
                                         ]),
                                         "titles": "Cultural restrictions"}
    dict_widgets['meal_probabilities'] = {"widget_list":
                                          form_probability_dict(
                                              meals_proba, widgets.FloatSlider, exclude_validator=True, 
                                              round_digits=2,
                                              min=0, 
                                              max=1.0, 
                                              step=0.05),
                                          "titles": "Meal probabilities"}
    dict_widgets['flexible_probabilities'] = {"widget_list":
                                              widgets.Accordion(children=[form_probability_dict(flexi_probabilities[k],
                                                                                                widgets.FloatSlider,
                                                                                                round_digits=2,
                                                                                                min=0,
                                                                                                max=1.0,
                                                                                                step=0.05) for k in flexi_probabilities.keys()],
                                                                titles=[k.replace("_", " ") for k in flexi_probabilities.keys()]),
                                              "titles": "Flexible probability"}
    dict_widgets['transition_probabilities'] = {"widget_list": widgets.Accordion(children=[form_probability_dict(bmi_transition_probabilities[k],
                                                                                                                 widgets.FloatSlider,
                                                                                                                 round_digits=2,
                                                                                                                 min=0,
                                                                                                                 max=1.0,
                                                                                                                 step=0.05) for k in bmi_transition_probabilities.keys()],
                                                                                 titles=[k.replace("_", " ") for k in bmi_transition_probabilities.keys()]),
                                                "titles": "BMI transition probability"
                                                }
    dict_widgets['meal_time'] = {"widget_list":
                                 widgets.Accordion(children=[form_probability_dict(proba_dict=meals_time_distribution[k],
                                                                                   widget_class=widgets.IntSlider,
                                                                                   exclude_validator=True,
                                                                                   min=0,
                                                                                   max=24,
                                                                                   step=1) for k in meals_time_distribution.keys()],
                                                   titles=[k.replace("_", " ") for k in meals_time_distribution.keys()]),
                                 "titles": "Meal time"}
    dict_widgets['places_meal'] = {
        "widget_list": form_probability_dict(places_dict,
                                             widgets.FloatSlider,
                                             round_digits=2,
                                             min=0.0,
                                             max=1.0,
                                             step=0.05),
        "titles": "Place of meal consumption probability"
    }
    dict_widgets['social_situation_meal'] = {
        "widget_list": form_probability_dict(social_situation_dict,
                                             widgets.FloatSlider,
                                             round_digits=2,
                                             min=0.0,
                                             max=1.0,
                                             step=0.05),
        "titles": "Social situation of meal consumption probability"
    }
    dict_widgets['delta_distribution'] = {"widget_list": distribution_selector.get_output(),
                                          "titles": "Appreciation feedback (delta)"}
    button_layout = widgets.Layout(width='auto', height='auto')
    recipes_file_upload = widgets.FileUpload(
        accept='.csv',
        multiple=False,
        description='Upload recipes:')
    file_parameters_upload = widgets.FileUpload(
        accept='.npy',
        multiple=False,
        description='Upload parameters file:',
        layout=button_layout
    )
    parameters_upload_button = widgets.Button(
                     description = "Set parameters",
                     style={'description_width': 'initial'},
                     layout=button_layout
                 )
    dict_widgets[''] = {
        "widget_list": widgets.VBox(
            [widgets.Label('Upload a parameters file (.npy)'),
             widgets.Box([
                file_parameters_upload,
                parameters_upload_button
             ])
             ]),
        "titles": "Upload parameter file (Optional)"
    }
    sep_char = '|'
    # dict_widgets['upload_recipes'] = {"widget_list": widgets.VBox(
    #     [
    #     widgets.Label('Upload a recipe in a csv file and specify the separator.'),
    #     widgets.Box([
    #     recipes_file_upload,
    #     widgets.Text(
    #         value=sep_char,
    #         placeholder='Introduce the csv separator',
    #         description='Separator:',
    #         disabled=False,
    #         style={'description_width': 'initial'}
    #     )
    #         ])
    #     ]),
    #     "titles": "Upload recipes (Optional)"}
    # UI displaying
    style = {'description_width': 'initial'}
    NUM_USERS = widgets.IntText(
        defaultValues.DEFAULT_NUM_USERS, description="Total users:")
    NUM_DAYS = widgets.IntText(
        defaultValues.DEFAULT_NUM_DAYS, description="Days to generate:", style=style)
    top_box = widgets.Box([NUM_USERS, NUM_DAYS], style=style)
    # Connect signals
    age_preset_event_handler = UpdateDropdown(
        value_dict=age_probabilities,
        presets_dict=defaultValues.age_presets_dict
    )
    allergies_preset_event_handler = UpdateDropdown(
        value_dict=allergies_probability,
        presets_dict=defaultValues.allergies_presets_dict
    )
    age_preset_combo.observe(age_preset_event_handler.dropDownChange,
                             names='value')
    allergies_preset_combo.observe(allergies_preset_event_handler.dropDownChange,
                                   names='value')
    recipes_file_upload.observe(on_file_change, names='value')
    # BMI preset connection
    bmi_preset_event_handler = UpdateDropdown(
        value_dict=BMI_probabilities,
        presets_dict=defaultValues.bmi_presets_dict
    )
    bmi_preset_combo.observe(bmi_preset_event_handler.dropDownChange,
                             names='value')
    # Cultural factor connection
    cultural_preset_event_handler = UpdateDropdown(
        value_dict=food_restriction_probability,
        presets_dict=defaultValues.cultural_restriction_presets_dict
    )
    cultural_preset_combo.observe(cultural_preset_event_handler.dropDownChange,
                                  names='value')
    # Create main widgets
    main_widget = NotebookUIBuilder(dict_widgets)
    # Create button to execute the simulation
    execution_button = widgets.Button(description="Start Generation",
                                      icon="play",
                                      tooltip="Click to start the data generation")
    p_bar = FloatProgressBar()
    # parameters
    # Test the function general

    general_dict = {
        'delta_probabilities': delta_dict,
        'age': age_probabilities,
        'gender_probabilities': gender_probabilities,
        'BMI_probabilities': BMI_probabilities,
        'allergies_probability_dict': allergies_probability,
        'food_restriction_probability_dict': food_restriction_probability,
        'flexi_probabilities': flexi_probabilities,
        'meals_proba': meals_proba,
        "bmi_transition_proba": bmi_transition_probabilities,
        "places_meal": places_dict,
        "social_situation_meal": social_situation_dict,
        "meal_time": meals_time_distribution
    }

    out = widgets.Output()
    button_control = ExecuteButton(p_bar,
                                   num_users=NUM_USERS,
                                   num_days=NUM_DAYS,
                                   num_simultaneous_allergies=multiple_allergies_number,
                                   delta_dist_chose=chose_dist,
                                   dictionaries=general_dict,
                                   out=out)
    partial_function_load_parameters = partial(load_parameters_from_file, 
                                               file_object=file_parameters_upload)
    parameters_upload_button.on_click(partial_function_load_parameters)
    execution_button.on_click(button_control.button_callback)
    display(top_box)
    main_widget.display()
    display(execution_button)
    display(out)
