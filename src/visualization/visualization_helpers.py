from IPython.display import display
from IPython.display import HTML
import base64
import ipywidgets as widgets
import io
from collections import OrderedDict
from typing import Any

from synthetic_data_generation.generators import (person_entity)


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
            self.progress_bar.value = value
            self.progress_bar.bar_style = bar_status
            self._respond_calculus()

    def display(self):
        display(self.progress_bar)

    def hide(self):
        self.progress_bar.layout.display = "none"

    def _respond_calculus(self, default_description='success'):
        value = self.progress_bar.value
        if value >= self.max_val:
            self.progress_bar.bar_style = default_description


class DictValidator:
    def __init__(self, dict_proba, description="Validity status:"):
        self.dict_proba = dict_proba
        style = {'description_width': 'initial'}
        self.valid_widget = widgets.Valid(
            value=True,
            description=description,
            style=style
        )

    def check_sum_proba(self):
        total_probability = sum([v.value for v in self.dict_proba.values()])
        if total_probability == 1.0:
            return True
        else:
            return False

    def get_validator_widget(self):
        return self.valid_widget

    def validator_event(self, change):
        valid_value = self.check_sum_proba()
        self.valid_widget.value = valid_value


def form_probability_dict(proba_dict, widget_class, **kwargs):
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


class ExecuteButton:
    def __init__(self, progress_bar: FloatProgressBar) -> None:
        self.progress_bar = progress_bar
        pass

    def execute_simulation(self):
        self.progress_bar.display()
        pass

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
    allergies_prob = [0.075, 0.075, 0.075,
                      0.075, 0.075, 0.075, 0.075, 0.075, 0.4]
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
    dict_widgets['meal_probabilities'] = {"widget_list": form_probability_dict(meals_proba, widgets.FloatSlider, min=0, max=1.0),
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
    button_control = ExecuteButton(p_bar)
    execution_button.on_click(button_control.button_callback)
    display(top_box)
    main_widget.display()
    display(execution_button)
    # create a progress bar for the simulation
    # execute the simulation
    # habilitate the buttons to download
