from collections import Counter
from textwrap import dedent

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, State, Output
from ..fhmm import fhmm
import numpy as np

from ..app import app


def get_layout(**kwargs):
    initial_text = kwargs.get("text", "Type some text into me!")

    # Note that if you need to access multiple values of an argument, you can
    # use args.getlist("param")
    return html.Div(
        [
            dcc.Markdown(
                dedent(
                    """
                    # Model Creator

                    This demo counts the number of characters in the text box and
                    updates a bar chart with their frequency as you type.
                    """
                )
            ),
            html.Button(id='create_model_button', n_clicks=0, children='Submit'),
            dbc.FormGroup(
                dbc.Textarea(
                    id="text-input",
                    value=initial_text,
                    style={"width": "40em", "height": "5em"},
                )
            ),
            dbc.FormGroup(
                [
                    dbc.Label("Sort by:"),
                    dbc.RadioItems(
                        id="sort-type",
                        options=[
                            {"label": "Frequency", "value": "frequency"},
                            {"label": "Character code", "value": "code"},
                        ],
                        value="frequency",
                    ),
                ]
            ),
            dbc.FormGroup(
                [
                    dbc.Label("Normalize character case?"),
                    dbc.RadioItems(
                        id="normalize",
                        options=[
                            {"label": "No", "value": "no"},
                            {"label": "Yes", "value": "yes"},
                        ],
                        value="no",
                    ),
                ]
            ),
            dcc.Graph(id="graph"),
        ]
    )


@app.callback(
    Output("graph", "figure"),
    [
        Input("text-input", "value"),
        Input("sort-type", "value"),
        Input("normalize", "value"),
    ],
    [],  # States
)
def callback(text, sort_type, normalize):
    if normalize == "yes":
        text = text.lower()

    if sort_type == "frequency":
        sort_func = lambda x: -x[1]
    else:
        sort_func = lambda x: ord(x[0])

    counts = Counter(text)

    if len(counts) == 0:
        x_data = []
        y_data = []
    else:
        x_data, y_data = zip(*sorted(counts.items(), key=sort_func))
    return {
        "data": [{"x": x_data, "y": y_data, "type": "bar", "name": "trace1"}],
        "layout": {
            "title": "Frequency of Characters",
            "height": "600",
            "font": {"size": 16},
        },
    }

@app.callback(
    Output('text-input', 'value'),
        [
            Input('create_model_button', 'n_clicks')

        ],
        [], #states
    )
def update_output(n_clicks):

    model = fhmm.FHMM(100,4)
    input_file = '../Data/PremierAutomation/Sample_Data_Short_100ms_8min_FixedHeaders.csv'
    data = np.genfromtxt(input_file, dtype=float, delimiter=',', names=True)
    X = np.array([data['Current_FB_Amps'],data['Armature_Firing_Angle_Deg_Angle_that_Voltage_waveform_fired_upon_for_pulses']]).transpose()
    X = X[:-(np.mod(X.shape[0],100)),:]
    LL, meanLL, stdLL = fhmm.train(X)
    return u'''
        The Button has been pressed {} times LL= {}
    '''.format(n_clicks, LL)
