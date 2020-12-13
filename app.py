import dash
import dash_html_components as html
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import joblib
from dash.dependencies import Input, Output, State

app = dash.Dash(external_stylesheets=[dbc.themes.DARKLY])
app.title = 'Cardiovascular Disease Predictor'
# ------------------------------------------------------------------------------
# Import and clean data (importing csv into pandas)
df = pd.read_csv("cardio_train_clean_1hot_featureselection.csv")
model = joblib.load('logistic_regression_model_1hot.sav')

layout = dict(
    autosize=True,
    height=500,
    font=dict(color="#191A1A"),
    titlefont=dict(color="#191A1A", size='14'),
    margin=dict(
        l=45,
        r=15,
        b=45,
        t=35
    )
)

# Layout
app.layout = html.Div([

    # Title - Row
    dbc.Row(html.Div(
        [
            html.H1(
                'Cardiovascular Disease Predictor',
                style={'font-family': 'Helvetica',
                       "margin-left": 20,
                       "margin-bottom": 0}
            )
        ]
    )),

    # block 2
    dbc.Row(html.Div([
        html.H3('Patient'),

        dbc.Row(
            [
                html.Div(
                    [
                        html.P('Gender (M/F): '),
                        dbc.RadioItems(
                            id="gender",
                            options=[
                                {'label': 'Male', 'value': 0},
                                {'label': 'Female', 'value': 1}],
                            value=0
                        ),
                    ],
                    className='one columns',
                    style={'margin-left': 10, 'padding': 10}
                ),

                html.Div(
                    [
                        html.P('Age (Years): '),
                        dbc.Input(
                            id="age", type="number",
                            debounce=True, placeholder="30 - 80 years", min=10, max=99

                        ),
                    ],
                    className='three columns',
                    style={'margin-left': 10, 'padding': 10}
                ),

                html.Div(
                    [
                        html.P('Height (cm): '),
                        dbc.Input(
                            id="height", type="number",
                            debounce=True, placeholder="140 - 180 cm", min=140, max=180

                        ),
                    ],
                    className='three columns',
                    style={'margin-left': 10, 'padding': 10}
                ),

                html.Div(
                    [
                        html.P('Weight (kg): '),
                        dbc.Input(
                            id="weight", type="number",
                            debounce=True, placeholder="40 - 110 kg", min=40, max=110

                        ),
                    ],
                    className='three columns',
                    style={'margin-left': 10, 'padding': 10}
                ),

                html.Div(
                    [
                        html.P('Systolic Blood Pressure (mmHg): '),
                        dbc.Input(
                            id="sys", type="number",
                            debounce=True, placeholder="90 - 170 mmHg", min=90, max=170

                        ),
                    ],
                    className='three columns',
                    style={'padding': 10}
                ),
            ], className='row1'),

        dbc.Row([
            dbc.Col(html.Div(
                [
                    html.P('Cholesterol Levels : '),
                    dbc.RadioItems(
                        id="chol",
                        options=[
                            {'label': 'Normal', 'value': 0},
                            {'label': 'High', 'value': 1},
                            {'label': 'Very High', 'value': 2}],
                        value=0
                    ),
                ],
                className='one columns',
                style={'padding': 10}
            )),
            dbc.Col(html.Div(
                [
                    html.P('Glucose Levels: '),
                    dbc.RadioItems(
                        id="gluc",
                        options=[
                            {'label': 'Normal', 'value': 0},
                            {'label': 'High', 'value': 1},
                            {'label': 'Very High', 'value': 2}],
                        value=0
                    ),
                ],
                className='two columns',
                style={'padding': 10}
            )),
            dbc.Col(html.Div(
                [
                    html.P('Smoker?: '),
                    dbc.RadioItems(
                        id="smoke",
                        options=[
                            {'label': 'Yes', 'value': 1},
                            {'label': 'No', 'value': 0}],
                        value=0
                    ),
                ],
                className='three columns',
                style={'padding': 10}
            )),
            dbc.Col(html.Div(
                [
                    html.P('Drinks Alcohol?: '),
                    dbc.RadioItems(
                        id="alc",
                        options=[
                            {'label': 'Yes', 'value': 1},
                            {'label': 'No', 'value': 0}],
                        value=0
                    ),
                ],
                className='four columns',
                style={'padding': 10}
            )),
            dbc.Col(html.Div(
                [
                    html.P('Active?: '),
                    dbc.RadioItems(
                        id="active",
                        options=[
                            {'label': 'Yes', 'value': 1},
                            {'label': 'No', 'value': 0}],
                        value=0
                    ),
                ],
                className='five columns',
                style={'padding': 10}
            ))

        ],
            className='row2'
        ),

        dbc.Button('Submit', size="lg", color="light", id='submit-val', n_clicks=0, style={'padding': 10}),
        dbc.Button('Reset', size="sm", color="light", id='reset', n_clicks=0, style={'margin-left': 20, 'padding': 10}),

    ], className='row0', style={'margin-top': 10, 'border':
        '1px solid #C6CCD5', 'padding': 10,
                                'border-radius': '5px'})),
    html.Br(),
    html.Br(),
    dbc.Row(html.Div(
        id='result',
        style={'margin-top': 10, 'border': '1px solid #C6CCD5',
               'padding': 10, 'border-radius': '5px'}))

], style={'padding': '25px'})


@app.callback(Output('result', 'children'),

              [Input('submit-val', 'n_clicks'),

               State('gender', 'value'),
               State('age', 'value'),
               State('height', 'value'),
               State('weight', 'value'),
               State('sys', 'value'),
               State('chol', 'value'),
               State('gluc', 'value'),
               State('smoke', 'value'),
               State('alc', 'value'),
               State('active', 'value')
               ])
def prediction(s_clicks, gender, age, height, weight, sys, chol, gluc, smoke, alc, active):
    # Initialization
    chol_norm_bool = 0
    chol_high_bool = 0
    chol_vhigh_bool = 0
    gluc_norm_bool = 0
    gluc_high_bool = 0
    gluc_vhigh_bool = 0

    # Checks clicks
    if s_clicks == 0:
        clicked = False
    else:
        clicked = True

    # Error Catch
    if not age:
        age = 0
    if not height:
        height = 0
    if not weight:
        weight = 0
    if not sys:
        sys = 0

    print("sys", sys)
    # Turns into 1hot implementation
    if chol == 0:
        chol_norm_bool = 1
    elif chol == 1:
        chol_high_bool = 1
    else:
        chol_vhigh_bool = 1

    if gluc == 0:
        gluc_norm_bool = 1
    elif gluc == 1:
        gluc_high_bool = 1
    else:
        gluc_vhigh_bool = 1

    # Input into dataset
    if clicked:
        data = np.array([[
            age, gender, height, weight, sys,
            chol_norm_bool, chol_high_bool, chol_vhigh_bool,
            gluc_norm_bool, gluc_high_bool, gluc_vhigh_bool,
            smoke, alc, active
        ]])
        data.reshape(1, -1)

        # Check probability
        y_prob = model.predict_proba(data)
        return "The prediction for the probability of CVD is: ", round(y_prob[0, 1] * 100, 3), '%'


@app.callback([Output('gender', 'value'),
               Output('age', 'value'),
               Output('height', 'value'),
               Output('weight', 'value'),
               Output('sys', 'value'),
               Output('chol', 'value'),
               Output('gluc', 'value'),
               Output('smoke', 'value'),
               Output('alc', 'value'),
               Output('active', 'value')],

              [Input('reset', 'n_clicks')])
def update(reset):
    print(reset)
    if reset != 0:
        gender = 0
        age = "30 - 80 years"
        height = 0
        weight = 0
        sys = 0
        chol = 0
        gluc = 0
        smoke = 0
        alc = 0
        active = 0
        return gender, age, height, weight, sys, chol, gluc, smoke, alc, active


if __name__ == '__main__':
    app.run_server(debug=True)
'''
    html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(id='chart-2')
                    ], className = "four columns", style = {'margin-top': 35,
                                                            'padding': '15',
                                                            'border': '1px solid #C6CCD5'}
                ),
                html.Div(id = 'table-box'),
                html.Div(dt.DataTable(id = 'table', data=[{}]), style={'display': 'none'})
            ], className = 'row'
        )
    '''

'''
# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div([

    html.H1("Cardiovascular Disease Predictor", style={'text-align': 'center'}),

    dcc.Input(
        id="age", type="number",
        debounce=True, placeholder="Age (years)",
    ),
    html.Hr(),
    html.Div(id="age-out"),

    dcc.Input(
        id="height", type="number",
        debounce=True, placeholder="Height (cm)",
    ),
    html.Hr(),
    html.Div(id="height-out"),

    dcc.Input(
        id="weight", type="number",
        debounce=True, placeholder="Weight (kg)",
    ),
    html.Hr(),
    html.Div(id="weight-out"),

    dcc.Input(
        id="sys", type="number",
        debounce=True, placeholder="Systolic Blood Pressure (mmHg)",
    ),
    html.Hr(),
    html.Div(id="sys-out"),

    dcc.RadioItems(
        id="chol",
        options=[
            {'label': 'Normal', 'value': 0},
            {'label': 'High', 'value': 1},
            {'label': 'Very High', 'value': 2}],
        value=0
    ),

    dcc.RadioItems(
        id="gluc",
        options=[
            {'label': 'Normal', 'value': 0},
            {'label': 'High', 'value': 1},
            {'label': 'Very High', 'value': 2}],
        value=0
    ),
    html.Hr(),

    dcc.Graph(id='my_bee_map', figure={})

])

'''
