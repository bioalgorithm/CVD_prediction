import copy
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import numpy as np
import os
import pandas as pd
import plotly
import time
import joblib

from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from flask import Flask
from plotly import graph_objs as go
from plotly.graph_objs import *

app = dash.Dash(__name__)
app.title = 'Cardiovascular Disease Predictor'
# ------------------------------------------------------------------------------
# Import and clean data (importing csv into pandas)
df = pd.read_csv("cardio_train_clean_1hot_featureselection.csv")
model = joblib.load('random_forest.sav')

'''
df = df.groupby(['Age', 'Gender', 'Height', 'Weight', "Systolic BP",
              "Chlosterol_Normal","Chlosterol_High","Chlosterol_Veryhigh",
              "Glucose_Normal","Glucose_High","Glucose_Veryhigh", 
              "Smoke", "Alcohol", "Active"])[['Cardio']].mean()
df.reset_index(inplace=True)
'''
age_dict = {}
for i in range(30, 65):
    input = str(i)
    age_dict[input] = i

weight_dict = {}
for i in range(40, 110):
    input = str(i)
    weight_dict[input] = i

height_dict = {}
for i in range(140, 180):
    input = str(i)
    height_dict[input] = i

sys_dict = {}
for i in range(90, 170):
    input = str(i)
    sys_dict[input] = i

gender_dict = {"Male": 0, "Female": 1}
smoke_dict = {"No": 0, "Yes": 1}
alc_dict = {"No": 0, "Yes": 1}
active_dict = {"No": 0, "Yes": 1}
chol_dict = {"Normal": 1, "High": 2, "Very High": 3}
gluc_dict = {"Normal": 1, "High": 2, "Very High": 3}

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
# -*- coding: utf-8 -*-
'''
server = app.server

# Datasets

# Boostrap CSS.
app.css.append_css({'external_url': 'https://codepen.io/amyoshino/pen/jzXypZ.css'})
'''
layout = dict(
    autosize=True,
    height=450,
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
    html.Div(
        [
            html.H1(
                'Cardiovascular Disease Predictor',
                style={'font-family': 'Helvetica',
                       "margin-left": "20",
                       "margin-bottom": "0"},
                className='eight columns',
            )
        ],
        className='row'
    ),

    #block 2
    html.Div([
        html.H3('Patient'),
        html.Div(
            [
                html.Div(
                    [
                        html.P('Gender (M/F): '),
                        dcc.RadioItems(
                            id="gender",
                            options=[
                                {'label': 'Male', 'value': 0},
                                {'label': 'Female', 'value': 1}],
                            value=0
                        ),
                    ],
                    className='three columns',
                    style={'margin-top': '10'}
                ),

                html.Div(
                    [
                        html.P('Age (Years): '),
                        dcc.Input(
                            id="age", type="number",
                            debounce=True, placeholder="Age (years)",
                            value = 0
                                                    ),
                    ],
                    className='three columns',
                    style={'margin-top': '10'}
                ),

                html.Div(
                    [
                        html.P('Height (cm): '),
                        dcc.Input(
                            id="height", type="number",
                            debounce=True, placeholder="Height (cm)",
                            value = 0
                        ),
                    ],
                    className='one columns',
                    style={'margin-top': '40'}
                ),

                html.Div(
                    [
                        html.P('Weight (kg): '),
                        dcc.Input(
                            id="weight", type="number",
                            debounce=True, placeholder="Weight (kg)",
                            value = 0
                        ),
                    ],
                    className='one columns',
                    style={'margin-top': '40'}
                ),

                html.Div(
                    [
                        html.P('Systolic Blood Pressure (mmHg): '),
                        dcc.Input(
                            id="sys", type="number",
                            debounce=True, placeholder="Systolic Blood Pressure (mmHg)",
                            value = 0
                        ),
                    ],
                    className='one columns',
                    style={'margin-top': '40'}
                ),
                html.Div(
                    [
                        html.P('Cholesterol Levels : '),
                        dcc.RadioItems(
                            id="chol",
                            options=[
                                {'label': 'Normal', 'value': 0},
                                {'label': 'High', 'value': 1},
                                {'label': 'Very High', 'value': 2}],
                            value=0
                        ),
                    ],
                    className='one columns',
                    style={'margin-top': '40'}
                ),
                html.Div(
                    [
                        html.P('Glucose Levels: '),
                        dcc.RadioItems(
                            id="gluc",
                            options=[
                                {'label': 'Normal', 'value': 0},
                                {'label': 'High', 'value': 1},
                                {'label': 'Very High', 'value': 2}],
                            value=0
                        ),
                    ],
                    className='one columns',
                    style={'margin-top': '40'}
                ),
                html.Div(
                    [
                        html.P('Smoker?: '),
                        dcc.RadioItems(
                            id="smoke",
                            options=[
                                {'label': 'No', 'value': 0},
                                {'label': 'Yes', 'value': 1}],
                            value=0
                        ),
                    ],
                    className='one columns',
                    style={'margin-top': '40'}
                ),
                html.Div(
                    [
                        html.P('Drinks Alcohol?: '),
                        dcc.RadioItems(
                            id="alc",
                            options=[
                                {'label': 'No', 'value': 0},
                                {'label': 'Yes', 'value': 1}],
                            value=0
                        ),
                    ],
                    className='one columns',
                    style={'margin-top': '40'}
                ),
                html.Div(
                    [
                        html.P('Active?: '),
                        dcc.RadioItems(
                            id="active",
                            options=[
                                {'label': 'No', 'value': 0},
                                {'label': 'Yes', 'value': 1}],
                            value=0
                        ),
                    ],
                    className='one columns',
                    style={'margin-top': '40'}
                ),

            ],
            className='row'
        ),


    html.Br(),
    html.Br(),
    html.Div(id='result')

    ], className = 'row',  style = {'margin-top': 20, 'border':
                                    '1px solid #C6CCD5', 'padding': 15,
                                    'border-radius': '5px'})
], style = {'padding': '25px'})

@app.callback(dash.dependencies.Output('result', 'children'),
              [dash.dependencies.Input('gender', 'value'),
               dash.dependencies.Input('age', 'value'),
               dash.dependencies.Input('height', 'value'),
               dash.dependencies.Input('weight', 'value'),
               dash.dependencies.Input('sys', 'value'),
               dash.dependencies.Input('chol', 'value'),
               dash.dependencies.Input('gluc', 'value'),
               dash.dependencies.Input('smoke', 'value'),
               dash.dependencies.Input('alc', 'value'),
               dash.dependencies.Input('active', 'value'),
               ])

def prediction(gender, age, height, weight, sys, chol, gluc, smoke, alc, active):
    chol_norm_bool = 0
    chol_high_bool = 0
    chol_vhigh_bool = 0
    gluc_norm_bool = 0
    gluc_high_bool = 0
    gluc_vhigh_bool = 0

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
    data = np.array([[
            age, gender, height, weight, sys,
            chol_norm_bool, chol_high_bool, chol_vhigh_bool,
            gluc_norm_bool, gluc_high_bool, gluc_vhigh_bool,
            smoke, alc, active
    ]])
    data.reshape(1, -1)
    print(data)
    y_prob = model.predict_proba(data)
    print(y_prob[0,1])
    return "The prediction for the probability of CVD is: ", y_prob[0, 1]

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
