# TODO: report for setorder = 1, ..., n for dplural = n > 1

import os
import pickle
import pandas as pd
import numpy as np
import bisect

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import xgboost

os.chdir(os.path.dirname(__file__))

cat_ranks = pickle.load(open("Full 2016 Target Mean Encodings.p", "rb"))

prenatal_model = pickle.load(open("Prenatal Model.p", "rb"))
prenatal_qtiles = pickle.load(open("Prenatal Quantiles.p", "rb"))

df_varlist = pd.read_csv('Neonatal Mortality Predictor List.csv')
prenatal_vars = df_varlist[df_varlist['prenatal']==1]['feature'].tolist()
prenatal_vars.remove('ilive')
del df_varlist

app = dash.Dash()

app.layout = html.Div(children=[
    html.H1(children='Prenatal Predictor of Neonatal Mortality'),
             
    html.Label('Mother\'s Age'),
    dcc.Dropdown(
        clearable=False,
        options=[{'label': str(i), 'value': i} for i in range(12, 51)],
        value = 'missing',
        id='mager'
    ),
             
    html.Label('Mother\'s Height in Inches'),
    dcc.Dropdown(
        clearable=False,
        options=[{'label': 'Unknown or Not Stated', 'value': 'missing'}] + \
                [{'label': str(i), 'value': i} for i in range(30, 78)],
        value = 'missing',
        id='m_ht_in'
    ),
             
    html.Label('Mother\'s Birthplace'),
    dcc.Dropdown(
        clearable=False,
        options=[{'label': 'Unknown or Not Stated', 'value': 3}] + \
                [
            {'label': 'In the 50 U.S. states', 'value': 1},
            {'label': 'Outside the 50 U.S. states', 'value': 2}
        ],
        value = 3,
        id='mbstate_rec'
    ),
    
    html.Label('Mother\'s Race'),
    dcc.Checklist(
        options=[
            {'label': 'American Indian or Alaskan Native', 'value': 'AIAN'},
            {'label': 'Asian', 'value': 'Asian'},
            {'label': 'Black', 'value': 'Black'},
            {'label': 'Native Hawaiian or Other Pacific Islander', 'value': 'NHOPI'},
            {'label': 'White', 'value': 'White'}
        ],
        values=[],
        id='mrace31'
    ),
             
    html.Label('Mother\'s Hispanic Origin'),
    dcc.Dropdown(
        clearable=False,
        options=[
            {'label': 'Non-hispanic', 'value': 0},
            {'label': 'Mexican', 'value': 1},
            {'label': 'Puerto Rican', 'value': 2},
            {'label': 'Cuban', 'value': 3},
            {'label': 'Central and South American', 'value': 4},
            {'label': 'Other and Unknown Hispanic origin ', 'value': 5},
            {'label': 'Hispanic Origin Not Stated', 'value': 9}
        ],
        value = 9,
        id='mhisp_r'
    ),
             
    html.Label('Mother\'s Education'),
    dcc.Dropdown(
        clearable=False,
        options=[
            {'label': '8th Grade or Less', 'value': 1},
            {'label': '9th through 12th grade with no diploma', 'value': 2},
            {'label': 'High school graduate or GED completed', 'value': 3},
            {'label': 'Some college credit, but not a degree', 'value': 4},
            {'label': 'Associate degree (AA,AS)', 'value': 5},
            {'label': 'Bachelor’s degree (BA, AB, BS)', 'value': 6},
            {'label': 'Master’s degree (MA, MS, MEng, MEd, MSW, MBA)', 'value': 7},
            {'label': 'Doctorate (PhD, EdD) or Professional Degree (MD, DDS, DVM, LLB, JD)', 'value': 8},
            {'label': 'Unknown', 'value': 9}
        ],
        value = 9,
        id='meduc'
    ),
    
    html.Label('Mother Married'),
    dcc.RadioItems(
        options=[
            {'label': 'Yes', 'value': '1'},
            {'label': 'No', 'value': '2'},
        ],
        id='dmar'
    ),
             
    html.Label('Paternity Acknowledged'),
    dcc.Dropdown(
        clearable=False,
        options=[
            {'label': 'Yes', 'value': 'Y'},
            {'label': 'No', 'value': 'N'},
            {'label': 'Unknown', 'value': 'U'},
            {'label': 'Not Applicable', 'value': 'X'}
        ],
        value = 'U',
        id='mar_p'
    ),
             
    html.Label('Father\'s Age'),
    dcc.Dropdown(
        clearable=False,
        options=[{'label': 'Unknown or Not Stated', 'value': 'missing'}] + \
                [{'label': str(i), 'value': i} for i in range(9, 99)],
        value = 'missing',
        id='fagecomb'
    ),
             
    html.Label('Father\'s Education'),
    dcc.Dropdown(
        clearable=False,
        options=[
            {'label': '8th Grade or Less', 'value': 1},
            {'label': '9th through 12th grade with no diploma', 'value': 2},
            {'label': 'High school graduate or GED completed', 'value': 3},
            {'label': 'Some college credit, but not a degree', 'value': 4},
            {'label': 'Associate degree (AA,AS)', 'value': 5},
            {'label': 'Bachelor’s degree (BA, AB, BS)', 'value': 6},
            {'label': 'Master’s degree (MA, MS, MEng, MEd, MSW, MBA)', 'value': 7},
            {'label': 'Doctorate (PhD, EdD) or Professional Degree (MD, DDS, DVM, LLB, JD)', 'value': 8},
            {'label': 'Unknown', 'value': 9}
        ],
        value = 9,
        id='feduc'
    ),
    
    html.Label('Father\'s Race'),
    dcc.Checklist(
        options=[
            {'label': 'American Indian or Alaskan Native', 'value': 'AIAN'},
            {'label': 'Asian', 'value': 'Asian'},
            {'label': 'Black', 'value': 'Black'},
            {'label': 'Native Hawaiian or Other Pacific Islander', 'value': 'NHOPI'},
            {'label': 'White', 'value': 'White'},
            {'label': 'Unknown', 'value': 99}
        ],
        values=[],
        id='frace31'
    ),
             
    html.Label('Father\'s Hispanic Origin'),
    dcc.Dropdown(
        clearable=False,
        options=[
            {'label': 'Non-hispanic', 'value': 0},
            {'label': 'Mexican', 'value': 1},
            {'label': 'Puerto Rican', 'value': 2},
            {'label': 'Cuban', 'value': 3},
            {'label': 'Central and South American', 'value': 4},
            {'label': 'Other and Unknown Hispanic origin ', 'value': 5},
            {'label': 'Hispanic Origin Not Stated', 'value': 9}
        ],
        value = 9,
        id='fhisp_r'
    ),
    
    html.Div([], style={'marginBottom': 1024}),
            
    html.Label('Mother\'s Cigarettes Smoked per Day Before Pregnancy:'),
    dcc.Dropdown(
        clearable=False,
        options=[{'label': 'Unknown or Not Stated', 'value': 'missing'}] + \
                [{'label': str(i), 'value': i} for i in range(0, 99)],
        value = 'missing',
        id='cig_0'
    ),
    html.Label('Mother\'s Cigarettes Smoked per Day During 1st Trimester:'),
    dcc.Dropdown(
        clearable=False,
        options=[{'label': 'Unknown or Not Stated', 'value': 'missing'}] + \
                [{'label': str(i), 'value': i} for i in range(0, 99)],
        value = 'missing',
        id='cig_1'
    ),
    html.Label('Mother\'s Cigarettes Smoked per Day During 2nd Trimester:'),
    dcc.Dropdown(
        clearable=False,
        options=[{'label': 'Unknown or Not Stated', 'value': 'missing'}] + \
                [{'label': str(i), 'value': i} for i in range(0, 99)],
        value = 'missing',
        id='cig_2'
    ),
    html.Label('Mother\'s Cigarettes Smoked per Day During 3rd Trimester:'),
    dcc.Dropdown(
        clearable=False,
        options=[{'label': 'Unknown or Not Stated', 'value': 'missing'}] + \
                [{'label': str(i), 'value': i} for i in range(0, 99)],
        value = 'missing',
        id='cig_3'
    ),
    
    html.Label('Mother Received WIC During Pregnancy'),
    dcc.RadioItems(
        options=[
            {'label': 'Yes', 'value': 'Y'},
            {'label': 'No', 'value': 'N'},
            {'label': 'Unknown', 'value': 'U'}
        ],
        value = 'U',
        id='wic'
    ),
             
    html.Label('Mother\'s Pre-pregnancy Weight (lbs)'),
    dcc.Dropdown(
        clearable=False,
        options=[{'label': 'Unknown or Not Stated', 'value': 'missing'}] + \
                [{'label': str(i), 'value': i} for i in range(75, 375)],
        value = 'missing',
        id='pwgt_r'
    ),
             
    html.Label('Mother\'s Weight at Delivery (lbs)'),
    dcc.Dropdown(
        clearable=False,
        options=[{'label': 'Unknown or Not Stated', 'value': 'missing'}] + \
                [{'label': str(i), 'value': i} for i in range(100, 400)],
        value = 'missing',
        id='dwgt_r'
    ),
             
    html.Label('Month of Pregnancy Prenatal Care Began'),
    dcc.Dropdown(
        clearable=False,
        options=[{'label': 'Unknown or Not Stated', 'value': 'missing'}] + \
                [
            {'label': 'No Prenatal Care', 'value': 0},
            {'label': '1', 'value': 1},
            {'label': '2', 'value': 2},
            {'label': '3', 'value': 3},
            {'label': '4', 'value': 4},
            {'label': '5', 'value': 5},
            {'label': '6', 'value': 6},
            {'label': '7', 'value': 7},
            {'label': '8', 'value': 8},
            {'label': '9', 'value': 9},
            {'label': '10', 'value': 10}
        ],
        value = 'missing',
        id='precare'
    ),
             
    html.Label('Number of Prenatal Visits'),
    dcc.Dropdown(
        clearable=False,
        options=[{'label': 'Unknown or Not Stated', 'value': 'missing'}] + \
                [{'label': str(i), 'value': i} for i in range(0, 99)],
        value = 'missing',
        id='previs'
    ),
             
    html.Label('Prior Births Now Living'),
    dcc.Dropdown(
        clearable=False,
        options=[{'label': 'Unknown or Not Stated', 'value': 'missing'}] + \
                [{'label': str(i), 'value': i} for i in range(31)],
        value = 'missing',
        id='priorlive'
    ),
             
    html.Label('Prior Births Now Dead'),
    dcc.Dropdown(
        clearable=False,
        options=[{'label': 'Unknown or Not Stated', 'value': 'missing'}] + \
                [{'label': str(i), 'value': i} for i in range(31)],
        value = 'missing',
        id='priordead'
    ),
             
    html.Label('Prior Other Terminations'),
    dcc.Dropdown(
        clearable=False,
        options=[{'label': 'Unknown or Not Stated', 'value': 'missing'}] + \
                [{'label': str(i), 'value': i} for i in range(31)],
        value = 'missing',
        id='priorterm'
    ),
             
    html.Label('Months Since Last Live Birth'),
    dcc.Dropdown(
        clearable=False,
        options=[{'label': 'Unknown, Not Stated, or Not Applicable', 'value': 'missing'}] + \
                [{'label': str(i), 'value': i} for i in range(301)],
        value = 'missing',
        id='illb_r'
    ),
             
    html.Label('Months Since Last Other Pregnancy'),
    dcc.Dropdown(
        clearable=False,
        options=[{'label': 'Unknown, Not Stated, or Not Applicable', 'value': 'missing'}] + \
                [{'label': str(i), 'value': i} for i in range(301)],
        value = 'missing',
        id='ilop_r'
    ),
             
    html.Label('Gestation Duration Based on Last Normal Menses (Weeks)'),
    dcc.Dropdown(
        clearable=False,
        options=[{'label': 'Unknown', 'value': 'missing'}] + \
                [{'label': str(i), 'value': i} for i in range(17, 48)],
        value = 'missing',
        id='combgest'
    ),
             
    html.Label('Gestation Duration Based on Obstetric Estimate (Weeks)'),
    dcc.Dropdown(
        clearable=False,
        options=[{'label': 'Not Stated', 'value': 'missing'}] + \
                [{'label': str(i), 'value': i} for i in range(17, 48)],
        value = 'missing',
        id='oegest_comb'
    ),
            
    html.Div([], style={'marginBottom': 1024}),
             
    html.Label('Expected Number of Births'),
    dcc.Dropdown(
        clearable=False,
        options=[{'label': str(i), 'value': i} for i in range(1, 5)] + \
                [{'label': '5 or more', 'value': 5}],
        value = 'missing',
        id='dplural'
    ),
             
    html.Label('Infant Sex'),
    dcc.Dropdown(
        clearable=False,
        options=[
            {'label': 'Male', 'value': 'M'},
            {'label': 'Female', 'value': 'F'}
        ],
        value = 'missing',
        id='sex'
    ),
    
    html.Label('Infections Present'),
    dcc.Checklist(
        options=[
            {'label': 'Gonorrhea', 'value': 'gon'},
            {'label': 'Syphilis', 'value': 'syph'},
            {'label': 'Chlamydia', 'value': 'chlam'},
            {'label': 'Hepatitis B', 'value': 'hepatb'},
            {'label': 'Hepatitis C', 'value': 'hepatc'},
            {'label': 'None of Those Listed', 'value': 'no_infec'}
        ],
        values=[],
        id='ip_'
    ),
             
    html.Label('Number of Previous Cesareans'),
    dcc.Dropdown(
        clearable=False,
        options=[{'label': str(i), 'value': i} for i in range(31)],
        value = 'missing',
        id='rf_cesarn'
    ),
    
    html.Label('Risk Factors'),
    dcc.Checklist(
        options=[
            {'label': 'Pre-pregnancy Diabetes', 'value': 'pdiab'},
            {'label': 'Gestational Diabetes', 'value': 'gdiab'},
            {'label': 'Pre-pregnancy Hypertension', 'value': 'phype'},
            {'label': 'Gestational Hypertension', 'value': 'ghype'},
            {'label': 'Hypertension Eclampsia', 'value': 'ehype'},
            {'label': 'Previous Preterm Birth', 'value': 'ppterm'},
            {'label': 'Infertility Treatment Used', 'value': 'inftr'},
            {'label': 'Fertility Enhancing Drugs', 'value': 'fedrg'},
            {'label': 'Assisted Reproductive Technology', 'value': 'artec'},
            {'label': 'None of Those Listed', 'value': 'no_risks'}
        ],
        values=[],
        id='rf_'
    ),
    
    html.Label('Obstetric Procedures'),
    dcc.Checklist(
        options=[
            {'label': 'Successful External Cephalic Version', 'value': 'ecvs'},
            {'label': 'Failed External Cephalic Version', 'value': 'ecvf'}
        ],
        values=[],
        id='ob_'
    ),
             
    html.Label('Birth Month'),
    dcc.Dropdown(
        clearable=False,
        options=[
            {'label': 'January', 'value': 1},
            {'label': 'February', 'value': 2},
            {'label': 'March', 'value': 3},
            {'label': 'April', 'value': 4},
            {'label': 'May', 'value': 5},
            {'label': 'June', 'value': 6},
            {'label': 'July', 'value': 7},
            {'label': 'August', 'value': 8},
            {'label': 'September', 'value': 9},
            {'label': 'October', 'value': 10},
            {'label': 'November', 'value': 11},
            {'label': 'December', 'value': 12}
        ],
        value = 'missing',
        id='dob_mm'
    ),
             
    html.Label('Payment Source for Delivery'),
    dcc.Dropdown(
        clearable=False,
        options=[
            {'label': 'Medicaid', 'value': 1},
            {'label': 'Private Insurance', 'value': 2},
            {'label': 'Self-Pay', 'value': 3},
            {'label': 'Indian Health Service', 'value': 4},
            {'label': 'CHAMPUS/TRICARE', 'value': 5},
            {'label': 'Other Government (Federal, State, Local)', 'value': 6},
            {'label': 'Other', 'value': 8},
            {'label': 'Unknown', 'value': 9}
        ],
        value = 9,
        id='pay'
    ),
    
    dcc.Markdown(id='output', children='No output yet'),
    
    html.Button('Predict', id='button'),
    
], style={'columnCount': 3})
    
def cat_map(cat, value):
    return cat_ranks[cat].get(str(value), 'missing')

def race_map(races):
    if 99 in races:
        return 99
    else:
        map_dict = {"[]":99,
                    "['White']":1,
                    "['Black']":2,
                    "['AIAN']":3,
                    "['Asian']":4,
                    "['NHOPI']":5,
                    "['Black', 'White']":6,
                    "['AIAN', 'Black']":7,
                    "['Asian', 'Black']":8,
                    "['Black', 'NHOPI']":9,
                    "['AIAN', 'White']":10,
                    "['AIAN', 'Asian']":11,
                    "['AIAN', 'NHOPI']":12,
                    "['Asian', 'White']":13,
                    "['Asian', 'NHOPI']":14,
                    "['NHOPI', 'White']":15,
                    "['AIAN', 'Black', 'White']":16,
                    "['AIAN', 'Asian', 'Black']":17,
                    "['AIAN', 'Black', 'NHOPI']":18,
                    "['Asian', 'Black', 'White']":19,
                    "['Asian', 'Black', 'NHOPI']":20,
                    "['Black', 'NHOPI', 'White']":21,
                    "['AIAN', 'Asian', 'White']":22,
                    "['AIAN', 'NHOPI', 'White']":23,
                    "['AIAN', 'Asian', 'NHOPI']":24,
                    "['Asian', 'NHOPI', 'White']":25,
                    "['AIAN', 'Asian', 'Black', 'White']":26,
                    "['AIAN', 'Asian', 'Black', 'NHOPI']":27,
                    "['AIAN', 'Black', 'NHOPI', 'White']":28,
                    "['Asian', 'Black', 'NHOPI', 'White']":29,
                    "['AIAN', 'Asian', 'NHOPI', 'White']":30,
                    "['AIAN', 'Asian', 'Black', 'NHOPI', 'White']":31
                    }
        return map_dict[repr(races)]

def racehisp_map(race31, hisp_r):
    if hisp_r == None:
        return 8
    elif hisp_r > 0:
        return 7
    elif race31 == []:
        return 9
    elif race31 == ['White']:
        return 1
    elif race31 == ['Black']:
        return 2
    elif race31 == ['AIAN']:
        return 3
    elif race31 == ['Asian']:
        return 4
    elif race31 == ['NHOPI']:
        return 5
    else:
        return 6

@app.callback(
    Output('output', 'children'),
    [Input('button', 'n_clicks')],
    state=[State('sex', 'value'),
           State('mbstate_rec', 'value'),
           State('wic', 'value'),
           State('dmar', 'value'),
           State('cig_0', 'value'),
           State('cig_1', 'value'),
           State('cig_2', 'value'),
           State('cig_3', 'value'),
           State('rf_', 'values'),
           State('ip_', 'values'),
           State('ob_', 'values'),
           State('rf_cesarn', 'value'),
           State('m_ht_in', 'value'),
           State('illb_r', 'value'),
           State('ilop_r', 'value'),
           State('mar_p', 'value'),
           State('dplural', 'value'),
           State('mrace31', 'values'),
           State('frace31', 'values'),
           State('mhisp_r', 'value'),
           State('fhisp_r', 'value'),
           State('pay', 'value'),
           State('meduc', 'value'),
           State('feduc', 'value'),
           State('dob_mm', 'value'),
           State('precare', 'value'),
           State('pwgt_r', 'value'),
           State('dwgt_r', 'value'),
           State('priordead', 'value'),
           State('priorlive', 'value'),
           State('priorterm', 'value'),
           State('combgest', 'value'),
           State('oegest_comb', 'value'),
           State('mager', 'value'),
           State('fagecomb', 'value'),
           State('previs', 'value')])
def compute(n_clicks, sex, mbstate_rec, wic, dmar, cig_0, cig_1, cig_2, cig_3,
            rf_, ip_, ob_, rf_cesarn, m_ht_in, illb_r, ilop_r, mar_p,
            dplural, mrace31, frace31, mhisp_r, fhisp_r, pay, meduc, feduc,
            dob_mm, precare, pwgt_r, dwgt_r, priordead, priorlive, priorterm,
            combgest, oegest_comb, mager, fagecomb, previs):
    df = pd.DataFrame({
    'sex':[cat_map('sex', sex)],
    'mbstate_rec':[cat_map('mbstate_rec', mbstate_rec)],
    'wic':[cat_map('wic', wic)],
    'dmar':[cat_map('dmar', dmar)],
    'cig_rec':[cat_map('cig_rec', 'N' if cig_1 == 0 and cig_2 == 0 and cig_3 == 0 else 'Y')],
    'rf_pdiab':[cat_map('rf_pdiab', 'Y' if 'pdiab' in rf_ else 'N')],
    'rf_gdiab':[cat_map('rf_gdiab', 'Y' if 'gdiab' in rf_ else 'N')],
    'rf_phype':[cat_map('rf_phype', 'Y' if 'phype' in rf_ else 'N')],
    'rf_ghype':[cat_map('rf_ghype', 'Y' if 'ghype' in rf_ else 'N')],
    'rf_ehype':[cat_map('rf_ehype', 'Y' if 'ehype' in rf_ else 'N')],
    'rf_ppterm':[cat_map('rf_ppterm', 'Y' if 'ppterm' in rf_ else 'N')],
    'rf_inftr':[cat_map('rf_inftr', 'Y' if 'inftr' in rf_ else 'N')],
    'rf_fedrg':[cat_map('rf_fedrg', 'Y' if 'fedrg' in rf_ else 'N')],
    'rf_artec':[cat_map('rf_artec', 'Y' if 'artec' in rf_ else 'N')],
    'no_risks':[cat_map('no_risks', 9 if (rf_ == [] and rf_cesarn == 0) else 1 if (rf_ == ['no_risks'] and rf_cesarn == 0) else 0)],
    'ip_gon':[cat_map('ip_gon', 'Y' if 'gon' in rf_ else 'N')],
    'ip_syph':[cat_map('ip_syph', 'Y' if 'syph' in rf_ else 'N')],
    'ip_chlam':[cat_map('ip_chlam', 'Y' if 'chlam' in rf_ else 'N')],
    'ip_hepatb':[cat_map('ip_hepatb', 'Y' if 'hepatb' in rf_ else 'N')],
    'ip_hepatc':[cat_map('ip_hepatc', 'Y' if 'hepatc' in rf_ else 'N')],
    'no_infec':[cat_map('no_risks', 9 if ip_ == [] else 1 if ip_ == ['no_infec'] else 0)],
    'ob_ecvs':[cat_map('ob_ecvs', 'Y' if 'ecvs' in ob_ else 'N')],
    'ob_ecvf':[cat_map('ob_ecvf', 'Y' if 'ecvf' in ob_ else 'N')],
    'mar_p':[cat_map('mar_p', mar_p)],
    'dplural':[cat_map('dplural', dplural)],
    'mrace31':[cat_map('mrace31', race_map(mrace31))],
    'frace31':[cat_map('frace31', race_map(frace31))],
    'mhisp_r':[cat_map('mhisp_r', mhisp_r)],
    'fhisp_r':[cat_map('fhisp_r', fhisp_r)],
    'mracehisp':[cat_map('mracehisp', racehisp_map(mrace31, mhisp_r))],
    'pay':[cat_map('pay', pay)],
    'meduc':[cat_map('meduc', meduc)],
    'fracehisp':[cat_map('mracehisp', racehisp_map(frace31, fhisp_r))],
    'feduc':[cat_map('feduc', feduc)],
    'dob_mm':[cat_map('dob_mm', dob_mm)],
    'precare':[cat_map('precare', precare)],
    'rf_cesarn':[rf_cesarn],
    'bmi':[703 * pwgt_r / (m_ht_in**2) if 'missing' not in [pwgt_r, m_ht_in] else 'missing'],
    'priordead':[priordead],
    'priorlive':[priorlive],
    'priorterm':[priorterm],
    'combgest':[combgest],
    'oegest_comb':[oegest_comb],
    'mager':[mager],
    'm_ht_in':[m_ht_in],
    'cig_3':[cig_3],
    'cig_2':[cig_2],
    'cig_1':[cig_1],
    'cig_0':[cig_0],
    'fagecomb':[fagecomb],
    'previs':[previs],
    'wtgain':[dwgt_r - pwgt_r if 'missing' not in [dwgt_r, pwgt_r] else 'missing'],
    'illb_r':[illb_r],
    'ilop_r':[ilop_r],
    'ilp_r':[min(illb_r, ilop_r) if 'missing' not in [illb_r, ilop_r] else 'missing'],
    'pwgt_r':[pwgt_r],
    'dwgt_r':[dwgt_r],
    'setorder_r':[cat_map('setorder_r', 1)],
    })
    df = df.replace('missing', np.nan)
    prediction = prenatal_model.predict(xgboost.DMatrix(df[prenatal_vars]), ntree_limit=prenatal_model.best_ntree_limit)
    percentile = bisect.bisect_left(prenatal_qtiles, prediction) / 10000.0
    sex_pronoun = {'M':'his',
                   'F':'her',
                   'missing':'its'}[sex]
    pred_message = 'A child born with these characteristics has a {0:.2%} predicted probability of surviving through completion of '.format(1-prediction[0]) + sex_pronoun + ' standard certificate of live birth, better than {0:.2%} of births based on 2016 data.'.format(1-percentile)
    if dplural in range(2, 6):
        pred_message = pred_message.replace('A child born', 'A child born first')
        num_order_to_word = {1:'first', 2:'second', 3:'third', 4:'fourth', 5:'fifth', 6:'sixth'}
        plural_predictions = {}
        plural_predictions[1] = prediction
        num_listed_children = 1
        for i in range(2, dplural+1):
            df['setorder_r'] = [cat_map('dplural', i)]
            plural_predictions[i] = prenatal_model.predict(xgboost.DMatrix(df[prenatal_vars]), ntree_limit=prenatal_model.best_ntree_limit)
            if np.round(plural_predictions[i], 3) == np.round(plural_predictions[i-1], 3):
                num_listed_children += 1
                pred_message = pred_message.replace(num_order_to_word[i-1], num_order_to_word[i-1] + ' or ' + num_order_to_word[i])
                if num_listed_children == 3:
                    pred_message = pred_message.replace(num_order_to_word[i-2] + ' or ' + num_order_to_word[i-1], num_order_to_word[i-2] + ', ' + num_order_to_word[i-1] + ',')
                elif num_listed_children > 3:
                    pred_message = pred_message.replace(num_order_to_word[i-2] + ', or ' + num_order_to_word[i-1], num_order_to_word[i-2] + ', ' + num_order_to_word[i-1] + ',')
            else:
                percentile = bisect.bisect_left(prenatal_qtiles, plural_predictions[i][0]) / 10000.0
                pred_message = pred_message + '\n\nA child born ' + num_order_to_word[i] + ' with these characteristics has a {0:.2%} predicted probability of surviving through completion of '.format(1-plural_predictions[i][0]) + sex_pronoun + ' standard certificate of live birth, better than {0:.2%} of births based on 2016 data.'.format(1-percentile)
                num_listed_children = 1
    return pred_message

if __name__ == '__main__':
    app.run_server(debug=True)

# Observe in browser at http://127.0.0.1:8050/