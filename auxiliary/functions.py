# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 22:42:19 2020

@author: admin
"""

##Functions for replication study

# Import modules.
import pandas as pd
import statsmodels.api as sm
import numpy as np

from scipy import stats
from linearmodels import IV2SLS

# Get significance asterix.
def significance(pval):
    if type(pval) == str:
        star = ''
    elif pval <= 0.01:
        star = '***'
    elif pval <= 0.05:
        star = '**'
    elif pval <= 0.1:
        star = '*'
    else:
        star = ' '
    return star

# Generate cohort dummies.
def get_cohort_dummy(df, col, c):
    '''
    Inputs are
    a DataFrame,
    a column col (string), and
    an input c (cohort) for which the output variable shall return 1.
    newcol
    '''
    #Load data.
    #path = ('data/Crime.dta')
    #df = pd.read_stata(path)
    # Get name of cohort dummy c.
    newcol = 'cohort_' + f'{c}'
    # Define a function that creates a dummy var. conditional on another column.
    def dummy_mapping(x):
        if x == c:
            return 1
        elif x == np.nan:
            return np.nan
        else:
            return 0
    df[newcol] = df[col].apply(dummy_mapping)
    return df

# Set up data frame and variables for regressions.
def get_variables():
    '''
    '''
    # Load data.
    path = ('data/Crime.dta')
    df = pd.read_stata(path)
    # For the regressions below, add a constant to the data frame.
    df['constant'] = 1
    
    # Get a variable representing the strings to add them to regression functions.
    constant = ['constant']
    highnumber = ['highnumber']
    conscription = ['sm']
    crimerate = ['crimerate']
    malvinas = ['malvinas']
    navy = ['navy']
    
    # Get list of origin dummy names. Omit 'argentine' i.o.t. avoid multicollinearity.
    origin = ['naturalized', 'indigenous']
    
    # Get cohort dummies from 1929 to 1965.
    for year in list(range(1929, 1966, 1)):
        get_cohort_dummy(df=df, col='cohort', c=year)
    
    # Get list of cohort dummy names.
    cohort_years = list(range(1930, 1966, 1))  # Omit cohort_1929 (multicollinearity).
    cohorts = []
    for i in cohort_years:
        cohorts.append('cohort_' + f'{i}')

    # Get list of district dummy names. District dummies have already been provided in data.
    district_numbers = list(range(2, 25, 1))  # Omit dist1 (multicollinearity).
    districts = []
    for i in district_numbers:
        districts.append('dist' + f'{i}')
        
    # Generate variable hn_malvinas: interaction term between highnumber and malvinas.
    df['hn_malvinas'] = df.highnumber*df.malvinas
    hn_malvinas = ['hn_malvinas']
    
    return constant, highnumber, conscription, crimerate, malvinas, navy, origin, cohorts, districts, hn_malvinas, df

# Get plot as in figure A.1.
def binned_plot(df, bin_num, ylim, years):
    '''
    Returns plots for crime rate. To smooth out fluctuations, data can be partitioned into bin_num bins. For each bin the mean
    of crime rate is computed. Number of plots returned depends on number of cohorts desired.
    bin_num: int, number of bins
    ylim: list/2-tuple, range of y-axis of plots
    years: list of cohorts
    '''
    bins = np.linspace(0, 1000, bin_num+1)
    for i in years:
        binned_stats = stats.binned_statistic(x=df[df.cohort == i].draftnumber, values=df[df.cohort == i].enfdummy, 
                                              statistic='mean', bins=bins)
        df_bin = pd.DataFrame()
        df_bin['Crime rate'] = binned_stats.statistic
        df_bin['Draftnumber'] = bins[1: bin_num+1]
        df_bin.plot.line(x='Draftnumber', y='Crime rate', title=f'Crime Rates for Cohort {i}', ylim=ylim)