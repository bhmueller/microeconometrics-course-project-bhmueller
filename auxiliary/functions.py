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
        
# Regressions (initially for table 4).
def regress(df, method, cohort_range, cohort_dummies, controls):
    '''
    df: data frame to use.
    method: string, either 'IV' for IV2SLSL by linearmodels or 'OLS' for OLS by statsmodels.
    cohort_range: list/2-tuple, indicating first and last cohort.
    cohorts: cohort dummies to include, for 1958-'62: cohorts=cohorts[29: 33].
    controls: string, either 'y' or 'n'.
    '''
    constant, highnumber, conscription, crimerate, malvinas, navy, origin, cohorts, districts, hn_malvinas, df = get_variables()
    
    if method == 'OLS':
        if controls == 'y':
            vars_controls = highnumber + cohort_dummies + origin + districts + constant
            df = df[df.cohort >= cohort_range[0]][df.cohort <= cohort_range[1]][vars_controls + crimerate].dropna().copy()
            X = df[vars_controls].copy()
            y = df.loc[:, 'crimerate']
                
            rslts = sm.OLS(y, X).fit()
            return rslts
        
        if controls == 'n':
            vars_no_controls = highnumber + cohort_dummies + constant
            df = df[df.cohort >= cohort_range[0]][df.cohort <= cohort_range[1]][vars_no_controls + crimerate].dropna().copy()
            X = df[vars_no_controls].copy()
            y = df.loc[:, 'crimerate']
                
            rslts = sm.OLS(y, X).fit()
            return rslts
        
    if method == 'IV':
        
        if controls == 'y':
            cohorts=cohorts[29: 33]
            vars_controls = highnumber + conscription + cohort_dummies + origin + districts + constant
            df = df[df.cohort >= cohort_range[0]][df.cohort <= cohort_range[1]][vars_controls + crimerate].copy().dropna(axis=0)
            y = df.loc[:, 'crimerate']
            
            rslts = IV2SLS(y, df[constant + cohorts + origin + districts], df['sm'], df['highnumber']).fit()
            return rslts
        
        if controls == 'n':
            cohorts=cohorts[29: 33]
            vars_no_controls = highnumber + conscription + cohort_dummies + constant
            df = df[df.cohort >= cohort_range[0]][df.cohort <= cohort_range[1]][vars_no_controls + crimerate].copy().dropna(axis=0)
            y = df.loc[:, 'crimerate']
            
            rslts = IV2SLS(y, df[constant + cohorts], df['sm'], df['highnumber']).fit()
            return rslts
        
# Regressions for table 4.
def regressions_table_4(df):
    '''
    Function returns regression results as in table 4 in Galiani et al. 2011.
    First, it computes the estimates.
    Arguments:
    df: data frame to use.
    '''
    constant, highnumber, conscription, crimerate, malvinas, navy, origin, cohorts, districts, hn_malvinas, df = get_variables()
    # Lists to store estimates, standard errors, no. of obs, percent change, and whether controls were used.
    est_sm = []
    est_hn = []
    std_hn = []
    std_sm = []
    pval_sm = []
    pval_hn = []
    percent_change = []
    num_obs = []
    
    # For computing percent change:
    p1 = df['sm'][df.cohort >= 1958][df.cohort <= 1962][df.highnumber == 1].dropna().mean()
    p2 = df['sm'][df.cohort >= 1958][df.cohort <= 1962][df.highnumber == 0].dropna().mean()
    
    # Get regressions.
    # Col 1.
    rslts = regress(df=df, method='OLS', cohort_range=[1958, 1962], cohort_dummies=cohorts[29: 33], controls='n')
            
    est_sm.append('-')
    est_hn.append(rslts.params['highnumber'])
    std_sm.append('-')
    std_hn.append(rslts.HC0_se['highnumber'])
    pval_sm.append('-')
    pval_hn.append(rslts.pvalues.highnumber)
    num_obs.append(rslts.nobs)
            
    wald = (rslts.params['highnumber']/(p1 - p2))
    mean_crime = df.crimerate[df.cohort >= 1958][df.cohort <= 1962][df.highnumber == 0].mean()
    percent_change.append(100*wald/mean_crime)
            
    # Col 2.
    rslts = regress(df=df, method='OLS', cohort_range=[1958, 1962], cohort_dummies=cohorts[29: 33], controls='y')
            
    est_sm.append('-')
    est_hn.append(rslts.params['highnumber'])
    std_sm.append('-')
    std_hn.append(rslts.HC0_se['highnumber'])
    pval_sm.append('-')
    pval_hn.append(rslts.pvalues.highnumber)
    num_obs.append(rslts.nobs)
            
    wald = (rslts.params['highnumber']/(p1 - p2))
    mean_crime = df.crimerate[df.cohort >= 1958][df.cohort <= 1962][df.highnumber == 0].mean()
    percent_change.append(100*wald/mean_crime)
            
    # Col 3.
    rslts = regress(df=df, method='IV', cohort_range=[1958, 1962], cohort_dummies=cohorts[29: 33], controls='n')
            
    est_sm.append(rslts.params['sm'])
    est_hn.append('-')
    std_sm.append(rslts.std_errors.sm)
    std_hn.append('-')
    pval_sm.append(rslts.pvalues.sm)
    pval_hn.append('-')
    num_obs.append(rslts.nobs)
            
    mean_crime = df.crimerate[df.cohort >= 1958][df.cohort <= 1962][df.highnumber == 0].mean()
    percent_change.append(100*rslts.params['sm']/mean_crime)
            
    #Col 4.
    rslts = regress(df=df, method='IV', cohort_range=[1958, 1962], cohort_dummies=cohorts[29: 33], controls='y')
            
    est_sm.append(rslts.params['sm'])
    est_hn.append('-')
    std_sm.append(rslts.std_errors.sm)
    std_hn.append('-')
    pval_sm.append(rslts.pvalues.sm)
    pval_hn.append('-')
    num_obs.append(rslts.nobs)
            
    mean_crime = df.crimerate[df.cohort >= 1958][df.cohort <= 1962][df.highnumber == 0].mean()
    percent_change.append(100*rslts.params['sm']/mean_crime)
    
    # Col 5.
    rslts = regress(df=df, method='OLS', cohort_range=[1929, 1965], cohort_dummies=cohorts[0: 36], controls='n')
            
    est_sm.append('-')
    est_hn.append(rslts.params['highnumber'])
    std_sm.append('-')
    std_hn.append(rslts.HC0_se['highnumber'])
    pval_sm.append('-')
    pval_hn.append(rslts.pvalues.highnumber)
    num_obs.append(rslts.nobs)
            
    wald = (rslts.params['highnumber']/(p1 - p2))
    mean_crime = df.crimerate[df.cohort >= 1929][df.cohort <= 1965][df.highnumber == 0].mean()
    percent_change.append(100*wald/mean_crime)
            
    # Col 6.
    rslts = regress(df=df, method='OLS', cohort_range=[1929, 1955], cohort_dummies=cohorts[0: 26], controls='n')
            
    est_sm.append('-')
    est_hn.append(rslts.params['highnumber'])
    std_sm.append('-')
    std_hn.append(rslts.HC0_se['highnumber'])
    pval_sm.append('-')
    pval_hn.append(rslts.pvalues.highnumber)
    num_obs.append(rslts.nobs)
            
    wald = (rslts.params['highnumber']/(p1 - p2))
    mean_crime = df.crimerate[df.cohort >= 1929][df.cohort <= 1955][df.highnumber == 0].mean()
    percent_change.append(100*wald/mean_crime)
            
    # Col 7.
    rslts = regress(df=df, method='OLS', cohort_range=[1958, 1965], cohort_dummies=cohorts[29: 36], controls='n')
            
    est_sm.append('-')
    est_hn.append(rslts.params['highnumber'])
    std_sm.append('-')
    std_hn.append(rslts.HC0_se['highnumber'])
    pval_sm.append('-')
    pval_hn.append(rslts.pvalues.highnumber)
    num_obs.append(rslts.nobs)
            
    wald = (rslts.params['highnumber']/(p1 - p2))
    mean_crime = df.crimerate[df.cohort >= 1958][df.cohort <= 1965][df.highnumber == 0].mean()
    percent_change.append(100*wald/mean_crime)
            
    return est_sm, est_hn, std_sm, std_hn, pval_sm, pval_hn, percent_change, num_obs

# Get table 4.
def table_4(df):
    '''
    Function returns table representing table 4 in Galiani et al. 2011.
    Arguments:
    df: data frame to use.
    '''
    #constant, highnumber, conscription, crimerate, malvinas, navy, origin, cohorts, districts, hn_malvinas, df = get_variables()
    # Get regression results.
    est_sm, est_hn, std_sm, std_hn, pval_sm, pval_hn, percent_change, num_obs = regressions_table_4(df)
    
    # Print table.
    print('\033[1m' 'Table 4 - Estimated Impact of Conscription on Crime Rates ' '\033[0m')
    print(128*'_')

    print('{:<15s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}'\
          .format('Cohort', '1958-1962', '', "1958-1962", '', "1958-1962", '', "1958-1962", '', "1929-1965", '', "1929-1955", '', \
                "1958-1965", '', ''))

    print('{:<15s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}'\
          .format('', '(1)', '', '(2)', '', '(3)', '', '(4)', '', '(5)', '', '(6)', '', \
                '(7)', '', ''))
    print(128*'_')


    for i in range(len(est_hn)):
        if i == 0:
            print('{:<15s}'.format("Draft Eligible"), end="")
        if type(est_hn[i]) == str:
            print('{:>13s}{:<3s}'.format('', ''), end="")
        else:   
            print('\033[1m' '{:>13.4f}{:<3s}' '\033[0m'.format(est_hn[i], significance(pval_hn[i])), end="")
    
    print('\n')

    for i in range(len(std_hn)):
        if i == 0:
            print('{:<15s}'.format(''), end="")
        if type(est_hn[i]) == str:
            print('{:>13s}{:<3s}'.format('', ''), end="")
        else:
            print('{:>13.4f}{:<3s}'.format(std_hn[i], ''), end="")
        
    print('\n')
        
    for i in range(len(est_sm)):
        if i == 0:
            print('{:<15s}'.format("Conscription"), end="")
        if type(est_sm[i]) == str:
            print('{:>13s}{:<3s}'.format('', ''), end="")
        else:   
            print('\033[1m' '{:>13.4f}{:<3s}' '\033[0m'.format(est_sm[i], significance(pval_sm[i])), end="")
    
    print('\n')

    for i in range(len(std_sm)):
        if i == 0:
            print('{:<15s}'.format(''), end="")
        if type(est_sm[i]) == str:
            print('{:>13s}{:<3s}'.format('', ''), end="")
        else:
            print('{:>13.4f}{:<3s}'.format(std_sm[i], ''), end="")
    
    print('\n')

    for i in range(len(percent_change)):
        if i == 0:
            print('{:<15s}'.format('Percent change'), end="")
        print('\033[1m' '{:>13.2f}{:<3s}' '\033[0m'.format(percent_change[i], ''), end="")
    
    print('\n')

    print('{:<15s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}'\
          .format('Controls', 'No', '', 'Yes', '', 'No', '', 'Yes', '', 'No', '', 'No', '', 'No', '', ''))
    
    print('\n')

    for i in range(len(num_obs)):
        if i == 0:
            print('{:<15s}'.format('Observations'), end="")
        print('\033[1m' '{:>13.0f}{:<3s}' '\033[0m'.format(num_obs[i], ''), end="")

    print('\n')
    
    print('{:<15s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}'\
          .format('Method', 'OLS', '', 'OLS', '', '2SLS', '', '2SLS', '', 'OLS', '', 'OLS', '', 'OLS', '', ''))

    print(128*'_')
    print('Notes: Robust standard errors are shown below estimates. The level of observation is the cohort-ID number combination. All models')
    print('All models include cohort dummies. The models in columns 2 and 4 include controls for origin (naturalized or indigenous) and dis-')
    print('trict (the country is divided in 24 districts). In 2SLS models, the instrument for Conscription is Draft Eligible. Percent change')
    print('for 2SLS models is calculated as 100 × Estimate/mean crime rate of draft-ineligible men. For intention-to-treat models, percent  ')
    print('change is reported as 100 × Wald estimate/mean crime rate of draft-ineligible men, where the Wald estimate is calculated as ITT  ')
    print('estitimate/(p1 − p2), where p1 is the probability of serving in the military among those that are draft-eligible, and p2 is the  ')
    print('probability of serving in the military among those that are not draft-eligible (since we do not have information on compliance   ')
    print('rates outside the cohorts of 1958 to 1962, in all cases we use the compliance rates for this period).                            ')
    print('*** Significant at 1 percent level.')
    print(' ** Significant at 5 percent level.')