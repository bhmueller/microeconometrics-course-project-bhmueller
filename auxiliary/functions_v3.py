# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 02:02:36 2020

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
    
# Table 6.
def table_6():
    constant, highnumber, conscription, crimerate, malvinas, navy, origin, cohorts, districts, hn_malvinas, df = get_variables()
    df_reg = df[df.cohort > 1957][df.cohort < 1963].copy()

    reg_sm = []
    pval_sm = []
    std_sm = []
    change_sm = []
    for crime in ['arms', 'property', 'sexual', 'murder', 'threat', 'drug', 'whitecollar']:
        # Dependent variable: crime
        y = df_reg.loc[:, crime]
        rslts = IV2SLS(y, df_reg[constant + cohorts[29: 33]], df_reg['sm'], df_reg['highnumber']).fit()
        
        # Get percent change.
        ineligible_mean = df_reg[crime][df_reg.highnumber == 0].mean()  # Mean crime rate of ineligible ID-groups by type of crime.
        change = ((rslts.params.sm)/(ineligible_mean))
    
        reg_sm.append(rslts.params.sm)
        pval_sm.append(rslts.pvalues.sm)
        std_sm.append(rslts.std_errors.sm)
        change_sm.append(change)
        
    print('\033[1m' 'Table 6 - Estimated Impact of Conscription on Crime rates, by Type of Crime' '\033[0m')
    print(128*'_')
    # Header.
    print('{:<15s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}'\
          .format('Dependent Var.', 'Weapons', "", 'Property', "", 'Sexual Attack', "", 'Murder', "", 'Threat', "", 'Drug Traff.', "", \
                  'White Collar', '', ''))
    print('{:<15s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}'\
          .format('Cohort', '1958-1962', '', "1958-1962", '', "1958-1962", '', "1958-1962", '', "1958-1962", '', "1958-1962", '', \
                  "1958-1962", '', ''))
    print('{:<15s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}'\
          .format('', '(1)', '', '(2)', '', '(3)', '', '(4)', '', '(5)', '', '(6)', '', \
                  '(7)', '', ''))
    print(128*'_')


    for i in range(len(reg_sm)):
        if i == 0:
            print('{:<15s}'.format("Conscription"), end="")
        print('\033[1m' '{:>13.5f}{:<3s}' '\033[0m'.format(reg_sm[i], significance(pval_sm[i])), end="")
    
    print('\n')

    for i in range(len(std_sm)):
        if i == 0:
            print('{:<15s}'.format(''), end="")
        print('{:>13.5f}{:<3s}'.format(std_sm[i], ''), end="")
    
    print('\n')

    for i in range(len(change_sm)):
        if i == 0:
            print('{:<15s}'.format('Percent change'), end="")
        print('\033[1m' '{:>13.2f}{:<3s}' '\033[0m'.format(change_sm[i], ''), end="")
    
    print('\n')

    print('{:<15s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}'\
          .format('Observations', '5000', '', '5000', '', '5000', '', '5000', '', '5000', '', '5000', '', \
                  '5000', '', ''))

    print('{:<15s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}'\
          .format('Method', '2SLS', '', '2SLS', '', '2SLS', '', '2SLS', '', '2SLS', '', '2SLS', '', '2SLS', '', ''))
    print(128*'_')
    print('Notes: Robust standard errors are shown below estimates. The level of observation is the cohort-ID number combination. All mo-')
    print('dels include cohort dummies. The instrument for Consription is Draft eligible. Percent change is calculated as 100*Estimate/mean')
    print('dependent variable of draft-ineligible men.')
    print('*** Significant at 1 percent level.')
    print(' ** Significant at 5 percent level.')

# Table 5.
def table_5():
    
    # Get data.
    constant, highnumber, conscription, crimerate, malvinas, navy, origin, cohorts, districts, hn_malvinas, df = get_variables()
    
    # Lists for regression results.
    est_hn = []
    std_hn = []
    est_mal = []
    std_mal = []
    est_na = []
    std_na = []
    n_obs = []
    pval_hn = []
    pval_mal = []
    pval_na = []
    
    # Col 1.
    
    df_29_65 = df[df.cohort > 1928][df.cohort < 1966][highnumber + hn_malvinas + constant + cohorts[0:36] + crimerate].copy().dropna(axis=0)
    X = df_29_65[highnumber + hn_malvinas + constant + cohorts[0:36]].copy()
    y = df_29_65.loc[:, 'crimerate']
    rslts = sm.OLS(y, X).fit()
    
    est_hn.append(rslts.params.highnumber)
    std_hn.append(rslts.HC0_se.highnumber)
    est_mal.append(rslts.params.hn_malvinas)
    std_mal.append(rslts.HC0_se.hn_malvinas)
    est_na.append('')
    std_na.append('')
    n_obs.append(rslts.nobs)
    pval_hn.append(rslts.pvalues.highnumber)
    pval_mal.append(rslts.pvalues.hn_malvinas)
    pval_na.append('')
    
    # Col 2.
    
    df_58_65 = df[df.cohort > 1957][df.cohort < 1966][highnumber + hn_malvinas + constant + cohorts[29:36] + crimerate].copy().dropna(axis=0)
    X = df_58_65[highnumber + hn_malvinas + constant + cohorts[29:36]].copy()
    y = df_58_65.loc[:, 'crimerate']
    rslts = sm.OLS(y, X).fit()
    
    est_hn.append(rslts.params.highnumber)
    std_hn.append(rslts.HC0_se.highnumber)
    est_mal.append(rslts.params.hn_malvinas)
    std_mal.append(rslts.HC0_se.hn_malvinas)
    est_na.append('')
    std_na.append('')
    n_obs.append(rslts.nobs)
    pval_hn.append(rslts.pvalues.highnumber)
    pval_mal.append(rslts.pvalues.hn_malvinas)
    pval_na.append('')
    
    # Col 3.
    
    df_29_65 = df[df.cohort > 1928][df.cohort < 1966][highnumber + navy + constant + cohorts[0:36] + crimerate].copy().dropna(axis=0)
    X = df_29_65[highnumber + navy + constant + cohorts[0:36]].copy()
    y = df_29_65.loc[:, 'crimerate']
    rslts = sm.OLS(y, X).fit()
    
    est_hn.append(rslts.params.highnumber)
    std_hn.append(rslts.HC0_se.highnumber)
    est_mal.append('')
    std_mal.append('')
    est_na.append(rslts.params.navy)
    std_na.append(rslts.HC0_se.navy)
    n_obs.append(rslts.nobs)
    pval_hn.append(rslts.pvalues.highnumber)
    pval_mal.append('')
    pval_na.append(rslts.pvalues.navy)
    
    # Col 4.
    df_58_65 = df[df.cohort > 1957][df.cohort < 1966][highnumber + navy + constant + cohorts[29:36] + crimerate].copy().dropna(axis=0)
    X = df_58_65[highnumber + navy + constant + cohorts[29:36]].copy()
    y = df_58_65.loc[:, 'crimerate']
    rslts = sm.OLS(y, X).fit()
    
    est_hn.append(rslts.params.highnumber)
    std_hn.append(rslts.HC0_se.highnumber)
    est_mal.append('')
    std_mal.append('')
    est_na.append(rslts.params.navy)
    std_na.append(rslts.HC0_se.navy)
    n_obs.append(rslts.nobs)
    pval_hn.append(rslts.pvalues.highnumber)
    pval_mal.append('')
    pval_na.append(rslts.pvalues.navy)
    
    # Print table.
    print('\033[1m' 'Table 5 - Estimated Impact of Conscription on Crime Rates for Peacetime' '\033[0m')
    print('\033[1m' 'versus Wartime Service and 1-Year versus 2-Year Service' '\033[0m')
    print(80*'_')

    print('{:<15s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}'\
          .format('Cohort', '1929-1965', '', "1958-1965", '', "1929-1965", '', "1958-1965", '', ''))

    print('{:<15s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}'\
          .format('', '(1)', '', '(2)', '', '(3)', '', '(4)', '', ''))
    print(80*'_')
    
    # Draft Eligible.

    for i in range(len(est_hn)):
        if i == 0:
            print('{:<15s}'.format("Draft Eligible"), end="")
        if type(est_hn[i]) == str:
            print('{:>13s}{:<3s}'.format('', ''), end="")
        else:   
            print('\033[1m' '{:>13.5f}{:<3s}' '\033[0m'.format(est_hn[i], significance(pval_hn[i])), end="")
    
    print('\n')

    for i in range(len(std_hn)):
        if i == 0:
            print('{:<15s}'.format(''), end="")
        if type(est_hn[i]) == str:
            print('{:>13s}{:<3s}'.format('', ''), end="")
        else:
            print('{:>13.4f}{:<3s}'.format(std_hn[i], ''), end="")
        
    print('\n')
    
    # Falkland War.
        
    for i in range(len(est_mal)):
        if i == 0:
            print('{:<15s}'.format("War Eligible"), end="")
        if type(est_mal[i]) == str:
            print('{:>13s}{:<3s}'.format('', ''), end="")
        else:   
            print('\033[1m' '{:>13.4f}{:<3s}' '\033[0m'.format(est_mal[i], significance(pval_mal[i])), end="")
    
    print('\n')

    for i in range(len(std_mal)):
        if i == 0:
            print('{:<15s}'.format(''), end="")
        if type(est_mal[i]) == str:
            print('{:>13s}{:<3s}'.format('', ''), end="")
        else:
            print('{:>13.4f}{:<3s}'.format(std_mal[i], ''), end="")
            
    print('\n')
    
    # Navy
    
    for i in range(len(est_na)):
        if i == 0:
            print('{:<15s}'.format("Navy Eligible"), end="")
        if type(est_na[i]) == str:
            print('{:>13s}{:<3s}'.format('', ''), end="")
        else:   
            print('\033[1m' '{:>13.4f}{:<3s}' '\033[0m'.format(est_na[i], significance(pval_na[i])), end="")
    
    print('\n')

    for i in range(len(std_na)):
        if i == 0:
            print('{:<15s}'.format(''), end="")
        if type(est_na[i]) == str:
            print('{:>13s}{:<3s}'.format('', ''), end="")
        else:
            print('{:>13.4f}{:<3s}'.format(std_na[i], ''), end="")
    
    
    print('\n')
    
    # Obs.

    for i in range(len(n_obs)):
        if i == 0:
            print('{:<15s}'.format('Observations'), end="")
        print('\033[1m' '{:>13.0f}{:<3s}' '\033[0m'.format(n_obs[i], ''), end="")

    print('\n')
    
    # Method.
    
    print('{:<15s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}'\
          .format('Method', 'OLS', '', 'OLS', '', 'OLS', '', 'OLS', '', ''))

    print(80*'_')
    print('Notes: Robust standard errors are shown below estimates. The level of observation')
    print('is the cohort-ID number combination. War Eligible is a dummy that takes the value')
    print('of one for the draft eligible from cohorts of 1962 and 1963. Navy Eligible is a  ')
    print('dummy variable that takes the value of one for those ID numbers eligible to serve')
    print('in the Navy. All models include cohort dummies.')
    print('** Significant at 5 percent level.')
    print(' * Significant at 10 percent level.')
    
# Extension table 4.
def extension_table_4():
    # Set up df etc.
    constant, highnumber, conscription, crimerate, malvinas, navy, origin, cohorts, districts, hn_malvinas, df = get_variables()
    
    # Lists to fill with estimates.
    reg_sm = []
    pval_sm = []
    std_sm = []
    change_sm = []
    
    # No controls.
    years = list(range(1958, 1963, 1))
    for i in years:
        df_reg = df[df.cohort == i].copy()
        # Dependent variable: conscription.
        rslts = IV2SLS(df_reg.loc[:, 'crimerate'], df_reg[constant], df_reg['sm'], df_reg['highnumber']).fit()
        
        # Get percent change.
        ineligible_mean = df_reg['crimerate'][df_reg.highnumber == 0].mean()  # Mean crime rate of ineligible ID-groups by type of crime.
        change = (100*(rslts.params.sm)/(ineligible_mean))
    
        reg_sm.append(rslts.params.sm)
        pval_sm.append(rslts.pvalues.sm)
        std_sm.append(rslts.std_errors.sm)
        change_sm.append(change)
    
    width = 97    
    print('\033[1m' 'Table E.4 - Estimated Impact of Conscription on Crime rates, for each Core Cohort Separately' '\033[0m')
    print('\033[1m' '(Dependent Variable: Crime Rate)' '\033[0m')
    print(width*'_')
    
    
    for i in years:
        if i == 1958:
            print('{:<15s}'.format('Cohort'), end="")
        print('{:>13.0f}{:<3s}'.format(i, ''), end="")
    
    print('\n')
    
    print('{:<15s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}'\
          .format('', '(1)', '', '(2)', '', '(3)', '', '(4)', '', '(5)', '', ''))
    print(width*'_')


    for i in range(len(reg_sm)):
        if i == 0:
            print('{:<15s}'.format("Conscription"), end="")
        print('\033[1m' '{:>13.5f}{:<3s}' '\033[0m'.format(reg_sm[i], significance(pval_sm[i])), end="")
    
    print('\n')

    for i in range(len(std_sm)):
        if i == 0:
            print('{:<15s}'.format(''), end="")
        print('{:>13.5f}{:<3s}'.format(std_sm[i], ''), end="")
    
    print('\n')

    for i in range(len(change_sm)):
        if i == 0:
            print('{:<15s}'.format('Percent change'), end="")
        print('\033[1m' '{:>13.2f}{:<3s}' '\033[0m'.format(change_sm[i], ''), end="")
    
    print('\n')

    print('{:<15s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}'\
          .format('Observations', '1000', '', '1000', '', '1000', '', '1000', '', '1000', '', ''))

    print('{:<15s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}'\
          .format('Method', '2SLS', '', '2SLS', '', '2SLS', '', '2SLS', '', '2SLS', '', ''))
    print(width*'_')
    print('Notes: Robust standard errors are shown below estimates. The level of observation is the cohort-')
    print('ID number combination. The instrument for Conscription is Draft Eligible. Percent change is cal-')
    print('culated as 100 × Estimate/mean crime rate of draft-ineligible men.')
    print('** Significant at 5 percent level.')
    print(' * Significant at 10 percent level.')

# Extension table 4 with controls.
def extension_table_4_controls():
    # Set up df etc.
    constant, highnumber, conscription, crimerate, malvinas, navy, origin, cohorts, districts, hn_malvinas, df = get_variables()
    
    # Lists to fill with estimates.
    reg_sm = []
    pval_sm = []
    std_sm = []
    change_sm = []
    
    # No controls.
    years = list(range(1958, 1963, 1))
    for i in years:
        df_reg = df[df.cohort == i].copy()
        # Dependent variable: conscription.
        rslts = IV2SLS(df_reg.loc[:, 'crimerate'], df_reg[constant + districts + origin], df_reg['sm'], df_reg['highnumber']).fit()
        
        # Get percent change.
        ineligible_mean = df_reg['crimerate'][df_reg.highnumber == 0].mean()  # Mean crime rate of ineligible ID-groups by type of crime.
        change = (100*(rslts.params.sm)/(ineligible_mean))
    
        reg_sm.append(rslts.params.sm)
        pval_sm.append(rslts.pvalues.sm)
        std_sm.append(rslts.std_errors.sm)
        change_sm.append(change)
    
    width = 97    
    print('\033[1m' 'Table E.4 - Estimated Impact of Conscription on Crime rates, by Cohort with Controls' '\033[0m')
    print('\033[1m' '(Dependent Variable: Crime Rate)' '\033[0m')
    print(width*'_')
    
    
    for i in years:
        if i == 1958:
            print('{:<15s}'.format('Cohort'), end="")
        print('{:>13.0f}{:<3s}'.format(i, ''), end="")
    
    print('\n')
    
    print('{:<15s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}'\
          .format('', '(1)', '', '(2)', '', '(3)', '', '(4)', '', '(5)', '', ''))
    print(width*'_')


    for i in range(len(reg_sm)):
        if i == 0:
            print('{:<15s}'.format("Conscription"), end="")
        print('\033[1m' '{:>13.5f}{:<3s}' '\033[0m'.format(reg_sm[i], significance(pval_sm[i])), end="")
    
    print('\n')

    for i in range(len(std_sm)):
        if i == 0:
            print('{:<15s}'.format(''), end="")
        print('{:>13.5f}{:<3s}'.format(std_sm[i], ''), end="")
    
    print('\n')

    for i in range(len(change_sm)):
        if i == 0:
            print('{:<15s}'.format('Percent change'), end="")
        print('\033[1m' '{:>13.2f}{:<3s}' '\033[0m'.format(change_sm[i], ''), end="")
    
    print('\n')

    print('{:<15s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}'\
          .format('Observations', '1000', '', '1000', '', '1000', '', '1000', '', '1000', '', ''))

    print('{:<15s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}'\
          .format('Method', '2SLS', '', '2SLS', '', '2SLS', '', '2SLS', '', '2SLS', '', ''))
    print(width*'_')
    print('Notes: Robust standard errors are shown below estimates. The level of observation is the cohort-')
    print('ID number combination. The instrument for Conscription is Draft Eligible. Percent change is cal-')
    print('culated as 100 × Estimate/mean crime rate of draft-ineligible men. Models include district and')
    print('origin dummies.')
    print('** Significant at 5 percent level.')
    print(' * Significant at 10 percent level.')
    
# Table 2.
def table_2():
    
    # Get data.
    constant, highnumber, conscription, crimerate, malvinas, navy, origin, cohorts, districts, hn_malvinas, df = get_variables()

    equal_var = False
    nan_policy = 'propagate'
    # Get years (=cohorts) for loop.
    years = list(range(1958, 1963, 1))
    
    # Lists to fill with statistics for each subgroup.
    t_arg = []
    pval_arg = []
    
    t_ind = []
    pval_ind = []
    
    t_nat = []
    pval_nat = []
    
    # Argentine.
    for i in years:
        a = df.argentine[df.cohort == i][df.highnumber == 0].copy()
        b = df.argentine[df.cohort == i][df.highnumber == 1].copy()
        ttest = stats.ttest_ind(a, b, axis=0, equal_var=equal_var, nan_policy=nan_policy)
        t_arg.append(ttest.statistic)
        pval_arg.append(ttest.pvalue)
        
    # Indigenous.
    for i in years:
        a = df.indigenous[df.cohort == i][df.highnumber == 0].copy()
        b = df.indigenous[df.cohort == i][df.highnumber == 1].copy()
        ttest = stats.ttest_ind(a, b, axis=0, equal_var=equal_var, nan_policy=nan_policy)
        t_ind.append(ttest.statistic)
        pval_ind.append(ttest.pvalue) 
        
    # Naturalized.
    for i in years:
        a = df.naturalized[df.cohort == i][df.highnumber == 0].copy()
        b = df.naturalized[df.cohort == i][df.highnumber == 1].copy()
        ttest = stats.ttest_ind(a, b, axis=0, equal_var=equal_var, nan_policy=nan_policy)
        t_nat.append(ttest.statistic)
        pval_nat.append(ttest.pvalue)
    
    # Print table.
    width = 110    
    print('\033[1m' 'Table 2 - Differences in Pre-Treatment Characteristics by Birth Cohort and Eligibility Group' '\033[0m')
    print('\033[1m' 'Differences by Cohort (draft exempt - draft eligible)' '\033[0m')
    print(width*'_')
    
    
    for i in years:
        if i == 1958:
            print('{:<30s}'.format('Cohort'), end="")
        print('{:>13.0f}{:<3s}'.format(i, ''), end="")
    
    print('\n')
    
    print(width*'_')
    
    # Argentine
    for i in range(len(t_arg)):
        if i == 0:
            print('{:<30s}'.format('Argentine-born, not indigenous'), end='')
        print('\033[1m' '{:>13.5f}{:<3s}' '\033[0m'.format(t_arg[i], significance(pval_arg[i])), end="")
    
    print('\n')

    for i in range(len(pval_arg)):
        if i == 0:
            print('{:<30s}'.format(''), end="")
        print('{:>13.5f}{:<3s}'.format(pval_arg[i], ''), end="")
    
    print('\n')
    
    # Indig.
    for i in range(len(t_ind)):
        if i == 0:
            print('{:<30s}'.format('Argentine-born, indigenous'), end='')
        print('\033[1m' '{:>13.5f}{:<3s}' '\033[0m'.format(t_ind[i], significance(pval_ind[i])), end="")
    
    print('\n')

    for i in range(len(pval_ind)):
        if i == 0:
            print('{:<30s}'.format(''), end="")
        print('{:>13.5f}{:<3s}'.format(pval_ind[i], ''), end="")
    
    print('\n')
    
    # Naturalized.
    
    for i in range(len(t_nat)):
        if i == 0:
            print('{:<30s}'.format('Born abroad, naturalized'), end='')
        print('\033[1m' '{:>13.5f}{:<3s}' '\033[0m'.format(t_nat[i], significance(pval_nat[i])), end="")
    
    print('\n')

    for i in range(len(pval_nat)):
        if i == 0:
            print('{:<30s}'.format(''), end="")
        print('{:>13.5f}{:<3s}'.format(pval_nat[i], ''), end="")
    
    print('\n')
    
    print(width*'_')
    print('Notes: P-values are shown below test statistics. The level of observation is the cohort-ID number combination.')
    
# Table 3.
# Define data set.
def table_3():
    # Get variables.
    constant, highnumber, conscription, crimerate, malvinas, navy, origin, cohorts, districts, hn_malvinas, df = get_variables()
    
    df_regression = df[df.cohort > 1957][df.cohort < 1963].copy()
    years_tab3 = list(range(1957, 1963, 1))  # 1957 included to get 6 instead of 5 columns (1 repeated cross-section + 5 separate cross-sections)
    estim_hn = []  # List for estimates for 'highnumber'.
    estim_const = []  # List for estimates for 'constant'.
    std_hn = []  # List for standard errors for 'highnumber'.
    std_const = []
    pval_hn = []
    pval_const = []
    
    for i in years_tab3:
    
        if i < 1958:
        
            X = df_regression[highnumber + cohorts[29: 33] + constant].copy()
            y = df_regression.loc[:, 'sm']
            # Fit OLS model.
            rslts = sm.OLS(y, X, cov_type='cluster').fit()
        
            #df_regression['fitted_conscription'] = rslts.fittedvalues
        
            estim_hn.append(rslts.params['highnumber'])
            std_hn.append(rslts.HC0_se['highnumber'])
            estim_const.append(rslts.params['constant'])
            std_const.append(rslts.HC0_se['constant'])
            pval_hn.append(rslts.pvalues['highnumber'])
            pval_const.append(rslts.pvalues['constant'])
        
        else:
        
            df_regression = df[df.cohort == i].copy()
            X = df_regression[['highnumber', 'constant']].copy()
            y = df_regression.loc[:, 'sm']
            rslts = sm.OLS(y, X).fit()
        
            estim_hn.append(rslts.params['highnumber'])
            std_hn.append(rslts.HC0_se['highnumber'])
            estim_const.append(rslts.params['constant'])
            std_const.append(rslts.HC0_se['constant'])
            pval_hn.append(rslts.pvalues['highnumber'])
            pval_const.append(rslts.pvalues['constant'])
            
    print('\033[1m' 'Table 3 - First Stage by Birth Cohort' '\033[0m')
    print('Dependent Variable: Conscription')
    print(112*'_')
    # Header.
    print('{:<15s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}'\
          .format('Cohort', '1958-1962', '', "1958", '', "1959", '', "1960", '', "1961", '', "1962", '', \
                 '', ''))
    print('{:<15s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}'\
          .format('', '(1)', '', '(2)', '', '(3)', '', '(4)', '', '(5)', '', '(6)', '', \
                 '', ''))
    print(112*'_')


    for i in range(len(estim_hn)):
        if i == 0:
            print('{:<15s}'.format('Draft Eligible'), end="")
        print('\033[1m' '{:>13.4f}{:<3s}' '\033[0m'.format(estim_hn[i], significance(pval_hn[i])), end="")
    
    print('\n')

    for i in range(len(std_hn)):
        if i == 0:
            print('{:<15s}'.format(''), end="")
        print('{:>13.4f}{:<3s}'.format(std_hn[i], ''), end="")
    
    print('\n')

    for i in range(len(estim_const)):
        if i == 0:
            print('{:<15s}'.format("Constant"), end="")
        print('\033[1m' '{:>13.4f}{:<3s}' '\033[0m'.format(estim_const[i], significance(pval_const[i])), end="")
    
    print('\n')

    for i in range(len(std_const)):
        if i == 0:
            print('{:<15s}'.format(''), end="")
        print('{:>13.4f}{:<3s}'.format(std_const[i], ''), end="")
    
    print('\n')

    print('{:<15s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}'\
          .format('Observations', '5,000', '', '1,000', '', '1,000', '', '1,000', '', '1,000', '', '1,000', '', \
                '', ''))

    print('{:<15s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}'\
          .format('Method', 'OLS', '', 'OLS', '', 'OLS', '', 'OLS', '', 'OLS', '', 'OLS', '', \
                 '', ''))
    print(112*'_')
    print('Notes: Robust standard errors are shown below estimates. The level of observation is the cohort-ID number combi-')
    print('nation. Column 1 includes cohort dummies.')
    print('*** Significant at 1 percent level.')

# Table 7.
def table_7_IV():
    # Get variables.
    constant, highnumber, conscription, crimerate, malvinas, navy, origin, cohorts, districts, hn_malvinas, df = get_variables()
    formal = ['formal']
    unemployment = ['unemployment']
    income = ['income'] 
    
    est_sm = []
    std_sm = []
    pval_sm = []
    percent_change = []
    n_obs = []
    
    # Define data set.
    df_reg = df[(df.cohort > 1957) & (df.cohort < 1963)].copy()
    
    for outcome in ['formal', 'unemployment', 'income']:
        y = df_reg.loc[:, outcome]
        rslts = IV2SLS(y, df_reg[constant + cohorts[29: 33]], df_reg['sm'], df_reg['highnumber']).fit()
        
        est_sm.append(rslts.params.sm)
        std_sm.append(rslts.std_errors.sm)
        pval_sm.append(rslts.pvalues.sm)
        n_obs.append(rslts.nobs)
        mean = df_reg[outcome][df_reg.highnumber == 0].mean()
        change = (100*(rslts.params['sm'])/(mean))
        percent_change.append(change)
    
    # Print header.
    width = 95
    print('\033[1m' 'Table 7 - Estimated Impact of Conscription on Labour Market Outcomes' '\033[0m')
    print(width*'_')
    
    print('{:<15s}{:>17s}{:<3s}{:>17s}{:<3s}{:>17s}{:<3s}{:>17s}'\
          .format('Dependent Var.', 'Formal job market', '', 'Unemployment rate', '', 'Earnings','', ''))
    print(width*'_')
    print('{:<15s}{:>17s}{:<3s}{:>17s}{:<3s}{:>17s}{:<3s}{:>17s}'\
          .format('', '1958-1962', '', '1958-1962', '', '1958-1962', '', ''))
    print('{:<15s}{:>17s}{:<3s}{:>17s}{:<3s}{:>17s}{:<3s}{:>17s}'\
          .format('', '(1)', '', '(2)', '', '(3)', '', ''))
    print(width*'_')

    
    # Conscription.
    for i in range(len(est_sm)):
        if i == 0:
            print('{:<15s}'.format('Conscription'), end="")
        if type(est_sm[i]) == str:
            print('{:>17s}{:<3s}'.format('', ''), end="")
        else:
            print('\033[1m' '{:>17.5f}{:<3s}' '\033[0m'.format(est_sm[i], significance(pval_sm[i])), end="")
    print('\n')
    
    # Standard errors.
    for i in range(len(std_sm)):
        if i == 0:
            print('{:<15s}'.format(''), end="")
        if type(est_sm[i]) == str:
            print('{:>17s}{:<3s}'.format(''), end="")
        else:
            print('\033[1m' '{:>17.4f}{:<3s}' '\033[0m'.format(std_sm[i], ''), end="")
    print('\n')
                  
    # Percent change.
    for i in range(len(percent_change)):
        if i == 0:
            print('{:<15s}'.format('Percent change'), end="")
        print('{:>17.2f}{:<3s}'.format(percent_change[i], ''), end="")
    print('\n')
    
    # Number of observations.
    for i in range(len(n_obs)):
        if i == 0:
            print('{:<15s}'.format('Observations'), end="")
        print('{:>17.0f}{:<3s}'.format(n_obs[i], ''), end="")
    print('\n')
    
    print('{:<15s}{:>17s}{:<3s}{:>17s}{:<3s}{:>17s}{:<3s}{:>17s}'\
          .format('', '2SLS', '', '2SLS', '', '2SLS', '', ''))
    print(width*'_')
    
    print('Notes: Robust standard errors are shown below estimates. The level of observation is the cohort-')
    print('ID number combination. Participation in the formal job market is as of 2004. Unemployment rates ')
    print('and earnings are as of 2003. Earnings are hourly earnings in Argentine pesos. All models include')
    print('cohort dummies. The instrument for Conscription is Draft Eligible. Percent change is calculated ')
    print('as 100 × Estimate/mean dependent variable of draft-ineligible men.')

# Table B.1.
def table_B_1():
    '''
    Gives summary statistics for the core cohorts 1958-1962.
    '''
    
    path = ('data/baseB.dta')
    baseb = pd.read_stata(path)
    
    # Get variables.
    constant, highnumber, conscription, crimerate, malvinas, navy, origin, cohorts, districts, hn_malvinas, df = get_variables()
    formal = ['formal']
    unemployment = ['unemployment']
    income = ['income']
    arms = ['arms']
    sexual = ['sexual']
    whitecollar = ['whitecollar']
    
    df_reg = df[(df.cohort > 1957) & (df.cohort < 1963)].copy()
    
    core_variables = [highnumber, conscription, crimerate, formal, unemployment, income, arms, sexual, whitecollar, navy, hn_malvinas]
    #var_names = {['sm']: 'Conscription', ['crimerate']: 'Crime rate', ['formal']: 'Formal job market', ['unemployment']:'Unemployment rate', \
    #             ['earnings']: 'Earnings'}
    
    width = 110
    print('\033[1m' 'Table B.1 - Descriptive Statistics of Selected Variables of Interest for Male Birth Cohorts 1958 to 1962' '\033[0m')
    print(width*'_')
    
    # Header: Mean, STD, etc.
    print('{:<17s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}'\
          .format('', '(1)', '', '(2)', '', '(3)', '', '(4)', '', '(5)', '', \
                  '', ''))
    print('{:<17s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}{:<3s}{:>13s}'\
          .format('', 'Number', '', 'Mean', '', 'St. dev.', '', 'Mean eligible', '', \
                  'Mean exempt', '', ''))
    print(width*'_')
    print('Variables: \n')
    for i in core_variables:
        
        var_list = []
        
        # Number mean.
        #stat = len(df_reg[i])
        #var_list.append(stat)

        stat = df_reg[i].mean()
        var_list.append(stat[0]*baseb.sizecohort[(df.cohort > 1957) & (df.cohort < 1963)].mean()*1000)
        
        # Mean.
        stat = df_reg[i].mean()
        var_list.append(stat[0])
        
        # Standard dev.
        stat = df_reg[i].std()
        var_list.append(stat[0])
        
        # Cond. mean.
        
        if i == highnumber:
            #stat = df_reg[i].mean()
            var_list.append(1)
            
            #stat = df_reg[i].mean()
            #var_list.append(stat[0]*baseb.sizecohort[(df.cohort > 1957) & (df.cohort < 1963)].mean()*1000)
        
            var_list.append(0)
        
            #var_list.append((1-stat[0])*baseb.sizecohort[(df.cohort > 1957) & (df.cohort < 1963)].mean()*1000)
            
        else:
            
            stat = df_reg[i][df_reg.highnumber == 1].mean()
            var_list.append(stat[0])
        
            #var_list.append(stat[0]*(df_reg[i].mean()*baseb.sizecohort[(df.cohort > 1957) & (df.cohort < 1963)].mean())*1000)
        
            stat = df_reg[i][df_reg.highnumber == 0].mean()
            var_list.append(stat[0])
        
            #var_list.append(stat[0]*(df_reg[i].mean()*baseb.sizecohort[(df.cohort > 1957) & (df.cohort < 1963)].mean())*1000)
        
        # t-test.
        #a = df_reg[i][df_reg.highnumber == 0].copy()
        #b = df_reg[i][df_reg.highnumber == 1].copy()
        #ttest = stats.ttest_ind(a, b, axis=0, equal_var=False, nan_policy='propagate')
        #var_list.append(ttest.pvalue[0])
        
        for d in range(len(var_list)):
            if d == 0:
                print('{:<17s}'.format(str(i)), end="")
            if var_list[d] >= 100:
                print('\033[1m' '{:>13.0f}{:<3s}' '\033[0m'.format(var_list[d], ''), end="")
            else:
                print('\033[1m' '{:>13.4f}{:<3s}' '\033[0m'.format(var_list[d], ''), end="")
        print('\n')
    print(width*'_')
    print('Notes: highnumber = draft eligible, sm = conscription, formal = formal job market participation, income = ')
    print('hourly earnings in Argentine pesos,arms = crimes involving usage of weapons, sexual = sexual attack, white-')
    print('collar = white collar crimes (such as fraud, scams, extortion), navy = eligible for navy, hn_malvinas = eli-')
    print('gible during Falklands War. The number in column 1 contains the mean number of individuals in the subgroup. ') 
    print('It was calculated by using the mean size of cohorts from 1958 to 1962, which is 236,656. Columns 4 and 5 con-')
    print('tain means conditional on draft eligible and draft exempt cohort-ID groups.')