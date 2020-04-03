import sys

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.distributions.mixture_rvs import mixture_rvs
from sklearn.linear_model import LinearRegression
from numpy.polynomial import chebyshev as cby

import random

import plotly
import plotly.graph_objects as go
plotly.offline.init_notebook_mode(connected=True)

def extract_coeffs(data, window_size, step, degree, start, stop, plot_pdfs = False, plot_coeffs = True):
    
    i = start
    window = window_size + start

    guide_xlist = []
    optimal_interval_total = []

    coefficient_matrix = np.empty(degree + 1)
    coefficient_matrix = np.reshape(coefficient_matrix,(-1,1))

    fig = go.Figure()

    while window < stop:

        sample = data[i:window]

        # KDE curve approximation
        kde = sm.nonparametric.KDEUnivariate(sample)
        kde.fit() # Estimate the densities

        cby_coefficients = cby.chebfit(kde.support, kde.density, degree)
        polynomial_fit = cby.chebval(kde.support, cby_coefficients)

        step_number = int((i-start)/step)

        #Add plot to series of plots
        fig.add_trace(go.Scatter(x=kde.support, y=kde.density,
                            mode='lines',
                            name= 'From = ' + str(i) + ' to = ' + str(window)))

        guide_xlist += [str(i) + ' - ' + str(window)]

        cby_coefficients = np.reshape(cby_coefficients,(-1,1))

        coefficient_matrix = np.concatenate([coefficient_matrix,cby_coefficients], axis=1)

        i += step
        window += step 
        
        optimal_interval_total += [[float(sample.min()),float(sample.max())]]

        #print(step_number)

    coefficient_matrix = coefficient_matrix[:,1:]
    
    if plot_pdfs == True:

        fig.update_layout(
            title_text='Chebyshev KDE approximation (Polynomial degree = ' + str(degree) + ')', # title of plot
            xaxis_title_text='Value', # xaxis label
            #yaxis_title_text='Count', # yaxis label
            bargap=0.2, # gap between bars of adjacent location coordinates
            bargroupgap=0.1 # gap between bars of the same location coordinates
        )

        fig.show()    
    
    if plot_coeffs == True:
        
        fig = go.Figure()

        for i in range(0,coefficient_matrix.shape[0]):

            fig.add_trace(go.Scatter(x=guide_xlist, y=coefficient_matrix[i,:],
                                mode='lines',
                                name= 'C' + str(i)))

        fig.update_layout(
            title_text= 'Chebyshev Coefficients', # title of plot
            xaxis_title_text='Value', # xaxis label
            #yaxis_title_text='Count', # yaxis label
            bargap=0.2, # gap between bars of adjacent location coordinates
            bargroupgap=0.1 # gap between bars of the same location coordinates
        )

        fig.show()   
    
    return coefficient_matrix, optimal_interval_total


def coefficients_regression(coefficient_matrix, n_steps_back, steps_ahead):

    new_coefficients = []

    for step in range(0,steps_ahead):

        new_coefficients_row = []

        for k in range(0, coefficient_matrix.shape[0]):

            X = np.arange(0, len(coefficient_matrix[k,-n_steps_back:]))
            y = coefficient_matrix[k,-n_steps_back:]

            #print(y)

            reg = LinearRegression().fit(X[:,None], y)
            new_coefficients_row += reg.predict(np.array([len(X)+step])[:,None]).tolist()


            #print("Next element")
            #print(reg.predict(np.array([len(X)+step])[:,None]).tolist())
            #print(" ")

        new_coefficients += [new_coefficients_row]

    # Save each step in a different column
    new_coefficients = np.array(new_coefficients).T
    
    return new_coefficients


def coefficients_regression(coefficient_matrix, n_steps_back, steps_ahead):
    
    for step in range(0,steps_ahead):

        new_coefficients_row = []

        for k in range(0, coefficient_matrix.shape[0]):

            X = np.arange(0, len(coefficient_matrix[k,-n_steps_back:]))
            y = coefficient_matrix[k,-n_steps_back:]

            #print(y)

            reg = LinearRegression().fit(X[:,None], y)
            new_coefficients_row += reg.predict(np.array([len(X)+1])[:,None]).tolist()


            #print("Next element")
            #print(reg.predict(np.array([len(X)+step])[:,None]).tolist())
            #print(" ")

        coefficient_matrix = np.concatenate([coefficient_matrix, np.array(new_coefficients_row)[:,None]], axis = 1)

        #new_coefficients += [new_coefficients_row]
        
    new_coefficients = coefficient_matrix[:,-steps_ahead:]
    
    return new_coefficients


def rejection_sampling(new_coefficients, intervals, degree, sample_size = [100], plot_pdfs = True, pdf_check = False):

    fig = go.Figure()
    sample_total = np.empty(0)

    for k in range(0, new_coefficients.shape[1]):
        
        interval = intervals[k]

        x = np.arange(interval[0], interval[1], (abs(interval[1]-interval[0]))/float(sample_size[k]))

        coeffs = np.array([[coef] for coef in new_coefficients[:,k]])
        polynomial_fit = cby.chebval(x, coeffs)
        
        polynomial_fit = polynomial_fit[0]
            
        for n in range(0,len(polynomial_fit)):
            #print(polynomial_fit[n])
            if polynomial_fit[n] < 0:
                polynomial_fit[n] = 0

        # PLOT pdf Chebyshev curve approximation
        fig.add_trace(go.Scatter(x=x, y=polynomial_fit, mode='lines',name='Step ahead number: ' + str(k+1)))

        sample = np.empty(sample_size[k])

        seed = 1

        np.random.seed(seed)

        #fx = f(x)
        max_fx = polynomial_fit.max()
        min_fx = polynomial_fit.min()

        i = 0

        while i < sample_size[k]:

            u1 = np.random.uniform(interval[0], interval[1])
            u2 = np.random.uniform(min_fx, max_fx)

            if u2 <= cby.chebval(u1, coeffs):

                sample[i] = u1
                i += 1

        sample_total = np.concatenate([sample_total, sample], axis = 0)
    
    if plot_pdfs == True:

        fig.update_layout(
            title_text='Chebyshev KDE approximation (Polynomial degree = ' + str(degree) + ')', # title of plot
            xaxis_title_text='Value', # xaxis label
            #yaxis_title_text='Count', # yaxis label
            bargap=0.2, # gap between bars of adjacent location coordinates
            bargroupgap=0.1 # gap between bars of the same location coordinates
        )
        
    if pdf_check == True:
        
        # KDE curve approximation
        kde_test = sm.nonparametric.KDEUnivariate(sample_total)
        kde_test.fit() # Estimate the densities

        cby_coefficients = cby.chebfit(kde_test.support, kde_test.density, degree)
        polynomial_fit = cby.chebval(kde_test.support, cby_coefficients)
        
        # Create traces
        fig.add_trace(go.Scatter(x=kde_test.support, y=kde_test.density,
                            mode='lines',
                            name='KDE of step prior to forecasting'))
        #fig.add_trace(go.Histogram(x = sample, nbinsx = 400,histnorm='percent', name = 'Histogram'))
        
    fig.show()
        
    return sample_total


def rejection_sampling_nruns(new_coefficients, intervals, sample_size = [100], plot_pdfs = True, pdf_check = False, n_runs = 20):
    
    n_sample_points = np.array(sample_size)[:new_coefficients.shape[1]].sum()
    
    control_sample = np.empty(n_sample_points)
    control_sample = control_sample[:,None]
    
    for n in range(0,n_runs):
    
        sample_total = np.empty(0)

        for k in range(0, new_coefficients.shape[1]):
            
            #print(k)

            interval = intervals[k]
            
            #print(interval[0])
            #print(interval[1])
            #print(sample_size[k])

            x = np.arange(interval[0], interval[1], (abs(interval[1]-interval[0]))/float(sample_size[k]))

            coeffs = np.array([[coef] for coef in new_coefficients[:,k]])
            polynomial_fit = cby.chebval(x, coeffs)

            polynomial_fit = polynomial_fit[0]

            for n in range(0,len(polynomial_fit)):
                #print(polynomial_fit[n])
                if polynomial_fit[n] < 0:
                    polynomial_fit[n] = 0

            sample = np.empty(sample_size[k])

            seed = 1

            np.random.seed(seed)

            #fx = f(x)
            max_fx = polynomial_fit.max()
            min_fx = polynomial_fit.min()

            i = 0

            while i < sample_size[k]:

                u1 = np.random.uniform(interval[0], interval[1])
                u2 = np.random.uniform(min_fx, max_fx)

                # collect a full sample here

                #print(u2)
                #print(f(u1))
                #print("")

                if u2 <= cby.chebval(u1, coeffs):

                    sample[i] = u1
                    i += 1

            sample_total = np.concatenate([sample_total, sample], axis = 0)
            
        random.shuffle(sample_total)
        control_sample = np.concatenate([control_sample, sample_total[:,None][:n_sample_points]], axis = 1)
    
    return control_sample

def solution_plot(data, sample_total, stop):
    
    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(x=data.index, y=data[0], mode='lines', name= 'Original time Series'))
    fig1.add_trace(go.Scatter(x=np.arange(stop,stop + len(sample_total)), y=sample_total, mode='lines', name= 'Forecast'))

    fig1.update_layout(
        title_text= 'Data and Forecast', # title of plot
        xaxis_title_text='Value', # xaxis label
        #yaxis_title_text='Count', # yaxis label
        #bargap=0.2, # gap between bars of adjacent location coordinates
        #bargroupgap=0.1 # gap between bars of the same location coordinates
    )

    fig1.show()