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
from typing import List, Tuple, Union

import plotly
import plotly.graph_objects as go
plotly.offline.init_notebook_mode(connected=True)

def extract_coeffs(data: pd.Series, window_size: int, step: int, degree: int, start: int, stop: int, 
                   plot_pdfs: bool = False, plot_coeffs: bool = True) -> tuple:
    """
    Extracts Chebyshev polynomial coefficients from data in sliding windows and optionally plots the results.
    
    Args:
        data (pd.Series): The input data series.
        window_size (int): The size of the sliding window.
        step (int): The step size for the sliding window.
        degree (int): The degree of the Chebyshev polynomial.
        start (int): The starting index for the analysis.
        stop (int): The stopping index for the analysis.
        plot_pdfs (bool): Flag to plot PDFs (default: False).
        plot_coeffs (bool): Flag to plot coefficients (default: True).
        
    Returns:
        tuple: A tuple containing the coefficient matrix and optimal intervals.
    """
    i = start
    window = window_size + start

    guide_xlist = []
    optimal_interval_total = []

    coefficient_matrix = np.empty((degree + 1, 0))

    fig = go.Figure()

    while window < stop:
        sample = data[i:window]

        # KDE curve approximation
        kde = sm.nonparametric.KDEUnivariate(sample)
        kde.fit()  # Estimate the densities

        cby_coefficients = cby.chebfit(kde.support, kde.density, degree)
        polynomial_fit = cby.chebval(kde.support, cby_coefficients)

        step_number = int((i - start) / step)

        # Add plot to series of plots
        fig.add_trace(go.Scatter(x=kde.support, y=kde.density,
                                 mode='lines',
                                 name=f'From = {i} to = {window}'))

        guide_xlist.append(f'{i} - {window}')
        cby_coefficients = cby_coefficients.reshape(-1, 1)
        coefficient_matrix = np.hstack((coefficient_matrix, cby_coefficients))

        i += step
        window += step

        optimal_interval_total.append([float(sample.min()), float(sample.max())])

    if plot_pdfs:
        fig.update_layout(
            title_text=f'Chebyshev KDE approximation (Polynomial degree = {degree})',  # title of plot
            xaxis_title_text='Value',  # x-axis label
            bargap=0.2,  # gap between bars of adjacent location coordinates
            bargroupgap=0.1  # gap between bars of the same location coordinates
        )
        fig.show()

    if plot_coeffs:
        fig = go.Figure()
        for i in range(coefficient_matrix.shape[0]):
            fig.add_trace(go.Scatter(x=guide_xlist, y=coefficient_matrix[i, :],
                                     mode='lines',
                                     name=f'C{i}'))

        fig.update_layout(
            title_text='Chebyshev Coefficients',  # title of plot
            xaxis_title_text='Window',  # x-axis label
            bargap=0.2,  # gap between bars of adjacent location coordinates
            bargroupgap=0.1  # gap between bars of the same location coordinates
        )
        fig.show()

    return coefficient_matrix, optimal_interval_total


def coefficients_regression(coefficient_matrix: np.ndarray, n_steps_back: int, steps_ahead: int) -> np.ndarray:
    """
    Perform linear regression on the Chebyshev coefficient matrix to predict future coefficients.

    Args:
        coefficient_matrix (np.ndarray): Matrix of Chebyshev coefficients with shape (n_coefficients, n_windows).
        n_steps_back (int): Number of steps back to consider for the regression model.
        steps_ahead (int): Number of future steps to predict.

    Returns:
        np.ndarray: Predicted coefficients for the specified steps ahead, shape (n_coefficients, steps_ahead).
    """
    new_coefficients = []

    for step in range(steps_ahead):
        new_coefficients_row = []

        for k in range(coefficient_matrix.shape[0]):
            X = np.arange(len(coefficient_matrix[k, -n_steps_back:])).reshape(-1, 1)
            y = coefficient_matrix[k, -n_steps_back:]

            reg = LinearRegression().fit(X, y)
            new_coefficients_row.append(reg.predict(np.array([[len(X) + step]]))[0])

        new_coefficients.append(new_coefficients_row)

    return np.array(new_coefficients).T


def coefficients_regression(coefficient_matrix: np.ndarray, n_steps_back: int, steps_ahead: int) -> np.ndarray:
    """
    Perform linear regression on the Chebyshev coefficient matrix to predict future coefficients.

    Args:
        coefficient_matrix (np.ndarray): Matrix of Chebyshev coefficients with shape (n_coefficients, n_windows).
        n_steps_back (int): Number of steps back to consider for the regression model.
        steps_ahead (int): Number of future steps to predict.

    Returns:
        np.ndarray: Predicted coefficients for the specified steps ahead, shape (n_coefficients, steps_ahead).
    """
    for step in range(steps_ahead):
        new_coefficients_row = []

        for k in range(coefficient_matrix.shape[0]):
            X = np.arange(len(coefficient_matrix[k, -n_steps_back:])).reshape(-1, 1)
            y = coefficient_matrix[k, -n_steps_back:]

            reg = LinearRegression().fit(X, y)
            new_coefficients_row.append(reg.predict(np.array([[len(X) + 1]]))[0])

        coefficient_matrix = np.hstack((coefficient_matrix, np.array(new_coefficients_row).reshape(-1, 1)))

    new_coefficients = coefficient_matrix[:, -steps_ahead:]
    
    return new_coefficients


def rejection_sampling(new_coefficients: np.ndarray, intervals: List[Tuple[float, float]], degree: int,
                       sample_size: List[int] = [100], plot_pdfs: bool = True, pdf_check: bool = False) -> np.ndarray:
    """
    Perform rejection sampling using Chebyshev polynomial fits to generate samples from estimated PDFs.

    Args:
        new_coefficients (np.ndarray): Array of new Chebyshev coefficients, shape (n_coefficients, steps_ahead).
        intervals (List[Tuple[float, float]]): List of tuples specifying the intervals for each step ahead.
        degree (int): Degree of the Chebyshev polynomial used for fitting.
        sample_size (List[int]): List specifying the number of samples for each step ahead.
        plot_pdfs (bool): Whether to plot the PDFs using Plotly. Default is True.
        pdf_check (bool): Whether to check the PDF using KDE. Default is False.

    Returns:
        np.ndarray: Array containing the generated samples.
    """
    fig = go.Figure()
    sample_total = np.empty(0)

    for k in range(new_coefficients.shape[1]):
        interval = intervals[k]
        x = np.arange(interval[0], interval[1], (interval[1] - interval[0]) / float(sample_size[k]))

        coeffs = new_coefficients[:, k].reshape(-1, 1)
        polynomial_fit = cby.chebval(x, coeffs).flatten()

        polynomial_fit[polynomial_fit < 0] = 0  # Ensure no negative values

        # Plot the Chebyshev curve approximation
        fig.add_trace(go.Scatter(x=x, y=polynomial_fit, mode='lines', name=f'Step ahead number: {k + 1}'))

        sample = np.empty(sample_size[k])
        np.random.seed(1)

        max_fx = polynomial_fit.max()
        min_fx = polynomial_fit.min()

        i = 0
        while i < sample_size[k]:
            u1 = np.random.uniform(interval[0], interval[1])
            u2 = np.random.uniform(min_fx, max_fx)
            if u2 <= cby.chebval(u1, coeffs):
                sample[i] = u1
                i += 1

        sample_total = np.concatenate([sample_total, sample], axis=0)

    if plot_pdfs:
        fig.update_layout(
            title=f'Chebyshev KDE approximation (Polynomial degree = {degree})',
            xaxis_title='Value',
            bargap=0.2,
            bargroupgap=0.1
        )
        
    if pdf_check:
        kde_test = sm.nonparametric.KDEUnivariate(sample_total)
        kde_test.fit()

        cby_coefficients = cby.chebfit(kde_test.support, kde_test.density, degree)
        polynomial_fit = cby.chebval(kde_test.support, cby_coefficients)
        
        fig.add_trace(go.Scatter(x=kde_test.support, y=kde_test.density, mode='lines', name='KDE of step prior to forecasting'))

    fig.show()
        
    return sample_total


def rejection_sampling_nruns(new_coefficients: np.ndarray, intervals: List[Tuple[float, float]], 
                             sample_size: List[int] = [100], plot_pdfs: bool = True, 
                             pdf_check: bool = False, n_runs: int = 20) -> np.ndarray:
    """
    Perform rejection sampling multiple times using Chebyshev polynomial fits to generate samples from estimated PDFs.

    Args:
        new_coefficients (np.ndarray): Array of new Chebyshev coefficients, shape (n_coefficients, steps_ahead).
        intervals (List[Tuple[float, float]]): List of tuples specifying the intervals for each step ahead.
        sample_size (List[int]): List specifying the number of samples for each step ahead.
        plot_pdfs (bool): Whether to plot the PDFs using Plotly. Default is True.
        pdf_check (bool): Whether to check the PDF using KDE. Default is False.
        n_runs (int): Number of runs to perform rejection sampling. Default is 20.

    Returns:
        np.ndarray: Array containing the generated samples for each run.
    """
    n_sample_points = np.array(sample_size)[:new_coefficients.shape[1]].sum()
    control_sample = np.empty((n_sample_points, 0))

    for _ in range(n_runs):
        sample_total = np.empty(0)

        for k in range(new_coefficients.shape[1]):
            interval = intervals[k]
            x = np.arange(interval[0], interval[1], (interval[1] - interval[0]) / float(sample_size[k]))

            coeffs = new_coefficients[:, k].reshape(-1, 1)
            polynomial_fit = cby.chebval(x, coeffs).flatten()
            polynomial_fit[polynomial_fit < 0] = 0  # Ensure no negative values

            sample = np.empty(sample_size[k])
            np.random.seed(1)

            max_fx = polynomial_fit.max()
            min_fx = polynomial_fit.min()

            i = 0
            while i < sample_size[k]:
                u1 = np.random.uniform(interval[0], interval[1])
                u2 = np.random.uniform(min_fx, max_fx)
                if u2 <= cby.chebval(u1, coeffs):
                    sample[i] = u1
                    i += 1

            sample_total = np.concatenate([sample_total, sample], axis=0)
            
        np.random.shuffle(sample_total)
        control_sample = np.concatenate([control_sample, sample_total[:n_sample_points, None]], axis=1)
    
    return control_sample

def solution_plot(data: Union[pd.DataFrame, pd.Series], sample_total: np.ndarray, stop: int) -> None:
    """
    Plot the original time series data and the forecasted values.

    Args:
        data (Union[pd.DataFrame, pd.Series]): The original time series data. Should have a datetime-like index.
        sample_total (np.ndarray): The forecasted values.
        stop (int): The point in time where the forecast begins.

    Returns:
        None: Displays the plot using Plotly.
    """
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