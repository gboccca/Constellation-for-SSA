import matplotlib
matplotlib.use('TkAgg')  # Or another backend like 'Agg'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.special import erf
import warnings 

file_path_alt = r"Optimization\master_results.txt"

columns_alt = [
    "Altitude", "Expl-Fragm", "Coll_Fragm", "Launch/Mis", "NaK-Drops", "SRM-Slag", "SRM-Dust",
    "Paint Flks", "Ejecta", "MLI", "Cloud 1", "Cloud 2", "Cloud 3", "Cloud 4", "Cloud 5", "Human-made",
    "Meteoroids", "Streams", "Total"
]

data_alt = pd.read_csv(file_path_alt, sep=r'\s+', skiprows=2, names=columns_alt)
data_alt['Probability'] = data_alt['Total'] / data_alt['Total'].sum()


def gaussian(x, w, mu, sigma):
    return w * norm.pdf(x, mu, sigma)  # Single Gaussian

def gaussian_mixture(x, w1, mu1, sigma1, w2, mu2, sigma2):
    g1 = w1 * norm.pdf(x, mu1, sigma1)  # First Gaussian
    g2 = w2 * norm.pdf(x, mu2, sigma2)  # Second Gaussian
    return g1 + g2

def skewed_gaussian(x, w, mu, sigma, alpha):
    norm_pdf = norm.pdf(x, mu, sigma)
    norm_cdf = (1 + erf(alpha * (x - mu) / (sigma * np.sqrt(2)))) / 2
    return 2 * w * norm_pdf * norm_cdf


def doublegaussian_fit():
    global data_alt
    
    p0 = [0.6, 800, 150, 0.4, 1500, 200]
    params, _ = curve_fit(gaussian_mixture, data_alt['Altitude'], data_alt['Probability'], p0=p0)
    return params


def single_gaussian_fit():
    global data_alt
    
    # Initial guess: weight=1, mean=average altitude, sigma=std deviation of altitude
    p0 = [1, np.mean(data_alt['Altitude']), np.std(data_alt['Altitude'])]
    
    params, _ = curve_fit(gaussian, data_alt['Altitude'], data_alt['Probability'], p0=p0)
    return params  # Returns weight, mu, sigma



def skewed_gaussian_fit():
    global data_alt
    
    # Initial guess: weight=1, mean=average altitude, sigma=std deviation of altitude, alpha=0 (no skew)
    p0 = [1, np.mean(data_alt['Altitude']), np.std(data_alt['Altitude']), 0]
    
    params, _ = curve_fit(skewed_gaussian, data_alt['Altitude'], data_alt['Probability'], p0=p0)
    return params  # Returns weight, mu, sigma, alpha


def satellite_dist(**kwargs):
    """
    Generate the satellite distribution from gasussian mixture model
    kwargs (all required):

        w1, mu1, sigma1, w2, mu2, sigma2, num_obrits, num_sats

    returns:
        discrete_dist, altitudes
    """

    required_kwargs = ['w1', 'mu1', 's1', 'w2', 'mu2', 's2', 'num_obrits', 'num_sats']

    # unpack parameters

    for key in required_kwargs:
        if key not in kwargs:
            raise ValueError(f'Missing required parameter {key}')

    superfluous_kwargs = []
    for key in kwargs:
        if key not in required_kwargs:
            superfluous_kwargs.append(key)

    # this will be triggered every time if the function is called correctly
    if superfluous_kwargs:
        warnings.warn(f'Superfluous parameters: {superfluous_kwargs}')
        # remove superfluous keys from kwargs
        for key in superfluous_kwargs:
            kwargs.pop(key)

    num_obrits = kwargs['num_obrits']
    num_sats = kwargs['num_sats']
    w1 = kwargs['w1']
    mu1 = kwargs['mu1']
    sigma1 = kwargs['s1']
    w2 = kwargs['w2']
    mu2 = kwargs['mu2']
    sigma2 = kwargs['s2']

    # generate discrete positions along distribution
    altitudes = np.linspace(450, 1500, num_obrits)
    # generate distribution based on mean and std_deviation
    gmm_dist = (w1 * norm.pdf(altitudes, mu1, sigma1) +
                w2 * norm.pdf(altitudes, mu2, sigma2))  # Sum of two Gaussians

    # generate discrete distribution
    normalized_dist = gmm_dist / np.sum(gmm_dist)
    scaled_dist = normalized_dist * num_sats
    discrete_dist = np.round(scaled_dist).astype(int)

    # distribute remaining satellites in orbits closest to the distribution peak
    res = num_sats - np.sum(discrete_dist)
    peak_index = np.argmax(scaled_dist)
    sorted_indices = sorted(range(num_obrits), key=lambda x: abs(x - peak_index))
    for i in range(res):
        discrete_dist[sorted_indices[i % num_obrits]] += 1

    return discrete_dist, altitudes


if __name__ == '__main__':
    
    w1_fit, mu1_fit, sigma1_fit, w2_fit, mu2_fit, sigma2_fit = doublegaussian_fit()
    print(f"Best Fit Parameters:\n w1={w1_fit}, mu1={mu1_fit}, sigma1={sigma1_fit}")
    print(f" w2={w2_fit}, mu2={mu2_fit}, sigma2={sigma2_fit}")

    x = np.linspace(200, 2000, 500)
    y = gaussian_mixture(x,w1_fit,mu1_fit,sigma1_fit,w2_fit,mu2_fit,sigma2_fit)

    plt.plot(data_alt['Altitude'], data_alt['Probability'])
    plt.plot(x,y, label="Gaussian Mixture")
    plt.xlabel('Altitude')
    plt.ylabel('Probability')
    plt.grid()
    plt.show()

    w, mu, sigma = single_gaussian_fit()
    print(f"Best Fit Parameters:\n w={w}, mu={mu}, sigma={sigma}")

    y = gaussian(x, w, mu, sigma)
    plt.plot(data_alt['Altitude'], data_alt['Probability'])
    plt.plot(x, y, label="Single Gaussian")
    plt.xlabel('Altitude')
    plt.ylabel('Probability')
    plt.grid()
    plt.show()

    w, mu, sigma, alpha = skewed_gaussian_fit()
    print(f"Best Fit Parameters:\n w={w}, mu={mu}, sigma={sigma}, alpha={alpha}")

    y = skewed_gaussian(x, w, mu, sigma, alpha)
    plt.plot(data_alt['Altitude'], data_alt['Probability'])
    plt.plot(x, y, label="Skewed Gaussian")
    plt.xlabel('Altitude')
    plt.ylabel('Probability')
    plt.grid()
    plt.show()