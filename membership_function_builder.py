# %% Import Libraries
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# %% Load the JSON file
with open("membership_functions_config.json", "r") as file:
    membership_functions_config_json = json.load(file)
    membership_functions_config = membership_functions_config_json['membership_functions']

# %% Load the dataset
file_path = "databases/bank_marketing_preprocessed.csv"

data = pd.read_csv(file_path)

print(data.head())

# %% Remove the target variable from the dataset
TARGET_VARIABLE = "y"
target_variable = data[TARGET_VARIABLE]
data = data.drop(columns=TARGET_VARIABLE)

# %% Process each attribute as specified in the configuration file
for attribute, details in membership_functions_config.items():
    num_clusters = details['num_clusters']
    cluster_info = details['clusters']

    # Check if the attribute is present in the dataset
    if attribute not in data.columns:
        raise ValueError(f"Attribute {attribute} not found in the dataset.")

    # Extract the attribute values
    attribute_data = data[attribute].values.astype(float)
    X = np.expand_dims(attribute_data, axis=0)

    # FCM Clustering
    # cntr - Cluster centers
    # u - Final fuzzy c-partitioned matrix
    # u0 - Initial guess at cluster centers
    # d - Final Euclidian distance matrix
    # jm - Objective function history
    # p - Final number of iterations
    # fpc - Final fuzzy partition coefficient
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X,
        c = num_clusters,
        m = 2,
        error=0.005,
        maxiter=1000,
        init=None
    )

    centers = np.sort(cntr.flatten())

    # Derive the membership function values
    if cluster_info[0]["shape"] == "gaussian":
        # For each cluster, estimate a "spread" (sigma) using a weighted standard deviation
        sigmas = []
        for i in range(num_clusters):
            diff = attribute_data - centers[i]
            # Compute weighted variance using membership degrees u[i]
            sigma = np.sqrt(np.sum(u[i] * diff ** 2) / np.sum(u[i]))
            sigmas.append(sigma)

        # Define a list of membership functions for each cluster
        membership_funcs = []
        for i in range(num_clusters):
            center = centers[i]
            sigma = sigmas[i]
            # Define a Gaussian membership function
            # The Gaussian function is defined as: exp(-0.5 * ((x - center) / sigma) ** 2)
            # Capture 'center' and 'sigma' in the lambda's default parameters
            func = lambda x, c=center, s=sigma: np.exp(-((x - c) ** 2) / (2 * s ** 2))
            membership_funcs.append(func)

    elif cluster_info[0]["shape"] == "triangular":
        # Computer (a, b, c) for each cluster
        a_vals, b_vals, c_vals = [], [], []
        for i, center in enumerate(centers):
            b = center
            if i == 0:
                a = np.min(attribute_data) # Lower bound
            else:
                a = (centers[i-1] + center) / 2
            if i == len(centers) - 1:
                c = np.max(attribute_data) # Upper bound
            else:
                c = (center + centers[i+1]) / 2
            a_vals.append(a)
            b_vals.append(b)
            c_vals.append(c)

        # Define a list of membership functions for each cluster
        membership_funcs = []

        for a, b, c in zip(a_vals, b_vals, c_vals):
            def tri_mu(x, a=a, b=b, c=c):
                # Piecewise function for triangular membership function
                x = np.asarray(x)
                mu = np.zeros_like(x, dtype=float)
                # Rising edge
                idx = np.logical_and(x > a, x <= b)
                if b != a:
                    mu[idx] = (x[idx] - a) / (b - a)
                # Falling edge
                idx = np.logical_and(x > b, x < c)
                if c != b:
                    mu[idx] = (c - x[idx]) / (c - b)
                return mu
            membership_funcs.append(tri_mu)

    elif cluster_info[0]["shape"] == "trapezoidal":
        # Compute (a, b, c, d) for each cluster
        a_vals, b_vals, c_vals, d_vals = [], [], [], []
        for i, center in enumerate(centers):
            if i == 0:
                a = np.min(attribute_data)
            else:
                a = (centers[i-1] + center) / 2
            if i == len(centers) - 1:
                d = np.max(attribute_data)
            else:
                d = (center + centers[i+1]) / 2

            # Here we define b and c to create a plateau
            plateau_width = 0.1 * (d - a)
            b = center - plateau_width / 2
            c = center + plateau_width / 2

            a_vals.append(a)
            b_vals.append(b)
            c_vals.append(c)
            d_vals.append(d)

        # Define a list of membership functions for each cluster
        membership_funcs = []
        for a, b, c, d in zip(a_vals, b_vals, c_vals, d_vals):
            def trap_mu(x, a=a, b=b, c=c, d=d):
                # Piecewise function for trapezoidal membership function
                x = np.asarray(x)
                mu = np.zeros_like(x, dtype=float)
                # Rising edge
                idx = np.logical_and(x > a, x <= b)
                if b != a:
                    mu[idx] = (x[idx] - a) / (b - a)
                # Plateau
                idx = np.logical_and(x > b, x < c)
                mu[idx] = 1
                # Falling edge
                idx = np.logical_and(x >= c, x < d)
                if d != c:
                    mu[idx] = (d - x[idx]) / (d - c)
                return mu
            membership_funcs.append(trap_mu)

    elif cluster_info[0]["shape"] == "crisp":
        # Define crisp membership functions for binary columns
        membership_funcs = [lambda x, c=center: 1 if x == c else 0 for center in centers]

    else:
        raise ValueError(f"Membership function shape '{cluster_info[0]['shape']}' not recognized.")

    # Plot the membership functions
    plt.figure()
    plt.title(f"Membership Functions for {attribute}")
    plt.xlabel(attribute)
    plt.ylabel("Membership Degree")
    plt.ylim(0, 1)
    for i, func in enumerate(membership_funcs):
        # don't plot the membership function for binary columns
        if cluster_info[0]["shape"] != "crisp":
            plt.plot(attribute_data, func(attribute_data), label=f"Cluster {i+1}")
        else:
            plt.axvline(centers[i], color='r', linestyle='--', label=f"Cluster {i+1}")
    plt.legend()
    plt.show()
