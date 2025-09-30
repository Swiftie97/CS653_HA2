import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numba
import os

pd.set_option("mode.copy_on_write", True) # Will be default in pandas 3.0

# Read in data
pd.set_option("mode.copy_on_write", True)
cwd = Path().cwd()
wine_quality_red_filename = Path('winequality-red.csv')
wine_quality_white_filename = Path('winequality-white.csv')

if not wine_quality_red_filename.exists():
    raise FileNotFoundError(f"Dataset file not found: {wine_quality_red_filename}")
if not wine_quality_white_filename.exists():
    raise FileNotFoundError(f"Dataset file not found: {wine_quality_white_filename}")

wine_red = pd.read_csv(wine_quality_red_filename, sep=';') # 1599 data points
wine_white = pd.read_csv(wine_quality_white_filename, sep=';') # 4898 data points

PLOTS_FOLDER = 'plots'

os.makedirs(PLOTS_FOLDER, exist_ok=True)

def question01():
    wine_red_quality = wine_red['quality'] # Range 3-8
    wine_white_quality = wine_white['quality'] # Range 3-9

    # Ranges found with below functions
    print("Red wine quality scores range: [{0},{1}]".format(wine_red_quality.min(), wine_red_quality.max()))
    print("White wine quality scores range: [{0},{1}]".format(wine_white_quality.min(), wine_white_quality.max()))

    # Format bin size
    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]

    # Make both plots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))  # side by side

    # Plot red wine quality
    ax1.hist(wine_red_quality, bins=bins, color='red', edgecolor='black')
    ax1.set_xticks(range(0, 11))
    ax1.set_title("Red Wine Quality")
    ax1.set_xlabel("Quality")
    ax1.set_ylabel("Count")

    # Plot white wine quality
    ax2.hist(wine_white_quality, bins=bins, color='gold', edgecolor='black')
    ax2.set_xticks(range(0, 11))
    ax2.set_title("White Wine Quality")
    ax2.set_xlabel("Quality")
    ax2.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_FOLDER, 'q01_histograms.png'))

def question02():
    def create_boxplots(attribute, ylim_range, call_index):
        wine_red_attribute = wine_red[attribute]
        wine_white_attribute = wine_white[attribute]

        # 4 boxplots arranged in 2x2 fashion
        # Row 1 has red wine and white wine y limits matched
        # Row 2 has red wine and white wine y limits automatically set
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        

        for i in range(len(axes)):

            # Red
            axes[i][0].boxplot(wine_red_attribute)
            axes[i][0].set_title("Red Wine - {}".format(attribute))
            axes[i][0].set_xticks([]) # Hides the 1

            # White
            axes[i][1].boxplot(wine_white_attribute)
            axes[i][1].set_title("White Wine - {}".format(attribute))
            axes[i][1].set_xticks([]) # Hides the 1

            # Match y limits to increase ease of comparing median, Q1, Q3, and IQR
            # Otherwise, keep y limits automatically set to increase ease of seeing outliers
            if i == 0: 
                axes[i][0].set_ylim(ylim_range[0], ylim_range[1])
                axes[i][1].set_ylim(ylim_range[0], ylim_range[1])
                axes[i][0].set_ylabel("{} - match y limits".format(attribute))
            else:
                axes[i][0].set_ylabel("{} - automatic y limits".format(attribute))

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_FOLDER, 'q02_' + str(call_index) + \
                                 '_' + attribute.replace(' ', '_') + '_' + 'boxplots.png'))

    create_boxplots('fixed acidity', (3,17), 0)
    create_boxplots('volatile acidity', (0, 1.7), 1)
    create_boxplots('pH', (2.6, 4.2), 2)
    create_boxplots('density', (0.98, 1.04), 3)

def question03():
    wine_red_scatter_subset = wine_red[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar']]
    wine_white_scatter_subset = wine_white[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar']]

    pd.plotting.scatter_matrix(wine_red_scatter_subset, alpha=0.2, figsize=(12, 12), c='red')
    pd.plotting.scatter_matrix(wine_white_scatter_subset, alpha=0.2, figsize=(12, 12), c='gold')

def question04():
    # Helper functions for distance functions
    @numba.njit
    def minkowski_distance(rowA, rowB, r):
        return np.power(np.power(np.abs(rowA - rowB), r).sum(), 1 / r)

    def calc_inv_covariance_matrix(data):
        x_mean = np.mean(data, axis=0, keepdims=True)
        x_centered = data - x_mean # Get centered data
        covariance_matrix = x_centered.T.dot(x_centered) / (len(data) - 1) # Covariance matrix
        return np.linalg.inv(covariance_matrix) # Inverse Covariance matrix

    # Distance functions
    @numba.njit
    def euclidean(rowA, rowB):
        return np.sqrt(np.square(rowA - rowB).sum())

    @numba.njit
    def minkowski_distance_1(rowA, rowB):
        return minkowski_distance(rowA, rowB, 1)

    @numba.njit
    def minkowski_distance_3(rowA, rowB):
        return minkowski_distance(rowA, rowB, 3)

    @numba.njit
    def minkowski_distance_5(rowA, rowB):
        return minkowski_distance(rowA, rowB, 5)

    @numba.njit
    def mahalanobis_distance(rowA, rowB, inverse_covariance_matrix):
        diff = rowA - rowB
        diff = diff.reshape(1, diff.shape[0])
        result = diff @ inverse_covariance_matrix @ diff.T
        result = result[0][0]
        result = np.sqrt(result)
        return result

    @numba.njit
    def cosine_similarity(rowA, rowB):
        return rowA.dot(rowB) / np.sqrt(np.square(rowA).sum()) / np.sqrt(np.square(rowB).sum())

    @numba.njit
    def correlation(rowA, rowB): 
        rowA_hat = (rowA - np.mean(rowA)) / np.std(rowA)
        rowB_hat = (rowB - np.mean(rowB)) / np.std(rowB)
        return rowA_hat.dot(rowB_hat)

    # Euclidean + Minkowski distance where r=1
    @numba.njit
    def linear_combo(rowA, rowB):
        return euclidean(rowA, rowB) + minkowski_distance_1(rowA, rowB)

    @numba.njit
    def get_distance_function_by_index(rowA, rowB, dist_function_index, inverse_covariance_matrix):
        if dist_function_index == 0:
            return euclidean(rowA, rowB)
        elif dist_function_index == 1:
            return minkowski_distance_1(rowA, rowB)
        elif dist_function_index == 2:
            return minkowski_distance_3(rowA, rowB)
        elif dist_function_index == 3:
            return minkowski_distance_5(rowA, rowB)
        elif dist_function_index == 4:
            return mahalanobis_distance(rowA, rowB, inverse_covariance_matrix)
        elif dist_function_index == 5:
            return cosine_similarity(rowA, rowB)
        elif dist_function_index == 6:
            return correlation(rowA, rowB)
        elif dist_function_index == 7:
            return linear_combo(rowA, rowB)

    @numba.njit
    def generateMatrix(data, distance_function_index, inverse_covariance_matrix):
        matrix = np.zeros((len(data), len(data)))
        for row_i in range(len(data)):
            for row_j in range(len(data)):
                matrix[row_i][row_j] = get_distance_function_by_index(data[row_i], data[row_j], distance_function_index, inverse_covariance_matrix)

        return matrix

    def generateMatricesForDataset(data, distance_func_list):
        inverse_covariance_matrix = calc_inv_covariance_matrix(data) # Set global inverse covariance matrix variable
        matrix_list = []
        for distance_function_index, distance_func in enumerate(distance_func_list):
            matrix = generateMatrix(data, distance_function_index, inverse_covariance_matrix)
            matrix_list.append(matrix)
        return matrix_list

    wine_red_sorted_quality = wine_red.sort_values(by='quality')
    wine_white_sorted_quality = wine_white.sort_values(by='quality')
    wine_red_sorted_quality = wine_red_sorted_quality.drop(columns=['quality', 'alcohol']) # Need just first 10
    wine_white_sorted_quality = wine_white_sorted_quality.drop(columns=['quality', 'alcohol']) # Need just first 10
    wine_red_sorted_quality = wine_red_sorted_quality.reset_index(drop=True)
    wine_white_sorted_quality = wine_white_sorted_quality.reset_index(drop=True)

    distance_func_list = [euclidean, minkowski_distance_1, minkowski_distance_3, minkowski_distance_5, 
                        mahalanobis_distance, cosine_similarity, correlation, linear_combo]
    distance_func_names = ['Euclidean', 'Minkowski Distance, r=1', 'Minkowski Distance, r=3', 'Minkowski Distance r=5', 
                        'Mahalanobis Distance', 'Cosine Similarity', 'Correlation', 'Linear Combo']

    red_wine_matrix_list = generateMatricesForDataset(wine_red_sorted_quality.to_numpy(), distance_func_list)
    white_wine_matrix_list = generateMatricesForDataset(wine_white_sorted_quality.to_numpy(), distance_func_list)

    for matrix_index, red_wine_matrix in enumerate(red_wine_matrix_list):
        ax = sns.heatmap(red_wine_matrix, rasterized=True)
        ax.set(title='Red Wine - {}'.format(distance_func_names[matrix_index]))
        ax.xaxis.set_ticks_position('top')
        plt.xticks(rotation=90)
        plt.savefig('red_wine_{0}_{1}.png'.format(str(matrix_index), distance_func_list[matrix_index].__name__ ), dpi=150)
        plt.close()

    for matrix_index, white_wine_matrix in enumerate(white_wine_matrix_list):
        ax = sns.heatmap(white_wine_matrix, rasterized=True)
        ax.set(title='White Wine - {}'.format(distance_func_names[matrix_index]))
        ax.xaxis.set_ticks_position('top')
        plt.xticks(rotation=90)
        plt.savefig('white_wine_{0}_{1}.png'.format(str(matrix_index), distance_func_list[matrix_index].__name__ ), dpi=150)
        plt.close()


