import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numba
import os
import gc

pd.set_option("mode.copy_on_write", True) # Will be default in pandas 3.0

# Read in data
cwd = Path().cwd()
wine_quality_red_filename = Path('winequality-red.csv')
wine_quality_white_filename = Path('winequality-white.csv')

# Check if dataset files exist
if not wine_quality_red_filename.exists():
    raise FileNotFoundError(f"Dataset file not found: {wine_quality_red_filename}")
if not wine_quality_white_filename.exists():
    raise FileNotFoundError(f"Dataset file not found: {wine_quality_white_filename}")

# Read in dataset
wine_red = pd.read_csv(wine_quality_red_filename, sep=';') # 1599 data points
wine_white = pd.read_csv(wine_quality_white_filename, sep=';') # 4898 data points

PLOTS_FOLDER = 'plots'

# Make plots subdirectory
os.makedirs(PLOTS_FOLDER, exist_ok=True)

def question01():
    """
    Question 1
    """
    # Get quality column data from datasets
    wine_red_quality = wine_red['quality'] # Range 3-8
    wine_white_quality = wine_white['quality'] # Range 3-9

    # Quality ranges found with below functions
    # print("Red wine quality scores range: [{0},{1}]".format(wine_red_quality.min(), wine_red_quality.max()))
    # print("White wine quality scores range: [{0},{1}]".format(wine_white_quality.min(), wine_white_quality.max()))

    # Format bin size
    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]

    # Make both plots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))  # side by side

    # Plot red wine quality histogram
    ax1.hist(wine_red_quality, bins=bins, color='red', edgecolor='black')
    ax1.set_xticks(range(0, 11))
    ax1.set_title("Red Wine Quality")
    ax1.set_xlabel("Quality")
    ax1.set_ylabel("Count")

    # Plot white wine quality histogram
    ax2.hist(wine_white_quality, bins=bins, color='gold', edgecolor='black')
    ax2.set_xticks(range(0, 11))
    ax2.set_title("White Wine Quality")
    ax2.set_xlabel("Quality")
    ax2.set_ylabel("Count")

    # Save plots
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_FOLDER, 'q01_histograms.png'))
    plt.close()

def question02():
    """
    Question 2
    """
    def create_boxplots(attribute, ylim_range, call_index):
        """
        Generates 4 boxplots for the given attribute in a 2x2 arrangement.
        First row: boxplots for red and white wine, with y scale same based on input y range.
        Second row: boxplots of the same figures, with y scale automatically defined.

        Args:
            attribute (str): attribute of wine metric to plot.
            ylim_range (tuple(float, float)): range to keep the y scale for the first set of boxplots even.
            call_index (int): the nth set of boxplots called (for file naming purposes).
        """
        # Get the series objects for the chosen attribute
        wine_red_attribute = wine_red[attribute]
        wine_white_attribute = wine_white[attribute]

        # 4 boxplots arranged in 2x2 fashion
        # Row 1 has red wine and white wine y limits matched
        # Row 2 has red wine and white wine y limits automatically set
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        

        for i in range(len(axes)):

            # Plot red wine boxplot
            axes[i][0].boxplot(wine_red_attribute)
            axes[i][0].set_title("Red Wine - {}".format(attribute))
            axes[i][0].set_xticks([]) # Hides the 1

            # Plot white wine boxplot
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

        # Save plots
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_FOLDER, 'q02_' + str(call_index) + \
                                 '_' + attribute.replace(' ', '_') + '_' + 'boxplots.png'))
        plt.close()

    # Generate boxplots for the chosen attributes
    create_boxplots('fixed acidity', (3,17), 0)
    create_boxplots('volatile acidity', (0, 1.7), 1)
    create_boxplots('pH', (2.6, 4.2), 2)
    create_boxplots('density', (0.98, 1.04), 3)

def question03():
    """
    Question 3
    """
    # Grab only these columns from the dataset
    wine_red_scatter_subset = wine_red[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar']]
    wine_white_scatter_subset = wine_white[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar']]

    # Generate scatter matrix for red wine
    pd.plotting.scatter_matrix(wine_red_scatter_subset, alpha=0.2, figsize=(12, 12), c='red')
    plt.savefig(os.path.join(PLOTS_FOLDER, 'q03_red_wine_scatter_matrix.png'))
    plt.close()
    # Generate scatter matrix for white wine
    pd.plotting.scatter_matrix(wine_white_scatter_subset, alpha=0.2, figsize=(12, 12), c='gold')
    plt.savefig(os.path.join(PLOTS_FOLDER, 'q03_white_wine_scatter_matrix.png'))
    plt.close()

def question04():
    """
    Question 4

    @numba.njit decorators used optimize helper functions to improve execution speed by compling
    Python code to machine code using C-style approach. Very useful for computationally intensive
    operations in numerical or array-based computations.
    """

    # Helper functions for distance functions
    @numba.njit
    def minkowski_distance(rowA, rowB, r):
        """
        Compute Minkowski distance.

        Args:
            rowA (np.ndarray): row of dataset as a 1D array.
            rowB (np.ndarray): row of dataset as a 1D array.
            r (int or float): Minkowski norm order.

        Returns:
            float: Minkowski distance of order `r` between `rowA` and `rowB`.
        """
        return np.power(np.power(np.abs(rowA - rowB), r).sum(), 1 / r)

    def calc_inv_covariance_matrix(data):
        """
        Get inverse covariance matrix from input wine dataset.

        Args:
            data (pd.DataFrame): wine dataset.

        Returns:
            np.array: Computed inverse covariance matrix.
        """
        x_mean = np.mean(data, axis=0, keepdims=True)
        x_centered = data - x_mean # Get centered data
        covariance_matrix = x_centered.T.dot(x_centered) / (len(data) - 1) # Covariance matrix
        return np.linalg.inv(covariance_matrix) # Inverse Covariance matrix

    # Distance functions
    @numba.njit
    def euclidean(rowA, rowB):
        """
        Compute Euclidean distance.

        Args:
            rowA (np.ndarray): row of dataset as a 1D array.
            rowB (np.ndarray): row of dataset as a 1D array.

        Returns:
            float: Euclidean distance between `rowA` and `rowB`.
        """
        return np.sqrt(np.square(rowA - rowB).sum())

    @numba.njit
    def minkowski_distance_1(rowA, rowB):
        """
        Compute Minkowski distance of order 1.

        Args:
            rowA (np.ndarray): row of dataset as a 1D array.
            rowB (np.ndarray): row of dataset as a 1D array.

        Returns:
            float: Minkowski distance of order 1 between `rowA` and `rowB`.
        """
        return minkowski_distance(rowA, rowB, 1)

    @numba.njit
    def minkowski_distance_3(rowA, rowB):
        """
        Compute Minkowski distance of order 3.

        Args:
            rowA (np.ndarray): row of dataset as a 1D array.
            rowB (np.ndarray): row of dataset as a 1D array.

        Returns:
            float: Minkowski distance of order 3 between `rowA` and `rowB`.
        """
        return minkowski_distance(rowA, rowB, 3)

    @numba.njit
    def minkowski_distance_5(rowA, rowB):
        """
        Compute Minkowski distance of order 5.

        Args:
            rowA (np.ndarray): row of dataset as a 1D array.
            rowB (np.ndarray): row of dataset as a 1D array.

        Returns:
            float: Minkowski distance of order 5 between `rowA` and `rowB`.
        """
        return minkowski_distance(rowA, rowB, 5)

    @numba.njit
    def mahalanobis_distance(rowA, rowB, inverse_covariance_matrix):
        """
        Compute Mahalanobis distance.

        Args:
            rowA (np.ndarray): row of dataset as a 1D array.
            rowB (np.ndarray): row of dataset as a 1D array.
            inverse_covariance_matrix (np.ndarray): Precomputed inverse covariance matrix.

        Returns:
            float: Mahalanobis distance between `rowA` and `rowB`.
        """
        diff = rowA - rowB
        # To allow for transpose
        diff = diff.reshape(1, diff.shape[0])
        result = diff @ inverse_covariance_matrix @ diff.T
        result = result[0][0]
        result = np.sqrt(result)
        return result

    @numba.njit
    def cosine_similarity(rowA, rowB):
        """
        Compute cosine similarity.

        Args:
            rowA (np.ndarray): row of dataset as a 1D array.
            rowB (np.ndarray): row of dataset as a 1D array.

        Returns:
            float: Cosine similarity between `rowA` and `rowB`.
        """
        return rowA.dot(rowB) / np.sqrt(np.square(rowA).sum()) / np.sqrt(np.square(rowB).sum())

    @numba.njit
    def correlation(rowA, rowB):
        """
        Compute correlation.

        Args:
            rowA (np.ndarray): row of dataset as a 1D array.
            rowB (np.ndarray): row of dataset as a 1D array.

        Returns:
            float: Correlation between `rowA` and `rowB`.
        """
        rowA_hat = (rowA - np.mean(rowA)) / np.std(rowA)
        rowB_hat = (rowB - np.mean(rowB)) / np.std(rowB)
        return rowA_hat.dot(rowB_hat)

    @numba.njit
    def linear_combo(rowA, rowB):
        """
        Compute a linear combination of Euclidean distance and Minkowski distance (r=1)
        between two 1D arrays, with equal weights of 0.5 for each.

        Args:
            rowA (np.ndarray): row of dataset as a 1D array.
            rowB (np.ndarray): row of dataset as a 1D array.

        Returns:
            float: Weighted sum (0.5 each) of Euclidean distance and Minkowski distance
                (r=1) between `rowA` and `rowB`.
        """
        return 0.5 * euclidean(rowA, rowB) + 0.5 * minkowski_distance_1(rowA, rowB)

    @numba.njit
    def get_distance_function_by_index(rowA, rowB, dist_function_index, inverse_covariance_matrix):
        """
        Compute a distance or similarity measure between two input 
        dataset rows based on the specified function index.
        Args:
            rowA (np.ndarray): row of dataset as a 1D array.
            rowB (np.ndarray): row of dataset as a 1D array.
            dist_function_index (_type_): Index specifying which distance or similarity function to use:
                0 = Euclidean distance
                1 = Minkowski distance (r=1)
                2 = Minkowski distance (r=3)
                3 = Minkowski distance (r=5)
                4 = Mahalanobis distance
                5 = Cosine similarity
                6 = Correlation distance
                7 = Linear combination of Euclidean and Minkowski (r=1)
            inverse_covariance_matrix (_type_): Precomputed inverse covariance matrix, 
                required only if `dist_function_index` is 4 (Mahalanobis distance); 
                can be None otherwise.

        Returns:
            float: Computed distance or similarity value between `rowA` and `rowB` 
                according to the selected function.
        """
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
        """
        Generate a square distance/similarity matrix for a dataset using a specified distance function.

        Args:
            data (np.ndarray): Dataset as a np 2d array.
            distance_function_index (int): Index of the distance/similarity function to use.
            inverse_covariance_matrix (np.ndarray): Precomputed inverse covariance matrix, required for Mahalanobis distance.

        Returns:
            np.ndarray: Square matrix of distances/similarities between all pairs of rows in `data`.
        """
        # Instantiate result matrix
        matrix = np.zeros((len(data), len(data)))
        # Iterate through all combos of pairs of rows in dataset
        for row_i in range(len(data)):
            for row_j in range(len(data)):
                matrix[row_i][row_j] = get_distance_function_by_index(data[row_i], data[row_j], distance_function_index, inverse_covariance_matrix)

        return matrix

    def generateMatricesForDataset(data, distance_func_list):
        """
        Generate distance/similarity matrices for a dataset using a list of distance functions.

        Args:
            data (np.ndarray): Dataset as a numpy 2D array.
            distance_func_list (list): List of distance/similarity functions.

        Returns:
            list: List of square matrices, one per distance function, containing pairwise distances/similarities.
        """
        # Compute inverse covariance matrix
        inverse_covariance_matrix = calc_inv_covariance_matrix(data)
        matrix_list = []
        # Acquire list of distance/similarity matrices
        for distance_function_index, distance_func in enumerate(distance_func_list):
            matrix = generateMatrix(data, distance_function_index, inverse_covariance_matrix)
            matrix_list.append(matrix)
        return matrix_list
    
    def plotMatrix(title_start, matrix, matrix_index, distance_func_names, distance_func_list):
        """
        Plot a heatmap of a distance/similarity matrix and then save as .png.

        Args:
            title_start (str): Prefix for the plot title.
            matrix (np.ndarray): 2D distance/similarity matrix to be plot.
            matrix_index (int): Index of the matrix/function in the lists. For file naming purposes.
            distance_func_names (list): List of distance function names for labeling.
            distance_func_list (list): List of distance functions corresponding to the matrices.

        Returns:
            None
        """
        # Set up plot layout and plot heatmap with Seaborn
        plt.figure(figsize=(14, 12))
        ax = sns.heatmap(matrix, rasterized=True)
        plot_title = title_start + ' - {}'.format(distance_func_names[matrix_index])
        ax.set(title=plot_title)
        ax.set_aspect('equal')
        ax.xaxis.set_ticks_position('top')
        plt.xticks(rotation=90)
        plt.tight_layout()

        # Get file name of output image
        dist_f_name = distance_func_list[matrix_index].__name__ 
        file_name = f"q04_{title_start.replace(' ', '_')}_{str(matrix_index)}_{dist_f_name}.png"

        # Save to disk and free memory
        # Not freeing memory causes crashes for me since I've only allocated 8GB for my WSL2 setup.
        file_path = os.path.join(PLOTS_FOLDER, file_name)
        plt.savefig(file_path, dpi=150)
        plt.close()
        del ax
        gc.collect()

    # Sort databases by quality
    wine_red_sorted_quality = wine_red.sort_values(by='quality')
    wine_white_sorted_quality = wine_white.sort_values(by='quality')

    # Drop two columns, since we only need the specified 10 features
    wine_red_sorted_quality = wine_red_sorted_quality.drop(columns=['quality', 'alcohol'])
    wine_white_sorted_quality = wine_white_sorted_quality.drop(columns=['quality', 'alcohol'])
    wine_red_sorted_quality = wine_red_sorted_quality.reset_index(drop=True)
    wine_white_sorted_quality = wine_white_sorted_quality.reset_index(drop=True)

    # List of distance/similarity functions
    distance_func_list = [euclidean, minkowski_distance_1, minkowski_distance_3, minkowski_distance_5, 
                        mahalanobis_distance, cosine_similarity, correlation, linear_combo]
    # List of names of distance/similarity functions
    distance_func_names = ['Euclidean', 'Minkowski Distance, r=1', 'Minkowski Distance, r=3', 'Minkowski Distance r=5', 
                        'Mahalanobis Distance', 'Cosine Similarity', 'Correlation', 'Linear Combo']

    # Get sampled database (every 10 rows)
    wine_red_sorted_quality_sampled10 = wine_red_sorted_quality[::10]
    wine_white_sorted_quality_sampled10 = wine_white_sorted_quality[::10]

    # Get distance matrices for sampled red wine database
    red_wine_sampled10_matrix_list = generateMatricesForDataset(wine_red_sorted_quality_sampled10.to_numpy(), distance_func_list)
    # Plot heatmaps
    for matrix_index, red_wine_matrix_sampled in enumerate(red_wine_sampled10_matrix_list):
        plotMatrix('Red Wine (sampled factor of 10)', red_wine_matrix_sampled, matrix_index, distance_func_names, distance_func_list)

    # Get distance matrices for sampled white wine database
    white_wine_sampled10_matrix_list = generateMatricesForDataset(wine_white_sorted_quality_sampled10.to_numpy(), distance_func_list)
    # Plot heatmaps
    for matrix_index, white_wine_matrix_sampled in enumerate(white_wine_sampled10_matrix_list):
        plotMatrix('White Wine (sampled factor of 10)', white_wine_matrix_sampled, matrix_index, distance_func_names, distance_func_list)

    # Get distance matrices for full red wine database
    red_wine_matrix_list = generateMatricesForDataset(wine_red_sorted_quality.to_numpy(), distance_func_list)
    # Plot heatmaps
    for matrix_index, red_wine_matrix in enumerate(red_wine_matrix_list):
        plotMatrix('Red Wine', red_wine_matrix, matrix_index, distance_func_names, distance_func_list)

    # Get distance matrices for full white wine database
    white_wine_matrix_list = generateMatricesForDataset(wine_white_sorted_quality.to_numpy(), distance_func_list)
    # Plot heatmaps
    for matrix_index, white_wine_matrix in enumerate(white_wine_matrix_list):
        plotMatrix('White Wine', white_wine_matrix, matrix_index, distance_func_names, distance_func_list)