"""
Preprocessing Module
--------------------
In this module there are functions to deal with Hi-C matrices.
In particular to perform basic spectral analysis and to reconstruct them with the summing of projectors.
"""
import networkx as nx
import pandas as pd
import numpy as np

def create_graph(matrix_df: pd.DataFrame, 
                 metadata_df: pd.DataFrame
                 )-> nx.Graph:
    """
    This function creates a nx.Graph object, starting from the adjacency matrix of the network and the label of each node.

    Parameters
    ----------
        matrix_df (pd.DataFrame): 
            Data Frame containing the adjacency matrix of the network.
        metadata_df (pd.DataFrame): 
            Data Frame containing the metadata of the nodes.

    Returns
    --------
        nx.Graph: 
            Graph object of NetworkX library representing described by the adjacency contained in the `matrix_df`.
    """
    network_graph = nx.from_pandas_adjacency(matrix_df)
    list = []

    for i in range(len(metadata_df.chr)):
        list += ([metadata_df.chr[i]] * (metadata_df.end[i] - metadata_df.start[i]+1))
  
    dict = {index: value for index, value in enumerate(list)}
    nx.set_node_attributes(network_graph, dict, 'chromosome')
    
    return network_graph

def remove_empty_axis(matrix: np.ndarray
                      )-> np.ndarray:
    """
    This function returns the given matrix removing the rows and columns that contain only 0.

    Parameters
    ----------
        matrix (np.ndrray): 
            original matrix from which remove empty rows and columns (containing all zeros).

    Returns
    --------
        np.ndarray: 
            submatrix of the original one, where the empty rows and columns have been removed.
    """
    matrix = matrix[np.any(matrix != 0, axis=1)]
    matrix = matrix[:, (matrix != 0).any(axis=0)]

    return matrix

def extract_diagonal_blocks(matrix: np.ndarray, 
                            block_size: list
                            ) -> list:
    """
    This function extract the diagonal blocks of dimension contained in the 
    block_size, and return them under the form of a list of np.ndarray.

    Parameters
    ----------
        matrix (np.ndarray): 
            initial matrix from which we want to extract the blocks of the diagonal
        block_size (list): 
            list containing the dimension of each block in the diagonal.

    Returns
    --------
        list: 
            list of np.ndarray containing the blocks of the diagonal extracted from the original matrix.
    """
    blocks = []
    start = 0
    for dim in block_size:
        submatrix = matrix[start:start+dim, start:start+dim]
        blocks.append(submatrix)
        start += dim
    return blocks

def get_chromosome_list(metadata_df: pd.DataFrame
                        ) -> list:
    """
    This function, given a dataframe as input, containing the columns `start` and `end`, containing the 
    number of row of start and end of each chromosome, returns a list with the difference between 
    `end` element and `start` element.

    Parameters
    ----------
        metadata_df (pd.DataFrame): 
            Dataframe containing a column named `start`, a column `end` and a 
            column with the name of the chromosome (not necessary).

    Returns
    --------
        list: 
            list containing the number of segments into which a chromosome has been divided.
    """
    return list(metadata_df.end-metadata_df.start+1)

def ooe_normalization(matrix: np.ndarray
                      )-> np.ndarray:
    """
    This function allows to normalize a given matrix with the Observed-Over-Expected (OOE) 
    algorithm for a symmetric matrix. In particular this algorithm provides to divide each 
    element for the mean of the diagonal it belongs to.

    Parameters
    ----------
        matrix (np.ndarray): 
            symmetric matrix to be normalized.

    Returns
    --------
        np.ndarray:     
            matrix normalized with the OOE algorithm.
    """
    mean_matrix = np.zeros(matrix.shape)
    for i in range(matrix.shape[0]):
        mean = np.mean(np.diag(matrix, i))
        np.fill_diagonal(mean_matrix[:, i:], mean)
    mean_matrix += mean_matrix.T
    mean_matrix[np.diag_indices(matrix.shape[0])] = np.diagonal(mean_matrix) / 2
    mean_matrix[mean_matrix == 0] = 1
    normalized_matrix = matrix/mean_matrix
    return normalized_matrix

def build_projectors(eigenvectors: np.array,
                     number_of_projector_desired: int
                     ) -> list:
    """
    This function allows to get a list containing the projectors corresponding to the eigenvectors provided
    as input. The eigenvectors must be contained in a np.array where the columns are the eigenvectors.

    Parameters
    ----------
    eigenvectors (np.array) :
        Array containing the in the columns the eigenvectors of a matrix.
    number_of_projector_desired (int) :
        Number of projectors that you want to build.

    Returns
    -------
    list: 
        Output list containing the projectors corresponding to the input eigenvectors.
    """
    projectors = []
    for i in range(number_of_projector_desired):
        x = eigenvectors[:,i]
        projector = np.outer(x, x.T) / np.dot(x.T, x)
        projectors.append(projector)
    
    return projectors

def reconstruct_matrix(projectors: list,
                       eigenvalues: list,
                       number_of_projectors: int
                       ) -> np.ndarray:
    """
    This function takes as input a list of matrices (projectors) and a number of projectors to be added
    together in order to reconstruct the matrix.

    Parameters
    ----------
    projectors (list) :
        List containing np.ndarray of the projectors. It is supposed that the projectors are ordered 
        from the most to the least important.
    eigenvalues (list) :
        List containing the eigenvalues corresponding to the `projectors`.
    number_of_projectors (int) :
        Number of projectors to be added together.

    Returns
    -------
    np.ndarray: 
        Reconstructed matrix from the addition of the first `number_of_projectors` projectors, multiplied by
        the corresponding eigenvectors.

    Raises
    ------
        ValueError: 
            When the `number_of_projectors` parameter is greater than the length of the `projectors` list.
        ValueError: 
            When the `number_of_projectors` parameter is greater than the length of the `eigenvalues` list.
    """
    if number_of_projectors > len(projectors):
        raise ValueError('Number of projectors must be smaller than the length of projectors list.')
    if number_of_projectors > len(eigenvalues):
        raise ValueError('Number of projectors must be smaller than the number of eigenvalues in the list.')
    reconstructed_matrix = np.zeros(shape=projectors[0].shape)
    for i in range(number_of_projectors):
        reconstructed_matrix += projectors[i] * eigenvalues[i]
    return reconstructed_matrix
# %%
