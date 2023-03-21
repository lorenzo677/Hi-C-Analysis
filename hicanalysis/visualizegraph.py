"""
Visualizegraph Module
------------------
In this module there are functions to visualize the results of the 
Hi-C matrices with a pre-defined stile.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from typing import Optional

def weights_distribuion(matrix: np.ndarray, 
                        savepath: Optional[str] = None
                        ):
    """
    This function, given an adjacency matrix of a weighted network, plots and shows 
    the distribution of the weights of the network. If the parameter `savepath` is given 
    the image will be saved in the location provided.

    Parameters
    ----------
        matrix : np.ndarray 
            Adjacency matrix under the form of np.ndarray.
        savepath : Optional[str, path-like], optional 
            Parameter to chose the path and name where to save the file. If this
            parameter is not given, the image will not be saved. Defaults to None.
    """
    sns.set_style("darkgrid")
    sns.histplot( data = matrix.ravel(), bins='auto', color = sns.color_palette()[2]) 
    plt.title('Weights distribution')
    if savepath is not None:
        plt.savefig(savepath, dpi=400)
    plt.show()

def degree_distribution(network_graph: nx.Graph, 
                        savepath: Optional[str] = None
                        ):
    """
    This function, given a network graph, plots and shows the distribution of the degree of the nodes of the network.
    If the parameter `savepath` is given the image will be saved in the location provided.

    Parameters
    ----------
        network_graph : nx.Graph
            Network graph under the form of object of the Graph class of Networkx library.
        savepath : Optional[str, path-like], optional 
            Parameter to chose the path and name where to save the file. If this
            parameter is not given, the image will not be saved. Defaults to None.
    """
    sns.set_style("darkgrid")
    values = np.array( network_graph.degree(network_graph.nodes))[:, 1]
    sns.histplot(data = values , bins='auto', color = sns.color_palette()[2]) 
    plt.title('Degree distribution')
    plt.yscale('log')
    if savepath is not None:
        plt.savefig(savepath, dpi=400)
    plt.show()

def plot_chromosome_hics(matrices: np.ndarray, 
                         names: np.ndarray, 
                         savepath: Optional[str] = None
                         ):
    """
    This function produce an image containing the HiC matrices of the 22+2 chromosomes of the DNA.
    The function don't apply any logarithm to the data.
    If the parameter `savepath` is given the image will be saved in the location provided.

    Parameters
    ----------
        matrices : np.ndarray 
            Array containing all the 24 matrices corresponding to the chromosomes
        names : np.ndarray
            Array containing the names to label each plot
        savepath : Optional[str, path-like], optional 
            Parameter to chose the path and name where to save the file. If this
            parameter is not given, the image will not be saved. Defaults to None.
    """
    rows, columns = 4, 6
    fig, axes = plt.subplots(rows, columns, figsize=(14,8))
    fig.subplots_adjust(top = 0.96, right=0.96, left=0.037, bottom=0.037, hspace=0.265, wspace=0.258)
    for i, ax in enumerate(axes.flat):
        image = ax.imshow(matrices[i])
        ax.set_title(names[i], fontsize = 8)
        plt.colorbar(image)
        ax.axis('off')
    if savepath is not None:
        plt.savefig(savepath, dpi=400)
    plt.show()

def plot_chromosomes_histograms(matrices: np.ndarray, 
                                names: np.ndarray, 
                                savepath: Optional[str] = None
                                ):
    """
    This function produce an image containing the histograms of the absolute value 
    of the eigenvalues of the 22+2 chromosomes of the DNA.
    The function don't apply any logarithm to the data.
    If the parameter `savepath` is given the image will be saved in the location provided.

    Parameters
    ----------
        matrices : np.ndarray 
            Array containing all the 24 matrices corresponding to the chromosomes.
        names : np.ndarray 
            Array containing the names to label each plot.
        savepath : Optional[str, path-like], optional 
            Parameter to chose the path and name where to save the file. If this
            parameter is not given, the image will not be saved. Defaults to None.
    """
    rows, columns =  6, 4
    fig, axes = plt.subplots(rows, columns, figsize=(10,14))
    fig.subplots_adjust(top = 0.96, right=0.96, left=0.037, bottom=0.037, hspace=0.265, wspace=0.258)
    for i, ax in enumerate(axes.flat):
        eigenvalues = np.abs(np.linalg.eigvals(matrices[i]))
        ax.hist(eigenvalues, bins='auto', density=True)  
        ax.set_title(names[i], fontsize = 8)
        ax.set_xlim(-1, 20)
    if savepath is not None:
        plt.savefig(savepath, dpi=400)
    plt.show()

def plot_matrix(matrix: np.ndarray, 
                title: Optional[str] = '', 
                savepath: Optional[str] = None
                ):
    """
    This function shows the image of the matrix with a colorbar. 
    It is possible to set a title with the parameter title.
    If the parameter `savepath` is given the image will be saved in the location provided.

    Parameters
    ----------
        matrix : np.ndarray 
            Matrix in the form of a np.ndarray.
        title : Optional[str], optional  
            Title to put in the image. Defaults to ''.
        savepath : Optional[str, path-like], optional 
            Parameter to chose the path and name where to save the file. If this
            parameter is not given, the image will not be saved. Defaults to None.
    """
    plt.imshow(matrix)
    sns.set_style("dark")
    plt.title(title)
    plt.colorbar()
    plt.subplots_adjust(top = 0.98, right=0.98, left=0.05, bottom=0.05)
    if savepath is not None:
        plt.savefig(savepath, dpi=400)
    plt.show()

def scatter_plot(image1: np.ndarray, 
                 image2: np.ndarray, 
                 label1: Optional[str] = 'Image 1', 
                 label2: Optional[str] = 'Image 2', 
                 savepath: Optional[str] = None
                 ):
    """
    This function generate a scatter plot to compare two images or matrices. Given 
    two numpy.ndarray, and optionally the label to assign to each image, it plots 
    the scatter plot of the values of the images. If the parameter `savepath` is 
    given the image will be saved in the location provided.

    Parameters
    ----------
        image1 : np.ndarray 
            First image, that lays on the x-axis.
        image2 : np.ndarray  
            Second image, that lays on the y-axis.
        label1 Optional[str], optional  
            Label to assign to the first image. Defaults to 'Image 1'.
        label2 Optional[str], optional  
            Label to assign to the second image. Defaults to 'Image 2'.
        savepath : Optional[str, path-like], optional  
            Parameter to chose the path and name where to save the file. If this
            parameter is not given, the image will not be saved. Defaults to None.
    """
    pixels1 = image1.flatten()
    pixels2 = image2.flatten()
    sns.set_style('darkgrid')
    fig, ax = plt.subplots()
    ax.scatter(pixels1, pixels2, s=2)
    plt.xlabel(label1, fontsize = 18)
    plt.ylabel(label2, fontsize = 18)
    plt.subplots_adjust(top = 0.96, right=0.98, left=0.1, bottom=0.1)
    if savepath is not None:
        plt.savefig(savepath, dpi=400)
    plt.show()

def histogram(data,
              label,
              xlabel: Optional[str] = '',
              ylabel: Optional[str] = '',
              xlim: Optional[tuple] = None,
              title: str = '', 
              ylogscale: bool = False, 
              savepath: Optional[str] = None
              ):
    """
    This function plot an histogram of the `data` with a predefined `darkgrid` style 
    of seaborn. In the case `data` is an array, it will be shown the histogram of 
    the values, if `data` is a sequence of array it will be shown an image 
    containing all the corresponding histograms.With the `data` parameter it is 
    necessary to provide also a `label` parameter, that must have the same 
    dimension of `data`. `xlabel` and `ylabel` are string that are used to set a 
    label to the corresponding axis, while `xlim` is a tuple that help to limits 
    in the visualization of the plot. It is possible to set a title for the 
    plot with the `title` parameter and to set the log scale in y-axis with 
    `ylogscale` parameter.
    If the parameter `savepath` is given the image will be saved in the location provided.

    Parameters
    ----------
        data : (n,) array or sequence of (n,) arrays
            Input values, this takes either a single array or a sequence of arrays 
            which are not required to be of the same length.
        label : str or sequence of str
            Names of input values, this takes either a single string or a sequence of 
            strings depending on the dimension of the input data.
        xlabel : Optional[str], optional
            Label to be shown in the x-axis of the plot. Default to ''.
        ylabel : Optional[str], optional
            Label to be shown in the y-axis of the plot. Default to ''.
        xlim : Optional[tuple], optional
            Tuple containing the lower and upper limits to display in the plot. Default 
            to None
        title : Optional[str], optional 
            Title of the plot to visualize in the image. Defaults to ''.
        ylogscale : Optional[bool], optional 
            Parameter to set the log scale on y-axis. Defaults to False.
        savepath : Optional[str, path-like], optional 
            Parameter to chose the path and name where to save the file. If this
            parameter is not given, the image will not be saved. Defaults to None.
    """
    
    sns.set_style("darkgrid")
    if title != '':
        plt.title(title, fontsize=18)
    if ylabel != '':
        plt.ylabel(ylabel, fontsize=18)
    if xlabel != '':
        plt.xlabel(xlabel, fontsize=18)
    plt.hist(data, label=label, bins='auto', density=True)
    plt.subplots_adjust(top = 0.91, right=0.98, left=0.108, bottom=0.11)
    if ylogscale:
        plt.yscale('log')
    if xlim is not None:
        plt.xlim(xlim)
    plt.legend()
    if savepath is not None:
        plt.savefig(savepath, dpi = 400)
    plt.show()

def show_10_projectors(projectors: list, 
                       savepath: Optional[str] = None 
                       ):
    """    
    This function shows in a (2 rows x 5 columns) grid the ten projectors given as input.
    If the parameter `savepath` is given the image will be saved in the location provided.

    Parameters
    ----------
    projectors : list
        List containing ten matrices of the projectors to be shown. 
    savepath : Optional[str, path-like], optional
        Parameter to chose the path and name where to save the file. If this
        parameter is not given, the image will not be saved. Defaults to None.
    """
    rows, columns = 2, 5
    fig, axes = plt.subplots(rows, columns, figsize=(12,5))
    fig.subplots_adjust(top = 0.96, right=0.96, left=0.037, bottom=0.037, hspace=0.265, wspace=0.258)
    for i, ax in enumerate(axes.flat):
        ax.imshow(projectors[i])
        ax.set_title(f'Projector - {i}')
    if savepath is not None:
        plt.savefig(savepath, dpi=400)
    plt.show()

def plot_matrix_comparison(matrix1: np.ndarray,
                           matrix2: np.ndarray,
                           title1: Optional[str]= '',
                           title2: Optional[str]= '', 
                           pixels_to_be_masked: Optional[list] = None,
                           savepath: Optional[str] = None
                           ):
    """
    This function plots two matrices in the same image in order to compare them visually.
    With the parameter `pixels_to_be_masked` it is possible to set to zero some specific pixels
    of the first image to make it a more readable.
    If the parameter `savepath` is given the image will be saved in the location provided.

    Parameters
    ----------
    matrix1 : np.ndarray
        Matrix containing the data of the first image.
    matrix2 : np.ndarray
        Matrix containing the data of the second image.
    title1 : Optional[str], optional
        String containing the title of the first image. Defaults to ''.
    title2 : Optional[str], optional
        String containing the title of the second image. Defaults to ''.
    pixels_to_be_masked : Optional[list], optional
        List of tuples that contain the coordinates of the pixel to be masked of the first 
        image. Defaults to None. Ex. With the list `[(1, 2), (3, 4)]` are masked the pixel 
        whose coordinates are: `(1, 2)` and `(3, 4)`.
    savepath : Optional[bool], optional
        Parameter to chose the path and name where to save the file. If this
        parameter is not given, the image will not be saved. Defaults to None.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10,4))

    if pixels_to_be_masked is not None:
        for pixel in pixels_to_be_masked:
            matrix1[pixel[0]][pixel[1]] = 0
    shw1 = ax[0].imshow(matrix1)
    ax[0].set_title(title1)
    ax[0].grid(False)
    shw2 = ax[1].imshow(matrix2)
    ax[1].set_title(title2)
    ax[1].grid(False)
    plt.colorbar(shw1)
    plt.colorbar(shw2)
    fig.subplots_adjust(top = 0.96, right=0.98, left=0.04, bottom=0.05)
    sns.set_style("dark")
    if savepath is not None:
        plt.savefig(savepath, dpi=400)
    plt.show()