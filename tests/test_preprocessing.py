"""
Test_preprocessing
------------------
In this module there are functions to test the preprocessing module.
"""
#%%
import hicanalysis.preprocessing as pre
import numpy as np
import pandas as pd

def generate_symmetric_matrix() -> np.ndarray:
    """
    This function generate a symmetric square random matrix only for testing purpose.
    It decide randomly from a 'discrete uniform' distribution a number of rows (and columns) 
    in the half-open interval `[2, 5000)`, then it randomly fill a matrix with the chosen dimension
    by using another 'discrete uniform' distribution in the half-open interval `[0, 10000)`.

    Returns
    -------
    np.ndarray: symmetric square random matrix.
    """
    dim = 10
    np.random.seed(42)
    matrix = np.random.randint(0, 10000, (dim, dim))
    matrix += matrix.T
    return matrix

def test_matrix_square():
    """
    This test function tests if the function `generate_symmetric_matrix()`, returns a square matrix.

    Given
    -----
    matrix: any matrix.
    
    Tests
    -----
    The matrix is a square matrix.
    """
    matrix = generate_symmetric_matrix()
    assert(matrix.shape[0]==matrix.shape[1])

def test_matrix_symmetry():
    """
    This test function tests if the function `generate_symmetric_matrix()`, returns a symmetric matrix.

    Given
    -----
    matrix: any square matrix.
    
    Tests
    -----
    The square matrix is symmetric.
    """
    matrix = generate_symmetric_matrix()
    assert((matrix == matrix.T).all())

def test_shape_normalization():
    """
    This test function tests if the function `ooe_normalization()` of the module `preprocessing`, returns a
    normalized matrix with the same shape of the original one

    Given
    -----
    matrix: any square symmetric matrix.
    
    Tests
    -----
    The normalized matrix has the same shape of the original one.
    """
    matrix = generate_symmetric_matrix()
    normalized_matrix = pre.ooe_normalization(matrix)
    assert(matrix.shape == normalized_matrix.shape)

def test_normalization():
    """
    This test function tests if the `ooe_normalization()` function of the module `preprocessing`, returns a
    normalized matrix compatible with the expected result. In this test a matrix (4x4) is used.

    Given
    -----
    matrix: any square symmetric matrix.
    
    Tests
    -----
    The normalized matrix is equal to the expected result.
    """
    matrix = np.array([[0, 1, 4, 6], [1, 0, 5, 7], [4, 5, 0, 2], [6, 7, 2, 0]])
    normalized_matrix = pre.ooe_normalization(matrix)
    expected = np.array([[0, 3/8, 8/11, 1], [3/8, 0, 15/8, 14/11], [8/11, 15/8, 0, 3/4], [1, 14/11, 3/4, 0]])
    assert((normalized_matrix == expected).all())

def test_normalization_with_empty_axis():
    """
    This test function tests if the `ooe_normalization()` function of the module `preprocessing`, returns a
    normalized matrix compatible with the expected result in the case that a row and a column are empty. 
    In this test a matrix (4x4) is used.

    Given
    -----
    matrix: square symmetric matrix with a row (and a column) filled with 0.
    
    Tests
    -----
    The normalized matrix is equal to the expected result.
    """
    matrix = np.array([[0, 0, 0, 0], [0, 0, 5, 7], [0, 5, 0, 2], [0, 7, 2, 0]])
    normalized_matrix = pre.ooe_normalization(matrix)
    expected = np.array([[0, 0, 0, 0], [0, 0, 15/7, 2], [0, 15/7, 0, 6/7], [0, 2, 6/7, 0]])
    assert((normalized_matrix == expected).all())

def test_normalization_identity_matrix():
    """
    This test function tests if the `ooe_normalization()` function of the module `preprocessing`, returns a
    normalized matrix compatible with the expected result for a identity matrix. In this test a matrix (4x4) is used.

    Given
    -----
    matrix: identity matrix.
    
    Tests
    -----
    The normalized matrix is equal to the expected result.
    """
    matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    normalized_matrix = pre.ooe_normalization(matrix)
    assert((normalized_matrix == matrix).all())


def test_remove_empty_axis_with_zero():
    """
    This test function tests if the `remove_empty_axis()` function of the module `preprocessing`, returns a
    matrix without the rows and columns containing only zeros, compatible with the expected result. 
    In this test a matrix (4x4) with a row and a column filled with 0 is used.

    Given
    -----
    matrix: a square symmetric matrix with a column and a row filled with 0.
    
    Tests
    -----
    The matrix returned by the `remove_empty_axis()` function is equal to the expected result, that is the original 
    matrix where the rows and columns containing only zeros have been removed.
    """
    matrix = np.array([[0, 0, 0, 0], [0, 0, 5, 7], [0, 5, 0, 2], [0, 7, 2, 0]])
    cleaned_matrix = pre.remove_empty_axis(matrix)
    expected = np.array([[0, 5, 7], [5, 0, 2], [7, 2, 0]])
    assert((expected == cleaned_matrix).all())

def test_remove_empty_axis_without_zero():
    """
    This test function tests if the `remove_empty_axis()` function of the module `preprocessing`, returns a
    matrix without the rows and columns containing only zeros, compatible with the expected result. 
    In this test a matrix (4x4) without empty rows and columns is used.

    Given
    -----
    matrix: any square symmetric matrix.
    
    Tests
    -----
    The matrix returned by the `remove_empty_axis()` function is equal to the expected result.
    """
    matrix = np.array([[0, 1, 0, 0], [1, 0, 5, 7], [0, 5, 0, 2], [0, 7, 2, 0]])
    cleaned_matrix = pre.remove_empty_axis(matrix)
    assert((matrix == cleaned_matrix).all())

def test_diagonal_blocks_shape():
    """
    This test function tests if the `extract_diagonal_blocks()` function of the module `preprocessing`, returns a
    list of square matrices whose dimension are equal to the one contained in the list provided to the 
    `extract_diagonal_blocks()` function.

    Given
    -----
    matrix: any square symmetric matrix.
    
    Tests
    -----
    The matrices contained inside the list returned by the `extract_diagonal_blocks()` function have a shape 
    equal to the one contained in the provided list.
    """
    np.random.seed(42)
    matrix = np.random.randint(0, 10000, (10, 10))
    list = [2, 3, 5]
    blocks = pre.extract_diagonal_blocks(matrix, list)
    assert(blocks[0].shape == (list[0], list[0]))
    assert(blocks[1].shape == (list[1], list[1]))
    assert(blocks[2].shape == (list[2], list[2]))

def test_diagonal_blocks():
    """
    This test function tests if the `extract_diagonal_blocks()` function of the module `preprocessing`, returns a
    list of square matrices containing the blocks on the diagonal of the input matrix.

    Given
    -----
    matrix: any square matrix.
    
    Tests
    -----
    The matrices contained inside the list returned by the `extract_diagonal_blocks()` function have a shape 
    equal to the one contained in the provided list.
    """
    
    matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4], [5, 6, 7, 8]])
    list = [1, 3]
    blocks = pre.extract_diagonal_blocks(matrix, list)
    expected = [np.array([[1]]), np.array([[6, 7, 8], [2, 3, 4], [6, 7, 8]])]
    assert((blocks[0] == expected[0]).all().all())
    

def test_get_chromosome_list():
    """
    This test function tests if the `get_chromosome_list()` function of the module `preprocessing`, returns a
    list of numbers equals to the difference between the columns he `end` and the `start` of a dataframe given as input.

    Given
    -----
    dataframe: pd.DataFrame with a column `start` and a column `end`.
    
    Tests
    -----
    The `get_chromosome_list()` returns a list containing the dimension of the matrices to extract from the original
    matrix, that is the difference between the column `end` and the column `start` plus 1.
    """
    dataframe = pd.DataFrame({'start': [1, 4, 7], 'end': [3, 8, 12]})
    expected = [3, 5, 6]
    obtained_list = pre.get_chromosome_list(dataframe)
    assert(expected == obtained_list)

def test_length_projectors():
    """
    This test function tests if the `build_projectors()` function of the module `preprocessing`, returns a
    list of projectors whose dimension is equal to the number of eigenvectors given as input. In this case we tested
    the case with three eigenvectors given as input.

    Given
    -----
    eig: np.array containing in the columns the eigenvectors of a matrix. 
    
    Tests
    -----
    If the `build_projectors()` function returns a list that have the same length as the number of eigenvectors passed as input.
    """
    eig = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 6]])
    projectors = pre.build_projectors(eig, 3)
    assert(len(projectors) == 3)

def test_projectors():
    """
    This test function tests if the `build_projectors()` function of the module `preprocessing`, returns a
    list of projectors. It is known that a projector `P` is idempotent, so in this test it is verified that
    `P^2=P`.

    Given
    -----
    eig: np.array containing in the columns the eigenvectors of a matrix. 
    
    Tests
    -----
    If the `build_projectors()` function returns a list of projectors.
    """
    eig = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 6]])
    projectors = pre.build_projectors(eig, 3)
    for projector in projectors:
        assert((np.dot(projector, projector) == projector).all())

def generate_list_of_matrices() -> list:
    """
    This function is used only for testing purpose and it is used to generate a list of matrices.
    This function extract randomly a number of matrices to be inserted in the list, then
    the function `generate_symmetric_matrix()` is called.

    Returns
    -------
    list
        list of matrices.
    """
    
    dim = 5
    list = []
    for _ in range(dim):
        list.append(generate_symmetric_matrix())
    return list

def test_generate_list_of_matrices():
    """
    This test function tests if the `generate_list_of_matrices()` function, returns a
    list object.
    
    Tests
    -----
    If the `generate_list_of_matrices()` function returns a list.
    """
    matrices = generate_list_of_matrices()
    assert(type(matrices)==list)

def test_reconstructed_matrix_shape():
    """
    This test function tests if the `reconstruct_matrix()` function of the module `preprocessing`, returns a
    matrix with the same shape of the matrices contained in the input.
    
    Given
    -----
    matrices: list of matrices of the same shape.

    Tests
    -----
    The shape of the reconstructed matrix is equal to the shape of one matrix of the matrices list given
    as input.
    """
    matrices = generate_list_of_matrices()
    reconstructed = pre.reconstruct_matrix(matrices, 4)
    assert(reconstructed.shape == matrices[0].shape)

def test_symmetry_reconstructed_matrix():
    """
    This test function tests if the `reconstruct_matrix()` function of the module `preprocessing`, returns a
    symmetric matrix.
    
    Given
    -----
    matrices: list of matrices of the same shape.

    Tests
    -----
    The reconstructed matrix is symmetric.
    """
    matrices = generate_list_of_matrices()
    reconstructed = pre.reconstruct_matrix(matrices, 4)
    assert((reconstructed == reconstructed.T).all())

def test_reconstructed_matrix():
    """
    This test function tests if the `reconstruct_matrix()` function of the module `preprocessing`, returns a
    matrix that is the sum, element by element, of the ones given as input.
    
    Given
    -----
    matrices: list of matrices of the same shape.

    Tests
    -----
    The reconstructed matrix is equal to the sum of the matrices given as input.
    """
    matrices = [np.array([[0, 1], [2, 3]]), np.array([[2, 3], [4, 5]])]
    expected = np.array([[2, 4], [6, 8]])
    reconstructed = pre.reconstruct_matrix(matrices, 2)
    assert((expected == reconstructed).all())

