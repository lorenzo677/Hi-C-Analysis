#%%
import networkx as nx
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from pathlib import Path
import hicanalysis.preprocessing as pre
import hicanalysis.visualizegraph as vg

#%%
def main():
    #%%
    # --------------
    # UPLOAD OF DATA
    # --------------    
    matrix_filepath = Path('../data/raw_GM12878_1Mb.csv')
    metadata_filepath = Path('../data/metadata.csv')
    ice_matrix_filepath = Path('../data/nrm_ICE_GM12878_1Mb.csv')

    df = pd.read_csv(matrix_filepath, header=None)
    metadata = pd.read_csv(metadata_filepath)
    
    adjacency = df.to_numpy()
    #%%
    # --------------------------------
    # NORMALIZATION OF THE FULL MATRIX
    # --------------------------------

    normalized_adj = pre.ooe_normalization(adjacency)
    normalized_adj = np.log(normalized_adj)
    normalized_adj[normalized_adj == -np.inf] = 0
    normalized_adj = pre.remove_empty_axis(normalized_adj)

    # vg.weights_distribuion(adjacency)
    #%%
    # ----------------------------------------------
    # NORMALIZATION OF THE MATRIX OF EACH CHROMOSOME
    # ----------------------------------------------

    list = pre.get_chromosome_list(metadata)

    blocks = pre.extract_diagonal_blocks(adjacency, list)
    normalized_blocks = []
    for i, block in enumerate(blocks):
        normalized_block = pre.ooe_normalization(block)
        normalized_block = pre.remove_empty_axis(normalized_block)
        normalized_block = np.log(normalized_block)
        normalized_block[normalized_block==-np.inf] = 0
        normalized_blocks.append(normalized_block)
        blocks[i] = pre.remove_empty_axis(block)

    #%%
    # -----------------------------------------------------------------
    # VISUALIZE ALL THE MATRICES OF THE CHROMOSOMES AND THE FULL MATRIX
    # -----------------------------------------------------------------

    vg.plot_matrix(adjacency, 'Original Adjacency Matrix')
    vg.plot_chromosome_hics(normalized_blocks, metadata.chr)
     
    #%% 
    # ---------------------------    
    # ANALYSIS OF EACH CHROMOSOME
    # ---------------------------

    # Analysis of the block number n
    n = 0
    H = nx.from_numpy_array(normalized_blocks[n])

    # VISUALIZE THE HISTOGRAMS OF THE EIGENVALUES

    # Create a random network with the same number of nodes to compare 
    random_graph = nx.gnp_random_graph(len(normalized_blocks[0]), 0.98, seed=42)
    adj_random_graph = nx.adjacency_matrix(random_graph)
    rn_eigenvalues = np.abs(np.linalg.eigvals(adj_random_graph.todense()))

    eigenvalues = np.abs(np.linalg.eigvals(normalized_blocks[0]))
    vg.histogram((rn_eigenvalues, eigenvalues), 
                 label=('Random Network', 'Chromosome 1'), 
                 xlabel=r'$p(|\lambda|$)', 
                 ylabel='Frequency', 
                 xlim=(-0.1, 20), 
                 title='Eigenvalues distribution of the chromosome 1',
                #   savepath='../images/histogram.png'
                 )

    vg.plot_chromosomes_histograms(normalized_blocks, 
                                   metadata.chr.values, 
                                #    savepath='../images/comparison_full.png'
                                   )
   

    # BETWEENNESS CENTRALITY
    bet_centrality = nx.betweenness_centrality(H, weight='weight', seed=42)
    vg.histogram([*bet_centrality.values()], 
                 label=f'Chromosome {n+1}', 
                 title='Betweenness Centrality', 
                 ylogscale=False,
                #   savepath='../images/bet_centrality.png'
                 )

    # DEGREE CENTRALITY
    deg_centrality = nx.degree_centrality(H)
    vg.histogram([*deg_centrality.values()], 
                 label=f'Chromosome {n+1}', 
                 title='Degree Centrality', 
                 ylogscale=True,
                #   savepath='../images/deg_centrality.png'
                 )
    #%%
    # ---------------------
    # MATRIX RECONSTRUCTION
    # ---------------------

    # EIGENVECTOR DECOMPOSITION 
    # eigenvalues, eigenvectors = np.linalg.eig(normalized_blocks[n])   # block n
    eigenvalues, eigenvectors = np.linalg.eig(normalized_adj)           # full matrix
    
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    # eigenvectors = np.array([eigenvectors[i] for i in idx])
    # ranked_eigenvector = eigenvectors[:,1] # select the first eigenvector
    
    # PROJECTORS CONSTRUCTION 
    projectors = pre.build_projectors(eigenvectors, 21)
    
    # MATRIX RECONSTRUCTION
    reconstructed_matrix = pre.reconstruct_matrix(projectors, i)
    # pixels = [(122,122), (122, 123), (123,122)]                       # pixel to be masked
  
    vg.plot_matrix_comparison(reconstructed_matrix, 
                              normalized_adj, 
                              'Reconstructed Matrix', 
                              'Original Matrix', 
                              # pixels_to_be_masked=pixels, 
                              # savepath='../images/comparison_full_{i}prj.png'
                              )
    
    vg.scatter_plot(reconstructed_matrix, 
                    normalized_adj, 
                    'Reconstructed', 
                    'Original', 
                    # savepath=f'../images/scatter_plot_full_ice_{i}prj.png'
                    )

    corr, p_value = pearsonr(reconstructed_matrix.flatten(), normalized_adj.flatten())
    print(f'Pearson {i}: {corr}')
  
#%%

if __name__ == "__main__":
    main()