
<!DOCTYPE html>

<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" /><meta name="generator" content="Docutils 0.19: https://docutils.sourceforge.io/" />

    <title>Welcome to Hi-C-Analysis’s documentation! &#8212; Hi-C-Analysis 0.1.0 documentation</title>
    <link rel="stylesheet" type="text/css" href="_static/pygments.css" />
    <link rel="stylesheet" type="text/css" href="_static/alabaster.css" />
    <script data-url_root="./" id="documentation_options" src="_static/documentation_options.js"></script>
    <script src="_static/doctools.js"></script>
    <script src="_static/sphinx_highlight.js"></script>
    <link rel="index" title="Index" href="genindex.html" />
    <link rel="search" title="Search" href="search.html" />
   
  <link rel="stylesheet" href="_static/custom.css" type="text/css" />
  
  
  <meta name="viewport" content="width=device-width, initial-scale=0.9, maximum-scale=0.9" />

  </head><body>
  

    <div class="document">
      <div class="documentwrapper">
        <div class="bodywrapper">
          

          <div class="body" role="main">
            
  <section id="welcome-to-hi-c-analysis-s-documentation">
<h1>Welcome to Hi-C-Analysis’s documentation!<a class="headerlink" href="#welcome-to-hi-c-analysis-s-documentation" title="Permalink to this heading">¶</a></h1>
<div class="toctree-wrapper compound">
</div>
<section id="preprocessing-module">
<h2>Preprocessing Module<a class="headerlink" href="#preprocessing-module" title="Permalink to this heading">¶</a></h2>
<section id="the-create-graph-function">
<h3>The create_graph function<a class="headerlink" href="#the-create-graph-function" title="Permalink to this heading">¶</a></h3>
<dl class="py function">
<dt class="sig sig-object py" id="hicanalysis.preprocessing.create_graph">
<span class="sig-prename descclassname"><span class="pre">hicanalysis.preprocessing.</span></span><span class="sig-name descname"><span class="pre">create_graph</span></span><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">matrix_df</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">DataFrame</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">metadata_df</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">DataFrame</span></span></em><span class="sig-paren">)</span> <span class="sig-return"><span class="sig-return-icon">&#x2192;</span> <span class="sig-return-typehint"><span class="pre">Graph</span></span></span><a class="headerlink" href="#hicanalysis.preprocessing.create_graph" title="Permalink to this definition">¶</a></dt>
<dd><p>This function creates a nx.Graph object, starting from the adjacency matrix of the network and the label of each node.</p>
<section id="parameters">
<h4>Parameters<a class="headerlink" href="#parameters" title="Permalink to this heading">¶</a></h4>
<blockquote>
<div><dl class="simple">
<dt>matrix_df (pd.DataFrame): </dt><dd><p>Data Frame containing the adjacency matrix of the network.</p>
</dd>
<dt>metadata_df (pd.DataFrame): </dt><dd><p>Data Frame containing the metadata of the nodes.</p>
</dd>
</dl>
</div></blockquote>
</section>
<section id="returns">
<h4>Returns<a class="headerlink" href="#returns" title="Permalink to this heading">¶</a></h4>
<blockquote>
<div><dl class="simple">
<dt>nx.Graph: </dt><dd><p>Graph object of NetworkX library representing described by the adjacency contained in the <cite>matrix_df</cite>.</p>
</dd>
</dl>
</div></blockquote>
</section>
</dd></dl>

</section>
<section id="the-remove-empty-axis-function">
<h3>The remove_empty_axis function<a class="headerlink" href="#the-remove-empty-axis-function" title="Permalink to this heading">¶</a></h3>
<dl class="py function">
<dt class="sig sig-object py" id="hicanalysis.preprocessing.remove_empty_axis">
<span class="sig-prename descclassname"><span class="pre">hicanalysis.preprocessing.</span></span><span class="sig-name descname"><span class="pre">remove_empty_axis</span></span><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">matrix</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">ndarray</span></span></em><span class="sig-paren">)</span> <span class="sig-return"><span class="sig-return-icon">&#x2192;</span> <span class="sig-return-typehint"><span class="pre">ndarray</span></span></span><a class="headerlink" href="#hicanalysis.preprocessing.remove_empty_axis" title="Permalink to this definition">¶</a></dt>
<dd><p>This function returns the given matrix removing the rows and columns that contain only 0.</p>
<section id="id1">
<h4>Parameters<a class="headerlink" href="#id1" title="Permalink to this heading">¶</a></h4>
<blockquote>
<div><dl class="simple">
<dt>matrix (np.ndrray): </dt><dd><p>original matrix from which remove empty rows and columns (containing all zeros).</p>
</dd>
</dl>
</div></blockquote>
</section>
<section id="id2">
<h4>Returns<a class="headerlink" href="#id2" title="Permalink to this heading">¶</a></h4>
<blockquote>
<div><dl class="simple">
<dt>np.ndarray: </dt><dd><p>submatrix of the original one, where the empty rows and columns have been removed.</p>
</dd>
</dl>
</div></blockquote>
</section>
</dd></dl>

</section>
<section id="the-extract-diagonal-blocks-function">
<h3>The extract_diagonal_blocks function<a class="headerlink" href="#the-extract-diagonal-blocks-function" title="Permalink to this heading">¶</a></h3>
<dl class="py function">
<dt class="sig sig-object py" id="hicanalysis.preprocessing.extract_diagonal_blocks">
<span class="sig-prename descclassname"><span class="pre">hicanalysis.preprocessing.</span></span><span class="sig-name descname"><span class="pre">extract_diagonal_blocks</span></span><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">matrix</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">ndarray</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">block_size</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">list</span></span></em><span class="sig-paren">)</span> <span class="sig-return"><span class="sig-return-icon">&#x2192;</span> <span class="sig-return-typehint"><span class="pre">list</span></span></span><a class="headerlink" href="#hicanalysis.preprocessing.extract_diagonal_blocks" title="Permalink to this definition">¶</a></dt>
<dd><p>This function extract the diagonal blocks of dimension contained in the 
block_size, and return them under the form of a list of np.ndarray.</p>
<section id="id3">
<h4>Parameters<a class="headerlink" href="#id3" title="Permalink to this heading">¶</a></h4>
<blockquote>
<div><dl class="simple">
<dt>matrix (np.ndarray): </dt><dd><p>initial matrix from which we want to extract the blocks of the diagonal</p>
</dd>
<dt>block_size (list): </dt><dd><p>list containing the dimension of each block in the diagonal.</p>
</dd>
</dl>
</div></blockquote>
</section>
<section id="id4">
<h4>Returns<a class="headerlink" href="#id4" title="Permalink to this heading">¶</a></h4>
<blockquote>
<div><dl class="simple">
<dt>list: </dt><dd><p>list of np.ndarray containing the blocks of the diagonal extracted from the original matrix.</p>
</dd>
</dl>
</div></blockquote>
</section>
</dd></dl>

</section>
<section id="the-get-chromosome-list-function">
<h3>The get_chromosome_list function<a class="headerlink" href="#the-get-chromosome-list-function" title="Permalink to this heading">¶</a></h3>
<dl class="py function">
<dt class="sig sig-object py" id="hicanalysis.preprocessing.get_chromosome_list">
<span class="sig-prename descclassname"><span class="pre">hicanalysis.preprocessing.</span></span><span class="sig-name descname"><span class="pre">get_chromosome_list</span></span><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">metadata_df</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">DataFrame</span></span></em><span class="sig-paren">)</span> <span class="sig-return"><span class="sig-return-icon">&#x2192;</span> <span class="sig-return-typehint"><span class="pre">list</span></span></span><a class="headerlink" href="#hicanalysis.preprocessing.get_chromosome_list" title="Permalink to this definition">¶</a></dt>
<dd><p>This function, given a dataframe as input, containing the columns <cite>start</cite> and <cite>end</cite>, containing the 
number of row of start and end of each chromosome, returns a list with the difference between 
<cite>end</cite> element and <cite>start</cite> element.</p>
<section id="id5">
<h4>Parameters<a class="headerlink" href="#id5" title="Permalink to this heading">¶</a></h4>
<blockquote>
<div><dl class="simple">
<dt>metadata_df (pd.DataFrame): </dt><dd><p>Dataframe containing a column named <cite>start</cite>, a column <cite>end</cite> and a 
column with the name of the chromosome (not necessary).</p>
</dd>
</dl>
</div></blockquote>
</section>
<section id="id6">
<h4>Returns<a class="headerlink" href="#id6" title="Permalink to this heading">¶</a></h4>
<blockquote>
<div><dl class="simple">
<dt>list: </dt><dd><p>list containing the number of segments into which a chromosome has been divided.</p>
</dd>
</dl>
</div></blockquote>
</section>
</dd></dl>

</section>
<section id="the-ooe-normalization-function">
<h3>The ooe_normalization function<a class="headerlink" href="#the-ooe-normalization-function" title="Permalink to this heading">¶</a></h3>
<dl class="py function">
<dt class="sig sig-object py" id="hicanalysis.preprocessing.ooe_normalization">
<span class="sig-prename descclassname"><span class="pre">hicanalysis.preprocessing.</span></span><span class="sig-name descname"><span class="pre">ooe_normalization</span></span><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">matrix</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">ndarray</span></span></em><span class="sig-paren">)</span> <span class="sig-return"><span class="sig-return-icon">&#x2192;</span> <span class="sig-return-typehint"><span class="pre">ndarray</span></span></span><a class="headerlink" href="#hicanalysis.preprocessing.ooe_normalization" title="Permalink to this definition">¶</a></dt>
<dd><p>This function allows to normalize a given matrix with the Observed-Over-Expected (OOE) 
algorithm for a symmetric matrix. In particular this algorithm provides to divide each 
element for the mean of the diagonal it belongs to.</p>
<section id="id7">
<h4>Parameters<a class="headerlink" href="#id7" title="Permalink to this heading">¶</a></h4>
<blockquote>
<div><dl class="simple">
<dt>matrix (np.ndarray): </dt><dd><p>symmetric matrix to be normalized.</p>
</dd>
</dl>
</div></blockquote>
</section>
<section id="id8">
<h4>Returns<a class="headerlink" href="#id8" title="Permalink to this heading">¶</a></h4>
<blockquote>
<div><dl class="simple">
<dt>np.ndarray:     </dt><dd><p>matrix normalized with the OOE algorithm.</p>
</dd>
</dl>
</div></blockquote>
</section>
</dd></dl>

</section>
<section id="the-build-projectors-function">
<h3>The build_projectors function<a class="headerlink" href="#the-build-projectors-function" title="Permalink to this heading">¶</a></h3>
<dl class="py function">
<dt class="sig sig-object py" id="hicanalysis.preprocessing.build_projectors">
<span class="sig-prename descclassname"><span class="pre">hicanalysis.preprocessing.</span></span><span class="sig-name descname"><span class="pre">build_projectors</span></span><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">eigenvectors</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">number_of_projector_desired</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">int</span></span></em><span class="sig-paren">)</span> <span class="sig-return"><span class="sig-return-icon">&#x2192;</span> <span class="sig-return-typehint"><span class="pre">list</span></span></span><a class="headerlink" href="#hicanalysis.preprocessing.build_projectors" title="Permalink to this definition">¶</a></dt>
<dd><p>This function allows to get a list containing the projectors corresponding to the eigenvectors provided
as input. The eigenvectors must be contained in a np.array where the columns are the eigenvectors.</p>
<section id="id9">
<h4>Parameters<a class="headerlink" href="#id9" title="Permalink to this heading">¶</a></h4>
<dl class="simple">
<dt>eigenvectors (np.array) :</dt><dd><p>Array containing the in the columns the eigenvectors of a matrix.</p>
</dd>
<dt>number_of_projector_desired (int) :</dt><dd><p>Number of projectors that you want to build.</p>
</dd>
</dl>
</section>
<section id="id10">
<h4>Returns<a class="headerlink" href="#id10" title="Permalink to this heading">¶</a></h4>
<dl class="simple">
<dt>list: </dt><dd><p>Output list containing the projectors corresponding to the input eigenvectors.</p>
</dd>
</dl>
</section>
</dd></dl>

</section>
<section id="the-reconstruct-matrix-function">
<h3>The reconstruct_matrix function<a class="headerlink" href="#the-reconstruct-matrix-function" title="Permalink to this heading">¶</a></h3>
<dl class="py function">
<dt class="sig sig-object py" id="hicanalysis.preprocessing.reconstruct_matrix">
<span class="sig-prename descclassname"><span class="pre">hicanalysis.preprocessing.</span></span><span class="sig-name descname"><span class="pre">reconstruct_matrix</span></span><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">projectors</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">list</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">number_of_projectors</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">int</span></span></em><span class="sig-paren">)</span> <span class="sig-return"><span class="sig-return-icon">&#x2192;</span> <span class="sig-return-typehint"><span class="pre">ndarray</span></span></span><a class="headerlink" href="#hicanalysis.preprocessing.reconstruct_matrix" title="Permalink to this definition">¶</a></dt>
<dd><p>This function takes as input a list of matrices (projectors) and a number of projectors to be added
together in order to reconstruct the matrix.</p>
<section id="id11">
<h4>Parameters<a class="headerlink" href="#id11" title="Permalink to this heading">¶</a></h4>
<dl class="simple">
<dt>projectors (list) :</dt><dd><p>List containing np.ndarray of the projectors. It is supposed that the projectors are ordered 
from the most to the least important.</p>
</dd>
<dt>number_of_projectors (int) :</dt><dd><p>Number of projectors to be added together.</p>
</dd>
</dl>
</section>
<section id="id12">
<h4>Returns<a class="headerlink" href="#id12" title="Permalink to this heading">¶</a></h4>
<dl class="simple">
<dt>np.ndarray: </dt><dd><p>Reconstructed matrix from the addition of the first <cite>number_of_projectors</cite> projectors.</p>
</dd>
</dl>
</section>
<section id="raises">
<h4>Raises<a class="headerlink" href="#raises" title="Permalink to this heading">¶</a></h4>
<blockquote>
<div><dl class="simple">
<dt>ValueError: </dt><dd><p>When the <cite>number_of_projectors</cite> parameter is greater than the length of the <cite>projectors</cite> list.</p>
</dd>
</dl>
</div></blockquote>
</section>
</dd></dl>

</section>
</section>
<section id="visualizegraph-module">
<h2>Visualizegraph Module<a class="headerlink" href="#visualizegraph-module" title="Permalink to this heading">¶</a></h2>
<section id="the-weights-distribuion-function">
<h3>The weights_distribuion function<a class="headerlink" href="#the-weights-distribuion-function" title="Permalink to this heading">¶</a></h3>
<dl class="py function">
<dt class="sig sig-object py" id="hicanalysis.visualizegraph.weights_distribuion">
<span class="sig-prename descclassname"><span class="pre">hicanalysis.visualizegraph.</span></span><span class="sig-name descname"><span class="pre">weights_distribuion</span></span><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">matrix</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">ndarray</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">savepath</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">str</span><span class="w"> </span><span class="p"><span class="pre">|</span></span><span class="w"> </span><span class="pre">None</span></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span></em><span class="sig-paren">)</span><a class="headerlink" href="#hicanalysis.visualizegraph.weights_distribuion" title="Permalink to this definition">¶</a></dt>
<dd><p>This function, given an adjacency matrix of a weighted network, plots and shows 
the distribution of the weights of the network. If the parameter <cite>savepath</cite> is given 
the image will be saved in the location provided.</p>
<section id="id13">
<h4>Parameters<a class="headerlink" href="#id13" title="Permalink to this heading">¶</a></h4>
<blockquote>
<div><dl class="simple">
<dt>matrix<span class="classifier">np.ndarray </span></dt><dd><p>Adjacency matrix under the form of np.ndarray.</p>
</dd>
<dt>savepath<span class="classifier">Optional[str, path-like], optional </span></dt><dd><p>Parameter to chose the path and name where to save the file. If this
parameter is not given, the image will not be saved. Defaults to None.</p>
</dd>
</dl>
</div></blockquote>
</section>
</dd></dl>

</section>
<section id="the-degree-distribution-function">
<h3>The degree_distribution function<a class="headerlink" href="#the-degree-distribution-function" title="Permalink to this heading">¶</a></h3>
<dl class="py function">
<dt class="sig sig-object py" id="hicanalysis.visualizegraph.degree_distribution">
<span class="sig-prename descclassname"><span class="pre">hicanalysis.visualizegraph.</span></span><span class="sig-name descname"><span class="pre">degree_distribution</span></span><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">network_graph</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">Graph</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">savepath</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">str</span><span class="w"> </span><span class="p"><span class="pre">|</span></span><span class="w"> </span><span class="pre">None</span></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span></em><span class="sig-paren">)</span><a class="headerlink" href="#hicanalysis.visualizegraph.degree_distribution" title="Permalink to this definition">¶</a></dt>
<dd><p>This function, given a network graph, plots and shows the distribution of the degree of the nodes of the network.
If the parameter <cite>savepath</cite> is given the image will be saved in the location provided.</p>
<section id="id14">
<h4>Parameters<a class="headerlink" href="#id14" title="Permalink to this heading">¶</a></h4>
<blockquote>
<div><dl class="simple">
<dt>network_graph<span class="classifier">nx.Graph</span></dt><dd><p>Network graph under the form of object of the Graph class of Networkx library.</p>
</dd>
<dt>savepath<span class="classifier">Optional[str, path-like], optional </span></dt><dd><p>Parameter to chose the path and name where to save the file. If this
parameter is not given, the image will not be saved. Defaults to None.</p>
</dd>
</dl>
</div></blockquote>
</section>
</dd></dl>

</section>
<section id="the-plot-chromosome-hics-function">
<h3>The plot_chromosome_hics function<a class="headerlink" href="#the-plot-chromosome-hics-function" title="Permalink to this heading">¶</a></h3>
<dl class="py function">
<dt class="sig sig-object py" id="hicanalysis.visualizegraph.plot_chromosome_hics">
<span class="sig-prename descclassname"><span class="pre">hicanalysis.visualizegraph.</span></span><span class="sig-name descname"><span class="pre">plot_chromosome_hics</span></span><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">matrices</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">ndarray</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">names</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">ndarray</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">savepath</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">str</span><span class="w"> </span><span class="p"><span class="pre">|</span></span><span class="w"> </span><span class="pre">None</span></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span></em><span class="sig-paren">)</span><a class="headerlink" href="#hicanalysis.visualizegraph.plot_chromosome_hics" title="Permalink to this definition">¶</a></dt>
<dd><p>This function produce an image containing the HiC matrices of the 22+2 chromosomes of the DNA.
The function don’t apply any logarithm to the data.
If the parameter <cite>savepath</cite> is given the image will be saved in the location provided.</p>
<section id="id15">
<h4>Parameters<a class="headerlink" href="#id15" title="Permalink to this heading">¶</a></h4>
<blockquote>
<div><dl class="simple">
<dt>matrices<span class="classifier">np.ndarray </span></dt><dd><p>Array containing all the 24 matrices corresponding to the chromosomes</p>
</dd>
<dt>names<span class="classifier">np.ndarray</span></dt><dd><p>Array containing the names to label each plot</p>
</dd>
<dt>savepath<span class="classifier">Optional[str, path-like], optional </span></dt><dd><p>Parameter to chose the path and name where to save the file. If this
parameter is not given, the image will not be saved. Defaults to None.</p>
</dd>
</dl>
</div></blockquote>
</section>
</dd></dl>

</section>
<section id="the-plot-chromosomes-histograms-function">
<h3>The plot_chromosomes_histograms function<a class="headerlink" href="#the-plot-chromosomes-histograms-function" title="Permalink to this heading">¶</a></h3>
<dl class="py function">
<dt class="sig sig-object py" id="hicanalysis.visualizegraph.plot_chromosomes_histograms">
<span class="sig-prename descclassname"><span class="pre">hicanalysis.visualizegraph.</span></span><span class="sig-name descname"><span class="pre">plot_chromosomes_histograms</span></span><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">matrices</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">ndarray</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">names</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">ndarray</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">savepath</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">str</span><span class="w"> </span><span class="p"><span class="pre">|</span></span><span class="w"> </span><span class="pre">None</span></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span></em><span class="sig-paren">)</span><a class="headerlink" href="#hicanalysis.visualizegraph.plot_chromosomes_histograms" title="Permalink to this definition">¶</a></dt>
<dd><p>This function produce an image containing the histograms of the absolute value 
of the eigenvalues of the 22+2 chromosomes of the DNA.
The function don’t apply any logarithm to the data.
If the parameter <cite>savepath</cite> is given the image will be saved in the location provided.</p>
<section id="id16">
<h4>Parameters<a class="headerlink" href="#id16" title="Permalink to this heading">¶</a></h4>
<blockquote>
<div><dl class="simple">
<dt>matrices<span class="classifier">np.ndarray </span></dt><dd><p>Array containing all the 24 matrices corresponding to the chromosomes.</p>
</dd>
<dt>names<span class="classifier">np.ndarray </span></dt><dd><p>Array containing the names to label each plot.</p>
</dd>
<dt>savepath<span class="classifier">Optional[str, path-like], optional </span></dt><dd><p>Parameter to chose the path and name where to save the file. If this
parameter is not given, the image will not be saved. Defaults to None.</p>
</dd>
</dl>
</div></blockquote>
</section>
</dd></dl>

</section>
<section id="the-plot-matrix-function">
<h3>The plot_matrix function<a class="headerlink" href="#the-plot-matrix-function" title="Permalink to this heading">¶</a></h3>
<dl class="py function">
<dt class="sig sig-object py" id="hicanalysis.visualizegraph.plot_matrix">
<span class="sig-prename descclassname"><span class="pre">hicanalysis.visualizegraph.</span></span><span class="sig-name descname"><span class="pre">plot_matrix</span></span><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">matrix</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">ndarray</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">title</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">str</span><span class="w"> </span><span class="p"><span class="pre">|</span></span><span class="w"> </span><span class="pre">None</span></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">''</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">savepath</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">str</span><span class="w"> </span><span class="p"><span class="pre">|</span></span><span class="w"> </span><span class="pre">None</span></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span></em><span class="sig-paren">)</span><a class="headerlink" href="#hicanalysis.visualizegraph.plot_matrix" title="Permalink to this definition">¶</a></dt>
<dd><p>This function shows the image of the matrix with a colorbar. 
It is possible to set a title with the parameter title.
If the parameter <cite>savepath</cite> is given the image will be saved in the location provided.</p>
<section id="id17">
<h4>Parameters<a class="headerlink" href="#id17" title="Permalink to this heading">¶</a></h4>
<blockquote>
<div><dl class="simple">
<dt>matrix<span class="classifier">np.ndarray </span></dt><dd><p>Matrix in the form of a np.ndarray.</p>
</dd>
<dt>title<span class="classifier">Optional[str], optional  </span></dt><dd><p>Title to put in the image. Defaults to ‘’.</p>
</dd>
<dt>savepath<span class="classifier">Optional[str, path-like], optional </span></dt><dd><p>Parameter to chose the path and name where to save the file. If this
parameter is not given, the image will not be saved. Defaults to None.</p>
</dd>
</dl>
</div></blockquote>
</section>
</dd></dl>

</section>
<section id="the-scatter-plot-function">
<h3>The scatter_plot function<a class="headerlink" href="#the-scatter-plot-function" title="Permalink to this heading">¶</a></h3>
<dl class="py function">
<dt class="sig sig-object py" id="hicanalysis.visualizegraph.scatter_plot">
<span class="sig-prename descclassname"><span class="pre">hicanalysis.visualizegraph.</span></span><span class="sig-name descname"><span class="pre">scatter_plot</span></span><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">image1</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">ndarray</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">image2</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">ndarray</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">label1</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">str</span><span class="w"> </span><span class="p"><span class="pre">|</span></span><span class="w"> </span><span class="pre">None</span></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">'Image</span> <span class="pre">1'</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">label2</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">str</span><span class="w"> </span><span class="p"><span class="pre">|</span></span><span class="w"> </span><span class="pre">None</span></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">'Image</span> <span class="pre">2'</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">savepath</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">str</span><span class="w"> </span><span class="p"><span class="pre">|</span></span><span class="w"> </span><span class="pre">None</span></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span></em><span class="sig-paren">)</span><a class="headerlink" href="#hicanalysis.visualizegraph.scatter_plot" title="Permalink to this definition">¶</a></dt>
<dd><p>This function generate a scatter plot to compare two images or matrices. Given 
two numpy.ndarray, and optionally the label to assign to each image, it plots 
the scatter plot of the values of the images. If the parameter <cite>savepath</cite> is 
given the image will be saved in the location provided.</p>
<section id="id18">
<h4>Parameters<a class="headerlink" href="#id18" title="Permalink to this heading">¶</a></h4>
<blockquote>
<div><dl class="simple">
<dt>image1<span class="classifier">np.ndarray </span></dt><dd><p>First image, that lays on the x-axis.</p>
</dd>
<dt>image2<span class="classifier">np.ndarray  </span></dt><dd><p>Second image, that lays on the y-axis.</p>
</dd>
<dt>label1 Optional[str], optional  </dt><dd><p>Label to assign to the first image. Defaults to ‘Image 1’.</p>
</dd>
<dt>label2 Optional[str], optional  </dt><dd><p>Label to assign to the second image. Defaults to ‘Image 2’.</p>
</dd>
<dt>savepath<span class="classifier">Optional[str, path-like], optional  </span></dt><dd><p>Parameter to chose the path and name where to save the file. If this
parameter is not given, the image will not be saved. Defaults to None.</p>
</dd>
</dl>
</div></blockquote>
</section>
</dd></dl>

</section>
<section id="the-histogram-function">
<h3>The histogram function<a class="headerlink" href="#the-histogram-function" title="Permalink to this heading">¶</a></h3>
<dl class="py function">
<dt class="sig sig-object py" id="hicanalysis.visualizegraph.histogram">
<span class="sig-prename descclassname"><span class="pre">hicanalysis.visualizegraph.</span></span><span class="sig-name descname"><span class="pre">histogram</span></span><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">data</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">label</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">xlabel</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">str</span><span class="w"> </span><span class="p"><span class="pre">|</span></span><span class="w"> </span><span class="pre">None</span></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">''</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">ylabel</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">str</span><span class="w"> </span><span class="p"><span class="pre">|</span></span><span class="w"> </span><span class="pre">None</span></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">''</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">xlim</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">tuple</span><span class="w"> </span><span class="p"><span class="pre">|</span></span><span class="w"> </span><span class="pre">None</span></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">title</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">str</span></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">''</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">ylogscale</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">bool</span></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">False</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">savepath</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">str</span><span class="w"> </span><span class="p"><span class="pre">|</span></span><span class="w"> </span><span class="pre">None</span></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span></em><span class="sig-paren">)</span><a class="headerlink" href="#hicanalysis.visualizegraph.histogram" title="Permalink to this definition">¶</a></dt>
<dd><p>This function plot an histogram of the <cite>data</cite> with a predefined <cite>darkgrid</cite> style 
of seaborn. In the case <cite>data</cite> is an array, it will be shown the histogram of 
the values, if <cite>data</cite> is a sequence of array it will be shown an image 
containing all the corresponding histograms.With the <cite>data</cite> parameter it is 
necessary to provide also a <cite>label</cite> parameter, that must have the same 
dimension of <cite>data</cite>. <cite>xlabel</cite> and <cite>ylabel</cite> are string that are used to set a 
label to the corresponding axis, while <cite>xlim</cite> is a tuple that help to limits 
in the visualization of the plot. It is possible to set a title for the 
plot with the <cite>title</cite> parameter and to set the log scale in y-axis with 
<cite>ylogscale</cite> parameter.
If the parameter <cite>savepath</cite> is given the image will be saved in the location provided.</p>
<section id="id19">
<h4>Parameters<a class="headerlink" href="#id19" title="Permalink to this heading">¶</a></h4>
<blockquote>
<div><dl class="simple">
<dt>data<span class="classifier">(n,) array or sequence of (n,) arrays</span></dt><dd><p>Input values, this takes either a single array or a sequence of arrays 
which are not required to be of the same length.</p>
</dd>
<dt>label<span class="classifier">str or sequence of str</span></dt><dd><p>Names of input values, this takes either a single string or a sequence of 
strings depending on the dimension of the input data.</p>
</dd>
<dt>xlabel<span class="classifier">Optional[str], optional</span></dt><dd><p>Label to be shown in the x-axis of the plot. Default to ‘’.</p>
</dd>
<dt>ylabel<span class="classifier">Optional[str], optional</span></dt><dd><p>Label to be shown in the y-axis of the plot. Default to ‘’.</p>
</dd>
<dt>xlim<span class="classifier">Optional[tuple], optional</span></dt><dd><p>Tuple containing the lower and upper limits to display in the plot. Default 
to None</p>
</dd>
<dt>title<span class="classifier">Optional[str], optional </span></dt><dd><p>Title of the plot to visualize in the image. Defaults to ‘’.</p>
</dd>
<dt>ylogscale<span class="classifier">Optional[bool], optional </span></dt><dd><p>Parameter to set the log scale on y-axis. Defaults to False.</p>
</dd>
<dt>savepath<span class="classifier">Optional[str, path-like], optional </span></dt><dd><p>Parameter to chose the path and name where to save the file. If this
parameter is not given, the image will not be saved. Defaults to None.</p>
</dd>
</dl>
</div></blockquote>
</section>
</dd></dl>

</section>
<section id="the-show-10-projectors-function">
<h3>The show_10_projectors function<a class="headerlink" href="#the-show-10-projectors-function" title="Permalink to this heading">¶</a></h3>
<dl class="py function">
<dt class="sig sig-object py" id="hicanalysis.visualizegraph.show_10_projectors">
<span class="sig-prename descclassname"><span class="pre">hicanalysis.visualizegraph.</span></span><span class="sig-name descname"><span class="pre">show_10_projectors</span></span><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">projectors</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">list</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">savepath</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">str</span><span class="w"> </span><span class="p"><span class="pre">|</span></span><span class="w"> </span><span class="pre">None</span></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span></em><span class="sig-paren">)</span><a class="headerlink" href="#hicanalysis.visualizegraph.show_10_projectors" title="Permalink to this definition">¶</a></dt>
<dd><p>This function shows in a (2 rows x 5 columns) grid the ten projectors given as input.
If the parameter <cite>savepath</cite> is given the image will be saved in the location provided.</p>
<section id="id20">
<h4>Parameters<a class="headerlink" href="#id20" title="Permalink to this heading">¶</a></h4>
<dl class="simple">
<dt>projectors<span class="classifier">list</span></dt><dd><p>List containing ten matrices of the projectors to be shown.</p>
</dd>
<dt>savepath<span class="classifier">Optional[str, path-like], optional</span></dt><dd><p>Parameter to chose the path and name where to save the file. If this
parameter is not given, the image will not be saved. Defaults to None.</p>
</dd>
</dl>
</section>
</dd></dl>

</section>
<section id="the-plot-matrix-comparison-function">
<h3>The plot_matrix_comparison function<a class="headerlink" href="#the-plot-matrix-comparison-function" title="Permalink to this heading">¶</a></h3>
<dl class="py function">
<dt class="sig sig-object py" id="hicanalysis.visualizegraph.plot_matrix_comparison">
<span class="sig-prename descclassname"><span class="pre">hicanalysis.visualizegraph.</span></span><span class="sig-name descname"><span class="pre">plot_matrix_comparison</span></span><span class="sig-paren">(</span><em class="sig-param"><span class="n"><span class="pre">matrix1</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">ndarray</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">matrix2</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">ndarray</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">title1</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">str</span><span class="w"> </span><span class="p"><span class="pre">|</span></span><span class="w"> </span><span class="pre">None</span></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">''</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">title2</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">str</span><span class="w"> </span><span class="p"><span class="pre">|</span></span><span class="w"> </span><span class="pre">None</span></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">''</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">pixels_to_be_masked</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">list</span><span class="w"> </span><span class="p"><span class="pre">|</span></span><span class="w"> </span><span class="pre">None</span></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span></em>, <em class="sig-param"><span class="n"><span class="pre">savepath</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">str</span><span class="w"> </span><span class="p"><span class="pre">|</span></span><span class="w"> </span><span class="pre">None</span></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span></em><span class="sig-paren">)</span><a class="headerlink" href="#hicanalysis.visualizegraph.plot_matrix_comparison" title="Permalink to this definition">¶</a></dt>
<dd><p>This function plots two matrices in the same image in order to compare them visually.
With the parameter <cite>pixels_to_be_masked</cite> it is possible to set to zero some specific pixels
of the first image to make it a more readable.
If the parameter <cite>savepath</cite> is given the image will be saved in the location provided.</p>
<section id="id21">
<h4>Parameters<a class="headerlink" href="#id21" title="Permalink to this heading">¶</a></h4>
<dl class="simple">
<dt>matrix1<span class="classifier">np.ndarray</span></dt><dd><p>Matrix containing the data of the first image.</p>
</dd>
<dt>matrix2<span class="classifier">np.ndarray</span></dt><dd><p>Matrix containing the data of the second image.</p>
</dd>
<dt>title1<span class="classifier">Optional[str], optional</span></dt><dd><p>String containing the title of the first image. Defaults to ‘’.</p>
</dd>
<dt>title2<span class="classifier">Optional[str], optional</span></dt><dd><p>String containing the title of the second image. Defaults to ‘’.</p>
</dd>
<dt>pixels_to_be_masked<span class="classifier">Optional[list], optional</span></dt><dd><p>List of tuples that contain the coordinates of the pixel to be masked of the first 
image. Defaults to None. Ex. With the list <cite>[(1, 2), (3, 4)]</cite> are masked the pixel 
whose coordinates are: <cite>(1, 2)</cite> and <cite>(3, 4)</cite>.</p>
</dd>
<dt>savepath<span class="classifier">Optional[bool], optional</span></dt><dd><p>Parameter to chose the path and name where to save the file. If this
parameter is not given, the image will not be saved. Defaults to None.</p>
</dd>
</dl>
</section>
</dd></dl>

</section>
</section>
</section>
<section id="indices-and-tables">
<h1>Indices and tables<a class="headerlink" href="#indices-and-tables" title="Permalink to this heading">¶</a></h1>
<ul class="simple">
<li><p><a class="reference internal" href="genindex.html"><span class="std std-ref">Index</span></a></p></li>
<li><p><a class="reference internal" href="search.html"><span class="std std-ref">Search Page</span></a></p></li>
</ul>
</section>


          </div>
          
        </div>
      </div>
      <div class="sphinxsidebar" role="navigation" aria-label="main navigation">
        <div class="sphinxsidebarwrapper">
<h1 class="logo"><a href="#">Hi-C-Analysis</a></h1>








<h3>Navigation</h3>

<div class="relations">
<h3>Related Topics</h3>
<ul>
  <li><a href="#">Documentation overview</a><ul>
  </ul></li>
</ul>
</div>
<div id="searchbox" style="display: none" role="search">
  <h3 id="searchlabel">Quick search</h3>
    <div class="searchformwrapper">
    <form class="search" action="search.html" method="get">
      <input type="text" name="q" aria-labelledby="searchlabel" autocomplete="off" autocorrect="off" autocapitalize="off" spellcheck="false"/>
      <input type="submit" value="Go" />
    </form>
    </div>
</div>
<script>document.getElementById('searchbox').style.display = "block"</script>








        </div>
      </div>
      <div class="clearer"></div>
    </div>
    <div class="footer">
      &copy;2023, Lorenzo Barsotti.
      
      |
      Powered by <a href="http://sphinx-doc.org/">Sphinx 6.1.3</a>
      &amp; <a href="https://github.com/bitprophet/alabaster">Alabaster 0.7.13</a>
      
      |
      <a href="_sources/index.rst.txt"
          rel="nofollow">Page source</a>
    </div>

    

    
  </body>
</html>