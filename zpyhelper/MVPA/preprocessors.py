
"""Methods for preprocessing activity pattern matrix
"""

import numpy
from sklearn.covariance._shrunk_covariance import _oas
from sklearn.decomposition import PCA
from typing import Union
from functools import partial, reduce

####################### Data Splitting and Concatenation Methods #######################
def split_data(X:numpy.ndarray,groups=None,return_groups=False,select_groups:list=None,**kwargs) -> Union[list,tuple]:
    """split data into groups

    Parameters
    ----------
    X : numpy.ndarray
        data to be split
    groups : array, optional
        array specifying the which group each row of data belongs to, if `None`, all rows are assumed to be from the same group. by default None
    return_groups: bool, optional
        whether or not to return the group corresponding to the split
    select_groups:list, optional
        the (subset of) groups used to return. If `None`, will return the splits corresponding to all groups. If a list, will only return the splits corresponding to the specified groups

    Returns
    -------
    list or tuple
        data list or tuple of (data list, unique group) if `return_groups` is ``True``
        Each element of data list is the data of one group.
    """
    X=numpy.atleast_2d(X)
    assert X.ndim == 2, f"Expected X to be 2D, but got {X.ndim} instead"
    groups = numpy.ones((X.shape[0],)) if groups is None else numpy.atleast_1d(groups)
    assert groups.size==X.shape[0], "number of samples in X and groups does not match"
    unique_groups = numpy.unique(groups)
    
    select_groups = unique_groups if select_groups is None else select_groups
    assert all([g in unique_groups for g in select_groups]), "some of the specified groups do not exist!"

    if return_groups:
        return [X[groups.flatten()==j,:] for j in select_groups], select_groups
    else:
        return [X[groups.flatten()==j,:] for j in select_groups]   

def concat_data(Xlist:list)->numpy.ndarray:
    """concatenate list of data array into one array

    Parameters
    ----------
    Xlist : list
        list of data array, all of which must have same number of columns

    Returns
    -------
    numpy.ndarray
    """
    Xlist = [numpy.array(X) for X in Xlist if all([numpy.array(X).ndim==2,numpy.array(X).size>0])]
    return numpy.concatenate(Xlist,axis=0)


####################### Standardization and Centering Methods #######################
def scale_feature(X:numpy.ndarray,s_dir:int=2,standardize:bool=True) -> numpy.ndarray:
    """ standardize or center a 1D or 2D numpy array by using ZX = (X - mean)/std

    Parameters
    ----------
    X : numpy.ndarray
        the 1D or 2D numpy array that needs to be normalized
    s_dir : int, optional
        the direction along which to perform standardization\n
        if 0, will perfrom standardization independently for each row  \n
        if 1, will perform standardization independently for each column  \n      
        if 2, will perform standardization on the whole matrix  \n
        by default 2
    standardize: bool, optional
        whether or not to standardize, by default `True`

    Returns
    -------
    numpy.ndarray
        standardized 1D or 2D array ZX
    """
    assert isinstance(X,numpy.ndarray), "X must be numpy array"
    assert X.ndim <= 2, "X must be 1D or 2D"

    if X.ndim == 1:
        s_dir = 2

    if s_dir == 0:
        ZX = _rowwise_standardize(X,standardize)
    elif s_dir == 1:
        X = X.T
        ZX = _rowwise_standardize(X,standardize)
        ZX = ZX.T
    elif s_dir == 2:
        denom = numpy.std(X) if standardize else 1
        ZX = (X - numpy.mean(X)) / denom
    return ZX

def _rowwise_standardize(X:numpy.ndarray,standardize:bool) -> numpy.ndarray:
    """standarderdize or center an array rowwise

    Parameters
    ----------
    X : numpy.ndarray
    standardize: bool
        whether or not to standardize

    Returns
    -------
    numpy.ndarray
        _description_
    """
    row_means = X.mean(axis=1)
    row_stds  = X.std(axis=1)
    denom = row_stds[:, numpy.newaxis] if standardize else 1
    return (X - row_means[:, numpy.newaxis]) / denom

def average_odd_even_session(activitypatternmatrix:numpy.ndarray,session:numpy.ndarray,**kwargs) -> numpy.ndarray:
    """average data in odd and even session separately

    Parameters
    ----------
    activitypatternmatrix : numpy.ndarray
        data matrix of shape `(nsample,nvoxel)`
    session : numpy.ndarray
        array specifying the which session each row of data belongs to.

    Returns
    -------
    numpy.ndarray
        transformed data matrix
    """
    assert activitypatternmatrix.shape[0] == session.size
    X_list  = split_data(activitypatternmatrix,session)
    X_odd   = numpy.mean(X_list[0::2],axis=0)
    X_even  = numpy.mean(X_list[1::2],axis=0) if len(X_list)>1 else []
    return concat_data([X_odd,X_even])

def average_flexi_session(activitypatternmatrix:numpy.ndarray,session:numpy.ndarray,average_by:list) -> numpy.ndarray:
    """average data in by flexible session separately

    Parameters
    ----------
    activitypatternmatrix : numpy.ndarray
        data matrix of shape `(nsample,nvoxel)`
    session : numpy.ndarray
        array specifying the which session each row of data belongs to.
    average_by : list
        list specifying groups of sessions that will be averaged across. \n
        For example, `average_by = [[0,1],[2,3]]` will return the average activity pattern of session 0 and 1, and the average activity pattern of session 2 and 3

    Returns
    -------
    numpy.ndarray
        transformed data matrix
    """
    assert activitypatternmatrix.shape[0] == session.size
    X_list,unique_sess  = split_data(activitypatternmatrix,session,return_groups=True)
    unique_sess = list(unique_sess)
    assert all([s in unique_sess for s in sum(average_by,[])]), "some of the specified session do not exist!"
    X_aves = [numpy.mean([X_list[unique_sess.index(s)] for s in ss],axis=0) for ss in average_by]
    return concat_data(X_aves)

####################### Multivariate-Noise-Normalisation Methods #######################
def _estimate_whitening_from_resid(res) -> numpy.ndarray:
    """estimate a whitening matrix from a residual matrix with Oracle Approximating Shrinkage method
    similar to the implementation@https://osf.io/dzqs2

    Parameters
    ----------
    res : numpy.ndarray
        residual matrix of size ``(nsample,nfeature)``

    Returns
    -------
    numpy.ndarray
        whitening matrix of size ``(nfeature,nfeature)``
    """
    res = numpy.array(res)
    shrunk_cov,_ = _oas(res-res.mean(0),assume_centered=True) 
    D, V = numpy.linalg.eigh(shrunk_cov)
    preci_sqrt = (V * (1/numpy.sqrt(D))) @ V.T 
    return preci_sqrt    

def normalise_multivariate_noise(activitypatternmatrix:numpy.ndarray,
                                 residualmatrix:numpy.ndarray,
                                 ap_groups:numpy.ndarray=None,
                                 resid_groups:numpy.ndarray=None,
                                 **kwargs) -> numpy.ndarray:
    """perform multivariate noise normalisation on the activity pattern matrix.
    The activity pattern matrix and residual matrix is first split into groups. \n
    For each group of data, a whitening matrix is estimated from the residual matrix and then applied to the acitvity pattern matrix. \n
    Splits are then concatenated back to the original shape.
    similar to the implementation@https://osf.io/3k759

    Parameters
    ----------
    activitypatternmatrix : numpy.ndarray
        data matrix of shape `(n_data_sample,nvoxel)`
    residualmatrix : numpy.ndarray
        residual matrix of shape `(n_residual_sample,nvoxel)`
    ap_groups : numpy.ndarray
        array specifying the which group each row of data belongs to, if `None`, all rows are assumed to be from the same group. by default None
    resid_groups : numpy.ndarray
        array specifying the which group each row of residual matrix belongs to, if `None`, all rows are assumed to be from the same group. by default None

    Returns
    -------
    numpy.ndarray
        whitened data matrix
    """
    assert activitypatternmatrix.shape[1] == residualmatrix.shape[1], "data matrix and residual matrix must have the same number of voxels"

    ap_groups    = numpy.ones((activitypatternmatrix.shape[0],)) if ap_groups is None else ap_groups
    resid_groups = numpy.ones((residualmatrix.shape[0],)) if resid_groups is None else resid_groups
    assert numpy.unique(ap_groups).size == numpy.unique(resid_groups).size 
    assert activitypatternmatrix.shape[0] == ap_groups.size
    assert residualmatrix.shape[0] == resid_groups.size

    res_list = split_data(residualmatrix,resid_groups)
    X_list   = split_data(activitypatternmatrix,ap_groups)
    wm_list  = [_estimate_whitening_from_resid(res) for res in res_list]
    X_whiten = [X@W for X,W in zip(X_list,wm_list)]
    return numpy.concatenate(X_whiten,axis=0)

####################### Dimensionality Reduction Methods #######################
def extract_pc(activitypatternmatrix:numpy.ndarray,n_components=None,**kwargs) -> numpy.ndarray:
    """Perform PCA and transform the data accordingly

    Parameters
    ----------
    activitypatternmatrix : numpy.ndarray
        data matrix of shape `(n_data_sample,nvoxel)`
    n_components : _type_, optional
        if `n_components>1`, it specifies the number of components. \n 
        if `n_components<1`, it specifies the percentage of variance explained. 
        if `None`, all pcs will be kept.
        by default None

    Returns
    -------
    numpy.ndarray
        transformed data matrix
    """
    if n_components is None:
        return PCA().fit_transform(activitypatternmatrix) 
    else:
        return PCA(n_components=n_components).fit_transform(activitypatternmatrix)


########################### Alignment Methods ###########################
def rmsd(X:numpy.ndarray,Y:numpy.ndarray)->float:
    """
    Calculate the root mean square deviation between two matrices

    Parameters
    ----------
    X : numpy.ndarray
        Matrix 1
    Y : numpy.ndarray
        Matrix 2
    
    Returns
    -------
    float
        Root mean square deviation between X and Y
    """
    assert X.shape == Y.shape, "Matrix dimensions must match"
    return numpy.sqrt( numpy.mean( (X-Y)**2 ) )

def kabsch_algorithm(P:numpy.ndarray,Q:numpy.ndarray, translation:bool=False)->tuple:
    """use Kabsch algorithm to compute the optimal rotation matrix that aligns P to Q.   \n
    P and Q are NxK matrices where K is the dimensionality and N is the number of points.
    Reference: https://github.com/charnley/rmsd/blob/master/rmsd/calculate_rmsd.py

    The algorithm is based on the following steps: \n
    Parameters
    ----------
    P : numpy.ndarray
        source matrix that needed to be aligned to Q
    Q : numpy.ndarray
        reference matrix that P will be aligned to
    translation : bool, optional
        if True, will use centroids to translate P and Q onto each other, by default False

    Returns
    -------
    tuple
        _description_
    """

    #check the dimensionality of the input matrices
    assert P.shape == Q.shape, "Matrix dimensions must match"
    
    if translation:
        P = P - P.mean(0)
        Q = Q - Q.mean(0)
    
    #compute the covariance matrix
    C = numpy.dot(P.T,Q)

    # SVD on the covariance matrix
    U, S, Vt = numpy.linalg.svd(C)

    # Validate right-handed coordinate system
    if numpy.linalg.det(U)*numpy.linalg.det(Vt) < 0.0:
        U[:, -1] = -U[:, -1]

    # Optimal rotation
    R = numpy.dot(U, Vt)
    Phat = P@R #np.dot(P,R)+centroid_Q-centroid_P

    return Phat, R, rmsd(Phat,Q)

########################### Chaining Functions ###########################
def chain_steps(*varargin):
    """Chaining different preprocessing functions together and compose it into one preprocessing function. \n
    parameters for different preprocessing steps will also be set. \n
    The composed function will only take the activity pattern matrix as input.\n
    The composed function will apply the steps sequentially
    Each argument must be a tuple of length `2` and specify `(function, function_fixed_argument)`: \n
        - `function`: a callable that takes the activity pattern as input and return preprocessed activity pattern
        - `function_fixed_argument`: a dictionary the contains arguments apart from the activity pattern matrix that will be passed to the callable \n
    For example: \n
    The following code \n
    ```
    step1 = (func1,param1)
    step2 = (func2,param2)
    preproc_func = chain_steps(step1,step2)`
    X_preproc = preproc_func(X)
    ```
    is equivalent to:\n
    ```
    X_preproc = func2(func1(X,**param1),**param2)
    ```

    Returns
    -------
    callable
        composed preprocessing function
    """
    steps = list(varargin)
    for j in range(len(steps)):
        if len(steps[j])==1:
            steps[j].append({})
    step_funcs = [partial(step[0],**step[1]) for step in steps if len(step)==2]
    chained_func = partial(reduce, lambda x,func: func(x), step_funcs)
    return chained_func

# to test the effect of chaining, run the following
# steps = [
#     [scale_feature],
#     [extract_pc],
#     [average_odd_even_session,{"session":numpy.concatenate([numpy.ones((25,))*j for j in range(4)])}],
# ]
# X = numpy.random.random((100,50))
# chain_steps(*steps)(X)

# chaining can also be done if step_funcs are returned instead of chained_func
#Xpreproc=X
#for sfunc in chain_steps(*steps):
#    Xpreproc = sfunc(Xpreproc)