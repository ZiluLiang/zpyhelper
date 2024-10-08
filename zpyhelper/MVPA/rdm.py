"""
This module contains helper function for computing representation (dis)similarity matrix

Zilu Liang @HIPlab Oxford
2024
"""

from scipy.spatial.distance import pdist, squareform
import numpy
from typing import Union
from sklearn.linear_model import LinearRegression
from .preprocessors import scale_feature

def check_array(x,ensure_dim:int=2,min_sample=1,min_feature=1) -> numpy.ndarray:
    """check if the dimensionality of array satisfies criteria

    Parameters
    ----------
    x : object
        the object to be transformed in to numpy.ndarray
    ensure_dim: int
        ensure dimensionality of the transformed array, by default ``1``
    min_sample : int, optional
        minimum number of samples (length of the first axis), by default ``1``
    min_feature : int, optional
        minimum number of features (length of the second axis), only matters if ``ensure_dim==2``, by default ``1``

    Returns
    -------
    numpy.ndarray
    """
    x = numpy.array(x)    
    ensure_dim = int(ensure_dim)
    assert ensure_dim>0, "ensure_dim must be positive integer"
    assert x.shape[0]>=min_sample, f"Expected at least {min_sample} sample"    
    assert x.ndim == ensure_dim, f"Expected 2D array, got a {x.ndim}D array instead"
    if ensure_dim == 2:
        assert x.shape[1]>=min_feature, f"Expected at least {min_feature} feature"        
    return x


def lower_tri(rdm:numpy.ndarray) -> tuple:
    """return the lower triangular part of the RDM, excluding the diagonal elements

    Parameters
    ----------
    rdm : numpy.ndarray
        the representation dissimilarity matrix of size ``(nsample,nsample)``

    Returns
    -------
    tuple: (rdm_tril,lower_tril_idx)
        rdm_tril: the lower triangular part of the RDM, excluding the diagonal elements
        lower_tril_idx: the index of the lower triangular part of the RDM, excluding the diagonal elements
    """
    rdm = check_array(rdm,ensure_dim=2,min_sample=2,min_feature=2)
    assert rdm.shape[0] == rdm.shape[1], "rdm must be of size ``(nsample,nsample)``"
    
    lower_tril_idx = numpy.tril_indices(rdm.shape[0], k = -1)
    rdm_tril = rdm[lower_tril_idx]
    return rdm_tril,lower_tril_idx

def upper_tri(rdm:numpy.ndarray) -> tuple:
    """return the upper triangular part of the RDM, excluding the diagonal elements

    Parameters
    ----------
    rdm : numpy.ndarray
        the representation dissimilarity matrix. of size ``(nsample,nsample)``

    Returns
    -------
    tuple: (rdm_tril,upper_tril_idx)
        rdm_tril: a 1D numpy array. the upper triangular part of the RDM, excluding the diagonal elements
        upper_tril_idx: a 1D numpy array. the index of the upper triangular part of the RDM, excluding the diagonal elements
    """

    rdm = check_array(rdm,ensure_dim=2,min_sample=2,min_feature=2)
    assert rdm.shape[0] == rdm.shape[1], "rdm must be of size ``(nsample,nsample)``"

    upper_triu_idx = numpy.triu_indices(rdm.shape[0], k = 1)
    rdm_triu = rdm[upper_triu_idx]
    return rdm_triu,upper_triu_idx

def compute_R2(y_pred:numpy.ndarray, y_true:numpy.ndarray, nparam: int) -> tuple:
    """compute the coefficient of determination (R-square or adjusted R-square) of a model based on prediction and true value
    based on formula in https://en.wikipedia.org/wiki/Coefficient_of_determination

    Parameters
    ----------
    y_pred : numpy.ndarray
        1D numpy array of predicted y values
    y_true : numpy.ndarray
        1D numpy array of true (observed) y values
    nparam : int
        number of parameters in the model

    Returns
    -------
    tuple
        a tuple of (r-squared, adjusted r-squared)
    """
    SS_Residual = numpy.sum((y_true-y_pred)**2)       
    SS_Total = numpy.sum((y_true-numpy.mean(y_true))**2)     
    R_squared = 1 - SS_Residual/SS_Total
    n_sample = len(y_true)
    adjusted_R_squared = 1 - (1-R_squared)*(n_sample-1)/(n_sample-nparam)
    return R_squared, adjusted_R_squared

def compute_rdm(pattern_matrix:numpy.ndarray,metric:str) -> numpy.ndarray:
    """compute the dissimilarity matrix of a nsample x nfeature matrix.

    Parameters
    ----------
    pattern_matrix : numpy.ndarray
        pattern_matrix containing samples' features, of size ``(nsample,nfeature)``
    metric : str
        dissimilarity/distance metric passed to `scipy.spatial.distance.pdist`

    Returns
    -------
    numpy.ndarray
        dissimliarity matrix of size ``(nsample,nsample)``

    Raises
    ------
    Exception
        pattern matrix must be 2D with minimum of 2 samples and 1 feature
    """
    # check X 
    X = check_array(pattern_matrix,ensure_dim=2,min_sample=2,min_feature=1)
    return squareform(pdist(X, metric=metric)) 

def nan_equal(a:numpy.ndarray,b:numpy.ndarray)->numpy.ndarray:
    """check element-wise if two numpy array are equal. nan value comparisons will result in nan. In the result array, boolean values are converted in to 1 and 0.

    Parameters
    ----------
    a : numpy.ndarray
        Input array
    b : numpy.ndarray
        Input array

    Returns
    -------
    numpy.ndarray
        The result array from comparison, has the same shape as input arrays.
    """
    res = 1.*(a == b)
    nan_filter = numpy.any([numpy.isnan(a),numpy.isnan(b)],axis=0)
    res[nan_filter] = numpy.nan
    return res

def compute_rdm_identity(pattern_matrix:numpy.ndarray) -> numpy.ndarray:
    """comput the dissimilarity matrix based on sample identity, if the pair have the same value, distance will be zero, otherwise will be one.
    This is similar to the 'scipy.spatial.distance.hamming' except that nan values will return nan instead of included in distance computation.

    Parameters
    ----------
    pattern_matrix: numpy.ndarray
        pattern_matrix containing samples' identity, of size ``(nsample,)``

    Returns
    -------
    numpy.ndarray
        dissimliarity matrix of size ``(nsample,nsample)``

    Raises
    ------
    Exception
        pattern matrix must be 1D with minimum of 2 samples
    """
    # check X 
    identity_arr = check_array(numpy.squeeze(pattern_matrix),ensure_dim=1,min_sample=2)
    X,Y = numpy.meshgrid(identity_arr,identity_arr)
    return 1. - nan_equal(X,Y)# if same, distance=0

def compute_rdm_nomial(pattern_matrix:numpy.ndarray) -> numpy.ndarray:
    """compute the dissimilarity matrix based on samples' nomial features. Must have at least 2 features, otherwise should use ``compute_rdm_identity`` instead.
    This is similar to the 'scipy.spatial.distance.hamming' except that nan values will return nan instead of included in distance computation.

    Parameters
    ----------
    pattern_matrix : numpy.ndarray
        pattern_matrix containing samples' features, of size ``(nsample,nfeature)``
    
    Returns
    -------
    numpy.ndarray
        dissimliarity matrix of size ``(nsample,nsample)``

    Raises
    ------
    Exception
        pattern matrix must be 2D with minimum of 2 samples and 2 features
    """
    # check X 
    X = check_array(pattern_matrix,ensure_dim=2,min_sample=2,min_feature=2)
    # compute rdm
    feature_rdms = [compute_rdm_identity(X[:,k]) for k in range(X.shape[1])]
    rdm = numpy.mean(feature_rdms,axis=0)
    return rdm

def compute_rdm_residual(rdm:numpy.ndarray,control_rdms:Union[list,numpy.ndarray],squareform=True):
    """compute the residual of rdm after explaining for control rdms

    Parameters
    ----------
    rdm : numpy.ndarray
        the target rdm
    control_rdms : list or numpy.ndarray
        a control rdm or a list of control rdms
    squareform : bool, optional
        to put residual back to square form or not, by default True

    Returns
    -------
    numpy.ndarray
        the residual rdm, if ``squareform==True``, the shape will be the same as the original rdm, otherwise, the lower triangular part will be returned
    """
    # check rdm 
    rdm = check_array(rdm,ensure_dim=2,min_sample=2,min_feature=2)
    assert rdm.shape[0] == rdm.shape[1]
    Y = lower_tri(rdm)[0]
    
    # check control rdm
    if isinstance(control_rdms,list):
        control_rdms = [check_array(m,ensure_dim=2,min_sample=2,min_feature=2) for m in control_rdms]
    elif isinstance(control_rdms,numpy.ndarray):
        control_rdms = [check_array(control_rdms,ensure_dim=2,min_sample=2,min_feature=2)]
    else:
        raise TypeError('control rdm must be numpy ndarray or a list of numpy ndarray')            

    assert all([numpy.logical_and(m.shape[0]==rdm.shape[0],m.shape[1]==rdm.shape[1]) for m in control_rdms]), "control rdms must have the same size as rdm"

    control_ = numpy.array([lower_tri(m)[0] for m in control_rdms])
    # find out rows that are not nans in all columns
    na_filters = numpy.all(
        ~numpy.isnan(numpy.vstack([control_,Y])),0
    )
    
    # standardize predictors and data
    X = control_.T[na_filters,:]
    #remove features with no variation
    if sum([numpy.unique(arr).size>1 for arr in X.T])>0:
        X = X[:,[numpy.unique(arr).size>1 for arr in X.T]]
        zX = scale_feature(X,1,True)
        zY = scale_feature(Y[na_filters],2,True)
        control_reg = LinearRegression().fit(zX,zY)     
        R = numpy.full_like(Y,fill_value=numpy.nan)
        R[na_filters] = zY - control_reg.predict(zX)

        if squareform:
            resid_rdm = numpy.zeros_like(rdm)
            resid_rdm[lower_tri(resid_rdm)[1]] = R
            return resid_rdm + resid_rdm.T
        else:
            return R
    else:
        raise TypeError("control rdms have less than 2 unique values")