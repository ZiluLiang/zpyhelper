""" Estimator classes for MVPA analysis
    All classes takes activity pattern matrix as an input and performs different types of RSA analysis based on the activity pattern matrix.
    An estimator class has at least four methods:
    (1) fit: by calling estimator.fit(), MVPA analysis is performed, `result` attribute of the estimator will be set, it will be an 1D numpy array.
    (2) __str__: return the name of estimator class
    (3) get_details: return the details of estimator class in a dictonary, data will be serialized so that it can be written into JSON 
    (4) visualize: by calling estimator.visualize(), the result of RSA analysis will visualized, a figure handle will be returned

Zilu Liang @HIPlab Oxford
2023
"""

import abc
import numpy
import scipy
from   sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import pandas
import itertools
import time
from typing import Union

from .rdm import lower_tri, compute_rdm
from .preprocessors import split_data, scale_feature

def _force_1d(x):
    """force array to be 1D"""
    return numpy.atleast_1d(x).flatten()

class MetaEstimator():
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass

    @abc.abstractmethod
    def get_details(self):
        pass

class PatternCorrelation(MetaEstimator):
    """calculate the correlation between neural rdm and model rdm

    Parameters
    ----------
    activitypattern : numpy.ndarray
        a 2D numpy array. the neural activity pattern matrix used for computing representation dissimilarity matrix. size = (nsample,nfeatures)
    modelrdm : numpy.ndarray or list of numpy.ndarray
        a 2D numpy array. the dissimilarity matrices from different models of representation. size = (nsample,nsample)
    modelnames: list
        a list of model names. If `None` models will be named as `m1,m2,...`, by default `None`
    rdm_metric: str, optional
        dissimilarity/distance metric passed to `scipy.spatial.distance.pdist` to compute neural rdm from activity pattern matrix, by default `"correlation"`
    type : str, optional
        type of correlation measure, by default `"spearman"`.
        must be one of: `"spearman", "pearson", "kendall", "linreg"`
    ztransform: bool, optional
        whether or not to perform fisher Z transform to the correlation coefficients, by default `False`.    
    """
    def __init__(self,
                 activitypattern:numpy.ndarray,
                 modelrdms:Union[numpy.ndarray,list],
                 modelnames:list=None,
                 rdm_metric:str="correlation",
                 type:str="spearman",
                 ztransform:bool=False) -> None:        

        #neural rdm
        neuralrdm = compute_rdm(activitypattern,rdm_metric)
        self.rdm_shape = neuralrdm.shape        
        self.Y,_ = lower_tri(neuralrdm)

        #model rdm and names        
        if isinstance(modelrdms,list):
            modelrdms = modelrdms
        elif isinstance(modelrdms,numpy.ndarray):
            modelrdms = [modelrdms]
        else:
            raise TypeError('model rdm must be numpy ndarray or a list of numpy ndarray')
        
        if modelnames is None:
            modelnames = [f'm{str(j)}' for j in range(len(modelrdms))]            
        assert len(modelnames) == len(modelrdms), 'number of model names must be equal to number of model rdms'   

        self.Xs = []
        for m in modelrdms:
            X,_ = lower_tri(m)
            self.Xs.append(X)
        self.modelnames = modelnames  
        self.nX = len(modelrdms)

        #correlation type
        corr_functions = {
            "spearman": lambda X,Y: scipy.stats.spearmanr(X,Y).correlation,
            "pearson":  lambda X,Y: scipy.stats.pearsonr(X,Y).correlation,
            "kendall":  lambda X,Y: scipy.stats.kendalltau(X,Y).statistic,
            "linreg":   lambda X,Y: scipy.stats.linregress(X,Y).slope
        }    
        if type not in corr_functions.keys():
            raise ValueError('unsupported type of correlation, must be one of: ' + ', '.join(list(corr_functions.keys())))
        else:
            self.type = type
            self.corrfun = corr_functions[self.type]
        if ztransform:
            self.outputtransform = lambda x: numpy.arctanh(x)
        else:
            self.outputtransform = lambda x: x
    
    def fit(self):
        self.na_filters = []
        result = []
        for X in self.Xs:
            na_filters = numpy.logical_and(~numpy.isnan(self.Y),~numpy.isnan(X))
            self.na_filters.append(na_filters)
            Y = self.Y[na_filters]
            X = X[na_filters]
            r = self.outputtransform(self.corrfun(X,Y))
            result.append(r)
        self.result = _force_1d(result)
        return self
    
    def visualize(self):
        try:
            self.result
        except Exception:
            self.fit()        

        fig,axes = plt.subplots(self.nX,2,figsize = (10,5*self.nX))
        for k,X in enumerate(self.Xs):
            plot_models = [self.Y,X]
            plot_titles = [f"neural rdm \n r(neural,model)={'%.3f' % self.result}", f"{self.modelnames[k]}"]
            for j,(t,m) in enumerate(zip(plot_titles,plot_models)):
                v = numpy.full(self.rdm_shape,numpy.nan)
                _,idx = lower_tri(v)
                fillidx = (idx[0][self.na_filters[k]],idx[1][self.na_filters[k]])
                v[fillidx] = m[self.na_filters[k]]
                if self.nX==1:
                    sns.heatmap(v,ax=axes[j],square=True,cbar_kws={"shrink":0.85})
                    axes[j].set_title(t)
                else:
                    sns.heatmap(v,ax=axes[k][j],square=True,cbar_kws={"shrink":0.85})
                    axes[k][j].set_title(t)
        fig.suptitle(f'{self.type} correlation:')
        return fig
    
    def __str__(self) -> str:
        return f"PatternCorrelation with {self.type}"
    
    def get_details(self):        
        details = {"name":self.__str__(),
                   "corrtype":self.type,
                   "NAfilters":dict(zip(self.modelnames,[x.tolist() for x in self.na_filters])),
                   "modelRDMs":dict(zip(self.modelnames,[x.tolist() for x in self.Xs]))
                  }
        return  details

class MultipleRDMRegression(MetaEstimator):
    """estimate the regression coefficient when using the model rdms to predict neural rdm

    Parameters
    ----------
    activitypattern : numpy.ndarray
        a 2D numpy array. the neural activity pattern matrix used for computing representation dissimilarity matrix. size = (nsample,nfeatures)
    modelrdms : numpy.ndarray or list of numpy.ndarray
        a 2D numpy array. the dissimilarity matrices from different models of representation. size = (nsample,nsample)
    modelnames: list, optional
        a list of model names. If `None` models will be named as m1,m2,..., by default `None`
    rdm_metric: str, optional
        dissimilarity/distance metric passed to `scipy.spatial.distance.pdist` to compute neural rdm from activity pattern matrix, by default `"correlation"`
    standardize: bool, optional
        whether or not to standardize the model rdms and neural rdms before running regression, by default `True`
    """
    def __init__(self,activitypattern:numpy.ndarray,modelrdms:Union[numpy.ndarray,list],modelnames:list=None,rdm_metric:str="correlation",standardize:bool=True) -> None:

        #model rdm and names        
        if isinstance(modelrdms,list):
            modelrdms = modelrdms
        elif isinstance(modelrdms,numpy.ndarray):
            modelrdms = [modelrdms]
        else:
            raise TypeError('model rdm must be numpy ndarray or a list of numpy ndarray')

        if modelnames is None:
            modelnames = [f'm{str(j)}' for j in range(len(modelrdms))]
        assert len(modelnames) == len(modelrdms), 'number of model names must be equal to number of model rdms'

        #neural rdm
        neuralrdm = compute_rdm(activitypattern,rdm_metric)
        self.rdm_shape = neuralrdm.shape
        self.modelnames = modelnames

        Y,_ = lower_tri(neuralrdm)

        self.n_reg = len(modelrdms) # number of model rdms
        X = numpy.empty((len(Y),self.n_reg)) # X is a nvoxel * nmodel matrix
        for j,m in enumerate(modelrdms):
            X[:,j],_ = lower_tri(m)

        self.X = X
        self.Y = Y

        self.standardize = standardize
    
    def fit(self):
        xna_filters = numpy.all(~numpy.isnan(self.X),1) # find out rows that are not nans in all columns
        self.na_filters = numpy.logical_and(~numpy.isnan(self.Y),xna_filters)

        if self.standardize:
            #standardize design matrix independently within each column
            X = scale_feature(self.X[self.na_filters,:],1)
            # standardize Y
            Y = scale_feature(self.Y[self.na_filters])
        else:
            X = self.X[self.na_filters,:]
            Y = self.Y[self.na_filters]

        reg = LinearRegression().fit(X,Y)
        self.reg = reg
        self.result = _force_1d(reg.coef_)
        self.score  = reg.score(X,Y)
        return self
    
    def visualize(self):
        try:
            self.result
        except Exception:
            self.fit()
        plot_models = list(numpy.concatenate((numpy.atleast_2d(self.Y),self.X.T),axis=0))
        plot_titles = ["neural rdm"] + self.modelnames

        fig,axes = plt.subplots(1,len(plot_models),figsize = (10,5))
        for j,(t,m) in enumerate(zip(plot_titles,plot_models)):
            v = numpy.full(self.rdm_shape,numpy.nan)
            _,idx = lower_tri(v)
            fillidx = (idx[0][self.na_filters],idx[1][self.na_filters])
            v[fillidx] = m[self.na_filters]
            sns.heatmap(v,ax=axes.flatten()[j],square=True,cbar_kws={"shrink":0.85})
            if j == 0:
                axes.flatten()[j].set_title(t)
            else:
                axes.flatten()[j].set_title(f"{t}:{self.result[j-1]}")
        fig.suptitle(f'R2: {self.score}')
        return fig

    def __str__(self) -> str:
        return "MultipleRDMRegression"
    
    def get_details(self):        
        details = {"name":self.__str__(),
                   "standardize":self.standardize*1,
                   "NAfilters":self.na_filters.tolist(),
                   "modelRDMs":dict(zip(self.modelnames,[x.tolist() for x in self.X.T])),
                   "score":self.score
                  }
        return  details

class NeuralRDMStability(MetaEstimator):
    def __init__(self,activitypattern:numpy.ndarray,groups:numpy.ndarray,
                 rdm_metric:str="correlation", type:str="spearman",ztransform:bool=False) -> None:
        """calculate the stability of neural RDM across different splits of data

        Parameters
        ----------
        activitypattern : numpy.ndarray
            a 2D numpy array. the neural activity pattern matrix used for computing representation dissimilarity matrix. size = (nsample,nfeatures)
        groups : numpy.ndarray
            array specifying the which group each row of activitypattern belongs to. Must have at least two unique values.
        rdm_metric : str, optional
            _dissimilarity/distance metric passed to `scipy.spatial.distance.pdist` to compute neural rdm from activity pattern matrix, by default `"correlation"`
        type : str, optional
            type of correlation measure, by default `"spearman"`.
            must be one of: `"spearman", "pearson", "kendall", "linreg"`
        ztransform: bool, optional
            whether or not to perform fisher Z transform to the correlation coefficients, by default `False`.

        Raises
        ------
        ValueError
            ``groups`` must have at least two unique values
        """
        assert numpy.unique(groups).size>1
        self.Xs = split_data(activitypattern,groups)
        self.RDMs = [compute_rdm(X,rdm_metric) for X in self.Xs]
        self.groups = numpy.unique(groups)
        
        #correlation type
        corr_functions = {
            "spearman": lambda X,Y: scipy.stats.spearmanr(X,Y).correlation,
            "pearson":  lambda X,Y: scipy.stats.pearsonr(X,Y).correlation,
            "kendall":  lambda X,Y: scipy.stats.kendalltau(X,Y).statistic,
            "linreg":   lambda X,Y: scipy.stats.linregress(X,Y).slope
        }    
        if type not in corr_functions.keys():
            raise ValueError('unsupported type of correlation, must be one of: ' + ', '.join(list(corr_functions.keys())))
        else:
            self.type = type
            self.corrfun = corr_functions[self.type]
        if ztransform:
            self.outputtransform = lambda x: numpy.arctanh(x)
        else:
            self.outputtransform = lambda x: x

    def fit(self):
        result = []
        for rdm1,rdm2 in itertools.combinations(self.RDMs,2):
            X,Y = lower_tri(rdm1)[0],lower_tri(rdm2)[0]
            r = self.outputtransform(self.corrfun(X,Y))
            result.append(r)
        self.stability_perpair = dict(zip([f"{x}_{y}" for x,y in itertools.combinations(numpy.unique(self.groups),2)],result))
        self.result = _force_1d(numpy.mean(result))
        return self
    
    def __str__(self) -> str:
        return "NeuralRDMStability"
    
    def visualize(self):
        fig,axes = plt.subplots(1,len(self.RDMs),figsize = (10,5))
        for j,(g,m,ax) in enumerate(zip(self.groups,self.RDMs,axes)):
            v = numpy.full_like(m,fill_value=numpy.nan)
            v[lower_tri(m)[1]] = lower_tri(m)[0]
            sns.heatmap(v,ax=ax,square=True,cbar_kws={"shrink":0.85})
            ax.set_title(f"rdm of data split {g}")
        fig.suptitle(f'average stability: {self.result}')
        return fig
    
    def get_details(self):        
        details = {"name":self.__str__(),
                   "corrtype":self.type
                  }
        return  details
