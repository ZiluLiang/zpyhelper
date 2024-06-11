""" Estimator classes for MVPA analysis
    All classes takes activity pattern matrix as an input and performs different types of RSA analysis based on the activity pattern matrix.  \n
    An estimator class has at least three methods:  \n
    (1) fit: by calling estimator.fit(), analysis is performed, `result` attribute of the estimator will be set, it will be an 1D numpy array.  \n
    (2) __str__: return the name of estimator class  \n
    (3) get_details: return the details of estimator class in a dictonary, data will be serialized so that it can be written into JSON    \n
    Some estimator also comes with a plotting method:  \n
    (4) visualize: by calling estimator.visualize(), the result of RSA analysis will visualized, a figure handle will be returned  \n
  
"""

import abc
import numpy
import scipy
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multioutput import MultiOutputRegressor,MultiOutputClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score,r2_score
from sklearn import svm

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import itertools
import time
from typing import Union

from .rdm import lower_tri, compute_rdm, compute_rdm_residual
from .preprocessors import split_data, scale_feature

def _force_1d(x):
    """force array to be 1D"""
    return numpy.atleast_1d(x).flatten()

class MetaEstimator():
    """Meta class for estimators
    """
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def fit(self):
        """Run estimator.   
         After self.fit() is called, `result` attribute of the estimator will be set, it will be an 1D numpy array.
        """
        pass

    @abc.abstractmethod
    def __str__(self):
        """Return the name of estimator class  
        """
        pass

    @abc.abstractmethod
    def get_details(self):
        """Return the details of estimator class in a dictonary, data will be serialized so that it can be written into JSON
        """
        pass

class PatternCorrelation(MetaEstimator):
    """Calculate the correlation between neural rdm and model rdm

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
                 ztransform:bool=False,
                 runonresidual:bool=False,
                 controlrdms:Union[numpy.ndarray,list]=[]) -> None:        

        #neural rdm
        neuralrdm = compute_rdm(activitypattern,rdm_metric)
        self.rdm_shape = neuralrdm.shape        
        # if run on residual, compute the residual after regressing out control rdm
        if runonresidual:
            self.Y = compute_rdm_residual(neuralrdm,controlrdms,squareform=False)
        else:
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
        """Run estimator.   
         After self.fit() is called, `result` attribute of the estimator will be set, it will be an 1D numpy array.

         Note that pairwise exclusion is used. For each model, the lower triangular part of the model rdm and that of neural rdm are put into two columns. 
         If one of the two columns contains a NaN , the corresponding row is omitted. This process is done separately for each model rdm when calculating correlation between that particular model rdm and neural rdm
        """
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
    
    def visualize(self)->matplotlib.figure.Figure:
        """show the model rdm(s) and neural rdm in heatmap.

        Each subplot shows a model or neural rdm. The correlation between neural and model rdm is shown in subplot's title.

        Returns
        -------
        matplotlib.figure.Figure
            the handle of plotted figure
        """
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
        """Return the name of estimator class  
        """
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
    def __init__(self,activitypattern:numpy.ndarray,
                 modelrdms:Union[numpy.ndarray,list],modelnames:list=None,
                 rdm_metric:str="correlation",standardize:bool=True,
                 runonresidual:bool=False,
                 controlrdms:Union[numpy.ndarray,list]=[]) -> None:

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

        # if run on residual, compute the residual after regressing out control rdm
        if runonresidual:
            Y = compute_rdm_residual(neuralrdm,controlrdms,squareform=False)
        else:
            Y,_ = lower_tri(neuralrdm)

        self.n_reg = len(modelrdms) # number of model rdms
        X = numpy.empty((len(Y),self.n_reg)) # X is a npair * nmodel matrix
        for j,m in enumerate(modelrdms):
            X[:,j],_ = lower_tri(m)

        self.X = X
        self.Y = Y

        self.standardize = standardize
    
    def fit(self):
        """Run estimator.   
         After self.fit() is called, `result` attribute of the estimator will be set, it will be an 1D numpy array.

         Note that list-wise exclusion is used. The lower triangular part of the model rdm(s) and that of neural rdm are put into columns. 
         If any one of the columns contains a NaN , the corresponding row is omitted.
        """
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
    
    def visualize(self)->matplotlib.figure.Figure:
        """show the model rdm(s) and neural rdm in heatmap.

        Each subplot shows a model or neural rdm. The regression coefficient of model rdm is shown in subplot's title.

        Returns
        -------
        matplotlib.figure.Figure
            the handle of plotted figure
        """
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
        """Return the name of estimator class  
        """
        return "MultipleRDMRegression"
    
    def get_details(self):  
        """Return the details of estimator class in a dictonary, data will be serialized so that it can be written into JSON
        """      
        details = {"name":self.__str__(),
                   "standardize":self.standardize*1,
                   "NAfilters":self.na_filters.tolist(),
                   "modelRDMs":dict(zip(self.modelnames,[x.tolist() for x in self.X.T])),
                   "score":self.score
                  }
        return  details

class NeuralRDMStability(MetaEstimator):
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
    def __init__(self,activitypattern:numpy.ndarray,groups:numpy.ndarray,
                 rdm_metric:str="correlation", type:str="spearman",ztransform:bool=False) -> None:
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
        """Run estimator.   
         After self.fit() is called, `result` attribute of the estimator will be set, it will be an 1D numpy array.

         Neural RDM is calculated for each group. Correlation between every possible pair of groups are calculated. Then the average is taken over all the calculated correlation coefficients.
        """
        result = []
        for rdm1,rdm2 in itertools.combinations(self.RDMs,2):
            X,Y = lower_tri(rdm1)[0],lower_tri(rdm2)[0]
            r = self.outputtransform(self.corrfun(X,Y))
            result.append(r)
        self.stability_perpair = dict(zip([f"{x}_{y}" for x,y in itertools.combinations(numpy.unique(self.groups),2)],result))
        self.result = _force_1d(numpy.mean(result))
        return self
    
    def __str__(self) -> str:
        """Return the name of estimator class  
        """
        return "NeuralRDMStability"
    
    def visualize(self)->matplotlib.figure.Figure:
        """show the neural rdm of different groups using heatmap.

        Each subplot shows the neural rdm of one group.

        Returns
        -------
        matplotlib.figure.Figure
            the handle of plotted figure
        """
        fig,axes = plt.subplots(1,len(self.RDMs),figsize = (10,5))
        for j,(g,m,ax) in enumerate(zip(self.groups,self.RDMs,axes)):
            v = numpy.full_like(m,fill_value=numpy.nan)
            v[lower_tri(m)[1]] = lower_tri(m)[0]
            sns.heatmap(v,ax=ax,square=True,cbar_kws={"shrink":0.85})
            ax.set_title(f"rdm of data split {g}")
        fig.suptitle(f'average stability: {self.result}')
        return fig
    
    def get_details(self):
        """Return the details of estimator class in a dictonary, data will be serialized so that it can be written into JSON
        """
        details = {"name":self.__str__(),
                   "corrtype":self.type
                  }
        return  details

class PatternDecoding(MetaEstimator):
    """decode from neural activity pattern with cross validation   
    Important note: this hasn't been tested in whole-brain searchlight yet

    Parameters
    ----------
    activitypattern : numpy.ndarray
        a 2D numpy array of size `(nsample,nfeature)`. the neural activity pattern matrix used for decoding.
    targets : numpy.ndarray 
        a 2D or 1D numpy array of size `(nsample,ntarget)` or `(nsample,)`.  Target values for prediction (class labels in classification, real numbers in regression).
    groups : numpy.ndarray
        a 1D numpy array of size `(nsample,)`. Group values used for splitting data into decoder fitting and evaluation set. Must have at least two unique values
    decoder: str or an estimator object implementing decoding analysis with `fit` and `predict`
        decoder used to perform the analysis
    decoder_kwarg: dict 
        arguments passed to decoder class to instantiate a decoder
    regression: bool
        regression or classification decoder. If ``True``, will find in regression decoders, if ``False``, will find in classification decoders. by default ``False``.
    targetnames: list, optional
        a list of model names. If `None` models will be named as m1,m2,..., by default `None`
    scoring_func: str or callable, optional
        a scoring function that will be used to compute the quality of prediction after fitting the decoder. If None, will use accuracy score for classification and use r_square for regression.
    scoring_funcname: str, optional
        name of the scoring function if an customised callable is passed in `scoring_funcname`
    """
    def __init__(self,activitypattern:numpy.ndarray,
                 targets:numpy.ndarray, groups:numpy.ndarray,
                 decoder:str,decoder_kwarg:dict={},
                 targetnames:list=None,regression=False,
                 scoring_func:Union[str,callable]=None,scoring_funcname:str=None) -> None:
        REGRESSION_CATALOG     = dict(svr= svm.SVR, linear = LinearRegression)
        CLASSIFICATION_CATALOG = dict(svc= svm.LinearSVC, logistic=LogisticRegression)
        SCORING_CATALOG        = dict(r2 = r2_score, acc = accuracy_score)

        #check targets
        if targets.ndim == 1:
            targets = numpy.atleast2d(targets).T
        assert activitypattern.shape[0] == targets.shape[0], f"Sample size mismatch: number of samples in activity pattern matrix is {activitypattern.shape[0]}, number of samples in ``targets`` is {targets.shape[0]}"
        self.multioutput = targets.shape[1] > 1

        if targetnames is None:
            targetnames = [f't{str(j)}' for j in range(len(targetnames))]
        assert len(targetnames) == targets.shape[1], 'number of target names must be equal to number of targetnames'
        self.targetnames = targetnames

        #check groups
        groups = numpy.array(groups).flatten()
        assert groups.size == activitypattern.shape[0],  f"Sample size mismatch: number of samples in activity pattern matrix is {activitypattern.shape[0]}, number of samples in ``groups`` is {groups.size}" 
        
        # decoder
        self.basedecoder = decoder
        if isinstance(decoder,str):
            if regression:
                self.basedecoder = REGRESSION_CATALOG[decoder]
                if self.multioutput:
                    self.multi_meta  = MultiOutputRegressor
                scoring_func = "r2" if scoring_func is None else scoring_func
            else:
                self.basedecoder = CLASSIFICATION_CATALOG[decoder]
                scoring_func = "acc" if scoring_func is None else scoring_func
                if self.multioutput:
                    self.multi_meta  = MultiOutputClassifier
        self.decoder_kwarg = decoder_kwarg
        
        self.score = scoring_func
        self.scorename = scoring_funcname if isinstance(scoring_func,str) else "customised"
        if isinstance(scoring_func,str):
            self.scorename = scoring_func
            self.score = SCORING_CATALOG[scoring_func]

        
        self.X = activitypattern
        self.Y = targets
        self.groups = groups
    
    def fit(self):
        """Run estimator.   
         After self.fit() is called, `result` attribute of the estimator will be set, it will be an 1D numpy array.

         Run decoding in cross-validation. data is splitted according to `self.groups`. Leave-one-group-out cross-validation is used. The result saved is [average fit score, average cv evaluation score]
        """
        logo = LeaveOneGroupOut()
        fit_scores, eval_scores = [], []
        for k, (fit_index, eval_index) in enumerate(logo.split(self.X, self.Y, self.groups)):
            #print(f"Fold {k}:")
            #print(f"  Fit: index={fit_index}, group={groups[fit_index]}")
            #print(f"  Evaluation:  index={eval_index}, group={groups[eval_index]}")
            X_fit,X_eval = self.X[fit_index,:], self.X[eval_index,:]
            if self.multioutput:
                decoder = self.multi_meta(self.basedecoder(**self.decoder_kwarg))
                Y_fit,Y_eval = self.Y[fit_index,:], self.Y[eval_index,:]
            else:
                decoder = self.basedecoder(**self.decoder_kwarg)
                Y_fit,Y_eval = self.Y[fit_index,0], self.Y[eval_index,0]

            decoder.fit(X_fit,Y_fit)
            PRED_fit,  PRED_eval  = decoder.predict(X_fit),      decoder.predict(X_eval)
            score_fit, score_eval = self.score(PRED_fit,Y_fit),  self.score(PRED_eval,Y_eval)

            fit_scores.append(score_fit)
            eval_scores.append(score_eval)

        self.result =_force_1d([numpy.mean(fit_scores),numpy.mean(eval_scores)])
        self.resultnames = ["fit","eval"]
        return self
    
    def visualize(self)->matplotlib.figure.Figure:
        """show the score of the decoder in fit and evaliatopm set using scatter plots

        Returns
        -------
        matplotlib.figure.Figure
            the handle of plotted figure
        """
        try:
            self.result
        except Exception:
            self.fit()
        fig,ax = plt.subplots(1,1,figsize = (5,5))
        ax.scatter(x=numpy.arange(self.result.size),y=self.result)
        fig.suptitle(f'Score: {self.result}')
        return fig

    def __str__(self) -> str:
        """Return the name of estimator class  
        """
        return "PatternDecoding"
    
    def get_details(self):
        """Return the details of estimator class in a dictonary, data will be serialized so that it can be written into JSON
        """        
        details = {"name":self.__str__(),
                   "scorenames":self.resultnames,
                   "scores": self.result
                  }
        return  details