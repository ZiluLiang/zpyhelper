"""
This module contains classes for running searchlight MVPA analysis in nifti images.
The code is adapted from nilearn.decoding.searchlight module @ https://github.com/nilearn/nilearn/blob/321494420/nilearn/decoding/searchlight.py
"""
import os
import sys
import time
import json
import warnings
import numpy
import contextlib

import nibabel as nib
import nibabel.processing
from joblib import Parallel, delayed, cpu_count
from scipy.sparse import vstack, find

from ..filesys import checkdir, is_jsonable
from ..MVPA.preprocessors import chain_steps

from .niidatahandler import _check_and_load_images,write_data_to_image

import nilearn
from nilearn import image, masking
from nilearn._utils.niimg_conversions import (
    _safe_get_data,
    check_niimg_3d,
)
from nilearn.decoding.searchlight import GroupIterator
from sklearn import neighbors


def _apply_mask_and_get_affinity(seeds,
                                 niimgs,
                                 radius:float=5,
                                 mask_img=None,
                                 empty_policy:str="ignore",
                                 n_vox:int=2,
                                 allow_overlap:bool=True,
                                 n_jobs:int=cpu_count()):
    """
    This function is adapted from `_apply_mask_and_get_affinity` from the `nilearn.maskers.nifti_spheres_masker` module
    (https://github.com/nilearn/nilearn/blob/0d379462d8f84344056308d4d096caf78954ca6d/nilearn/input_data/nifti_spheres_masker.py)
    
    Original Documentation: 
    ----------
    Get only the rows which are occupied by sphere at given seed locations and the provided radius.
    Rows are in target_affine and target_shape space.
    
    Adaptation 
    ----------
    This function is adapted to:
    1. impose minimum voxel number contraints and avoid throwing warnings when an empty sphere/insufficient voxel number is detected.
    The current script handle empty sphere or sphere with insufficient voxel number using one of the three approaches by specifying `empty_policy` argument:  
        - ``'fill'``: conduct new search with the same algorithm but incrementing searchlight sphere radius for empty spheres  
        - ``'ignore'``: return empty  
        - ``'raise'``: throw an error       

    2. add parallelization to speed up the process

    3. extract from multiple nii img. This will be handy when estimating whitening matrix for each sphere from residual images of the first level GLM.
    
    Parameters
    ----------
    seeds : List of triplets of coordinates in native space
        Seed definitions. List of coordinates of the seeds in the same space
        as target_affine.

    niimgs : a list of or one 3D/4D Niimg-like object
        See :ref:`extracting_data`.
        Images to process.
        If a 3D niimg is provided, a singleton dimension will be added to
        the output to represent the single scan in the niimg.

    radius : float
        Indicates, in millimeters, the radius for the sphere around the seed.  By default `5`.

    mask_img : Niimg-like object, optional
        Mask to apply to regions before extracting signals. If niimg is None,
        mask_img is used as a reference space in which the spheres 'indices are
        placed. By default `None`.

    empty_policy: str, optional
        how to deal with empty spheres

    n_vox : int, optional
        minimum number of voxels in each searchlight sphere. By default `2`.

    allow_overlap : boolean
        If False, a ValueError is raised if VOIs overlap. By default `True`.

    Returns
    -------
    X : 2D numpy.ndarray
        Signal for each brain voxel in the (masked) niimgs.
        shape: (number of scans, number of voxels)

    A : scipy.sparse.lil_matrix
        Contains the boolean indices for each sphere.
        shape: (number of seeds, number of voxels)

    """

    t0 = time.time()

    seeds = list(seeds)

    # Compute world coordinates of all in-mask voxels.
    if niimgs is None:
        mask, affine = masking._load_mask_img(mask_img)
        # Get coordinate for all voxels inside of mask
        mask_coords = numpy.asarray(numpy.nonzero(mask)).T.tolist()
        X = None

    elif mask_img is not None:
        niimgs = [niimgs] if not isinstance(niimgs,list) else niimgs
        Xs = []
        for niimg in niimgs:
            affine = niimg.affine
            mask_img = check_niimg_3d(mask_img)
            mask_img = image.resample_img(
                mask_img,
                target_affine=affine,
                target_shape=niimg.shape[:3],
                interpolation='nearest',
            )
            mask, _ = masking._load_mask_img(mask_img)
            mask_coords = list(zip(*numpy.where(mask != 0)))

            X = masking._apply_mask_fmri(niimg, mask_img)
            Xs.append(X)

    elif niimgs is not None:
        niimgs = [niimgs] if not isinstance(niimgs,list) else niimgs
        Xs = []
        for niimg in niimgs:
            affine = niimg.affine
            if numpy.isnan(numpy.sum(_safe_get_data(niimg))):
                warnings.warn(
                    'The imgs you have fed into fit_transform() contains NaN '
                    'values which will be converted to zeroes.'
                )
                X = _safe_get_data(niimg, True).reshape([-1, niimg.shape[3]]).T
            else:
                X = _safe_get_data(niimg).reshape([-1, niimg.shape[3]]).T
            Xs.append(X)

            mask_coords = list(numpy.ndindex(niimg.shape[:3]))
    else:
        raise ValueError("Either a list of or one niimg or a mask_img must be provided.")
    t1 = time.time()

    # For each seed, get coordinates of nearest voxel
    def find_ind(m_coords, nearest):
        try:
            return m_coords.index(nearest)
        except ValueError:
            return None

    def search_nearest_for_seedschunk(m_coords,seedschunk,aff):
        tmp_nearests = numpy.round(image.resampling.coord_transform(
            numpy.array(seedschunk)[:,0], numpy.array(seedschunk)[:,1], numpy.array(seedschunk)[:,2], numpy.linalg.inv(aff)
        )).T.astype(int)
        nearest_ = [find_ind(m_coords,tuple(nearest)) for nearest in tmp_nearests]
        return nearest_
    
    with Parallel(n_jobs=n_jobs) as parallel:
        n_splits = n_jobs
        split_idx = numpy.array_split(numpy.arange(len(seeds)), n_splits)
        seed_chunks = [numpy.array(seeds)[idx] for idx in split_idx]
        nearests = parallel(delayed(search_nearest_for_seedschunk)(mask_coords,sc,affine) for sc in seed_chunks)
    nearests = sum(nearests,[])
    
    mask_coords = numpy.asarray(list(zip(*mask_coords)))
    mask_coords = image.resampling.coord_transform(
        mask_coords[0], mask_coords[1], mask_coords[2], affine
    )
    mask_coords = numpy.asarray(mask_coords).T

    clf = neighbors.NearestNeighbors(radius=radius)
    A = clf.fit(mask_coords).radius_neighbors_graph(seeds)
    A = A.tolil()        
    for i, nearest in enumerate(nearests):
        if nearest is None:
            continue
        A[i, nearest] = True

    # Include the voxel containing the seed itself if not masked
    mask_coords = mask_coords.astype(int).tolist()
    for i, seed in enumerate(seeds):
        with contextlib.suppress(ValueError):
            A[i, mask_coords.index(list(map(int, seed)))] = True

    # Check for empty/insufficient voxel number spheres    
    sphere_sizes = numpy.asarray(A.tocsr().sum(axis=1)).ravel()
    redo_spheres = numpy.nonzero(sphere_sizes < n_vox)[0]
    
    j = 0
    if empty_policy == "raise":
        raise ValueError(f'The following spheres have less than {n_vox} voxels: {redo_spheres}')
    elif empty_policy == "ignore":
        pass
    elif empty_policy == "fill":
        #expand radius if doesn't meet voxel count criteria
        while len(redo_spheres)>0:
            j+=1
            redo_seeds = list(numpy.array(seeds)[redo_spheres])
            redo_nearests = list(numpy.array(nearests)[redo_spheres])
            radius += 0.5
            redo_A = neighbors.NearestNeighbors(radius=radius).fit(mask_coords).radius_neighbors_graph(redo_seeds)
            redo_A = redo_A.tolil()
            for i, nearest in enumerate(redo_nearests):
                if nearest is None:
                    continue
                redo_A[i, nearest] = True
            for i, seed in enumerate(redo_seeds):
                with contextlib.suppress(ValueError):
                    redo_A[i, mask_coords.index(list(map(int, seed)))] = True
            A[redo_spheres, :] = redo_A
            sphere_sizes = numpy.asarray(A.tocsr().sum(axis=1)).ravel()
            redo_spheres = numpy.nonzero(sphere_sizes < n_vox)[0]
        
    if (not allow_overlap) and numpy.any(A.sum(axis=0) >= 2):
        raise ValueError('Overlap detected between spheres')
    
    print(f'reading nii took: {t1-t0} seconds, neibourhood specification took: {time.time()-t1} seconds, redo iterations = {j},  max radius = {radius}')
    return Xs, A

class RSASearchLight:
    def __init__(self,
                 patternimg_paths,
                 mask_img_path:str,
                 residimg_paths:list=[],
                 process_mask_img_path:str=None,
                 radius:float=12.5,
                 preproc_steps:dict={},
                 njobs:int=1):
        """_summary_

        Parameters
        ----------
        patternimg_paths : str or list
            path to the activity pattern images. It can be path to a 4D image or paths to multiple 3D images.
        mask_img_path : str
            path to the mask image. The mask image is a boolean image specifying voxels whose signals should be included into computation (of neural rdm etc)
        residimg_paths : str or list
            path to the residual images. It can be path to a 4D image or paths to multiple 3D images. It will be used if multivariate noise normalization is required
        process_mask_img_path : str, optional
            path to the process mask image. The process mask image is a boolean image specifying voxels on which searchlight analysis is performed. If None, will use the mask_img_path by default None
        radius : float, optional
            the radius of the searchlight sphere, by default 12.5
        preproc_steps:dict, optional
            preprocesss steps applied to the activity pattern matrix before calling estimator\n
            for example: \n
            ```
            preproc_steps = {
                "MVNN": [normalise_multivariate_noise,
                         {"ap_groups":apgroup,"resid_groups":rsgroup}
                        ],
                "PCA": [extract_pc,{"n_components":3}],
                "AOE": [average_odd_even_session,{"session":session}],
                }
            ```

        njobs : int, optional
            number of parallel jobs, by default 1
        """
        self.pattern_img      = _check_and_load_images(patternimg_paths,mode="concatenate")
        if len(residimg_paths)>1:
            self.resid_img    = _check_and_load_images(residimg_paths,mode="concatenate")
        else:
            self.resid_img    = None
        self.mask_img         = _check_and_load_images(mask_img_path,mode="intersect")
        self.process_mask_img = self.mask_img if process_mask_img_path is None else _check_and_load_images(process_mask_img_path,mode="intersect")
        self.radius           = radius
        self.njobs            = njobs

        # get preproc steps
        self.preproc_steps    = preproc_steps
        preproc_summary = {} if len(preproc_steps)>0 else "None"
        for j, (step_name,stepcfg) in enumerate(preproc_steps.items()):
            func_param = {}
            for k,v in stepcfg[1].items():
                if is_jsonable(v):
                    func_param[k] = v
                else:
                    if isinstance(v,numpy.ndarray):
                        func_param[k] = v.tolist()
                    
            preproc_summary[f"step{j}-{step_name}"] = {
                "function":stepcfg[0].__name__,
                "function_param": func_param
            }

        # get search light spheres
        if self.resid_img is not None:
            [self.X, self.R], self.A, self.neighbour_idx_lists = self.genPatches()
        else:
            [self.X], self.A, self.neighbour_idx_lists = self.genPatches()
            self.R = None
        print(f"total number of voxels to perform searchlight: {len(self.neighbour_idx_lists)}")

        ## create a searchlight summary
        self.config ={"mask":mask_img_path,
                      "radius":radius,
                      "scans":patternimg_paths,
                      "njobs":njobs,
                      "preproc":preproc_summary}

    def run(self,estimator,estimator_kwargs:dict,outputpath:str,outputregexp:str="beta_%04d.nii",verbose:bool=True):
        """run searchlight analysis and save results.  

        Whole brain searchlight spheres are split into `self.jobs` number of chunks and the searchlight for these chunks are performed in parallel.

        In each searchlight sphere, its neural activity pattern as well as the `estimator_kwargs` are passed to `estimator` to instantiate an estimator class for running the analysis

        After all jobs are done, results are written into a nii file and saved to the output directory

        Parameters
        ----------
        estimator : class
            an estimator class that is called to perform rsa analysis in each sphere
        estimator_kwargs : dict
            other arguments passed to instantiate an estimator class in each sphere
        outputpath : str
            output directory, results will be written to a nii file and saved to this directory
        outputregexp : str
            regular expression that is used to contruct output nii file name.  by default "beta_%04d.nii"
        verbose : bool, optional
            display progress or not, by default True
        """
        t0 = time.time()
        self.estimator = estimator
        # split patches for parallelization
        group_iter = GroupIterator(len(self.neighbour_idx_lists), self.njobs)
        with Parallel(n_jobs=self.njobs) as parallel:
            results = parallel(
                delayed(self.fitPatchGroup)(
                    [self.neighbour_idx_lists[i] for i in list_i],
                    thread_id + 1,
                    self.A.shape[0],
                    estimator_kwargs,
                    verbose
                )
                for thread_id, list_i in enumerate(group_iter)
            )

        result = numpy.vstack([x[0] for x in results])
        estimator_details = results[0][1]
        sys.stderr.write((f" completed searchlight on {len(self.neighbour_idx_lists)} voxels in {time.time()-t0} seconds. \n"))
        self.write(result,estimator_details,outputpath,outputregexp)
        return self
    
    def write(self,result,estimator_details,outputpath,outputregexp,ensure_finite:bool=False):
        if '.nii' not in outputregexp:
            outputregexp = f'{outputregexp}.nii'
        checkdir(outputpath)
        mean_img = nilearn.image.mean_img(self.pattern_img)       
        for k in numpy.arange(numpy.shape(result)[1]):
            curr_fn = os.path.join(outputpath,outputregexp % (k))
            write_data_to_image(data=result[:,k],
                                mask_imgs=self.process_mask_img,
                                ref_img=mean_img,
                                ensure_finite=ensure_finite,outputpath=curr_fn)

        # create a json file storing regressor information
        with open(os.path.join(outputpath,'searchlight.json'), "w") as outfile:
            json.dump({"searchlight_config":self.config,"estimator":estimator_details}, outfile)        
        return self 

    
    def fitPatchGroup(self,neighbour_idx_list:list,thread_id:int,total:int, estimator_kwargs:dict,verbose:bool = True):
        """_summary_

        Parameters
        ----------
        neighbour_idx_list : list
            a list of indices of voxels that should be included in each searchlight sphere
        thread_id : int
            thread id for the current job, used for display
        total : int
            total number of voxels to perform searchlight on across all jobs
        estimator_kwargs : dict
            _other arguments passed to instantiate an estimator class in each sphere
        verbose : bool, optional
            display progress or not, by default True

        Returns
        -------
        result: numpy.array
            searchlight results for the list of voxels 
        estimator_details: dict
            details of estimator returned by the estimator class. values in the dictionary should be serialized and ready to be written in to json format.
        """
        voxel_results = []
        t0 = time.time()
        for i,neighbour_idx in enumerate(neighbour_idx_list):
            if neighbour_idx.size==0:
                voxel_results.append([])
            else:
                # create preproc function
                ####find the step of mvnn and pass the residual matrix as input
                if "MVNN" in self.preproc_steps.keys():
                    self.preproc_steps["MVNN"][1]["residualmatrix"] = self.R[:,neighbour_idx]
                preproc_func = chain_steps(*list(self.preproc_steps.values()))
                preproc_X = preproc_func(self.X[:,neighbour_idx])
                # instantiate estimator for current voxel
                curr_estimator =  self.estimator(
                    preproc_X,
                    **estimator_kwargs)
                # perform estimation
                voxel_results.append(curr_estimator.fit().result)
            if verbose:
                step = 10000 # print every 10000 voxels
                if  i % step == 0:
                    crlf = "\r" if total == len(neighbour_idx_list) else "\n"
                    pt = round(float(i)/len(neighbour_idx_list)*100,2)
                    dt = time.time()-t0
                    remaining = (100-pt)/max(0.01,pt)*dt
                    sys.stderr.write(
                        f"job {thread_id}, processed {i}/{len(neighbour_idx_list)} voxels"
                        f"({pt:0.2f}%, {remaining} seconds remaining){crlf}"
                    )
        
        #double check if results are of same length (except for empty voxels)
        voxresult_count = [len(x) for x in voxel_results if len(x)!=0]
        if numpy.all([x == voxresult_count[0] for x in voxresult_count]):
            voxresult_count = voxresult_count[0]
        else:
            raise ValueError("shape of voxel-wise searchlight results mismatch!")
        
        #find empty voxel and fill with nans
        empty_vox = [j for j,x in enumerate(voxel_results) if len(x)==0]
        for k in empty_vox:
            voxel_results[k] = numpy.full_like(voxresult_count,fill_value=numpy.nan)
        
        #get estimator information to write into json file
        estimator_details = curr_estimator.get_details()
        sys.stderr.write(f"job {thread_id}, processed {len(neighbour_idx_list)} voxels in {time.time()-t0} seconds\n")
        return numpy.asarray(voxel_results),estimator_details

    def genPatches(self):
        """
        extract wholebrain activity pattern matrix and find indices of voxels that should be included in each searchlight sphere
        """
        print("generating searchlight patches")
        t0 = time.time()
        ## voxels to perform searchlight on
        process_mask_data = self.process_mask_img.get_fdata()
        process_coords = numpy.where(process_mask_data!=0)
        process_coords = numpy.asarray(
            nilearn.image.coord_transform(
                process_coords[0],
                process_coords[1],
                process_coords[2],
                self.process_mask_img.affine
                )
            ).T

        niimgs = [self.pattern_img,self.resid_img] if self.resid_img is not None else self.pattern_img
        Xs,A = _apply_mask_and_get_affinity(
                seeds    = process_coords,
                niimgs    = niimgs,
                radius   = self.radius,
                mask_img = self.mask_img,# only include voxel in the mask
                empty_policy = "ignore",
                n_vox = 1,
                allow_overlap = True
                )        
        A = A.tocsr()
        print(f"finished generating searchlight patches in {time.time()-t0}") 

        voxel_neighbours = []
        for _,row in enumerate(A):
            _, vidx, _ = find(row)
            voxel_neighbours.append(vidx) 
    
        return Xs, A, voxel_neighbours