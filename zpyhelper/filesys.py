
import os
from typing import Union
import json

def checkdir(dirs:Union[list,str]):
    """check if directories exist, if not, generate directories

    Parameters
    ----------
    dirs : list or str
        a list of directory strings or a directory string

    Raises
    ------
    Exception
        input must be list or string
    """
    if isinstance(dirs, list):
        raise AssertionError("dirs must be a list of directory strings or a directory string")

    if isinstance(dirs,str):
        dirs = [dirs]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

def is_jsonable(x)->bool:
    """check if an object is JSON serializable

    Parameters
    ----------
    x : any
        the object to check

    Returns
    -------
    bool
        `True`: serializable
    """
    try:
        json.dumps(x)
        return True
    except:
        return False
    
