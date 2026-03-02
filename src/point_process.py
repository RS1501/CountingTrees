import sys

sys.path.append(".")

from typing import Callable, Literal
import numpy as np
from scipy.optimize import minimize
from abc import abstractmethod
from src.utils import *
from matplotlib import pyplot as plt
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from scipy.special import j0
from numpy.linalg import norm


class PointProcess(): 
    def __init__(self):
        pass
    @abstractmethod
    def generate(self):
        pass
    
    
class PoissonProcess(PointProcess):
    def __init__(self, lam: Callable[[tuple[float, float]], float]|float|int):
        super().__init__()
        self.lam = lam

    def generate(self, W: tuple[float, float], seed: int=None):
        if seed is not None: np.random.seed(seed)
        width, height = W
        
        # Case 1: Homogeneous Poisson process
        if isinstance(self.lam, (float, int)): 
            nb_points = np.random.poisson(self.lam*width*height)
            points = np.random.uniform([0, 0], [width, height], size=(nb_points, 2))
            
        # Case 2; Inhomogeneous Poisson process
        elif isinstance(self.lam, Callable):
            x_max = minimize(lambda x: -self.lam(x), x0 = (width/2, height/2), bounds=[(0, width), (0, height)]).x
            lam_max = self.lam(x_max)
            nb_points = np.random.poisson(lam_max*width*height)
            points = np.random.uniform([0, 0], [width, height], size=(nb_points, 2))
            points_thinned = []
            np.random.seed(None)
            for x in points:
                u = np.random.random()
                if u <= self.lam(x)/lam_max: 
                    points_thinned.append(x)
            points = np.array(points_thinned)
                
        return points
    

class ThomasProcess(PointProcess):
    def __init__(self, lam_p: Callable[[tuple[float, float]], float]|float|int, nu: float, sig: float):
        super().__init__()
        self.lam_p = lam_p
        self.nu = nu
        self.sig = sig
        
    def generate(self, W: tuple[float, float], seed: int=None, format: Literal['separate', 'concatenated']='concatenated', trim: bool=True):
        if seed is not None: np.random.seed(seed=seed)
        parent_points = PoissonProcess(self.lam_p).generate(W, seed)
        nb_parents = len(parent_points)
        
        nb_offsprings = np.random.poisson(self.nu, size=nb_parents)
        offsprings = []
        
        for i in range(nb_parents):
            daughters = np.random.normal(loc=parent_points[i, :], scale=self.sig, size=(nb_offsprings[i], 2))
            if trim: daughters = trim_process(daughters, (0, 0), W)
            offsprings.append(daughters)
        
        if format == 'separate':
            return parent_points, offsprings
        elif format == 'concatenated':
            return parent_points, np.concatenate(offsprings)


def display(parents: np.ndarray=None, 
            offsprings: list[np.ndarray]|np.ndarray=None, 
            window: tuple[float, float]=(1, 1), 
            title: str="Point process", 
            path: str=None): 
    
    plt.rcParams.update({
    "text.usetex": True,          # Use LaTeX to render text
    "font.family": "serif",       # Match LaTeX serif font
    "font.size": 12
    })

    
    marker = '.'; label = None; legend=False
    if offsprings is not None:
        marker = 'x'; label = 'Parent points'; legend=True
        if isinstance(offsprings, list):
            for offspring in offsprings:
                plt.scatter(offspring[:, 0], offspring[:, 1], marker='.')
        elif isinstance(offsprings, np.ndarray):
            plt.scatter(offsprings[:, 0], offsprings[:, 1], marker='.', color='#1f77b4')
    
    if parents is not None:
        plt.scatter(parents[:, 0], parents[:, 1], marker=marker, label=label, color='black')
    
    if legend: plt.legend()
    plt.xlim(0, window[0])
    plt.ylim(0, window[1])
    plt.xlabel('Abscissa x')
    plt.ylabel('Ordinate y')
    plt.title(title)
    if path is not None:
        plt.savefig(path, format='eps', bbox_inches='tight')
    plt.show()
    plt.close()
    
    
def display_superposition(points: list[np.ndarray]=None,
                          window: tuple[float,float]=(1,1),
                          title: str="Superposition of point processes",
                          labels: list[str]=[], 
                          path: str=None):
    
    plt.rcParams.update({
    "text.usetex": True,          # Use LaTeX to render text
    "font.family": "serif",       # Match LaTeX serif font
    "font.size": 12
    })

    if points is not None:
        marker='.'
        for i in range(len(points)):
            plt.scatter(points[i][:,0],points[i][:,1],marker=marker,label=labels[i] if i < len(labels) else f"label{i+1}")
        plt.legend()
    plt.xlim(0, window[0])
    plt.ylim(0, window[1])
    plt.xlabel('Abscissa x')
    plt.ylabel('Ordinate y')
    plt.title(title)
    if path is not None:
        plt.savefig(path, format='eps', bbox_inches='tight')
    plt.show()
    plt.close()


def intensity(points: np.ndarray, W: tuple[float, float], estimator: Literal['standard', 's']='standard', r: float=None):
    w, h = W
    
    # Case 1: Standard intensity estimator
    if estimator == 'standard': 
        return points.shape[0]/(w*h)
    
    # Case 2: S intensity estimator
    elif estimator == 's':
        if r is None: raise ValueError("Assign a value to the parameter r")
        s = 0
        for point in points:
            x, y = point
            nu_1 = rect(W, x, y, r)[1]
            s += nu_1/isotropised_set_covariance(r,W)
            
        return s/(2*np.pi*r)
    
    else: raise ValueError("Invalid estimator name")


def rho(points: np.ndarray, W: tuple[float, float], h: float, r: float) -> float:
    s = 0
    if r < h:
        for x1 in points:
            for x2 in points:
                if (x1 != x2).all():
                    s += box_kernel(h, r - np.linalg.norm(x1 - x2))/(volume_intersection(W, x1 , x2)*np.linalg.norm(x1 - x2))
        return (2*h/(r+h))*s/(2*np.pi)
    else:
        for x1 in points:
            for x2 in points:
                if (x1 != x2).all():
                    s += box_kernel(h, r - np.linalg.norm(x1 - x2))/volume_intersection(W, x1 , x2)
        return s/(2*np.pi*r)


def g(points: np.ndarray, W: tuple[float, float], h: float = None, r_values: np.ndarray = None, correction: str = 'iso') -> tuple[float, float]:
    """Uses an Epanechnikov kernel of bandwidth `h` to estimate the pcf of a point process, for a radius range `r_values` (if not specified the R function pcf chooses a default range).

    Returns:
       tuple: `(r_values, pcf_values)`
    """
    width, height = W
    x, y = points[:, 0], points[:, 1]
    
    # Initialize global variables in R
    ro.globalenv['width'] = width
    ro.globalenv['height'] = height
    ro.globalenv['x'] = ro.FloatVector(x)
    ro.globalenv['y'] = ro.FloatVector(y)
    if h:
        ro.globalenv['h'] = h
    if r_values is not None:
        ro.globalenv['r_values'] = ro.FloatVector(r_values)
        
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1)  # select a CRAN mirror

    if not rpackages.isinstalled('spatstat'):
        utils.install_packages('spatstat')

    # # Install and load the spatstat package in R
    # if not rpackages.isinstalled('spatstat'):
    #     rpackages.install('spatstat')
    ro.r('library(spatstat)')
    
    # Define the corresponding point process as a spatstat ppp object
    ro.r('W <- owin(c(0, width), c(0, height))')
    ro.r('pp_thomas <- ppp(x, y, window = W, marks = NULL)')
    
    # Estimate the pcf
    if r_values is not None and h:
        ro.r('pcf_results <- pcf(pp_thomas, r = r_values, bw = h)')
    elif h:
        ro.r('pcf_results <- pcf(pp_thomas, bw = h)')
    elif r_values is not None:
        ro.r('pcf_results <- pcf(pp_thomas, r = r_values)')
    else:
        ro.r('pcf_results <- pcf(pp_thomas)')
        
    # Retrieve the results
    r_values = np.array(ro.r('pcf_results$r'))
    if correction == 'iso':
        g_values = np.array(ro.r('pcf_results$iso'))
    elif correction == 'trans':
        g_values = np.array(ro.r('pcf_results$trans'))

    return r_values, g_values


def periodogram(x,y,lambda_hat,points,area):  # To compute the peridogram
    n=lambda_hat

    for T in points:
        for D in points:
            if T[0]==D[0] and T[1]==D[1]:
                n+=0
            else:
                n+=(np.exp(-2*np.pi*1j*(x*(T[0]-D[0])+y*(T[1]-D[1]))))/area

    return n.real 

def periodogram_bessel(points, W, r):
    s = 0
    for x in points:
        for y in points:
            if (x != y).any():
                s += j0(2*np.pi*r*norm(x - y))
    s = intensity(points, W) + s/(W[0]*W[1])
    return s


# def h(r, W): #To compute the data taper with a Gaussian window.
#     sigma=W[0]/5 
#     return np.exp(-(r**2)/(2*sigma**2))/(np.sqrt(sigma*np.pi**(3/2)))

# def h(r, W): #To compute the data taper with a Gaussian window.
#     alpha = 25/(4*W[0]**2)
#     return np.exp(-alpha*r**2)

# def h(r, W): #To compute the data taper with a Gaussian window.
#     a = 0.008
#     return np.exp(-r**2/(2*a**2))

'''
def h(r,W): #To compute the data taper with a Spherical cosine window
    R= (W[0]+W[1])/2

    return np.cos((r*np.pi)/(2*R))/np.sqrt(R*(np.pi-1))
'''

'''
def h(r,W): #To compute the data taper with an isotropic Laplace window
    c=W[0]/5 

    return np.exp(-r/c)/np.sqrt(c*np.pi)
'''


def periodogram_tapering(points, W, r, taper):
    s = 0
    for x in points:
        for y in points:
            if (x != y).any():
                s += j0(2*np.pi*r*norm(x - y))*taper(norm(x - y), W)
    s = intensity(points, W) + s
    return s