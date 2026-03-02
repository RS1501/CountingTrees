import numpy as np
import math
import time

# Color palette

color = {'darkblue': '#1f77b4', 
         'teal': '#17becf', 
         'darkgreen': '#2ca02c', 
         'darkgray': '#7f7f7f', 
         'orange': '#ff7f0e'}


## Utility functions

def volume_intersection(W: tuple[float, float], x1: np.ndarray, x2: np.ndarray) -> float:
    return (W[0] - abs(x2[0] - x1[0])) * (W[1] - abs(x2[1] - x1[1]))


def trim_process(points: np.ndarray, bottom_left: tuple[float, float], W: tuple[float, float]) -> np.ndarray:
    
    x1, x2 = bottom_left[0], bottom_left[0] + W[0]
    y1, y2 = bottom_left[1], bottom_left[1] + W[1]
    
    valid_x = (points[:, 0] >= x1) & (points[:, 0] <= x2)
    valid_y = (points[:, 1] >= y1) & (points[:, 1] <= y2)
    
    return points[valid_x & valid_y]


def isotropised_set_covariance(r: float, W: tuple[float, float]) -> float:
    w, h = W
    if r <= w:
        return w*h - 2*r*(w + h)/np.pi + r**2/np.pi
    elif w < r <= h:
        return w*h*(2*math.asin(w/r) - w/h - 2*(r/w - np.sqrt(r**2/w**2 - 1)))/np.pi
    
    
def f(d1: float, d2: float, r: float, ra: list) -> None:
    r2 = r * r
    w = 0  # Initialize w

    if d1 < r:
        d3 = math.sqrt(r2 - d1 * d1)
        if d3 < d2:
            w = math.atan(d2 / d1) - math.atan(d3 / d1)
            ra[0] += 0.5 * (d1 * d3 + r2 * w)
            ra[1] += w * r
        else:
            ra[0] += 0.5 * d1 * d2
    else:
        w = math.atan(d2 / d1)
        ra[0] += 0.5 * r2 * w
        ra[1] += w * r


def rect(W: tuple[float, float], x: float, y: float, r: float) -> list:
    w, h = W

    ra = [0.0, 0.0]  # Initialize ra array

    f(w - x, h - y, r, ra)
    f(w - x, y, r, ra)
    f(y, w - x, r, ra)
    f(y, x, r, ra)
    f(x, y, r, ra)
    f(x, h - y, r, ra)
    f(h - y, x, r, ra)
    f(h - y, w - x, r, ra)

    return ra


def box_kernel(h:float, z:float) -> float:
    return np.where(np.abs(z) <= h, 1 / (2 * h), 0)


def w(W, x1: tuple[float, float], x2: tuple[float, float]) -> float:
    return rect(W, x1[0], x1[1], np.linalg.norm(x1 - x2))[1]


def format_time(seconds: float|int):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))


def g_theo_thomas(r, theta: tuple[float, float]) -> float:
    """Returns the theoretical value of the PCF for a Thomas process.

    Args:
        theta (tuple): (lam_p, sig)
    """
    lam_p, sig = theta
    return 1 + np.exp(-r**2/(4*sig**2)) / (4*np.pi*lam_p*sig**2)

def g_theo_2thomas(r, lam_hat, theta):  # theta = [mu, lam_p1, sig1, sig2]
    mu, lam_p1, sig1, sig2 = theta
    nu1 = mu*sig1**2
    nu2 = mu*sig2**2
    lam_p2 = (lam_hat - nu1*lam_p1)/nu2
    g1 = g_theo_thomas(r, (lam_p1, sig1))
    g2 = g_theo_thomas(r, (lam_p2, sig2))
    lam1 = lam_p1*nu1
    lam2 = lam_p2*nu2
    return  (lam1**2*g1 + lam2**2*g2 + 2*lam1*lam2)/((lam1 + lam2)**2)


def contrast(g_estim, g_theo, dr):
    diff = g_estim - g_theo
    return np.sum((diff**2)*dr)  # Riemann's left sum


# def f(r, theta):  # To compute the Fourier Transform
#     lam_p, nu, sig = theta
#     lam = lam_p*nu
#     return lam + lam*nu*np.exp(-4*np.pi**2*r**2*sig**2)


def f2(r, lambda_hat, theta):  # To compute the Fourier Transform with a tuple of 2 parameters: theta = [nu, sigma]
    nu, sig = theta
    return lambda_hat*(1+nu*np.exp(-4*np.pi**2*r**2*sig**2))

def f3(r, lam_hat, theta):
    '''
    Compute the Fourier transform of a superposition of two Thomas processes, given theta = [mu, lam_p1, sig1, sig2] (nu = mu*sig**2)
    '''
    mu, lam_p1, sig1, sig2 = theta
    nu1 = mu*sig1**2
    nu2 = mu*sig2**2
    f_1 = lam_p1*nu1*(1 + nu1*np.exp(-4*np.pi**2*r**2*sig1**2))
    f_2 = (lam_hat - nu1*lam_p1)*(1 + nu2*np.exp(-4*np.pi**2*r**2*sig2**2))  # Identifiability problem --> nu1*lam_p1 <--
    return f_1 + f_2

def f4(r, lam_hat, theta):
    '''
    Compute the Fourier transform of a superposition of two Thomas processes, given theta = [mu, sig1, sig2] (nu = mu*sig**2), 
    with the assumption lam_1 = lam_2
    '''
    mu, sig1, sig2 = theta
    nu1 = mu*sig1**2
    nu2 = mu*sig2**2
    f_1 = (lam_hat/2)*(1 + nu1*np.exp(-4*np.pi**2*r**2*sig1**2))
    f_2 = (lam_hat/2)*(1 + nu2*np.exp(-4*np.pi**2*r**2*sig2**2))
    return f_1 + f_2

def taper_triangle(r, W, frac=0.2): #To compute the data taper with a Triangular window.
    a = W[0]*frac
    if 1 - r/a > 0:
        return (1 - r/a )/(W[0]*W[1])
    return 0

def taper_gaussian(r, W, frac=0.2): #To compute the data taper with a Gaussian window.
    a = W[0]*frac
    if 1 - r/a > 0:
        return np.exp(-r**2 / (2 * a**2))/(W[0]*W[1])
    return 0
