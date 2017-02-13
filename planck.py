# Planck brightness functions
import numpy as np
from astropy.constants import h, k_B, c
from astropy import units as u
from IPython.core.debugger import Tracer; debug_here=Tracer()

def planck(f,T):
    """Return intensity for planck blackbody function. If f or T do
    not have astropy units associated, they are assumed to be in Hz and
    K. Output is astropy quantity, default W/m^2/Hz/sr."""

    if type(f) is not u.quantity.Quantity:
        f=f*u.Hz

    if type(T) is not u.quantity.Quantity:
        T=T*u.K

    I = (2*h*f**3/c**2)*(np.exp(h*f/(k_B*T))-1)**(-1)*(1/u.steradian)

    return I.to(u.W/(u.m)**2/u.Hz/u.sr)


def Tb(f,I):
    """Thermodynamic temperature for given intensity, i.e. inverse of planck
    blackbody function. If f or I do not have astropy units associated, they are
    assumed to be Hz and W/m^2/Hz/sr."""
    
    if type(f) is not u.quantity.Quantity:
        f=f*u.Hz

    if type(I) is not u.quantity.Quantity:
        I=I*u.W/(u.m)**2/u.Hz/u.sr

    T = (h*f/k_B)/np.log(1+2*h*f**3/(I*u.sr*c**2))

    return T.to(u.K)

def I2Ta(f,I):
    """Antenna temperature (i.e. Rayleigh Jeans temperature, i.e. brightness
    temperature) for given intensity. If f or I do not have astropy units
    associated, they are assumed to be Hz and W/m^2/Hz/sr."""
    
    if type(f) is not u.quantity.Quantity:
        f=f*u.Hz

    if type(I) is not u.quantity.Quantity:
        I=I*u.W/(u.m)**2/u.Hz/u.sr

    Ta = I*u.sr*c**2/(2*f**2*k_B)

    return Ta.to(u.K)

def Ta2I(f,Ta):
    """Intensity for given Antenna temperature, Ta. Inverse of Ta. If f or Ta do not
    have astropy units associated, they are assumed to be Hz and K. Returns
    intensity as astropy unit quantity, default W/m^2/Hz/sr
    """
    
    if type(f) is not u.quantity.Quantity:
        f=f*u.Hz

    if type(Ta) is not u.quantity.Quantity:
        Ta=Ta*u.K

    I = 2*Ta*f**2*k_B/(c**2*u.sr)

    return I.to(u.W/(u.m)**2/u.Hz/u.sr)
