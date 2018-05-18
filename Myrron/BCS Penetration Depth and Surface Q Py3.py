# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 13:18:52 2018

@author: ASUS
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit # for curve fitting
import cmath
import time
import pandas as pd

start_time = time.time()

"""Class Use====================="""
"""
class my_array(np.array):
    def find(self, b):
        r = array(range(len(b)))
        return r(b)
"""
"""FUNCTIONS====================="""
#Extrapolation Function
def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y
    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)
    def ufunclike(xs):
        return np.array(map(pointwise, np.array(xs)))
    return ufunclike

def complex_quad(func, a, b, **kwargs):
    def real_func(x):
        return scipy.real(func(x))
    def imag_func(x):
        return scipy.imag(func(x))
    real_integral = integrate.quad(real_func, a, b, **kwargs)
    imag_integral = integrate.quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])

def sigma_1(omega, T): #omega = resfreq / delta, T = T/T_C, epsilon = epsilon / delta #1/T = 0 =? float division by zero
    Tcerr = (1/T) if T != 0.0 else 0.0 # this is to prevent zero division error: float division by zero python error
    integrand = lambda epsilon: (((epsilon + omega) * epsilon + 1.0)/ \
                (cmath.sqrt((epsilon + omega)**2.0 - 1.0)*cmath.sqrt(epsilon**2.0 - 1.0)))* \
                (np.tanh((epsilon + omega)*Tcerr/(2.0 * T_C)) - np.tanh(epsilon*Tcerr/(2.0 * T_C)))
                #the evaluation of omega can be zero due to the weird declaration made by catelani. If that's the case. Then the difference of np.tanh is integer 0.
    result = complex_quad(integrand, 1.0, +np.infty)[0]
    if omega > 2.0: #this argument should be omega < 2 in Catelani's notebook
        result += complex_quad(integrand, 1.0 - omega, -1.0)[0]
    return result/np.pi

def sigma_2(omega, T):
    Tcerr = (1/T) if T != 0.0 else 0.0 # this is to prevent zero division error: float division by zero python error
    #omega = np.float_(omega)
    #must intercept zero division error in integrand
    integrand = lambda epsilon: (((epsilon + omega) * epsilon + 1.0)/ \
                (np.emath.sqrt((epsilon + omega)**2.0 - 1.0)*np.emath.sqrt(1.0 - epsilon**2.0)))* \
                np.tanh((epsilon + omega)*Tcerr/(2.0 * T_C))
    result = complex_quad(integrand, max(-1.0,1.0-omega), 1.0)[0]
    return result/np.pi
#BCS surface impedance based on gianlucci

def rsf(omega, Delta, t1, n):
    """Real Impedance - copied from notebook by Gianluigi Catelani of RWTH"""
    """Catelani assigns i; we assign j in liue with python programming"""
    """n=-1/2 for the dirty limit (pippard) or n=-1/3 for the anomalous limit (london limit)"""
    """Python is really weird. Results are opposite of Gianluigi's mathematica notebook. I think j = -i"""
    r = (1/omega) if omega != 0.0 else 0.0
    z = (1/Delta) if Delta != 0.0 else 0.0 # this is to prevent zero-division error
    ans = ((-1)*np.exp(-1j*(np.pi/2)*(1.0-n)))*(((sigma_1(t1*z, omega*z) - 1j*sigma_2(t1*z, omega*z))*(np.pi*Delta*r))**n)
    """modified from Catelani's data"""
    return ans.real
# The zero division error is so annoying. Resolution is to delete the unwanted zeros 
def xsf(omega, Delta, t1, n): #there's no problem with this part since we made an exception with 0. Problem lies in Sigma_2
    """Python is really weird. Results are opposite of mathematica. I think j = -i"""
    """Imaginary Impedance - copied from notebook by Gianluigi Catelani of RWTH"""
    """n=-1/2 for the dirty limit (pippard) or n=-1/3 for the anomalous limit (london limit)"""
    r = (1/omega) if omega != 0.0 else 0.0
    z = (1/Delta) if Delta != 0.0 else 0.0 # this is to prevent zero-division error
    ans = ((-1.0)*np.exp(-1j*(np.pi/2)*(1.0-n)))*(((sigma_1(t1*z, omega*z) - 1j*sigma_2(t1*z, omega*z))*(np.pi*Delta*r))**n)
    """Modified Catelani's Notebook"""
    return ans.imag
# The zero division error is so annoying. 
def delf(tmK, TC, a, b): #tmk here is a numpy array, TC is the guess TC data, a and b are the guess data
    # print(tmK)
    tau = tmK/TC
    res = np.zeros(len(tmK))
    DeltaF = Delta_f(tmK / TC) #by LTH
    for i in range(len(tmK)):
        res[i] += a*(xsf(omega_r, DeltaF[i], _Tnormexp_[0], n_dirty)-xsf(omega_r, DeltaF[i], tau[i], n_dirty)) * (tau[i] <= 1) # this is the culprit
        res[i] += b*(tau[i] > 1)
        res[i] *= fdata/(1E6) #data is expressed in kHz
    return res
#Delta_f must be single-valued, not an array
"""Interpolate BCS Gap Equation==============================="""
df = pd.read_csv('BCS Gap Equation 2.txt', sep='\t')
BCSvalues = df.values
_Tnorm_= BCSvalues[:,0]
_Gapnorm_= BCSvalues[:,1]
Delta_f = interp1d(_Tnorm_, _Gapnorm_, kind='quadratic')
Delta_f = extrap1d(Delta_f) #Delta is given
"""Import Data for Interpolation of BCS function=============================="""
dfexp=pd.read_csv('Frequency shift vs Temperature text file.txt', delimiter=',')
values_1 = dfexp.values
TCtest = 1189 #TC in mK
T_C = TCtest/1000

_TmK_ = values_1[:,0] #x-data from experiment
_Fshift_ = values_1[:,1] #y-data from experiment
_Tnormexp_ = _TmK_/TCtest
# how do we elegantly select temperatures that has BCS frequencies
 #finding the index whose temperature is less than one        
a = np.where(_Tnormexp_ <= 1)[0] # argument for finding index of the array less than 1
# creates an option for length of data to consider all of the data
opt = 0 # if opt = 0, we face ZeroDivisionError due to 0 entries in xsf1 and rsf1 but if we make opt = 1, we don't face that error
if opt == 1: 
    b = len(_Tnormexp_)
else:
    b = a.size # we automatically find the index relevant for BCS analysis

#print b
_Tnormgap_ = np.zeros(b)
Delta_gap = np.zeros(b)
_Fshiftgap_ = np.zeros(b)
Delta_dum = Delta_f(_Tnormexp_)
print(Delta_dum)
for i in range(b):
        _Tnormgap_[i]=_Tnormexp_[i]
        _Fshiftgap_[i]=_Fshift_[i]
        #if _Tnormgap_[i] <= 1:
        # print(list(Delta_dum)) #by LTH
        Delta_gap[i]=Delta_dum[i]
        #print i #this is for diagnostics
#plt.plot(_TmK_, _Fshift_, marker='.', ls='', color='b')
plt.plot(_Tnorm_, _Gapnorm_, color='b')
plt.plot(_Tnormgap_, Delta_gap, marker='.', ls='', color='r')
plt.xlabel(r'$T/T_C$')
plt.ylabel(r'$\Delta(T)/\Delta_0$')
plt.show()
"""CURVE FIT PARAMETERS for penetration depth determination==============="""
hbar =  6.582119514*10**-16  #eV*s
mu0 = 1.25663706E-6 #m kg s-2 A-2 

fdata = 9.106 * 10 ** 9 #Our cavity has 7.33787 * 10 ** 9 Hz. Reagor has about 9 GHz
delta_Al = 0.186
freqmeV = (2.0 * np.pi * fdata * hbar) * 1000.0 #
omega_r = (freqmeV / delta_Al) #omega_r = resfreq / delta_Al in eV (normalized for universality)
n_dirty = -0.5 

pmag = 6.0E-6 #Reagor p.  Better make this iterative for completeness # we face errors due 
lambda0 = 100E-9 #London penetration depth in nm
Axs = (pmag / (4.0 * np.pi * mu0 * lambda0)) / 1000.0  #guess for frequency shift in kHz
Bxs = -45.0 #guess for linear fit in kHz

"""SETTING ARRAYS================================"""

sigma1 = []
sigma2 = []
xsf1 = []
rsf1 = []
delxsf = []
rsfinv =[]

for i in range(len(_Tnormgap_)):
    sigma1.append(sigma_1(omega_r, _Tnormgap_[i]))
    sigma2.append(sigma_2(omega_r, _Tnormgap_[i]))
    xsf1.append(xsf(omega_r ,Delta_gap[i], _Tnormgap_[i], n_dirty))
    rsf1.append(rsf(omega_r ,Delta_gap[i], _Tnormgap_[i], n_dirty))
    delxsf.append(xsf1[0]-xsf1[i])
    rsfinv.append(1/rsf1[i])

sigma1=np.asarray(sigma1)
sigma2=np.asarray(sigma2)
xsf1=np.asarray(xsf1)
rsf1=np.asarray(rsf1)
delxsf=np.asarray(delxsf)
rsfinv=np.asarray(rsfinv)
# Plot is working so far
"""Employing Curve Fit for frequency shift================================="""
guess = np.array([T_C, Axs, Bxs]) # Cannot fit the data for some reason...
p1, cov1 = curve_fit(delf, _Tnormgap_, _Fshiftgap_, p0=guess, maxfev=10000000)
# Error due to ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
"""Plot Data======================================"""
"""Plot Sigma1 and Sigma2=============================="""
plt.figure()
plt.plot(_Tnormgap_, sigma1, color='b', label='$\sigma_1$')
plt.plot(_Tnormgap_, sigma2, color='r', label='$\sigma_2$')
plt.title('BCS Conductivities at fc=7.33787 GHz')
plt.xlabel(r'$ T/T_C$')
plt.ylabel(r'$\sigma_s\omega/\Delta$')
plt.grid(alpha=0.5)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('plot00.png', bbox_inches='tight', dpi=1000)
"""Plot xsf1 and rsf1============================="""
plt.figure()
plt.plot(_Tnormgap_, xsf1, color='b', label='xsf1')
plt.plot(_Tnormgap_, rsf1, color='r', label='rsf1')
plt.title('BCS Reactance')
plt.xlabel(r'$ T/T_C$')
plt.ylabel(r'$ X_S/X_N $')
plt.grid(alpha=0.5)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('plot01.png', bbox_inches='tight', dpi=1000)
"""Plot Q and fshift============================="""
plt.figure()
plt.plot(_Tnormgap_, delxsf, color='b', label='xsf1')
plt.plot(_Tnormgap_, rsfinv, color='r', label='rsf1')
plt.title('BCS Reactance')
plt.xlabel(r'$ T/T_C$')
plt.ylabel(r'$ X_S/X_N $')
plt.grid(alpha=0.5)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('plot02.png', bbox_inches='tight', dpi=1000)

"""Appendix of Codes======================================================="""
"""1. Setting 0 to Delta beyond TC"""
#Delta_dum = Delta_f(_Tnormexp_)
#Delta_exp=np.zeros(len(_Tnormexp_))
#for i in range(len(_Tnormexp_)):
#   if _Tnormexp_[i] <= 1.0:
#        Delta_exp[i]=Delta_dum[i]   
"""""" 
