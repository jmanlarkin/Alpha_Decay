import numpy as np
import pandas as pd
import spinmob as sm
import mcphysics as mphys
import matplotlib.pyplot as plt
from os import listdir
from os import path
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.stats.distributions import chi2

plt.ioff()

#Load all .Chn files in specified directory 'loc'
def Load(loc):
    var = sorted(listdir(str(loc)))
    index = len(var)
    data = np.zeros((index, 2048))
    for i in range(index):
        data[i] = mphys.data.load_chn(str(loc) + '/' + var[i])[1]
    return data

#Produce a Plot of Curve-Fitted Data
def Plot(x, y, yfit, err, Name, plotFig = True, saveFig = True, xlab = 'Channel Number', ylab = 'Counts'):
    residuals = y - yfit
    
    fig, ax = plt.subplots(2, 1, sharey='row', tight_layout = True, gridspec_kw={'height_ratios':[4,1]})
    ax[0].errorbar(x, y, yerr=err, fmt='o', markersize='4', label='Data Points')
    ax[0].plot(x, yfit, color='orange', zorder=-5, linewidth=1.85, label='Gaussian Fit')
    ax[1].errorbar(x, residuals, yerr=err, fmt='o', markersize='4')
    ax[1].plot(x, np.zeros(len(x)), zorder=-5, color='orange', linewidth=1.75)
    ax[0].set_xlim(x[0], x[-1])
    ax[1].set_xlim(x[0], x[-1])
    ax[0].ticklabel_format(style='plain')
    ax[1].ticklabel_format(style='plain')
    ax[0].ticklabel_format(useOffset=False)
    ax[1].ticklabel_format(useOffset=False)
    
    ax[1].set_xlabel(str(xlab), fontsize=14)
    ax[1].set_ylabel('Residuals', fontsize=14)
    ax[0].set_ylabel(str(ylab), fontsize=14)
    ax[0].set_title(str(Name), fontsize=14)
    ax[0].legend()
    
    if (plotFig == True) and (saveFig == True):
        fig.savefig('Plots/' + str(Name) + '.svg')
        plt.show()
        
    if (plotFig == True) and (saveFig == False):
        plt.show()
        
    if (plotFig == False) and (saveFig == True):
        plt.close(fig)
        fig.savefig('Plots/' + str(Name) + '.svg')
        plt.show()
        
    if (plotFig == False) and (saveFig == False):
        plt.close(fig)
        plt.show()
    
#Linear Function
def Linear(x, a, b):
    return a*x + b

def LinearInv(y, a, b):
    return (y - b) / a

#Chi Squared for Linear
def ChiSq_Linear(x, y, err, param, sig = 0.05):
    input_chi2 = 1. - sig
    index = len(err)
    yfit = Linear(x, *param)
    residuals = y - yfit
    out = []
    for i in range(index):
        if err[i] != 0:
            value = (residuals[i]/err[i])**2
            out.append(value)
    dof = len(out) - 2
    chi_theoretical = chi2.ppf(input_chi2, df=dof)/dof
    chi_calc = np.sum(out)/dof
    return [chi_calc, chi_theoretical]
    
#Gaussian Function
def Gaussian(x, m, o, A):
    return A*np.exp(-0.5*(((x-m)/o)**2))

#Chi Squared for Gaussian -- removes 0 err values from dof calc
#Optional parameter for signficance 'sig'
def ChiSq_Gauss(x, y, err, param, sig = 0.05):
    input_chi2 = 1. - sig
    index = len(err)
    yfit = Gaussian(x, *param)
    residuals = y - yfit
    out = []
    for i in range(index):
        if err[i] != 0:
            value = (residuals[i]/err[i])**2
            out.append(value)
    dof = len(out) - 2
    chi_theoretical = chi2.ppf(input_chi2, df=dof)/dof
    chi_calc = np.sum(out)/dof
    return [chi_calc, chi_theoretical]

#Fitting function for Gaussian peaks with specified width
#Optional parameters for plotting and saving the resulting fits
#DATA MUST BE SINGLY PEAKED
def Fit_Gauss(data, width = 10, plotFig = False, saveFig = False):
    xdata = np.arange(0, 2048)
    index = len(data)
    out = []
    for i in range(index):
        ydata = np.asarray(data[i])
        peaks = find_peaks(ydata, height = 10)[0]
        if len(peaks) > 1:
            peak = int(np.mean(peaks))
        else:
            peak = peaks[0]
        xfit = xdata[peak-width: peak+width+1]
        yfit = ydata[peak-width: peak+width+1]
        pov, cov =  curve_fit(Gaussian, xfit, yfit, p0 = [peak, width/2, 200])
        err = np.sqrt(np.diag(cov))
        center = int(pov[0])
        x = xdata[center-width: center+width+1]
        y = ydata[center-width: center+width+1]
        yerr = np.sqrt(y)
        chi2 = ChiSq_Gauss(x, y, yerr, pov)
        out.append([pov, err, *chi2])
        
        y_gauss = Gaussian(x, *pov)
        
        Plot(x, y, y_gauss, yerr, 'Peak_' + str(i+1), plotFig, saveFig)
        
    return out

#Fitting for double-Gaussian peak of Am-241
def linear_Gauss(x, m, o, A, B, C):
    return Gaussian(x, m, o, A) + B * x + C

def Chi2(x, y, err, param, func, sig = 0.05):
    input_chi2 = 1. - sig
    index = len(err)
    yfit = func(x, *param)
    residuals = y - yfit
    out = []
    for i in range(index):
        if err[i] != 0:
            value = (residuals[i]/err[i])**2
            out.append(value)
    dof = len(out) - len(param)
    chi_theoretical = chi2.ppf(input_chi2, df=dof)/dof
    chi_calc = np.sum(out)/dof
    return [chi_calc, chi_theoretical]

def ChiSq_Gauss(x, y, err, param, sig = 0.05):
    input_chi2 = 1. - sig
    index = len(err)
    yfit = Gaussian(x, *param)
    residuals = y - yfit
    out = []
    for i in range(index):
        if err[i] != 0:
            value = (residuals[i]/err[i])**2
            out.append(value)
    dof = len(out) - 2
    chi_theoretical = chi2.ppf(input_chi2, df=dof)/dof
    chi_calc = np.sum(out)/dof
    return [chi_calc, chi_theoretical]

def Fit(data, center, width = 30):
    xdata = np.asarray(range(2048))
    ydata = np.asarray(data)
    center = int(center)
    interval = [center - width, center + width + 1]
    pov, cov = curve_fit(Gaussian, xdata[interval[0]:interval[1]], ydata[interval[0]:interval[1]], p0 = [center, width/2, 200])
    err = np.sqrt(np.diag(cov))
    center = int(pov[0])
    interval = [center - width, center + width + 1]
    x = xdata[interval[0]:interval[1]]
    y = ydata[interval[0]:interval[1]]
    yerr = np.sqrt(y)
    chi2 = ChiSq_Gauss(x, y, yerr, pov)
    return [pov, err, *chi2]