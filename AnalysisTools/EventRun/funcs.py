from constants import *
import numpy as np
import math


def linear_res_function(x,return_bool = False):
        if return_bool:
            return np.full_like(x,True).astype(bool)
        else:
            return np.ones_like(x)

def eta_res_function(eta):
        res = 0.1 + 0.2*eta**2
        return 1/res
    
def mva_res_function(MVA,threshold=0.7):
        res = MVA > threshold
        return res

def OnlyPV_function(fake):
        res = fake != 0
        return res

def MaxPt(pt,maxpt = 128):
        res = np.clip(pt, 1, maxpt)
        return res 

def MinPt(pt,minpt = 3):
        res = pt > minpt 
        return res 

def comb_res_function(mva,eta,threshold=0.99):
        sel = (mva > threshold).astype(int)
        res = 0.1 + 0.2*eta**2
        return sel/res

def chi_res_function(chi2rphi,chi2rz,bendchi2,nstub):
        qrphi = chi2rphi/(nstub-2) < 20 
        qrz =  chi2rz/(nstub-2) < 5
        qbend = bendchi2 < 2.25
        qstub = nstub > 3
        q = np.logical_and(qrphi,qrz)
        q = np.logical_and(q,qbend)
        q = np.logical_and(q,qstub)
        return q

def gen_selection(pdgid):
    sel12 = (pdgid != 12)
    sel14 = (pdgid != 14)
    sel16 = (pdgid != 16)
    sel1000022 = (pdgid != 1000022)
    sel = np.logical_and(sel12,sel14)
    sel = np.logical_and(sel,sel16)
    sel = np.logical_and(sel,sel1000022)
    return sel

def predictFastHisto(value,weight,num_vertices=1,weighting=linear_res_function):
    halfBinWidth = 0.5*(2*FH_max_z0)/FH_nbins
    hist,bin_edges = np.histogram(value,FH_nbins,range=(-1*FH_max_z0,FH_max_z0),weights=weight*weighting)
    z0_list = []
    for i in range(num_vertices):
        temp_hist = np.convolve(hist,[1,1,1],mode='same')
        z0Index= np.argmax(temp_hist)
        if z0Index > 0:
            hist[z0Index - 1] = 0
        if z0Index < (FH_nbins-1):
            hist[z0Index + 1] = 0
        hist[z0Index] = 0

        z0 = -1*FH_max_z0 +(2*FH_max_z0)*z0Index/FH_nbins+halfBinWidth
        z0_list.append(z0)
    return z0_list

def FastHistoAssoc(PV,trk_z0,trk_eta):

    deltaz_bins = np.array([0.0,0.37,0.5,0.6,0.75,1.0,1.6,0.0])
    eta_bins = np.array([0.0,0.7,1.0,1.2,1.6,2.0,2.4])
    

    eta_bin = np.digitize(abs(trk_eta),eta_bins)
    deltaz = abs(trk_z0 - PV)

    assoc = (deltaz < deltaz_bins[eta_bin])

    return np.array(assoc,dtype=bool)

def CreateHisto(value,weight, res_func, return_index = False,factor=1):

    hist,bin_edges = np.histogram(value,FH_nbins,range=(-1*FH_max_z0,FH_max_z0),weights=weight,density=True)
    hist = np.clip(hist,0,1)
    
    return hist/factor,bin_edges

def predictMET(pt,phi,selection):

    pt = pt[selection]
    phi = phi[selection]

    met_px = np.sum(pt*np.cos(phi))
    met_py = np.sum(pt*np.sin(phi))
    met_pt = math.sqrt(met_px**2+met_py**2)
    return met_pt

def calc_widths(actual,predicted,relative=False):
    if relative:
        diff = (predicted-actual)/actual
    else:
        diff = (predicted-actual)
    RMS = np.sqrt(np.mean(diff**2))
    qs = np.percentile(diff,[32,50,68])
    qwidth = qs[2] - qs[0]
    qcentre = qs[1]

    return [RMS,qwidth,qcentre]