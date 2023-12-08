# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 18:49:38 2022

@author: Dipanjan
"""


import argparse, os, sys, time
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
import numpy as np
from scipy.interpolate import griddata
from scipy.signal import butter, filtfilt, welch
from scipy.stats import chi2, chi2_contingency
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from statsmodels.stats.multitest import multipletests


def r_xyz(a):
    """Read EEG electrode locations in xyz format

    Args:
        filename: full path to the '.xyz' file
    Returns:
        locs: n_channels x 3 (numpy.array)
    """
    ch_names = []
    locs = []
    ch_names=['Fp1','Fp2','F3','F4','F7','F8','T7','T8','C3','C4','P7','P8','P3','P4','O1','O2','Fz','Cz','Pz']
    locs=[[-2.7,  8.6,  3.6],[ 2.7,  8.6,  3.6],[-4.7,  6.2,  8. ],[ 4.7,  6.2,  8. ],[-6.7,  5.2,  3.6],[ 6.7,  5.2,  3.6],[-7.8,  0. ,  3.6],[ 7.8,  0. ,  3.6],[-6.1,  0. ,  9.7],[ 6.1,  0. ,  9.7],[-7.3, -2.5,  0. ],[ 7.3, -2.5,  0. ],[-4.7, -6.2,  8. ],[ 4.7, -6.2,  8. ],[-2.7, -8.6,  3.6],[ 2.7, -8.6,  3.6],[ 0. ,  6.7,  9.5],[ 0. ,  0. , 12. ],[ 0. , -6.7,  9.5]]
    return ch_names, np.array(locs)


def read_edf(filename):
    """Basic EDF file format reader

    EDF specifications: http://www.edfplus.info/specs/edf.html

    Args:
        filename: full path to the '.edf' file
    Returns:
        chs: list of channel names
        fs: sampling frequency in [Hz]
        data: EEG data as numpy.array (samples x channels)
    """

    def readn(n):
        """read n bytes."""
        return np.fromfile(fp, sep='', dtype=np.int8, count=n)

    def bytestr(bytes, i):
        """convert byte array to string."""
        return np.array([bytes[k] for k in range(i*8, (i+1)*8)]).tostring()

    fp = open(filename, 'r')
    x = np.fromfile(fp, sep='', dtype=np.uint8, count=256).tostring()
    header = {}
    header['version'] = x[0:8]
    header['patientID'] = x[8:88]
    header['recordingID'] = x[88:168]
    header['startdate'] = x[168:176]
    header['starttime'] = x[176:184]
    header['length'] = int(x[184:192]) # header length (bytes)
    header['reserved'] = x[192:236]
    header['records'] = int(x[236:244]) # number of records
    header['duration'] = float(x[244:252]) # duration of each record [sec]
    header['channels'] = int(x[252:256]) # ns - number of signals
    n_ch = header['channels']  # number of EEG channels
    header['channelname'] = (readn(16*n_ch)).tostring()
    header['transducer'] = (readn(80*n_ch)).tostring().split()
    header['physdime'] = (readn(8*n_ch)).tostring().split()
    header['physmin'] = []
    b = readn(8*n_ch)
    for i in range(n_ch):
        header['physmin'].append(float(bytestr(b, i)))
    header['physmax'] = []
    b = readn(8*n_ch)
    for i in range(n_ch):
        header['physmax'].append(float(bytestr(b, i)))
    header['digimin'] = []
    b = readn(8*n_ch)
    for i in range(n_ch):
        header['digimin'].append(int(bytestr(b, i)))
    header['digimax'] = []
    b = readn(8*n_ch)
    for i in range(n_ch):
        header['digimax'].append(int(bytestr(b, i)))
    header['prefilt'] = (readn(80*n_ch)).tostring().split()
    header['samples_per_record'] = []
    b = readn(8*n_ch)
    for i in range(n_ch):
        header['samples_per_record'].append(float(bytestr(b, i)))
    nr = header['records']
    n_per_rec = int(header['samples_per_record'][0])
    n_total = int(nr*n_per_rec*n_ch)
    fp.seek(header['length'],os.SEEK_SET)  # header end = data start
    data = np.fromfile(fp, sep='', dtype=np.int16, count=n_total)  # count=-1
    fp.close()

    # re-order
    #print("EDF reader:")
    #print("[+] n_per_rec: {:d}".format(n_per_rec))
    #print("[+] n_ch: {:d}".format(n_ch))
    #print("[+] nr: {:d}".format(nr))
    #print("[+] n_total: {:d}".format(n_total))
    #print(data.shape)
    data = np.reshape(data,(n_per_rec,n_ch,nr),order='F')
    data = np.transpose(data,(0,2,1))
    data = np.reshape(data,(n_per_rec*nr,n_ch),order='F')

    # convert to physical dimensions
    for k in range(data.shape[1]):
        d_min = float(header['digimin'][k])
        d_max = float(header['digimax'][k])
        p_min = float(header['physmin'][k])
        p_max = float(header['physmax'][k])
        if ((d_max-d_min) > 0):
            data[:,k] = p_min+(data[:,k]-d_min)/(d_max-d_min)*(p_max-p_min)

    #print(header)
    return header['channelname'].split(),\
           header['samples_per_record'][0]/header['duration'],\
           data


def bp_filter(data, f_lo, f_hi, fs):
    """Digital band pass filter (6-th order Butterworth)

    Args:
        data: numpy.array, time along axis 0
        (f_lo, f_hi): frequency band to extract [Hz]
        fs: sampling frequency [Hz]
    Returns:
        data_filt: band-pass filtered data, same shape as data
    """
    data_filt = np.zeros_like(data)
    f_ny = fs/2.  # Nyquist frequency
    b_lo = f_lo/f_ny  # normalized frequency [0..1]
    b_hi = f_hi/f_ny  # normalized frequency [0..1]
    # band-pass filter parameters
    p_lp = {"N":6, "Wn":b_hi, "btype":"lowpass", "analog":False, "output":"ba"}
    p_hp = {"N":6, "Wn":b_lo, "btype":"highpass", "analog":False, "output":"ba"}
    bp_b1, bp_a1 = butter(**p_lp)
    bp_b2, bp_a2 = butter(**p_hp)
    data_filt = filtfilt(bp_b1, bp_a1, data, axis=0)
    data_filt = filtfilt(bp_b2, bp_a2, data_filt, axis=0)
    return data_filt



def topo(data, n_grid=64):
    """Interpolate EEG topography onto a regularly spaced grid

    Args:
        data: numpy.array, size = number of EEG channels
        n_grid: integer, interpolate to n_grid x n_grid array, default=64
    Returns:
        data_interpol: cubic interpolation of EEG topography, n_grid x n_grid
                       contains nan values
    """
    channels=['Fp1','Fp2','F3','F4','F7','F8','T7','T8','C3','C4','P7','P8','P3','P4','O1','O2','Fz','Cz','Pz']
    locs=[[-2.7,  8.6,  3.6],[ 2.7,  8.6,  3.6],[-4.7,  6.2,  8. ],[ 4.7,  6.2,  8. ],[-6.7,  5.2,  3.6],[ 6.7,  5.2,  3.6],[-7.8,  0. ,  3.6],[ 7.8,  0. ,  3.6],[-6.1,  0. ,  9.7],[ 6.1,  0. ,  9.7],[-7.3, -2.5,  0. ],[ 7.3, -2.5,  0. ],[-4.7, -6.2,  8. ],[ 4.7, -6.2,  8. ],[-2.7, -8.6,  3.6],[ 2.7, -8.6,  3.6],[ 0. ,  6.7,  9.5],[ 0. ,  0. , 12. ],[ 0. , -6.7,  9.5]]
    n_channels = len(channels)
    #locs /= np.sqrt(np.sum(locs**2,axis=1))[:,np.newaxis]
    locs /= np.linalg.norm(locs, 2, axis=1, keepdims=True)
    c = findstr('Cz', channels)[0]
    # print 'center electrode for interpolation: ' + channels[c]
    #w = np.sqrt(np.sum((locs-locs[c])**2, axis=1))
    w = np.linalg.norm(locs-locs[c], 2, axis=1)
    #arclen = 2*np.arcsin(w/2)
    arclen = np.arcsin(w/2.*np.sqrt(4.-w*w))
    phi_re = locs[:,0]-locs[c][0]
    phi_im = locs[:,1]-locs[c][1]
    #print(type(phi_re), phi_re)
    #print(type(phi_im), phi_im)
    tmp = phi_re + 1j*phi_im
    #tmp = map(complex, locs[:,0]-locs[c][0], locs[:,1]-locs[c][1])
    #print(type(tmp), tmp)
    phi = np.angle(tmp)
    #phi = np.angle(map(complex, locs[:,0]-locs[c][0], locs[:,1]-locs[c][1]))
    X = arclen*np.real(np.exp(1j*phi))
    Y = arclen*np.imag(np.exp(1j*phi))
    r = max([max(X),max(Y)])
    Xi = np.linspace(-r,r,n_grid)
    Yi = np.linspace(-r,r,n_grid)
    data_ip = griddata((X, Y), data, (Xi[None,:], Yi[:,None]), method='cubic')
    return data_ip


def eeg2map(data):
    """Interpolate and normalize EEG topography, ignoring nan values

    Args:
        data: numpy.array, size = number of EEG channels
        n_grid: interger, interpolate to n_grid x n_grid array, default=64
    Returns:
        top_norm: normalized topography, n_grid x n_grid
    """
    n_grid = 64
    top = topo(data, n_grid)
    mn = np.nanmin(top)
    mx = np.nanmax(top)
    top_norm = (top-mn)/(mx-mn)
    return top_norm



def locmax(x):
    """Get local maxima of 1D-array

    Args:
        x: numeric sequence
    Returns:
        m: list, 1D-indices of local maxima
    """

    dx = np.diff(x) # discrete 1st derivative
    zc = np.diff(np.sign(dx)) # zero-crossings of dx
    m = 1 + np.where(zc == -2)[0] # indices of local max.
    return m


def findstr(s, L):
    """Find string in list of strings, returns indices.

    Args:
        s: query string
        L: list of strings to search
    Returns:
        x: list of indices where s is found in L
    """

    x = [i for i, l in enumerate(L) if (l==s)]
    return x


# importing matplot lib

# importing matplot lib

display_pic=1


filename=[]
filename1 = ["data_Subject00_1_1.edf","data_Subject00_1_2.edf","data_Subject00_1_3.edf","data_Subject00_1_4.edf","data_Subject00_1_5.edf","data_Subject00_1_6.edf","data_Subject00_1_7.edf","data_Subject00_1_8.edf","data_Subject00_1_9.edf","data_Subject00_1_10.edf","data_Subject00_1_11.edf","data_Subject00_1_12.edf","data_Subject00_1_13.edf","data_Subject00_1_14.edf","data_Subject00_1_15.edf","data_Subject00_1_16.edf","data_Subject00_1_18.edf","data_Subject00_1_19.edf","data_Subject00_1_20.edf","data_Subject00_1_21.edf","data_Subject00_1_22.edf","data_Subject00_1_23.edf","data_Subject00_1_24.edf","data_Subject00_1_25.edf","data_Subject00_1_26.edf","data_Subject00_1_27.edf","data_Subject00_1_28.edf","data_Subject00_1_29.edf","data_Subject00_1_30.edf","data_Subject00_1_31.edf"]

filename.append(filename1)


import umap

mapspca1=np.array([])
L_ind=np.array([])
arr =np.array([])  
datasto =np.array([])
L_individual =np.array([],dtype='i')  
L_group=np.array([],dtype='i')  


import numpy as np
import seaborn as sns
sns.set(style='white', rc={'figure.figsize':(12,8)})
np.random.seed(42)


pca1 = []
pca1 =   []

umap1 = []
umap2 =   []

scores1=[]
scores2=[]


score1=[]
score2=[]




gfpold =   []  # microstate label sequence for each k-means run
CE_array = [] 
import numpy as np
for i1 in range(1):    
    for j1 in range(30):
        for n_maps in range(4,5):
            chs, fs, data_raw = read_edf(filename[i1][j1])
            data = bp_filter(data_raw, f_lo=1, f_hi=35, fs=fs)
            datasto=np.append(datasto,data)
            
            
            
datasto=np.resize(datasto,(30000,19))   
data=datasto


interpol=False
n_maps=1000
doplot=True

n_ch = data.shape[1]
#print("[+] EEG channels: n_ch = {:d}".format(n_ch))

# --- normalized data ---
data_norm = data - data.mean(axis=1, keepdims=True)
data_norm /= data_norm.std(axis=1, keepdims=True)

# --- GFP peaks ---
gfp = np.nanstd(data, axis=1)
gfp2 = np.sum(gfp**2) # normalizing constant
gfp_peaks = locmax(gfp)
data_cluster = data[gfp_peaks,:]
#data_cluster = data_cluster[:100,:]
data_cluster_norm = data_cluster - data_cluster.mean(axis=1, keepdims=True)
data_cluster_norm /= data_cluster_norm.std(axis=1, keepdims=True)

import numpy as np
import seaborn as sns

n_maps=4
fit = umap.UMAP(n_neighbors=2,min_dist=1, n_components=n_maps,random_state=42)
u = fit.fit_transform(data_cluster.T)
maps = u.T


pca = PCA(n_components=n_maps)
#pca.fit(data_cluster)
#maps1 = np.array([pca.components_[k,:] for k in range(n_maps)])





if interpol:
    C = np.dot(data_cluster_norm, maps.T)/n_ch
    L_gfp = np.argmax(C**2, axis=1) # microstates at GFP peak
    del C
    n_t = data_norm.shape[0]
    L = np.zeros(n_t)
    for t in range(n_t):
        if t in gfp_peaks:
            i = gfp_peaks.tolist().index(t)
            L[t] = L_gfp[i]
        else:
            i = np.argmin(np.abs(t-gfp_peaks))
            L[t] = L_gfp[i]
    L = L.astype('int')
else:
    C = np.dot(data_norm, maps.T)/n_ch
    L = np.argmax(C**2, axis=1)
    del C

# visualize microstate sequence
if False:
    t_ = np.arange(n_t)
    fig, ax = plt.subplots(2, 1, figsize=(15,8), sharex=True)
    ax[0].plot(t_, gfp, '-k', lw=2)
    for p in gfp_peaks:
        ax[0].axvline(t_[p], c='k', lw=0.5, alpha=0.3)
    ax[0].plot(t_[gfp_peaks], gfp[gfp_peaks], 'or', ms=10)
    ax[1].plot(L)
    plt.show()

''' --- temporal smoothing ---
L2 = np.copy(L)
for i in range(n_win, len(L)-n_win):
    s = np.array([np.sum(L[i-n_win:i+n_win]==j) for j in range(n_clusters)])
    L2[i] = np.argmax(s)
L = L2.copy()
del L2
'''

# --- GEV ---
maps_norm = maps - maps.mean(axis=1, keepdims=True)
maps_norm /= maps_norm.std(axis=1, keepdims=True)

# --- correlation data, maps ---
C = np.dot(data_norm, maps_norm.T)/n_ch
#print("C.shape: " + str(C.shape))
#print("C.min: {C.min():.2f}   Cmax: {C.max():.2f}")

# --- GEV_k & GEV ---
gev = np.zeros(n_maps)
for k in range(n_maps):
    r = L==k
    gev[k] = np.sum(gfp[r]**2 * C[r,k]**2)/gfp2
gev_total=sum(gev)
    
        
dis=np.array([])
import math
for i in range(data_cluster_norm.shape[0]):
   for j in range(data_cluster_norm.shape[0]):
      dis=np.append(dis,math.dist(data_cluster_norm[i],data_cluster_norm[j]))        
bhai=dis<2

gev_accomodate1=[]
n_maps=6
for n_neighbors in range(2,40,1):
   for min_dist in range(1,3,1):
    fit = umap.UMAP(n_neighbors=4,min_dist=2*0.1,n_components=n_maps,random_state=42)
    u = fit.fit_transform(data_cluster.T);
    maps=u.T
    maps_norm = maps - maps.mean(axis=1, keepdims=True)
    maps_norm /= maps_norm.std(axis=1, keepdims=True)

    # --- correlation data, maps ---
    C = np.dot(data_norm, maps_norm.T)/n_ch
    L = np.argmax(C**2, axis=1)
    #print("C.shape: " + str(C.shape))
    #print("C.min: {C.min():.2f}   Cmax: {C.max():.2f}")

    # --- GEV_k & GEV ---
    gev = np.zeros(n_maps)
    for k in range(n_maps):
        r = L==k
        gev[k] = np.sum(gfp[r]**2 * C[r,k]**2)/gfp2
    gev_total=sum(gev)
    gev_accomodate1.append(gev_total)
    
    
    
    n_clusters=n_maps
    if(display_pic):
        #plt.ion()
        # matplotlib's perceptually uniform sequential colormaps:
        # magma, inferno, plasma, viridis
        #cm = plt.cm.magma
        cm = plt.cm.seismic
        fig, axarr = plt.subplots(1, n_clusters, figsize=(20,5))
        for imap in range(n_clusters):
            axarr[imap].imshow(eeg2map(maps[imap, :]), cmap=cm, origin='lower')
            axarr[imap].set_xticks([])
            axarr[imap].set_xticklabels([])
            axarr[imap].set_yticks([])
            axarr[imap].set_yticklabels([])
        title = f"Microstate maps"
        axarr[0].set_title(title, fontsize=16, fontweight="bold")