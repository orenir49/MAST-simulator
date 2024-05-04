#!/usr/bin/env python
# coding: utf-8

#In[1]
# Import the required libraries
# astroquery for retrieving data from Gaia
from astroquery.gaia import Gaia

# astropy libraries for calculations
from astropy import units as u
from astropy import wcs
from astropy.constants import h, c
from astropy import convolution
from astropy.io import fits

# Data science libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from PIL import Image

# Manipulating & parsing files
import os
import shutil

# Parallel computation
from multiprocessing import Pool
from functools import partial

# Making the notebook interactive
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import PySimpleGUI as sg

#In[2]
# Defining the constants of the notebook
# Constants from files
qe = pd.read_csv(os.path.join('.','essential_files','ccd_data','QE.csv'), delimiter=',') # Quantum Efficiency
dark = pd.read_csv(os.path.join('.','essential_files','ccd_data','dark_current.csv'), delimiter=',') # Dark Current

# Gaia Flux Density
fd_unit = u.J/u.m**2/u.s/u.nm
d_filters = {'Band':['G'],'Wavelength':[639.07*u.nm],'Bandpass':[454.82*u.nm],'clam':[1.346e-21*fd_unit]} 
filters = pd.DataFrame(data=d_filters)

# MAST CCD constants
# Number of pixels in the CCD
# ccd_width_px = 5496 # ASI183MM   
# ccd_height_px = 3672 # ASI183MM 
ccd_width_px = 8288 # ASI294MM
ccd_height_px = 5644 # ASI294MM

# Buffer of pixels when querying data and performing the convolution
ccd_width_buff_px = 512
ccd_height_buff_px = 512

# The length of each pixel in the CCD
# ccd_px_side_length_micron = 2.4 * u.micron # ASI183MM
ccd_px_side_length_micron = 2.3 * u.micron # ASI294MM
ccd_width_micron = ccd_width_px * ccd_px_side_length_micron
ccd_height_micron = ccd_height_px * ccd_px_side_length_micron
ccd_width_buff_micron = ccd_width_buff_px * ccd_px_side_length_micron
ccd_height_buff_micron = ccd_height_buff_px * ccd_px_side_length_micron

# more MAST optical specs
mast_diameter = 0.6 * u.m
mast_collecting_area_m2 = (mast_diameter/2)**2 * np.pi * u.m**2
mast_focal_length = 1.8 * u.m
incoming_light_fov = 2*np.arctan(mast_diameter/2/mast_focal_length)

# The platescale of MAST
plate_scale_arcsec_micron = 206265 * u.arcsec / mast_focal_length.to(u.micron)

# Pixel field of view
plate_scale_arcsec_px = plate_scale_arcsec_micron * ccd_px_side_length_micron

# CCD field of view
ccd_width_range_arcsec = ccd_width_micron * plate_scale_arcsec_micron
ccd_height_range_arcsec = ccd_height_micron * plate_scale_arcsec_micron
ccd_width_buff_range_arcsec = ccd_width_buff_micron * plate_scale_arcsec_micron
ccd_height_buff_range_arcsec = ccd_height_buff_micron * plate_scale_arcsec_micron

# Converting from arcsec to degrees
ccd_width_range_deg = ccd_width_range_arcsec.to(u.deg)
ccd_height_range_deg = ccd_height_range_arcsec.to(u.deg)
ccd_width_buff_range_deg = ccd_width_buff_range_arcsec.to(u.deg)
ccd_height_buff_range_deg = ccd_height_buff_range_arcsec.to(u.deg)

# Saturation of the CCD per pixel
# ccd_full_well = 15000 * u.electron # ASI183MM
ccd_full_well = 14400 * u.electron # ASI294MM

# Arbitrary count added to each pixel before printing image
baseline_per_px = 20 * u.electron 

# Focal points (middle of field) projection on CCD "width" axis, for different modes of operation
spectra_middle_width = int(ccd_width_px/2 - 1)
# wide_middle_width = 4300 # ASI183MM
# narrow_middle_width = 803 # ASI183MM
wide_middle_width = 7400 # ASI294MM
narrow_middle_width = 600 # ASI294MM

# Gaia constants
gaia_collecting_area_m2 = 0.7 * u.m**2

# Gaia-MAST constants
atm_extinction_telescope_eff = 0.5
mag_ZP = 9.953577 # magnitude of arbitrary star I picked from the simulation
flux_ZP = 230465 # flux detected with MAST, calculated using the spectrum

# PSF properties
# r_lst = [0, 423, 847, 1271, 1695, 2119, 2543, 2967, 3391, 3815, 4239, 4663, 5087, 5511, 5935, 6358, 6782] # ASI183MM
# kernels_dirname = 'kernels_183' # ASI183MM
ang0 = 90
r_lst = [0, 467, 934, 1401, 1868, 2337, 2804, 3271, 3738, 4205, 4672, 5139, 5607, 6074, 6541, 7009, 7476, 7944, 8411, 8878] # ASI294MM
kernels_dirname = 'kernels_294' # ASI294MM

# General function to query from Gaia using ADQL
# Help can be found be found at: https://www.cosmos.esa.int/web/gaia-users/archive/writing-queries
def query_gaia(query):
    job = Gaia.launch_job(query)
    return job.get_results()

# In[5]:

# Defining the query to get sources captured by Gaia in a specific region
# Documentation about the table GAIADR3.GAIA_SOURCE can be found at: https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html

def query_gaia_for_sources_in_region(max_mag, ra_deg, dec_deg, ccd_width_range_deg, ccd_height_range_deg, op_mode):
    if(op_mode == 'Spectra'):   
        wc = int(ccd_width_buff_px/2) + spectra_middle_width
    elif(op_mode == 'Image'):
        wc = int(ccd_width_buff_px/2) + wide_middle_width
    elif(op_mode == 'Image_narrow'):
        wc = int(ccd_width_buff_px/2) + narrow_middle_width

    fraction_left_of_focus = wc/(ccd_width_buff_px + ccd_width_px) # we know: RA range the CCD covers, and reference "wc" corresponding to input ra_deg
    fraction_right_of_focus = 1 - fraction_left_of_focus # using the reference, ra_deg, and range- determine RA boundaries for the query  

    query = '''SELECT source_id, ra, dec, phot_g_mean_mag, has_xp_sampled,
    PHOT_G_MEAN_FLUX AS phot_g_mean_flux_gaia, PHOT_G_MEAN_FLUX_ERROR AS phot_g_mean_flux_error_gaia
    FROM gaiadr3.gaia_source
    WHERE (ra BETWEEN {ra_min} AND {ra_max})
    AND (dec BETWEEN {dec_min} AND {dec_max})
    AND phot_g_mean_mag < {max_mag}
    '''.format(
        max_mag = max_mag,
        ra_min = ra_deg-(fraction_left_of_focus * ccd_width_range_deg.value), ra_max=ra_deg+(fraction_right_of_focus * ccd_width_range_deg.value),
        dec_min = dec_deg-(ccd_height_range_deg.value/2), dec_max=dec_deg+(ccd_height_range_deg.value/2)
    )   
    return query_gaia(query)

# In[6]:

# Input the flux as seen by Gaia (in electrons/sec as given by GAIA_SOURCE)
# Output the flux density of the source
# The conversion is done according to the documentation: https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu5pho/cu5pho_sec_photProc/cu5pho_ssec_photCal.html#SSS3.P2

def calc_gaia_flux_density_from_gaia_flux(sources_table):
    for filter in d_filters['Band']:
        sources_table['flux_' + filter + '_density_gaia'] = filters[filters['Band'] == filter]['clam'].iloc[0] * sources_table['phot_' + str.lower(filter) + '_mean_flux_gaia'].data
        sources_table['flux_' + filter + '_density_error_gaia'] =  filters[filters['Band'] == filter]['clam'].iloc[0] * sources_table['phot_' + str.lower(filter) + '_mean_flux_error_gaia'].data
# In[7]:

# more accurate way to estimate the MAST counts for stars brighter than magnitude 15
# however, this is more time consuming
def get_flux_from_spectrum(table):
    lam0 = filters[filters.Band == 'G'].iloc[0]['Wavelength']
    bandpass = filters[filters.Band == 'G'].iloc[0]['Bandpass']
    lmin = lam0 - bandpass/2
    lmax = lam0 + bandpass/2
    source_ids = table['source_id'][table['has_xp_sampled']==True]
    datalink = Gaia.load_data(ids=source_ids, data_release = 'Gaia DR3', retrieval_type='XP_SAMPLED', data_structure = 'INDIVIDUAL', verbose = False, output_file = None)
    dl_keys  = [inp for inp in datalink.keys()]
    dl_keys.sort()
    for dl_key in dl_keys:
        t = datalink[dl_key][0]
        source_id = t.get_field_by_id('source_id').value
        t = t.to_table()
        y1arr = qe['QE183'].values[int((lmin.value-336)/2):int((lmax.value-336)/2)]
        y2arr = np.ma.getdata(t['flux'].value)[int((lmin.value-336)/2):int((lmax.value-336)/2)]
        xarr = qe['Wavelength'].values[int((lmin.value-336)/2):int((lmax.value-336)/2)]
        fmast = np.trapz(y1arr*y2arr*xarr*1e-9/h.value/c.value,xarr)*atm_extinction_telescope_eff*mast_collecting_area_m2/u.m**2
        table['phot_G_flux_mast'][table['source_id'] == source_id] = fmast
# In[8]:

# Calculate the mean quantum efficiency of Gaia, based on the wavelenghts of intrest
def calculate_effective_qe(min_wavelen, max_wavelen):
    # qe_colname = 'QE183' # ASI183MM
    qe_colname = 'QE294' # ASI294MM
    return np.mean(qe[qe['Wavelength'].isin(qe['Wavelength'].where((qe['Wavelength'] >= min_wavelen) & (qe['Wavelength'] < max_wavelen)).dropna())][qe_colname])
# In[9]:

# Given the flux density as seen at Gaia, calculate the flux as measured in MAST
def convert_gaia_flux_density_to_mast_flux(sources_table):
    for filter in d_filters['Band']:
        lam0 = filters[filters.Band == filter].iloc[0]['Wavelength']
        bandpass = filters[filters.Band == filter].iloc[0]['Bandpass']
        flux_density_inc_mast =  atm_extinction_telescope_eff * sources_table['flux_' + filter + '_density_gaia']
        flux_meas_mast = calculate_effective_qe(lam0-bandpass/2,lam0+bandpass/2)* flux_density_inc_mast * bandpass
        power_meas_mast = flux_meas_mast * mast_collecting_area_m2
        sources_table['phot_' + filter + '_flux_mast'] = (power_meas_mast * lam0.to(u.m) / (h * c)) * u.electron 

        flux_density_inc_error_mast =  atm_extinction_telescope_eff * sources_table['flux_' + filter + '_density_error_gaia']
        flux_meas_error_mast = calculate_effective_qe(lam0-bandpass/2,lam0+bandpass/2) * flux_density_inc_error_mast * bandpass
        power_meas_error_mast = flux_meas_error_mast * mast_collecting_area_m2
        sources_table['phot_' + filter + '_flux_rel_error_mast'] = np.sqrt(((power_meas_error_mast * lam0.to(u.m) / (h * c)).value)**2 + ((power_meas_mast * bandpass.to(u.m) / (2 * h * c)).value)**2) / sources_table['phot_' + filter + '_flux_mast'].data * 100 * u.percent
# In[10]:

# Deprecated!
# This function uses a toy model to differentiate images on a sub-pixel scale
# A given source doesn't fall at an exact location, so split it between the four pixels in the area. This makes sure the center of the PSF moves together with the source.
# w_px, h_px are floats of the exact location of the point-like source
# width, height are the dimensions of the simulated image
def four_px(w_px,h_px,width,height):
    lst = []
    if (w_px <0 or w_px >= width or h_px < 0 or h_px >= height):
        return lst
    wf = np.floor(w_px)
    wc = np.ceil(w_px)
    hf = np.floor(h_px)
    hc = np.ceil(h_px)
    if np.allclose(w_px,wf): 
        w_floor_res = 1
        w_ceil_res = 0
    else: 
        w_floor_res = np.round(w_px - wf,3)
        w_ceil_res = np.round(wc - w_px,3)
    if np.allclose(h_px,hf): 
        h_floor_res = 1
        h_ceil_res = 0
    else: 
        h_floor_res = np.round(h_px - hf,3)
        h_ceil_res = np.round(hc - h_px,3)    

    if (wf >= 0 and hf >= 0):
        lst.append([wf ,hf ,w_floor_res*h_floor_res])
    if (wc < width and hf >= 0):
        lst.append([wc ,hf ,w_ceil_res*h_floor_res])
    if (wf >= 0 and hc < height):
        lst.append([wf ,hc ,w_floor_res*h_ceil_res])
    if (wc < width and hc < height):
        lst.append([wc , hc ,w_ceil_res*h_ceil_res])
    return lst
# In[11]

# gets w_px, h_px (source location given by wcs)
# checks if inside image boundaries (width/height)
# if in bounds, returns rounded values (to use as array indices)

def to_px(w_px,h_px,width,height):
    wf = np.floor(w_px)
    hf = np.floor(h_px)    
    if ( 0 <= wf < width and 0 <= hf < height):
        return [[wf ,hf ,1]]
    else:
        return []


# In[12]:

# The function takes a super-resolution image, and bins the pixels
# to return an image with the 
def bin_image(image,new_shape):
    shape = (new_shape[0], image.shape[0] // new_shape[0],
            new_shape[1], image.shape[1] // new_shape[1])
    return image.reshape(shape).sum(-1).sum(1)

# In[13]

# This function takes a low-resolution kernel
# and returns a (normalized) kernel at higher resolution
def interp_kernel(ker,resolution):
    oldwidth , oldheight = ker.shape[1],ker.shape[0]
    newwidth , newheight = resolution * oldwidth, resolution * oldheight
    low_res = np.array([np.linspace(0,newheight,oldheight),np.linspace(0,newwidth,oldwidth)])
    ut, vt = np.meshgrid(np.linspace(0,newheight,newheight), np.linspace(0,newwidth,newwidth), indexing='ij')
    hi_res = np.array([ut.ravel(), vt.ravel()]).T
    rgi = RegularGridInterpolator(low_res,ker,'linear')
    ker = rgi(hi_res).reshape((newheight,newwidth))
    ker = ker/np.sum(ker)
    return ker

# In[14]

## create coma kernels convolved with seeing, at the specified resolution
## return the path to the temporary directory where the kernels are saved

def create_kernels(resolution,seeing):
    dirpath = os.path.join('.','essential_files',kernels_dirname)
    tmpdirpath = os.path.join('.','essential_files','kernels_tmp')
    os.mkdir(tmpdirpath)
    filenames = next(os.walk(dirpath), (None, None, []))[2]
    for f in filenames: # go through all the coma kernels
        kerpath = os.path.join(dirpath,f)    
        ker = np.load(kerpath)
        if resolution > 1: # if the resolution is higher than 1, interpolate the kernel to the higher resolution
            ker = interp_kernel(ker,resolution)
        seeing_px = seeing / (plate_scale_arcsec_px) * resolution
        sigma = seeing_px/2.355 # fhwm = 2*sqrt(2*ln(2))*sigma
        ker2 = convolution.Gaussian2DKernel(x_stddev = sigma , y_stddev = sigma)
        ker = np.pad(ker,int(ker2.shape[0]/2))
        ker = convolution.convolve(ker,ker2,normalize_kernel=True) # convolve coma with seeing to create the final kernel
        np.save(os.path.join(tmpdirpath,f),ker) # save the kernel
    return tmpdirpath

# In[15]:

# This function is auxiliary to 'psf_rms'
# finds the center (first moment) of a kernel
def psf_center(ker):
    numerator_w = 0
    numerator_h = 0
    denominator = np.sum(ker)
    for h in range(ker.shape[0]):
        for w in range(ker.shape[1]):
            numerator_w += ker[h,w] * w
            numerator_h += ker[h,w] * h
    return numerator_h/denominator , numerator_w/denominator
# In[16]:

# This function finds the RMS width of a PSF (centralized second moment)
# This is an okay approximation to what Zemax gives 
# I would love suggestions for more accurate methods.
def psf_rms(ker):
    center_h , center_w = psf_center(ker)
    numerator = 0
    denominator = np.sum(ker)
    for h in range(ker.shape[0]):
        for w in range(ker.shape[1]):
            numerator += ker[h,w] * ((h-center_h)**2 + (w-center_w)**2)
    return np.sqrt(numerator/denominator)
# In[17]:

# aligns an image by angle 'ang' from the x-axis, counter-clockwise 
# used to rotate the kernels, which are generated with an initial orientation 'ang0'
def rot(psf,ang,ang0):
    im = Image.fromarray(psf)
    im = im.rotate(ang0-ang)
    im = np.array(im)
    im = im/np.sum(im)
    return im
# In[18]:

# for a source located at column w (px), and row h (px)
# adds the PSF of the source (Coma from mirror + Gaussian from seeing)
def add_PSF(image, w, h, seeing_arcsec, op_mode, resolution, kernel_dir): 
    
    # the center of the FoV is not necessarily in the center of the image
    if(op_mode == 'Spectra'):   
        wc = int(ccd_width_buff_px/2) + spectra_middle_width
    elif(op_mode == 'Image'):
        wc = int(ccd_width_buff_px/2) + wide_middle_width
    elif(op_mode == 'Image_narrow'):
        wc = int(ccd_width_buff_px/2) + narrow_middle_width
    hc = int(ccd_height_buff_px/2) + int(ccd_height_px/2) - 1
        
    wc *= resolution 
    hc *= resolution 

    dis_from_center = np.sqrt((w-wc)**2 + (h-hc)**2) / resolution # distance from center in px- used to choose the right kernel
    ang = np.rad2deg(np.arctan2(h-hc,w-wc)) # angle from center in degrees- used to rotate the kernel
    r_ker = r_lst[np.argmin(np.abs(r_lst - dis_from_center))]
    ker_name = os.path.join(kernel_dir, str.format(r'r={}.npy',r_ker)) 
    ker = np.load(ker_name) # load the appropriate kernel
    ker = rot(ker,ang,ang0)
    
    amplitude = image[h,w] # the flux of the source is saved in the image
    image[h,w] = 0 
    patch = amplitude * ker # multiply the normalized kernel by the total flux of the source

    w_start = w - int(patch.shape[1]/2)
    h_start = h - int(patch.shape[0]/2)
    w_finish = w_start + patch.shape[1]
    h_finish = h_start + patch.shape[0]
    image[h_start:h_finish,w_start:w_finish] += patch # add the PSF to the image
    return image

# In[19]:

# Add the sources to the base image
# image - Matrix for the image
# sources_table - Sources found by Gaia in that region
# wcs_dict - astropy.wcs object to convert from coordinates to pixels
# t_exp - Image exposure time

def add_sources(image, table, wcs_dict, t_exp, seeing_arcsec, op_mode, resolution, kernel_dir):
    pixs = wcs_dict.wcs_world2pix(table['ra'].data, table['dec'].data, 0) # convert from sky coordinates to pixels
    flux = np.round(table['phot_G_flux_mast'].data) # flux in electrons
    brightness = np.random.poisson(lam = flux) * t_exp / u.s # randomly assign brightness from Poisson distribution 
    for i in range(len(pixs[0])):
        lst = to_px(pixs[0][i],pixs[1][i],image.shape[1],image.shape[0]) 
        for ls in lst:
            image[int(ls[1]),int(ls[0])] += ls[2] * brightness[i] # add the source flux to the corresponding pixel
            lim_lft,lim_rgt = ccd_width_buff_px/2 * resolution, (ccd_width_px + ccd_width_buff_px/2) * resolution 
            lim_down,lim_up = ccd_height_buff_px/2 * resolution, (ccd_height_px + ccd_height_buff_px/2) * resolution
            if(lim_lft < int(ls[0]) < lim_rgt and lim_down < int(ls[1]) < lim_up): # if the source is inside the FoV, add the PSF
                add_PSF(image, int(ls[0]), int(ls[1]), seeing_arcsec, op_mode,resolution, kernel_dir)
    return image

# In[20]:

# Deprecated!
# Convolve the image with a Gaussian atmospheric seeing model
def add_seeing(image, seeing_arcsec):
    seeing_px = seeing_arcsec / plate_scale_arcsec_px
    sigma = seeing_px/2.355 # fhwm = 2*sqrt(2*ln(2))*sigma
    ker = convolution.Gaussian2DKernel(x_stddev = sigma , y_stddev = sigma)
    return convolution.convolve(image,ker)
# In[21]:

# Add background noise to the image (randomly for each pixel)
# depends on the mean background level and exposure time (user inputs)
def add_bgd_noise(image, mean_bgd_mag_to_arcsec_squared, t_exp):
    rate = flux_ZP * (u.electron/u.s/u.arcsec**2) * 10**(-0.4*(mean_bgd_mag_to_arcsec_squared-mag_ZP))*(ccd_px_side_length_micron*plate_scale_arcsec_micron)**2
    mean = (rate * t_exp).value
    image += np.random.poisson(lam = mean, size = (image.shape[0],image.shape[1])) # for now, brightness = number of photo electrons 
    return image
# In[22]:

# Add the dark noise to the image (randomly for each pixel)
# depends on the temperature, readout noise rms, and exposure time (all user inputs)
def add_read_dark_noise(image, temperature, read_rms, t_exp):
    dark_rate = dark[dark.temp == temperature].iloc[0]['e/s/pix'] * u.electron/u.s
    mean = (dark_rate * t_exp).value
    var = (dark_rate * t_exp).value + read_rms.value**2
    image += np.random.normal(loc = mean, scale = np.sqrt(var), size = (image.shape[0],image.shape[1]))
    return image
# In[23]:

# Given positions on the sensor 'w_arr' (the columns of interest in units of px)
# Return a number between 0-1 to indicate how much light reaches that column (a fraction of the photons are obscured by FIFFA)
def fiffa_shadow_at_col(w_arr):     
    a = np.fromfile(os.path.join('.','essential_files','fiffa_data','MAST_BFD17_ASI1600_GuideMode.dat'),sep=' ') # Zemax transmission vs. field position
    y = np.fromfile(os.path.join('.','essential_files','fiffa_data','MAST_BFD17_ASI1600_YCord.dat'),sep=' ') # Corresponding positions on "width" axis (in degrees)
    w_vec0 = np.unique(y)
    eta_vec0 = [np.mean(a[y == w0]) for w0 in w_vec0] # The transmission is almost independent of "height" axis: values almost uniform for constant "width" pos
    w_vec_deg = (w_arr - int(ccd_width_px/2 - 1))* plate_scale_arcsec_px.value / 3600 
    cs = CubicSpline(w_vec0,eta_vec0)
    eta_vec = cs(w_vec_deg) / 100 # Zemax gives transmission in range 0-100; I use range 0-1
    eta_vec[eta_vec > 1] = 1
    eta_vec[eta_vec < 0] = 0
    return np.flip(eta_vec)
# In[24]:

# accounts for shadow of the fiffa folding mirror
# attenuates the count in every column appropriately, removing light that is blocked by FIFFA 
def add_fiffa_shadow(image):
    columns = np.arange(0,ccd_width_px,1)
    fiffa_shadow = np.diag(fiffa_shadow_at_col(columns))
    image = image @ fiffa_shadow
    return image
# In[25]:

# estimates PSF spot size at different positions on the CCD, due to coma + seeing 
def get_psf_rms_after_seeing(field_arr_px, wc ,seeing_arcsec = 1.5*u.arcsec):
    rms_lst = []
    max_lst = []
    for r in r_lst:
        ker = np.load(os.path.join('.','essential_files',kernels_dirname, str.format(r'r={}.npy',r)))
        sigma = seeing_arcsec/plate_scale_arcsec_px/2.355 # fhwm = 2*sqrt(2*ln(2))*sigma
        gaus_ker = convolution.Gaussian2DKernel(sigma,sigma)
        ker = convolution.convolve(ker,gaus_ker)
        rms_lst.append(psf_rms(ker))
        max_lst.append(np.max(ker))
    cs_rms = CubicSpline(r_lst,rms_lst)
    cs_max = CubicSpline(r_lst,max_lst)
    rms_lst = cs_rms(np.abs(field_arr_px - wc))
    max_lst = cs_max(np.abs(field_arr_px - wc))
    return rms_lst , max_lst
# In[26]:

# Create the final sources table from Gaia (from querying Gaia to getting the correct measurment for MAST)
def create_sources_table(max_mag, ra_center, dec_center, op_mode ,use_available_spectra = False):
    coor_width_range_deg = (ccd_width_range_deg + ccd_width_buff_range_deg) / np.cos(np.deg2rad(dec_center))
    coor_height_range_deg = ccd_height_range_deg + ccd_height_buff_range_deg
    entries = query_gaia_for_sources_in_region(max_mag, ra_center, dec_center, coor_width_range_deg, coor_height_range_deg, op_mode)
    j = 0
    while len(entries) == 2000: ## synchronous queries are limited to 2000 rows, if hit limit: image probably not complete.
        max_mag -= 0.3**j
        j += 1
        entries = query_gaia_for_sources_in_region(max_mag, ra_center, dec_center, coor_width_range_deg, coor_height_range_deg, op_mode)
    calc_gaia_flux_density_from_gaia_flux(entries)
    convert_gaia_flux_density_to_mast_flux(entries)
    if(use_available_spectra):
        get_flux_from_spectrum(entries)
    return entries

# In[27]:
def create_wcs(ra_center, dec_center, op_mode, resolution):
    if(op_mode == 'Spectra'):   
        axis_1_focus = int(ccd_width_buff_px/2) + spectra_middle_width
    elif(op_mode == 'Image'):
        axis_1_focus = int(ccd_width_buff_px/2) + wide_middle_width
    elif(op_mode == 'Image_narrow'):
        axis_1_focus = int(ccd_width_buff_px/2) + narrow_middle_width
    axis_2_focus = int(ccd_height_buff_px/2) + int(ccd_height_px/2) - 1
        
    axis_1_focus *= resolution
    axis_2_focus *= resolution

    px_scale = (plate_scale_arcsec_px.to(u.deg)).value / resolution
    ncols = (ccd_width_px+ccd_width_buff_px) * resolution
    nrows = (ccd_height_px+ccd_height_buff_px) * resolution

    wcs_input_dict = {
    'CTYPE1': 'RA---TAN', 
    'CUNIT1': 'deg', 
    'CDELT1': px_scale, 
    'CRPIX1': axis_1_focus, 
    'CRVAL1': ra_center, 
    'NAXIS1': ncols,
    'CTYPE2': 'DEC--TAN', 
    'CUNIT2': 'deg', 
    'CDELT2': px_scale, 
    'CRPIX2': axis_2_focus, 
    'CRVAL2': dec_center, 
    'NAXIS2': nrows
    }
    return wcs.WCS(wcs_input_dict)
# In[28]

# Deprecated! 
# parallelize 'add_sources' to save time
def add_sources_parallel(image, table, wcs_dict, t_exp, seeing_arcsec, op_mode, resolution):
    add = partial(add_sources, image, wcs_dict=wcs_dict, t_exp=t_exp, seeing_arcsec=seeing_arcsec, op_mode=op_mode, resolution=resolution)
    n = os.cpu_count()
    start_idx = (np.arange(0,n) * len(table) / n).astype(int)
    finish_idx = np.concatenate([start_idx[1:],[len(table)]])
    partial_tbl_lst = []
    for s,f in zip(start_idx,finish_idx):
        partial_tbl_lst.append(table[s:f])
    
    with Pool(n) as p:
        image_layers = p.map(add,partial_tbl_lst)
    
    return np.sum(image_layers,axis=0)
# In[29]:

# Create the simulated image from MAST based on the sources table prepered previously
def create_image(sources_table, ra_center , dec_center, t_exp, bdg_mean_mag_to_arcsec_squared, seeing_arcsec, temperature, read_rms ,op_mode, resolution = 1 ,kernel_dir = None):
    wcs_dict = create_wcs(ra_center, dec_center, op_mode, resolution)
    nrows = resolution * (ccd_height_px + ccd_height_buff_px)
    ncols = resolution * (ccd_width_px + ccd_width_buff_px)
    image = np.zeros((nrows, ncols)) # create empty image with the correct dimensions
    delete_kernels = False
    if kernel_dir is None: # if no kernels were given, create them- and delete after use
        kernel_dir = create_kernels(resolution,seeing_arcsec)
        delete_kernels = True
    image = add_sources(image, sources_table, wcs_dict, t_exp, seeing_arcsec, op_mode, resolution, kernel_dir) # add the PSF of each source in the FoV
    if delete_kernels: # delete the kernels if they were created in this function
        shutil.rmtree(kernel_dir)
    if resolution > 1: # bin the image back to the correct size (if needed)
        image = bin_image(image,(int(nrows/resolution),int(ncols/resolution)))
    image = image[int(ccd_height_buff_px/2):int(ccd_height_px+ccd_height_buff_px/2),int(ccd_width_buff_px/2):int(ccd_width_px+ccd_width_buff_px/2)] # remove the buffer
    image = add_bgd_noise(image, bdg_mean_mag_to_arcsec_squared, t_exp) 
    image = add_fiffa_shadow(image) 
    image = add_read_dark_noise(image,temperature,read_rms,t_exp) 
    image += baseline_per_px.value 
    return image

# In[30]

# Class for making a figure interactive
class Toolbar(NavigationToolbar2Tk):
    def __init__(self, *args, **kwargs):
        super(Toolbar, self).__init__(*args, **kwargs)

# Plot a regular figure in some window
def draw_figure(canvas, fig):
    if canvas.children:
        for child in canvas.winfo_children():
            child.destroy()
    figure_canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='right', fill='both', expand=1)    

# Plot an interactive figure in some window
def draw_figure_w_toolbar(canvas, fig, canvas_toolbar):
    if canvas.children:
        for child in canvas.winfo_children():
            child.destroy()
    if canvas_toolbar.children:
        for child in canvas_toolbar.winfo_children():
            child.destroy()
    figure_canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
    figure_canvas_agg.draw()
    toolbar = Toolbar(figure_canvas_agg, canvas_toolbar)
    toolbar.update()
    figure_canvas_agg.get_tk_widget().pack(side='right', fill='both', expand=1)

# In[31]

# Plot the simulated image as interactive figure in new window
def create_image_figure(image):
    fig,ax = plt.subplots(figsize=(9,7), dpi=120)
    mappable = ax.imshow(image, origin='lower', cmap='Greys_r', norm=colors.LogNorm())
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.7, pad = 0.05, format='%1.1E')
    cbar.set_label('Electron Count', labelpad=10, rotation = 270, fontdict={'size':12})
    fig.tight_layout()
    return fig

def save_as_fits(image,params,filename):
    hd_dict = [('NAXIS',2),('NAXIS1',image.shape[1]),('NAXIS2',image.shape[0]),('RA',params['ra'],'degrees'),('DEC',params['dec'],'degrees'),('t_exp',params['t_exp'],'sec'),
                        ('bgdmagG',params['bgd'],'mag per arcsec squared'),
                        ('Readout',params['read_rms'],'electrons per pix'),('Seeing',params['seeing'],'arcsec')]
    hdu=fits.PrimaryHDU(data=image.astype(np.float32))
    for hd in hd_dict:
        hdu.header.append(hd)
    hdu.writeto(filename,overwrite=True)

def interactive_image_window(image, params):
    fig = create_image_figure(image)
    canvas_size = (fig.get_dpi()*fig.get_size_inches()[0],fig.get_dpi()*fig.get_size_inches()[1])
    layout = [[sg.T('Controls:')],[sg.Canvas(key='controls_cv')],
    [sg.Column(layout=[[sg.Canvas(key='fig_cv',size=canvas_size)]],background_color='#DAE0E6')],
    [sg.FolderBrowse(button_text='choose output folder',key='folder'),sg.InputText(default_text='file_name',key='name',size=(20,5)),sg.B('Save as FITS'),sg.B('Exit')]]
    
    window = sg.Window('Image', layout,finalize = True, resizable = True,element_justification='c')
    window.move_to_center()
    draw_figure_w_toolbar(window['fig_cv'].TKCanvas, fig, window['controls_cv'].TKCanvas)

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):  # always,  always give a way out!
            break
        if event == 'Save as FITS':
            if os.path.exists(values['folder']) and len(values['name']) > 0:
                filename = os.path.normpath(os.path.join(values['folder'],values['name']))+'.fits'
                save_as_fits(image, params, filename)
                sg.popup_ok('Saved successfully')
            else:
                sg.popup_error('Invalid folder or name!')
    window.close()
# In[32]

# Plot the SNR vs the position on the sensor, for magnitude of interest- in a new window
# As a bonus, the function plots the spot size (in microns) vs the field
def create_snr_figure(magnitudes, read_rms, t_exp = 1*u.s, seeing_arcsec = 1.5*u.arcsec ,mag_arcsec_squared = 20.5 , temp = -15, op_mode = 'Spectra'):
    if(op_mode == 'Spectra'):   
        wc = spectra_middle_width
    elif(op_mode == 'Image'):
        wc = wide_middle_width
    elif(op_mode == 'Image_narrow'):
        wc = narrow_middle_width
    field_arr_px = []
    for r in r_lst:
        if(wc - r > 0):
            field_arr_px.append(wc-r)
        if(wc + r < ccd_width_px):
            field_arr_px.append(wc+r)
    field_arr_px = np.sort(field_arr_px)
    field_arr_deg = (field_arr_px - wc) * plate_scale_arcsec_px.to(u.deg).value
    
    N_source = flux_ZP * 10**(-0.4*(magnitudes - mag_ZP)) * t_exp.value # rough estimate of electron flux from source of known magnitude
    rms_arr_after_seeing_px , psf_max_vs_px = get_psf_rms_after_seeing(field_arr_px, wc , seeing_arcsec)
    fiffa_shadow = fiffa_shadow_at_col(field_arr_px)
    n_pix = 1 + (rms_arr_after_seeing_px)**2 # rough estimate of total pixel number in the PSF
    N_bgd = flux_ZP * 10**(-0.4*(mag_arcsec_squared - mag_ZP)) * (plate_scale_arcsec_px.value)**2 * t_exp.value
    dark_rate = dark[dark.temp == temp].iloc[0]['e/s/pix'] * u.electron/u.s
    N_dark = (dark_rate * t_exp).value 
    N_read = read_rms.value
    SNR = np.zeros((len(N_source),len(field_arr_px)))

    fig,(ax1,ax2) = plt.subplots(ncols=2,dpi=120)
    ax1.set_xlabel('CCD column (px)', fontdict={'size': 12})
    ax1.set_ylabel('SNR', fontdict={'size': 12})
    for i,ns in enumerate(N_source):
        ns_detected = ns * fiffa_shadow
        SNR[i] = ns_detected/np.sqrt(ns_detected + n_pix*(N_bgd*fiffa_shadow + N_dark + N_read**2))
        ax1.plot(field_arr_px,SNR[i],label=f'mag={magnitudes[i]}')
        ns_peak = np.multiply(ns_detected,psf_max_vs_px)
        for j,peak in enumerate(ns_peak):
            if (peak >= ccd_full_well.value):
                ax1.scatter(field_arr_px[j],SNR[i][j],marker='x',c='Crimson',label='Saturation')

    handles, labels = ax1.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax1.legend(*zip(*unique),loc=[0.55,0.05])
    ax1.grid()
    ax2.set_xlabel('CCD column (px)', fontdict={'size':12})
    ax2.set_ylabel('Spot size ($\mu m$)',fontdict={'size':12})
    ax2.scatter(field_arr_px, rms_arr_after_seeing_px * ccd_px_side_length_micron, s = 10, c = 'Crimson')
    ax2.grid()
    fig.tight_layout()
    return fig, field_arr_px, SNR

def snr_window(magnitudes, read_rms, t_exp = 1*u.s, seeing_arcsec = 1.5*u.arcsec ,mag_arcsec_squared = 20.5 , temp = -15, op_mode = 'Spectra'):
    fig , field_arr_px, SNR = create_snr_figure(magnitudes, read_rms, t_exp, seeing_arcsec, mag_arcsec_squared, temp, op_mode)
    params_str = '|'.join((
    r't_exp=%d (sec) ' % (t_exp.value, ),
    r' background=%.1f (mag/arcsec**2) ' % (mag_arcsec_squared, ),
    r' seeing=%.2f" (arcsec) ' % (seeing_arcsec.value, ),
    r' read out rms=%.1f electrons/px ' % (read_rms.value, )))

    canvas_size = (fig.get_dpi()*fig.get_size_inches()[0],fig.get_dpi()*fig.get_size_inches()[1])
    layout = [[sg.Column(layout=[[sg.Canvas(key='fig_cv',size=canvas_size)]],background_color='#DAE0E6')],[sg.T(params_str)],
    [sg.FolderBrowse(button_text='choose output folder',key='folder'),sg.InputText(default_text='file_name',key='name'),
     sg.B('Save SNR plot as txt'),sg.B('Exit')]]

    window = sg.Window('Data', layout,finalize = True, resizable = True,element_justification='c')
    window.move_to_center()
    draw_figure(window['fig_cv'].TKCanvas, fig)
    
    params_str = params_str + '\n left column: Column (px) , right column: SNR'
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):  # always,  always give a way out!
            break
        if event == 'Save SNR plot as txt':
            if os.path.exists(values['folder']) and len(values['name']) > 0:
                for i,mag in enumerate(magnitudes):
                    df = pd.DataFrame({'Column_(px)':field_arr_px,'SNR':SNR[i]})
                    np.savetxt(os.path.normpath(os.path.join(values['folder'] , values['name'] + f'_G_mag{mag}')) +'.txt', df.values , header= params_str)
                sg.popup_ok('Saved successfully')
            else:
                sg.popup_error('Invalid folder or name!')

    window.close()
# In[33]:

# Take user inputs, and turn them into an image
def simulate_image(params):
    sources_table = create_sources_table(float(params['max_mag']), float(params['ra']), float(params['dec']) , params['op_mode'])
    image = create_image(sources_table, float(params['ra']), float(params['dec']), float(params['t_exp'])*u.s,
                          float(params['bgd']), float(params['seeing']) * u.arcsec, int(params['temp']),
                          float(params['read_rms'])*u.electron, params['op_mode'], params['resolution'],params['kernel_dir'])
    return image

# In[32]
def main_window():
    layout_main = [[sg.T('RA (deg)'),sg.InputText(default_text = 279.36 ,key='ra',size=(10,2)),
            sg.T('DEC (deg)'),sg.InputText(default_text = 38.79, key='dec',size=(10,2)),
            sg.T('Max G magnitude'),sg.InputText(default_text = 16 ,key='max_mag',size=(5,2)),
            sg.T('Exposure (sec)'), sg.InputText(default_text = 1, key = 't_exp',size=(5,2)),sg.T('Seeing FWHM (arcsec)'),
            sg.InputText(default_text = 1.5 ,key='seeing',size=(5,2))],[sg.T('Background (G-mag/arcsec**2)'),
                sg.InputText(default_text = 20.5, key = 'bgd',size=(5,2)), sg.T('Read out RMS (e/px)'),
                sg.InputText(default_text = 3.0 ,key='read_rms',size=(5,2)),sg.T('Temperature (C)'),sg.DropDown(values=list(dark['temp'].values),
                default_value=dark['temp'][1],key='temp',size=(5,4)),sg.T('Resolution (binning)'),sg.DropDown(values=[1,2,3,4,5],default_value=1,key='resolution',size=(5,4))]
                ,[sg.T('G-band magnitudes for SNR plot'),sg.Listbox(values = [10,11,12,13,14,15,16,17,18], default_values = [15,16],select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE,key='mag_lst',size=(10,3)),
                sg.T('Mode of operation'),sg.DropDown(values=['Spectra','Image','Image_narrow'],default_value='Spectra',key='op_mode'),
                sg.B(button_text='Simulate Image'),sg.B(button_text='Plot SNR vs. CCD column')]]
    sg.set_options(font=("Arial", 13))
    main_wind = sg.Window('Imaging parameters',layout_main,element_justification='c',finalize=True)
    main_wind.move_to_center()
    
    while True:
        event,params = main_wind.read()
        if event == sg.WIN_CLOSED:
            break
        elif event == 'Simulate Image':
            image = simulate_image(params)
            interactive_image_window(image,params)
        elif event == 'Plot SNR vs. CCD column':
            snr_window(np.array(params['mag_lst']),float(params['read_rms']) * u.electron,float(params['t_exp'])* u.s,
                                float(params['seeing']) * u.arcsec, float(params['bgd']), int(params['temp']),params['op_mode'])

    main_wind.close()

if __name__=='__main__':
    main_window()
        