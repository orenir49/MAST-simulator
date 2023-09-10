Can run the program without any prerequisites, on Windows-
please contact me to get the file/user manual.

With Python, can also run the simulation on Python. Download to your workspace:
* simulate.py (python script)
* essential_files (folder)
Basic instructions:
1. Verify dependencies with 'essential_files/requirements.txt'
2. Open a Python notebook and import 'simulate.py' (alternatively use the command line)
3. Create a dictionary of parameters, i.e:
    params = {'max_mag':maxmag,  # faintest magnitude to query for the image. I recommend starting with 15, going up only if necessary.
          'ra':ra, 'dec': dec, # coordinates of the target (ICRS)
          'op_mode':'Feed', # 'Feed'- image of the target obscured by FIFFA, 'Shifted_wide'- target acquired on the detector.
          't_exp':texp, # exposure time in seconds
          'bgd':bgd, # background level in G band magnitude per arcsecond squared- start with ~20.
          'seeing':seeing, # seeing FWHM in arcsec- start with ~1.5-2.0. Code will run slower with increasing seeing. 
          'temp':-15, # temperature of the CCD in celsius- controls dark noise. Pretty negligible, no need to change.
          'read_rms':3.0, # read noise in electrons per pixel. For the current detector, between 1.6-3.0 depending on gain.
          'resolution':resolution # more advanced setting- for basic applications, keep it to be 1. Higher values will slow down code.
          }
4. Pass the parameters to the 'simulate_image' function (save the result- an array):
     image = simulate.simulate_image(params)
5. Save the image in a directory of your choice:
    filepath = path.join('.','images',dirname,filename) + '.fits'
    simulate.save_as_fits(image,params,filepath)
