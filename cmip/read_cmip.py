#Read in hybrid-height coordinate data from the NCI CMIP archive, for computation in wrf_parallel.py

import metpy.units as units
import metpy.calc as mpcalc
import sys
import xarray as xr
import netCDF4 as nc
import numpy as np
import datetime as dt
import glob
import pandas as pd
from barra_read import date_seq
from calc_param import get_dp

def read_cmip(model, experiment, ensemble, year, domain, cmip_ver=5, group = "", al33=False, ver6hr="", ver3hr="", project="CMIP"):

	if cmip_ver == 5:

		#Get CMIP5 file paths

		if al33:
			#NOTE unknown behaviour for al33 directories with multiple versions

			if group == "":
				raise ValueError("Group required")
			if ver6hr == "":
				ver6hr = "v*"
			if ver3hr == "":
				ver3hr = "v*"

			hus_files = np.sort(glob.glob("/g/data/al33/replicas/CMIP5/combined/"+\
				    group+"/"+model+"/"+experiment+\
				    "/6hr/atmos/6hrLev/"+ensemble+"/"+ver6hr+"/hus/*6hrLev*"))
			ta_files = np.sort(glob.glob("/g/data/al33/replicas/CMIP5/combined/"+\
				    group+"/"+model+"/"+experiment+\
				    "/6hr/atmos/6hrLev/"+ensemble+"/"+ver6hr+"/ta/*6hrLev*"))
			ua_files = np.sort(glob.glob("/g/data/al33/replicas/CMIP5/combined/"+\
				    group+"/"+model+"/"+experiment+\
				    "/6hr/atmos/6hrLev/"+ensemble+"/"+ver6hr+"/ua/*6hrLev*"))
			va_files = np.sort(glob.glob("/g/data/al33/replicas/CMIP5/combined/"+\
				    group+"/"+model+"/"+experiment+\
				    "/6hr/atmos/6hrLev/"+ensemble+"/"+ver6hr+"/va/*6hrLev*"))

			huss_files = np.sort(glob.glob("/g/data/al33/replicas/CMIP5/combined/"+\
				    group+"/"+model+"/"+experiment+\
				    "/3hr/atmos/3hr/"+ensemble+"/"+ver3hr+"/huss/*"))
			tas_files = np.sort(glob.glob("/g/data/al33/replicas/CMIP5/combined/"+\
				    group+"/"+model+"/"+experiment+\
				    "/3hr/atmos/3hr/"+ensemble+"/"+ver3hr+"/tas/*"))
			uas_files = np.sort(glob.glob("/g/data/al33/replicas/CMIP5/combined/"+\
				    group+"/"+model+"/"+experiment+\
				    "/3hr/atmos/3hr/"+ensemble+"/"+ver3hr+"/uas/*"))
			vas_files = np.sort(glob.glob("/g/data/al33/replicas/CMIP5/combined/"+\
				    group+"/"+model+"/"+experiment+\
				    "/3hr/atmos/3hr/"+ensemble+"/"+ver3hr+"/vas/*"))
			ps_files = np.sort(glob.glob("/g/data/al33/replicas/CMIP5/combined/"+\
				    group+"/"+model+"/"+experiment+\
				    "/3hr/atmos/3hr/"+ensemble+"/"+ver3hr+"/ps/*"))
			pr_files = np.sort(glob.glob("/g/data/al33/replicas/CMIP5/combined/"+\
				    group+"/"+model+"/"+experiment+\
				    "/3hr/atmos/3hr/"+ensemble+"/"+ver3hr+"/pr/*"))

		else:

			hus_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP5/"+\
				    model+"/"+experiment+\
				    "/6hr/atmos/"+ensemble+"/hus/latest/*6hrLev*"))
			ta_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP5/"+\
				    model+"/"+experiment+\
				    "/6hr/atmos/"+ensemble+"/ta/latest/*6hrLev*"))
			ua_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP5/"+\
				    model+"/"+experiment+\
				    "/6hr/atmos/"+ensemble+"/ua/latest/*6hrLev*"))
			va_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP5/"+\
				    model+"/"+experiment+\
				    "/6hr/atmos/"+ensemble+"/va/latest/*6hrLev*"))

			huss_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP5/"+\
				    model+"/"+experiment+\
				    "/3hr/atmos/"+ensemble+"/huss/latest/*"))
			tas_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP5/"+\
				    model+"/"+experiment+\
				    "/3hr/atmos/"+ensemble+"/tas/latest/*"))
			uas_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP5/"+\
				    model+"/"+experiment+\
				    "/3hr/atmos/"+ensemble+"/uas/latest/*"))
			vas_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP5/"+\
				    model+"/"+experiment+\
				    "/3hr/atmos/"+ensemble+"/vas/latest/*"))
			ps_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP5/"+\
				    model+"/"+experiment+\
				    "/3hr/atmos/"+ensemble+"/ps/latest/*"))
			pr_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP5/"+\
				    model+"/"+experiment+\
				    "/3hr/atmos/"+ensemble+"/pr/latest/*"))
	elif cmip_ver == 6:

		#Get CMIP6 file paths

		hus_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP6/"+project+"/"+group+"/"+model+\
			"/"+experiment+"/"+ensemble+"/6hrLev/hus/gn/latest/*"))
		ta_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP6/"+project+"/"+group+"/"+model+\
			"/"+experiment+"/"+ensemble+"/6hrLev/ta/gn/latest/*"))
		ua_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP6/"+project+"/"+group+"/"+model+\
			"/"+experiment+"/"+ensemble+"/6hrLev/ua/gn/latest/*"))
		va_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP6/"+project+"/"+group+"/"+model+\
			"/"+experiment+"/"+ensemble+"/6hrLev/va/gn/latest/*"))


		huss_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP6/"+project+"/"+group+"/"+model+\
			"/"+experiment+"/"+ensemble+"/3hr/huss/gn/latest/*"))
		tas_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP6/"+project+"/"+group+"/"+model+\
			"/"+experiment+"/"+ensemble+"/3hr/tas/gn/latest/*"))
		uas_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP6/"+project+"/"+group+"/"+model+\
			"/"+experiment+"/"+ensemble+"/3hr/uas/gn/latest/*"))
		vas_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP6/"+project+"/"+group+"/"+model+\
			"/"+experiment+"/"+ensemble+"/3hr/vas/gn/latest/*"))
		ps_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP6/"+project+"/"+group+"/"+model+\
			"/"+experiment+"/"+ensemble+"/3hr/ps/gn/latest/*"))

	#Isolate the files relevant for the current "year"
	#NOTE will have to change to incorperate months if there is more than one file per year

	hus_fid = get_fid( hus_files, year)
	ta_fid = get_fid( ta_files, year)
	ua_fid = get_fid( ua_files, year)
	va_fid = get_fid( va_files, year)
	huss_fid = get_fid( huss_files, year)
	tas_fid = get_fid( tas_files, year)
	uas_fid = get_fid( uas_files, year)
	vas_fid = get_fid( vas_files, year)
	ps_fid = get_fid( ps_files, year)
	pr_fid = get_fid( pr_files, year)

	#Load the data
	hus = xr.open_mfdataset([hus_files[i] for i in hus_fid], use_cftime=True)
	ta = xr.open_mfdataset([ta_files[i] for i in ta_fid], use_cftime=True)
	ua = xr.open_mfdataset([ua_files[i] for i in ua_fid], use_cftime=True)
	va = xr.open_mfdataset([va_files[i] for i in va_fid], use_cftime=True)

	#Load surface data and match to 6 hourly
	huss = xr.open_mfdataset([huss_files[i] for i in huss_fid], use_cftime=True)
	tas = xr.open_mfdataset([tas_files[i] for i in tas_fid], use_cftime=True)
	uas = xr.open_mfdataset([uas_files[i] for i in uas_fid], use_cftime=True)
	vas = xr.open_mfdataset([vas_files[i] for i in vas_fid], use_cftime=True)
	ps = xr.open_mfdataset([ps_files[i] for i in ps_fid], use_cftime=True)
	try:
		pr = xr.open_mfdataset([pr_files[i] for i in pr_fid], use_cftime=True).interp({"time":ps.time}, method="linear")
	except:
		pass

	#If lon ranges from 0-360, reassign coordinates to be in the range -180-180
	if hus.lon.values.max() >= 350:
		hus.coords['lon'] = (hus.coords['lon'] + 180) % 360 - 180; hus = hus.sortby(hus.lon)
		ta.coords['lon'] = (ta.coords['lon'] + 180) % 360 - 180; ta = ta.sortby(ta.lon)
		ua.coords['lon'] = (ua.coords['lon'] + 180) % 360 - 180; ua = ua.sortby(ua.lon)
		va.coords['lon'] = (va.coords['lon'] + 180) % 360 - 180; va = va.sortby(va.lon)
		huss.coords['lon'] = (huss.coords['lon'] + 180) % 360 - 180; huss = huss.sortby(huss.lon)
		tas.coords['lon'] = (tas.coords['lon'] + 180) % 360 - 180; tas = tas.sortby(tas.lon)
		uas.coords['lon'] = (uas.coords['lon'] + 180) % 360 - 180; uas = uas.sortby(uas.lon)
		vas.coords['lon'] = (vas.coords['lon'] + 180) % 360 - 180; vas = vas.sortby(vas.lon)
		ps.coords['lon'] = (ps.coords['lon'] + 180) % 360 - 180; ps = ps.sortby(ps.lon)
		try:
			pr.coords['lon'] = (pr.coords['lon'] + 180) % 360 - 180; pr = pr.sortby(pr.lon)
		except:
			pass

	#Trim to the domain given by "domain", as well as the year given by "year". Expand domain 
	# for later compairsons with ERA5, and for interpolation of U/V
	domain[0] = domain[0]-5
	domain[1] = domain[1]+5
	domain[2] = domain[2]-5
	domain[3] = domain[3]+5
	hus = trim_cmip5(hus, domain, year)
	ta = trim_cmip5(ta, domain, year)

	huss = trim_cmip5(huss, domain, year)
	tas = trim_cmip5(tas, domain, year)
	ps = trim_cmip5(ps, domain, year)
	try:
		pr = trim_cmip5(pr, domain, year)
	except:
		pass

	#Interpolate u, v, uas and vas, using a slightly bigger domain, then trim to "domain"
	ua = trim_cmip5(ua, [domain[0],domain[1],domain[2],domain[3]], year)
	va = trim_cmip5(va, [domain[0],domain[1],domain[2],domain[3]], year)
	uas = trim_cmip5(uas, [domain[0],domain[1],domain[2],domain[3]], year)
	vas = trim_cmip5(vas, [domain[0],domain[1],domain[2],domain[3]], year)
	ua = ua.ua.interp({"lon":hus.lon},method="linear", assume_sorted=True)
	va = va.va.interp({"lat":hus.lat},method="linear", assume_sorted=True)
	uas = uas.uas.interp({"lat":hus.lat,"lon":hus.lon},method="linear", assume_sorted=True)
	vas = vas.vas.interp({"lat":hus.lat,"lon":hus.lon},method="linear", assume_sorted=True)
	ua = trim_cmip5(ua, domain, year)
	va = trim_cmip5(va, domain, year)
	uas = trim_cmip5(uas, domain, year)
	vas = trim_cmip5(vas, domain, year)

	#Get common times for all datasets
	try:
		common_times = np.array(list(set(hus.time.values) & set(ta.time.values) & set(ua.time.values)\
			 & set(va.time.values) & set(huss.time.values) & set(tas.time.values)\
			& set(uas.time.values) & set(vas.time.values) & set(ps.time.values) & set(pr.time.values)))
	except:
		common_times = np.array(list(set(hus.time.values) & set(ta.time.values) & set(ua.time.values)\
			 & set(va.time.values) & set(huss.time.values) & set(tas.time.values)\
			& set(uas.time.values) & set(vas.time.values) & set(ps.time.values)))

	#Restrict all data to common times
	hus = hus.sel({"time": np.in1d(hus.time, common_times)})
	ta = ta.sel({"time": np.in1d(ta.time, common_times)})
	ua = ua.sel({"time": np.in1d(ua.time, common_times)})
	va = va.sel({"time": np.in1d(va.time, common_times)})
	huss = huss.sel({"time": np.in1d(huss.time, common_times)})
	tas = tas.sel({"time": np.in1d(tas.time, common_times)})
	uas = uas.sel({"time": np.in1d(uas.time, common_times)})
	vas = vas.sel({"time": np.in1d(vas.time, common_times)})
	ps = ps.sel({"time": np.in1d(ps.time, common_times)})
	try:
		pr = pr.sel({"time": np.in1d(pr.time, common_times)})
	except:
		pass

	#Either convert vertical coordinate to height or pressure, depending on the model 
	names = [] 
	for name, da in hus.data_vars.items(): 
		names.append(name) 
	if "orog" in names:
		#If the model has been stored on a hybrid height coordinate, it should have the
		#   variable "orog". Convert height coordinate to height ASL, and calculate 
		#   pressure via the hydrostatic equation
		z = hus.lev + (hus.b * hus.orog)
		orog = hus.orog.values
		q = hus.hus / (1 - hus.hus)
		tv = ta.ta * ( ( q + 0.622) / (0.622 * (1+q) ) )
		if ((model in ["ACCESS1-3","ACCESS1-0","ACCESS-CM2","ACCESS-ESM1-5"])):
			z = np.swapaxes(z, 0, 1).values
			orog = orog[0]
		else:
			z = np.tile(z.values.astype("float32"), [ta.ta.shape[0], 1, 1, 1], )
		p = hypsometric_p(ps.ps.values, z, tv.values, np.tile(orog[np.newaxis], (ta.ta.shape[0],1,1)))
	elif np.any(np.in1d(["p0","ap"], names)):
		#If the model has been stored on a hybrid pressure coordinate, it should have the
		#   variable "p0". Convert hybrid pressure coordinate to pressure, and calculate 
		#   height via the hydrostatic equation
		if "p0" in names:
			p = (hus.a * hus.p0 + hus.b * hus.ps).transpose("time","lev","lat","lon").values
		elif "ap" in names:
			p = (hus.ap + hus.b * hus.ps).transpose("time","lev","lat","lon").values
		else:
			raise ValueError("Check the hybrid-pressure coordinate of this model")
		q = hus.hus / (1 - hus.hus)
		tv = ta.ta * ( ( q + 0.622) / (0.622 * (1+q) ) )
		#z = (-287 * tv * (np.log( p / ps.ps)).transpose("time","lev","lat","lon") ) / 9.8
		orog = xr.open_dataset(glob.glob("/g/data/r87/DRSv3/CMIP5/"+\
		    model+"/historical/fx/atmos/r0i0p0/orog/latest/orog*.nc")[0])
		if orog.lon.values.max() >= 350:
			orog.coords['lon'] = (orog.coords['lon'] + 180) % 360 - 180; orog = orog.sortby(orog.lon)
		orog = trim_cmip5(orog.orog, domain, year).values
		orog[orog<0] = 0
		#z = (z + orog).values
		z = hypsometric_z(ps.ps.values, p, tv.values, np.tile(orog[np.newaxis], (ta.ta.shape[0],1,1)))
	else:
		raise ValueError("Check the vertical coordinate of this model")

	#Sanity checks on pressure and height, one of which is calculated via hydrostatic approx. Note ACCESS-CM2 is
	# missing a temperature level, and so that level will have zero pressure. Ignore sanity check for this model.
	if (z.min() < -1000) | (z.max() > 100000):
		raise ValueError("Potentially erroneous Z values (less than -1000 or greater than 100,000 km")
	if (p.max() > 200000) | (p.min() < 0):
		if model != "ACCESS-CM2":
			raise ValueError("Potentially erroneous pressure (less than 0 or greater than 200,000 Pa")

	#Convert quantities into those expected by wrf_(non)_parallel.py
	lon = ta.lon.values
	lat = ta.lat.values
	date_list = ta.time.values
	date_list = np.array([dt.datetime.strptime(date_list[t].strftime(), "%Y-%m-%d %H:%M:%S") for t in np.arange(len(date_list))]) 
	ta = ta.ta.values - 273.15
	hur = mpcalc.relative_humidity_from_specific_humidity(hus.hus.values, \
		    ta*units.units.degC, p*units.units.pascal) * 100
	pres = p / 100.
	sfc_pres = ps.ps.values / 100.
	tas = tas.tas.values - 273.15
	ta2d = mpcalc.dewpoint_from_specific_humidity(huss.huss.values, tas*units.units.degC, \
		    ps.ps.values*units.units.pascal)

	ua=ua.values
	va=va.values
	uas=uas.values
	vas=vas.values

	try:
		pr = pr.pr.values * (60*60*3)
		pr = np.where(pr<0, 0, pr)
	except:
		pr = np.zeros(sfc_pres.shape)

	#Mask all data above 100 hPa. For ACCESS-CM2, mask data below 20 m
	if model == "ACCESS-CM2":
		ta[(pres < 100) | (p == 0) | (p == np.inf)] = np.nan
		hur[(pres < 100) | (p == 0) | (p == np.inf)] = np.nan
		z[(pres < 100) | (p == 0) | (p == np.inf)] = np.nan
		ua[(pres < 100) | (p == 0) | (p == np.inf)] = np.nan
		va[(pres < 100) | (p == 0) | (p == np.inf)] = np.nan
	else:
		ta[pres < 100] = np.nan
		hur[pres < 100] = np.nan
		z[pres < 100] = np.nan
		ua[pres < 100] = np.nan
		va[pres < 100] = np.nan


	return [ta, hur, z, orog, pres, sfc_pres, ua, va, uas, vas, tas, ta2d, pr, lon,\
		    lat, date_list]

def read_cmip6(group, model, experiment, ensemble, year, domain):

	#DEPRECIATED - USE READ_CMIP INSTEAD, SPECIFYING CMIP_VER=6


	#Read CMIP6 data from the r87 project (from oi10 and fs38)

	#For the given model, institute, experiment, get the relevant file paths.
	hus_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP6/CMIP/"+group+"/"+model+"/"+experiment+"/"+ensemble+"/6hrLev/hus/gn/latest/*"))
	ta_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP6/CMIP/"+group+"/"+model+"/"+experiment+"/"+ensemble+"/6hrLev/ta/gn/latest/*"))
	ua_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP6/CMIP/"+group+"/"+model+"/"+experiment+"/"+ensemble+"/6hrLev/ua/gn/latest/*"))
	va_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP6/CMIP/"+group+"/"+model+"/"+experiment+"/"+ensemble+"/6hrLev/va/gn/latest/*"))


	huss_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP6/CMIP/"+group+"/"+model+"/"+experiment+"/"+ensemble+"/3hr/huss/gn/latest/*"))
	tas_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP6/CMIP/"+group+"/"+model+"/"+experiment+"/"+ensemble+"/3hr/tas/gn/latest/*"))
	uas_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP6/CMIP/"+group+"/"+model+"/"+experiment+"/"+ensemble+"/3hr/uas/gn/latest/*"))
	vas_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP6/CMIP/"+group+"/"+model+"/"+experiment+"/"+ensemble+"/3hr/vas/gn/latest/*"))
	ps_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP6/CMIP/"+group+"/"+model+"/"+experiment+"/"+ensemble+"/3hr/ps/gn/latest/*"))

	#Isolate the files relevant for the current "year"
	#NOTE will have to change to incorperate months if there is more than one file per year

	hus_fid = get_fid( hus_files, year)
	ta_fid = get_fid( ta_files, year)
	ua_fid = get_fid( ua_files, year)
	va_fid = get_fid( va_files, year)
	huss_fid = get_fid( huss_files, year)
	tas_fid = get_fid( tas_files, year)
	uas_fid = get_fid( uas_files, year)
	vas_fid = get_fid( vas_files, year)
	ps_fid = get_fid( ps_files, year)

	#Load the data, match 3 hourly and 6 hourly data
	hus = xr.open_mfdataset([hus_files[i] for i in hus_fid])
	ta = xr.open_mfdataset([ta_files[i] for i in ta_fid])
	ua = xr.open_mfdataset([ua_files[i] for i in ua_fid])
	va = xr.open_mfdataset([va_files[i] for i in va_fid])

	huss = xr.open_mfdataset([huss_files[i] for i in huss_fid])
	huss = huss.sel({"time":np.in1d(huss.time, hus.time)})
	tas = xr.open_mfdataset([tas_files[i] for i in tas_fid])
	tas = tas.sel({"time":np.in1d(tas.time, ta.time)})
	uas = xr.open_mfdataset([uas_files[i] for i in uas_fid])
	uas = uas.sel({"time":np.in1d(uas.time, ua.time)})
	vas = xr.open_mfdataset([vas_files[i] for i in vas_fid])
	vas = vas.sel({"time":np.in1d(vas.time, va.time)})
	ps = xr.open_mfdataset([ps_files[i] for i in ps_fid])
	ps = ps.sel({"time":np.in1d(ps.time, hus.time)})


	#and trim to the domain given by "domain", as well as the year given by "year"
	hus = trim_cmip5(hus, domain, year)
	ta = trim_cmip5(ta, domain, year)

	huss = trim_cmip5(huss, domain, year)
	tas = trim_cmip5(tas, domain, year)
	ps = trim_cmip5(ps, domain, year)

	#Interpolate u, v, uas and vas, using a slightly bigger domain, then trim to "domain"
	ua = trim_cmip5(ua, [domain[0]-5,domain[1]+5,domain[2]-5,domain[3]+5], year)
	va = trim_cmip5(va, [domain[0]-5,domain[1]+5,domain[2]-5,domain[3]+5], year)
	uas = trim_cmip5(uas, [domain[0]-5,domain[1]+5,domain[2]-5,domain[3]+5], year)
	vas = trim_cmip5(vas, [domain[0]-5,domain[1]+5,domain[2]-5,domain[3]+5], year)
	ua = ua.ua.interp({"lon":hus.lon},method="linear", assume_sorted=True)
	va = va.va.interp({"lat":hus.lat},method="linear", assume_sorted=True)
	uas = uas.uas.interp({"lat":hus.lat,"lon":hus.lon},method="linear", assume_sorted=True)
	vas = vas.vas.interp({"lat":hus.lat,"lon":hus.lon},method="linear", assume_sorted=True)
	ua = trim_cmip5(ua, domain, year).values
	va = trim_cmip5(va, domain, year).values
	uas = trim_cmip5(uas, domain, year).values
	vas = trim_cmip5(vas, domain, year).values

	#Convert vertical coordinate to height 
	z = hus.lev + (hus.b * hus.orog)
	orog = hus.orog.values

	#Calculate pressure via hydrostatic equation
	q = hus.hus / (1 - hus.hus)
	tv = ta.ta * ( ( q + 0.622) / (0.622 * (1+q) ) )
	p = np.swapaxes(np.swapaxes(ps.ps * np.exp( -9.8*z / (287*tv)), 3, 2), 2, 1)

	#Convert quantities into those expected by wrf_parallel.py
	ta = ta.ta.values - 273.15
	hur = mpcalc.relative_humidity_from_specific_humidity(hus.hus.values, \
		    ta*units.units.degC, p.values*units.units.pascal) * 100
	z = np.tile(z.values, [ta.shape[0], 1, 1, 1])
	pres = p.values / 100.
	sfc_pres = ps.ps.values / 100.
	tas = tas.tas.values - 273.15
	ta2d = mpcalc.dewpoint_from_specific_humidity(hus.hus.values, ta*units.units.degC, \
		    p.values*units.units.pascal)
	lon = p.lon.values
	lat = p.lat.values
	date_list = p.time.values

	#Mask all data above 100 hPa
	ta[pres < 100] = np.nan
	hur[pres < 100] = np.nan
	z[pres < 100] = np.nan
	ua[pres < 100] = np.nan
	va[pres < 100] = np.nan

	return [ta, hur, z, orog, pres, sfc_pres, ua, va, uas, vas, tas, ta2d, lon,\
		    lat, date_list]

def get_fid(files, year):

	file_years = []
	for i in np.arange(len(files)):
		year0 = int(files[i].split("_")[-1][:-3].split("-")[0][0:4])
		year1 = int(files[i].split("_")[-1][:-3].split("-")[1][0:4])
		if (year0 == year1) | (int(files[i].split("_")[-1][:-3].split("-")[1][4:6])):
			file_years.append( np.arange(year0, year1+1) )
		else:
			file_years.append( np.arange(year0, year1) )
	fid = np.where([np.in1d(year, file_years[i]).sum() \
		    for i in np.arange(len(files))])[0]

	return fid
	

def trim_cmip5(dataset, domain, year):

	try:
		dataset = dataset.sel({"lon":slice(domain[2], domain[3]),\
		    "lat":slice(domain[0],domain[1]),\
		    "time":np.in1d(dataset["time.year"],year)})
	except:
		dataset = dataset.sel({"lon":slice(domain[2], domain[3]),\
		    "lat":slice(domain[0],domain[1])})

	return dataset

def get_lsm(model, experiment, cmip_ver=5, group = "", al33=False):

	if cmip_ver == 5:

		#Get CMIP5 file paths

		if al33:
			#NOTE unknown behaviour for al33 directories with multiple versions

			if group == "":
				raise ValueError("Group required")

			lsm_files = np.sort(glob.glob("/g/data/al33/replicas/CMIP5/combined/"+\
				    group+"/"+model+"/"+experiment+\
				    "/fx/atmos/fx/r0i0p0/*/sftlf/sftlf*"))

		else:

			lsm_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP5/"+\
				    model+"/"+experiment+\
				    "/fx/atmos/r0i0p0/sftlf/latest/sftlf*"))
	elif cmip_ver == 6:

		#Get CMIP6 file paths

		lsm_files = np.sort(glob.glob("/g/data/r87/DRSv3/CMIP6/CMIP/"+group+"/"+model+\
			"/"+experiment+"/r1i1p1f1/fx/sftlf/gn/latest/sftlf*"))

	lsm = xr.open_dataset(lsm_files[0]).sftlf

	return lsm
	
def hypsometric_p(p_sfc, z, tv, terrain):

	#TODO Sort out the axis/shape. Check this works on ERA5. Check this works on hybrid-pressure level data (to calculate Z)

	#Get the difference in height between each level. Need to give this function the height axis
	zdiff = np.diff(z, axis=1, prepend=terrain[:,np.newaxis,:,:])
	#Calculate the second term of the hypsometric equation
	T2 = np.exp( (-9.8 * zdiff) / (287 * (tv) ) )
	#Initialise the full pressure array
	p_full = np.empty(tv.shape)
	p_full[:,0,:,:] = p_sfc * T2[:,0,:,:]
	for i in np.arange(1,p_full.shape[1]):
		p_full[:,i,:,:] = p_full[:,i-1,:,:] * T2[:,i,:,:]
	
	return p_full	

def hypsometric_z(p_sfc, p, tv, terrain):

        #Calculate the second term of the hypsometric equation
        T2 = (287 * tv) / 9.8 
        
        #Initialise the full pressure array
        z_full = np.empty(tv.shape)
        z_full[:,0,:,:] = np.log(p_sfc / p[:,0,:,:]) * T2[:,0,:,:] + terrain
        for i in np.arange(1,z_full.shape[1]):
                z_full[:,i,:,:] = np.log(p[:,i-1,:,:]/p[:,i,:,:]) * T2[:,i,:,:] + z_full[:,i-1,:,:]
                        
        return z_full

if __name__ == "__main__":

	#[ta, hur, z, orog, pres, sfc_pres, ua, va, uas, vas, tas, ta2d, lon,lat, date_list] = \
	 #   read_cmip5("ACCESS1-0", "historical", "r1i1p1", 1990, [-44.525, -9.975, 111.975, 156.275])
	[ta, hur, z, orog, pres, sfc_pres, ua, va, uas, vas, tas, ta2d, lon,lat, date_list] = \
	    read_cmip6("CSIRO-ARCCSS","ACCESS-CM2", "historical", "r1i1p1f1",\
	    1990, [-44.525, -9.975, 111.975, 156.275])
