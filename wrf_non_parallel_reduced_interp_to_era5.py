#import getcape
import argparse
from SkewT import get_dcape
import gc
import warnings
import sys
import itertools
import multiprocessing
import netCDF4 as nc
import numpy as np
import datetime as dt
import glob
import pandas as pd
import os
try:
	import metpy.units as units
	import metpy.calc as mpcalc
except:
	pass
import wrf
from calc_param import save_netcdf, get_dp
import xarray as xr
from erai_read import read_erai
from era5_read import read_era5, read_era5_rt52
from erai_read import get_mask as  get_erai_mask
from barra_read import read_barra, read_barra_fc
from barra_ad_read import read_barra_ad
from barra_read import get_mask as  get_barra_mask
from barpa_read import read_barpa
from read_cmip import read_cmip
from wrf_parallel import *

#-------------------------------------------------------------------------------------------------

#This file is the same as wrf_parallel.py, except that the work is not done in parallel

#Note that changes in diagnostic definitions within wrf_parallel.py, which are not contained within
#   functions, will need to be copied here.

#-------------------------------------------------------------------------------------------------

def interp_era5(field, lon, lat, era5_lon, era5_lat, d3=True):
	if d3:
		return xr.DataArray(data=field, dims=["lev","lat","lon"], coords={"lev":np.arange(field.shape[0]), "lat":lat, "lon":lon}).\
			interp({"lat":era5_lat,"lon":era5_lon},kwargs={"fill_value":None}).values
	else:
		return xr.DataArray(data=field, dims=["lat","lon"], coords={"lat":lat, "lon":lon}).\
			interp({"lat":era5_lat,"lon":era5_lon},kwargs={"fill_value":None}).values

def getcape_driver(sfc_p_3d, sfc_ta, sfc_dp, ps):
		#Apply code of George Bryan (NCAR). Options:
	        #pinc   ! Pressure increment (Pa)
				      # (smaller number yields more accurate
                                      #  results,larger number makes code 
                                      #  go faster)

		#source    ! Source parcel:
                                        # 1 = surface
                                        # 2 = most unstable (max theta-e)
                                        # 3 = mixed-layer (specify ml_depth)

		#ml_depth depth (m) of mixed layer 
                                          # for source=3

		#adiabat   ! Formulation of moist adiabat:
                                        # 1 = pseudoadiabatic, liquid only
                                        # 2 = reversible, liquid only
                                        # 3 = pseudoadiabatic, with ice
                                        # 4 = reversible, with ice
		cape_gb_mu1 = np.zeros((sfc_p_3d.shape[1], sfc_p_3d.shape[2]))
		cape_gb_mu4 = np.zeros((sfc_p_3d.shape[1], sfc_p_3d.shape[2]))
		for i in np.arange(sfc_p_3d.shape[1]):
			for j in np.arange(sfc_p_3d.shape[2]):
				agl_idx = sfc_p_3d[:,i,j] <= ps[i,j]
				temp_cape_mu1, temp_cin_mu1 = getcape.getcape(sfc_p_3d[agl_idx,i,j], sfc_ta[agl_idx,i,j], sfc_dp[agl_idx,i,j],\
					pinc=100.0, source=2, ml_depth=200.0, adiabat=1)
				temp_cape_mu4, temp_cin_mu4 = getcape.getcape(sfc_p_3d[agl_idx,i,j], sfc_ta[agl_idx,i,j], sfc_dp[agl_idx,i,j],\
					pinc=100.0, source=2, ml_depth=200.0, adiabat=4)
				cape_gb_mu1[i,j] = temp_cape_mu1
				cape_gb_mu4[i,j] = temp_cape_mu4
		return [cape_gb_mu1, cape_gb_mu4]

def fill_output(output, t, param, ps, p, data):

	output[:,:,:,np.where(param==p)[0][0]] = data

	return output

def main():
	load_start = dt.datetime.now()
	#Try parsing arguments using argparse
	parser = argparse.ArgumentParser(description='wrf non-parallel convective diagnostics processer')
	parser.add_argument("-m",help="Model name",required=True)
	parser.add_argument("-r",help="Region name (default is aus)",default="aus")
	parser.add_argument("-t1",help="Time start YYYYMMDDHH",required=True)
	parser.add_argument("-t2",help="Time end YYYYMMDDHH",required=True)
	parser.add_argument("-e", help="CMIP5 experiment name (not required if using era5, erai or barra)", default="")
	parser.add_argument("--barpa_forcing_mdl", help="BARPA forcing model (erai or ACCESS1-0). Default erai.", default="erai")
	parser.add_argument("--ens", help="CMIP5 ensemble name (not required if using era5, erai or barra)", default="r1i1p1")
	parser.add_argument("--group", help="CMIP6 modelling group name", default="")
	parser.add_argument("--project", help="CMIP6 modelling intercomparison project", default="CMIP")
	parser.add_argument("--ver6hr", help="Version on al33 for 6hr data", default="")
	parser.add_argument("--ver3hr", help="Version on al33 for 3hr data", default="")
	parser.add_argument("--issave",help="Save output (True or False, default is False)", default="False")
	parser.add_argument("--ub4",help="Where to get era5 data. Default True for ub4 project, otherwise rt52",default="True")
	parser.add_argument("--outname",help="Name of saved output. In the form *outname*_*t1*_*t2*.nc. Default behaviour is the model name",default=None)
	parser.add_argument("--is_dcape",help="Should DCAPE be calculated? (1 or 0. Default is 1)",default=1)
	parser.add_argument("--al33",help="Should data be gathered from al33? Default is False, and data is gathered from r87. If True, then group is required",default="False")
	parser.add_argument("--delta_t",help="Time step spacing for ERA5 data, in hours. Default is one the minimum spacing (1 hour)",default="1")
	parser.add_argument("--era5_interp",help="Horizontally interpolate model data before calculating convective parameters",default="False")
	args = parser.parse_args()

	#Parse arguments from cmd line and set up inputs (date region model)
	model = args.m
	region = args.r
	t1 = args.t1
	t2 = args.t2
	issave = args.issave
	ub4 = args.ub4
	al33 = args.al33
	if args.outname==None:
		out_name = model
	else:
		out_name = args.outname
	is_dcape = args.is_dcape
	barpa_forcing_mdl = args.barpa_forcing_mdl
	experiment = args.e
	ensemble = args.ens
	group = args.group
	project = args.project
	ver6hr = args.ver6hr
	ver3hr = args.ver3hr
	delta_t = int(args.delta_t)
	era5_interp = args.era5_interp
	if region == "sa_small":
		start_lat = -38; end_lat = -26; start_lon = 132; end_lon = 142
	elif region == "aus":
		start_lat = -44.525; end_lat = -9.975; start_lon = 111.975; end_lon = 156.275
	elif region == "global":
		start_lat = -70; end_lat = 70; start_lon = -180; end_lon = 179.75
	else:
		raise ValueError("INVALID REGION\n")
	domain = [start_lat,end_lat,start_lon,end_lon]
	try:
		time = [dt.datetime.strptime(t1,"%Y%m%d%H"),dt.datetime.strptime(t2,"%Y%m%d%H")]
	except:
		raise ValueError("INVALID START OR END TIME. SHOULD BE YYYYMMDDHH\n")
	if era5_interp=="True":
		era5_interp = True
	elif era5_interp=="False":
		era5_interp = False
	else:
		raise ValueError("\n INVALID era5_interp...SHOULD BE True OR False")
	if ub4=="True":
		ub4 = True
	elif ub4=="False":
		ub4 = False
	else:
		raise ValueError("\n INVALID ub4...SHOULD BE True OR False")
	if issave=="True":
		issave = True
	elif issave=="False":
		issave = False
	else:
		raise ValueError("\n INVALID ISSAVE...SHOULD BE True OR False")
	if al33=="True":
		al33 = True
	elif al33=="False":
		al33 = False
	else:
		raise ValueError("\n INVALID al33...SHOULD BE True OR False")

	#Load data
	print("LOADING DATA...")
	if model == "erai":
		ta,temp1,hur,hgt,terrain,p,ps,wap,ua,va,uas,vas,tas,ta2d,\
			cp,tp,wg10,mod_cape,lon,lat,date_list = \
			read_erai(domain,time)
		cp = cp.astype("float32", order="C")
		tp = tp.astype("float32", order="C")
		mod_cape = mod_cape.astype("float32", order="C")
	elif model == "era5":
		if ub4:
			ta,temp1,hur,hgt,terrain,p,ps,ua,va,uas,vas,tas,ta2d,\
				cp,wg10,mod_cape,lon,lat,date_list = \
				read_era5(domain,time,delta_t=delta_t)
		else:
			ta,temp1,hur,hgt,terrain,p,ps,ua,va,uas,vas,tas,ta2d,\
				cp,tp,wg10,mod_cape,lon,lat,date_list = \
				read_era5_rt52(domain,time,delta_t=delta_t)
		cp = cp.astype("float32", order="C")
		tp = tp.astype("float32", order="C")
		mod_cape = mod_cape.astype("float32", order="C")
		wap = np.zeros(hgt.shape)
	elif model == "barra":
		ta,temp1,hur,hgt,terrain,p,ps,wap,ua,va,uas,vas,tas,ta2d,wg10,lon,lat,date_list = \
			read_barra(domain,time)
	elif model == "barra_fc":
		ta,temp1,hur,hgt,terrain,p,ps,wap,ua,va,uas,vas,tas,ta2d,wg10,lon,lat,date_list = \
			read_barra_fc(domain,time)
	elif model == "barpa":
		ta,hur,hgt,terrain,p,ps,ua,va,uas,vas,tas,ta2d,wg10,lon,lat,date_list = \
			read_barpa(domain, time, experiment, barpa_forcing_mdl, ensemble)
		wap = np.zeros(hgt.shape)
		temp1 = None
	elif model == "barra_ad":
		wg10,temp2,ta,temp1,hur,hgt,terrain,p,ps,wap,ua,va,uas,vas,tas,ta2d,lon,lat,date_list = \
			read_barra_ad(domain, time, False)
	elif model in ["ACCESS1-0","ACCESS1-3","GFDL-CM3","GFDL-ESM2M","CNRM-CM5","MIROC5",\
		    "MRI-CGCM3","IPSL-CM5A-LR","IPSL-CM5A-MR","GFDL-ESM2G","bcc-csm1-1","MIROC-ESM",\
		    "BNU-ESM"]:
		#Check that t1 and t2 are in the same year
		year = np.arange(int(t1[0:4]), int(t2[0:4])+1)
		ta, hur, hgt, terrain, p_3d, ps, ua, va, uas, vas, tas, ta2d, tp, lon, lat, \
		    date_list = read_cmip(model, experiment, \
		    ensemble, year, domain, cmip_ver=5, al33=al33, group=group, ver6hr=ver6hr, ver3hr=ver3hr)
		wap = np.zeros(hgt.shape)
		wg10 = np.zeros(ps.shape)
		mod_cape = np.zeros(ps.shape)
		p = np.zeros(p_3d[0,:,0,0].shape)
		#date_list = pd.to_datetime(date_list).to_pydatetime()
		temp1 = None
		tp = tp.astype("float32", order="C")
	elif model in ["ACCESS-ESM1-5", "ACCESS-CM2"]:
		year = np.arange(int(t1[0:4]), int(t2[0:4])+1)
		ta, hur, hgt, terrain, p_3d, ps, ua, va, uas, vas, tas, ta2d, lon, lat, \
		    date_list = read_cmip(model, experiment,\
		    ensemble, year, domain, cmip_ver=6, group=group, project=project)
		wap = np.zeros(hgt.shape)
		wg10 = np.zeros(ps.shape)
		p = np.zeros(p_3d[0,:,0,0].shape)
		#date_list = pd.to_datetime(date_list).to_pydatetime()
		temp1 = None
	else:
		raise ValueError("Model not recognised")
	del temp1
	ta = ta.astype("float32", order="C")
	hur = hur.astype("float32", order="C")
	hgt = hgt.astype("float32", order="C")
	terrain = terrain.astype("float32", order="C")
	p = p.astype("float32", order="C")
	ps = ps.astype("float32", order="C")
	wap = wap.astype("float32", order="C")
	ua = ua.astype("float32", order="C")
	va = va.astype("float32", order="C")
	uas = uas.astype("float32", order="C")
	vas = vas.astype("float32", order="C")
	tas= tas.astype("float32", order="C")
	ta2d = ta2d.astype("float32", order="C")
	wg10 = wg10.astype("float32", order="C")
	lon = lon.astype("float32", order="C")
	lat = lat.astype("float32", order="C")

	gc.collect()

	param = np.array(["mu_cape", "mu_cin", "muq", "s06", "s0500", "lr700_500", "mhgt", "ta500","tp"])

	if model in ["erai","era5"]:
		param = np.concatenate([param, ["mod_cape"]])

	#Option to interpolate to the ERA5 grid
	if era5_interp:
		#Interpolate model data to the ERA5 grid
		from era5_read import get_lat_lon_rt52 as get_era5_lat_lon
		era5_lon,era5_lat = get_era5_lat_lon()
		era5_lon_ind = np.where((era5_lon >= domain[2]) & (era5_lon <= domain[3]))[0]
		era5_lat_ind = np.where((era5_lat >= domain[0]) & (era5_lat <= domain[1]))[0]
		era5_lon = era5_lon[era5_lon_ind]
		era5_lat = era5_lat[era5_lat_ind]
		terrain = interp_era5(terrain,lon,lat,era5_lon,era5_lat,d3=False)
		#Set output array
		output_data = np.zeros((ps.shape[0], era5_lat.shape[0], era5_lon.shape[0], len(param)))
	else:
		output_data = np.zeros((ps.shape[0], ps.shape[1], ps.shape[2], len(param)))



	#Assign p levels to a 3d array, with same dimensions as input variables (ta, hgt, etc.)
	#If the 3d p-lvl array already exists, then declare the variable "mdl_lvl" as true. 
	try:
		p_3d;
		mdl_lvl = True
		full_p3d = p_3d
	except:
		mdl_lvl = False
		if era5_interp:
			p_3d = np.moveaxis(np.tile(p,[ta.shape[2],ta.shape[3],1]),[0,1,2],[1,2,0]).\
			    astype(np.float32)
		else:
			p_3d = np.moveaxis(np.tile(p,[era5_lat.shape[0],era5_lon.shape[0],1]),[0,1,2],[1,2,0]).\
			    astype(np.float32)

	print("LOAD TIME..."+str(dt.datetime.now()-load_start))
	tot_start = dt.datetime.now()


	for t in np.arange(0,ta.shape[0]):
		cape_start = dt.datetime.now()

		if era5_interp:
		    ta_t = interp_era5(ta[t],lon,lat,era5_lon,era5_lat,d3=True)
		    hur_t = interp_era5(hur[t],lon,lat,era5_lon,era5_lat,d3=True)
		    hgt_t = interp_era5(hgt[t],lon,lat,era5_lon,era5_lat,d3=True)
		    ps_t = interp_era5(ps[t],lon,lat,era5_lon,era5_lat,d3=False)
		    wap_t = interp_era5(wap[t],lon,lat,era5_lon,era5_lat,d3=True)
		    ua_t = interp_era5(ua[t],lon,lat,era5_lon,era5_lat,d3=True)
		    va_t = interp_era5(va[t],lon,lat,era5_lon,era5_lat,d3=True)
		    uas_t = interp_era5(uas[t],lon,lat,era5_lon,era5_lat,d3=False)
		    vas_t = interp_era5(vas[t],lon,lat,era5_lon,era5_lat,d3=False)
		    tas_t = interp_era5(tas[t],lon,lat,era5_lon,era5_lat,d3=False)
		    ta2d_t = interp_era5(ta2d[t],lon,lat,era5_lon,era5_lat,d3=False)
		    tp_t = interp_era5(tp[t],lon,lat,era5_lon,era5_lat,d3=False)
		    mod_cape_t = interp_era5(mod_cape[t],lon,lat,era5_lon,era5_lat,d3=False)
		else:
		    ta_t = ta[t]
		    hur_t = hur[t]
		    hgt_t = hgt[t]
		    ps_t = ps[t]
		    wap_t = wap[t]
		    ua_t = ua[t]
		    va_t = va[t]
		    uas_t = uas[t]
		    vas_t = vas[t]
		    tas_t = tas[t]
		    ta2d_t = ta2d[t]
		    tp_t = tp[t]
		    mod_cape_t = mod_cape[t]
		print(date_list[t])
		output = np.zeros((1, ps_t.shape[0], ps_t.shape[1], len(param)))

		if mdl_lvl:
			if era5_interp:
				p_3d = interp_era5(full_p3d[t],lon,lat,era5_lon,era5_lat,d3=True)
			else:
				p_3d = full_p3d[t]

		dp = get_dp(hur=hur_t, ta=ta_t, dp_mask = False)

		#Insert surface arrays, creating new arrays with "sfc" prefix
		sfc_ta = np.insert(ta_t, 0, tas_t, axis=0) 
		sfc_hgt = np.insert(hgt_t, 0, terrain, axis=0) 
		sfc_dp = np.insert(dp, 0, ta2d_t, axis=0) 
		sfc_p_3d = np.insert(p_3d, 0, ps_t, axis=0) 
		sfc_ua = np.insert(ua_t, 0, uas_t, axis=0) 
		sfc_va = np.insert(va_t, 0, vas_t, axis=0) 
		sfc_wap = np.insert(wap_t, 0, np.zeros(vas_t.shape), axis=0) 

		#Sort by ascending p
		a,temp1,temp2 = np.meshgrid(np.arange(sfc_p_3d.shape[0]) , np.arange(sfc_p_3d.shape[1]),\
			 np.arange(sfc_p_3d.shape[2]))
		sort_inds = np.flip(np.lexsort([np.swapaxes(a,1,0),sfc_p_3d],axis=0), axis=0)
		sfc_hgt = np.take_along_axis(sfc_hgt, sort_inds, axis=0)
		sfc_dp = np.take_along_axis(sfc_dp, sort_inds, axis=0)
		sfc_p_3d = np.take_along_axis(sfc_p_3d, sort_inds, axis=0)
		sfc_ua = np.take_along_axis(sfc_ua, sort_inds, axis=0)
		sfc_va = np.take_along_axis(sfc_va, sort_inds, axis=0)
		sfc_ta = np.take_along_axis(sfc_ta, sort_inds, axis=0)

		#Calculate q and wet bulb for pressure level arrays with surface values
		sfc_ta_unit = units.units.degC*sfc_ta
		sfc_dp_unit = units.units.degC*sfc_dp
		sfc_p_unit = units.units.hectopascals*sfc_p_3d
		hur_unit = mpcalc.relative_humidity_from_dewpoint(ta_t*units.units.degC, dp*units.units.degC)*\
			100*units.units.percent
		q_unit = mpcalc.mixing_ratio_from_relative_humidity(hur_unit,\
			ta_t*units.units.degC,np.array(p_3d)*units.units.hectopascals)
		sfc_hur_unit = mpcalc.relative_humidity_from_dewpoint(sfc_ta_unit, sfc_dp_unit)*\
			100*units.units.percent
		sfc_q_unit = mpcalc.mixing_ratio_from_relative_humidity(sfc_hur_unit,\
			sfc_ta_unit,sfc_p_unit)
		sfc_theta_unit = mpcalc.potential_temperature(sfc_p_unit,sfc_ta_unit)
		sfc_thetae_unit = mpcalc.equivalent_potential_temperature(sfc_p_unit,sfc_ta_unit,sfc_dp_unit)
		sfc_thetae = np.array(mpcalc.equivalent_potential_temperature(ps_t*units.units.hectopascals,tas_t*units.units.degC,\
				    ta2d_t*units.units.degC))
		sfc_q = np.array(sfc_q_unit)
		sfc_hur = np.array(sfc_hur_unit)
		#sfc_wb = np.array(wrf.wetbulb( sfc_p_3d*100, sfc_ta+273.15, sfc_q, units="degC"))

		#Use getcape.f90
		#cape_gb_mu1, cape_gb_mu4 = getcape_driver(sfc_p_3d, sfc_ta, sfc_dp, ps_t)

		#Now get most-unstable CAPE (max CAPE in vertical, ensuring parcels used are AGL)
		cape3d = wrf.cape_3d(sfc_p_3d,sfc_ta+273.15,\
				sfc_q,sfc_hgt,\
				terrain,ps_t,\
				True,meta=False, missing=0)
		cape = cape3d.data[0]
		cin = cape3d.data[1]
		lfc = cape3d.data[2]
		lcl = cape3d.data[3]
		el = cape3d.data[4]
		#Mask values which are below the surface and above 350 hPa AGL
		cape[(sfc_p_3d > ps_t) | (sfc_p_3d<(ps_t-350))] = np.nan
		cin[(sfc_p_3d > ps_t) | (sfc_p_3d<(ps_t-350))] = np.nan
		lfc[(sfc_p_3d > ps_t) | (sfc_p_3d<(ps_t-350))] = np.nan
		lcl[(sfc_p_3d > ps_t) | (sfc_p_3d<(ps_t-350))] = np.nan
		el[(sfc_p_3d > ps_t) | (sfc_p_3d<(ps_t-350))] = np.nan
		#Get maximum (in the vertical), and get cin, lfc, lcl for the same parcel
		mu_cape_inds = np.tile(np.nanargmax(cape,axis=0), (cape.shape[0],1,1))
		mu_cape = np.take_along_axis(cape, mu_cape_inds, 0)[0]
		mu_cin = np.take_along_axis(cin, mu_cape_inds, 0)[0]
		mu_lfc = np.take_along_axis(lfc, mu_cape_inds, 0)[0]
		mu_lcl = np.take_along_axis(lcl, mu_cape_inds, 0)[0]
		mu_el = np.take_along_axis(el, mu_cape_inds, 0)[0]
		muq = np.take_along_axis(sfc_q, mu_cape_inds, 0)[0] * 1000

		#Calculate other parameters
		#Thermo
		thermo_start = dt.datetime.now()
		lr700_500 = get_lr_p(ta_t, p_3d, hgt_t, 700, 500)
		melting_hgt = get_t_hgt(sfc_ta,np.copy(sfc_hgt),0,terrain)
		melting_hgt = np.where((melting_hgt < 0) | (np.isnan(melting_hgt)), 0, melting_hgt)
		ta500 = get_var_p_lvl(np.copy(sfc_ta), sfc_p_3d, 500)
		ta925 = get_var_p_lvl(np.copy(sfc_ta), sfc_p_3d, 925)
		ta850 = get_var_p_lvl(np.copy(sfc_ta), sfc_p_3d, 850)
		ta700 = get_var_p_lvl(np.copy(sfc_ta), sfc_p_3d, 700)
		rho = mpcalc.density( np.array(sfc_p_3d) * (units.units.hectopascal), sfc_ta * units.units.degC, sfc_q_unit)
		rho925 = np.array(get_var_p_lvl(np.array(rho), sfc_p_3d, 925))
		rho850 = np.array(get_var_p_lvl(np.array(rho), sfc_p_3d, 850))
		rho700 = np.array(get_var_p_lvl(np.array(rho), sfc_p_3d, 700))
		#Winds
		winds_start = dt.datetime.now()
		s06 = get_shear_hgt(sfc_ua, sfc_va, np.copy(sfc_hgt), 0, 6000, terrain)
		s0500 = get_shear_p(ua_t, va_t, p_3d, "sfc", np.array([500]), p_3d, uas=uas_t, vas=vas_t)[0]

		#WAP
		if model in ["erai","era5"]:
			sfc_w = mpcalc.vertical_velocity( wap_t * (units.units.pascal / units.units.second),\
				np.array(p_3d) * (units.units.hectopascal), \
				ta_t * units.units.degC,  q_unit)
			w925 = np.array(get_var_p_lvl(np.array(sfc_w), p_3d, 925))
			w850 = np.array(get_var_p_lvl(np.array(sfc_w), p_3d, 850))
			w700 = np.array(get_var_p_lvl(np.array(sfc_w), p_3d, 700))

		#Convergence
		if era5_interp:
			x, y = np.meshgrid(era5_lon,era5_lat)
		else:
			x, y = np.meshgrid(lon,lat)
		dx, dy = mpcalc.lat_lon_grid_deltas(x,y)
		u925 = np.array(get_var_p_lvl(np.copy(sfc_ua), sfc_p_3d, 925))
		u850 = np.array(get_var_p_lvl(np.copy(sfc_ua), sfc_p_3d, 850))
		u700 = np.array(get_var_p_lvl(np.copy(sfc_ua), sfc_p_3d, 700))
		v925 = np.array(get_var_p_lvl(np.copy(sfc_va), sfc_p_3d, 925))
		v850 = np.array(get_var_p_lvl(np.copy(sfc_va), sfc_p_3d, 850))
		v700 = np.array(get_var_p_lvl(np.copy(sfc_va), sfc_p_3d, 700))
		conv925 = -1e5*np.array(mpcalc.divergence(u925 * (units.units.meter / units.units.second), v925  * (units.units.meter / units.units.second), dx, dy))
		conv850 = -1e5*np.array(mpcalc.divergence(u850 * (units.units.meter / units.units.second), v850  * (units.units.meter / units.units.second), dx, dy))
		conv700 = -1e5*np.array(mpcalc.divergence(u700 * (units.units.meter / units.units.second), v700  * (units.units.meter / units.units.second), dx, dy))

		#CS6
		mucs6 = mu_cape * np.power(s06, 1.67)

		#Fill output
		output = fill_output(output, t, param, ps, "mu_cape", mu_cape)
		output = fill_output(output, t, param, ps, "mu_cin", mu_cin)
		output = fill_output(output, t, param, ps, "muq", muq)
		output = fill_output(output, t, param, ps, "s06", s06)
		output = fill_output(output, t, param, ps, "s0500", s0500)
		output = fill_output(output, t, param, ps, "lr700_500", lr700_500)
		output = fill_output(output, t, param, ps, "ta500", ta500)
		output = fill_output(output, t, param, ps, "mhgt", melting_hgt)
		output = fill_output(output, t, param, ps, "tp", tp_t)
		if (model == "erai") | (model == "era5"):
			output = fill_output(output, t, param, ps, "mod_cape", mod_cape_t)

		output_data[t] = output

	print("SAVING DATA...")
	param_out = []
	for param_name in param:
		temp_data = output_data[:,:,:,np.where(param==param_name)[0][0]]
		param_out.append(temp_data)

	#If the mhgt variable is zero everywhere, then it is likely that data has not been read.
	#In this case, all values are missing, set to zero.
	for t in np.arange(param_out[0].shape[0]):
		if param_out[np.where(param=="mhgt")[0][0]][t].max() == 0:
			for p in np.arange(len(param_out)):
				param_out[p][t] = np.nan

	if issave:
		if era5_interp:
			save_netcdf(region, model, out_name, date_list, era5_lat, era5_lon, param, param_out, \
				out_dtype = "f4", compress=True)
		else:
			save_netcdf(region, model, out_name, date_list, lat, lon, param, param_out, \
				out_dtype = "f4", compress=True)

	print(dt.datetime.now() - tot_start)

if __name__ == "__main__":


	warnings.simplefilter("ignore")

	main()
