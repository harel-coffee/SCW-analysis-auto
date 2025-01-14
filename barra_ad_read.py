from tqdm import tqdm
import gc
from dask.diagnostics import ProgressBar
import sys
import netCDF4 as nc
import numpy as np
import datetime as dt
import glob
import pandas as pd
import xarray as xr

from metpy.calc import vertical_velocity_pressure as omega
import metpy.calc as mpcalc
from metpy.units import units

from barra_read import get_aus_stn_info

def read_barra_ad(domain,times,wg_only):
	#Open BARRA_AD netcdf files and extract variables needed for a range of times and given
	# spatial domain

	#ref = dt.datetime(1970,1,1,0,0,0)
	date_list = date_seq(times,"hours",6)
	if len(times) > 1:
		date_list = date_seq(times,"hours",6)
	else:
		date_list = times

	#Get time-invariant pressure and spatial info
	no_p, pres, p_ind = get_pressure(100)
	pres = pres[p_ind]
	lon,lat = get_lat_lon()
	lon_ind = np.where((lon >= domain[2]) & (lon <= domain[3]))[0]
	lat_ind = np.where((lat >= domain[0]) & (lat <= domain[1]))[0]
	lon = lon[lon_ind]
	lat = lat[lat_ind]
	terrain = get_terrain(lat_ind,lon_ind)

	#Initialise arrays
	ta = np.empty((0,no_p,len(lat_ind),len(lon_ind)))
	dp = np.empty((0,no_p,len(lat_ind),len(lon_ind)))
	hur = np.empty((0,no_p,len(lat_ind),len(lon_ind)))
	hgt = np.empty((0,no_p,len(lat_ind),len(lon_ind)))
	ua = np.empty((0,no_p,len(lat_ind),len(lon_ind)))
	va = np.empty((0,no_p,len(lat_ind),len(lon_ind)))
	wap = np.empty((0,no_p,len(lat_ind),len(lon_ind)))
	uas = np.empty((0,len(lat_ind),len(lon_ind)))
	vas = np.empty((0,len(lat_ind),len(lon_ind)))
	tas = np.empty((0,len(lat_ind),len(lon_ind)))
	ta2d = np.empty((0,len(lat_ind),len(lon_ind)))
	ps = np.empty((0,len(lat_ind),len(lon_ind)))
	max_max_wg = np.empty((0,len(lat_ind),len(lon_ind)))
	max_wg = np.empty((0,len(lat_ind),len(lon_ind)))
	date_times = np.empty((0))
	p_3d = np.moveaxis(np.tile(pres,[ta.shape[2],ta.shape[3],1]),2,0)

	for t in np.arange(0,len(date_list)):
		year = dt.datetime.strftime(date_list[t],"%Y")
		month =	dt.datetime.strftime(date_list[t],"%m")
		day = dt.datetime.strftime(date_list[t],"%d")
		hour = dt.datetime.strftime(date_list[t],"%H")

		#Load BARRA forecast files
		max_max_wg_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_AD/v1/forecast/spec/"\
			+"max_max_wndgust10m/"+year+"/"+month+"/max_max_wndgust10m-fc-spec-PT1H-BARRA_AD-v*-"\
			+year+month+day+"T"+hour+"*.nc")[0])
		max_wg_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_AD/v1/forecast/spec/"\
			+"max_wndgust10m/"+year+"/"+month+"/max_wndgust10m-fc-spec-PT1H-BARRA_AD-v*-"\
			+year+month+day+"T"+hour+"*.nc")[0])
		if not wg_only:
			ta_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_AD/v1/forecast/prs/air_temp/"\
				+year+"/"+month+"/air_temp-fc-prs-PT1H-BARRA_AD-v*-"+year+month+day+"T"+hour+"*.nc")[0])
			z_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_AD/v1/forecast/prs/geop_ht/"\
				+year+"/"+month+"/geop_ht-fc-prs-PT1H-BARRA_AD-v*-"+year+month+day+"T"+hour+"*.nc")[0])
			ua_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_AD/v1/forecast/prs/wnd_ucmp/"\
				+year+"/"+month+"/wnd_ucmp-fc-prs-PT1H-BARRA_AD-v*-"+year+month+day+"T"+hour+"*.nc")[0])
			va_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_AD/v1/forecast/prs/wnd_vcmp/"\
				+year+"/"+month+"/wnd_vcmp-fc-prs-PT1H-BARRA_AD-v*-"+year+month+day+"T"+hour+"*.nc")[0])
			hur_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_AD/v1/forecast/prs/relhum/"\
				+year+"/"+month+"/relhum-fc-prs-PT1H-BARRA_AD-v*-"+year+month+day+"T"+hour+"*.nc")[0])
			w_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_AD/v1/forecast/prs/vertical_wnd/"\
				+year+"/"+month+"/vertical_wnd-fc-prs-PT1H-BARRA_AD-v*-"+year+month+day+"T"+hour+"*.nc")[0])
			uas_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_AD/v1/forecast/spec/uwnd10m/"\
				+year+"/"+month+"/uwnd10m-fc-spec-PT1H-BARRA_AD-v*-"+year+month+day+"T"+hour+"*.nc")[0])
			vas_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_AD/v1/forecast/spec/vwnd10m/"\
				+year+"/"+month+"/vwnd10m-fc-spec-PT1H-BARRA_AD-v*-"+year+month+day+"T"+hour+"*.nc")[0])
			ta2d_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_AD/v1/forecast/slv/dewpt_scrn/"\
				+year+"/"+month+"/dewpt_scrn-fc-slv-PT1H-BARRA_AD-v1*"+year+month+day+"T"+hour+"*.nc")[0])
			tas_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_AD/v1/forecast/spec/temp_scrn/"\
				+year+"/"+month+"/temp_scrn-fc-spec-PT1H-BARRA_AD-v1*"+year+month+day+"T"+hour+"*.nc")[0])
			ps_file = nc.Dataset(glob.glob("/g/data/ma05/BARRA_AD/v1/forecast/spec/sfc_pres/"\
				+year+"/"+month+"/sfc_pres-fc-spec-PT1H-BARRA_AD-v*-"+year+month+day+"T"+hour+"*.nc")[0])

		#Get times to load in from file
		date_times = np.append(date_times,\
			nc.num2date(np.round(max_max_wg_file["time"][:]),\
				max_max_wg_file["time"].units))

		temp_max_max_wg = max_max_wg_file["max_max_wndgust10m"][:,lat_ind,lon_ind]
		temp_max_wg = max_wg_file["max_wndgust10m"][:,lat_ind,lon_ind]
		max_max_wg = np.append(max_max_wg,temp_max_max_wg,axis=0)
		max_wg = np.append(max_wg,temp_max_wg,axis=0)
		max_max_wg_file.close()
		max_wg_file.close()
		
		if not wg_only:
			#Load data
			temp_ta = ta_file["air_temp"][:,p_ind,lat_ind,lon_ind] - 273.15
			temp_ua = ua_file["wnd_ucmp"][:,p_ind,lat_ind,lon_ind]
			temp_va = va_file["wnd_vcmp"][:,p_ind,lat_ind,lon_ind]
			temp_hgt = z_file["geop_ht"][:,p_ind,lat_ind,lon_ind]
			temp_hur = hur_file["relhum"][:,p_ind,lat_ind,lon_ind]
			temp_hur[temp_hur<0] = 0
			temp_dp = get_dp(temp_ta,temp_hur)
			temp_wap = omega( w_file["vertical_wnd"][:,p_ind,lat_ind,lon_ind] * (units.metre / units.second),\
				p_3d * (units.hPa), \
				temp_ta * units.degC )
			temp_uas = uas_file["uwnd10m"][:,lat_ind,lon_ind]
			temp_vas = vas_file["vwnd10m"][:,lat_ind,lon_ind]
			temp_ps = ps_file["sfc_pres"][:,lat_ind,lon_ind]/100 
			temp_tas = tas_file["temp_scrn"][:,lat_ind,lon_ind]/100 
			temp_ta2d = ta2d_file["dewpt_scrn"][:,lat_ind,lon_ind]/100 

			#Flip pressure axes for compatibility with SHARPpy
			temp_ta = np.flipud(temp_ta)
			temp_dp = np.flipud(temp_dp)
			temp_hur = np.flipud(temp_hur)
			temp_hgt = np.flipud(temp_hgt)
			temp_ua = np.flipud(temp_ua)
			temp_va = np.flipud(temp_va)
			temp_wap = np.flipud(temp_wap)
	
			#Fill arrays with current time steps
			ta = np.append(ta,temp_ta,axis=0)
			ua = np.append(ua,temp_ua,axis=0)
			va = np.append(va,temp_va,axis=0)
			hgt = np.append(hgt,temp_hgt,axis=0)
			hur = np.append(hur,temp_hur,axis=0)
			dp = np.append(dp,temp_dp,axis=0)
			wap = np.append(wap,temp_wap,axis=0)
			uas = np.append(uas,temp_uas,axis=0)
			vas = np.append(vas,temp_vas,axis=0)
			tas = np.append(tas,temp_vas,axis=0)
			ta2d = np.append(ta2d,temp_vas,axis=0)
			ps = np.append(ps,temp_ps,axis=0)

			ta_file.close();z_file.close();ua_file.close();va_file.close();hur_file.close();uas_file.close();vas_file.close();ps_file.close()
	
	#Flip pressure	
	pres = np.flipud(pres)
	
	if wg_only:
		return [max_max_wg,max_wg,lon,lat,date_times]
	else:
		return [max_max_wg,max_wg,ta,dp,hur,hgt,terrain,pres,ps,wap,ua,va,uas,vas,tas,ta2d,lon,lat,date_times]
	
def to_points_loop_wg10_sy(loc_id,points,fname,start_year,end_year,djf=False):
	#As in to_points_loop_wg10 but for BARRA-SY

	from dask.diagnostics import ProgressBar
	import gc
	ProgressBar().register()

	dates = []
	if djf:
	    for y in np.arange(start_year,end_year+1):
		    for m in [1,2,12]:
			    dates.append(dt.datetime(y,m,1,0,0,0))
	else:
	    for y in np.arange(start_year,end_year+1):
		    for m in np.arange(1,13):
			    dates.append(dt.datetime(y,m,1,0,0,0))

	df = pd.DataFrame()

	lsm = xr.open_dataset("/g/data/ma05/BARRA_SY/v1/static/lnd_mask-fc-slv-PT0H-BARRA_SY-v1.nc")

	#Read netcdf data
	for t in np.arange(len(dates)):
		print(dates[t])
		year = dt.datetime.strftime(dates[t],"%Y")
		month =	dt.datetime.strftime(dates[t],"%m")
		f = xr.open_mfdataset("/g/data/ma05/BARRA_SY/v1/forecast/spec/max_wndgust10m/"+\
			year+"/"+month+"/*.sub.nc", concat_dim="time")

		#Setup lsm
		lat = f.coords.get("latitude").values
		lon = f.coords.get("longitude").values
		x,y = np.meshgrid(lon,lat)
		x[lsm.lnd_mask==0] = np.nan
		y[lsm.lnd_mask==0] = np.nan

		dist_lon = []
		dist_lat = []
		for i in np.arange(len(loc_id)):

			dist = np.sqrt(np.square(x-points[i][0]) + \
				np.square(y-points[i][1]))
			temp_lat,temp_lon = np.unravel_index(np.nanargmin(dist),dist.shape)
			dist_lon.append(temp_lon)
			dist_lat.append(temp_lat)

		temp_df = f["max_wndgust10m"].isel(latitude = xr.DataArray(dist_lat, dims="points"), \
                                longitude = xr.DataArray(dist_lon, dims="points")).persist().to_dataframe()
		temp_df = temp_df.reset_index()

		for p in np.arange(len(loc_id)):
			temp_df.loc[temp_df.points==p,"loc_id"] = loc_id[p]

		temp_df = temp_df.drop(["points",\
			"forecast_period", "forecast_reference_time"],axis=1)
		df = pd.concat([df, temp_df])
		f.close()
		gc.collect()

	df.sort_values(["loc_id","time"]).to_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"+fname+".pkl")

def to_points_loop_wg10(loc_id,points,fname,start_year,end_year,djf=False):

	from dask.diagnostics import ProgressBar
	import gc
	ProgressBar().register()

	dates = []
	if djf:
	    for y in np.arange(start_year,end_year+1):
		    for m in [1,2,12]:
			    dates.append(dt.datetime(y,m,1,0,0,0))
	else:
	    for y in np.arange(start_year,end_year+1):
		    for m in np.arange(1,13):
			    dates.append(dt.datetime(y,m,1,0,0,0))

	df = pd.DataFrame()

	lsm = xr.open_dataset("/g/data/ma05/BARRA_AD/v1/static/lnd_mask-fc-slv-PT0H-BARRA_AD-v1.nc")

	#Read netcdf data
	for t in np.arange(len(dates)):
		print(dates[t])
		year = dt.datetime.strftime(dates[t],"%Y")
		month =	dt.datetime.strftime(dates[t],"%m")
		f = xr.open_mfdataset("/g/data/ma05/BARRA_AD/v1/forecast/spec/max_max_wndgust10m/"+\
			year+"/"+month+"/*.sub.nc", concat_dim="time")

		#Setup lsm
		lat = f.coords.get("latitude").values
		lon = f.coords.get("longitude").values
		x,y = np.meshgrid(lon,lat)
		x[lsm.lnd_mask==0] = np.nan
		y[lsm.lnd_mask==0] = np.nan

		dist_lon = []
		dist_lat = []
		for i in np.arange(len(loc_id)):

			dist = np.sqrt(np.square(x-points[i][0]) + \
				np.square(y-points[i][1]))
			temp_lat,temp_lon = np.unravel_index(np.nanargmin(dist),dist.shape)
			dist_lon.append(temp_lon)
			dist_lat.append(temp_lat)

		temp_df = f["max_max_wndgust10m"].isel(latitude = xr.DataArray(dist_lat, dims="points"), \
                                longitude = xr.DataArray(dist_lon, dims="points")).persist().to_dataframe()
		temp_df = temp_df.reset_index()

		for p in np.arange(len(loc_id)):
			temp_df.loc[temp_df.points==p,"loc_id"] = loc_id[p]

		temp_df = temp_df.drop(["points",\
			"forecast_period", "forecast_reference_time"],axis=1)
		df = pd.concat([df, temp_df])
		f.close()
		gc.collect()

	df.sort_values(["loc_id","time"]).to_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"+fname+".pkl")

def date_seq(times,delta_type,delta):
	start_time = times[0]
	end_time = times[1]
	current_time = times[0]
	date_list = [current_time]
	while (current_time < end_time):
		if delta_type == "hours":
			current_time = current_time + dt.timedelta(hours = delta)	
		date_list.append(current_time)
	return date_list

def get_pressure(top):
	ta_file = nc.Dataset("/g/data/ma05/BARRA_AD/v1/forecast/prs/air_temp/2012/12/air_temp-fc-prs-PT1H-BARRA_AD-v1-20121201T0000Z.sub.nc")
	p =ta_file["pressure"][:]
	p_ind = np.where(p>=top)[0]
	return [len(p_ind), p, p_ind]

def get_lat_lon():
	ta_file = nc.Dataset("/g/data/ma05/BARRA_AD/v1/forecast/prs/air_temp/2012/12/air_temp-fc-prs-PT1H-BARRA_AD-v1-20121201T0000Z.sub.nc")
	lon = ta_file["longitude"][:]
	lat = ta_file["latitude"][:]
	return [lon,lat]

def get_lat_lon_inds(points,lon,lat):
	lsm = nc.Dataset("/g/data/ma05/BARRA_AD/v1/static/lnd_mask-fc-slv-PT0H-BARRA_AD-v1.nc").variables["lnd_mask"][:]
	x,y = np.meshgrid(lon,lat)
	x[lsm==0] = np.nan
	y[lsm==0] = np.nan
	lat_ind = np.empty(len(points))
	lon_ind = np.empty(len(points))
	lat_used = np.empty(len(points))
	lon_used = np.empty(len(points))
	for point in np.arange(0,len(points)):
		dist = np.sqrt(np.square(x-points[point][0]) + \
				np.square(y-points[point][1]))
		dist_lat,dist_lon = np.unravel_index(np.nanargmin(dist),dist.shape)
		lat_ind[point] = dist_lat
		lon_ind[point] = dist_lon
		lon_used[point] = lon[dist_lon]
		lat_used[point] = lat[dist_lat]
	return [lon_ind, lat_ind, lon_used, lat_used]

def get_terrain(lat_ind,lon_ind):
	terrain_file = nc.Dataset("/g/data/ma05/BARRA_AD/v1/static/topog-fc-slv-PT0H-BARRA_AD-v1.nc")
	terrain = terrain_file.variables["topog"][lat_ind,lon_ind]
	terrain_file.close()
	return terrain


def remove_corrupt_dates(date_list):
	corrupt_dates = [dt.datetime(2014,11,22,6,0)]
	date_list = np.array(date_list)
	for i in np.arange(0,len(corrupt_dates)):
		date_list = date_list[~(date_list==corrupt_dates[i])]
	return date_list

if __name__ == "__main__":
	
	if len(sys.argv) > 1:
		start_year = int(sys.argv[1])
		end_year = int(sys.argv[2])
	if len(sys.argv) > 3:
		variable = sys.argv[3]
	
	#All 35 locs with AWS data
	loc_id, points = get_aus_stn_info()

	#BARRA-SY comparison locs
	to_keep = ["Coffs Harbour","Sydney","Wagga Wagga","Williamtown"]
	points = np.array(points)[np.in1d(loc_id,to_keep)]
	loc_id = np.array(loc_id)[np.in1d(loc_id,to_keep)]
	to_points_loop_wg10_sy(loc_id,points,"barra_sy_wg10_"+str(start_year)+"_"+str(end_year),start_year,end_year)

	#All 35 locs with AWS data
	loc_id, points = get_aus_stn_info()

	#BARRA-AD comparison locs
	to_keep = ["Adelaide","Ceduna","Coffs Harbour","Mount Gambier","Sydney","Wagga Wagga","Williamtown","Woomera"]
	points = np.array(points)[np.in1d(loc_id,to_keep)]
	loc_id = np.array(loc_id)[np.in1d(loc_id,to_keep)]
	to_points_loop_wg10(loc_id,points,"barra_ad_wg10_"+str(start_year)+"_"+str(end_year),start_year,end_year)
