import tqdm
import metpy.units as units
import metpy.calc as mpcalc
import wrf
import numpy as np
import pandas as pd
import datetime as dt
import math
import os
import pytz
#from event_analysis import get_aus_stn_info
import netCDF4 as nc
from SkewT import get_dcape
from wrf_parallel import get_shear_hgt, trapz_int3d
try:
	import sharppy.sharptab.profile as profile
	import sharppy.sharptab.utils as utils
	import sharppy.sharptab.params as params
	import sharppy.sharptab.interp as interp
	import sharppy.sharptab.winds as winds
except:
	pass

def get_eff_cape(cape, cin, sfc_p_3d, sfc_ta, sfc_hgt, sfc_q, ps, terrain):

	#Define the effective layer cape condition for the 3d grid
	cape_cond = (cape >= 100) & (cin <= 250) & (sfc_p_3d <= ps)
	eff_cape_cond = np.zeros(cape_cond.shape, dtype=bool)
	is_first = np.ones((cape_cond.shape[1], cape_cond.shape[2]), dtype=bool)
	eff_cape_cond[0] = cape_cond[0] 
	for i in np.arange(1,cape_cond.shape[0]):
		eff_cape_cond[i] = cape_cond[i]
		is_first[is_first & (~cape_cond[i] & cape_cond[i-1])]=False
		eff_cape_cond[i, ~is_first] = False

	#Extract pressure and height for effective levels
	eff_p = np.where(eff_cape_cond,\
		sfc_p_3d, np.nan)
	eff_hgt = np.where(eff_cape_cond,\
		sfc_hgt, np.nan)

	#Define "average" conditions over the effective layer. For air temp and water vapour,
	# the pressure-weighted average is used. For height and pressure, use the halfway point. 
	#If the layer is of one-level depth, use that layer's conditions
	eff_avg_p = ((np.nanmin(eff_p,axis=0) + np.nanmax(eff_p,axis=0)) / 2).astype(np.float32)
	eff_avg_hgt = ((np.nanmin(eff_hgt,axis=0) + np.nanmax(eff_hgt,axis=0)) / 2).astype(np.float32)
	eff_avg_ta = trapz_int3d(sfc_ta, sfc_p_3d, eff_cape_cond).astype(np.float32)
	eff_avg_q = trapz_int3d(sfc_q, sfc_p_3d, eff_cape_cond).astype(np.float32)

	#So that the wrf-python code behaves nicely, fill the points with no effective layer, using surface conditions.
	#These points will be masked later
	eff_avg_p = np.where(np.isnan(eff_avg_p),\
	       np.ma.masked_where(~((sfc_p_3d==ps)),\
	       sfc_p_3d).max(axis=0).filled(0)\
	       ,eff_avg_p).astype(np.float32)
	eff_avg_hgt = np.where(np.isnan(eff_avg_p),\
	       np.ma.masked_where(~((sfc_p_3d==ps)),\
	       sfc_hgt).max(axis=0).filled(0)\
	       ,eff_avg_hgt).astype(np.float32)
	eff_avg_ta = np.where(np.isnan(eff_avg_p),\
	       np.ma.masked_where(~((sfc_p_3d==ps)),\
	       sfc_ta).max(axis=0).filled(0)\
	       ,eff_avg_ta).astype(np.float32)
	eff_avg_q = np.where(np.isnan(eff_avg_p),\
	       np.ma.masked_where(~((sfc_p_3d==ps)),\
	       sfc_q).max(axis=0).filled(0)\
	       ,eff_avg_q).astype(np.float32)

	#Insert the effective layer conditions into the bottom of the 3d arrays pressure-level arrays
	eff_ta_arr = np.insert(sfc_ta,0,eff_avg_ta,axis=0)
	eff_q_arr = np.insert(sfc_q,0,eff_avg_q,axis=0)
	eff_hgt_arr = np.insert(sfc_hgt,0,eff_avg_hgt,axis=0)
	eff_p3d_arr = np.insert(sfc_p_3d,0,eff_avg_p,axis=0)

	#Sort arrays by ascending pressure
	a,temp1,temp2 = np.meshgrid(np.arange(eff_p3d_arr.shape[0]) ,\
		 np.arange(eff_p3d_arr.shape[1]), np.arange(eff_p3d_arr.shape[2]))
	sort_inds = np.flipud(np.lexsort([np.swapaxes(a,1,0),eff_p3d_arr],axis=0))
	eff_ta_arr = np.take_along_axis(eff_ta_arr, sort_inds, axis=0)
	eff_p3d_arr = np.take_along_axis(eff_p3d_arr, sort_inds, axis=0)
	eff_hgt_arr = np.take_along_axis(eff_hgt_arr, sort_inds, axis=0)
	eff_q_arr = np.take_along_axis(eff_q_arr, sort_inds, axis=0)

	#Calculate CAPE using wrf-python. 
	cape3d_effavg = wrf.cape_3d(eff_p3d_arr[:,0,0],eff_ta_arr[:,0,0] + 273.15,\
		eff_q_arr[:,0,0],eff_hgt_arr[:,0,0],terrain,ps,False,meta=False, missing=0)

	#From the 3d CAPE array, return just the effective layer vaues
	eff_cape = np.ma.masked_where(~((eff_ta_arr==eff_avg_ta) & \
		(eff_p3d_arr==eff_avg_p)),cape3d_effavg.data[0]).max(axis=0).filled(0)
	eff_cin = np.ma.masked_where(~((eff_ta_arr==eff_avg_ta) & \
		(eff_p3d_arr==eff_avg_p)),cape3d_effavg.data[1]).max(axis=0).filled(0)
	eff_lfc = np.ma.masked_where(~((eff_ta_arr==eff_avg_ta) & \
		(eff_p3d_arr==eff_avg_p)),cape3d_effavg.data[2]).max(axis=0).filled(0)
	eff_lcl = np.ma.masked_where(~((eff_ta_arr==eff_avg_ta) & \
		(eff_p3d_arr==eff_avg_p)),cape3d_effavg.data[3]).max(axis=0).filled(0)
	eff_el = np.ma.masked_where(~((eff_ta_arr==eff_avg_ta) & \
		(eff_p3d_arr==eff_avg_p)),cape3d_effavg.data[4]).max(axis=0).filled(0)

	#Finally, make sure to mask points where there is no effective layer
	eff_cape[eff_cape_cond.sum(axis=0) == 0] = 0
	eff_cin[eff_cape_cond.sum(axis=0) == 0] = 0
	eff_lfc[eff_cape_cond.sum(axis=0) == 0] = 0
	eff_lcl[eff_cape_cond.sum(axis=0) == 0] = 0
	eff_el[eff_cape_cond.sum(axis=0) == 0] = 0

	return eff_cape, eff_cin, eff_lfc, eff_lcl, eff_el, eff_hgt, eff_avg_hgt

def get_aus_stn_info():

	names = ["id", "stn_no", "district", "stn_name", "1", "2", "lat", "lon", "3", "4", "5", "6", "7", "8", \
			"9", "10", "11", "12", "13", "14", "15", "16"]	

	df = pd.read_csv("/g/data/eg3/ab4502/ExtremeWind/obs/aws/daily_aus_full/DC02D_StnDet_999999999643799.txt",\
		names=names, header=0)

	renames = aus_stn_info()

	df = df.replace({"stn_name":renames})

	points = [(df.lon.iloc[i], df.lat.iloc[i]) for i in np.arange(df.shape[0])]

	return [df.stn_name.values,points]

def analyse_events(event_type, domain, model=None, lightning_only=False, lightning_thresh=0, wg_thresh=0,\
			remove_tc_affected=False, add_sta=True):
        #Read data and combine

	#Load AWS, lightning and model data
	if domain == "sa_small":
		aws = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/obs/aws/"\
                        +"all_daily_max_wind_gusts_sa_1979_2017.pkl")
		lightning = load_lightning(domain="sa_small",daily=False)

		if model == "erai":
			erai = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"\
				+"erai_points_sa_small_1979_2017.pkl")
			erai_fc = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"\
				+"erai_fc_points_1979_2017.pkl")
		elif model == "barra":
			barra = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"+\
				"barra_points_sa_small_2003_2016_daily_max.pkl")
			barra_r_fc = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/barra_r_fc/"\
				+"barra_r_fc_points_daily_2003_2016.pkl")
	elif domain == "aus":
		lightning = load_lightning(domain="aus",daily=False)
		aws = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/obs/aws/"\
                        	+"all_daily_max_wind_gusts_6hrly_aus_1979_2017.pkl")

		if model == "erai":
			erai = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"\
				+"erai_points_aus1979_2017.pkl")

	#If interested in the jdh dataset, read and combine with AWS, model and lightning
	if event_type == "jdh":
		jdh = read_non_synoptic_wind_gusts().set_index("station",append=True)
		if model == "erai":
			df = pd.concat([aws["wind_gust"],jdh["gust (m/s)"],erai,erai_fc["wg10"],\
				lightning["lightning"]] ,axis=1)
		elif model == "barra":
			df = pd.concat([aws["wind_gust"],jdh["gust (m/s)"],barra,barra_r_fc["max_wg10"],\
				lightning["lightning"]], axis=1)
			df = df.dropna(subset=["ml_cape"])
		df["jdh"] = 0
		df.loc[((~df["gust (m/s)"].isna()) & (~df["wind_gust"].isna()) & (df["wind_gust"]>=20)),"jdh"] = 1
		df = df.dropna(subset=["wind_gust"])
		print("No. of JDH events = "+str(df.jdh.sum()))

	#Else, combine just the AWS, model and lightning
	elif event_type == "aws":
		if domain == "sa_small":
			df = pd.concat([aws["wind_gust"].set_index(["an_gust_time_utc","stn_name"]),\
				erai.set_index(["loc_id"],append=True),\
				erai_fc.set_index(["date","loc_id"])["wg10"],\
				lightning.set_index(["date","loc_id"])["lightning"],\
                		],axis=1)
		elif domain == "aus":
			df = pd.concat([aws.set_index(["an_gust_time_utc","stn_name"]),\
				lightning.set_index(["date","loc_id"])["lightning"]],axis=1)
			if add_sta:
				df = verify_sta(df)

		if model == "erai":
			df = pd.concat([df,erai.set_index(["loc_id"],append=True)],axis=1)

		df = df.dropna(subset=["wind_gust"])
		df = df[df.wind_gust >= wg_thresh]

		if lightning_only:
			df = df.dropna(subset=["lightning"])
			df = df[df.lightning >= lightning_thresh]

		if remove_tc_affected:
			df = df[df.tc_affected==0]
	
	return df

def read_upperair_obs(start_date,end_date,fout,code="wrfpython"):

	#Read in upper air obs for Adelaide, Woomera, Sydney and Darwin, and compute convective indices using 
	# either wrfpython or SHARPpy. Note that if wrfpython is chosen, then all wind indices, k-index, total 
	# totals and DCAPE are still calculated using SHARPpy

	names = ["record_id","stn_id","date_time","ta","ta_quality","dp","dp_quality",\
		"rh","rh_quality","ws","ws_quality","wd","wd_quality","p","p_quality",
		"z","z_quality","symbol"]
	path = "/g/data/eg3/ab4502/ExtremeWind/obs/upper_air/"
	fnames = [path + "UA01D_Data_023034_999999999723955.txt",\
		path + "UA01D_Data_014015_999999999723955.txt",\
		path + "UA01D_Data_016001_999999999723955.txt",\
		path + "UA01D_Data_066037_999999999723955.txt" ]
	df = pd.DataFrame()
	for f in fnames:
		df = pd.concat([df, pd.read_csv(f ,\
			names=names,dtype={"ta":np.float64},na_values = ["     ","      ","   ","          "," "])] )
	times = [dt.datetime.strptime(x,"%d/%m/%Y %H:%M") for x in df["date_time"]]
	df["date"] = times
	df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].reset_index()

	#Convert wind speed and direction to U and V
	df = uv(df)

	#Group by time/date of observation
	groups = df.groupby(["date", "stn_id"])

	#Loop over date/times/groups and keep ones which have >12 heights for all variables
	min_no_of_points = 12	#Set min no of points required to consider sounding
	mu_cape = []
	times = []

	df = pd.DataFrame(columns=["stn_id","mu_cape","ml_cape","k_index","s06","Umean800_600",\
			"Umean01","dcape","t_totals"])

	stn_det = pd.read_csv("/g/data/eg3/ab4502/ExtremeWind/obs/aws/"+\
			"daily_aus_full/DC02D_StnDet_999999999643799.txt")

	no_points = []
	for name, group in tqdm.tqdm(groups):

		group.loc[group.dp < -100, "dp"] = np.nan

		group = group.dropna(subset=["ta","dp","rh","ua","va","z"])

		if ( (group.shape[0] > min_no_of_points) & (group.p.min()<200) & (group.p.max()>850) ):

			if (group.p[:-1].values - group.p[1:].values).max() < 100:

				no_points.append(group.shape[0])

				if code == "sharppy":
					prof = profile.create_profile(pres = group.p, \
							hght = group.z, tmpc = group.ta, \
							dwpc = group.dp, u = group.ua, v=group.va, strictQC=False)
					
					sb_parcel = params.parcelx(prof, flag=1, dp=-10)
					mu_parcel = params.parcelx(prof, flag=3, dp=-10)
					ml_parcel = params.parcelx(prof, flag=4, dp=-10)
					eff_parcel = params.parcelx(prof, flag=6, ecape=100,\
					     ecinh=-250, dp=-10)
					p1km = interp.pres(prof, interp.to_msl(prof, 1000.))
					p6km = interp.pres(prof, interp.to_msl(prof, 6000.))
					sfc = prof.pres[prof.sfc]
					s06_u, s06_v = winds.wind_shear(prof, sfc, p6km)
					u01, v01  = winds.mean_wind(prof, sfc, p1km)
					u800_600, v800_600  = winds.mean_wind(prof, 800, 600)
					s06 = utils.KTS2MS( utils.mag(s06_u, s06_v) )
					Umean800_600 = utils.KTS2MS( utils.mag(u800_600, v800_600) )
					Umean01 = utils.KTS2MS( utils.mag(u01, v01) )
					v_totals = params.v_totals(prof)
					c_totals = params.c_totals(prof)
					t_totals = c_totals + v_totals
					dcape = params.dcape(prof)[0]
					if dcape < 0:
						dcape = 0
					ml_el = ml_parcel.elhght
					if np.ma.is_masked(ml_el):
						ml_el = np.nanmax(prof.hght)

				elif code == "wrfpython":
					
					terrain = stn_det.loc[np.in1d(\
						stn_det["Bureau of Meteorology Station Number"], \
						group.stn_id.unique()), \
						"Height of station above mean sea level in metres"]\
						.values[0]
					rh = np.array(mpcalc.relative_humidity_from_dewpoint(group.ta.values\
						*units.units.degC, \
						group.dp.values*units.units.degC)*100)
					group.loc[:,"q"] = mpcalc.mixing_ratio_from_relative_humidity(\
						rh*units.units.percent, \
						group.ta.values*units.units.degC,\
						group.p.values*units.units.hPa)
					res = wrf.cape_3d(group.p.values, group.ta.values+273.15, \
						group.q.values, group.z.values, \
						terrain, group.p.max(), ter_follow=False).values
					
					ml_inds = ((group.p <= group.p.max()) & \
						(group.p >= (group.p.max() - 100)))
					ml_ta_avg = np.squeeze( trapz_int3d( \
						group.ta.values[np.newaxis][np.newaxis].T, \
						group.p.values[np.newaxis][np.newaxis].T, \
						np.array(ml_inds)[np.newaxis][np.newaxis].T ) )\
						.astype(np.float32)
					ml_q_avg = np.squeeze( trapz_int3d( \
						group.q.values[np.newaxis][np.newaxis].T, \
						group.p.values[np.newaxis][np.newaxis].T, \
						np.array(ml_inds)[np.newaxis][np.newaxis].T ) )\
						.astype(np.float32)
					ml_hgt_avg = np.ma.masked_where(~ml_inds, group.z).mean()
					ml_p3d_avg = np.ma.masked_where(~ml_inds, group.p).mean()
					#Insert the mean values into the bottom of the 3d arrays
					# pressure-level arrays
					ml_ta_arr = np.insert(group.ta.values,0,ml_ta_avg)
					ml_q_arr = np.insert(group.q.values,0,ml_q_avg)
					ml_hgt_arr = np.insert(group.z.values,0,ml_hgt_avg)
					ml_p3d_arr = np.insert(group.p.values,0,ml_p3d_avg)
					#Sort by ascending p
					idx = np.argsort(ml_p3d_arr)
					ml_ta_arr = ml_ta_arr[idx]
					ml_q_arr = ml_q_arr[idx]
					ml_hgt_arr = ml_hgt_arr[idx]
					ml_p3d_arr = ml_p3d_arr[idx]
					#Calculate CAPE using wrf-python. 
					cape3d_mlavg = np.squeeze(wrf.cape_3d(ml_p3d_arr,\
						(ml_ta_arr + 273.15),\
						ml_q_arr,\
						ml_hgt_arr,terrain,\
						group.p.max(),False,meta=False, missing=0))
					ml_cape = np.ma.masked_where(~((ml_ta_arr==ml_ta_avg) &\
						 (ml_p3d_arr==ml_p3d_avg)),\
						cape3d_mlavg.data[0]).max()
					ml_cin = np.ma.masked_where(~((ml_ta_arr==ml_ta_avg) &\
						(ml_p3d_arr==ml_p3d_avg)),\
						cape3d_mlavg.data[1]).max()
					ml_el = np.ma.masked_where(~((ml_ta_arr==ml_ta_avg) &\
						(ml_p3d_arr==ml_p3d_avg)),\
						cape3d_mlavg.data[4]).max()
					cape = res[0]
					cin = res[1]
					lfc = res[2]
					lcl = res[3]
					el = res[4]

					#Get effective CAPE
					eff_cape, eff_cin, eff_lfc, eff_lcl, eff_el, eff_hgt, eff_avg_hgt = get_eff_cape(\
						cape=cape[np.newaxis][np.newaxis].T,\
						cin=cin[np.newaxis][np.newaxis].T,\
						sfc_p_3d=group.p.values[np.newaxis][np.newaxis].T,\
						sfc_ta=group.ta.values[np.newaxis][np.newaxis].T,\
						sfc_hgt=group.z.values[np.newaxis][np.newaxis].T,\
						sfc_q=group.q.values[np.newaxis][np.newaxis].T,\
						ps=group.p.max()[np.newaxis][np.newaxis].T,\
						terrain=terrain[np.newaxis][np.newaxis].T)
					eff_cape = np.where(np.isnan(eff_cape), 0, eff_cape)
					eff_cin = np.where(np.isnan(eff_cin), 0, eff_cin)
					eff_lfc = np.where(np.isnan(eff_lfc), 0, eff_lfc)
					eff_lcl = np.where(np.isnan(eff_lcl), 0, eff_lcl)
					eff_el = np.where(np.isnan(eff_el), 0, eff_el)


					#Mass-weighted
					umean01 = np.squeeze(trapz_int3d( group.ua.values[np.newaxis][np.newaxis].T,\
						group.p.values[np.newaxis][np.newaxis].T, \
						np.array((group.z - terrain) <= 1000)[np.newaxis][np.newaxis].T,)\
						.data)
					vmean01 = np.squeeze(trapz_int3d( group.va.values[np.newaxis][np.newaxis].T,\
						group.p.values[np.newaxis][np.newaxis].T, \
						np.array((group.z - terrain) <= 1000)[np.newaxis][np.newaxis].T,)\
						.data)
					umean06 = np.squeeze(trapz_int3d( group.ua.values[np.newaxis][np.newaxis].T,\
						group.p.values[np.newaxis][np.newaxis].T, \
						np.array((group.z - terrain) <= 6000)[np.newaxis][np.newaxis].T,)\
						.data)
					vmean06 = np.squeeze(trapz_int3d( group.va.values[np.newaxis][np.newaxis].T,\
						group.p.values[np.newaxis][np.newaxis].T, \
						np.array((group.z - terrain) <= 6000)[np.newaxis][np.newaxis].T,)\
						.data)
					umean800_600 =\
						np.squeeze(trapz_int3d( group.ua.values[np.newaxis][np.newaxis].T,\
						group.p.values[np.newaxis][np.newaxis].T, \
						np.array((group.p <= 800) & \
						(group.p >= 600) )[np.newaxis][np.newaxis].T,)\
						.data)
					vmean800_600 =\
						np.squeeze(trapz_int3d( group.va.values[np.newaxis][np.newaxis].T,\
						group.p.values[np.newaxis][np.newaxis].T, \
						np.array((group.p <= 800) & \
						(group.p >= 600) )[np.newaxis][np.newaxis].T,)\
						.data)

					#Non pressure weighted
					#umean01 = group[(group.z - terrain) <= 1000]["ua"].mean()
					#vmean01 = group[(group.z - terrain) <= 1000]["va"].mean()
					#umean800_600 = group[((group.p - terrain) <= 800) & \
					#	((group.p - terrain) >= 600)]["ua"].mean()
					#vmean800_600 = group[((group.p - terrain) <= 800) & \
					#	((group.p - terrain) >= 600)]["va"].mean()
		
					Umean01 = np.sqrt( umean01**2 + vmean01**2 )
					Umean06 = np.sqrt( umean06**2 + vmean06**2 )
					Umean800_600 = np.sqrt( umean800_600**2 + vmean800_600**2 )
					u0 = np.interp([0], group.z - terrain, group.ua)[0]
					v0 = np.interp([0], group.z - terrain, group.va)[0]
					u6 = np.interp([6000], group.z - terrain, group.ua)[0]
					v6 = np.interp([6000], group.z - terrain, group.va)[0]
					s06 = np.sqrt(np.square(u6-u0)+np.square(v6-v0))
					ta850 = np.interp([850], np.flip(group.p) , np.flip(group.ta))[0]
					ta500 = np.interp([500], np.flip(group.p) , np.flip(group.ta))[0]
					v_totals = ta850 - ta500
					dp850 = np.interp([850], np.flip(group.p) , np.flip(group.dp))[0]
					c_totals = dp850 - ta500
					t_totals = c_totals + v_totals
					z700 = np.interp([700], np.flip(group.p) , np.flip(group.z))
					z500 = np.interp([500], np.flip(group.p) , np.flip(group.z))
					lr03 = -((np.interp([3000], (group.z)-terrain , (group.ta))[0]) - (np.interp([0], (group.z)-terrain , (group.ta))[0])) /\
						((3000-0)/1000)
					lr13 = -((np.interp([3000], (group.z)-terrain , (group.ta))[0]) - (np.interp([1000], (group.z)-terrain , (group.ta))[0])) /\
						((3000-1000)/1000)
					lr700_500 = -((np.interp([z500], (group.z)-terrain , (group.ta))[0]) - (np.interp([z700], (group.z)-terrain , (group.ta))[0])) /\
						((z500-z700)/1000)
					rhmin13 = np.min(rh[((group.z)-terrain >= 1000) & ((group.z)-terrain <= 3000)])
					q_melting = np.interp([0], np.flip(group.ta), np.flip(group.q))[0] * 1000
					mu_cape = np.max(cape)
					mu_cin = cin[np.nanargmax(cape)]
					dcape = np.squeeze(get_dcape(group.p.values[np.newaxis][np.newaxis].T, \
						group.ta.values[np.newaxis][np.newaxis].T, \
						group.q.values[np.newaxis][np.newaxis].T, \
						group.z.values[np.newaxis][np.newaxis].T, \
						group.p.iloc[0], p_lvl=True, p=group.p.values, sfc_included=True))
					thetae = mpcalc.equivalent_potential_temperature(\
						group.p.values * units.units.hectopascal, \
						group.ta.values * units.units.degC, \
						group.dp.values * units.units.degC)
					#thetae[abs(group.p - group.p.iloc[0]) > 400] = np.nan
					thetae[(abs(group.p - group.p.iloc[0]) > 400) | (abs(group.p - group.p.iloc[0]) < 50)] = np.nan
					try:
						dcape = dcape[np.nanargmin(thetae)]
					except:
						dcape = 0
					if dcape < 0:
						dcape = 0
					dp700 = np.interp([700], np.flip(group.p) , np.flip(group.dp))[0]
					ta700 = np.interp([700], np.flip(group.p) , np.flip(group.ta))[0]
					k_index = ta850 - ta500 + (ta850 - dp850) - (ta700-dp700)

					ue = np.interp([np.nanmin(eff_hgt)], group.z - terrain, group.ua)[0]
					ve = np.interp([np.nanmin(eff_hgt)], group.z - terrain, group.va)[0]
					um = np.interp([(el[np.nanargmax(cape)] * 0.5)], group.z - terrain, group.ua)[0]
					vm = np.interp([(el[np.nanargmax(cape)] * 0.5)], group.z - terrain, group.va)[0]
					ebwd = np.sqrt(np.square(um-ue)+np.square(vm-ve))

					#srhe
					us6 = u6 - u0
					vs6 = v6 - v0
					tmp = 7.5 / (np.sqrt(np.square(us6) + np.square(vs6)))
					u_storm_right = umean06 + (tmp * vs6)
					v_storm_right = vmean06 - (tmp * us6)
					u_storm_left = umean06 - (tmp * vs6)
					v_storm_left = vmean06 + (tmp * us6)
					hgt = group.z.values - terrain
					u = group.ua.values
					v = group.va.values
					hgt_bot = np.nanmin(eff_hgt,axis=0)
					hgt_top = np.nanmax(eff_hgt,axis=0)                                        
					u_ma = np.ma.masked_where(np.squeeze((hgt < hgt_bot) | (hgt > hgt_top) | (np.isnan(u)) | \
							(np.isnan(hgt_bot)) | (np.isnan(hgt_top)) | (np.isnan(hgt))), u)
					v_ma = np.ma.masked_where(np.squeeze((hgt < hgt_bot) | (hgt > hgt_top) | (np.isnan(v)) | \
							(np.isnan(hgt_bot)) | (np.isnan(hgt_top)) | (np.isnan(hgt))) , v)
					sru_left = u_ma - u_storm_left
					srv_left = v_ma - v_storm_left
					layers_left = (sru_left[1:] * srv_left[:-1]) - (sru_left[:-1] * srv_left[1:])
					srhe_left = abs(np.sum(layers_left))

					###
					ebwd = np.where(np.isnan(ebwd), 0, ebwd)
					srhe_left = np.where(np.isnan(srhe_left), 0, srhe_left)
					bdsd = np.squeeze(1 / ( 1 + np.exp( -(
						ebwd * 6.1e-2
						+ Umean800_600 * 1.5e-1
						+ lr13 * 9.4e-1
						+ rhmin13 * 3.9e-2
						+ srhe_left * 1.7e-2
						+ q_melting * 3.8e-1
						+ eff_lcl * 4.7e-4
						- 1.3e+1) ) ) )
					eff_sherb = np.squeeze((ebwd / 27.) * (lr03 / 5.2) * (lr700_500 / 5.6))
					dcp = (dcape / 980.) * (mu_cape / 2000.) * ( (s06*1.944) / 20.) * ((Umean06*1.944) / 16.)

				t =  (group.date.iloc[0].hour + group.date.iloc[0].minute/60.)

				if code == "sharppy":
					df = pd.concat([ df, \
						pd.DataFrame({"stn_id":group.stn_id.iloc[0],\
							"mu_cape":mu_parcel.bplus, \
							"ml_cape":ml_parcel.bplus, "mu_cin":abs(mu_parcel.bminus),\
							"ml_cin":abs(ml_parcel.bminus),"s06":s06,\
							"Umean800_600":Umean800_600, "k_index":params.k_index(prof),\
							"Umean01":Umean01,"t_totals":t_totals, "ml_el":ml_el,\
							"dcape":dcape}, \
							index=[group.date.iloc[0].replace(hour=int(t), minute=0)])],\
						axis=0)
				elif code == "wrfpython":
					df = pd.concat([ df, \
						pd.DataFrame({"stn_id":group.stn_id.iloc[0],"mu_cape":mu_cape, \
							"ml_cape":ml_cape, "mu_cin":mu_cin,\
							"ml_cin":ml_cin,"s06":s06,\
							"Umean800_600":Umean800_600, "k_index":k_index ,\
							"Umean01":Umean01,"t_totals":t_totals, "ml_el":ml_el,\
							"dcape":dcape, "bdsd":bdsd, "dcp":dcp, "eff_sherb":eff_sherb}, \
							index=[group.date.iloc[0].replace(hour=int(t), minute=0)])],\
						axis=0)

	df.to_pickle("/g/data/eg3/ab4502/ExtremeWind/points/"+fout+".pkl")

	return df

def read_aws_daily_sa(loc):

	#Read daily S.A. AWS data which has been downloaded for 1979-2017

	names = ["hm","stn_no","stn_name","lat","lon","date_str","wind_gust","quality",\
			"#"]
	if loc == "Adelaide AP":
		fname = "/g/data/eg3/ab4502/ExtremeWind/obs/aws/daily_1979_2017/DC02D_Data_023034_999999999565266.txt"
	elif loc == "Woomera":
		fname = "/g/data/eg3/ab4502/ExtremeWind/obs/aws/daily_1979_2017/DC02D_Data_016001_999999999565266.txt"
	elif loc == "Coober Pedy AP":
		fname = "/g/data/eg3/ab4502/ExtremeWind/obs/aws/daily_1979_2017/DC02D_Data_016090_999999999565266.txt"
	elif loc == "Port Augusta":
		fname = "/g/data/eg3/ab4502/ExtremeWind/obs/aws/daily_1979_2017/DC02D_Data_018201_999999999565266.txt"
	elif loc == "Clare HS":
		fname = "/g/data/eg3/ab4502/ExtremeWind/obs/aws/daily_1979_2017/DC02D_Data_021131_999999999565266.txt"
	elif loc == "Marree":
		fname = "/g/data/eg3/ab4502/ExtremeWind/obs/aws/daily_1979_2017/DC02D_Data_017126_999999999565266.txt"
	elif loc == "Munkora":
		fname = "/g/data/eg3/ab4502/ExtremeWind/obs/aws/daily_1979_2017/DC02D_Data_025557_999999999565266.txt"
	elif loc == "Robe":
		fname = "/g/data/eg3/ab4502/ExtremeWind/obs/aws/daily_1979_2017/DC02D_Data_026105_999999999565266.txt"
	elif loc == "Loxton":
		fname = "/g/data/eg3/ab4502/ExtremeWind/obs/aws/daily_1979_2017/DC02D_Data_024024_999999999565266.txt"
	elif loc == "Coonawarra":
		fname = "/g/data/eg3/ab4502/ExtremeWind/obs/aws/daily_1979_2017/DC02D_Data_026091_999999999565266.txt"
	elif loc == "Renmark":
		fname = "/g/data/eg3/ab4502/ExtremeWind/obs/aws/daily_1979_2017/DC02D_Data_024048_999999999565266.txt"
	elif loc == "Whyalla":
		fname = "/g/data/eg3/ab4502/ExtremeWind/obs/aws/daily_1979_2017/DC02D_Data_018120_999999999565266.txt"
	elif loc == "Padthaway South":
		fname = "/g/data/eg3/ab4502/ExtremeWind/obs/aws/daily_1979_2017/DC02D_Data_026100_999999999565266.txt"
	elif loc == "Nuriootpa":
		fname = "/g/data/eg3/ab4502/ExtremeWind/obs/aws/daily_1979_2017/DC02D_Data_023373_999999999565266.txt"
	elif loc == "Rayville Park":
		fname = "/g/data/eg3/ab4502/ExtremeWind/obs/aws/daily_1979_2017/DC02D_Data_021133_999999999565266.txt"
	elif loc == "Mount Gambier":
		fname = "/g/data/eg3/ab4502/ExtremeWind/obs/aws/daily_1979_2017/DC02D_Data_026021_999999999565266.txt"
	elif loc == "Naracoorte":
		fname = "/g/data/eg3/ab4502/ExtremeWind/obs/aws/daily_1979_2017/DC02D_Data_026099_999999999565266.txt"
	elif loc == "The Limestone":
		fname = "/g/data/eg3/ab4502/ExtremeWind/obs/aws/daily_1979_2017/DC02D_Data_026095_999999999565266.txt"
	elif loc == "Parafield":
		fname = "/g/data/eg3/ab4502/ExtremeWind/obs/aws/daily_1979_2017/DC02D_Data_023013_999999999565266.txt"
	elif loc == "Austin Plains":
		fname = "/g/data/eg3/ab4502/ExtremeWind/obs/aws/daily_1979_2017/DC02D_Data_025562_999999999565266.txt"
	elif loc == "Roseworthy":
		fname = "/g/data/eg3/ab4502/ExtremeWind/obs/aws/daily_1979_2017/DC02D_Data_023122_999999999565266.txt"
	elif loc == "Tarcoola":
		fname = "/g/data/eg3/ab4502/ExtremeWind/obs/aws/daily_1979_2017/DC02D_Data_016098_999999999565266.txt"
	elif loc == "Edinburgh":
		fname = "/g/data/eg3/ab4502/ExtremeWind/obs/aws/daily_1979_2017/DC02D_Data_023083_999999999565266.txt"
	elif loc == "Port Augusta Power Station":
		fname = "/g/data/eg3/ab4502/ExtremeWind/obs/aws/daily_1979_2017/DC02D_Data_019066_999999999565467.txt"
	
	aws = pd.read_csv(fname\
				,names=names,dtype={"wind_gust":float},\
				na_values={"wind_gust":'     '})
	aws["day"] = aws.date_str.str.slice(0,2).astype("int")
	aws["month"] = aws.date_str.str.slice(3,5).astype("int")
	aws["year"] = aws.date_str.str.slice(6,10).astype("int")
	aws_dt = []
	for i in np.arange(0,aws.shape[0]):
		aws_dt.append(dt.datetime((aws["year"][i]),(aws["month"][i]),\
			(aws["day"][i])))
	aws["date"] = aws_dt
	
	return aws

def read_convective_wind_gusts():

	#Creates a (dataframe of) "convective wind gust" data

	#Read daily AWS data which has been downloaded for 35 stations Australia wide, 1979-2017
	#Remove suspect, wrong or inconsistent quality controlled data
	#Converts to UTC, and adds lightning data, as well as STA reports

	#AWS daily max gusts are moved to the closest analysis time (00, 06, 12, 18 UTC), based on the 
	# "time of max gust" in UTC.
	# If the gust is exactly halfway between analysis times, the earlier time is taken.
	#
	#Lightning data (2005-2015) is then assigned as the maximum stroke count at the AWS station 
	# between the most recent and next analysis time. For example, if a maximum gust is recorded 
	# at 2:20 UTC, then the associated lightning count is taken as the maximum between the 00 and 
	# 06 UTC values. If the gust occurs at an analysis time, the current time and the previous time 
	# are considered. Lightning data has also already been summed over +/- 4 grid points (lat/lon) 
	# relative to each AWS station (around +/- 1 degree).

	#STA reports are considered if the report occurs within 0.5 degrees of lat and lon of an AWS station. 
	# Similar to the AWS data, the reports are assigned to the closest analysis time.

	#In addition, for AWS daily gusts data, a TC_affected flag is raised if a TC was present on the same day
	# within 2 degrees of latitude and longitude, from the BoM best track data 

	#An INCOMPLETE_MONTH flag is raised at a station if for a given month, less than 90% of days are recorded

	#Set csv column names
	names = ["record_id","stn_no","stn_name","locality", "state","lat","lon","height","date_str",\
		"wind_gust","quality","wind_dir", "wind_dir_quality", "max_gust_str_lt", \
		"max_gust_time_quality", "eof"]

	#Set csv read data types
	data_types = dict(record_id=str, stn_no=int, stn_name=str, locality=str, state=str, lat=float, lon=float,\
				height=str, date_str=str, wind_gust=float, quality=str, \
				wind_dir=str, wind_dir_quality=str, max_gust_str_lt=str,\
				max_gust_time_quality=str,\
				eof=str)

	renames = aus_stn_info()

	#Load csv file
	print("LOADING TEXT FILE")
	#f = "/g/data/eg3/ab4502/ExtremeWind/obs/aws/daily_aus_full/DC02D_Data_999999999643799.txt"
	f = "/g/data/eg3/ab4502/ExtremeWind/obs/aws/daily_aus_full/DC02D_Data_999999999720188.txt"
	df = pd.read_csv(f, names=names, dtype=data_types, \
		na_values={"wind_gust":'     ', "max_gust_str_lt":"    "})
	df = df.replace({"stn_name":renames})
	df["locality"] = df["locality"].str.strip()
	#df["wind_dir"] = df["wind_dir"].str.strip()
	df["wind_dir"] = pd.to_numeric(df.wind_dir, errors="coerce")
	df["stn_name"] = df["stn_name"].str.strip()

	#Get station info
	loc_id = list(df.stn_name.unique())
	lon_list = []
	lat_list = []
	points = []
	for loc in loc_id:
		lon = df[df.stn_name==loc]["lon"].unique()[0]
		lat = df[df.stn_name==loc]["lat"].unique()[0]
		lon_list.append(lon)
		lat_list.append(lat)
		points.append((lon,lat))
	stn_info = pd.DataFrame({"stn_name":loc_id, "lon":lon_list, "lat":lat_list})

	#Split the date column and convert to datetime object
	print("CONVERTING DATES TO DATETIME OBJECTS...")
	df["year"] = df.date_str.str.slice(6,10).astype("int")
	df["month"] = df.date_str.str.slice(3,5).astype("int")
	df["day_lt"] = df.date_str.str.slice(0,2).astype("int")
	df["daily_date_lt"] = pd.to_datetime({"year":df["year"],"month":df["month"],"day":df["day_lt"]})

	#For each month/station, see whether greater than 90% of the month has data. Create a new column with this
	# information
	df["incomplete_month"] = 1
	df["is_data"] = (~(df["wind_gust"].isna())).astype(int)
	missing = df.groupby(["stn_name","year","month"]).agg({"incomplete_month":"sum", "is_data":"sum"}).\
		reset_index()
	missing["perc"] = missing["is_data"] / missing["incomplete_month"]
	df = pd.merge(df, missing, on=["stn_name","year","month"])
	df["incomplete_month"] = np.where(df["perc"] < 0.9, 1, 0)
	df = df.drop(labels=["eof","incomplete_month_x","incomplete_month_y","is_data_y","is_data_x","perc"], axis=1)

	#Split the time of observed gust column and convert to datetime object. If a gust hasn't been recorded,
	# then assign the gust time the same as the date time for that day
	print("CONVERTING THE TIME OF MAX GUST INTO A DATETIME OBJECT...")
	df.loc[df["max_gust_str_lt"].isna(),"max_gust_str_lt"] = "0000"
	df["hour_lt"] = df.max_gust_str_lt.str.slice(0,2).astype("int")
	df["min_lt"] = df.max_gust_str_lt.str.slice(2,4).astype("int")
	df["gust_time_lt"] = pd.to_datetime({"year":df["year"], "month":df["month"], "day":df["day_lt"],\
			"hour":df["hour_lt"], "minute":df["min_lt"]})

	#Remove gusts where the quality flag is "Suspect", "Inconsistent (with other known information" or 
	# "Wrong". These represent 3, 0 and 1 gusts over 25 m/s. Out of the other flags for gusts over 25 m/s,
	# 80 don't have a flag, 72 have "N"ot been quality controlled, 2031 have been qualit"Y" controlled.
	df.loc[np.in1d(df.quality,np.array(["S","W","I"])),"wind_gust"] = np.nan
	
	#Set up TC dataframe. Only worry about 2005 onwards as this is when lightning is available
	tc_df = read_bom_shtc("/g/data/eg3/ab4502/ExtremeWind/shtc_2604.csv")
	tc_df = tc_df[(tc_df.datetime >= dt.datetime(2005,1,1)) & (tc_df.datetime<dt.datetime(2019,1,1))]
	#Loop through the TCs and assign a stn_name if it is within 500 km a TC on
	# that day
	tc_affected_date = []
	tc_affected_stn = []
	for i in np.arange(tc_df.shape[0]):
		tc_affected_stns = stn_info[latlon_dist(tc_df.iloc[i].lat, tc_df.iloc[i].lon,\
			stn_info.lat, stn_info.lon) < 500]
		if tc_affected_stns.shape[0] > 0:
			for j in np.arange(tc_affected_stns.shape[0]):
				tc_affected_date.append(dt.datetime( tc_df.iloc[i].datetime.year, \
					tc_df.iloc[i].datetime.month, tc_df.iloc[i].datetime.day))
				tc_affected_stn.append(tc_affected_stns.iloc[j].stn_name)
	tc_affected_df = pd.DataFrame({"stn_name":tc_affected_stn, "date":tc_affected_date})
	tc_affected_df = tc_affected_df.drop_duplicates(subset=["stn_name","date"])

	#For each gust from 2005-2015, get lightning information. Data is 6 hourly, and has been binned temporally
	# centred on the time stamp, and spatially into a 0.25 degree grid
	lightning = read_lightning("lightning_aus_50", rad = 50, dmax = False, loc_id=loc_id, points=points).\
			set_index(["date","loc_id"]).reset_index("loc_id")

	#Convert the AWS date-time object to UTC. Needs to be done separately for each station (different time zones)
	#Time zones from Olson dataset and includes DST info. Not sure if this includes sporadic changes in DST
	#practices e.g. trials of DST in W.A.
	#For days when there is not a unique time due to clocks being moved back, use the UTC offset for the 
	# next day
	print("\nCONVERTING FROM LT TO UTC...\n")
	temp_df2 = pd.DataFrame()
	tzs = {"WA":"Australia/Perth", "NT":"Australia/Darwin", "SA":"Australia/Adelaide", \
		"QLD":"Australia/Brisbane","NSW":"Australia/Sydney", "VIC":"Australia/Melbourne", \
		"TAS":"Australia/Hobart"}
	for l in df.stn_name.unique():
		temp_df = df.loc[df.stn_name==l].reset_index()
		temp_tz = tzs[temp_df.iloc[0].state.strip()]
		if l == "Giles":
			temp_tz = tzs["SA"] 	#Apparently Giles follows the S.A. time zone
		temp_df["gust_time_utc"] = pd.DatetimeIndex(temp_df.gust_time_lt, tz=temp_tz, \
						ambiguous='NaT').tz_convert("UTC")
		temp_df2 = pd.concat([temp_df2, temp_df], axis=0)
	#For times which aren't able to be converted (NaT) use the UTC offset from the next day
	temp_df2 = temp_df2.reset_index()
	nat_inds = np.array(temp_df2[temp_df2.gust_time_utc.isnull()].index)
	for i in np.arange(0,len(nat_inds)):
		utc_offset = (pd.DatetimeIndex(temp_df2.iloc[nat_inds[i]+1, \
			np.where(temp_df2.columns=="gust_time_lt")[0]]).tz_localize(None) - \
			pd.DatetimeIndex(temp_df2.iloc[nat_inds[i]+1, \
			np.where(temp_df2.columns=="gust_time_utc")[0]]).tz_localize(None))
		temp_df2.iloc[nat_inds[i], np.where(temp_df2.columns=="gust_time_utc")[0]] = \
			(temp_df2.iloc[nat_inds[i], np.where(temp_df2.columns=="gust_time_lt")[0]].values[0]\
			- utc_offset).tz_localize("UTC")
	df = pd.merge(df, temp_df2[["stn_name","daily_date_lt","gust_time_utc"]], on=["stn_name","daily_date_lt"])

	print("CONVERTING UTC GUST TIME BACK TO DAILY DATE")
	df["gust_time_utc"] = pd.DatetimeIndex(df["gust_time_utc"]).tz_convert(None)
	temp_time = pd.DatetimeIndex(df.gust_time_utc)
	df["daily_date_utc"] = pd.to_datetime({"year":temp_time.year, "month":temp_time.month, "day":temp_time.day})

	#Get the closest 6-hourly time. This will be used to compare with 6-hourly lightning data
	#Also, get the closest 1-hourly time for analysis
	print("\nCONVERTING FROM UTC TO CLOSEST 6-hourly TIME...")
	df["an_date"] = pd.DatetimeIndex(temp_time.round("6H"))
	df["hourly_time_utc"] = pd.DatetimeIndex(temp_time.round("1H"))
	df["hourly_floor_utc"] = pd.DatetimeIndex(temp_time.floor("1H"))
	df["hourly_ceil_utc"] = pd.DatetimeIndex(temp_time.ceil("1H"))

	#Insert TC flag from TC dataframe
	tc_affected_df["tc_affected"] = 1
	df = pd.merge(df, tc_affected_df, left_on=["stn_name","daily_date_utc"], \
		right_on=["stn_name","date"], how="left").drop(labels="date",axis=1)
	df["tc_affected"] = np.where(np.isnan(df["tc_affected"]), 0, 1)

	#Combine lightning and AWS data
	df = pd.merge(df, lightning[["lightning","loc_id"]], how="left", right_on=["date","loc_id"], \
		left_on=["an_date","stn_name"])

	#Add STA. Limit to one STA report per day by creating an "is_sta" flag, and masking duplicates
	print("ADDING STA REPORTS...")
	df = verify_sta(df)
	df["is_sta"] = (~(df.sta_date_ceil.isna()))*1
	i = df.duplicated(subset=["sta_daily_date","stn_name"])
	df.loc[i, "is_sta"] = 0
	
	#Drop where there is no lightning
	df = df.dropna(subset=["lightning"])

	#Save as pickle
	df[["stn_name","an_date","state","lat","lon","height","wind_gust","wind_dir","year","month",\
		"day_lt","hour_lt","min_lt","daily_date_utc","gust_time_lt","gust_time_utc", "hourly_time_utc",\
		"hourly_floor_utc","hourly_ceil_utc","tc_affected","lightning","incomplete_month",\
		"sta_wind","sta_wind_id","sta_date","sta_date_floor","sta_date_ceil","sta_daily_date","is_sta"]].\
		to_pickle("/g/data/eg3/ab4502/ExtremeWind/obs/aws/convective_wind_gust_aus_2005_2018.pkl")

	return df

def read_aws_half_hourly():

	#Read AWS half hourly data

	#Set csv column names
	names = ["record_id","stn_no","stn_name","locality", "state","lat","lon","height","date_str",\
		"wind_dir","wind_dir_quality","wind_gust","quality","aws_flag","eof"]

	#Dict to map station names to
	renames = {'ALICE SPRINGS AIRPORT                   ':"Alice Springs",\
			'GILES METEOROLOGICAL OFFICE             ':"Giles",\
			'COBAR MO                                ':"Cobar",\
			'AMBERLEY AMO                            ':"Amberley",\
			'SYDNEY AIRPORT AMO                      ':"Sydney",\
			'MELBOURNE AIRPORT                       ':"Melbourne",\
			'MACKAY M.O                              ':"Mackay",\
			'WEIPA AERO                              ':"Weipa",\
			'MOUNT ISA AERO                          ':"Mount Isa",\
			'ESPERANCE                               ':"Esperance",\
			'ADELAIDE AIRPORT                        ':"Adelaide",\
			'CHARLEVILLE AERO                        ':"Charleville",\
			'CEDUNA AMO                              ':"Ceduna",\
			'OAKEY AERO                              ':"Oakey",\
			'WOOMERA AERODROME                       ':"Woomera",\
			'TENNANT CREEK AIRPORT                   ':"Tennant Creek",\
			'GOVE AIRPORT                            ':"Gove",\
			'COFFS HARBOUR MO                        ':"Coffs Harbour",\
			'MEEKATHARRA AIRPORT                     ':"Meekatharra",\
			'HALLS CREEK METEOROLOGICAL OFFICE       ':"Halls Creek",\
			'ROCKHAMPTON AERO                        ':"Rockhampton",\
			'MOUNT GAMBIER AERO                      ':"Mount Gambier",\
			'PERTH AIRPORT                           ':"Perth",\
			'WILLIAMTOWN RAAF                        ':"Williamtown",\
			'CARNARVON AIRPORT                       ':"Carnarvon",\
			'KALGOORLIE-BOULDER AIRPORT              ':"Kalgoorlie",\
			'DARWIN AIRPORT                          ':"Darwin",\
			'CAIRNS AERO                             ':"Cairns",\
			'MILDURA AIRPORT                         ':"Mildura",\
			'WAGGA WAGGA AMO                         ':"Wagga Wagga",\
			'BROOME AIRPORT                          ':"Broome",\
			'EAST SALE                               ':"East Sale",\
			'TOWNSVILLE AERO                         ':"Townsville",\
			'HOBART (ELLERSLIE ROAD)                 ':"Hobart",\
			'PORT HEDLAND AIRPORT                    ':"Port Hedland"}

	#Set csv read data types
	data_types = dict(record_id=str, stn_no=int, stn_name=str, locality=str, state=str, lat=float, lon=float,\
				district=str, height=str, date_str=str, closest_date_str=str,\
				wind_gust=float, quality=str, \
				wind_dir=str, wind_dir_quality=str, max_gust_str_lt=str, max_gust_time_quality=str,\
				eof=str,aws_flag=str)

	#Read csv
	fname = "/g/data/eg3/ab4502/ExtremeWind/obs/aws/aws_half_hour_aus_2005_2015/HM01X_Data_999999999678486.txt"
	aws = pd.read_csv(fname,header=0,names=names)
	aws.loc[:,"wind_gust"] = pd.to_numeric(aws["wind_gust"], errors="coerce")
	aws["wind_dir"] = pd.to_numeric(aws.wind_dir, errors="coerce")
	aws["time"] =pd.to_datetime(aws.date_str, format="%d/%m/%Y %H:%M")
	aws = aws[["stn_name","time","wind_dir","wind_gust"]]
	aws = aws.replace({"stn_name":renames})
	aws["wind_gust"] = aws["wind_gust"] / 3.6

	return aws
	

def half_hourly_conv_events(loc,resample=False):
	#Read half-hourly AWS data which has been downloaded for 2005-2015, and construct a dataset
	# of convective events

	#Set csv column names
	names = ["record_id","stn_no","stn_name","locality", "state","lat","lon","district","height","date_str",\
		"closest_date_str","wind_dir","wind_dir_quality","wind_gust","quality","aws_flag","eof"]

	#Dict to map station names to
	renames = {'ALICE SPRINGS AIRPORT                   ':"Alice Springs",\
			'GILES METEOROLOGICAL OFFICE             ':"Giles",\
			'COBAR MO                                ':"Cobar",\
			'AMBERLEY AMO                            ':"Amberley",\
			'SYDNEY AIRPORT AMO                      ':"Sydney",\
			'MELBOURNE AIRPORT                       ':"Melbourne",\
			'MACKAY M.O                              ':"Mackay",\
			'WEIPA AERO                              ':"Weipa",\
			'MOUNT ISA AERO                          ':"Mount Isa",\
			'ESPERANCE                               ':"Esperance",\
			'ADELAIDE AIRPORT                        ':"Adelaide",\
			'CHARLEVILLE AERO                        ':"Charleville",\
			'CEDUNA AMO                              ':"Ceduna",\
			'OAKEY AERO                              ':"Oakey",\
			'WOOMERA AERODROME                       ':"Woomera",\
			'TENNANT CREEK AIRPORT                   ':"Tennant Creek",\
			'GOVE AIRPORT                            ':"Gove",\
			'COFFS HARBOUR MO                        ':"Coffs Harbour",\
			'MEEKATHARRA AIRPORT                     ':"Meekatharra",\
			'HALLS CREEK METEOROLOGICAL OFFICE       ':"Halls Creek",\
			'ROCKHAMPTON AERO                        ':"Rockhampton",\
			'MOUNT GAMBIER AERO                      ':"Mount Gambier",\
			'PERTH AIRPORT                           ':"Perth",\
			'WILLIAMTOWN RAAF                        ':"Williamtown",\
			'CARNARVON AIRPORT                       ':"Carnarvon",\
			'KALGOORLIE-BOULDER AIRPORT              ':"Kalgoorlie",\
			'DARWIN AIRPORT                          ':"Darwin",\
			'CAIRNS AERO                             ':"Cairns",\
			'MILDURA AIRPORT                         ':"Mildura",\
			'WAGGA WAGGA AMO                         ':"Wagga Wagga",\
			'BROOME AIRPORT                          ':"Broome",\
			'EAST SALE                               ':"East Sale",\
			'TOWNSVILLE AERO                         ':"Townsville",\
			'HOBART (ELLERSLIE ROAD)                 ':"Hobart",\
			'PORT HEDLAND AIRPORT                    ':"Port Hedland"}

	#Set csv read data types
	data_types = dict(record_id=str, stn_no=int, stn_name=str, locality=str, state=str, lat=float, lon=float,\
				district=str, height=str, date_str=str, closest_date_str=str,\
				wind_gust=float, quality=str, \
				wind_dir=str, wind_dir_quality=str, max_gust_str_lt=str, max_gust_time_quality=str,\
				eof=str,aws_flag=str)

	#Read csv
	fname = "/g/data/eg3/ab4502/ExtremeWind/obs/aws/aws_half_hour_aus_2005_2015/HM01X_Data_999999999678486.txt"
	aws = pd.read_csv(fname\
				,header=0,names=names,dtype={"wind_gust":float},\
				na_values={"wind_gust":'     '})
	aws = aws.replace({"stn_name":renames})
	aws.loc[:,"day"] = aws.date_str.str.slice(0,2).astype("int")
	aws.loc[:,"month"] = aws.date_str.str.slice(3,5).astype("int")
	aws.loc[:,"year"] = aws.date_str.str.slice(6,10).astype("int")
	aws.loc[:,"hour"] = aws.date_str.str.slice(11,13).astype("int")
	aws.loc[:,"minute"] = aws.date_str.str.slice(14,16).astype("int")
	aws_events = aws.query("wind_gust >= 25")
	aws_dt = []
	aws_daily_dt = []
	for i in np.arange(0,aws_events.shape[0]):
		aws_dt.append(dt.datetime((aws_events["year"].iloc[i]),(aws_events["month"].iloc[i]),\
			(aws_events["day"].iloc[i]),(aws_events["hour"].iloc[i]),(aws_events["minute"].iloc[i])))
		aws_daily_dt.append(dt.datetime((aws_events["year"].iloc[i]),(aws_events["month"].iloc[i]),\
			(aws_events["day"].iloc[i])))
	aws_events["date"] = aws_dt
	aws_events["daily_date"] = aws_daily_dt

	#if resample:
	#	aws = aws.resample("6H",on="date",base=3,\
	#		loffset=dt.timedelta(hours=3),\
	#		closed="right").max()
	
	#Check the half-hourly data by looking at the daily data
	daily = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/obs/aws/convective_wind_gust_aus_2005_2015.pkl").\
		reset_index()
	daily_strong_gusts = [daily.loc[(daily["level_1"]==aws_events.stn_name.iloc[i]) & \
			(daily["daily_date_utc"]==aws_events.daily_date.iloc[i]), "wind_gust"].max() \
			for i in np.arange(aws_events.shape[0]) ]
	daily_tc_affected = [daily.loc[(daily["level_1"]==aws_events.stn_name.iloc[i]) & \
			(daily["daily_date_utc"]==aws_events.daily_date.iloc[i]), "tc_affected"].max() \
			for i in np.arange(aws_events.shape[0]) ]
	aws_events.loc[:, "daily_gust"] = daily_strong_gusts
	aws_events.loc[:, "tc_affected"] = daily_tc_affected

	#Throw out half-hourly data where either
	#	- The gust is "TC affected" (TC within 2 degrees of lat-lon on the day)
	#	- Daily maximum wind gust is less than 20 (noting there is a few instances where the daily data
	#		is say, 24.7
	#	- Daily data is missing
	#On the above conditions, this removes 175 half-hourly points (from 525 total points, leaving 350 points)
	aws_events = aws_events[~((aws_events["daily_gust"] < 20) | (aws_events["tc_affected"]==1) | (aws_events["daily_gust"].isna()))]

	aws = aws.sort_values(["stn_name", "date_str"]).reset_index()
	aws_conv_stn_name = []
	aws_conv_date = []
	aws_conv_wind_gust = []
	for i in np.arange(aws_events.shape[0]):
		print(i)

		ind = aws[(aws.stn_name == aws_events.iloc[i].stn_name) & (aws.date_str == aws_events.iloc[i].date_str)].index[0]
		prev_speed = aws.iloc[ind-4:ind]["wind_gust"].values
		next_speed = aws.iloc[ind+1:ind+5]["wind_gust"].values
		prev_speed[aws.iloc[ind-4:ind]["stn_name"] != aws_events.iloc[i].stn_name] = np.nan
		next_speed[aws.iloc[ind+1:ind+5]["stn_name"] != aws_events.iloc[i].stn_name] = np.nan
		prev_avg = np.nanmax(prev_speed)
		next_avg = np.nanmax(next_speed)

		if ( ( aws_events.iloc[i]["wind_gust"] / prev_avg) >= 2.0 ) | ( ( aws_events.iloc[i]["wind_gust"] / next_avg) > 2.0 ):
			aws_conv_stn_name.append(aws_events.iloc[i]["stn_name"])
			aws_conv_date.append(aws_events.iloc[i]["date"])
			aws_conv_wind_gust.append(aws_events.iloc[i]["wind_gust"])
	aws_conv_events = pd.DataFrame({"stn_name":aws_conv_stn_name, "date":aws_conv_date, "wind_gust":aws_conv_wind_gust})

	return aws

def uv(df):
	ua = np.empty(df.shape[0])
	va = np.empty(df.shape[0])
	for i in np.arange(0,df.shape[0]):
		ua[i] = df["ws"][i] * math.sin(math.radians(df["wd"][i]))
		va[i] = df["ws"][i] * math.cos(math.radians(df["wd"][i]))
	df["ua"] = ua
	df["va"] = va
	return df

def get_min_var(var):
	lengths = np.empty(len(var))
	for i in np.arange(0,len(var)):
		lengths[i] = len(var[i])
	ind = np.argmin(lengths)
	return(ind)

def read_synoptic_wind_gusts(loc):
	# Read .xlsx file provided by JDH containing "synoptic wind gust events"
	#loc corresponds to the names of the sheets in the spreadsheet
	#e.g. " Adelaide AP" 

	xl_file = pd.ExcelFile("/g/data/eg3/ab4502/ExtremeWind/obs/jdh/sa_synoptic_gusts.xlsx")
	dfs = {sheet_name: xl_file.parse(sheet_name,header=3,skip_footer=5,skiprows=[4]) \
			for sheet_name in xl_file.sheet_names}
	ad_df = dfs[loc]
	ad_df_dates = []
	for i in np.arange(0,ad_df.shape[0]):
		string = str(ad_df["Year"][i])+"-"+str(ad_df["Month"][i])+"-"+str(ad_df["Day"][i])+\
			" "+str(ad_df["Time"][i])
		ad_df_dates.append(dt.datetime.strptime(string,"%Y-%m-%d %H:%M"))
	ad_df["dates"] = ad_df_dates
	ad_df["dates_utc"] = [x - dt.timedelta(hours=10,minutes=30) for x in ad_df_dates]
	
	return ad_df

def read_non_synoptic_wind_gusts():
	# Read .xlsx file provided by JDH containing "non-synoptic wind gust events"

	xl_file = pd.ExcelFile("/g/data/eg3/ab4502/ExtremeWind/obs/jdh/sa_non_synoptic_gusts.xlsx")
	df = xl_file.parse(header=3,skiprows=[4]).reset_index()
	df_dates = []
	for i in np.arange(0,df.shape[0]):
		if type(df["Date"][i]) == dt.datetime:
			df_dates.append(df["Date"][i])
		else:
			df_dates.append(dt.datetime.strptime(df["Date"][i],"%d/%m/%Y"))
	df["dates"] = df_dates

	#Add gusts 2-4 to dataframe as extra rows
	ind1 = ~(df["station.1"].isna())
	ind2 = ~(df["station.2"].isna())
	ind3 = ~(df["station.3"].isna())
	ind4 = ~(df["station.4"].isna())
	df0 = df[["gust (m/s)","direction (deg.)","station","dates"]]
	df1 = df[["gust (m/s).1","direction (deg.).1","station.1","dates"]][ind1]
	df2 = df[["gust (m/s).2","direction (deg.).2","station.2","dates"]][ind2]
	df3 = df[["gust (m/s).3","direction (deg.).3","station.3","dates"]][ind3]
	df4 = df[["gust (m/s).4","direction (deg.).4","station.4","dates"]][ind4]
	df1.columns = df2.columns = df3.columns = df4.columns = df0.columns
	df_full = pd.concat([df0,df1,df2,df3,df4])

	#Fix data points for which day and month are the wrong way around in JDH data
	#Checked for Woomera, Adelaide AP and Port Augusta
	#NOTE THAT 28-11-2011 at Port Augusta is dubious (daily max AWS = 17 m/s)
	#NOTE THAT 30-12-2012 at Renmark is dubious (daily max AWS = 12.3 m/s)
	df_full.dates.loc[df_full["dates"]==dt.datetime(2010,7,12)] = dt.datetime(2010,12,7)
	df_full.dates.loc[df_full["dates"]==dt.datetime(2014,6,10)] = dt.datetime(2014,10,6)
	df_full.dates.loc[df_full["dates"]==dt.datetime(2015,7,12)] = dt.datetime(2015,12,7)

	df_full.dates.loc[df_full["dates"]==dt.datetime(2004,12,10)] = dt.datetime(2004,10,12)
	df_full.dates.loc[df_full["dates"]==dt.datetime(2005,10,6)] = dt.datetime(2005,6,10)
	df_full.dates.loc[df_full["dates"]==dt.datetime(2008,5,12)] = dt.datetime(2008,12,5)
	df_full.dates.loc[df_full["dates"]==dt.datetime(2010,1,12)] = dt.datetime(2010,12,1)
	df_full.dates.loc[df_full["dates"]==dt.datetime(2011,9,11)] = dt.datetime(2011,11,9)
	df_full.dates.loc[df_full["dates"]==dt.datetime(2012,5,9)] = dt.datetime(2012,9,5)
	df_full.dates.loc[df_full["dates"]==dt.datetime(2015,7,1)] = dt.datetime(2015,1,7)
	df_full.dates.loc[df_full["dates"]==dt.datetime(2016,12,7)] = dt.datetime(2016,7,12)

	df_full.dates.loc[df_full["dates"]==dt.datetime(1983,2,3)] = dt.datetime(1983,3,2)
	df_full.dates.loc[df_full["dates"]==dt.datetime(1986,6,12)] = dt.datetime(1986,12,6)
	df_full.dates.loc[df_full["dates"]==dt.datetime(1989,1,12)] = dt.datetime(1989,12,1)
	df_full.dates.loc[df_full["dates"]==dt.datetime(1992,12,8)] = dt.datetime(1992,8,12)
	df_full.dates.loc[df_full["dates"]==dt.datetime(1996,12,9)] = dt.datetime(1996,9,12)
	df_full.dates.loc[df_full["dates"]==dt.datetime(1998,5,11)] = dt.datetime(1998,11,5)
	df_full.dates.loc[df_full["dates"]==dt.datetime(2002,5,8)] = dt.datetime(2002,8,5)

	#Remove points which have been identified as incorrect
	wrong_points = [dt.datetime(1983,12,24),dt.datetime(2001,11,17)]
	df_full = df_full[~(np.in1d(np.array(df_full.dates),np.array(wrong_points).astype(np.datetime64)))]

	#THINK JDH DATA MAY ALREADY BE IN UTC
	#df_full["dates_utc_start"] = [x - dt.timedelta(hours=10,minutes=30) \
	#		for x in df_full["dates"]]
	df_full.index = df_full["dates"]
	df_full = df_full.sort_index()
	
	#Rename a few stations
	df_full.station.loc[df_full.station=="Adelaide Airport"] = "Adelaide AP"
	df_full.station.loc[df_full.station=="Mt Gambier"] = "Mount Gambier"
	df_full.station.loc[df_full.station=="Coober Pedy"] = "Coober Pedy AP"
	df_full.station.loc[df_full.station=="Port Augusta Power Station"] = "Port Augusta"

	return df_full

def read_lightning(fname, rad = 100, dmax = False, loc_id=None, points=None):
	#Read Andrew Dowdy's lightning dataset for a list of points.
	if (loc_id is None) | (points is None):
		loc_id,points = get_aus_stn_info()
	
	path = "/g/data/eg3/ab4502/ExtremeWind/ad_data/lightning/"
	years = np.arange(2005,2019)
	df = pd.DataFrame()
	for year in np.arange(0,len(years)):
		print("READING LIGHTNING DATA FOR YEAR "+str(years[year]))
		f = nc.Dataset(path+"lightning_Australasia0.250000degree_6.00000hr_"+\
			str(years[year])+".nc")
		if year == 0:
			x,y = np.meshgrid(f.variables["lon"][:], f.variables["lat"][:])
		lightning_year = f.variables["Lightning_observed"][:]
		times = f.variables["time"][:]
		if years[year] >= 2016:
			time_dt = [dt.datetime(years[year],1,1,0,0,0) + dt.timedelta(hours=6*int(times[i])) \
				for i in np.arange(0,len(times))]
		else:
			time_dt = [dt.datetime(years[year],1,1,0,0,0) + dt.timedelta(hours=int(times[i])) \
				for i in np.arange(0,len(times))]
		for p in np.arange(0,len(points)):
			dist = latlon_dist(points[p][1], points[p][0], y, x)
			mask = np.zeros(lightning_year.shape, dtype=bool)
			#Sum all lightning counts within "rad" (in km) 
			mask[:,:,:] = (dist > rad)[np.newaxis,:,:]
			temp_lightning = np.ma.masked_where(mask, \
				lightning_year ).sum(axis=(1,2))
			
			temp_df = pd.DataFrame(\
				{"lightning":temp_lightning,\
				"loc_id":loc_id[p],"date":time_dt,"lon":points[p][0],\
				"lat":points[p][1]})
			df = df.append(temp_df)
	df.to_pickle("/g/data/eg3/ab4502/ExtremeWind/ad_data/"+fname+".pkl")

	if dmax:

		print("\n\n RESAMPLING TO DMAX...")
		df = df.set_index("date")
		df_daily = pd.DataFrame()
		for loc in np.unique(df.loc_id):
			print(loc)
			temp_df = pd.DataFrame(df[df.loc_id==loc][["lightning"]].resample("1D").max())
			temp_df["loc_id"] = loc
			df_daily = pd.concat([df_daily,temp_df])
		df_daily = df_daily.set_index("loc_id",append=True)

		df_daily.to_pickle("/g/data/eg3/ab4502/ExtremeWind/ad_data/lightning_aus_daily.pkl")

	return df

def load_lightning(domain="aus",daily=True,smoothing=True):
	#Load csv created by read_lightning
	#Domain can be "aus" or "sa_small"
	if domain == "aus":
		if smoothing:
			if daily:
				df = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/ad_data/lightning_aus_smoothed_daily.pkl")
			else:
				df = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/ad_data/lightning_aus_smoothed.pkl")
		else:
			if daily:
				df = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/ad_data/lightning_aus_daily.pkl")
			else:
				df = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/ad_data/lightning_aus.pkl")
	if domain == "aus_large":
		df = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/ad_data/lightning_572_stns_smoothed.pkl")
	elif domain == "sa_small":
		if smoothing:
			if daily:
				df = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/ad_data/lightning_sa_smoothed_daily.pkl")
			else:
				df = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/ad_data/lightning_sa_smoothed.pkl")
		else:
			if daily:
				df = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/ad_data/lightning_sa_daily.pkl")
			else:
				df = pd.read_pickle("/g/data/eg3/ab4502/ExtremeWind/ad_data/lightning_sa.pkl")
	return df

def load_wind_sa():
	#Load wind_SA.csv into a dataframe
	df = pd.read_csv("/g/data/eg3/ab4502/ExtremeWind/obs/wind_sa.csv",usecols=np.arange(0,12)\
		,header=0,skiprows=[966])
	#Convert date
	df["date"] = [dt.datetime.strptime(df["Date time commenced"][i],"%Y-%m-%d %H:%M:%S") \
			for i in np.arange(0,df.shape[0])]
	return df

def read_clim_ind(ind, seasons=[[3,4,5],[6,7,8],[9,10,11],[12,1,2]], names = ["MAM","JJA","SON","DJF"],years=np.arange(1979,2019)):

	#Create annual time series' for each season

	if ind == "nino34":
		#NINO3.4
		df = pd.read_table("/g/data/eg3/ab4502/ExtremeWind/clim_ind/nina34.anom.data",names=np.arange(0,13,1),\
			index_col=0,sep="  ",skiprows=[0,1,2],skipfooter=3,engine="python")
		df.loc[2019,9] = 26.68
		df.loc[:,9] = pd.to_numeric(df[9])
		time_series = pd.DataFrame(columns=np.append(names,"ANN"),index=years)
		for y in years:
			for s in np.arange(0,len(seasons)):
				if s == 3:	#IF NDJ
					mean = np.mean([df.loc[y,seasons[s][0]],df.loc[y+1,seasons[s][1]],\
					df.loc[y+1,seasons[s][2]]])
				else:
					mean = np.mean([df.loc[y,seasons[s][0]],df.loc[y,seasons[s][1]],\
					df.loc[y,seasons[s][2]]])
				time_series.loc[y,names[s]] = mean
				time_series.loc[y,"ANN"] = np.mean(df.loc[y,:])

	#DMI
	elif ind == "dmi":
		df = pd.read_table("/g/data/eg3/ab4502/ExtremeWind/clim_ind/dmi.txt",names=np.arange(0,13,1),\
			index_col=0,sep="    ",skiprows=[0,1,2],skipfooter=6,engine="python")
		time_series = pd.DataFrame(columns=np.append(names,"ANN"),index=years)
		for y in years:
			for s in np.arange(0,len(seasons)):
				if s == 3:	#IF NDJ
					mean = np.mean([df.loc[y,seasons[s][0]],df.loc[y+1,seasons[s][1]],\
					df.loc[y+1,seasons[s][2]]])
				else:
					mean = np.mean([df.loc[y,seasons[s][0]],df.loc[y,seasons[s][1]],\
					df.loc[y,seasons[s][2]]])
				time_series.loc[y,names[s]] = mean
				time_series.loc[y,"ANN"] = np.mean(df.loc[y,:])

	#SAM
	elif ind == "sam":
		df = pd.read_table("/g/data/eg3/ab4502/ExtremeWind/clim_ind/sam.txt",names=np.arange(0,13,1),\
			index_col=0,sep="  | ",engine="python")
		time_series = pd.DataFrame(columns=np.append(names,"ANN"),index=years)
		for y in years:
			for s in np.arange(0,len(seasons)):
				if s == 3:	#IF NDJ
					mean = np.mean([df.loc[y,seasons[s][0]],df.loc[y+1,seasons[s][1]],\
					df.loc[y+1,seasons[s][2]]])
				else:
					mean = np.mean([df.loc[y,seasons[s][0]],df.loc[y,seasons[s][1]],\
					df.loc[y,seasons[s][2]]])
				time_series.loc[y,names[s]] = mean
				time_series.loc[y,"ANN"] = np.mean(df.loc[y,:])
	elif ind == "mjo":
		mjo = pd.read_csv("/g/data/eg3/ab4502/ExtremeWind/clim_ind/mjo_rmm.txt", \
			header=1,  delim_whitespace=True,names=["year","month","day","rmm1","rmm2","phase",\
			"amplitude","a","b","c","d"],index_col=False).iloc[:,0:7]
		mjo.loc[:,"datetime"] = pd.to_datetime({"year":mjo.year,"month":mjo.month,"day":mjo.day})
		mjo = mjo.loc[(mjo["datetime"] >= dt.datetime(1979,1,1)) \
			& (mjo["datetime"]<dt.datetime(2019,1,1)),:]
		mjo.loc[:,"active"] = np.where((np.in1d(mjo["phase"],[4,5,6,7])) & (mjo["amplitude"] >= 1), 1, 0)
		mjo.loc[:,"inactive"] = np.where((np.in1d(mjo["phase"],[8,1,2,3])) & (mjo["amplitude"] >= 1), 1, 0)
		time_series = mjo.set_index("datetime")[["active","inactive"]]
	try:
		return time_series
	except:
		raise NameError("MUST BE ""sam"", ""nino34"", ""dmi"" or ""mjo""")

def read_bom_shtc(infile):
    """
    Read CSV file downloaded from http://www.bom.gov.au/cyclone/history/#db called shtc_<date downloaded mmdd>.csv
    """

    #print " INFO : Extracting data from site file %s" %(infile)

    #Read infile. Note that lines before header contains all named TCs in the data file. Will change if downloaded again with new TCs
    #Read only columns of Name, datetime, lat, lon, central pressure and max wind speed
    #usecols=[0,1,2,3,5,7,8,16,49,53]
    data = pd.read_csv(infile,error_bad_lines=False,header=62,low_memory = False)
    data = data[['NAME', 'DISTURBANCE_ID', 'TM', 'TYPE', 'SURFACE_CODE', 'LAT', 'LON','CENTRAL_PRES', 'MAX_WIND_SPD', 'MAX_WIND_GUST',\
		'MN_RADIUS_OUTER_ISOBAR', 'MN_RAD_OUT_ISOBAR_UNCERTAINTY', 'MN_RADIUS_GF_WIND','MAX_WIND_SPD_UNCERTAINTY', \
		'MAX_WIND_SPD_METHOD', 'MAX_WIND_GUST_METHOD', 'MAX_REP_WIND_SPD', 'MAX_REP_WIND_METHOD', 'COMMENT']]

    #Change time axis to datetime objects. Remove data where there is no time
    data_subset = data[data.TM != ' ']
    #data_subset['TM'] = pd.to_datetime(map(str,data_subset['TM'].tolist()),format='%Y-%m-%d %H:%M')    
    data_subset.TM = pd.to_datetime(data_subset.TM,format='%Y-%m-%d %H:%M')

    #Rename to be consistent with read_obs.read_obs
    data_subset = data_subset.rename(columns = {'NAME':'name','DISTURBANCE_ID':'storm_no','TM':'datetime','LAT':'lat','LON':'lon','CENTRAL_PRES':'pressure','MAX_WIND_SPD':'wind_mps', 'MAX_WIND_GUST':'gust_mps', 'SURFACE_CODE':'sfc_code','TYPE':'type'})

    #Change lat, lon to numeric
    data_subset['lat'] = pd.to_numeric(data_subset['lat'])
    data_subset['lon'] = pd.to_numeric(data_subset['lon'])
    data_subset['sfc_code'] = pd.to_numeric(data_subset['sfc_code'], errors="coerce")
    data_subset['gust_mps'] = pd.to_numeric(data_subset['gust_mps'], errors="coerce")

    #Create a column composed of storm number and storm year to create a unique storm_id
    def storm_id(row):
        return '{}_{}'.format(row['datetime'].year, row['storm_no'])

    data_subset['storm_id'] = data_subset.apply(storm_id, axis=1)


    return data_subset


def read_sta():

	#Read the "cleaned" STA wind report dataset. Cleaned means that some 
	# comments from the original dataset have been removed such that it can actually be
	# read in as a csv

	#Possibly need to screen for non-straight line winds and/or downbursts (e.g. tornadoes
	# but with no squall/burst, tropical cyclones). Also, there are ~6000 reports, so can afford
	# to be strict. E.g. remove reports without comments, reports without a wind speed, etc. 

	sta = pd.read_csv("/g/data/eg3/ab4502/ExtremeWind/obs/sta_wind_clean.csv",usecols=np.arange(0,8),\
			engine="python")

	temp_date = pd.DatetimeIndex( pd.to_datetime(sta["Date/Time"], format="%d/%m/%Y %H:%M") )
	sta["sta_date"] = temp_date
	sta["date_floor"] = temp_date.floor("1H")
	sta["date_ceil"] = temp_date.ceil("1H")
	sta["daily_date"] = pd.to_datetime({"year":temp_date.year, "month":temp_date.month, \
			"day":temp_date.day})

	sta["Wind ID"] = sta["Wind ID"].astype(str)
	sta["Comments"] = sta["Comments"].str.lower()
	sta["Max Gust speed"] = sta["Max Gust speed"] * 0.514444

	return sta


def verify_sta(df):

	#Take a dataframe of "events", which contains a date field, and add the Severe Thunderstorm
	# Archive (STA) from the BoM. Output a dataframe with an additional column, "sta_wind"

	#Comments are available in the csv, but are not read in 

	#The input dataframe must have columns indexed by "hourly_time_utc" and "stn_name"

	sta = read_sta()

	#Get station info
	loc_id = list(df.stn_name.unique())
	lon_list = []
	lat_list = []
	points = []
	for loc in loc_id:
		lon = df[df.stn_name==loc]["lon"].unique()[0]
		lat = df[df.stn_name==loc]["lat"].unique()[0]
		lon_list.append(lon)
		lat_list.append(lat)
		points.append((lon,lat))
	stn_info = pd.DataFrame({"stn_name":loc_id, "lon":lon_list, "lat":lat_list})
	
	#For each STA report, find station locations within 50 km (from the 
	# passed AWS dataframe). Assign the report with that station name. If there is more than 
	# one station, take the one with minimum distance
	sta_stn = []
	sta_wind_id = []
	sta_date = []
	sta_date_floor = []
	sta_date_ceil = []
	sta_daily_date = []
	sta_wind = []
	for i in np.arange(0,sta.shape[0]):
		
		dist = latlon_dist(sta.Latitude[i], sta.Longitude[i], stn_info.lat, stn_info.lon)

		if ( dist.min() <= 50 ):
			temp = stn_info[dist<=50]
			if dist[dist<=50].shape[0] > 1:
				print("MULTIPLE STATIONS")
				print(temp.stn_name.values)
			for j in np.arange(dist[dist<=50].shape[0]):
				sta_stn.append(temp.iloc[j].stn_name)
				sta_wind_id.append(sta["Wind ID"][i])
				sta_date.append(sta["sta_date"][i])
				sta_date_floor.append(sta["date_floor"][i])
				sta_date_ceil.append(sta["date_ceil"][i])
				sta_daily_date.append(sta["daily_date"][i])
				sta_wind.append(sta["Max Gust speed"][i])

	sta_stn = pd.DataFrame({"stn_name":sta_stn, "sta_wind":sta_wind, "sta_date_floor":sta_date_floor, "sta_date_ceil":sta_date_ceil, \
		"sta_date":sta_date,"sta_wind_id":sta_wind_id, "sta_daily_date":sta_daily_date}).\
		sort_values("sta_wind", ascending=False).\
		drop_duplicates(["stn_name","sta_daily_date"])

	df = pd.merge(df, sta_stn, how="left", left_on=["daily_date_utc","stn_name"], \
		right_on=["sta_daily_date","stn_name"])

	return df
			
def latlon_dist(lat, lon, lats, lons):

	#Calculate great circle distance (Harversine) between a lat lon point (lat, lon) and a list of lat lon
	# points (lats, lons)

	R = 6373.0

	lat1 = np.deg2rad(lat)
	lon1 = np.deg2rad(lon)
	lat2 = np.deg2rad(lats)
	lon2 = np.deg2rad(lons)

	dlon = lon2 - lon1
	dlat = lat2 - lat1

	a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
	c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

	return (R * c)

def aus_stn_info():

	#For the 35 Aus stations currently being used for convective wind gusts, get some info
	#Namely, the dictionary mapping renames from native data, and dictionary mapping stations to time zones

	#Dict to map station names to
	renames = {'ALICE SPRINGS AIRPORT                   ':"Alice Springs",\
			'GILES METEOROLOGICAL OFFICE             ':"Giles",\
			'COBAR MO                                ':"Cobar",\
			'AMBERLEY AMO                            ':"Amberley",\
			'SYDNEY AIRPORT AMO                      ':"Sydney",\
			'MELBOURNE AIRPORT                       ':"Melbourne",\
			'MACKAY M.O                              ':"Mackay",\
			'WEIPA AERO                              ':"Weipa",\
			'MOUNT ISA AERO                          ':"Mount Isa",\
			'ESPERANCE                               ':"Esperance",\
			'ADELAIDE AIRPORT                        ':"Adelaide",\
			'CHARLEVILLE AERO                        ':"Charleville",\
			'CEDUNA AMO                              ':"Ceduna",\
			'OAKEY AERO                              ':"Oakey",\
			'WOOMERA AERODROME                       ':"Woomera",\
			'TENNANT CREEK AIRPORT                   ':"Tennant Creek",\
			'GOVE AIRPORT                            ':"Gove",\
			'COFFS HARBOUR MO                        ':"Coffs Harbour",\
			'MEEKATHARRA AIRPORT                     ':"Meekatharra",\
			'HALLS CREEK METEOROLOGICAL OFFICE       ':"Halls Creek",\
			'ROCKHAMPTON AERO                        ':"Rockhampton",\
			'MOUNT GAMBIER AERO                      ':"Mount Gambier",\
			'PERTH AIRPORT                           ':"Perth",\
			'WILLIAMTOWN RAAF                        ':"Williamtown",\
			'CARNARVON AIRPORT                       ':"Carnarvon",\
			'KALGOORLIE-BOULDER AIRPORT              ':"Kalgoorlie",\
			'DARWIN AIRPORT                          ':"Darwin",\
			'CAIRNS AERO                             ':"Cairns",\
			'MILDURA AIRPORT                         ':"Mildura",\
			'WAGGA WAGGA AMO                         ':"Wagga Wagga",\
			'BROOME AIRPORT                          ':"Broome",\
			'EAST SALE                               ':"East Sale",\
			'TOWNSVILLE AERO                         ':"Townsville",\
			'HOBART (ELLERSLIE ROAD)                 ':"Hobart",\
			'PORT HEDLAND AIRPORT                    ':"Port Hedland"}

	return renames

if __name__ == "__main__":

	#df = calc_obs()
	
	#df.to_csv("/home/548/ab4502/working/ExtremeWind/data_obs_"+\
	#	"Nov2012"+".csv",float_format="%.3f")
	
	#read_convective_wind_gusts()

	#df = read_lightning(False)
	#read_aws_daily_aus()
	df = read_upperair_obs(dt.datetime(2005,1,1),dt.datetime(2018,12,31),\
	    "UA_wrfpython", "wrfpython")
	#df = read_upperair_obs(dt.datetime(2005,1,1),dt.datetime(2018,12,31), \
	 #   "UA_sharppy", "sharppy")

