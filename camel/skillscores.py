"""
Copyright © 2023 United States Government as represented by the 
Administrator of the National Aeronautics and Space Administration.  
No copyright is claimed in the United States under Title 17, U.S. Code. 
All Other Rights Reserved.

@author: remullinix
"""

import pandas as pd
import numpy as np
from datetime import datetime,timedelta,timezone

def interpolate_dataframe_to_base(frame, baseFrame, method):
	'''
    Returns interpolated panda data frame.
    Any method compatible with pandas.DataFrame.interpolate

            Parameters:
                    frame (DataFrame): dataframe with datatime as index
                    baseFrame (DataFrame): dataframe with datatime as index
		    		method (string): "linear", "nearest", others

            Returns:
                    interpolated_frame (DataFrame): dataframe with datatime as index
    '''
    
	baseTimes = baseFrame.index.values
	frameTimes = frame.index.values
	baseAndFrameTimes = pd.to_datetime(np.unique(np.hstack((baseTimes, frameTimes))), utc=True)
	interpolated_frame = frame.reindex(baseAndFrameTimes).sort_index().interpolate(method, limit_area = 'inside')
	interpolated_frame = interpolated_frame.reindex(baseFrame.index)
	return interpolated_frame


def remove_gaps(modDF, obsDF, maxSeconds):
	'''
    Returns observation dataframe with gaps removed.

            Parameters:
                    modDF (DataFrame): model dataframe with datatime as index
                    obsDF (DataFrame): observation dataframe with datatime as index
		    		maxSeconds (int): number of seconds allowed before removing gap

            Returns:
                    obsDF (DataFrame): observation dataframe with gaps removed
    '''
	timeDF = pd.DataFrame(index=modDF.index)
	timeDF['nearest'] = pd.to_numeric(modDF.index.values)

	modTimes = modDF.index.values
	obsTimes = obsDF.index.values
	obsModTimes = pd.to_datetime(np.unique(np.hstack((obsTimes, modTimes))), utc=True)
	timeDF = timeDF.reindex(obsModTimes).sort_index().interpolate("nearest")

	timeDF['timestamp'] = pd.to_numeric(timeDF.index.values)
	timeDF['diff'] = (timeDF.timestamp - timeDF.nearest)/1000000000

	timeDF = timeDF.reindex(obsDF.index).sort_index()
	timeDF = timeDF.dropna()


	timeDF = timeDF.loc[abs(timeDF['diff']) <= maxSeconds ]

	return obsDF.reindex(timeDF.index)

def timestamp2datetime(timestamp,format="%Y-%m-%dT%H:%M:%SZ"):
	datetimestamp = datetime.strptime(timestamp,format).replace(tzinfo=timezone.utc)
	return datetimestamp

def dBH_dt(df,compute_derivatives=False):
        '''
        Computes the magnititude of time derivatives of horizontal magnetic perturbations.
        This is the parameter used in Pulkkinen et al (2013) that is compared against threshold values of 0.3 0.7, 1.1 and 1.5 nT/s.
        Parameters: dataframe with tiemstampos, DeltaB components (Down, North, East) and (optional) time derivatives of the DeltaB components
        Returns:
                df_dbH_dt - dataframe with timestamps and dBH_dt.
                      Timestamps are taken from existing dataframe if derivatives already exist.
                      Time stamps are calculated as the average of adjacent time stamps if
                        derivatives are calculated from raw DeltaB data.
        '''
        df_size = df.shape[0]
        if "Derivative_DeltaB_East" in df.columns and "Derivative_DeltaB_North" in df.columns and not compute_derivatives:
                # if dataframes have the time derivatives already calculated
                df_dBE_dt = np.array(df['Derivative_DeltaB_East'])
                df_dBN_dt = np.array(df['Derivative_DeltaB_North'])
                df_timestamp = np.array(df["timestamp"])
                df_datetimes = np.array([timestamp2datetime(timestamp) for timestamp in df_timestamp])
        else:
                # dataframes with the raw DeltaB (magnetic perturbation) data only
                df_timestamp = np.array(df['timestamp'])
                df_datetimes = np.array([timestamp2datetime(timestamp) for timestamp in df_timestamp])
                df_datetime_deltas = df_datetimes[1:df_size] - df_datetimes[0:df_size-1]
                dt_seconds = np.array([df_datetime_deltas[i].total_seconds() for i in range(df_size-1)])
                df_dBN_dt = (np.array(df['DeltaB_North'][1:df_size]) 
                     - np.array(df['DeltaB_North'][0:df_size-1]))/dt_seconds
                df_dBE_dt = (np.array(df['DeltaB_East'][1:df_size]) 
                     - np.array(df['DeltaB_East'][0:df_size-1]))/dt_seconds
                df_datetimes=df_datetimes[0:df_size-1] + df_datetime_deltas/2
        df_dbH_dt = pd.DataFrame(np.sqrt(df_dBN_dt*df_dBN_dt+df_dBE_dt*df_dBE_dt),
	                         columns=['dBH_dt'], index=df_datetimes)
        return(df_dbH_dt)


def rmsScore(modDF, obsDF):
	'''
    Computes Root Mean Square error based on modDF and obsDF.  
	Dataframes must be same size and have matching indicies.
    
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datatime as index
                    obsDF (pandas.DataFrame): dataframe with datatime as index
		    
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	obsDF = obsDF.reindex(modDF.index)
	squaredError = (modDF - obsDF)**2
	rms = squaredError.mean()**0.5
	return rms

def maeScore(modDF, obsDF):
	'''
    Computes Mean Absolute Error based on modDF and obsDF.  
	Dataframes must be same size and have matching indicies.
    
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datatime as index
                    obsDF (pandas.DataFrame): dataframe with datatime as index
		    
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	obsDF = obsDF.reindex(modDF.index)
	absError = abs(modDF - obsDF)
	mae = absError.mean()
	return mae

def meScore(modDF, obsDF):
	'''
    Computes Mean Error based on modDF and obsDF.  
	Dataframes must be same size and have matching indicies.
    
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datatime as index
                    obsDF (pandas.DataFrame): dataframe with datatime as index
		    
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	obsDF = obsDF.reindex(modDF.index)
	error = modDF - obsDF
	me = error.mean()
	return me

def meanSquaredError(modDF, obsDF):
	'''
    Computes Mean Squared Error based on modDF and obsDF.  
	Dataframes must be same size and have matching indicies.
    
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datatime as index
                    obsDF (pandas.DataFrame): dataframe with datatime as index
		    
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	obsDF = obsDF.reindex(modDF.index)
	mse = ((obsDF - modDF) ** 2).mean()
	return mse

def ccScore(modDF, obsDF):
	'''
    Computes Correlation Coefficient based on modDF and obsDF.  
	Dataframes must be same size and have matching indicies.
    
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datatime as index
                    obsDF (pandas.DataFrame): dataframe with datatime as index
		    
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	obsDF = obsDF.reindex(modDF.index)
	modelMean = modDF.mean()
	obsMean = obsDF.mean()
	sigmaObs = sigma(obsDF)
	sigmaModel = sigma(modDF)
	diffMeanSquare = (obsDF - obsMean) * (modDF - modelMean)
	diffMeanSquareSum = diffMeanSquare.sum()
	N = diffMeanSquare.shape[0]
	score = (diffMeanSquareSum / (N-1)) / (sigmaObs * sigmaModel)
	return score

def peScore(modDF, obsDF):
	'''
    Computes Prediction Efficiency based on modDF and obsDF.  
	Dataframes must be same size and have matching indicies.
    
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datatime as index
                    obsDF (pandas.DataFrame): dataframe with datatime as index
		    
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	obsDF = obsDF.reindex(modDF.index)
	sigmaObsSquared = sigma(obsDF)**2
	diff = (obsDF - modDF)*(obsDF - modDF) / sigmaObsSquared
	return 1 - diff.mean();

def sspbScore(modDF, obsDF):
	'''
    Computes Symmetric Signed Percentage Bias based on modDF and obsDF.  
	Dataframes must be same size and have matching indicies.
    
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datatime as index
                    obsDF (pandas.DataFrame): dataframe with datatime as index
		    
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	# assumes all values are greater then 0
	# assumes values are already log base 10

	obsDF = obsDF.reindex(modDF.index)
	mod_over_obs = obsDF - modDF
	median = mod_over_obs.median()
	sign = median.apply(lambda x: np.sign(x))
	return 100 * sign * (10**median - 1)


def ssopScore(modDF, obsDF):
	'''
    Computes Skill Score of Prediction based on modDF and obsDF.  
	Dataframes must be same size and have matching indicies.
    
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datatime as index
                    obsDF (pandas.DataFrame): dataframe with datatime as index
		    
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	obsDF = obsDF.reindex(modDF.index)
	mseRef = variance(obsDF)
	msePred = (modDF - obsDF)**2
	ss = msePred / mseRef
	ssop = 1 - ss.mean()**0.5
	return ssop


def truePostiveRate(mod, obs, threshold):
	'''
    Computes true positive rate based on modDF and obsDF.  
	Dataframes must be same size and have matching indicies.
    
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datatime as index
                    obsDF (pandas.DataFrame): dataframe with datatime as index
		    		threshold (float): threshold to determine an 'event'
		    
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	obs = obs.reindex(mod.index)
	ct = contingencyTable(mod, obs, threshold)
	return ct['tp'] / (ct['tp'] + ct['fn'])

def truePostiveRate_from_ct(ct):
	'''
    Computes true positive rate based on contingency table
            Parameters:
                    ct (pandas.DataFrame): contingecny table with true positive (tp)
                        and false negative (fn) counts
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	return ct['tp'] / (ct['tp'] + ct['fn'])

def falsePostiveRate(mod, obs, threshold):
	'''
    Computes false positive rate based on modDF and obsDF.  
	Dataframes must be same size and have matching indicies.
    
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datatime as index
                    obsDF (pandas.DataFrame): dataframe with datatime as index
		    		threshold (float): threshold to determine an 'event'
		    
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	obs = obs.reindex(mod.index)
	ct = contingencyTable(mod, obs, threshold)
	return ct['fp'] / (ct['fp'] + ct['tn'])

def falsePostiveRate_from_ct(ct):
	'''
    Computes false positive rate based on contingency table 
    
            Parameters:
                    ct: contingency table with false positive (fp) and true negative (tn) counts
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	return ct['fp'] / (ct['fp'] + ct['tn'])

def threatScore(mod, obs, threshold):
	'''
    Computes threat score based on modDF and obsDF.  
	Dataframes must be same size and have matching indicies.

            Parameters:
                    modDF (pandas.DataFrame): dataframe with datatime as index
                    obsDF (pandas.DataFrame): dataframe with datatime as index
		    		threshold (float): threshold to determine an 'event'
		    
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	obs = obs.reindex(mod.index)
	ct = contingencyTable(mod, obs, threshold)
	return ct['tp'] / (ct['tp'] + ct['fp'] + ct['fn'])

def threatScore_from_ct(ct):
	'''
    Computes threat score based on contingency table calculated from modDF and obsDF.  
            Parameters:
                    ct: contingency table with true positive (tp), false positive (fp)
                        and false negative (fn) counts
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	return ct['tp'] / (ct['tp'] + ct['fp'] + ct['fn'])

def trueSkillStatistics(mod, obs, threshold):
	'''
    Computes true skill based on modDF and obsDF.  
	Dataframes must be same size and have matching indicies.  

            Parameters:
                    modDF (pandas.DataFrame): dataframe with datatime as index
                    obsDF (pandas.DataFrame): dataframe with datatime as index
		    		threshold (float): threshold to determine an 'event'
		    
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	obs = obs.reindex(mod.index)
	ct = contingencyTable(mod, obs, threshold)
	return ct['tp'] / (ct['tp'] + ct['fn']) - ct['fp'] / (ct['fp'] + ct['tn'])

def bias(mod, obs, threshold):
	'''
    Computes bias based on modDF and obsDF.  
	Dataframes must be same size and have matching indicies.  
	  
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datatime as index
                    obsDF (pandas.DataFrame): dataframe with datatime as index
		    		threshold (float): threshold to determine an 'event'
		    
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	obs = obs.reindex(mod.index)
	ct = contingencyTable(mod, obs, threshold)
	return (ct['tp'] + ct['fp']) / (ct['tp'] + ct['fn'])

def bias_from_ct(ct):
        '''
    Computes bias based on contingency table
       
            Parameters:
                    ct: contingency table with true positive (tp) and false negative (fn) counts
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
        '''
        return (ct['tp'] + ct['fp']) / (ct['tp'] + ct['fn'])

def pod_from_ct(ct):
        '''
    Computes Probability of Detection based on contingency table
       
            Parameters:
                    ct: contingency table with true positive (tp) and false negative (fn) counts
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
        '''
        return (ct['tp']) / (ct['tp'] + ct['fn'])

def pofd_from_ct(ct):
        '''
    Computes Probability of False Detection based on contingency table
       
            Parameters:
                    ct - contingency table with false positive (fp) and true negative (tn) counts
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
        '''
        return (ct['fp']) / (ct['fp'] + ct['tn'])

def HeidkeScore(ct):
        '''
    Computes Heidke Skill Score based on modDF and obsDF.  
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datatime as index
                    obsDF (pandas.DataFrame): dataframe with datatime as index
		    		threshold (float): threshold to determine an 'event'
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
        '''
        obs = obs.reindex(mod.index)
        ct = contingencyTable(mod, obs, threshold)
        h = ct['tp']
        f = ct['fp']
        m = ct['fn']
        n = ct['tn']       
        return (2 * (h * n - m * f)/( (h + m) * (m + n) + (h + f) * (f + n) ) )

def HeidkeScore_from_ct(ct):
        '''
    Computes Heidke Skill Score based on contingency table
            Parameters:
                    ct: contingency table with true positive (tp), false positive (fp),
                        false negative (fn) and true negative (tn) counts
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
        '''
        h = ct['tp']
        f = ct['fp']
        m = ct['fn']
        n = ct['tn']       
        return (2 * (h * n - m * f)/( (h + m) * (m + n) + (h + f) * (f + n) ) )

# internal charging metrics
# Median symmetric accuracy 
def msaScore(modDF, obsDF):
	'''
    Computes Median Symmetric Accuracy based on modDF and obsDF.  
	Dataframes must be same size and have matching indicies.  
	  
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datatime as index
                    obsDF (pandas.DataFrame): dataframe with datatime as index
		    
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	obsDF = obsDF.reindex(modDF.index)
	Q = modDF / obsDF
	medianAbs = (abs(np.log(Q))).median()
	msa = 100 * (pow(np.e, medianAbs)- 1); 
	return msa


def sspbScore(modDF, obsDF):
	'''
    Computes Symmetric Signed Percentage Bias based on modDF and obsDF.  
	Dataframes must be same size and have matching indicies.  
	  
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datatime as index
                    obsDF (pandas.DataFrame): dataframe with datatime as index
		    
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	obsDF = obsDF.reindex(modDF.index)
	Q = modDF / obsDF
	absMedian = abs(np.log(Q).median())
	sspb = 100 * np.sign(np.log(Q).median()) *  (pow(np.e, absMedian)- 1); 
	return sspb

def mocrScore(modDF, obsDF):
	'''
    Computes Mean Observed to Computed (O/C) Ratio based on modDF and obsDF.  
	Dataframes must be same size and have matching indicies.  
	  
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datatime as index
                    obsDF (pandas.DataFrame): dataframe with datatime as index
		    
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	obsDF = obsDF.reindex(modDF.index)
	R = obsDF / modDF
	return R.mean()

def sdocrScore(modDF, obsDF):
	'''
    Computes Standard Deviation Observed to computed (O/C) Ratio based on modDF and obsDF.  
	Dataframes must be same size and have matching indicies.  
	  
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datatime as index
                    obsDF (pandas.DataFrame): dataframe with datatime as index
		    
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	obsDF = obsDF.reindex(modDF.index)
	R = obsDF / modDF
	A = np.log(R)
	B = np.log(R.mean())
	deviation =  ((A - B) ** 2).mean() ** 0.5
	return deviation

#
# Helper functions
#

def sigma(df):
	'''
    Computes sigma of dataframe. Helper function.
    
            Parameters:
                    df (pandas.DataFrame): dataframe with datatime as index and single parameter    
            Returns:
                    sigma (float): sigma of the dataframe
    '''
	mean = df.mean()
	squaredMeanDiff = (df - mean) * (df - mean)
	squaredMeanDiffSum = squaredMeanDiff.sum()
	N = squaredMeanDiff.shape[0]
	squaredSigma = squaredMeanDiffSum / (N-1)
	return squaredSigma**0.5

def variance(df):
	'''
    Computes variance of dataframe. Helper function.
    
            Parameters:
                    df (pandas.DataFrame): dataframe with datatime as index and single parameter    
            Returns:
                    variance (float): variance of the dataframe
    '''
	mean = df.mean()
	squaredMeanDiff = (df - mean) * (df - mean)
	squaredMeanDiffSum = squaredMeanDiff.sum()
	N = squaredMeanDiff.shape[0]
	return squaredMeanDiffSum / (N-1)

def contingencyTable(mod, obs, threshold):
	'''
    Computes contingency table. Helper function.  Columns returned include
    true positive, false positive, false negative, and true negative
    
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datatime as index
                    obsDF (pandas.DataFrame): dataframe with datatime as index
		    
            Returns:
                    contingency table (pandas.DataFrame): 
    '''
	eventMod = np.full(mod.shape,0)
	eventMod[mod > threshold] = 1
	eventObs = np.full(obs.shape,0)
	eventObs[obs > threshold] = 1

	tp = sum(eventMod + eventObs == 2)[0]
	fp = sum(eventMod)[0] - tp
	fn = sum(eventObs)[0] - tp
	tn = sum(eventMod + eventObs == 0)[0] 
	df = pd.DataFrame([[tp, fp, fn, tn]], columns = ['tp', 'fp', 'fn', 'tn'])
	return df


def contingencyTable_with_timewindows(mod, obs, threshold, windowlength,use_obs_times=False,use_model_times=False,debug=False):
	'''
    Computes contingency table. Helper function.  Columns returned include
    true positive, false positive, false negative, and true negative
    
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datatime as index
                    obsDF (pandas.DataFrame): dataframe with datatime as index
                    threshold: numerical value indicating a hit when exceeded
                    windowlength: length in minutes of time intervals that the observation and modeled time 
                           interval is divided into. Time windows do not overlap.
                           in each time interval, at least one value exceeding the threshold indicates a hit
                    missing data in either observation or model will result in missed hits.
            Returns:
                    contingency table (pandas.DataFrame): 
    '''
	t0m = mod.index[0]
	t1m = mod.index[-1]
	t0o = obs.index[0]
	t1o = obs.index[-1]
	t0 = min([t0m,t0o])
	t1 = max([t1m,t1o])
	if use_obs_times:
		t0 = t0o
		t1 = t1o
	if use_model_times:
		t0 = t0m
		t1 = t1m
	dt = timedelta(minutes=windowlength)
	n_dt = int(np.ceil((t1-t0)/dt))
	print('t1-t0: ',t1-t0,' dt: ',dt,' number of windows: ',n_dt)
    
	eventMod = np.full(n_dt,0)
	for i in range(n_dt):
		t0_ = t0 + i*dt
		t1_ = t0 +(i+1)*dt
		eventMod[i] = np.any(mod[ np.logical_and(mod.index >= t0_, mod.index < t1_) == True ] > threshold)

	eventObs = np.full(n_dt,0)
	for i in range(n_dt):
		t0_ = t0 + i*dt
		t1_ = t0 +(i+1)*dt
		eventObs[i] = np.any(obs[ np.logical_and(obs.index >= t0_, obs.index < t1_) == True ] > threshold)
	if debug:
		print("OY ON",np.sum(eventObs),n_dt-np.sum(eventObs))

	tp = sum(eventMod + eventObs == 2)
	fp = sum(eventMod) - tp
	fn = sum(eventObs) - tp
	tn = sum(eventMod + eventObs == 0) 
	df = pd.DataFrame([[tp, fp, fn, tn]], columns = ['tp', 'fp', 'fn', 'tn'])
	return df
