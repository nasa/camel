"""
Copyright © 2023 United States Government as represented by the 
Administrator of the National Aeronautics and Space Administration.  
No copyright is claimed in the United States under Title 17, U.S. Code. 
All Other Rights Reserved.

@author: remullinix
"""

import pandas as pd
import numpy as np


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
