"""
@author: remullinix, mareiss
Version: 1.0
"""

import pandas as pd
import numpy as np
from scipy.stats import linregress, skew, kurtosis
from scipy.signal import find_peaks

def rmsScore(modDF, obsDF):
	'''
    Computes Root Mean Square Error based on modDF and obsDF.  
	Dataframes must be same size and have matching indices.
    
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datetime as index
                    obsDF (pandas.DataFrame): dataframe with datetime as index
		    
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
	Dataframes must be same size and have matching indices.
    
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datetime as index
                    obsDF (pandas.DataFrame): dataframe with datetime as index
		    
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
	Dataframes must be same size and have matching indices.
    
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datetime as index
                    obsDF (pandas.DataFrame): dataframe with datetime as index
		    
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
	Dataframes must be same size and have matching indices.
    
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datetime as index
                    obsDF (pandas.DataFrame): dataframe with datetime as index
		    
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	obsDF = obsDF.reindex(modDF.index)
	mse = ((obsDF - modDF) ** 2).mean()
	return mse

def ccScore(modDF, obsDF):
	'''
    Computes Pearson Correlation Coefficient based on modDF and obsDF.  
	Dataframes must be same size and have matching indices.
    
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datetime as index
                    obsDF (pandas.DataFrame): dataframe with datetime as index
		    
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
	Dataframes must be same size and have matching indices.
    
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datetime as index
                    obsDF (pandas.DataFrame): dataframe with datetime as index
		    
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	obsDF = obsDF.reindex(modDF.index)
	sigmaObsSquared = sigma(obsDF)**2
	diff = (obsDF - modDF)*(obsDF - modDF) / sigmaObsSquared
	return 1 - diff.mean()

def sspbScore(modDF, obsDF):
	'''
    Computes Symmetric Signed Percentage Bias based on modDF and obsDF.  
	Dataframes must be same size and have matching indices.
    
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datetime as index
                    obsDF (pandas.DataFrame): dataframe with datetime as index
		    
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
	Dataframes must be same size and have matching indices.
    
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datetime as index
                    obsDF (pandas.DataFrame): dataframe with datetime as index
		    
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	obsDF = obsDF.reindex(modDF.index)
	mseRef = variance(obsDF)
	msePred = (modDF - obsDF)**2
	ss = msePred / mseRef
	ssop = 1 - ss.mean()**0.5
	return ssop


def truePositiveRate(ct):
	'''
    Computes true positive rate based on modDF and obsDF.  
	Dataframes must be same size and have matching indices.
    
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datetime as index
                    obsDF (pandas.DataFrame): dataframe with datetime as index
		    		threshold (float): threshold to determine an 'event'
		    
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	return ct['tp'] / (ct['tp'] + ct['fn'])

def falsePostiveRate(ct):
	'''
    Computes false positive rate based on modDF and obsDF.  
	Dataframes must be same size and have matching indices.
    
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datetime as index
                    obsDF (pandas.DataFrame): dataframe with datetime as index
		    		threshold (float): threshold to determine an 'event'
		    
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	return ct['fp'] / (ct['fp'] + ct['tn'])

def falseNegativeRate(ct):
	'''
    Computes false negative rate based on modDF and obsDF.  
	Dataframes must be same size and have matching indices.
    
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datetime as index
                    obsDF (pandas.DataFrame): dataframe with datetime as index
		    		threshold (float): threshold to determine an 'event'
		    
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	return ct['fn'] / (ct['tp'] + ct['fn'])

def positivePredictiveValue(ct):
	'''
    Computes predictive value rate based on modDF and obsDF.  
	Dataframes must be same size and have matching indices.
    
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datetime as index
                    obsDF (pandas.DataFrame): dataframe with datetime as index
		    		threshold (float): threshold to determine an 'event'
		    
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	return ct['tp'] / (ct['tp'] + ct['fp'])

def falseAlarmRatio(ct):
	'''
    Computes false alarm ratio based on modDF and obsDF.  
	Dataframes must be same size and have matching indices.
    
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datetime as index
                    obsDF (pandas.DataFrame): dataframe with datetime as index
		    		threshold (float): threshold to determine an 'event'
		    
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	return ct['fp'] / (ct['tp'] + ct['fp'])

def threatScore(ct):
	'''
    Computes threat score based on modDF and obsDF.  
	Dataframes must be same size and have matching indices.

            Parameters:
                    modDF (pandas.DataFrame): dataframe with datetime as index
                    obsDF (pandas.DataFrame): dataframe with datetime as index
		    		threshold (float): threshold to determine an 'event'
		    
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	return ct['tp'] / (ct['tp'] + ct['fp'] + ct['fn'])

def trueSkillStatistics(ct):
	'''
    Computes true skill based on modDF and obsDF.  
	Dataframes must be same size and have matching indices.  

            Parameters:
                    modDF (pandas.DataFrame): dataframe with datetime as index
                    obsDF (pandas.DataFrame): dataframe with datetime as index
		    		threshold (float): threshold to determine an 'event'
		    
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	return ct['tp'] / (ct['tp'] + ct['fn']) - ct['fp'] / (ct['fp'] + ct['tn'])

def bias(ct):
	'''
    Computes bias based on modDF and obsDF.  
	Dataframes must be same size and have matching indices.  
	  
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datetime as index
                    obsDF (pandas.DataFrame): dataframe with datetime as index
		    		threshold (float): threshold to determine an 'event'
		    
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	return (ct['tp'] + ct['fp']) / (ct['tp'] + ct['fn'])


# internal charging metrics
# Median symmetric accuracy 
def msaScore(modDF, obsDF):
	'''
    Computes Median Symmetric Accuracy based on modDF and obsDF.  
	Dataframes must be same size and have matching indices.  
	  
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datetime as index
                    obsDF (pandas.DataFrame): dataframe with datetime as index
		    
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	obsDF = obsDF.reindex(modDF.index)
	Q = modDF / obsDF
	medianAbs = (abs(np.log(Q))).median()
	msa = 100 * (pow(np.e, medianAbs)- 1)
	return msa


def sspbScore(modDF, obsDF):
	'''
    Computes Symmetric Signed Percentage Bias based on modDF and obsDF.  
	Dataframes must be same size and have matching indices.  
	  
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datetime as index
                    obsDF (pandas.DataFrame): dataframe with datetime as index
		    
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	obsDF = obsDF.reindex(modDF.index)
	Q = modDF / obsDF
	absMedian = abs(np.log(Q).median())
	sspb = 100 * np.sign(np.log(Q).median()) *  (pow(np.e, absMedian)- 1)
	return sspb

def mocrScore(modDF, obsDF):
	'''
    Computes Mean Observed to Computed (O/C) Ratio based on modDF and obsDF.  
	Dataframes must be same size and have matching indices.  
	  
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datetime as index
                    obsDF (pandas.DataFrame): dataframe with datetime as index
		    
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	obsDF = obsDF.reindex(modDF.index)
	R = obsDF / modDF
	return R.mean()

def sdocrScore(modDF, obsDF):
	'''
    Computes Standard Deviation Observed to computed (O/C) Ratio based on modDF and obsDF.  
	Dataframes must be same size and have matching indices.  
	  
            Parameters:
                    modDF (pandas.DataFrame): dataframe with datetime as index
                    obsDF (pandas.DataFrame): dataframe with datetime as index
		    
            Returns:
                    score (pandas.Series): series with parameter name(s) as index
    '''
	obsDF = obsDF.reindex(modDF.index)
	R = obsDF / modDF
	A = np.log(R)
	B = np.log(R.mean())
	deviation =  ((A - B) ** 2).mean() ** 0.5
	return deviation

def sigma(df):
	'''
    Computes sigma of dataframe. Helper function.
    
            Parameters:
                    df (pandas.DataFrame): dataframe with datetime as index and single parameter    
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
                    df (pandas.DataFrame): dataframe with datetime as index and single parameter    
            Returns:
                    variance (float): variance of the dataframe
    '''
	mean = df.mean()
	squaredMeanDiff = (df - mean) * (df - mean)
	squaredMeanDiffSum = squaredMeanDiff.sum()
	N = squaredMeanDiff.shape[0]
	return squaredMeanDiffSum / (N-1)

def regressionStats(modDF, obsDF):
	'''
    Computes linear regression statistics between model and observed series.

            Parameters:
                    modDF (pandas.Series): predicted values
                    obsDF (pandas.Series): observed values
		    
            Returns:
                    dict: contains slope, intercept, r_value, r_squared
    '''

	obsDF = obsDF.reindex(modDF.index)
	
	valid = modDF.notna() & obsDF.notna()
	modDF = modDF[valid]
	obsDF = obsDF[valid]

	slope, intercept, r_value, _, _ = linregress(obsDF, modDF)
	return {
		'slope': slope,
		'intercept': intercept,
		'r_value': r_value,
		'r_squared': r_value ** 2
	}

def descriptiveStats(series):
    '''
    Computes a set of descriptive statistics for a pandas Series.

            Parameters:
                    series (pandas.Series): Series of numerical data
                    
            Returns:
                    dict: Dictionary with keys:
                        - 'mean'
                        - 'median'
                        - 'std'
                        - 'min'
                        - 'max'
                        - 'var'
                        - 'skewness'
                        - 'kurtosis'
    '''
    series = series.dropna()
    return {
        'mean': round(series.mean(), 2),
        'median': round(series.median(), 2),
        'std': round(series.std(ddof=0), 2),
        'min': round(series.min(), 2),
        'max': round(series.max(), 2),
        'var': round(series.var(ddof=0), 2),
        'skewness': round(skew(series), 2),
        'kurtosis': round(kurtosis(series), 2)
    }

def classifyBinaryEvents(modDF, obsDF, threshold=400):
    '''
    Classifies each prediction–observation pair into binary event categories
    and computes the overall contingency table.

            Parameters:
                    modDF (pandas.DataFrame): Predicted data with a 'Model' column
                    obsDF (pandas.DataFrame): Observed data with an 'Obs' column
                    threshold (float, optional): Threshold used to determine an event [default = 400]
                    
            Returns:
                    tuple:
                        - labels (pandas.Series): Per-point classification with values:
                            'Hit', 'Miss', 'False Alarm', 'Correct Rejection'
                            (aligned to modDF index)
                        - table (pandas.DataFrame): 1×4 contingency table with columns:
                            'tp' (true positives)
                            'fp' (false positives)
                            'fn' (false negatives)
                            'tn' (true negatives)
    '''
    obsDF = obsDF.reindex(modDF.index)

    m = modDF['Model']
    o = obsDF['Obs']

    m_evt = m >= threshold
    o_evt = o >= threshold

    conditions = [
        m_evt & o_evt,            # Hit
        (~m_evt) & (~o_evt),      # Correct Rejection
        m_evt & (~o_evt),         # False Alarm
        (~m_evt) & o_evt          # Miss
    ]
    choices = ['Hit', 'Correct Rejection', 'False Alarm', 'Miss']
    labels = pd.Series(np.select(conditions, choices), index=m.index, name='classification')

    tp = int((m_evt & o_evt).sum())
    fp = int((m_evt & ~o_evt).sum())
    fn = int((~m_evt & o_evt).sum())
    tn = int((~m_evt & ~o_evt).sum())
    table = pd.DataFrame([[tp, fp, fn, tn]], columns=['tp', 'fp', 'fn', 'tn'])

    return labels, table

def compute_peaks(dates, observed_data, predicted_data, minimum_speed, maximum_dt):
    '''
    Detects peaks in observed and predicted time series, matches them within a time tolerance,
    and computes event-based statistics (hits, misses, false alarms) along with timing and
    magnitude differences.

            Parameters:
                    dates (pandas.Series): Datetime values corresponding to the data points
                    observed_data (pandas.Series): Observed values
                    predicted_data (pandas.Series): Predicted/model values
                    minimum_speed (float): Minimum peak height (speed) to qualify as an event
                    maximum_dt (float): Maximum allowed time difference (in days) between matched peaks
                    
            Returns:
                    tuple:
                        - peaks_obs (numpy.ndarray): Indices of detected observed peaks
                        - peaks_pred (numpy.ndarray): Indices of detected predicted peaks
                        - hit_positions (pandas.Index): Indices of matched predicted peaks ("hits")
                        - misses_positions (pandas.Index): Indices of unmatched observed peaks ("misses")
                        - false_alarms_positions (pandas.Index): Indices of unmatched predicted peaks ("false alarms")
                        - time_diffs (list of float): Time differences [hours] for each matched (hit) peak
                        - data_diffs (list of float): Value differences [same units as data] for each matched (hit) peak
    '''
    dates = pd.to_datetime(dates)

    observed_values = observed_data.to_numpy()
    predicted_values = predicted_data.to_numpy()

    # Detect peaks
    peaks_obs, _ = find_peaks(
        observed_values, height=minimum_speed, prominence=100, distance=48, width=24
    )
    peaks_pred, _ = find_peaks(
        predicted_values, height=minimum_speed, prominence=100, distance=48, width=24
    )

    peaks_obs_dates = dates.iloc[peaks_obs]
    peaks_pred_dates = dates.iloc[peaks_pred]

    # Build all valid (obs, pred) pairs within maximum_dt
    candidate_pairs = []
    for i, obs_date in enumerate(peaks_obs_dates):
        for j, pred_date in enumerate(peaks_pred_dates):
            dt_days = abs((pred_date - obs_date).days)
            if dt_days <= maximum_dt:
                candidate_pairs.append((i, j, dt_days))

    # Sort pairs by smallest time difference
    candidate_pairs.sort(key=lambda x: x[2])

    matched_obs = set()
    matched_pred = set()
    matches = []

    for i, j, _ in candidate_pairs:
        if i not in matched_obs and j not in matched_pred:
            matched_obs.add(i)
            matched_pred.add(j)
            matches.append((i, j))

    # Extract hits
    hits = [j for _, j in matches]
    hit_positions = peaks_pred_dates.index[hits]

    # Extract misses and false alarms
    misses = [i for i in range(len(peaks_obs)) if i not in matched_obs]
    false_alarms = [j for j in range(len(peaks_pred)) if j not in matched_pred]

    misses_positions = peaks_obs_dates.index[misses]
    false_alarms_positions = peaks_pred_dates.index[false_alarms]

    # Compute time and data diffs for hits
    time_diffs, data_diffs = [], []
    for i, j in matches:
        time_diff = (peaks_pred_dates.iloc[j] - peaks_obs_dates.iloc[i]).total_seconds() / 3600
        data_diff = predicted_values[peaks_pred[j]] - observed_values[peaks_obs[i]]
        time_diffs.append(time_diff)
        data_diffs.append(data_diff)

    return (
        peaks_obs,
        peaks_pred,
        hit_positions,
        misses_positions,
        false_alarms_positions,
        time_diffs,
        data_diffs
    )