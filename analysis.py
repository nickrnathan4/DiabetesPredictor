import csv as csv
import numpy as np
import pandas as pd

from datetime import datetime


# ======================= LOAD DATASETS =======================

allergy_raw = pd.read_csv('trainingSet/training_SyncAllergy.csv', header=0)
condition_raw = pd.read_csv('trainingSet/SyncCondition.csv', header=0)
diagnosis_raw = pd.read_csv('trainingSet/training_SyncDiagnosis.csv', header=0)
lab_observation_raw = pd.read_csv('trainingSet/training_SyncLabObservation.csv', header=0)
lab_panel_raw = pd.read_csv('trainingSet/training_SyncLabPanel.csv', header=0)
lab_result_raw = pd.read_csv('trainingSet/training_SyncLabResult.csv', header=0)
medication_raw = pd.read_csv('trainingSet/training_SyncMedication.csv', header=0)
patient_raw = pd.read_csv('trainingSet/training_SyncPatient.csv', header=0)
patient_condition_raw = pd.read_csv('trainingSet/training_SyncPatientCondition.csv', header=0)
patient_smoke_status_raw = pd.read_csv('trainingSet/training_SyncPatientSmokingStatus.csv', header=0)
prescription_raw = pd.read_csv('trainingSet/training_SyncPrescription.csv', header=0)
smoking_status_raw = pd.read_csv('trainingSet/SyncSmokingStatus.csv', header=0)
transcript_raw = pd.read_csv('trainingSet/training_SyncTranscript.csv', header=0)
transcript_allergy_raw = pd.read_csv('trainingSet/training_SyncTranscriptAllergy.csv', header=0)
transcript_diagnosis_raw = pd.read_csv('trainingSet/training_SyncTranscriptDiagnosis.csv', header=0)
transcript_medication_raw = pd.read_csv('trainingSet/training_SyncTranscriptMedication.csv', header=0)
regions = pd.read_csv('trainingSet/state_regions.csv', header=0)


# ======================= BUILD DATASET =======================
patient = patient_raw[['PatientGuid', 
					  'DMIndicator', 
					  'Gender',
					  'YearOfBirth',
					  'State']] 


# ADD SMOKING STATUS FEATURE
smoking_status = pd.merge(patient_smoke_status_raw, smoking_status_raw, on='SmokingStatusGuid', how='left')
smoking_status = smoking_status[['PatientGuid', 'Severity', 'EffectiveYear']]
smoking_status = smoking_status.groupby('PatientGuid').agg({'EffectiveYear': np.max, 'Severity' : np.max}) # take most recent diagnosis
smoking_status = smoking_status.drop('EffectiveYear',1)
patient = pd.merge(patient, smoking_status, left_on='PatientGuid', right_index=True, how='left')
patient = patient.rename(columns = {'Severity' : 'SmokingStatus' })


# ENCODE GENDER
patient['GenderDummy'] = (patient['Gender'] == 'M').astype(int)


# CALCULATE AGE
patient['Age'] = datetime.today().year - patient['YearOfBirth']


# GENERATE TRANSCRIPT FEATURES
def get_transcripts(patientGuid):
	return transcript_raw[transcript_raw['PatientGuid']==patientGuid]

def avg_transcript_feature(patientGuid, feature):
	feature_series = transcript_raw[transcript_raw['PatientGuid']==patientGuid][feature]
	return feature_series.replace(0.0, np.nan).mean()


# ADD BMI FEATURE
BMI = []
for guid in patient['PatientGuid']:
	mean_BMI = avg_transcript_feature(guid, 'BMI')
	BMI.append({'PatientGuid': guid,'MeanBMI' : mean_BMI})

BMI_df = pd.DataFrame(BMI, columns=['PatientGuid', 'MeanBMI'])
patient = pd.merge(patient, BMI_df, on='PatientGuid', how='left')

# ADD SYSTOLIC FEATURE
systolic = []
for guid in patient['PatientGuid']:
	mean_systolic = avg_transcript_feature(guid, 'SystolicBP')
	systolic.append({'PatientGuid': guid,'MeanSystolic' : mean_systolic})

systolic_df = pd.DataFrame(systolic, columns=['PatientGuid', 'MeanSystolic'])
patient = pd.merge(patient, systolic_df, on='PatientGuid', how='left')

# ADD DIASTOLIC FEATURE
diastolic = []
for guid in patient['PatientGuid']:
	mean_diastolic = avg_transcript_feature(guid, 'DiastolicBP')
	diastolic.append({'PatientGuid': guid,'MeanDiastolic' : mean_diastolic})

diastolic_df = pd.DataFrame(diastolic, columns=['PatientGuid', 'MeanDiastolic'])
patient = pd.merge(patient, diastolic_df, on='PatientGuid', how='left')

# ADD RESPIRATORY RATE FEATURE
resp_rate = []
for guid in patient['PatientGuid']:
	mean_resp_rate = avg_transcript_feature(guid, 'RespiratoryRate')
	resp_rate.append({'PatientGuid': guid,'MeanRespRate' : mean_resp_rate})

resp_rate_df = pd.DataFrame(resp_rate, columns=['PatientGuid', 'MeanRespRate'])
patient = pd.merge(patient, resp_rate_df, on='PatientGuid', how='left')

# ADD HEART RATE FEATURE
heart_rate = []
for guid in patient['PatientGuid']:
	mean_heart_rate = avg_transcript_feature(guid, 'HeartRate')
	heart_rate.append({'PatientGuid': guid,'MeanHeartRate' : mean_heart_rate})

heart_rate_df = pd.DataFrame(heart_rate, columns=['PatientGuid', 'MeanHeartRate'])
patient = pd.merge(patient, heart_rate_df, on='PatientGuid', how='left')


# REMOVE OUTLIERS
patient = patient[patient.MeanBMI > 100]

# ENCODE CATEGORICAL VARIABLES
patient['MaleDummy'] = 0
patient['MaleDummy'] = (patient['Gender'] == 'M').astype(int)

patient = pd.merge(patient, regions, on='State', how='left')
patient['CentralRegion'] = 0
patient['NorthRegion'] = 0
patient['SouthRegion'] = 0
patient['WestRegion'] = 0
patient['CentralRegion'] = (patient['Region'] == 'C').astype(int)
patient['NorthRegion'] = (patient['Region'] == 'N').astype(int)
patient['SouthRegion'] = (patient['Region'] == 'S').astype(int)
patient['WestRegion'] = (patient['Region'] == 'W').astype(int)


# DROP NON-NUMERIC FEATURES
patient = patient[['DMIndicator',
					'Age',
					'SmokingStatus',
					'MeanBMI',
					'MeanSystolic',
					'MeanDiastolic',
					'MeanRespRate',
					'MaleDummy',
					'CentralRegion',
					'SouthRegion',
					'NorthRegion',
					'WestRegion']]

# REMOVE NULL VALUES
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(patient)
patient = imp.transform(patient)

col_labels = {'DMIndicator':patient[:,0],
				'Age':patient[:,1],
				'SmokingStatus':patient[:,2],
				'MeanBMI':patient[:,3],
				'MeanSystolic':patient[:,4],
				'MeanDiastolic':patient[:,5],
				'MeanRespRate':patient[:,6],
				'MaleDummy':patient[:,7],
				'CentralRegion':patient[:,8],
				'SouthRegion':patient[:,9],
				'NorthRegion':patient[:,10],
				'WestRegion':patient[:,11]}

patient = pd.DataFrame(col_labels)

# ======================= ANALYSIS =======================
from sklearn import linear_model

# CONVERT TO NUMPY ARRAYS
X = pd.DataFrame.as_matrix(patient.drop('DMIndicator',1))
y = patient['DMIndicator'].values

# SPLIT INTO TRAINING AND TEST
split_ratio = 0.2
np.random.seed(34)
indices = np.random.permutation(len(X))
X_train = X[indices[:int(-1*len(X)*split_ratio)]]
y_train = y[indices[:int(-1*len(y)*split_ratio)]]
X_test = X[indices[int(-1*len(X)*split_ratio):]]
y_test = y[indices[int(-1*len(y)*split_ratio):]]

logit = linear_model.LogisticRegression()
logit.fit(X_train,y_train)
prediction = logit.predict(X_test)					# predict class labels
prediction_prob = logit.predict_proba(X_test)		# probability estimates
results = prediction == y_test
print "True Positive: " + str(float(np.sum(results))/float(len(results))) + "%"

# ======================= LOG LOSS EVALUATION =======================

import scipy as sp
def llfun(act, pred):
	epsilon = 1e-15
	pred = sp.maximum(epsilon, pred)
	pred = sp.minimum(1-epsilon, pred)
	ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
	ll = sum(ll)
	ll = ll * -1.0/len(act)
	return ll

def create_test_probs(inprobs, outprobs):
	for i in inprobs:
		if i == 1.0:
			outprobs.append([0.0,1.0])
		else:
			outprobs.append([1.0,0.0])

y_test_prob = []
create_test_probs(y_test, y_test_prob)
print llfun(y_test_prob, prediction_prob)

# VARIABLE LIST
# DMIndicator
# Age
# SmokingStatus 		
# MeanBMI
# MeanSystolic 			
# MeanDiastolic 		
# MeanRespRate 			
# MaleDummy
# CentralRegion
# SouthRegion
# NorthRegion
# WestRegion









