import pandas 
import requests 
import json 
import numpy as np 
from sklearn import tree 
import pydotplus

#Google API information
GOOGLE_MAPS_API_KEY = 'AIzaSyDi3j-gUjpdC_DXhZrk8LjH4BGz6FyE1rQ'
GOOGLE_MAPS_API_ENDPOINT = 'https://maps.googleapis.com/maps/api/geocode/json' 

'''
crimeToUrl = {
	'ANTI-BLACK': 'http://maps.google.com/mapfiles/ms/icons/blue-dot.png', 
	'ANTI-FEMALE HOMOSEXUAL (LESBIAN)': 'http://maps.google.com/mapfiles/ms/icons/red-dot.png',
	'ANTI-MALE HOMOSEXUAL (GAY)': 'http://maps.google.com/mapfiles/ms/icons/red-dot.png', 
	'ANTI-TRANSGENDER': 'http://maps.google.com/mapfiles/ms/icons/red-dot.png', 
	'ANTI-GENDER NON-CONFORMING': 'http://maps.google.com/mapfiles/ms/icons/red-dot.png',
	'ANTI-WHITE': 'http://maps.google.com/mapfiles/ms/icons/purple-dot.png',
	'ANTI-ARAB': 'http://maps.google.com/mapfiles/ms/icons/yellow-dot.png',
	'ANTI-ISLAMIC (MUSLIM)': 'http://maps.google.com/mapfiles/ms/icons/yellow-dot.png',
	'ANTI-JEWISH': 'http://maps.google.com/mapfiles/ms/icons/green-dot.png'
}
'''

crimeToClass = {
	'ANTI-BLACK': 0, 
	'ANTI-FEMALE HOMOSEXUAL (LESBIAN)': 1,
	'ANTI-MALE HOMOSEXUAL (GAY)': 1, 
	'ANTI-TRANSGENDER': 1, 
	'ANTI-GENDER NON-CONFORMING': 1,
	'ANTI-WHITE': 2,
	'ANTI-ARAB': 3,
	'ANTI-ISLAMIC (MUSLIM)': 3,
	'ANTI-JEWISH': 4
}

#Request latitude/longitude from Google to cluster on
def geolocationDataForAddress(address):
	params = {'address': address, 'key': GOOGLE_MAPS_API_KEY}
	req = requests.get(GOOGLE_MAPS_API_ENDPOINT, params)
	try:
		apiResponse = json.loads(req.text)
		locationData = apiResponse['results'][0]['geometry']['location']
	except: 
		return None 

	return locationData

#open data in pandas
with open('HATE_CRIME_DATA.CSV', 'r') as hateCrimeFile: 
	pandasObject = pandas.read_csv(hateCrimeFile)

#trim data tables
relevant_columns = ['BIAS_MOTIVATION_GROUP', 'BLOCK_ADDRESS', 'City', 'ZIP_CODE'] 
pandasObject = pandasObject[relevant_columns] 

#to store relevant data 
finalDf = pandas.DataFrame(columns=['LABELS', 'LATITUDE', 'LONGITUDE'])

pandasObject['LATITUDE'] = np.nan
pandasObject['LONGITUDE'] = np.nan

print("Loading data.")
#iterate over rows of pandas dataframe
for index, row in pandasObject.iterrows(): 
	currentAddress = row['BLOCK_ADDRESS'].rstrip() + ', ' + row['City'] + ', KY '
	addressLocation = geolocationDataForAddress(currentAddress)
	crimeType = row['BIAS_MOTIVATION_GROUP'].strip() 
	if crimeType in crimeToClass: 
		if addressLocation != None: 
			finalDf.set_value(index, 'LATITUDE', float(addressLocation['lat']))
			finalDf.set_value(index, 'LONGITUDE', float(addressLocation['lng']))
			finalDf.set_value(index, 'LABELS', crimeToClass[crimeType])
			print("Datapoint appended.")

		else: 
			print("RETRIEVAL FAILED FOR CRIME ENDPOINT:", currentAddress)

values = finalDf.values 

trainingData = [row[-2:] for row in values]  
labels = [row[0] for row in values]

print("Training decision tree classifier.")
classifier = tree.DecisionTreeClassifier()
classifier.fit(trainingData, labels)

print("Rendering visualization.")
with open("vis.dot", 'w') as f:
	f = tree.export_graphviz(classifier, out_file=f)