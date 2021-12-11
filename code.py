"""
Title: Subway Accessible ATM's
URL: https://saimantmg01.github.io/DS_Project/
Resources: Used https://www.geeksforgeeks.org/count-the-number-of-rows-and-columns-of-a-pandas-dataframe/ on how to see row and column data
           Used https://www.geeksforgeeks.org/find-duplicate-rows-in-a-dataframe-based-on-all-or-selected-columns/ on duplicated values to see in pandas 
           Used https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html on dropping_duplicates in pandas
           Used https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.groupby.DataFrameGroupBy.nunique.html on  counts of unique elements in each position.
           Used https://towardsdatascience.com/using-scikit-learns-binary-trees-to-efficiently-find-latitude-and-longitude-neighbors-909979bd929b as reference on how to use Ball Tree Algorithm to find nearest neighbor
           Used https://docs.huihoo.com/scikit-learn/0.20/modules/generated/sklearn.neighbors.BallTree.html as reference on Ball Tree method available in sklearn
           Used https://pandas.pydata.org/docs/user_guide/index.html#user-guide as reference for how to use pandas and its functions
           Used https://pandas.pydata.org/pandas-docs/dev/getting_started/intro_tutorials/index.html as reference on pandas basics
           Used https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iterrows.html as reference on iterrows function in pandas
           Used https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isin.html as reference for how isin function works in pandas
           Used Stack overflow for debugging purposes
           Used https://python-visualization.github.io/folium/modules.html#folium.vector_layers.PolyLine as reference on folium features
           Used https://python-visualization.github.io/folium/quickstart.html as reference on how to use folium.
           Used own program from program #14- library cleaning to extract lat and lon and modify it little bit based on functionality
           Used https://towardsdatascience.com/tree-algorithms-explained-ball-tree-algorithm-vs-kd-tree-vs-brute-force-9746debcd940 as reference on undrstanding balltree algorithm
           Used https://en.wikipedia.org/wiki/Haversine_formula as reference on haversine function
           Got advice from Fourcan Abdullah on folium
           Datasets from following website for the project:
           https://catalog.data.gov/dataset/bank-owned-atm-locations-in-new-york-state 	
           https://data.cityofnewyork.us/Transportation/Subway-Stations/arq3-7z49
"""

import pandas as pd
import numpy as np
import re
from sklearn.neighbors import BallTree
import folium

"""
Gets the latitude and longitude.

This function isolates the targeted data which is a string and extracts the latitude and longitude. 
@param : row of a dataframe
@return: a series with longitude on 0 index and latitude on 1st index.

"""
def extractLatLon(row):
    row = str(row)
    #pattern represent the string with numerical values between parenthesis and spaces stored in a string
    pattern = r'(-[0-9]*\.[0-9]*\s[0-9]*\.[0-9]*)'
    #look for targeted values in a row
    answer = re.findall(pattern,row)
    #iterate through each element in the pattern
    for elem in answer:
        #split the targted values based on space which exist between two number
        answer = elem.split()
    #convert lon and lat from original degree to radians to perform haversine function upon them while using the Ball tree.
    return pd.Series([np.deg2rad(float(answer[0])), np.deg2rad(float(answer[1]))])


#get the dataframe from respective csv
bankATM = pd.read_csv("Bank-Owned_ATM_Locations_in_New_York_State.csv")
SubwayStation = pd.read_csv("DOITT_SUBWAY_STATION_01_13SEPT2010.csv")

###Data cleaning

#for  Bank_Owned_ATM locations
#dropping all the duplicates and only keeping the first instance
bank_unique_rows = bankATM.drop_duplicates() # 5497 rows. 25 removes
#dropNA -> get rid of rows in bank ATM data where there is no value 
bank_unique_rows = bank_unique_rows.dropna(subset = ['Georeference']) # 5162 rows. 335 rows removed
#reset the index of bank ATM
bank_unique_rows = bank_unique_rows.reset_index(drop= True) #reset the index

#uniformity - get all values of below mentioned to be capitalized  
cols = ['Name of Institution', 'Street Address', 'City', 'County']
#go through each column and capitalized its values.
for c in cols:
    bank_unique_rows[c] = bank_unique_rows[c].astype(str).str.upper()

#filtering Bank ATM dataframe to get ATM within NYC only
county_list = ['NEW YORK','KINGS', 'QUEENS','BRONX', 'RICHMOND'] #RICHMOND means staten island
#only get ATM within NYC
NYC_bank_ATMs = bank_unique_rows[bank_unique_rows['County'].isin(county_list)] #1653 rows
#reset the index
NYC_bank_ATMs = NYC_bank_ATMs.reset_index(drop=True)

#for Subway Station locations
#drop all the duplicates only keep the first instance
Subway_Station_unique_rows = SubwayStation.drop_duplicates() 
#drop rows with null values for the_geom column
Subway_Station_unique_rows = Subway_Station_unique_rows.dropna(subset = ['the_geom'])
#reset the index
Subway_Station_unique_rows = Subway_Station_unique_rows.reset_index(drop= True)

#uniformity -> capitalize following columns in the subway dataframe
cols_for_subway = ['NAME','LINE', 'NOTES']
for c in cols_for_subway:
    Subway_Station_unique_rows[c] = Subway_Station_unique_rows[c].astype(str).str.upper()

#extract latitude and longitude from Georeference dataframe and assign respective latitude and longitude to new columns
NYC_bank_ATMs[['Lon','Lat']] = NYC_bank_ATMs.apply(extractLatLon,axis=1, result_type='expand')
#extract latitude and longitude from the_geom in Subway Station dataframe and assign respective latitude and longitude to new columns
Subway_Station_unique_rows[['Lon','Lat']] = Subway_Station_unique_rows.apply(extractLatLon,axis=1, result_type='expand')


###Data Analysis

#take the NYC_bank_ATM latitude and longitude values to construct a ball tree
ball = BallTree(NYC_bank_ATMs[["Lat","Lon"]].values, metric='haversine')
#execute the query with Subway_Station_unique_rows on NYC bank ATM to look for neighbors and return the only one closest neighbor 
distances, indices = ball.query(Subway_Station_unique_rows[["Lat","Lon"]],k = 1)
#convert the distances to kilometers
distances = distances * 6371 #km

#flatten the indices from second dimension to one dimension to get number of rows.
flattened_indices = indices.flatten()
#get the respective name of ATM, its longitude, latitude in NYC_bank_ATM whose distance were closest to the subway station.
nearest_station_names, nearest_lon, nearest_lat = zip(*NYC_bank_ATMs.loc[flattened_indices][['Name of Institution', 'Lon', 'Lat']].to_numpy())
#create new columns to hold nearest atm, nearest lon, nearest lat, and distance of particular ATM from a given subway
Subway_Station_unique_rows['nearest_ATM'] = nearest_station_names
Subway_Station_unique_rows['nearest_lon'] = nearest_lon
Subway_Station_unique_rows['nearest_lat'] = nearest_lat
Subway_Station_unique_rows['nearest_dist'] = distances

#only get walkable distance of less than 0.5 km 
Subway_Station_unique_rows = Subway_Station_unique_rows[Subway_Station_unique_rows["nearest_dist"] < 0.5]

### Data Visualization

#create a world map centered in average location of all Subway Station dataframe
map = folium.Map(location=[np.degrees(Subway_Station_unique_rows["Lat"].mean()), np.degrees(Subway_Station_unique_rows["Lon"].mean())], zoom_start= 12, control_scale=True)

#iterate through each row (as index and row) in Subway_Station_unique_rows
for index, row in Subway_Station_unique_rows.iterrows():
    #create a marker in folium map which shows location of subway with popup of its name added to map.
    folium.Marker([np.degrees(row["Lat"]), np.degrees(row["Lon"])], icon= folium.Icon(color='blue'), popup=row["NAME"], tooltip= "Subway").add_to(map)
    #create a marker in folium map which shows location of ATM's with popup of its name added to map.
    folium.Marker([np.degrees(row["nearest_lat"]), np.degrees(row["nearest_lon"])], icon= folium.Icon(color='red'), popup=row["nearest_ATM"], tooltip= "ATMs").add_to(map)  
    location = [(np.degrees(row["Lat"]), np.degrees(row["Lon"])),(np.degrees(row["nearest_lat"]), np.degrees(row["nearest_lon"]))]
    #create a line in map using folium
    folium.PolyLine(location,color='red').add_to(map)
map.save('map.html')
