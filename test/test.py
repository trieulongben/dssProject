
KM = __import__("cluster")
LOC= __import__("location")
CT = __import__("content")

import numpy as np
import pandas as pd 
import sklearn
from sklearn.cluster import KMeans


# business_URL= "/home/cate/Cate/recommender_system/business_final.csv"
# def load_data(url):
#     data = pd.read_csv(url)
#     return data
data=pd.read_csv('data/business_final.csv')

# Creating Location-Based Recommendation Function
def location_based_recommendation(data, latitude, longitude):

    # Putting the Coordinates of Restaurants together into a dataframe
    coordinates = data[['longitude','latitude']]

    kmeans = KMeans(n_clusters = 10, init = 'k-means++')
    kmeans.fit(coordinates)
    y = kmeans.labels_

    data['cluster'] = kmeans.predict(data[['longitude','latitude']])
    top_restaurants_toronto = data.sort_values(by=['stars', 'review_count'], ascending=False)

    
    """Predict the cluster for longitude and latitude provided"""
    cluster = kmeans.predict(np.array([longitude,latitude]).reshape(1,-1))[0]
    
   
    """Get the best restaurant in this cluster along with the relevant information for a user to make a decision"""
    return top_restaurants_toronto[top_restaurants_toronto['cluster']==cluster].iloc[0:10][['name', 'latitude','longitude','categories','stars', 'review_count','cluster']]


df=location_based_recommendation(data, 43.6677, -79.3948)
#Infomation to output
names=df['name'].to_numpy()
lat=df['latitude'].to_numpy()
long=df['longitude'].to_numpy()
categories=df['categories'].to_numpy()
print(df)