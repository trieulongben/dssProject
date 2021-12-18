from flask import Flask,render_template,request,redirect,url_for,session
app = Flask(__name__,template_folder='')
app.secret_key = 'motconvitxeorahaicaicanh'
import requests

################################################################################
#Plot
import plotly
import plotly.express as px
import json
#Function 1 Location
import numpy as np
import pandas as pd 
import sklearn
from sklearn.cluster import KMeans
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
def location_based_recommendation(data, latitude, longitude):

    # Putting the Coordinates of Restaurants together into a dataframe
    coordinates = data[['longitude','latitude']]

    kmeans = KMeans(n_clusters = 10, init = 'k-means++')
    kmeans.fit(coordinates)
    y = kmeans.labels_

    data['cluster'] = kmeans.predict(data[['longitude','latitude']])
    top_restaurants_portland = data.sort_values(by=['stars', 'review_count'], ascending=False)

    
    """Predict the cluster for longitude and latitude provided"""
    cluster = kmeans.predict(np.array([longitude,latitude]).reshape(1,-1))[0]
    
   
    """Get the best restaurant in this cluster along with the relevant information for a user to make a decision"""
    return top_restaurants_portland[top_restaurants_portland['cluster']==cluster].iloc[0:10][['name', 'latitude','longitude','categories','stars', 'review_count','cluster']]

################################################################################


def makeObject(df):
    objectList=[]
    for index, row in df.iterrows():
        item={
            'type':'scattermapbox',
            'name':row['name'],
            'lat':[row['latitude']],
            'lon':[row['longitude']],
            'cate':[row['categories']],
            'mode':'markers',
            'marker': {
            'size': 14
        }
        }
        objectList.append(item)
    return objectList
    ################################################################################

#Function 2


def content_based_recommendations(name):
    data2=pd.read_csv('data/bagofWord.csv')
    data2=data2.set_index('name')
    count = CountVectorizer()
    count_matrix = count.fit_transform(data2['bag_of_words'])
    indices = pd.Series(data2.index)

    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    recommended_restaurants = []
    
    idx = indices[indices == name].index[0]

    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    top_10_indexes = list(score_series.iloc[1:11].index)
    
    for i in top_10_indexes:
        recommended_restaurants.append(list(data2.index)[i])
        
    return recommended_restaurants
################################################################################
#Routes

@app.route("/")
def main():
    return "Welcome!"
@app.route("/location", methods=['GET','POST'])
def location():
    output=None
    isFunc1=1
    isFunc2=1
    try:
        print(request.form['latitude'])
    except:
        isFunc1=None
    #
    
    
    if request.method == 'POST' and isFunc1!=None:
        data1=pd.read_csv('data/business_final.csv')
        lat=float(request.form['latitude'])
        long=float(request.form['longitude'])
        df=(location_based_recommendation(data1,long,lat))
        #Infomation to output
        output=makeObject(df)
        item={
            'type':'scattermapbox',
            'name':'Your position',
            'lat':[lat],
            'lon':[long],
            'cate':['None'],
            'mode':'markers',
            'marker': {
            'size': 18
            }
        }
        output.append(item)
        print(output)
        #tEST
    elif request.method == 'GET' and request.args.get('name')!=None and request.args.get('func')=='2'  :
        print(request.args.get('name'))
        name= request.args.get('name')
        session['name']=name
        return redirect(url_for('content',name=name))
    elif request.method == 'GET' and request.args.get('name')!=None and request.args.get('func')=='3'  :
        print(request.args.get('name'))
        name= request.args.get('name')
        session['name']=name
        return redirect(url_for('collborative',name=name))
    else:
        print('absasa')
        print(request.args.get('name'))
        print(type(request.args.get('func')))
        print(isFunc1)
    return render_template('func1.html',output=output)

@app.route("/content", methods=['GET','POST'])
def content():
    if session['name'] :
        data1=pd.read_csv('data/business_final.csv')
        input=session['name']
        
        name2=content_based_recommendations(input)

        df_output2=data1[data1['name']==name2[0]]
        for i in name2[1:]:
            df_output2=df_output2.append(data1[data1['name']==i])
        output=makeObject(df_output2)
        print( df_output2)
    return render_template('func2.html',output=output)

@app.route("/collaborative", methods=['GET','POST'])
def collborative():
    if session['name'] :
        data1=pd.read_csv('data/business_final.csv')
        input=session['name']
        
        name3=func3(input)

        df_output3=data1[data1['name']==name3.index[0]]
        for i in name3.index[1:]:
            df_output3=df_output3.append(data1[data1['name']==i])
        output=makeObject(df_output3)
        print( df_output3)
    return render_template('func3.html',output=output)
################################################################################










#Func3
def mean_center_rows(df):
    return (df.T - df.mean(axis = 1)).T
def score(data):
    # Computing Super-Score Rating for Reviews
    data['super_score'] = data['polarity'] *  data['compound']
    data['super_score'] = data['super_score'] + data['stars']

    return data
def cos_matrix(datas):
    # Combining the text in Keywords and categories columns
    # data['All_Keywords'] = data['categories'].str.cat(data['Keywords'],sep=", ")
    
#####################################################################################################
#Optimizeeeeeeeeeeeeeeeee
    #### Delete not needed
    datas['name']=datas['name'].astype('string')
    datas['super_score']=datas['super_score'].astype('float32')
    datas.drop(['Unnamed: 0','business_id','Keywords','categories'],axis=1,inplace=True)
    
    
    datas['user_id']=datas['user_id'].sort_values()
    
    
    #Attribute for making dataframe function(1)
    columns=datas['name'].unique()
#####################################################################################################
    datas=datas.pivot_table(index = 'user_id', columns = 'name', values = 'super_score')
    datas=datas.fillna(0)
    #Attribute for making dataframe function(2)
    user_ids=list(datas.index)
    
    
    
    
#Data numpy()
    
    datas=datas.to_numpy()
    
    #Mean
    datas=(datas.T - datas.mean(axis = 1)).T
    
    
# SVD
    U, sigma, Vt = svds(datas, k = 15)
    sigma = np.diag(sigma)
    
    # Overview of user ratings across all Restaurants in Toronto
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 

#Output
    # Converting the reconstructed matrix back to a Pandas dataframe
    print(all_user_predicted_ratings.shape)
    print(len(user_ids))
    print(len(columns))
    cf_preds_df = pd.DataFrame(all_user_predicted_ratings, columns = columns, index=user_ids).transpose()

    return cf_preds_df

def item_matrix(cf_preds_df):
    # Creating Item-Item Matrix based on Cosine Similarity
    item_item_matrix = cosine_similarity(cf_preds_df)
    item_item_matrix= pd.DataFrame(item_item_matrix, columns=cf_preds_df.index, index = cf_preds_df.index)

    return item_item_matrix
def load_data(url):
    data = pd.read_csv(url)
    return data
def func3(name):
    
    portland_data = load_data('data/portland_data2.csv')
    cf_preds_df = cos_matrix(portland_data)
    item_item_matrix = item_matrix(cf_preds_df)
    return cf_recommender(name,cf_preds_df,item_item_matrix)



# Creating Collaborative Filtering Function for Restaurant-Restaurant Recommendation System
def cf_recommender(restaurant,cf_preds_df,item_item_matrix):
    
    restaurant_ratings = cf_preds_df.T[restaurant]
    similar_restaurant_ratings = cf_preds_df.T.corrwith(restaurant_ratings)
    corr_ratings = pd.DataFrame(similar_restaurant_ratings, columns=['Correlation'])
    corr_ratings.dropna(inplace=True)
    
    ratings_sim = item_item_matrix[restaurant]
    
    ratings_sim = ratings_sim[ratings_sim>0]
    
    return ratings_sim.sort_values(ascending= False).head(11)[1:]
if __name__ == "__main__":
    app.run(debug=True)