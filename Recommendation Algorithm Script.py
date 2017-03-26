# -*- coding: utf-8 -*-
"""
Created on Sat Mar 04 09:33:45 2017

@author: Luis
"""

import pandas as pd
import graphlab

"""To run this script, simply import the attached excel files I have
structured and queried. There should be 4 different .csv files to read."""

"""All you have to do is read in those files, and simply run the either of the two functions {popularitymodel(), 
and item_based_content_model(). } Each of these two functions will take you to a local web server
where you can analyze your recommendations, type a user_id, video_id, or title, and see the recommendations 
as well as the rank of the recommendations. Hope you enjoy my script."""

"""E.g.: popularitymodel('8f33a9e0-35f6-b8ee-3e7e-5805950b6b4d', 5) returns the top 5 most recommended videos
for user_id '8f33a9e0-35f6-b8ee-3e7e-5805950b6b4d'

item_based_content_model('18bjCHBZBEQAqS22WgeeCk', 5) returns the top 5 most recommended videos
for video_id '18bjCHBZBEQAqS22WgeeCk'
"""

# pass in column names for each CSV and read them using pandas. 
events_data_popular = pd.read_csv('C:/Users/Luis/Documents\python_import.csv')
events_data_proportionwatched = pd.read_csv('C:/Users/Luis/Documents\python_import2.csv')
user_data = pd.read_csv('C:/Users/Luis/Documents\python_import3.csv')
video_data = pd.read_csv('C:/Users/Luis/Documents\popularvideos.csv')

#Create Sframes for video data and user data
user_data_graph = graphlab.SFrame(user_data)
video_data_graph = graphlab.SFrame(video_data)

del events_data_popular['Unnamed: 0'] #deletes extra column
del events_data_proportionwatched['Unnamed: 0'] #deletes extra column
del user_data['Unnamed: 0'] #deletes extra column
del video_data['Unnamed: 0'] #deletes extra column

def popularitymodel(userlookup, numrank):
    #split the user events information between training, and test set
    events_base_popular = events_data_popular.iloc[:582385]
    events_test_popular= events_data_popular[582385:]


    #turn the training and testing data into SFrames
    train_data_popular = graphlab.SFrame(events_base_popular)
    test_data_popular = graphlab.SFrame(events_test_popular)

    """Here I create a recommendation engine that is based on the popularity 
    of the different tastemade shows"""

    popularity_model = graphlab.popularity_recommender.create(train_data_popular, user_id='user_id', item_id='video_id', target='cumulative_rank')

    #Get recommendations for first 5 users and print them
    #users = range(1,6) specifies user ID of first 5 users
    #k=5 specifies top 5 recommendations to be given
    popularity_recomm = popularity_model.recommend([userlookup], k=numrank)
    #For a specific user: example: '8f33a9e0-35f6-b8ee-3e7e-5805950b6b4d'
    #popularity_model.recommend(['8f33a9e0-35f6-b8ee-3e7e-5805950b6b4d'], k=5)
   # popularity_model.recommend(userlookup, k=5)
    popularity_recomm.print_rows(num_rows=5*numrank)

    view = popularity_model.views.overview(validation_set=test_data_popular,
                                           user_data=user_data_graph,
                                           user_name_column='user_id',
                                           item_data=video_data_graph,
                                           item_name_column='title',
                                           item_url_column='web_url')
    view.show()
    
    
def item_based_content_model(userlookup, numrank):

    """ Data Preparation for User Based Collaberative Filtering Model"""

    #split the user events information between training, and test set
    events_base_proportionwatched = events_data_proportionwatched.iloc[:582385]
    events_test_proportionwatched= events_data_proportionwatched[582385:]


    #turn the training and testing data into SFrames
    train_data_proportionwatched = graphlab.SFrame(events_base_proportionwatched)
    test_data_proportionwatched = graphlab.SFrame(events_test_proportionwatched)


    """Here I create a recommendation algorithm that is based on similar users,
    using the Pearson similarity matrix"""

    item_sim_model = graphlab.item_similarity_recommender.create(train_data_proportionwatched, user_id='user_id', item_id='video_id', target='proportion_watched', similarity_type='pearson')
    #Make Recommendations:
    item_sim_recomm = item_sim_model.recommend([userlookup],k=numrank)
    item_sim_recomm.print_rows(num_rows=5*numrank)

    view = item_sim_model.views.overview(validation_set=test_data_proportionwatched,
                                         user_data=user_data_graph,
                                         user_name_column='user_id',
                                         item_data=video_data_graph,
                                         item_name_column='title',
                                         item_url_column='web_url')
                            
    view.show()
    

"""Assess model performance"""
#model_performance = graphlab.compare(test_data_proportionwatched, [popularity_model, item_sim_model])
#graphlab.show_comparison(model_performance,[popularity_model, item_sim_model])

"""View model"""



