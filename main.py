def main():
    import pandas as pd
    import numpy as np
    import sys
    #Machine Learning modeling librabries
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import r2_score
    from sklearn.metrics.pairwise import linear_kernel
    from sklearn.feature_extraction import text
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    from sklearn.metrics.pairwise import haversine_distances
    from math import radians,sqrt
    from tqdm import tqdm
    from sklearn.cluster import KMeans
    import pickle
    #############################################################################################################################
    df=pd.read_csv("C:/Users/DELL/Desktop/Restaurant recomendation System/python files/data.csv")
    df_copy=df.copy()
    df_copy_onehot = pd.get_dummies(df_copy[['type']])
    df_copy_onehot['location'] = df_copy['location']
    rest_name_grouped = df_copy_onehot.groupby('location').mean().reset_index()
    """ 
    def return_most_restaurants(row, top_restaurants):
        row_categories = row.iloc[1:]
        row_categories_sorted = row_categories.sort_values(ascending=False)
        return row_categories_sorted.index.values[0:top_restaurants]
    #############################################################################################################################

    top_restaurants = 8
    indicators = ['st', 'nd', 'rd']

    # create columns according to number of top venues
    columns = ['location']
    for ind in np.arange(top_restaurants):
        try:
            columns.append('{}{} Most Common Type'.format(ind+1, indicators[ind]))
        except:
            columns.append('{}th Most Common Type'.format(ind+1))

    # create a new dataframe
    restaurant_names_sorted = pd.DataFrame(columns=columns)
    restaurant_names_sorted['location'] = rest_name_grouped['location']

    for ind in np.arange(rest_name_grouped.shape[0]):
        restaurant_names_sorted.iloc[ind, 1:] = return_most_restaurants(rest_name_grouped.iloc[ind, :], top_restaurants)


    #######################################################################################################################
    
    sse = {}
    for k in range(1,8):
        kmeans = KMeans(n_clusters=k,random_state=0)
        kmeans.fit(rest_name_grouped.drop('location',axis=1))
        rest_name_grouped['Cluster'] = kmeans.labels_
        sse[k] = kmeans.inertia_
    kmeans = KMeans(n_clusters=3,random_state=2)
    kmeans.fit(rest_name_grouped.drop('location',axis=1))
    rest_name_grouped.groupby('Cluster')['location'].count()
    restaurant_names_sorted = restaurant_names_sorted.merge(rest_name_grouped,on='location')
    restaurant_names_sorted = restaurant_names_sorted.merge(df,on='location')

    ########################################################################################################################
    delivery_type = df_copy_onehot.groupby(['location']).sum().reset_index()
    delivery_type = delivery_type[['location','type_Delivery']]
    delivery_type['other_types'] = 1-(delivery_type['type_Delivery']/delivery_type['type_Delivery'].max(axis=0))
    delivery_type.rename(columns={'location':'locs'},inplace=True)
    target_cluster_dataframe = restaurant_names_sorted.loc[restaurant_names_sorted['location']=='BTM']
    target_cluster = target_cluster_dataframe.iloc[0].at['Cluster']
    recommendable_restaurants = restaurant_names_sorted[restaurant_names_sorted['Cluster']==target_cluster]
    recommendable_restaurants['Ranking'] = (recommendable_restaurants['rate']/recommendable_restaurants['rate'].max(axis=0)) * 0.5 + (recommendable_restaurants['cost']/recommendable_restaurants['cost'].max(axis=0)) * 0.1 + (recommendable_restaurants['votes']/recommendable_restaurants['votes'].max(axis=0)) * 0.1
    recommended_locations = recommendable_restaurants.sort_values(by='Ranking',ascending=False)
    recommended_locations.reset_index(inplace=True, drop=True)
    top3 = recommended_locations.groupby(['location','1st Most Common Type','2nd Most Common Type','3rd Most Common Type'])['Ranking'].unique()
    top3_df = pd.DataFrame(top3).reset_index()
    top3_df.head(3)
"""
    ######################################################################################################################W####

    def get_top_words(column, top_nu_of_words, nu_of_word):
        vec = CountVectorizer(ngram_range= nu_of_word, stop_words='english')
        bag_of_words = vec.fit_transform(column)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:top_nu_of_words]
    df.sample(3)
    df_percent = df.sample(frac=0.5)
    df_percent.set_index('name', inplace=True)
    df_percent.fillna('0',inplace=True)
    indices = pd.Series(df_percent.index)
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_percent['reviews_list'])
    cosine_similarities = linear_kernel(tfidf_matrix)

    def recommend(name, cosine_similarities = cosine_similarities):
        recommend_restaurant = []
        idx = indices[indices == name].index[0]
        score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)
        top30_indexes = list(score_series.iloc[0:31].index)
        
        for each in top30_indexes:
            recommend_restaurant.append(list(df_percent.index)[each])
        
        df_new = pd.DataFrame(columns=['name','cuisines', 'Rating', 'cost', 'Top reviews', 'address', 'phone number','type','url'])
        
        for each in recommend_restaurant:
            df_new = df_new.append(pd.DataFrame(df_percent[['name','cuisines', 'Rating', 'cost', 'Top reviews', 'address', 'phone number','type','url']][df_percent.index == each].sample()))
        
        df_new = df_new.drop_duplicates(subset=['name','cuisines', 'Rating', 'cost', 'Top reviews', 'address', 'phone number','type','url'], keep=False)
        df_new = df_new.sort_values(by='Rating', ascending=False).head(10)
        
        print('TOP %s RESTAURANTS LIKE %s WITH SIMILAR REVIEWS: ' % (str(len(df_new)), name))
        
        return df_new

if __name__==main():
    main().run()

