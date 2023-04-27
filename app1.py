from flask import Flask, request, render_template
import main as m

app=Flask(__name__)

@app.route('/')
@app.route('/home')
def index():
    return render_template('main.html')

@app.route('/result', methods=['POST','GET'])

def result():
    input_value=request.form.to_string()
      
    output=m.recommend(input_value)
    return render_template('main.html', name=output)
""" 
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
        """

if __name__ == '__main__':
    app.run(debug=True)
    
    

