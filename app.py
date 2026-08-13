import streamlit as st
import requests
import matplotlib.pyplot as plt
import joblib 
import pandas as pd
from random import sample
import wandb

API = 'http://127.0.0.1:8000'

@st.cache_data
def load_data():
    return pd.read_csv('filtered_books.csv')

filtered = load_data()

genres = ['Other', 'Fiction', 'Children', 'Religion & Spirituality', 'Biography', 'History', 'Business', 'Self-Help', 'Mystery & Thriller', 'Sci-Fi & Fantasy']

@st.cache_data(ttl = 30)
def load_model():
    '''Lodas current production model from W&B model registry'''
    api = wandb.Api()
    artifact = api.artifact('wandb-registry-model/book-genre-classifier:production')
    artifact_dir = artifact.download()
    model = joblib.load(f'{artifact_dir}/review_model.pkl')
    return model 

loaded_model = load_model()

st.title('Book Recommendations System')
st.markdown('Tell me your favorite book and I will give you two recommendations!')

book_input = st.text_area(label = 'Favorite book', placeholder = 'What is the title of your favorite book?')
author_input = st.text_area(label = 'Author', placeholder = 'Who is the author?')

if st.button(label = 'Analyze'): 
    if book_input.strip() == '' and author_input.strip() == '':
        st.error('***Please input your favorite book so that I can make recommendations***')
    else: 
        try: 
            response = requests.post(
                f'{API}/predict',
                json = {'title': book_input.strip(), 'auhtor': author_input.strip() or None}
            )
            if response.status_code == 200:
                st.session_state['result'] = response.json()
                st.session_state.pop('error',None)
            else:
                st.session_state['error'] = response.json().get('detail', 'Something went wrong')
                st.session_state.pop('result',None)
        except requests.exceptions.ConnectionError:
            st.session_state['error'] = 'Could not connect to the API. Is it running?'
            st.session_state.pop('result',None)
        #st.session_state['prediction'] = loaded_model.predict([book_input])

if 'error' in st.session_state:
    st.error(st.session_state['error'])

if 'result' in st.session_state:
    result = st.session_state['result']
    st.write(f"Genre: {result['genre']}")
    st.success(
        f'Here are recommedndations based on your favorite book:  \n'
        f'Recommendation One: "{result['rec_1']}"  \n'
        f'Recommendation Two: "{result['rec_2']}'
    )

# if 'prediction' in st.session_state:
#     pred = st.session_state['prediction'][0]

#     if pred in genres: 
#         genre_filtered = filtered[filtered['genre'] == pred]

#         if author_input in genre_filtered['authors'].values:
#             rec_1 = genre_filtered[genre_filtered['authors']==author_input].sample(1).iloc[0]
#             rec_2 = genre_filtered.sample(1).iloc[0]
#             st.write(f'Genre: {pred}')
#             st.success(f'Here are recommendations based on your favorite book:  \nRecommendation One (Same Author): "{rec_1["Title"]}" by {rec_1["authors"]}.  \nRecommendation Two: "{rec_2["Title"]}" by {rec_2["authors"]}')
#         else: 
#             recs = genre_filtered.sample(2)
#             st.write(f'Genre: {pred}')
#             st.success(f'Here are recommendations based on your favorite book:  \nRecommendation One: "{recs.iloc[0]["Title"]}" by {recs.iloc[0]["authors"]}.  \nRecommendation Two: "{recs.iloc[1]["Title"]}" by {recs.iloc[1]["authors"]}')


# print(filtered['genre'].value_counts())


