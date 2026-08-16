import streamlit as st
import requests
import os

API = os.environ.get('API_URL','http://127.0.0.1:8000')

st.title('Book Recommendations System')
st.markdown('Tell me your favorite book and I will give you two recommendations!')

genres = ['Other', 'Fiction', 'Children', 'Religion & Spirituality', 'Biography', 'History', 'Business', 'Self-Help', 'Mystery & Thriller', 'Sci-Fi & Fantasy']

book_input = st.text_area(label = 'Favorite book', placeholder = 'What is the title of your favorite book?')
author_input = st.text_area(label = 'Author', placeholder = 'Who is the author?')

if st.button(label = 'Analyze'): 
    if book_input.strip() == '' and author_input.strip() == '':
        st.error('***Please input your favorite book so that I can make recommendations***')
    else: 
        try: 
            response = requests.post(
                f'{API}/predict',
                json = {'title': book_input.strip(), 'author': author_input.strip() or None}
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


if 'error' in st.session_state:
    st.error(st.session_state['error'])

if 'result' in st.session_state:
    result = st.session_state['result']
    st.write(f"Genre: {result['genre']}")
    st.success(
        f'Here are recommedndations based on your favorite book:  \n'
        f'Recommendation One: "{result["rec_1"]}"  \n'
        f'Recommendation Two: "{result["rec_2"]}"'
    )

    st.markdown('**Was this a good recommendation?**')
    col1, col2 = st.columns(2)
    with col1: 
        if st.button('Yes'):
            requests.post(f'{API}/feedback', json={
                'request_id': result['request_id'],
                'is_good_recommendation': True
            })
            st.toast('Thank you for the feedback')
    with col2:
        if st.button('No'):
            requests.post(f'{API}/feedback', json= {
                'request_id': result['request_id'],
                'is_good_recommendation': False
            })
            st.toast('Thank you for the feedback')


