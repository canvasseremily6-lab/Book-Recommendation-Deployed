import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score
import boto3

st.set_page_config(page_title='Model Monitoring Dashboard', layout = 'wide')

@st.cache_data(ttl = 30)
def load_logs(): 
    dynamodb = boto3.resource('dynamodb', region_name = 'us-east-1')
    table = dynamodb.Table('prediction_logs')

    items = []
    response = table.scan()
    while 'LastEvaluatedKey' in response: 
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        items.extend(response['Items'])

    df = pd.DataFrame(items)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['latency_ms'] = pd.to_numeric(df['latency_ms'], errors = 'coerce')
        df = df.sort_values('timestamp')

    return df

st.title('Model Monitoring Dashboard')

try: 
    df = load_logs()
except Exception as e: 
    st.error(f'COuld not load logs from DynamoDB: {e}')
    st.stop()

if df.empty:
    st.warning('No prediction logs found')
    if st.button('Refresh Data'):
        st.cache_data.clear()
        st.rerun()
    st.stop()

# prediction latency over time 
st.subheader('Predicction Latency Over Time')
latency_by_time = df.set_index('timestamp')['latency_ms']
st.line_chart(latency_by_time)
col1, col2, col3 = st.columns(3)
col1.metric('Average Latency', f'{df["latency_ms"].mean():.1f} ms')
col2.metric('Minimum Latency', f'{df["latency_ms"].min():.1f} ms')
col3.metric('Maximum Latency', f'{df["latency_ms"].max():.1f} ms')

st.divider()

#target drift 
st.subheader('Distribution of Prediction Genres')
genre_counts = df['genre'].value_counts()
st.bar_chart(genre_counts)

df['date'] = df['timestamp'].dt.date
drift_pivot = df.groupby(['date', 'genre']).size().unstack(fill_value=0)
st.line_chart(drift_pivot)

st.divider()

#accuracy from suer feedback
st.subheader('Live Accuracy (User Feedback)')
if 'feedback' in df.columns:
    feedback_df = df.dropna(subset=['feedback'])
    if not feedback_df.empty: 
        accuracy = feedback_df['feedback'].mean() *100
        col1,col2 = st.columns(2)
        col1.metric('Live Accuracy', f'{accuracy:.1f}%')
        col2.metric('Total Feedback Recived', len(feedback_df))

        st.markdown('**Accuracy over time:**')
        feedback_df = feedback_df.sort_values('timestamp')
        feedback_df['rolling_accuracy'] = feedback_df['feedback'].expanding().mean() *100
        st.line_chart(feedback_df.set_index('timestamp')['rolling_accuracy'])
    else: 
        st.info('No feedback submitted yet')
else:
    st.info('No feedback submitted yet')

st.divider()

st.subheader('Recent Requsts')
display_cols = ['timestamp', 'title', 'genre', 'latency_ms', 'cached']
if 'feedback' in df.columns:
    display_cols.append('feedback')
st.dataframe(df[display_cols].sort_values('timestamp', ascending=False).head(50), use_container_width=True)

if st.button('Refresh Data'):
    st.cache_data.clear()
    st.rerun()