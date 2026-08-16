import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
import wandb
import pandas as pd
import boto3
import uuid
import time
from datetime import datetime, timezone

class RecommendationRequest(BaseModel):
    title: str
    author: Optional[str] = None

class FeedbackRequest(BaseModel):
    request_id: str
    is_good_recommendation: bool

class RecommendationResponse(BaseModel):
    genre: str
    rec_1: str
    rec_2: str
    cached: bool = False
    request_id: str

def load_model():
    '''Lodas current production model from W&B model registry'''
    try:
        api = wandb.Api()
        artifact = api.artifact('wandb-registry-model/book-genre-classifier:production')
        artifact_dir = artifact.download()
        model = joblib.load(f'{artifact_dir}/review_model.pkl')
        return model
    except Exception as e: 
        print(f'Error: {e}')
        return None 

def load_books():
    '''Loads filtered books dataframe used for genre-based recommendations'''
    try:
        return pd.read_csv('filtered_books.csv')
    except Exception as e:
        print(f'Error loading books data: {e}')
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    app.state.books =load_books()
    dynamodb = boto3.resource('dynamodb', region_name = 'us-east-1')
    app.state.logs_table = dynamodb.Table('prediction_logs')
    app.state.cache_table = dynamodb.Table('recommendation_cache')
    yield

app = FastAPI(
    title = 'Prediction Model from Registry',
    version = '1.0.0',
    lifespan=lifespan
)

@app.get('/')
def intro():
    '''
    Intro when API is loaded
    '''
    return{'message': 'Hello! Welcomet to the Book Recommendation API'}

@app.get('/health')
def health_check():
    '''
    Health Check Endpoint
    Endpoint that verifies the API server is running and responding
    '''
    model_loaded = app.state.model is not None
    books_loaded = app.state.books is not None
    ok = model_loaded and books_loaded
    return{'status': 'ok' if ok else 'degraded', 'model_loaded': model_loaded, 'books_loaded': books_loaded}

@app.post('/feedback')
async def submit_feedback(feedback: FeedbackRequest):
    '''
    Records user feedback on a prediction for live accuracy tracking
    '''
    try: 
        app.state.logs_table.update_item(
            Key={'request_id': feedback.request_id},
            UpdateExpression = 'SET feedback = :fb',
            ExpressionAttributeValues={':.fb': feedback.is_good_recommendation}
        )
        return {'status': 'feedback recorded'}
    except Exception as e: 
        raise HTTPException(status_code=500, detail = f'failed to record feedback: {e}')

def log_prediction(app,title,genre,rec_1,rec_2, cached,latency_ms,request_id=None):
    '''
    Logs every prediction request to DynamoDB
    '''
    try: 
        app.state.logs_table.put_item(Item={
            'request_id': str(uuid.uuid4()),
            'title': title,
            'genre': genre,
            'rec_1': rec_1,
            'rec_2': rec_2,
            'cached': cached, 
            'latency_ms': str(round(latency_ms,2)),
            'timestamp': datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e: 
        print(f'Failed to log prediction: {e}')

@app.post('/predict', response_model=RecommendationResponse)
async def predict_recommendation(request: RecommendationRequest):
    '''
    Predict Endpoint
    Endpoint that takes user input text and returns two book recommendations
    '''
    start_time = time.time()

    if app.state.model is None or app.state.books is None:
        raise HTTPException(status_code=503, detail='Model or data is not loaded')

    title = request.title.strip()
    title_key= title.lower()

    #check if already cached
    try: 
        cached_item = app.state.cache_table.get_item(Key={'title_key': title_key}).get('Item')
    except Exception as e: 
        print(f'Cache read failed: {e}')
        cached_item = None

    if cached_item:
        latency_ms = (time.time() - start_time) * 1000       
        log_prediction(app,title,cached_item['genre'], cached_item['rec_1'], cached_item['rec_2'], cached = True, latency_ms=latency_ms)
        return RecommendationResponse(
            genre=cached_item['genre'],
            rec_1=cached_item['rec_1'],
            rec_2=cached_item['rec_2'],
            cached = True,
            request_id = cached_item.get('request_id', str(uuid.uuid4())),
        )

    # not cached

    try:
        predicted_genre = app.state.model.predict([title])[0]
    except Exception:
        raise HTTPException(status_code=500, detail='Inference failed')

    pool = app.state.books[app.state.books['genre'] == predicted_genre]

    if len(pool) < 2:
        raise HTTPException(status_code=404, detail=f'Not enough books found in genre "{predicted_genre}"')

    picks = pool.sample(2)
    rec_1 = picks.iloc[0]['Title']
    rec_2=picks.iloc[1]['Title']

    # write the cache for next time 

    try: 
        app.state.cache_table.put_item(Item={
            'title_key': title_key,
            'genre': predicted_genre,
            'rec_1': rec_1,
            'rec_2': rec_2,
        })
    except Exception as e: 
        print(f'Failed to write cache: {e}')

    # log the request 
    latency_ms = (time.time() - start_time) * 1000
    request_id = str(uuid.uuid4())
    log_prediction(app,title,predicted_genre,rec_1,rec_2,cached=False,latency_ms=latency_ms, request_id=request_id)

    return RecommendationResponse(
        genre=predicted_genre,
        rec_1=picks.iloc[0]['Title'],
        rec_2=picks.iloc[1]['Title'],
        cached=False,
        request_id=request_id,
    )

