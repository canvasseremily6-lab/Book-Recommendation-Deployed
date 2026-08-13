import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
import wandb
import pandas as pd

class RecommendationRequest(BaseModel):
    title: str
    author: Optional[str] = None

class RecommendationResponse(BaseModel):
    genre: str
    rec_1: str
    rec_2: str

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

@app.post('/predict', response_model=RecommendationResponse)
async def predict_recommendation(request: RecommendationRequest):
    '''
    Predict Endpoint
    Endpoint that takes user input text and returns two book recommendations
    '''
    if app.state.model is None or app.state.books is None:
        raise HTTPException(status_code=503, detail='Model or data is not loaded')

    title = request.title.strip().lower()

    try:
        predicted_genre = app.state.model.predict([title])[0]
    except Exception:
        raise HTTPException(status_code=500, detail='Inference failed')

    pool = app.state.books[app.state.books['genre'] == predicted_genre]

    if len(pool) < 2:
        raise HTTPException(status_code=404, detail=f'Not enough books found in genre "{predicted_genre}"')

    picks = pool.sample(2)

    return RecommendationResponse(
        genre=predicted_genre,
        rec_1=picks.iloc[0]['Title'],
        rec_2=picks.iloc[1]['Title'],
    )

