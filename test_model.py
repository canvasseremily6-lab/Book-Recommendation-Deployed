import pytest 
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest
import streamlit as st
from test_preprocessing import clean_category, map_genres
from main import app
from unittest.mock import MagicMock, patch
import pandas as pd

backend = TestClient(app)

class TesetCleanCategory:
    def test_list_string_returns_elements(self):
        assert clean_category("['Fiction]") == 'Fiction'

    def test_list_string_with_mult_elements(self):
        assert clean_category("['Fiction', 'Drama']") == 'Fiction'

    def test_empty_list_returns(self):
        assert clean_category('[]') == '[]'

    def test_non_list_input(self):
        assert clean_category('not a list') == 'not a list'

    def test_none_input(self):
        result = clean_category(None)
        assert result is None

class TestMapGenres:
    def test_fiction(self):
        assert map_genres('Fiction') == 'Fiction'

    def test_science_fiction_and_fantasy(self):
        assert map_genres('Science Fiction & Fantasy') == 'Sci-Fi & Fantasy'

    def test_mystery_thrillre(self):
        assert map_genres('Mystery, Thriller & Suspense') == 'Mystery & Thriller'

    def test_children(self):
        assert map_genres('Juvenile Fiction') == 'Children'

    def test_case_insensitive(self):
        assert map_genres('FICTION') == 'Fiction'
        assert map_genres('fiction') == 'Fiction'

    def test_unmatched_cat(self):
        assert map_genres('random category') == 'Other'

    def test_nan_input(self):
        result = map_genres(float('nan'))
        assert result == 'Other'

class TestStreamlitInteraction:
    def test_app_run(self):
        app = AppTest.from_file('app.py')
        app.run(timeout=30)
        assert not app.exception

    def test_title(self):
        at = AppTest.from_file('app.py')
        at.run(timeout=30)
        assert 'Book Recommendations System'

@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = ['Fiction']
    return model


@pytest.fixture
def mock_books():
    return pd.DataFrame({
        'Title': ['Book A', 'Book B', 'Book C'],
        'authors': ['Author A', 'Author B', 'Author C'],
        'genre': ['Fiction', 'Fiction', 'History'],
    })


@pytest.fixture
def client(mock_model, mock_books):
    with patch('main.load_model', return_value=mock_model), \
         patch('main.load_books', return_value=mock_books), \
         patch('main.boto3.resource') as mock_boto:

        mock_dynamodb = MagicMock()
        mock_boto.return_value = mock_dynamodb

        mock_cache_table = MagicMock()
        mock_cache_table.get_item.return_value = {}  # simulate cache miss
        mock_logs_table = MagicMock()

        def table_side_effect(name):
            return mock_cache_table if name == 'recommendation_cache' else mock_logs_table

        mock_dynamodb.Table.side_effect = table_side_effect

        from main import app
        with TestClient(app) as test_client:
            yield test_client


def test_root_endpoint(client):
    response = client.get('/')
    assert response.status_code == 200
    assert 'message' in response.json()


def test_health_check_returns_ok_when_loaded(client):
    response = client.get('/health')
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'ok'
    assert data['model_loaded'] is True
    assert data['books_loaded'] is True


def test_predict_returns_valid_response(client):
    response = client.post('/predict', json={'title': 'Some Book Title'})
    assert response.status_code == 200
    data = response.json()
    assert data['genre'] == 'Fiction'
    assert data['rec_1'] in ['Book A', 'Book B']
    assert data['rec_2'] in ['Book A', 'Book B']
    assert data['cached'] is False
    assert 'request_id' in data


def test_predict_requires_title(client):
    response = client.post('/predict', json={})
    assert response.status_code == 422


def test_predict_genre_with_insufficient_books_returns_404(client, mock_model):
    mock_model.predict.return_value = ['History']  # only 1 History book in mock_books
    response = client.post('/predict', json={'title': 'Some Book'})
    assert response.status_code == 404
