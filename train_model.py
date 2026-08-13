import kagglehub
import os
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import ast
import wandb
import subprocess

def get_git_commit():
    try: 
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    except Exception:
        return 'no-git-repo'

run = wandb.init(
    project = 'book-genre-classifier', 
    dir = 'logs',
    config={
        'max_features': 5000,
        'test_size': 0.2,
        'random_state': 303,
        'model': 'MultinomialNB',
        'vectorizer': 'TfidfVectorizer',
        'min_category_count': 20,
        'git_commit': get_git_commit(),
    }
)

path = kagglehub.dataset_download("mohamedbakhet/amazon-books-reviews")
reviews = pd.read_csv(os.path.join(path,'Books_rating.csv'))
books = pd.read_csv(os.path.join(path, 'books_data.csv'))

wandb.config.update({
    'dataset_path': path,
    'reviews_shape': reviews.shape,
    'books_shape': books.shape,
})

avg_ratings = reviews.groupby('Title')['review/score'].agg(['mean','count']).reset_index()
avg_ratings.columns = ['Title', 'avg_rating', 'num_of_ratings']
avg_ratings = avg_ratings[avg_ratings['num_of_ratings'] >= 10]

merged_books = pd.merge(books, avg_ratings, on ='Title', how = 'inner')
merged_books = merged_books.dropna(subset=['Title', 'categories'])

def clean_category(x): 
    '''Function to clean list-string formatting within dataset'''
    try: 
        parsed = ast.literal_eval(x)
        return parsed[0] if isinstance(parsed,list) and parsed else x
    except: 
        return x

merged_books['clean_categories'] = merged_books['categories'].apply(clean_category)
merged_books['authors'] = merged_books['authors'].apply(clean_category)
y = merged_books['clean_categories']

counts = y.value_counts()
valid_categories = counts[counts >= 20].index
filtered = merged_books[merged_books['clean_categories'].isin(valid_categories)].copy()

def map_genres(cat):
    cat = str(cat).lower()
    if 'fiction' in cat and 'juvenile' not in cat and 'science' not in cat: 
        return 'Fiction'
    elif 'science fiction' in cat or 'suspense' in cat: 
        return 'Sci-Fi & Fantasy'
    elif 'mystery' in cat or 'thriller' in cat or 'suspense' in cat: 
        return 'Mystery & Thriller'
    elif 'romance' in cat: 
        return 'Romance'
    elif 'juvenile' in cat or 'children' in cat: 
        return 'Children'
    elif 'biography' in cat or 'autobiography' in cat: 
        return 'Biography'
    elif 'history' in cat: 
        return 'History'
    elif 'business' in cat or 'economics' in cat: 
        return 'Business'
    elif 'self-help' in cat or 'self help' in cat: 
        return 'Self-Help'
    elif 'religion' in cat or 'spirituality' in cat: 
        return 'Religion & Spirituality'
    elif 'poetry' in cat:
        return 'Poetry'
    elif 'horror' in cat:
        return 'Horror'
    elif 'cooking' in cat or 'cookbook' in cat or 'food' in cat:
        return 'Cooking'
    elif 'health' in cat or 'fitness' in cat or 'medical' in cat:
        return 'Health & Fitness'
    elif 'travel' in cat:
        return 'Travel'
    elif 'humor' in cat or 'comic' in cat or 'graphic novel' in cat:
        return 'Humor & Comics'
    elif 'science' in cat and 'fiction' not in cat:
        return 'Science & Nature'
    elif 'fiction' in cat and 'juvenile' not in cat:
        return 'Fiction'
    else: 
        return 'Other'

filtered['genre'] = filtered['clean_categories'].apply(map_genres)
filtered['text'] = filtered['Title'].fillna(' ') + ' ' + filtered['description'].fillna('')
filtered.to_csv('filtered_books.csv', index = False)

wandb.log({'genre_distribution': wandb.Table(dataframe=filtered['genre'].value_counts().reset_index())})

X = filtered['text']
y = filtered['genre']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=303, stratify=y)

pipeline = Pipeline(steps = [
    ('vectorizer', TfidfVectorizer(max_features =5000)),
    ('naive_bayes', MultinomialNB())
])

pipeline.fit(X_train,y_train)
preds = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average='weighted')

wandb.log({'accuracy': accuracy, 'f1_score': f1})

report_dict = classification_report(y_test, preds, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().reset_index()
wandb.log({'classification_report': wandb.Table(dataframe=report_df)})

joblib.dump(pipeline, 'review_model.pkl')

artifact = wandb.Artifact('review_model', type = 'model',
                          description='Naive Bayes genre classifier trained on book title + description',
                          metadata= {
                              'accuracy': accuracy,
                              'f1_score': f1, 
                              'max_features': 5000,
                              'git_commit': get_git_commit(),
                          })
artifact.add_file('review_model.pkl')

run.log_artifact(artifact)

api = wandb.Api()

def get_best_staging():
    try:
        staged = api.artifact('wandb-registry-model/book-genre-classifier:staging')
        return staged.metadata.get('f1_score', 0)
    except Exception as e:
        print(f'get_best_staging failed: {e}')
        return 0 

best_f1 = get_best_staging()
aliases = ['staging'] if f1 > best_f1 else []

run.link_artifact(
    artifact=artifact, 
    target_path='wandb-registry-model/book-genre-classifier',
    aliases = aliases
)


if f1 > best_f1:
    print(f'New model promoted to staging (f1 = {f1:.3f} > previous best f1: {best_f1:.3f})')
else: 
    print(f'Model not promoted (f1 = {f1:.3f} <= staging f1 = {best_f1:.3f})')

run.finish()