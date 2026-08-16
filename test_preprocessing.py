# preprocessing.py
import ast

def clean_category(x):
    try:
        parsed = ast.literal_eval(x)
        return parsed[0] if isinstance(parsed, list) and parsed else x
    except:
        return x

def map_genres(cat):
    cat = str(cat).lower()
    if 'science fiction' in cat or 'fantasy' in cat:
        return 'Sci-Fi & Fantasy'
    elif 'mystery' in cat or 'thriller' in cat or 'suspense' in cat or 'crime' in cat or 'detective' in cat:
        return 'Mystery & Thriller'
    elif 'romance' in cat:
        return 'Romance'
    elif 'juvenile' in cat or 'children' in cat or 'young adult' in cat:
        return 'Children'
    elif 'biography' in cat or 'autobiography' in cat or 'memoir' in cat:
        return 'Biography'
    elif 'history' in cat or 'historical' in cat:
        return 'History'
    elif 'business' in cat or 'economics' in cat or 'finance' in cat:
        return 'Business'
    elif 'self-help' in cat or 'self help' in cat:
        return 'Self-Help'
    elif 'religion' in cat or 'spirituality' in cat or 'christian' in cat:
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