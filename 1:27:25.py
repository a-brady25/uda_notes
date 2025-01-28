import nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd
import pyarrow

lyrics_pd = pd.read_feather('/Users/alainabrady/Desktop/Unstructured Data/complete_lyrics_2025.feather')

lyrics_pd.lyrics = lyrics_pd.lyrics.astype(str)

lyrics_sample = lyrics_pd.sample(1, random_state=42)

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

lyrics_sample['lyric_lemma'] = lyrics_sample.lyrics.apply(lemmatize_text)

print(lyrics_sample.lyric_lemma)
lyrics_sample.columns
lyrics_sample.song_x
lyrics_sample.artist_x
lyrics_sample.lyrics

Closer = lyrics_sample

Closer.lyric_lemma

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer()

Closer_tfidf = tfidf_vec.fit_transform(Closer['lyrics'])

tfidf_tokens = tfidf_vec.get_feature_names_out()

df_countvect = pd.DataFrame(data = songs_tfidf.toarray(), 
  columns = tfidf_tokens)

df_countvect.columns


closer = Closer["lyrics"].str.contains("closer", case = False, na = False)
closer

lyrics_sample['lyric_lemma']
lyrics_sample['lyrics']


## ok really not sure what's going on here because the lyrics aren't lining up with the song and artist names so I'm going to try the last statements

statements = pd.read_csv('/Users/alainabrady/Desktop/Unstructured Data/last_statements.csv')
statements.head()
print(statements.columns)

#tf-idf
tfidf_vec = TfidfVectorizer()

statements = statements.dropna(subset=['statements'])
statement_tfidf = tfidf_vec.fit_transform(statements['statements'])

tfidf_tokens = tfidf_vec.get_feature_names_out()

df_countvect = pd.DataFrame(data = statement_tfidf.toarray(), 
  columns = tfidf_tokens)

first_statement = statements.iloc[0]['statements']
second_statment = statements.iloc[1]['statements']

print(first_statement)
print(second_statment)

first_statement_split = first_statement.split(',')[0]
print(first_statement_split)
second_statement_split = second_statment.split(',')[0]
print(second_statement_split)

### Levenshtein
import textdistance

textdistance.levenshtein(first_statement_split, second_statement_split)

