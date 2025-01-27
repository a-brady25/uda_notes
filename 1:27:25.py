import nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd
import pyarrow

# You will only need to do this once:

#nltk.download('wordnet')
#nltk.download('omw-1.4')

# Obs -- you'll need to import nltk.

lyrics_pd = pd.read_feather('/Users/alainabrady/Desktop/Unstructured Data/complete_lyrics_2025.feather')

lyrics_pd.lyrics = lyrics_pd.lyrics.astype(str)

lyrics_sample = lyrics_pd.sample(1)

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

lyrics_sample['lyric_lemma'] = lyrics_sample.lyrics.apply(lemmatize_text)

print(lyrics_sample.lyric_lemma)
lyrics_sample.head()
lyrics_sample.columns
lyrics_sample.song_x
lyrics_sample.song_y
lyrics_sample.lyrics

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer()

songs_tfidf = tfidf_vec.fit_transform(lyrics_sample['lyrics'])

tfidf_tokens = tfidf_vec.get_feature_names_out()

df_countvect = pd.DataFrame(data = songs_tfidf.toarray(), 
  columns = tfidf_tokens)

df_countvect.columns
df_countvect = df_countvect.drop(columns=['3embed'])
