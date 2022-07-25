import string

import nltk
import ssl
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from lxml import etree
from sklearn.feature_extraction.text import TfidfVectorizer


try:
     _create_unverified_https_context =     ssl._create_unverified_context
except AttributeError:
     pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
punct = list(string.punctuation)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
root = etree.parse('news.xml').getroot()
articles = root[0]
stories = {}
all_nouns = []
for news in articles:
    news_string = news[1].text.lower()
    all_words = word_tokenize(news_string)
    normalised_words = [lemmatizer.lemmatize(w) for w in all_words]
    filtered_words = [w for w in normalised_words if w not in stop_words]
    punct_removed = [w for w in filtered_words if w not in punct]
    nouns = [w for w in punct_removed if nltk.pos_tag([w])[0][1] == 'NN']
    all_nouns.append(' '.join(nouns))
    stories[news[0].text] = nouns

vectorizer = TfidfVectorizer()
vectorizer.fit(all_nouns)

for index, story in enumerate(stories):
    print(f'{story}:')
    x = vectorizer.transform([all_nouns[index]])
    word_dict = dict(zip(vectorizer.get_feature_names_out(), x.toarray()[0]))
    story_words = {word: word_dict[word] for word in stories[story] if word in word_dict}
    story_words_ordered = [v[0] for v in sorted(story_words.items(), key=lambda x: (x[1], x[0]), reverse=True)][:5]
    print(*story_words_ordered)
