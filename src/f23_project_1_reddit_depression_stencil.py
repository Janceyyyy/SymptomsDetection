# -*- coding: utf-8 -*-


! pip install happiestfuntokenizing
! pip install transformers
! pip install joblib

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from happiestfuntokenizing.happiestfuntokenizing import Tokenizer
from google.colab import drive
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
from transformers import RobertaModel, RobertaTokenizer
import torch
import re
from joblib import dump
import joblib


drive.mount('/content/drive')

FILEPATH = '/content/drive/MyDrive/Colab Notebooks/nlp/student.pkl'

"""## Preprocessing"""

def load(file_path):
  """Load pickles"""
  return pd.read_pickle(file_path)

def dataset_generation(df, mental_health_subreddits):
    """Build control and symptom datasets."""
    # Set a time threshold of 180 days in seconds
    time_threshold=180*24*60*60
    # Filter the DataFrame to include only posts from mental health subreddits
    symptom_posts = df[df['subreddit'].isin(mental_health_subreddits)]

    earliest_mental_health_post = symptom_posts.groupby('author')['created_utc'].min()

    # Filter control posts to only include those that are at least 180 days older than the earliest mental health post
    # and are not from mental health subreddits
    control_posts = pd.merge(df, earliest_mental_health_post, on='author')
    control_posts = control_posts[(control_posts['created_utc_x'] <= control_posts['created_utc_y'] - time_threshold) &
                                  (~control_posts['subreddit'].isin(mental_health_subreddits))]
    # Remove posts where the author is '[deleted]' to ensure data quality
    control_posts = control_posts[control_posts['author'] != '[deleted]']
    symptom_posts = symptom_posts[symptom_posts['author'] != '[deleted]']

    return control_posts, symptom_posts

# List of depression subreddits in the paper
depression_subreddits = ["Anger",
    "anhedonia", "DeadBedrooms",
    "Anxiety", "AnxietyDepression", "HealthAnxiety", "PanicAttack",
    "DecisionMaking", "shouldi",
    "bingeeating", "BingeEatingDisorder", "EatingDisorders", "eating_disorders", "EDAnonymous",
    "chronicfatigue", "Fatigue",
    "ForeverAlone", "lonely",
    "cry", "grief", "sad", "Sadness",
    "AvPD", "SelfHate", "selfhelp", "socialanxiety", "whatsbotheringyou",
    "insomnia", "sleep",
    "cfs", "ChronicPain", "Constipation", "EssentialTremor", "headaches", "ibs", "tinnitus",
    "AdultSelfHarm", "selfharm", "SuicideWatch",
    "Guilt", "Pessimism", "selfhelp", "whatsbotheringyou"
]

symptom_subreddit = {'Anger': ['Anger'],
                     'Anhedonia': ["anhedonia", "DeadBedrooms"],
                     'Anxiety': ["Anxiety", "AnxietyDepression", "HealthAnxiety", "PanicAttack"],
                     'Disordered eating': ["bingeeating", "BingeEatingDisorder", "EatingDisorders", "eating_disorders", "EDAnonymous"],
                     'Loneliness': ["ForeverAlone", "lonely"],
                     'Sad mood': ["cry", "grief", "sad", "Sadness"],
                     'Self-loathing': ["AvPD", "SelfHate", "selfhelp", "socialanxiety", "whatsbotheringyou"],
                     'Sleep problem': ["insomnia", "sleep"],
                     'Somatic complaint': ["cfs", "ChronicPain", "Constipation", "EssentialTremor", "headaches", "ibs", "tinnitus"],
                     'Worthlessness': ["Guilt", "Pessimism", "selfhelp", "whatsbotheringyou"]
                     }

def mask_urls(tokens):
  url_pattern = re.compile(r'http[s]?://\S+')
  url = ['[URL]' if url_pattern.match(token) else token for token in tokens]
  return url
def tokenize(data):
    """Tokenize"""

    tokenizer = Tokenizer(preserve_case=False)

    # Tokenize
    data['tokenized_text'] = data['text'].apply(lambda x: tokenizer.tokenize(x))
    #mask URLs
    data['tokenized_text'] = data['tokenized_text'].apply(mask_urls)

    # Convert tokens back to string
    data['tokenized_text_str'] = data['tokenized_text'].apply(lambda x: ' '.join(x))

    return data

def stop_words(texts,n_top_words):
  """Find top 100 words from Reddit dataset to use as stop words"""
  all_words = [word for text in texts for word in text.split()]
  word_counts = Counter(all_words)

  common_words = [word for word, count in word_counts.most_common(n_top_words)]
  return common_words

"""## Reddit Topics with LDA

 - Don't use MALLET (as the paper does), use some other LDA implementation.
"""

# TODO: Your LDA code!


def preprocess_and_apply_lda(data, n_components=200, n_top_words=100):
    data = tokenize(df)
    texts = data['tokenized_text_str'].tolist()
    common_words =  stop_words(texts,n_top_words)
    vectorizer = CountVectorizer(stop_words=common_words)
    X = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(n_components=n_components, random_state=0)
    lda.fit(X)
    doc_topic_distributions = lda.transform(X)

    return lda,doc_topic_distributions

"""## RoBERTa Embeddings"""

# TODO: Your RoBERTa code!
def get_embedding(texts,tokenizer,device, model,layer_num=10):
    encoded_input = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**encoded_input)
        hidden_states = outputs.hidden_states
    target_layer = hidden_states[layer_num]
    mean_embedding = torch.mean(target_layer, dim=1)
    return mean_embedding.cpu().numpy()

def roberta_embeddings(df, layer_num=10):
    """Generate embeddings for texts using RoBERTa."""
    model_name = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name, output_hidden_states=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    embeddings = df['text'].apply(lambda x: get_embedding(x,tokenizer,device,model))
    np.save('/content/drive/MyDrive/Colab Notebooks/nlp/embedding33.npy', embeddings)
    return np.vstack(embeddings.tolist())

"""## Main"""

def rf(embeddings,control_posts, symptom_posts,df,symptom_subreddit):
    """random forest"""

    train_scores_dict = {}
    test_scores_dict = {}

    # Iterate over each symptom and its corresponding subreddits
    for symptom, subreddits in symptom_subreddit.items():
        # Create a mask to filter out posts belonging to the current symptom's subreddits
        symptom_mask = df['subreddit'].isin(subreddits)
        # Extract embeddings for control and symptom posts
        control_embeddings = embeddings[:len(control_posts)]
        symptom_embeddings = embeddings[symptom_mask]
        # Combine control and symptom embeddings and create labels (0 for control, 1 for symptom)
        X = np.concatenate((control_embeddings, symptom_embeddings))
        y = np.concatenate((np.zeros(len(control_embeddings)), np.ones(len(symptom_embeddings))))
        rf_classifier = RandomForestClassifier()
        cv = KFold(n_splits=5, shuffle=True)
        results = cross_validate(rf_classifier, X=X, y=y, cv=cv, scoring='roc_auc', return_train_score=True)
        #Store the training and testing scores
        train_scores = results['train_score']
        test_scores = results['test_score']
        train_scores_dict[symptom] = results['train_score']
        test_scores_dict[symptom] = results['test_score']


    return train_scores_dict , test_scores_dict

def main():
  """
  Here's the basic structure of the main block! It should run
  5-fold cross validation with random forest to evaluate your RoBERTa and LDA
  performance.
  """
  df = load(FILEPATH)
  control_posts, symptom_posts = dataset_generation(df,depression_subreddits)
  df = pd.concat([control_posts, symptom_posts])
  lda_model,doc_topic_distributions = preprocess_and_apply_lda(df)
  #dump(doc_topic_distributions, '/content/drive/MyDrive/Colab Notebooks/doc_topic_distributions.joblib')
  #doc_topic_distributions = joblib.load('/content/drive/MyDrive/Colab Notebooks/doc_topic_distributions.joblib')
  roberta = roberta_embeddings(df)
  #embeddings =np.load('/content/drive/MyDrive/Colab Notebooks/nlp/embedding33.npy', allow_pickle=True)
  #roberta = np.vstack(embeddings.tolist())
  lda_train,lda_test = rf(doc_topic_distributions,control_posts, symptom_posts,df,symptom_subreddit)
  roberta_train,roberta_test = rf(roberta,control_posts, symptom_posts,df,symptom_subreddit)

  print(f"{'Symptom':<20} {'LDA Test Score':<15} {'RoBERTa Test Score':<15}")
  for symptom in symptom_subreddit.keys():
      lda_test_score = np.mean(lda_test[symptom])
      roberta_test_score = np.mean(roberta_test[symptom])
      print(f"{symptom:<20} {lda_test_score:<15.3f} {roberta_test_score:<15.3f}")

main()