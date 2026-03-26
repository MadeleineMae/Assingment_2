from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from wordcloud import WordCloud
import umap
import hdbscan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

##importing CSV
df = pd.read_csv('fraud_email_.csv')
df.head()

##assessing datset:
df = df[df['Class'] == 1].copy()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # remove HTML tags (anything between < and >)
    text = re.sub(r'<.*?>', '', text)
    
    # remove URLs (starts with http or https)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[_]{2,}', ' ', text)
    text = re.sub(r'[-]{2,}', ' ', text)
    text = re.sub(r'[=]{2,}', ' ', text) # Also handles "======"
    text = text.replace(',', ' ')
    
    # clean up extra whitespace 
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
df['Text_clean'] = df['Text'].apply(clean_text)
# drop empty rows
df = df[df['Text_clean'] != ""]
df = df.dropna(subset=['Text'])
text = df['Text_clean'].astype(str).tolist()
df["Text_clean"] = df["Text_clean"].str.strip()

##display the entire column width to ensure its all clean:
pd.set_option("display.max_colwidth", 500)
print(df["Text_clean"].head(20).to_string())

##ensuring only fraud emails are filtered out:
print("Current label distribution:")
print("-" * 30)
print(df['Class'].value_counts())
print(f"\nUnique classes found: {df['Class'].unique()}")
# for a "True/False" check:
is_clean = (df['Class'] == 1).all()
print(f"Are all rows in Class 1? {'YES' if is_clean else 'NO'}")

##testing bertopic model before analysis: -default test:
topic_model = BERTopic()
topics, probabilities = topic_model.fit_transform(text)

topic_info = topic_model.get_topic_info()

print(f"Found {len(topic_info) - 1} topics (+1 outlier cluster)\n")
print("Topic ID | Count | Keywords")
print("-" * 80)
for _, row in topic_info.iterrows():
    keywords = ", ".join(row["Representation"])
    print(f"{row['Topic']:8d} | {row['Count']:5d} | {keywords}")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# mechanisms behind how it works - how what a single document embedding looks like
embedding = embedding_model.encode(text[0])
print(f'Embedding dimensions: {embedding.shape[0]}')
print(f'First 10 values: {embedding[:10]}')

embeddings = embedding_model.encode(text, show_progress_bar=True)
print(f'Embeddings shape: {embeddings.shape}')  # (n_documents, n_dimensions)

##umaps:
umap_model = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine') ##defaults
umap_embeddings = umap_model.fit_transform(embeddings)

print(f'Original dimensions: {embeddings.shape[1]}')
print(f'Reduced dimensions:  {umap_embeddings.shape[1]}')

##hdbscan:

hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=10,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)
hdbscan_model.fit(umap_embeddings)

n_topics = len(set(hdbscan_model.labels_)) - 1  # subtract 1 to exclude outlier cluster (-1)
n_outliers = (hdbscan_model.labels_ == -1).sum()
print(f'Topics found:   {n_topics}')
print(f'Outlier docs:   {n_outliers}')
print(f'First 10 labels: {hdbscan_model.labels_[:10]}')

def aggregate_docs_by_topic(docs, topic_assignments):
    topic_docs = {}
    for doc, topic in zip(docs, topic_assignments):
        if topic in topic_docs:
            topic_docs[topic] += " " + doc
        else:
            topic_docs[topic] = doc
    return topic_docs

topic_docs = aggregate_docs_by_topic(df.Text.tolist(), hdbscan_model.labels_)
print(f'Topics (including outlier -1): {sorted(topic_docs.keys())[:10]} ...')


##vectoriser for term frequencies:
vectorizer = CountVectorizer()
aggregated_texts = list(topic_docs.values())
tf_matrix = vectorizer.fit_transform(aggregated_texts).toarray()
print(f'TF matrix shape: {tf_matrix.shape}  (topics × terms)') ##count the words in the mega document

tf_norm = tf_matrix / np.sum(tf_matrix, axis=1, keepdims=True)

n_topics_total = tf_matrix.shape[0]
docfreq = np.sum(tf_matrix > 0, axis=0)
idf = np.log((n_topics_total + 1) / (docfreq + 1)) + 1

# Gets c-TF-IDF:
ctfidf_matrix = tf_norm * idf
print(f'c-TF-IDF matrix shape: {ctfidf_matrix.shape}  (topics × terms)')

feature_names = vectorizer.get_feature_names_out()
top_n = 5

print("Topic Representations via c-TF-IDF:")
print("-" * 60)
for topic_idx, row in enumerate(ctfidf_matrix):
    top_indices = np.argsort(row)[::-1][:top_n]
    top_words = [feature_names[i] for i in top_indices]
    topic_id = list(topic_docs.keys())[topic_idx]
    print(f"Topic {topic_id:3d}: {top_words}")

##cleaner bertopics defualt test to ensure it workds- includes stop words

vectorizer_model = CountVectorizer(stop_words='english', min_df=2, ngram_range=(1, 2))
representation_model = KeyBERTInspired()

topic_model = BERTopic(
    embedding_model=embedding_model, ##takes the whole document - embedds it then takes each word and sees distance between each word
    vectorizer_model=vectorizer_model,
    representation_model=representation_model
)

topics, probabilities = topic_model.fit_transform(text, embeddings=embeddings)

topic_info = topic_model.get_topic_info()

print(f"Found {len(topic_info) - 1} topics\n")
print("Topic ID | Count | Keywords")
print("-" * 80)
for _, row in topic_info.iterrows():
    keywords = ", ".join(row["Representation"])
    print(f"{row['Topic']:8d} | {row['Count']:5d} | {keywords}")


##final test with tuned components- see Test_assingment_2 for different hyperparametre tuning:
##UMAP:
umap_model_5 = umap.UMAP(n_neighbors=20, n_components=5, min_dist=0.15, metric='cosine', random_state = 42) ##defaults
umap_embeddings_5 = umap_model_5.fit_transform(embeddings)

print(f'Original dimensions: {embeddings.shape[1]}')
print(f'Reduced dimensions:  {umap_embeddings_5.shape[1]}') 

##HDBSCAN:
hdbscan_model_5 = hdbscan.HDBSCAN(
    min_cluster_size=33,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)
hdbscan_model_5.fit(umap_embeddings_5)

n_topics_5 = len(set(hdbscan_model_5.labels_)) - 1  
n_outliers_5 = (hdbscan_model_5.labels_ == -1).sum()
print(f'Topics found:   {n_topics_5}')
print(f'Outlier docs:   {n_outliers_5}')
print(f'First 10 labels: {hdbscan_model_5.labels_[:10]}') 

def aggregate_docs_by_topic(docs, topic_assignments):
    topic_docs = {}
    for doc, topic in zip(docs, topic_assignments):
        if topic in topic_docs:
            topic_docs[topic] += " " + doc
        else:
            topic_docs[topic] = doc
    return topic_docs

topic_docs_5 = aggregate_docs_by_topic(df.Text.tolist(), hdbscan_model_5.labels_)
print(f'Topics (including outlier -1): {sorted(topic_docs_5.keys())[:10]} ...') 

##TF-IDF:
vectorizer = CountVectorizer()
aggregated_texts_5 = list(topic_docs_5.values())
tf_matrix_5 = vectorizer.fit_transform(aggregated_texts_5).toarray()
print(f'TF matrix shape: {tf_matrix_5.shape}  (topics × terms)')

tf_norm_5 = tf_matrix_5 / np.sum(tf_matrix_5, axis=1, keepdims=True)

n_topics_total_5 = tf_matrix_5.shape[0]
docfreq_5 = np.sum(tf_matrix_5 > 0, axis=0)
idf_5 = np.log((n_topics_total_5 + 1) / (docfreq_5 + 1)) + 1

ctfidf_matrix_5 = tf_norm_5 * idf_5
print(f'c-TF-IDF matrix shape: {ctfidf_matrix_5.shape}  (topics × terms)')


feature_names_5 = vectorizer.get_feature_names_out()
top_n = 5

print("Topic Representations via c-TF-IDF:")
print("-" * 60)
for topic_idx, row in enumerate(ctfidf_matrix_5):
    top_indices = np.argsort(row)[::-1][:top_n]
    top_words = [feature_names_5[i] for i in top_indices]
    topic_id = list(topic_docs_5.keys())[topic_idx]
    print(f"Topic {topic_id:3d}: {top_words}")

##BERTopic with english stop words and common fraud email stop words:
my_stop_words = list(ENGLISH_STOP_WORDS) + ['bank', "transfer", "transaction", "funds", "fund"]
vectorizer_model = CountVectorizer(stop_words=my_stop_words, min_df=2, ngram_range=(1, 2))
representation_model = KeyBERTInspired()

topic_model_5 = BERTopic(
    embedding_model=embedding_model, 
    umap_model=umap_model_5, ##ensures tuned parametres are in the model
    hdbscan_model=hdbscan_model_5,
    vectorizer_model=vectorizer_model,
    representation_model=representation_model
)

topics, probabilities = topic_model_5.fit_transform(text, embeddings=embeddings) 

topic_info_5 = topic_model_5.get_topic_info()

print(f"Found {len(topic_info_5) - 1} topics\n")
print("Topic ID | Count | Keywords")
print("-" * 80)
for _, row in topic_info_5.iterrows():
    keywords = ", ".join(row["Representation"])
    print(f"{row['Topic']:8d} | {row['Count']:5d} | {keywords}")


##visualisations:

# use topic_ids instead of range(12) to make sure REAL topics are pulled
topic_ids = topic_info_5[topic_info_5['Topic'] != -1]['Topic'].head(12).tolist()
n_topics_to_show = len(topic_ids)
rows, cols = 6, 2
fig, axes = plt.subplots(rows, cols, figsize=(12, 18)) 
# loop through the IDs and pull the name
for i, topic_id in enumerate(topic_ids):
    # get the word cloud data
    word_probs = {word: prob for word, prob in topic_model_5.get_topic(topic_id)}
    wordcloud = WordCloud(width=500, height=300,
                          background_color='white',
                          prefer_horizontal=1.0).generate_from_frequencies(word_probs)
    
    # pulled from topic_info: gets the 'Name' or keywords
    # this will be something like "0_bank_money_transfer"
    topic_name = topic_info_5[topic_info_5.Topic == topic_id]['Name'].values[0]
    
    # set the title with the ID and keywords
    ax = axes[i // cols, i % cols]
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Topic {topic_id}:\n{topic_name}', fontsize=10, fontweight='bold')
# hide any empty plots
for j in range(i + 1, rows * cols):
    axes[j // cols, j % cols].axis('off')
plt.tight_layout()
plt.show()


##validating the model:
##silhouette score:
from sklearn.metrics import silhouette_score
# filter out outliers (-1) for a clean score
mask = hdbscan_model_5.labels_ != -1
clean_embeddings = umap_embeddings_5[mask]
clean_labels = hdbscan_model_5.labels_[mask]
sil_score = silhouette_score(clean_embeddings, clean_labels)
print(f"Silhouette Score (Topic Separation): {sil_score:.4f}")

##random seeds
# define the seeds you want to test
seeds = [33, 11, 98]
results = []
print("Starting Stability Test...")
print("-" * 50)
for seed in seeds:
    # re-create the UMAP model with the new seed
    temp_umap = umap.UMAP(n_neighbors=20, n_components=5, min_dist=0.15, 
                          metric='cosine', random_state=seed)
    temp_embeddings = temp_umap.fit_transform(embeddings)
    # re-run HDBSCAN (with existing parameters)
    temp_hdbscan = hdbscan.HDBSCAN(min_cluster_size=33, metric='euclidean', cluster_selection_method='eom')
    temp_labels = temp_hdbscan.fit_predict(temp_embeddings)
    # calculate metrics
    n_topics = len(set(temp_labels)) - 1
    
    # calculate silhouette score (excluding outliers)
    mask = temp_labels != -1
    sil_score = silhouette_score(temp_embeddings[mask], temp_labels[mask])
    
    results.append({'Seed': seed, 'Topics': n_topics, 'Silhouette': round(sil_score, 4)})
    print(f"Seed {seed}: Found {n_topics} topics | Silhouette: {sil_score:.4f}")
# display the final comparison table
df_stability = pd.DataFrame(results)
print("\nStability Summary Table:")
print(df_stability)

# get IDs for top 12
top_12_topic_ids = topic_model_5.get_topic_info().iloc[1:13]['Topic'].tolist()
# get document info 
document_info = topic_model_5.get_document_info(text)
# create a balanced sample (exactly 8 docs from EACH of the top 12 topics)
# .sample(frac=1) shuffles the whole list first 
# .groupby('Topic').head(9) pulls exactly 9 from each group 
# and .sample(frac=1) again makes sure Topic 1 isn't all at the top of the file!
final_balanced_sample = (
    document_info[document_info['Topic'].isin(top_12_topic_ids)]
    .sample(frac=1, random_state=42)
    .groupby('Topic')
    .head(8)
    .sample(frac=1, random_state=42) #"mixer" shuffle
    .reset_index(drop=True)
)


##important note before running is ensure this line is a comment otherwise the annotations will be overwritten.
#final_balanced_sample[['Document', 'Topic', 'Name', 'Representation']].to_csv('top_12_topic_annotation_5.csv', index=False)
# print to confirm the counts are equal
#print("Sample of 96 created! Every topic now has exactly 8 documents:")
#print(final_balanced_sample[['Topic', "Name"]].value_counts())

##analysing the annotations: - please note that the annotations were overwritten- if this is overun please see Test_assingment for unclean results
print("\nLabel distribution in training set:")
df_2 = pd.read_csv("top_12_topic_annotation_4.csv", encoding='ISO-8859-1')
print(df_2["TopicID"].value_counts().sort_index(), df_2["Name_model"].value_counts().sort_index())