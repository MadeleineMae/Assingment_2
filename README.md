Machine Learning Assignment 1

This project assessed whether identifiable topics can be found in fraud emails to mitigate phishing attacks. 
A BERTopics NLP unsupervised learning model was applied to a fraud email dataset. After some tuning of the UMAP and HDBSCAN it produces 26 topics. This was then validated through means of silhouette score, random seeds and manual annotations. 


Getting Started Dependencies: 
The libraries needed for this study include: 

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

This code was completed and executed on a MacOS.
Installing: The dataset used is https://www.kaggle.com/datasets/llabhishekll/fraud-email-dataset/data?select=fraud_email_.csv on Kaggle. Ensure to download it in zip format. Also the dataset is classified into 1= fraud email and 0=non-fraud email so ensure that you filter out the ones you want.
Dataset is found in .gitignore, it is extremely large so it has been compressed.
Authors: ex. Madeleine Butcher
License: This porject is liscensed under the MIT License - see the LICENSE file for detials


