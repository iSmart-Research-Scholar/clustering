from django.http import HttpResponse, JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
from wordcloud import WordCloud
import json

@api_view(['GET'])
def cluster(request):
    keywords = request.GET.get('keywords')
    recent = request.GET.get('recent')
    c_weight = request.GET.get('citation_weight')
    sorting_factor = request.GET.get('factor')
    URL = "http://127.0.0.1:8000/?keywords=" + keywords + "&recent=" + recent + "&citation_weight=" + c_weight
    data = requests.get(url = URL )
    data = data.json()
    abstracts = []
    title = []
    for article in data['articles']:
        abstracts.append(article['abstract'])
        title.append(article['title'])
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(abstracts)
    true_k = 3
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=200, n_init=10)
    model.fit(X)
    labels=model.labels_
    result={'cluster':labels,'abstracts':abstracts}
    vals = {0: [], 1: [], 2:[], 3:[], 4:[]}
    for i in range(len(labels)):
        vals[labels[i]].append(data["articles"][i])
    # for ele in vals:
    #     print(len(vals[ele]))
    return JsonResponse(vals, safe=False)

