#!/usr/bin/env python
# coding: utf-8
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
 
def read_article(file_name):
    file = open(file_name, "r")
    filedata = file.readlines()
    article = []
    for data in filedata:
        article.append(data.split(". "))
    sentences = []

    for paragraph in article:
        for sentence in paragraph:
            sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
            
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def generate_summary(file_name, top_n=5):
    nltk.download("stopwords")
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text anc split it
    sentences =  read_article(file_name)

    # indexing unranked
    dict_of_unranked_sentences = {}
    keys = range(len(sentences))
    # values = ["Hi", "I", "am", "John"]
    for i in keys:
        dict_of_unranked_sentences[i] = sentences[i]
    # print(dict_of_unranked_sentences)

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)
    # Step 4 - Sort the rank and pick top sentences
    
    ranked_sentence = []
    for i in range (len(scores)):
        ranked_sentence.append([scores[i], sentences[i], i])
    
    ranked_sentence.sort(key = lambda x: x[0], reverse=True)
    summary = ranked_sentence[0:top_n]
    summary = [i[1] for i in summary]
    for i in range(top_n):
        # print("Sentence rank ",i,":",ranked_sentence[i][1])
        summarize_text.append(" ".join(ranked_sentence[i][1]))

    # Step 5 - Offcourse, output the summarize texr
    print("Summarize Text: \n", ". ".join(summarize_text))
   
# let's begin
generate_summary( "dataset/titanic.txt", 10)
