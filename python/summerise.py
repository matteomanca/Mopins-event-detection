
# coding: utf-8

# In[31]:

from general_functions import *
#from collections import Counter, OrderedDict


#from gensim import corpora, models, similarities

"""PARAMETERS"""
path = 'test-disaster/' ## Set your path
cluster_res_file = 'cluster_res2_step2.json'


# ##### Read clustering result

"""Read clustering result"""
with open(path + cluster_res_file) as data_file:    
    cluster_res = json.load(data_file)
    
"""Read documents"""
with open(path + "documents.json") as doc_data_file:    
    documents = json.load(doc_data_file)

with open(path + "documents_nopreproc.json") as documents_nopreproc_data_file:    
    documents_nopreproc = json.load(documents_nopreproc_data_file)




dims = {}
for k, v in cluster_res.items():
    dims[k] = len(v)
# dims


# #### Select clusters with more than 100 docs

# In[35]:

th_event = len(documents) / 300 ##
active_clusters = dict((k, v) for k, v in dims.items() if v >= 0)
# active_clusters


# In[36]:

"""Compute centroid for a set of documents"""
def get_centroid(v):

    global tf_idf_docs
    global dictionary
    global tfidf

    cont = 0
    curr_cent = {}
    for i in v: #list of docs in current cluster
        try:
            curr_tfidf_d1 = tf_idf_docs[str(i)]
        except:
            tf_d1 =  dictionary.doc2bow(documents[str(i)]['doc'].split())
            curr_tfidf_d1 = tfidf[tf_d1]      
            tf_idf_docs[str(i)] = curr_tfidf_d1 #save to avoid multiple computation of tfidf of the same tweet

        curr_doc = dict(curr_tfidf_d1)

        for key in set(curr_doc.keys()):    
#             print key
            try:
                curr_cent[key] = curr_cent[key] + curr_doc[key]
            except:
                curr_cent[key] = curr_doc[key]
    curr_centroid = [(term_id,(tfidfsum/len(v)) ) for term_id,tfidfsum in curr_cent.items()]
    return curr_centroid


# In[37]:

def get_nn(cluster_centroid, docs):
    dist_th = 2
    nn_doc = ""
    global tf_idf_docs    
    for d in docs:
        tf_idf_d = tf_idf_docs[str(d)]
        dist_doc_centroid = round(1 - sparse_cos_sim(cluster_centroid, tf_idf_d),4)
        if dist_doc_centroid < dist_th:
            dist_th = dist_doc_centroid
            nn_doc = d
    return nn_doc


# In[38]:

def get_div_doc(cand_set, docs):
    cs_centroid = get_centroid(cand_set)
#     print cs_centroid
    dist_th = 0
    div_doc = ""
    global tf_idf_docs    
    for d in docs:
        tf_idf_d = tf_idf_docs[str(d)]
        dist_doc_cs_centroid = round(1 - sparse_cos_sim(cs_centroid, tf_idf_d),4)
        if dist_doc_cs_centroid > dist_th:
            dist_th = dist_doc_cs_centroid
            div_doc = d
#     print tf_idf_docs[str(div_doc)], dist_th
    return {'doc':div_doc,'dist':dist_th}


# In[ ]:

#summ_sets = {}
summ_sets = OrderedDict()
tf_idf_docs = {} ## Reset tf_idf_docs, the docs need to be re-computed based on the new corpus    
texts = [[word for word in document['doc'].lower().split() ]for id_doc,document in documents.items()]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
tfidf = models.TfidfModel(corpus)

for k, v in active_clusters.items():
    docs = cluster_res[k]
    cluster_centroid = get_centroid(docs)    
    summ_sets[k] = []
    nn_doc = get_nn(cluster_centroid, docs)
    docs.remove(nn_doc) ## this remove also from cluster_res[k]['docs']
    summ_sets[k].append(nn_doc)
    add_docs = True
    while add_docs:
        div_doc = get_div_doc(summ_sets[k],docs)
#         print div_doc
        ## if the document is enough "diverse" I add to the list of docs to retrieve to the user
        if div_doc['dist'] > 0.8 :
            summ_sets[k].append(div_doc['doc'])
            docs.remove(div_doc['doc']) ## this remove also from cluster_res[k]['docs']
        else:
            add_docs = False


# In[22]:

with open(path + "cl_tags.json") as data_file:
    cl_tags_dict = json.load(data_file)

html = '<html><head></head><body>'

summary_name = 'summary.txt'
summary_file = open(path + summary_name,'a') ## this file contains the cluster created in real time during the algorithm execution

summary_file_html = open(path + 'summary.html','a') ## this file contains the cluster created in real time during the algorithm execution

summary_file.write('===============================================================================\n')
summary_file.write(str(len(documents)) + ' documents\n')
summary_file.write('=============================================\n\n')

html = html + '<br/><br/>' + str(len(documents)) + ' documents\n' + '<br />'
html = html + '<table style="width:80%" >'

print '\n\n'
for k, docs in summ_sets.items():
#    summary_file.write('Cluster ' + str(k) +  ' - ' + 'original ' +  str(active_clusters[k]) +  ' - '  +  'summary ' +  str(len(docs)) +  '\n')
    summary_file.write('Summary of Event ' + str(k)  +  ' (Total tweets: ' + str(len(cluster_res[k])) + ') \n')
    html = html + '<th bgcolor="#FAD7A0" colspan=2>Event ' + str(k) + ' - (Total tweets: ' + str(len(cluster_res[k])) + ')</th>'
    try:
        summary_file.write(cl_tags_dict[k] +  '\n')
        html = html + '<tr bgcolor="#FAD7A0"><td colspan=2>' + cl_tags_dict[k]  + '</td></tr>'
    except:
        print 'cccc'

    summary_sorted = OrderedDict()
    for d in docs:
        summary_sorted[documents_nopreproc[str(d)]['timestamp']] = remove_mentions(remove_rt_str(documents_nopreproc[str(d)]['doc']))
    summary_sorted = collections.OrderedDict(sorted(summary_sorted.items()))

    i = 0
    for time, body in summary_sorted.items():
        i += 1
        summary_file.write('- ' + str(time) + ' ' + str(body)+ '\n')
        if i%2 == 0:
            html = html + '<tr bgcolor="#D6EAF8"><td width = 10%>' +str(time)  + '</td><td>' + str(body)  + '</td></tr>'
        else:
            html = html + '<tr bgcolor="#D1F2EB"><td width = 10%>' +str(time)  + '</td><td>' + str(body)  + '</td></tr>'
    html = html + '<br/><br/>'
    summary_file.write("\n\n")
summary_file.write('===============================================================================\n\n\n')

html = html + '</table></body></html>'
summary_file_html.write(html)

summary_file_html.close()

summary_file.close()


