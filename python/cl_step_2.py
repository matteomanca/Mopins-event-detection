
# ## Clustering phase 2
# 
# The script `Clustering_phase_1` generates clusters containing documents that are very similar each others. Now, we want to join clusters that belong to the same topic/event.

###
#Imports
###
from general_functions import *


"""PARAMETERS"""
path = 'test-disaster/' ## Set your path

def get_tags(text):
    return[i for i in text.lower().split() if '#' in i]

"""Given a query_doc, its NN and their distance, create a new cluster with the query_doc or add it to the same cluster of its NN """
def add_doc_to_clusters(clust_id, dis_min, doc_min):
    global clusters, inv_cl, usrs_cl, documents_nopreproc, dis_min_t, cl_index, original_clusters

    if dis_min > dis_min_t:
#        print 'Create  new cluster with ', clust_id, '\n\n  '
        ## Create new cluster
        cl_index = min(original_clusters[clust_id]['docs'])
        clusters[str(cl_index)] = []
        clusters[str(cl_index)].append(clust_id)
        inv_cl[str(clust_id)] = cl_index
        clust = cl_index
    else:
        ## Retrieve the cluster of the most similar doc
        clust = inv_cl[str(doc_min)]
        ## Add query doc to the same cluster of the most similar doc
        clusters[str(clust)].append(clust_id)
        inv_cl[str(clust_id)] = clust ## store in which cluster each doc is 
    return clust



# #### Create bag-of-words for each cluster and compute tf-idf



"""Read clusters step 1"""
all_clusters_file_name = path + "clusters_step1.json" #file containing all obtained clusters cluster_id:[clust_ids]
# all_users_clusters_file_name = "users_clusters.cat0315.end.json" #file containing all obtained clusters cluster_id:[clust_ids]

users_clusters_file_name = path + "users_clusters_step1.json" #file containing all obtained clusters cluster_id:[clust_ids]

with open(all_clusters_file_name) as data_file:    
    original_clusters = json.load(data_file)

with open(users_clusters_file_name) as data_file:    
    original_clusters_users = json.load(data_file)

with open(path + "documents.json") as doc_data_file:    
    documents = json.load(doc_data_file)

with open(path + "documents_nopreproc.json") as documents_nopreproc_data_file:    
    documents_nopreproc = json.load(documents_nopreproc_data_file)

#num_doc_x_clust_th = 2        ### CHECK
#num_users_x_clust_th = 2      ### CHECK
#
num_doc_x_clust_th = len(documents) * 0.1 /100 # 0.2%       ### CHECK: for 3000 tweets I consider a cluster only if ti contains at least 10 docs
#num_doc_x_clust_th = len(documents) * 0.5 /100 # 0.2%       ### CHECK: for 3000 tweets I consider a cluster only if ti contains at least 10 docs
num_users_x_clust_th = num_doc_x_clust_th /20      ### CHECK

"""Create bag of words for each cluster"""
bags = OrderedDict() #dictionary cluster_id: bag of words
#print original_clusters.keys()





###
# create bag of words for each cluser
###
for k, v in collections.OrderedDict(sorted(original_clusters.items())).iteritems():
    """TODO : PAY ATTENTION TO THE FOLLOWING CONDITION: I' M NOT CONSIDERING ALL CLUSTERS BUT JUST THOSE BIGGER THAN num_doc_x_clust_th DOCS"""

## remove neutral and spam cluster
    if len(v['docs']) > num_doc_x_clust_th and len(original_clusters_users[k]) > num_users_x_clust_th : 
        for i in range(len(v['docs'])):
            try:
                bags[k] = bags[k] + " " + documents[str(v['docs'][i])]['doc']  #texts are already stopped and stemmed
            except:
                bags[k] = documents[str(v['docs'][i])]['doc']

# save bags of words
with open(path + "bags.json", 'w') as fp:
    json.dump(bags, fp)





###
#create corpus with all terms of my clusters"""
###
tf_idf_clusters = OrderedDict()

texts_bag = [[word for word in text.lower().split()] for id_clus, text in bags.items()]
# print texts_bag[1]

dictionary_bag = corpora.Dictionary(texts_bag)
# dictionary.save('/tmp/deerwester.dict') # store the dictionary, for future reference
# print(len(dictionary_bag))

corpus_bag = [dictionary_bag.doc2bow(text) for text in texts_bag]

tfidf_bag = models.TfidfModel(corpus_bag)


""" Each cluster is represented by a tf-idf vector """ 

for k, v in bags.items():
    curr_tf = dictionary_bag.doc2bow(v.split()) 
    curr_tfidf = tfidf_bag[curr_tf]                    
    tf_idf_clusters[k] = curr_tfidf ##add tf_idf for curr doc to avoid to compute it at each loop

k = 13 #number of hyperplanes k, put 13
L = 16 #math.log(0.025,0.8) ==> 16

dimension = len(dictionary_bag)    

doc_x_buck_th = 2000

## Structures
#planes = [] # list of K*L hyperplanes
lsh = LSHash(k, dimension, L)
inv_doc_index = [] ## For each tweet contains the corresponding hash_key

clusters = OrderedDict() #Contain the result of the clustering. cluster_id:[list of tweets]
#try:
#    with open(path + "cluster_res_XXX.json") as data_file:
#        clusters = json.load(data_file)
#except:
#    clusters = OrderedDict()     ###TODO: update and do not re-initialize


usrs_cl = {} #Contain the user of the result of the clustering. cluster_id:[list of users]
inv_cl = {} # for each tweet it contains the cluster_id


dis_min_t = 0.85 #a high value leads to more spread clusters

cl_index = 0



for cl_id, tfidf_values in tf_idf_clusters.items():
#    print cl_id
    ## Initialize distance and NN doc
    near_neigh_doc = -1
    near_neigh_dist = 1

    query_dense = np.zeros(dimension)
    for k,v in dict(tfidf_values).items():
        query_dense[k] = v
    try:
        near_neigh_data = lsh.query(query_dense, num_results=1, distance_func="cosine") ## query the lsh for the nn
    except Exception as ex:
        print 'Exception ', ex

    try:
        near_neigh_doc = near_neigh_data[0][0][1]
        near_neigh_dist = near_neigh_data[0][1]
        
#        print 'near_neigh_doc ' , near_neigh_doc
#        print 'near_neigh_dist ', near_neigh_dist
    except:
        pass
    
    lsh.index(query_dense, cl_id) ## add new doc to lsh
    
    add_doc_to_clusters(cl_id, near_neigh_dist, near_neigh_doc)                


# In[7]:


"""Print clusters of clusters"""
cluster_res = OrderedDict()
#prev_cl2_file_name = path + "cluster_res_step2.json" #file containing all obtained clusters cluster_id:[clust_ids]
#    
#try:
#    with open(prev_cl2_file_name) as data_file:
#        cluster_res = json.load(data_file)
#except:
#    cluster_res = OrderedDict()     ###TODO: update and do not re-initialize


"""save cluster result step 2"""
with open(path + "cluster_res_XXX.json", 'w') as fp:
    json.dump(clusters, fp)


for cl_id, cl_list in clusters.items(): #each cluster of clusters
    if len(cl_list) > 1:
        docs = []
        for cluster in sorted(cl_list): #get documents for each cluster_id
            try:
                if {cluster: original_clusters[cluster]} not in cluster_res[cl_id]:
                    cluster_res[cl_id].append({cluster: original_clusters[cluster]})
            except:
                cluster_res[cl_id] = []
                if {cluster: original_clusters[cluster]} not in cluster_res[cl_id]:
                    cluster_res[cl_id].append({cluster: original_clusters[cluster]})

"""save cluster result step 2"""
with open(path + "cluster_res_step2.json", 'w') as fp:
    json.dump(cluster_res, fp)



#cluster_res2 = {}
cluster_res2 = OrderedDict()
for cl_id2, cl_data2 in cluster_res.items():
#    print cl_data2
    for cls_1 in sorted(cl_data2):
        for cl_id1, cl_data1 in cls_1.items():
            try:
                cluster_res2[str(cl_id2)] += cl_data1['docs']
            except:
                cluster_res2[str(cl_id2)] = cl_data1['docs']

"""save cluster result step 2 in diferent format: cl_id: list of all docs (that belong to diferent clusters in step 1)"""
with open(path + "cluster_res2_step2.json", 'w') as fp:
    json.dump(cluster_res2, fp)
    

with open(path + "cluster_res_step2.json") as data_file:    
    cluster_res = json.load(data_file)

    
##
#Print event
##

cluster_file_name = 'cluster_step2XXX.txt'
cluster_file = open(path + cluster_file_name,'a') ## this file contains the cluster created in real time during the algorithm

cluster_file.write('===============================================================================\n')
cluster_file.write(str(len(documents)) + ' documents\n')
cluster_file.write('=============================================\n\n')

cl_tags_dict = OrderedDict()



for cl_id, cl_data in cluster_res.items():
    cl_tags = []
    tweets_concat_str = ''
    temp_list = []
    to_print = {}
    cluster_file.write('Event: ' + cl_id + '\n')
    cluster_file.write('Cluter_dim: ' + str(len(cluster_res2[cl_id]) ) + '\n')
    for cluster_1 in cl_data:
        for cl_step1_id , cl_step1_data in cluster_1.items():
            temp_list.append(cl_step1_id)
            current_cluster_doc_list = cl_step1_data['docs']
            for d in current_cluster_doc_list:
                text_current_doc = remove_all_punct(remove_mentions(remove_rt_str(remove_urls(documents_nopreproc[str(d)]['doc']))))
                time_current_doc = documents_nopreproc[str(d)]['timestamp']

                if text_current_doc not in tweets_concat_str:
                    tweets_concat_str = tweets_concat_str + ' ' + text_current_doc
#                    cluster_file.write(' - ' + str(d) + ' ' + time_current_doc + ' ' + text_current_doc + '\n')
                    to_print[time_current_doc] = {'text':text_current_doc, 'id': cl_id}
#                    cluster_file.write(' - ' +  time_current_doc + ' ' + text_current_doc + '\n')

                curr_tags = get_tags(documents_nopreproc[str(d)]['doc'])
#                print 'curr_tags',  curr_tags
                if len(curr_tags) > 0 :
                    cl_tags.append(get_tags(documents_nopreproc[str(d)]['doc']) )
            cl_tags_dict[cl_id] = ' '.join(set(list(itertools.chain(*cl_tags))) )

    cluster_file.write(cl_tags_dict[cl_id] + '\n')
    to_print = collections.OrderedDict(sorted(to_print.items()))
    for time, data in to_print.items():
        cluster_file.write(' - ' +  time + ' ' + data['text'] + '\n')
    cluster_file.write('Cluters step 1: ' + ' '.join(sorted(temp_list)) + '\n')
    cluster_file.write('\n\n')






























#for cl_id, cl_list in clusters.items(): #each cluster of clusters
#    if len(cl_list) > 1:
#        cluster_file.write('Event: ' + cl_id + '\n')
#        doc_list = []
#        cluster_file.write('Cluters step 1: ' + ' '.join(sorted(cl_list)) + '\n')
#        for cl1 in cl_list:
#            doc_list = doc_list + original_clusters[cl1]['docs']
#        for d in doc_list:
#            cluster_file.write(' - ' + str(d) + ' ' + documents_nopreproc[str(d)]['timestamp'] + ' ' + remove_mentions(remove_rt_str(documents_nopreproc[str(d)]['doc'] )) + '\n')
#            curr_tags = get_tags(remove_mentions(remove_rt_str(documents_nopreproc[str(d)]['doc'] )))
#            if len(curr_tags) > 0 :
#                cl_tags.append(get_tags(remove_mentions(remove_rt_str(documents_nopreproc[str(d)]['doc'] ))) )
#        cl_tags_dict[cl_id] = ' '.join(set(list(itertools.chain(*cl_tags))) )
#        cluster_file.write('\n\n')
#
#cl_tags_dict = OrderedDict()


#
#for cl_id, cl_data in cluster_res.items():
##    print ' ----------------------- ', cl_id
#    cluster_file.write('\n\nCluter_dim: ' + str(len(cluster_res2[cl_id]) ) + '\n')
##    cluster_file.write('Cluters step 1: ' + ' '.join(sorted(cl_data.keys())) + '\n')
#    cl_tags = []
#    tmp_print_txt = OrderedDict()
#    cl1_id_temp_list = []
#    for cls_1 in sorted(cl_data):
#        for cl1_id, data_step1 in cls_1.items():
#            cl1_id_temp_list.append(cl1_id)
##            print '===> ', cl1_id
#    #    for cl1_id, data_step1 in cl_data.items():
#            for d in data_step1['docs']:
#                """Use the following for the usual date format"""
#                text_without_nline = ''.join(ch for ch in documents_nopreproc[str(d)]['doc'] if ch not in ['\n','\r'])
#                if remove_all_punct(remove_mentions(remove_rt_str(remove_urls(text_without_nline)))) not in tmp_print_txt.keys():
#                    tmp_print_txt[remove_all_punct(remove_mentions(remove_rt_str(remove_urls(text_without_nline))))] =  {'cont':1,'timestamp':documents[str(d)]['timestamp'], 'tweet_id':d, 'cl1_id':cl1_id}
#                else:
#                    tmp_print_txt[remove_all_punct(remove_mentions(remove_rt_str(remove_urls(text_without_nline))))] =  {'cont':tmp_print_txt[remove_all_punct(remove_mentions(remove_rt_str(remove_urls(text_without_nline))))]['cont']+1,'timestamp':documents[str(d)]['timestamp'], 'tweet_id':d, 'cl1_id':cl1_id}
#                curr_tags = get_tags(text_without_nline)
#                if len(curr_tags) > 0 :
#                    cl_tags.append(get_tags(text_without_nline) )
#    #            print cl_tags
#    cluster_file.write('Event: ' + cl_id + '\n')
#    cluster_file.write('Cluters step 1: ' + ' '.join(sorted(cl1_id_temp_list)) + '\n')
#    cluster_file.write(' '.join(set(list(itertools.chain(*cl_tags))) ))
#    cluster_file.write('\n')
#
#    cl_tags_dict[cl_id] = ' '.join(set(list(itertools.chain(*cl_tags))) )
#    
#    ## sort by timestamp
#    tmp_print_txt_sorted = OrderedDict()
#    for text, data in tmp_print_txt.items():
#        tmp_print_txt_sorted[data['timestamp']] = {'text' : remove_mentions(remove_rt_str(documents_nopreproc[str(data['tweet_id'])]['doc'])).encode('ascii','ignore'), 'cont': data['cont'], 'id':str(data['tweet_id'])}
#
#    tmp_print_txt_sorted = collections.OrderedDict(sorted(tmp_print_txt_sorted.items()))
#
### print to file
#    for time, body in tmp_print_txt_sorted.items():
#        cluster_file.write ('- ' + body['id'] + ' ' + str(time) + ' | ' + str(body['cont']) + ' | ' + body['text']  + '\n')
#
#cluster_file.write('===============================================================================\n\n\n')
#
#cluster_file.close()
#
with open(path + "cl_tags.json", 'w') as fp:
    json.dump(cl_tags_dict, fp)


