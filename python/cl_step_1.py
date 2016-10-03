
from general_functions import *

stop_en = stopwords.words('english')
stop_en = stop_en + list(['rt'])
stop_sp = stopwords.words('spanish')


# - ***Create corpus with all documents***

# In[2]:

def update_corpus():

#    print 'UPDATE CORPUS'

    global tf_idf_docs
    global dictionary
    global tfidf
    global documents
    
    tf_idf_docs = {} ## Reset tf_idf_docs, the docs need to be re-computed based on the new corpus
    
    texts = [[word for word in document['doc'].lower().split() ]for id_doc,document in documents.items()]
    # print texts[1]

    dictionary = corpora.Dictionary(texts)
    # dictionary.save('/tmp/deerwester.dict') # store the dictionary, for future reference
    print(len(dictionary))

    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus)
    return True


# - ***given a query doc q, retrieve a set of 2000 recent documents that have some term co-occurrence with q***

# In[3]:

""" Retrieve a list of most (2000) recent tweets containing at least one term contained in the input tweet """
def get_docs_with_same_terms(tweet_id):
    global documents, td_inv_ind, recent_docs
    d2_list = set([])#list of docs containing at least one common term with tweet_id doc
    for term in documents[tweet_id]['doc'].split():
        if term in td_inv_ind.keys(): 
            d2_list = set(d2_list).union(set(td_inv_ind[term])) ## set of docs containing the terms of the current tweet
            
        ### Add term to inverted index
        try: ## term already in td_inv_ind 
            if tweet_id not in td_inv_ind[term]:
                td_inv_ind[term].append(tweet_id)
        except: ##term NOT in td_inv_ind
            td_inv_ind[term] = []
            td_inv_ind[term].append(tweet_id)

    ###   intersection between d2_list and last 2000 docs
    try:
        d2_list_ret = set(d2_list).intersection(set(recent_docs))
    except Exception as ex:
        print ex
    return d2_list_ret


# - ***Given a query doc q, retrieve its nearest neighbour from the list of docs retrieved by `get_docs_with_same_terms`***

# In[4]:

""" Look for the doc NN comparing the query doc with those documents retrieved by the get_docs_with_same_terms function """
def check_nn_inverted_index(d2_list, tweet_id, query):
    dis_min = 1
    doc_min = -1
    global tf_idf_docs, dictionary, tfidf, documents
    for d_2 in d2_list:
        if d_2 != tweet_id:
            try:
                d_cand = tf_idf_docs[d_2]
            except:
                d_cand_tf = dictionary.doc2bow(documents[d_2]['doc'].split()) 
                d_cand = tfidf[d_cand_tf]
                tf_idf_docs[d_2] = d_cand ##add tf_idf for curr doc to avoid to compute it at each loop
            c = round(1 - sparse_cos_sim(query, d_cand),4)
            if c < dis_min:
                dis_min = c
                doc_min = d_2
    return {'doc':doc_min, 'distance':dis_min}



# - ***Add the query doc q to a cluster***

# In[5]:

"""Given a query_doc, its NN and their distance, create a new cluster with the query_doc or add it to the same cluster of its NN """
def add_doc_to_clusters(tweet_id, dis_min, doc_min):
#     date_format = '%Y-%m-%dT%H:%M:%S'
    date_format = '%a %b %d %H:%M:%S +0000 %Y'
    
    global clusters, inv_cl, usrs_cl, documents_nopreproc, dis_min_t, cl_index
#     print 'dis_min ', dis_min
    if dis_min >= dis_min_t:     
#        print 'Create  new cluster with ', tweet_id
        ## Create new cluster
        clusters[str(cl_index)] = {}
        clusters[str(cl_index)]['docs']=[]
        clusters[str(cl_index)]['docs'].append(tweet_id)
        clusters[str(cl_index)]['last_timestamp'] = documents_nopreproc[tweet_id]['timestamp']
        inv_cl[str(tweet_id)] = cl_index
        usrs_cl[str(cl_index)] = []
        usrs_cl[str(cl_index)].append(documents_nopreproc[tweet_id]['user'])
        clust = cl_index
        cl_index += 1
    else:       
        ## Retrieve the cluster of the most similar doc
        clust = inv_cl[str(doc_min)]
        
        clusters[str(clust)]['docs'].append(tweet_id)
        usrs_cl[str(clust)].append(documents_nopreproc[tweet_id]['user'])
        inv_cl[str(tweet_id)] = clust ## store in which cluster each doc is 
    return clust


# - ***Print clusters in real tieme during the algorithm execution***

"""Print clusters in real tieme during the algorithm execution"""
def print_cluster(clust): 
    global clusters, thres_min_doc_x_cl, usrs_cl, cluster_file
    if len(clusters[str(clust)]['docs']) > thres_min_doc_x_cl: ## I m considering cluster with less than thres_min_doc_x_cl items as neutrals

        cluster_file.write('cluster_id:' + str(clust) + '\n')
        cluster_file.write('num_doc:' + str(len(clusters[str(clust)]['docs'])) + '\n')        
        cluster_file.write('num_users:' + str(len(set(usrs_cl[str(clust)])) ) + '\n')
            
        if float(float(len(clusters[str(clust)]['docs'])) / len(set(usrs_cl[str(clust)]))) > 2: # Compare #docs wrt #users
            cluster_file.write('type: SPAM  \n')
        else:
            cluster_file.write('type: EVENT  \n')
            
        tmp_print_txt = {}        
        cluster_file.write('Docs:  \n')
        
        for i in range(0, len(clusters[str(clust)]['docs'])):   
                
                if remove_all_punct(remove_mentions(remove_rt_str(remove_urls(documents[clusters[str(clust)]['docs'][i]]['doc'])))) not in tmp_print_txt.keys():
                    try:                        
                        tmp_print_txt[remove_all_punct(remove_mentions(remove_rt_str(remove_urls(documents[clusters[str(clust)]['docs'][i]]['doc']))))] = {'cont':1,'timestamp':documents[clusters[str(clust)]['docs'][i]]['timestamp']}
                    except Exception as ex: print "----------------------------------------------",ex
                else:
                    try:
                    ## if this doc has already been posted I'm printing the last timestamp
                        tmp_print_txt[remove_all_punct(remove_mentions(remove_rt_str(remove_urls(documents[clusters[str(clust)]['docs'][i]]['doc']))))] = {'cont':tmp_print_txt[remove_all_punct(remove_mentions(remove_rt_str(remove_urls(documents[clusters[str(clust)]['docs'][i]]['doc']))))]['cont'] + 1,'timestamp':documents[clusters[str(clust)]['docs'][i]]['timestamp']}
                    except Exception as ex: print '=========================================== ', ex
        
        for text, data in tmp_print_txt.items():    
            cluster_file.write('- '+ str(data['timestamp']) + ' --> ' + str(data['cont']) + ' x ' + text + '\n')
        cluster_file.write('\n=============================== end cluster   ===============================\n\n\n\n' )
        cluster_file.flush()
            
            



def code(self, input_point):
    """ Calculate LSH code for a single input point. Returns one code of
        length `hash_size` for each `hash_table`.
        :param input_point:
        A list, or tuple, or numpy ndarray object that contains numbers
        only. The dimension needs to be 1 * `input_dim`.
        This object will be converted to Python tuple and stored in the
        selected storage.
        """
    
    if isinstance(input_point, np.ndarray):
        input_point = input_point.tolist()
    
    return [self._hash(self.uniform_planes[i], input_point)
            for i in xrange(self.num_hashtables)]


# - ***for a given doc q, query the lsh structure and/or the inverted index to find its nearest neighbour***

def build_clusters(query, tweet_id):

    global lsh, inv_doc_index, documents, documents_nopreproc, tfidf, cluster_file, clusters,tf_idf_docs, dis_min_t, dimension,L

    query_dense = np.zeros(dimension)
    for k,v in dict(query).items():
        query_dense[k] = v

################ lsh.query rallenta l esecuzione, come limitare il numero di punti in ciascun bucket ??????
    try:
        near_neigh_data = lsh.query(query_dense, num_results=1, distance_func="cosine") ## query the lsh for the nn
    except Exception as ex:
        print 'Exception ', ex
    
    near_neigh_doc = -1
    near_neigh_dist = 1
    try:
        near_neigh_doc = near_neigh_data[0][0][1]
        near_neigh_dist = near_neigh_data[0][1]
        
#        print 'near_neigh_doc ' , near_neigh_doc
#        print 'near_neigh_dist ', near_neigh_dist
    except:
        pass
    
    lsh.index(query_dense, tweet_id) ## add new doc to lsh
####new: limit the number of elements for bucket
    hash_keys = code(lsh,query_dense) ## retrieve the list of keys (bucket), 1 key for each hashtable, containing the query
    for i in xrange(L):
        if len(lsh.hash_tables[i].get_list(hash_keys[i]) ) > doc_x_buck_th:
            temp_list = lsh.hash_tables[i].get_list(hash_keys[i])
            temp_list.pop(0)
            lsh.hash_tables[i].set_val(hash_keys[i],temp_list)
####end new

################ questa parte rallenta l esecuzione END

                        
#    """ if the NN has not been found, check in the inverted index """    
    if near_neigh_dist >= dis_min_t: # if the NN is not enough similar check in the inverted index of term-document
        d2_list = get_docs_with_same_terms(tweet_id) #docs that contain query terms
        near_neigh_data_id = check_nn_inverted_index(d2_list, tweet_id, query)
        near_neigh_doc = near_neigh_data_id['doc']
        near_neigh_dist = near_neigh_data_id['distance']            

    clust = add_doc_to_clusters(tweet_id, near_neigh_dist, near_neigh_doc)                

#     print 'clust == ', clust
#    print_cluster(clust)



# - ***Save the clustering results and the analyzed documents in json format***

# In[8]:

"""Save the clustering results and the analyzed documents in json format"""

def print_final_clusters(all_cluster_file_name,all_users__cluster_file_name):   
    global dis_min_t, L, k, dataset
    """create json containing cluster_id:[list of tweet_ids]"""
    with open(all_cluster_file_name, 'w') as fp:
        json.dump(clusters, fp)
    """create json containing cluster_id:[list of user_ids]    """
    with open(all_users__cluster_file_name, 'w') as fp:
        json.dump(usrs_cl, fp)

    """  Save documents with processed text  """
    with open(path + 'documents.json', 'w') as fp:
        json.dump(documents, fp)

    """  Save documents with NOT processed text  """
    with open(path + 'documents_nopreproc.json', 'w') as fp:
        json.dump(documents_nopreproc, fp)

    settings = {'L_hashtables':L, 'k_hyperplanes': k, 'threshold_distance': dis_min_t , 'n_docs':len(documents), 'data':dataset}
    """  Save execution settings  """
    with open(path + 'settings.json', 'w') as fp:
        json.dump(settings, fp)




# In[9]:

def custom_print(path, all_cluster_file_name, all_users_cluster_file_name):
    clusters_file = open(path + 'final_print_clusters.txt','a') ##output file
    with open(all_cluster_file_name) as data_file:    
        cluster_res = json.load(data_file)

    with open(all_users_cluster_file_name) as users_data_file:    
        usrs_cl = json.load(users_data_file)

    with open(path + "documents.json") as doc_data_file:    
        documents = json.load(doc_data_file)

    with open(path + "documents_nopreproc.json") as documents_nopreproc_data_file:    
        documents_nopreproc = json.load(documents_nopreproc_data_file)
            
    min_ndocs = 0 ## treshold
    for k,v in cluster_res.items():
#        tmp_print_txt = {}
        tmp_print_txt = OrderedDict()

        distinct_users = len(set(usrs_cl[k])) #number of distinct users in cluster k
        tot_docs = len(v['docs']) #number of total docs in cluster k
        if tot_docs > min_ndocs:  ## decide what we want to print...not relevant now, we can read the result  of the clustering and print whatever we want
            for i in range(len(v['docs'])):
                text_without_nline = ''.join(ch for ch in remove_all_punct(remove_mentions(remove_rt_str(remove_urls(documents[str(v['docs'][i])]['doc'])))) if ch not in ['\n','\r'])
                if text_without_nline not in tmp_print_txt.keys():
                    tmp_print_txt[text_without_nline] =  {'cont':1,'timestamp':documents[str(v['docs'][i])]['timestamp'], 'tweet_id':v['docs'][i]}
                else:
                    try:
                        tmp_print_txt[text_without_nline] =  {'cont':tmp_print_txt[text_without_nline]['cont']+1,'timestamp':documents[str(v['docs'][i])]['timestamp'], 'tweet_id':v['docs'][i]}
                    except Exception as ex:
                        print 'qqq ' ,  ex
            distinct_docs = len(tmp_print_txt)
            if(len(tmp_print_txt) > 0):
                clusters_file.write('\n')
                clusters_file.write('\n')
                clusters_file.write('CLUSTER ' + str(k) + '||  DISTINCT USERS '+ str( len(set(usrs_cl[k])) ) + '|| TOT DOCS '+ str(len(set(cluster_res[k]['docs'])) ) + '|| DISTINCT DOCS ' + str(len(tmp_print_txt) ) + '\n')
                clusters_file.write('Timestamp | freq | Text \n')

                for text, data in tmp_print_txt.items():    
                    clusters_file.write('- ' + str(data['timestamp']) + ' | ' + str(data['cont']) + ' | ' + remove_rt_str(remove_mentions(documents_nopreproc[str(data['tweet_id'])]['doc'])).encode('ascii','ignore') + '\n')
    clusters_file.close()


# - ***Read documents strteam and crate a collection***

# In[10]:

def create_collection(data):
    global dictionary, tfidf, documents, n_docs, tweets_db_file, recent_docs, L, lsh, dimension
    try:
        if (1): #data.lang=='en'

            #get tweet info
            tweet_text = data['text']
            tweet_id = data['id']
            
            #Avoid duplicates
            if tweet_id in documents.keys():
                return
            
            try:
                tweet_timestamp = data['created_at'] ## Correct way
            except:
                tweet_timestamp = datetime.fromtimestamp(int(str(data['createdAt'])[0:len( str(data['createdAt']) )-3])).strftime('%a %b %d %H:%M:%S +0000 %Y') ## for catalunya BDigital dataset

#            print tweet_timestamp
            user_id = data['user']['id']
            
            tweet_text = stem_doc(remove_stop_words(remove_punct(remove_mentions(remove_rt_str(remove_urls(tweet_text))))) )

#            print 'n_docs = ', n_docs
            n_docs += 1
            documents[tweet_id] = {}
            documents_nopreproc[tweet_id] = {}
            documents[tweet_id] = {'doc':tweet_text, 'timestamp':tweet_timestamp,'user':user_id}
            documents_nopreproc[tweet_id] = {'doc':data['text'], 'timestamp':tweet_timestamp,'user':user_id}
            
            recent_docs.append(tweet_id)
            
            if n_docs % training_thres == 0:
                
                #######
                """  Print clusters. Set the output file name based on your input dataset  """
                all_cluster_file_name = path + "clusters_step1.json"
                all_users__cluster_file_name = path + "users_clusters_step1.json"
                print_final_clusters(all_cluster_file_name,all_users__cluster_file_name)

                custom_print(path, all_cluster_file_name,all_users__cluster_file_name)
                
                os.system("python cl_step_2.py")
                os.system("python summerise.py")

                #######


                update_corpus()
                dimension = len(dictionary)
                lsh = LSHash(k, dimension, L)   


            if n_docs > training_thres:
                query_vec = dictionary.doc2bow(tweet_text.split())
                query = tfidf[query_vec]
                tf_idf_docs[tweet_id] = query
                if len(query_vec) > 0:
                    if float(float(len(tweet_text.split()) ) / len(query_vec)) <= 2:
                        build_clusters(query,tweet_id)
                return True
    except Exception as ex: 
        print ex



# In[11]:

if __name__ == '__main__':  
    cl_index = 0
    documents_nopreproc = OrderedDict()
    documents = OrderedDict()

    recent_docs_threshold = 2000 ## number of docs to consider for the inverted index
    recent_docs = collections.deque(maxlen=recent_docs_threshold) #contains the 'recent_docs_threshold' most recent tweet_ids

    ##----- Parameters

    # For every training_thres docs arrived in the streaming The corpus is updated
    training_thres = 3000

    dis_min_t = 0.8 #minimum distance to add doc to NN cluster
    k = 13 #number of hyperplanes k 
    L = 16 #math.log(0.025,0.8)

    thres_min_doc_x_cl = 3   ## useful just during the real time printing
    doc_x_buck_th = 100 ## limit of the #docs in a single bucket ... SHOULD WE CHANGE THIS NUMBER ???

    ## Structures    
    inv_doc_index = [] ## For each tweet contains the corresponding hash_key

#    clusters = {} #Contain the result of the clustering. cluster_id:[list of tweets]
    clusters = OrderedDict()
    usrs_cl = {} #Contain the user of the result of the clustering. cluster_id:[list of users]
    inv_cl = {} # for each tweet it contains the cluster_id

    td_inv_ind = {} ##Inverted index structure that contains for each term the list of tweets (only tweets_id is stored) that contain that term
    n_docs = 0 ## doc counter
    tf_idf_docs = {} # Store the tf-idf of each arrived tweet (to avoid to compute it multiple times)


    """  Set the output file name based on your input dataset  """    
    cluster_file_name = 'real_time_clusters.txt'
    cluster_file = open(cluster_file_name,'w') ## this file contains the cluster created in real time during the algorithm execution
    
    ## Initialize corpus
    texts = [[word for word in document['doc'].lower().split() if (word not in stop_en) and (word not in stop_sp)]for id_doc,document in documents.items()]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus)
    
    
    dimension = len(dictionary)
    lsh = LSHash(k, dimension, L)

    
    """  SET THE PATH OF YOUR INPUT DATASET   """
#    dataset = 'twitter-crisis-test'
    path = 'test-disaster/'
    indir = 'data'
    for f in os.listdir(indir+'/'):
        try:            
            user_file = open(indir +'/'+ f)
            for l in user_file:
                try: 
                    tweet_js = json.loads(l.strip())
                    create_collection(tweet_js)
                except Exception as ex:
                    print 'xxx ',  ex

            user_file.close()
            print len(documents)
        except:pass
    cluster_file.close()        
    

    """  Print clusters. Set the output file name based on your input dataset  """    
    all_cluster_file_name = path + "clusters_step1.json"
    all_users__cluster_file_name = path + "users_clusters_step1.json"
    print_final_clusters(all_cluster_file_name,all_users__cluster_file_name)
    print '\n number of documents = ', len(documents)
    print 'L = ', L
    print 'min distance = ', dis_min_t


#all_clusters_file_name = path + "clusters_step1.json" #file containing all obtained clusters cluster_id:[tweet_ids]
#all_users_clusters_file_name = path + "users_clusters_step1.json" #file containing all obtained clusters cluster_id:[tweet_ids]
#custom_print(path, all_clusters_file_name,all_users_clusters_file_name)


# In[ ]:




# In[ ]:



