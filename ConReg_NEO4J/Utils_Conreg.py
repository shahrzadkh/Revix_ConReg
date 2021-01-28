
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.decomposition import PCA, LatentDirichletAllocation, TruncatedSVD
#from lexnlp.nlp.en.segments.sections import get_sections 
import json
import nltk
nltk.download()
from nltk.corpus import stopwords
import string
#print(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer
import re
import gensim
from gensim import corpora
#!pip install pyLDAvis
import pyLDAvis.gensim
from matplotlib import pyplot as plt
from wordcloud import WordCloud
import matplotlib.colors as mcolors
import math
import numpy as np
import pandas as pd
import holoviews as hv
import networkx as nx
#!pip install datashader
from holoviews import opts
from holoviews.operation.datashader import datashade, bundle_graph
from holoviews.element.graphs import layout_nodes
hv.extension('bokeh')
defaults = dict(width=800, height=800)
hv.opts.defaults(opts.EdgePaths(**defaults), opts.Graph(**defaults), opts.Nodes(**defaults))


#Goes to Util
class Text_parsing(object):
    def __init__(self):
        
        return 

    def parse_article(self, article, section_title, section_subtitle, act_name):
        article_title = article.get("TITLE")
        article_subtitle = article.get("SUBTITLE")
        article_content = article.get("CONTENT")
        assert article_title is not None
        assert article_subtitle is not None
        assert article_content is not None
        article = {"act_name":act_name,
                   "section_title": section_title,
                   "section_subtitle": section_subtitle,
                   "article_title": article_title + '_' + act_name,# + article_subtitle,
                   "article_subtitle":article_subtitle,
                   "article_content": article_content,
                  }
        return article

    def parse_section(self, section, collection=None, title=None, subtitle=None, act_name = ''):
        articles = [] if collection is None else collection
        if isinstance(section, list):
            for sec in section:
                articles = self.parse_section(sec, collection=articles,
                                         title=title, subtitle=subtitle, act_name=act_name)
            return articles
        section_title = section.get("TITLE") if title is None else f"{title}_{section.get('TITLE')}"
        section_subtitle = section.get("SUBTITLE") if title is None else f"{subtitle}_{section.get('SUBTITLE')}"

        # Only iterate if it contains a list of articles
        if "ARTICLE" in section:
            for art in section.get("ARTICLE",[]):
                article_entry = self.parse_article(art, section_title=section_title, section_subtitle=section_subtitle, act_name=act_name)
                articles.append(article_entry)
            return articles
        return self.parse_section(section["SECTION"], collection=articles,
                                  title=section_title, subtitle=section_subtitle, act_name=act_name)


    def parse_articles(self, json_data, act_name=''):
        articles = []
        sections = json_data['DOCUMENT']['SECTION']
        for i, section in enumerate(sections):
            if isinstance(section, list):
                for sec in section:
                    section_data = self.parse_section(sec,act_name=act_name)
                    articles.extend(section_data)
            else:
                section_data = self.parse_section(section,act_name=act_name)
                articles.extend(section_data)
        return pd.DataFrame(articles)

    def Read_in_Json_from_text(self, Orig_dir ='', all_jason_list_path =''):
        with open(all_jason_list_path) as fp: 
            Lines = fp.readlines() 
            files =[]
            acts =[]
            for line in Lines: 
                with open(os.path.join(Orig_dir, line.strip()), "r") as f:
                    data_1 = json.load(f)
                files.append(data_1)
                acts.append(re.sub('\.json$', '', line.strip()))
        return files, acts
    
    def merge_replicated_entries(self, df, matching_coloumn_name='', merging_coloumn_name=''):
        K = df[df.duplicated(subset=[matching_coloumn_name], keep=False)]
        df = df[~df.duplicated(subset=[matching_coloumn_name], keep=False)]

        q = K.article_title.unique()
        j =1
        for i in q:
            P =K.loc[K[matching_coloumn_name] == q[0]].reset_index(drop=True)
            temp_df = P.iloc[[0]]
            for j in range(1, P.shape[0]):
                temp_df.loc[0, merging_coloumn_name] = temp_df.loc[0, merging_coloumn_name]+ P.loc[j, merging_coloumn_name]
            df = df.append(temp_df, ignore_index=True, sort=False)

        return df
#%% preprocess text
# Goes to Util:
class Text_cleaning(object):
    def __init__(self, additional_stopwords_FullPath):
        stop_words_nltk = set(stopwords.words('english'))
        stopwords_final = set(w.rstrip() for w in open(additional_stopwords_FullPath))#'/home/brain/AiTalents/Notebooks/stopwords.txt'))
        stopwords_final = set([s.replace('\n', '') for s in stopwords_final])
        stopwords_final = stopwords_final.union(stop_words_nltk)
        self.stopwords_final = stopwords_final
    def clean_html(self, raw_text):
        cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        cleantext = re.sub(cleanr, '', raw_text)
        return cleantext

    def remove_URL(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'',text)

    
    def my_tokenizer(self, s):
        s = s.lower() # downcase
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
        tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
        tokens = [t for t in tokens if t not in string.punctuation] # remove punctuation        
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove any digits, i.e. "3rd edition"
        return tokens

    def preprocess_text(self, text):
        temp_text = self.remove_URL(text)
        no_html = self.clean_html(temp_text)
        return self.my_tokenizer(no_html)
    def Word_map_generation(self, df):
        #put clean Text into an array:
        contents = df["article_content"].values
        word_index_map = {}
        current_index = 0
        all_tokens = []
        all_content = []
        index_word_map = []
        error_count = 0
        for content in contents:
            try:
                content = content.encode('ascii', 'ignore').decode('utf-8') # this will throw exception if bad characters
                all_content.append(content)
                tokens = self.preprocess_text(content)
                all_tokens.append(tokens)
                for token in tokens:
                    if token not in word_index_map:
                        word_index_map[token] = current_index
                        current_index += 1
                        index_word_map.append(token)
            except Exception as e:
                print(e)
                print(title)
                error_count += 1
        return word_index_map, index_word_map,all_tokens, contents, error_count, self.stopwords_final


    def sklearn_Countvectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
                
        tf_vectorizer = CountVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                        token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                        tokenizer= self.preprocess_text)

        dtm_tf = tf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return tf_vectorizer, dtm_tf ## This is already a sparse feature matrix that could be fed into lda or SVD.
    def sklearn_TfidfVectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
        
        Tfidf_vectorizer = TfidfVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                           token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                           tokenizer= self.preprocess_text)

        dtm_Tfidf = Tfidf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return Tfidf_vectorizer, dtm_Tfidf ## This is already a sparse feature matrix that could be fed into lda or SVD.
            
    
    
#%% Feature generation:
#Goes to Util
class Feature_design(object):
    def __init__(self, word_index_map, all_tokens):
        self.word_index_map = word_index_map
        self.all_tokens = all_tokens
    # now let's create our input matrices - just indicator variables for this example - works better than proportions
    def tokens_to_vector(self, tokens):
        x = np.zeros(len(self.word_index_map))
        for t in tokens:
            i = self.word_index_map[t]
            x[i] = 1
        return x
    
    def Feature_matrix_generation(self):
        N = len(self.all_tokens)
        D = len(self.word_index_map)
        X = np.zeros((D, N)) # terms will go along rows, documents along columns
        i = 0
        for tokens in all_tokens:
            X[:,i] = self.tokens_to_vector(tokens)
            i += 1
        return X
    def Genism_Feature_matrix_generation(self, saving_dir):
        #analysis_dir= '/home/brain/AiTalents/Notebooks/Genism/'
        try:
            os.makedirs(saving_dir)
        except OSError:
            if not os.path.isdir(saving_dir):
                raise

        #os.chdir(analysis_dir)
        dictionary = corpora.Dictionary(self.all_tokens)
        corpus = [dictionary.doc2bow(text) for text in self.all_tokens]
        pickle.dump(corpus, open(os.path.join(saving_dir,'corpus.pkl'), 'wb'))
        dictionary.save(os.path.join(saving_dir,'dictionary.gensim'))
        return corpus, dictionary

#%% Topic models generation:
#Goes to Util
class Topic_modeling(object):
    def __init__(self, num_topics, saving_dir):
        
        
        self.num_topics = num_topics
        self.saving_dir = saving_dir
        try:
            os.makedirs(self.saving_dir)
        except OSError:
            if not os.path.isdir(self.saving_dir):
                raise
    def Genism_LDA(self, corpus, dictionary):
        #saving_dir= '/home/brain/AiTalents/Notebooks/Genism/'
            
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = self.num_topics, id2word=dictionary, passes=15)
        ldamodel.save(os.path.join(self.saving_dir,'LDA_model' + str(self.num_topics)+ '.gensim'))
        return ldamodel
    def Genism_LSI(self, corpus, dictionary):
        lsimodel = gensim.models.lsimodel.LsiModel(corpus, num_topics = self.num_topics, id2word=dictionary)
        lsimodel.save(os.path.join(self.saving_dir,'LSI_model' + str(self.num_topics)+ '.gensim'))
        return lsimodel
    def Sklearn_LDA(self, Feature_X):
        ldamodel = LatentDirichletAllocation()
        ldamodel.fit(Feature_X)
        Z = ldamodel.transform(Feature_X) # size: #term X #Component    >> For word Cloud
        # How about the graph connections?   componennts_ : size: #comps X #Articles  >> each column gives weights.
        U = ldamodel.components_    
        return ldamodel, Z, U 

            
#topics = ldamodel.print_topics(num_words=5)
#for topic in topics:
#    print(topic)

#%% Figures:

#Goes to Util
class Genism_plotting(object):
    def __init__(self,Figure_saving_dir):
        self.Figure_saving_dir = Figure_saving_dir
        try:
            os.makedirs(self.Figure_saving_dir)
        except OSError:
            if not os.path.isdir(self.Figure_saving_dir):
                raise
    def Convert(self, tup, di): 
        di = dict(tup) 
        return di

    def Reverse(self, tuples): 
        new_tup = () 
        for k in reversed(tuples): 
            new_tup = new_tup + (k,) 
        return new_tup 

    def pyLDA_Dashboard(self,ldamodel, corpus, dictionary,html_name, sort_topics=False):
        
        lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=sort_topics)
        pyLDAvis.save_html(lda_display, os.path.join(self.Figure_saving_dir,html_name + '_Dashboard.html'))
        #pyLDAvis.display(lda_display)
        return lda_display
    def Genism_word_cloud(self,html_name, ldamodel):
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

        cloud = WordCloud(background_color='#D8D8D8',
                          width=2500,
                          height=1800,
                          max_words=10,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)
        
        topic_word_box = ldamodel.show_topics(num_topics= ldamodel.num_topics, num_words= np.int(ldamodel.num_terms/100), formatted=False)
        # For each topic:
        topic_word_box_R= []
        for i in np.arange(ldamodel.num_topics):
            temp=[]
            temp = list(self.Reverse(topic_word_box[i][1]))
            topic_word_box_R.append(temp)
        
        D = ldamodel.num_topics
        CL = round(math.sqrt(D))
        RW = CL
        while RW * CL < D:
            RW += 1
        fig, axes = plt.subplots(RW, CL, figsize=(10,10), sharex=True, sharey=True)
        plt.style.context('grayscale')
        fig.patch.set_facecolor('#D8D8D8')

        Topic_names = ['Reporting', 'Investment', 'Risk_Management','Accounting', 'Payments_and_Settlement']
        #['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']
        #
        for i, ax in enumerate(axes.flatten()):
            if i >= D:
                axes.flatten()[i].axis('off')
            else:
            
                fig.add_subplot(ax)
                #topic_words = dict(topics[i][1])
                topic_words={}
                print(i)
                topic_words = self.Convert(topic_word_box_R[i], topic_words)
                cloud.generate_from_frequencies(topic_words, max_font_size=300)
                plt.gca().imshow(cloud)
                
                plt.gca().set_title(Topic_names[i] , fontdict=dict(size=16))
                #Topic_names.append('Topic_' + str(i +1))
                plt.gca().axis('off')


        plt.subplots_adjust(wspace=0, hspace=0)
        
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        
        #fig.patch.set_alpha(0.7)
        #plt.gca().set_facecolor('tab:gray')
        #cloud.to_file(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'))
        fig.savefig(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'), transparent=True)
        
        return 
                
class Genism_Graph(object):
    def __init__(self, Figure_saving_dir='', verbose = True):
        
        self.Figure_saving_dir = Figure_saving_dir
        self.verbose = verbose
        
    def Genism_networkX_graph(self, df, ldamodel):
        Article_names = df['article_title'].values
        
        Topic_names = ['Topic_'+ str(i) for i in range(1, len(ldamodel.show_topics())+1)]
        nx_graph = nx.Graph()
        Nodes = Topic_names + list(Article_names)
        # Add node for each Topic
        for T in Nodes:
            nx_graph.add_node(T)

        i = 0   
        # add edges
        for a in range(0,len(Article_names)):
            temp = ldamodel.get_document_topics(corpus[a])
            for c in range(0, len(temp)):
                if self.verbose:
                    
                    print('C: ', c)
                    print(len(ldamodel.get_document_topics(corpus[a])))

                T = Topic_names[temp[c][0]]
                if self.verbose:
                    print('T, A: ',T, Article_names[a])
                i += 1
                if self.verbose:
                    print('i:', i)
                # Only add edge if the count is positive

                nx_graph.add_edge(Article_names[a],T, weight = temp[c][1])
                if self.verbose:
                    print('Done_edge')
        return nx_graph
    def plot_Genism_networkX_graph(self,df, ldamodel, to_bundel=True ):
        nx_graph = self.Genism_networkX_graph(df, ldamodel)
        #Create the graph using networkX data:
        act_names = []
        kwargs = dict(width=800, height=800, xaxis=None, yaxis=None)
        opts.defaults(opts.Nodes(**kwargs), opts.Graph(**kwargs))
        
        #colors = ['#000000']+hv.Cycle('Category20').values
        topic_graph = hv.Graph.from_networkx(nx_graph, nx.layout.fruchterman_reingold_layout, k=1)
        topic_graph.nodes.data['act'] = topic_graph.nodes.data['index']
        for i in np.arange(topic_graph.nodes.data.shape[0]):
            if topic_graph.nodes.data.loc[i, 'act'] not in ['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']:
                topic_graph.nodes.data.loc[i, 'act'] = str(topic_graph.nodes.data.loc[i, 'act'].split('_')[1])
                
        
        tmp2= topic_graph.nodes.data[~topic_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        #colors = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*np.arange(len(tmp)), 20+2*np.arange(len(tmp)))]
        
        if to_bundel:

            bundled = bundle_graph(topic_graph)
            #.opts(opts.Nodes(color='act', size=10, width=1000, cmap=colors, legend_position='right'))
#opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            #
            return bundled, act_names, article_names
        else:
            return topic_graph, act_names, article_names
        #.opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            
    
    def subselect_Genism_single_Topic_graph(self, Topic_id, topic_graph):
        
        Sub_graph = topic_graph.select(start=Topic_id).opts(node_fill_color='blue', edge_line_width=hv.dim('weight'), edge_line_color='red')
        tmp2= Sub_graph.nodes.data[~Sub_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        
        return Sub_graph, act_names, article_names
    def subselect_Genism_single_Article_graph(self, Article_title, topic_graph, circular_layout=True):
        
        D = topic_graph.select(end=Article_title).opts(node_fill_color='white', edge_line_width=-1*np.log10(hv.dim('weight')))
        
        if circular_layout:
            
            F = layout_nodes(D)
            labels = hv.Labels(F.nodes, ['x', 'y'], 'index')

            return (F * labels.opts(text_font_size='10pt', text_color='black', bgcolor='white'))
        else: 
            
            return D#Goes to Util
class Text_parsing(object):
    def __init__(self):
        
        return 

    def parse_article(self, article, section_title, section_subtitle, act_name):
        article_title = article.get("TITLE")
        article_subtitle = article.get("SUBTITLE")
        article_content = article.get("CONTENT")
        assert article_title is not None
        assert article_subtitle is not None
        assert article_content is not None
        article = {"act_name":act_name,
                   "section_title": section_title,
                   "section_subtitle": section_subtitle,
                   "article_title": article_title + '_' + act_name,# + article_subtitle,
                   "article_subtitle":article_subtitle,
                   "article_content": article_content,
                  }
        return article

    def parse_section(self, section, collection=None, title=None, subtitle=None, act_name = ''):
        articles = [] if collection is None else collection
        if isinstance(section, list):
            for sec in section:
                articles = self.parse_section(sec, collection=articles,
                                         title=title, subtitle=subtitle, act_name=act_name)
            return articles
        section_title = section.get("TITLE") if title is None else f"{title}_{section.get('TITLE')}"
        section_subtitle = section.get("SUBTITLE") if title is None else f"{subtitle}_{section.get('SUBTITLE')}"

        # Only iterate if it contains a list of articles
        if "ARTICLE" in section:
            for art in section.get("ARTICLE",[]):
                article_entry = self.parse_article(art, section_title=section_title, section_subtitle=section_subtitle, act_name=act_name)
                articles.append(article_entry)
            return articles
        return self.parse_section(section["SECTION"], collection=articles,
                                  title=section_title, subtitle=section_subtitle, act_name=act_name)


    def parse_articles(self, json_data, act_name=''):
        articles = []
        sections = json_data['DOCUMENT']['SECTION']
        for i, section in enumerate(sections):
            if isinstance(section, list):
                for sec in section:
                    section_data = self.parse_section(sec,act_name=act_name)
                    articles.extend(section_data)
            else:
                section_data = self.parse_section(section,act_name=act_name)
                articles.extend(section_data)
        return pd.DataFrame(articles)

    def Read_in_Json_from_text(self, Orig_dir ='', all_jason_list_path =''):
        with open(all_jason_list_path) as fp: 
            Lines = fp.readlines() 
            files =[]
            acts =[]
            for line in Lines: 
                with open(os.path.join(Orig_dir, line.strip()), "r") as f:
                    data_1 = json.load(f)
                files.append(data_1)
                acts.append(re.sub('\.json$', '', line.strip()))
        return files, acts
    
    def merge_replicated_entries(self, df, matching_coloumn_name='', merging_coloumn_name=''):
        K = df[df.duplicated(subset=[matching_coloumn_name], keep=False)]
        df = df[~df.duplicated(subset=[matching_coloumn_name], keep=False)]

        q = K.article_title.unique()
        j =1
        for i in q:
            P =K.loc[K[matching_coloumn_name] == q[0]].reset_index(drop=True)
            temp_df = P.iloc[[0]]
            for j in range(1, P.shape[0]):
                temp_df.loc[0, merging_coloumn_name] = temp_df.loc[0, merging_coloumn_name]+ P.loc[j, merging_coloumn_name]
            df = df.append(temp_df, ignore_index=True, sort=False)

        return df
#%% preprocess text
# Goes to Util:
class Text_cleaning(object):
    def __init__(self, additional_stopwords_FullPath):
        stop_words_nltk = set(stopwords.words('english'))
        stopwords_final = set(w.rstrip() for w in open(additional_stopwords_FullPath))#'/home/brain/AiTalents/Notebooks/stopwords.txt'))
        stopwords_final = set([s.replace('\n', '') for s in stopwords_final])
        stopwords_final = stopwords_final.union(stop_words_nltk)
        self.stopwords_final = stopwords_final
    def clean_html(self, raw_text):
        cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        cleantext = re.sub(cleanr, '', raw_text)
        return cleantext

    def remove_URL(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'',text)

    
    def my_tokenizer(self, s):
        s = s.lower() # downcase
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
        tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
        tokens = [t for t in tokens if t not in string.punctuation] # remove punctuation        
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove any digits, i.e. "3rd edition"
        return tokens

    def preprocess_text(self, text):
        temp_text = self.remove_URL(text)
        no_html = self.clean_html(temp_text)
        return self.my_tokenizer(no_html)
    def Word_map_generation(self, df):
        #put clean Text into an array:
        contents = df["article_content"].values
        word_index_map = {}
        current_index = 0
        all_tokens = []
        all_content = []
        index_word_map = []
        error_count = 0
        for content in contents:
            try:
                content = content.encode('ascii', 'ignore').decode('utf-8') # this will throw exception if bad characters
                all_content.append(content)
                tokens = self.preprocess_text(content)
                all_tokens.append(tokens)
                for token in tokens:
                    if token not in word_index_map:
                        word_index_map[token] = current_index
                        current_index += 1
                        index_word_map.append(token)
            except Exception as e:
                print(e)
                print(title)
                error_count += 1
        return word_index_map, index_word_map,all_tokens, contents, error_count, self.stopwords_final


    def sklearn_Countvectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
                
        tf_vectorizer = CountVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                        token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                        tokenizer= self.preprocess_text)

        dtm_tf = tf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return tf_vectorizer, dtm_tf ## This is already a sparse feature matrix that could be fed into lda or SVD.
    def sklearn_TfidfVectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
        
        Tfidf_vectorizer = TfidfVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                           token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                           tokenizer= self.preprocess_text)

        dtm_Tfidf = Tfidf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return Tfidf_vectorizer, dtm_Tfidf ## This is already a sparse feature matrix that could be fed into lda or SVD.
            
    
    
#%% Feature generation:
#Goes to Util
class Feature_design(object):
    def __init__(self, word_index_map, all_tokens):
        self.word_index_map = word_index_map
        self.all_tokens = all_tokens
    # now let's create our input matrices - just indicator variables for this example - works better than proportions
    def tokens_to_vector(self, tokens):
        x = np.zeros(len(self.word_index_map))
        for t in tokens:
            i = self.word_index_map[t]
            x[i] = 1
        return x
    
    def Feature_matrix_generation(self):
        N = len(self.all_tokens)
        D = len(self.word_index_map)
        X = np.zeros((D, N)) # terms will go along rows, documents along columns
        i = 0
        for tokens in all_tokens:
            X[:,i] = self.tokens_to_vector(tokens)
            i += 1
        return X
    def Genism_Feature_matrix_generation(self, saving_dir):
        #analysis_dir= '/home/brain/AiTalents/Notebooks/Genism/'
        try:
            os.makedirs(saving_dir)
        except OSError:
            if not os.path.isdir(saving_dir):
                raise

        #os.chdir(analysis_dir)
        dictionary = corpora.Dictionary(self.all_tokens)
        corpus = [dictionary.doc2bow(text) for text in self.all_tokens]
        pickle.dump(corpus, open(os.path.join(saving_dir,'corpus.pkl'), 'wb'))
        dictionary.save(os.path.join(saving_dir,'dictionary.gensim'))
        return corpus, dictionary

#%% Topic models generation:
#Goes to Util
class Topic_modeling(object):
    def __init__(self, num_topics, saving_dir):
        
        
        self.num_topics = num_topics
        self.saving_dir = saving_dir
        try:
            os.makedirs(self.saving_dir)
        except OSError:
            if not os.path.isdir(self.saving_dir):
                raise
    def Genism_LDA(self, corpus, dictionary):
        #saving_dir= '/home/brain/AiTalents/Notebooks/Genism/'
            
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = self.num_topics, id2word=dictionary, passes=15)
        ldamodel.save(os.path.join(self.saving_dir,'LDA_model' + str(self.num_topics)+ '.gensim'))
        return ldamodel
    def Genism_LSI(self, corpus, dictionary):
        lsimodel = gensim.models.lsimodel.LsiModel(corpus, num_topics = self.num_topics, id2word=dictionary)
        lsimodel.save(os.path.join(self.saving_dir,'LSI_model' + str(self.num_topics)+ '.gensim'))
        return lsimodel
    def Sklearn_LDA(self, Feature_X):
        ldamodel = LatentDirichletAllocation()
        ldamodel.fit(Feature_X)
        Z = ldamodel.transform(Feature_X) # size: #term X #Component    >> For word Cloud
        # How about the graph connections?   componennts_ : size: #comps X #Articles  >> each column gives weights.
        U = ldamodel.components_    
        return ldamodel, Z, U 

            
#topics = ldamodel.print_topics(num_words=5)
#for topic in topics:
#    print(topic)

#%% Figures:

#Goes to Util
class Genism_plotting(object):
    def __init__(self,Figure_saving_dir):
        self.Figure_saving_dir = Figure_saving_dir
        try:
            os.makedirs(self.Figure_saving_dir)
        except OSError:
            if not os.path.isdir(self.Figure_saving_dir):
                raise
    def Convert(self, tup, di): 
        di = dict(tup) 
        return di

    def Reverse(self, tuples): 
        new_tup = () 
        for k in reversed(tuples): 
            new_tup = new_tup + (k,) 
        return new_tup 

    def pyLDA_Dashboard(self,ldamodel, corpus, dictionary,html_name, sort_topics=False):
        
        lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=sort_topics)
        pyLDAvis.save_html(lda_display, os.path.join(self.Figure_saving_dir,html_name + '_Dashboard.html'))
        #pyLDAvis.display(lda_display)
        return lda_display
    def Genism_word_cloud(self,html_name, ldamodel):
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

        cloud = WordCloud(background_color='#D8D8D8',
                          width=2500,
                          height=1800,
                          max_words=10,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)
        
        topic_word_box = ldamodel.show_topics(num_topics= ldamodel.num_topics, num_words= np.int(ldamodel.num_terms/100), formatted=False)
        # For each topic:
        topic_word_box_R= []
        for i in np.arange(ldamodel.num_topics):
            temp=[]
            temp = list(self.Reverse(topic_word_box[i][1]))
            topic_word_box_R.append(temp)
        
        D = ldamodel.num_topics
        CL = round(math.sqrt(D))
        RW = CL
        while RW * CL < D:
            RW += 1
        fig, axes = plt.subplots(RW, CL, figsize=(10,10), sharex=True, sharey=True)
        plt.style.context('grayscale')
        fig.patch.set_facecolor('#D8D8D8')

        Topic_names = ['Reporting', 'Investment', 'Risk_Management','Accounting', 'Payments_and_Settlement']
        #['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']
        #
        for i, ax in enumerate(axes.flatten()):
            if i >= D:
                axes.flatten()[i].axis('off')
            else:
            
                fig.add_subplot(ax)
                #topic_words = dict(topics[i][1])
                topic_words={}
                print(i)
                topic_words = self.Convert(topic_word_box_R[i], topic_words)
                cloud.generate_from_frequencies(topic_words, max_font_size=300)
                plt.gca().imshow(cloud)
                
                plt.gca().set_title(Topic_names[i] , fontdict=dict(size=16))
                #Topic_names.append('Topic_' + str(i +1))
                plt.gca().axis('off')


        plt.subplots_adjust(wspace=0, hspace=0)
        
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        
        #fig.patch.set_alpha(0.7)
        #plt.gca().set_facecolor('tab:gray')
        #cloud.to_file(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'))
        fig.savefig(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'), transparent=True)
        
        return 
                
class Genism_Graph(object):
    def __init__(self, Figure_saving_dir='', verbose = True):
        
        self.Figure_saving_dir = Figure_saving_dir
        self.verbose = verbose
        
    def Genism_networkX_graph(self, df, ldamodel):
        Article_names = df['article_title'].values
        
        Topic_names = ['Topic_'+ str(i) for i in range(1, len(ldamodel.show_topics())+1)]
        nx_graph = nx.Graph()
        Nodes = Topic_names + list(Article_names)
        # Add node for each Topic
        for T in Nodes:
            nx_graph.add_node(T)

        i = 0   
        # add edges
        for a in range(0,len(Article_names)):
            temp = ldamodel.get_document_topics(corpus[a])
            for c in range(0, len(temp)):
                if self.verbose:
                    
                    print('C: ', c)
                    print(len(ldamodel.get_document_topics(corpus[a])))

                T = Topic_names[temp[c][0]]
                if self.verbose:
                    print('T, A: ',T, Article_names[a])
                i += 1
                if self.verbose:
                    print('i:', i)
                # Only add edge if the count is positive

                nx_graph.add_edge(Article_names[a],T, weight = temp[c][1])
                if self.verbose:
                    print('Done_edge')
        return nx_graph
    def plot_Genism_networkX_graph(self,df, ldamodel, to_bundel=True ):
        nx_graph = self.Genism_networkX_graph(df, ldamodel)
        #Create the graph using networkX data:
        act_names = []
        kwargs = dict(width=800, height=800, xaxis=None, yaxis=None)
        opts.defaults(opts.Nodes(**kwargs), opts.Graph(**kwargs))
        
        #colors = ['#000000']+hv.Cycle('Category20').values
        topic_graph = hv.Graph.from_networkx(nx_graph, nx.layout.fruchterman_reingold_layout, k=1)
        topic_graph.nodes.data['act'] = topic_graph.nodes.data['index']
        for i in np.arange(topic_graph.nodes.data.shape[0]):
            if topic_graph.nodes.data.loc[i, 'act'] not in ['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']:
                topic_graph.nodes.data.loc[i, 'act'] = str(topic_graph.nodes.data.loc[i, 'act'].split('_')[1])
                
        
        tmp2= topic_graph.nodes.data[~topic_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        #colors = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*np.arange(len(tmp)), 20+2*np.arange(len(tmp)))]
        
        if to_bundel:

            bundled = bundle_graph(topic_graph)
            #.opts(opts.Nodes(color='act', size=10, width=1000, cmap=colors, legend_position='right'))
#opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            #
            return bundled, act_names, article_names
        else:
            return topic_graph, act_names, article_names
        #.opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            
    
    def subselect_Genism_single_Topic_graph(self, Topic_id, topic_graph):
        
        Sub_graph = topic_graph.select(start=Topic_id).opts(node_fill_color='blue', edge_line_width=hv.dim('weight'), edge_line_color='red')
        tmp2= Sub_graph.nodes.data[~Sub_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        
        return Sub_graph, act_names, article_names
    def subselect_Genism_single_Article_graph(self, Article_title, topic_graph, circular_layout=True):
        
        D = topic_graph.select(end=Article_title).opts(node_fill_color='white', edge_line_width=-1*np.log10(hv.dim('weight')))
        
        if circular_layout:
            
            F = layout_nodes(D)
            labels = hv.Labels(F.nodes, ['x', 'y'], 'index')

            return (F * labels.opts(text_font_size='10pt', text_color='black', bgcolor='white'))
        else: 
            
            return D#Goes to Util
class Text_parsing(object):
    def __init__(self):
        
        return 

    def parse_article(self, article, section_title, section_subtitle, act_name):
        article_title = article.get("TITLE")
        article_subtitle = article.get("SUBTITLE")
        article_content = article.get("CONTENT")
        assert article_title is not None
        assert article_subtitle is not None
        assert article_content is not None
        article = {"act_name":act_name,
                   "section_title": section_title,
                   "section_subtitle": section_subtitle,
                   "article_title": article_title + '_' + act_name,# + article_subtitle,
                   "article_subtitle":article_subtitle,
                   "article_content": article_content,
                  }
        return article

    def parse_section(self, section, collection=None, title=None, subtitle=None, act_name = ''):
        articles = [] if collection is None else collection
        if isinstance(section, list):
            for sec in section:
                articles = self.parse_section(sec, collection=articles,
                                         title=title, subtitle=subtitle, act_name=act_name)
            return articles
        section_title = section.get("TITLE") if title is None else f"{title}_{section.get('TITLE')}"
        section_subtitle = section.get("SUBTITLE") if title is None else f"{subtitle}_{section.get('SUBTITLE')}"

        # Only iterate if it contains a list of articles
        if "ARTICLE" in section:
            for art in section.get("ARTICLE",[]):
                article_entry = self.parse_article(art, section_title=section_title, section_subtitle=section_subtitle, act_name=act_name)
                articles.append(article_entry)
            return articles
        return self.parse_section(section["SECTION"], collection=articles,
                                  title=section_title, subtitle=section_subtitle, act_name=act_name)


    def parse_articles(self, json_data, act_name=''):
        articles = []
        sections = json_data['DOCUMENT']['SECTION']
        for i, section in enumerate(sections):
            if isinstance(section, list):
                for sec in section:
                    section_data = self.parse_section(sec,act_name=act_name)
                    articles.extend(section_data)
            else:
                section_data = self.parse_section(section,act_name=act_name)
                articles.extend(section_data)
        return pd.DataFrame(articles)

    def Read_in_Json_from_text(self, Orig_dir ='', all_jason_list_path =''):
        with open(all_jason_list_path) as fp: 
            Lines = fp.readlines() 
            files =[]
            acts =[]
            for line in Lines: 
                with open(os.path.join(Orig_dir, line.strip()), "r") as f:
                    data_1 = json.load(f)
                files.append(data_1)
                acts.append(re.sub('\.json$', '', line.strip()))
        return files, acts
    
    def merge_replicated_entries(self, df, matching_coloumn_name='', merging_coloumn_name=''):
        K = df[df.duplicated(subset=[matching_coloumn_name], keep=False)]
        df = df[~df.duplicated(subset=[matching_coloumn_name], keep=False)]

        q = K.article_title.unique()
        j =1
        for i in q:
            P =K.loc[K[matching_coloumn_name] == q[0]].reset_index(drop=True)
            temp_df = P.iloc[[0]]
            for j in range(1, P.shape[0]):
                temp_df.loc[0, merging_coloumn_name] = temp_df.loc[0, merging_coloumn_name]+ P.loc[j, merging_coloumn_name]
            df = df.append(temp_df, ignore_index=True, sort=False)

        return df
#%% preprocess text
# Goes to Util:
class Text_cleaning(object):
    def __init__(self, additional_stopwords_FullPath):
        stop_words_nltk = set(stopwords.words('english'))
        stopwords_final = set(w.rstrip() for w in open(additional_stopwords_FullPath))#'/home/brain/AiTalents/Notebooks/stopwords.txt'))
        stopwords_final = set([s.replace('\n', '') for s in stopwords_final])
        stopwords_final = stopwords_final.union(stop_words_nltk)
        self.stopwords_final = stopwords_final
    def clean_html(self, raw_text):
        cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        cleantext = re.sub(cleanr, '', raw_text)
        return cleantext

    def remove_URL(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'',text)

    
    def my_tokenizer(self, s):
        s = s.lower() # downcase
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
        tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
        tokens = [t for t in tokens if t not in string.punctuation] # remove punctuation        
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove any digits, i.e. "3rd edition"
        return tokens

    def preprocess_text(self, text):
        temp_text = self.remove_URL(text)
        no_html = self.clean_html(temp_text)
        return self.my_tokenizer(no_html)
    def Word_map_generation(self, df):
        #put clean Text into an array:
        contents = df["article_content"].values
        word_index_map = {}
        current_index = 0
        all_tokens = []
        all_content = []
        index_word_map = []
        error_count = 0
        for content in contents:
            try:
                content = content.encode('ascii', 'ignore').decode('utf-8') # this will throw exception if bad characters
                all_content.append(content)
                tokens = self.preprocess_text(content)
                all_tokens.append(tokens)
                for token in tokens:
                    if token not in word_index_map:
                        word_index_map[token] = current_index
                        current_index += 1
                        index_word_map.append(token)
            except Exception as e:
                print(e)
                print(title)
                error_count += 1
        return word_index_map, index_word_map,all_tokens, contents, error_count, self.stopwords_final


    def sklearn_Countvectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
                
        tf_vectorizer = CountVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                        token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                        tokenizer= self.preprocess_text)

        dtm_tf = tf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return tf_vectorizer, dtm_tf ## This is already a sparse feature matrix that could be fed into lda or SVD.
    def sklearn_TfidfVectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
        
        Tfidf_vectorizer = TfidfVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                           token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                           tokenizer= self.preprocess_text)

        dtm_Tfidf = Tfidf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return Tfidf_vectorizer, dtm_Tfidf ## This is already a sparse feature matrix that could be fed into lda or SVD.
            
    
    
#%% Feature generation:
#Goes to Util
class Feature_design(object):
    def __init__(self, word_index_map, all_tokens):
        self.word_index_map = word_index_map
        self.all_tokens = all_tokens
    # now let's create our input matrices - just indicator variables for this example - works better than proportions
    def tokens_to_vector(self, tokens):
        x = np.zeros(len(self.word_index_map))
        for t in tokens:
            i = self.word_index_map[t]
            x[i] = 1
        return x
    
    def Feature_matrix_generation(self):
        N = len(self.all_tokens)
        D = len(self.word_index_map)
        X = np.zeros((D, N)) # terms will go along rows, documents along columns
        i = 0
        for tokens in all_tokens:
            X[:,i] = self.tokens_to_vector(tokens)
            i += 1
        return X
    def Genism_Feature_matrix_generation(self, saving_dir):
        #analysis_dir= '/home/brain/AiTalents/Notebooks/Genism/'
        try:
            os.makedirs(saving_dir)
        except OSError:
            if not os.path.isdir(saving_dir):
                raise

        #os.chdir(analysis_dir)
        dictionary = corpora.Dictionary(self.all_tokens)
        corpus = [dictionary.doc2bow(text) for text in self.all_tokens]
        pickle.dump(corpus, open(os.path.join(saving_dir,'corpus.pkl'), 'wb'))
        dictionary.save(os.path.join(saving_dir,'dictionary.gensim'))
        return corpus, dictionary

#%% Topic models generation:
#Goes to Util
class Topic_modeling(object):
    def __init__(self, num_topics, saving_dir):
        
        
        self.num_topics = num_topics
        self.saving_dir = saving_dir
        try:
            os.makedirs(self.saving_dir)
        except OSError:
            if not os.path.isdir(self.saving_dir):
                raise
    def Genism_LDA(self, corpus, dictionary):
        #saving_dir= '/home/brain/AiTalents/Notebooks/Genism/'
            
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = self.num_topics, id2word=dictionary, passes=15)
        ldamodel.save(os.path.join(self.saving_dir,'LDA_model' + str(self.num_topics)+ '.gensim'))
        return ldamodel
    def Genism_LSI(self, corpus, dictionary):
        lsimodel = gensim.models.lsimodel.LsiModel(corpus, num_topics = self.num_topics, id2word=dictionary)
        lsimodel.save(os.path.join(self.saving_dir,'LSI_model' + str(self.num_topics)+ '.gensim'))
        return lsimodel
    def Sklearn_LDA(self, Feature_X):
        ldamodel = LatentDirichletAllocation()
        ldamodel.fit(Feature_X)
        Z = ldamodel.transform(Feature_X) # size: #term X #Component    >> For word Cloud
        # How about the graph connections?   componennts_ : size: #comps X #Articles  >> each column gives weights.
        U = ldamodel.components_    
        return ldamodel, Z, U 

            
#topics = ldamodel.print_topics(num_words=5)
#for topic in topics:
#    print(topic)

#%% Figures:

#Goes to Util
class Genism_plotting(object):
    def __init__(self,Figure_saving_dir):
        self.Figure_saving_dir = Figure_saving_dir
        try:
            os.makedirs(self.Figure_saving_dir)
        except OSError:
            if not os.path.isdir(self.Figure_saving_dir):
                raise
    def Convert(self, tup, di): 
        di = dict(tup) 
        return di

    def Reverse(self, tuples): 
        new_tup = () 
        for k in reversed(tuples): 
            new_tup = new_tup + (k,) 
        return new_tup 

    def pyLDA_Dashboard(self,ldamodel, corpus, dictionary,html_name, sort_topics=False):
        
        lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=sort_topics)
        pyLDAvis.save_html(lda_display, os.path.join(self.Figure_saving_dir,html_name + '_Dashboard.html'))
        #pyLDAvis.display(lda_display)
        return lda_display
    def Genism_word_cloud(self,html_name, ldamodel):
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

        cloud = WordCloud(background_color='#D8D8D8',
                          width=2500,
                          height=1800,
                          max_words=10,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)
        
        topic_word_box = ldamodel.show_topics(num_topics= ldamodel.num_topics, num_words= np.int(ldamodel.num_terms/100), formatted=False)
        # For each topic:
        topic_word_box_R= []
        for i in np.arange(ldamodel.num_topics):
            temp=[]
            temp = list(self.Reverse(topic_word_box[i][1]))
            topic_word_box_R.append(temp)
        
        D = ldamodel.num_topics
        CL = round(math.sqrt(D))
        RW = CL
        while RW * CL < D:
            RW += 1
        fig, axes = plt.subplots(RW, CL, figsize=(10,10), sharex=True, sharey=True)
        plt.style.context('grayscale')
        fig.patch.set_facecolor('#D8D8D8')

        Topic_names = ['Reporting', 'Investment', 'Risk_Management','Accounting', 'Payments_and_Settlement']
        #['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']
        #
        for i, ax in enumerate(axes.flatten()):
            if i >= D:
                axes.flatten()[i].axis('off')
            else:
            
                fig.add_subplot(ax)
                #topic_words = dict(topics[i][1])
                topic_words={}
                print(i)
                topic_words = self.Convert(topic_word_box_R[i], topic_words)
                cloud.generate_from_frequencies(topic_words, max_font_size=300)
                plt.gca().imshow(cloud)
                
                plt.gca().set_title(Topic_names[i] , fontdict=dict(size=16))
                #Topic_names.append('Topic_' + str(i +1))
                plt.gca().axis('off')


        plt.subplots_adjust(wspace=0, hspace=0)
        
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        
        #fig.patch.set_alpha(0.7)
        #plt.gca().set_facecolor('tab:gray')
        #cloud.to_file(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'))
        fig.savefig(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'), transparent=True)
        
        return 
                
class Genism_Graph(object):
    def __init__(self, Figure_saving_dir='', verbose = True):
        
        self.Figure_saving_dir = Figure_saving_dir
        self.verbose = verbose
        
    def Genism_networkX_graph(self, df, ldamodel):
        Article_names = df['article_title'].values
        
        Topic_names = ['Topic_'+ str(i) for i in range(1, len(ldamodel.show_topics())+1)]
        nx_graph = nx.Graph()
        Nodes = Topic_names + list(Article_names)
        # Add node for each Topic
        for T in Nodes:
            nx_graph.add_node(T)

        i = 0   
        # add edges
        for a in range(0,len(Article_names)):
            temp = ldamodel.get_document_topics(corpus[a])
            for c in range(0, len(temp)):
                if self.verbose:
                    
                    print('C: ', c)
                    print(len(ldamodel.get_document_topics(corpus[a])))

                T = Topic_names[temp[c][0]]
                if self.verbose:
                    print('T, A: ',T, Article_names[a])
                i += 1
                if self.verbose:
                    print('i:', i)
                # Only add edge if the count is positive

                nx_graph.add_edge(Article_names[a],T, weight = temp[c][1])
                if self.verbose:
                    print('Done_edge')
        return nx_graph
    def plot_Genism_networkX_graph(self,df, ldamodel, to_bundel=True ):
        nx_graph = self.Genism_networkX_graph(df, ldamodel)
        #Create the graph using networkX data:
        act_names = []
        kwargs = dict(width=800, height=800, xaxis=None, yaxis=None)
        opts.defaults(opts.Nodes(**kwargs), opts.Graph(**kwargs))
        
        #colors = ['#000000']+hv.Cycle('Category20').values
        topic_graph = hv.Graph.from_networkx(nx_graph, nx.layout.fruchterman_reingold_layout, k=1)
        topic_graph.nodes.data['act'] = topic_graph.nodes.data['index']
        for i in np.arange(topic_graph.nodes.data.shape[0]):
            if topic_graph.nodes.data.loc[i, 'act'] not in ['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']:
                topic_graph.nodes.data.loc[i, 'act'] = str(topic_graph.nodes.data.loc[i, 'act'].split('_')[1])
                
        
        tmp2= topic_graph.nodes.data[~topic_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        #colors = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*np.arange(len(tmp)), 20+2*np.arange(len(tmp)))]
        
        if to_bundel:

            bundled = bundle_graph(topic_graph)
            #.opts(opts.Nodes(color='act', size=10, width=1000, cmap=colors, legend_position='right'))
#opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            #
            return bundled, act_names, article_names
        else:
            return topic_graph, act_names, article_names
        #.opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            
    
    def subselect_Genism_single_Topic_graph(self, Topic_id, topic_graph):
        
        Sub_graph = topic_graph.select(start=Topic_id).opts(node_fill_color='blue', edge_line_width=hv.dim('weight'), edge_line_color='red')
        tmp2= Sub_graph.nodes.data[~Sub_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        
        return Sub_graph, act_names, article_names
    def subselect_Genism_single_Article_graph(self, Article_title, topic_graph, circular_layout=True):
        
        D = topic_graph.select(end=Article_title).opts(node_fill_color='white', edge_line_width=-1*np.log10(hv.dim('weight')))
        
        if circular_layout:
            
            F = layout_nodes(D)
            labels = hv.Labels(F.nodes, ['x', 'y'], 'index')

            return (F * labels.opts(text_font_size='10pt', text_color='black', bgcolor='white'))
        else: 
            
            return D#Goes to Util
class Text_parsing(object):
    def __init__(self):
        
        return 

    def parse_article(self, article, section_title, section_subtitle, act_name):
        article_title = article.get("TITLE")
        article_subtitle = article.get("SUBTITLE")
        article_content = article.get("CONTENT")
        assert article_title is not None
        assert article_subtitle is not None
        assert article_content is not None
        article = {"act_name":act_name,
                   "section_title": section_title,
                   "section_subtitle": section_subtitle,
                   "article_title": article_title + '_' + act_name,# + article_subtitle,
                   "article_subtitle":article_subtitle,
                   "article_content": article_content,
                  }
        return article

    def parse_section(self, section, collection=None, title=None, subtitle=None, act_name = ''):
        articles = [] if collection is None else collection
        if isinstance(section, list):
            for sec in section:
                articles = self.parse_section(sec, collection=articles,
                                         title=title, subtitle=subtitle, act_name=act_name)
            return articles
        section_title = section.get("TITLE") if title is None else f"{title}_{section.get('TITLE')}"
        section_subtitle = section.get("SUBTITLE") if title is None else f"{subtitle}_{section.get('SUBTITLE')}"

        # Only iterate if it contains a list of articles
        if "ARTICLE" in section:
            for art in section.get("ARTICLE",[]):
                article_entry = self.parse_article(art, section_title=section_title, section_subtitle=section_subtitle, act_name=act_name)
                articles.append(article_entry)
            return articles
        return self.parse_section(section["SECTION"], collection=articles,
                                  title=section_title, subtitle=section_subtitle, act_name=act_name)


    def parse_articles(self, json_data, act_name=''):
        articles = []
        sections = json_data['DOCUMENT']['SECTION']
        for i, section in enumerate(sections):
            if isinstance(section, list):
                for sec in section:
                    section_data = self.parse_section(sec,act_name=act_name)
                    articles.extend(section_data)
            else:
                section_data = self.parse_section(section,act_name=act_name)
                articles.extend(section_data)
        return pd.DataFrame(articles)

    def Read_in_Json_from_text(self, Orig_dir ='', all_jason_list_path =''):
        with open(all_jason_list_path) as fp: 
            Lines = fp.readlines() 
            files =[]
            acts =[]
            for line in Lines: 
                with open(os.path.join(Orig_dir, line.strip()), "r") as f:
                    data_1 = json.load(f)
                files.append(data_1)
                acts.append(re.sub('\.json$', '', line.strip()))
        return files, acts
    
    def merge_replicated_entries(self, df, matching_coloumn_name='', merging_coloumn_name=''):
        K = df[df.duplicated(subset=[matching_coloumn_name], keep=False)]
        df = df[~df.duplicated(subset=[matching_coloumn_name], keep=False)]

        q = K.article_title.unique()
        j =1
        for i in q:
            P =K.loc[K[matching_coloumn_name] == q[0]].reset_index(drop=True)
            temp_df = P.iloc[[0]]
            for j in range(1, P.shape[0]):
                temp_df.loc[0, merging_coloumn_name] = temp_df.loc[0, merging_coloumn_name]+ P.loc[j, merging_coloumn_name]
            df = df.append(temp_df, ignore_index=True, sort=False)

        return df
#%% preprocess text
# Goes to Util:
class Text_cleaning(object):
    def __init__(self, additional_stopwords_FullPath):
        stop_words_nltk = set(stopwords.words('english'))
        stopwords_final = set(w.rstrip() for w in open(additional_stopwords_FullPath))#'/home/brain/AiTalents/Notebooks/stopwords.txt'))
        stopwords_final = set([s.replace('\n', '') for s in stopwords_final])
        stopwords_final = stopwords_final.union(stop_words_nltk)
        self.stopwords_final = stopwords_final
    def clean_html(self, raw_text):
        cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        cleantext = re.sub(cleanr, '', raw_text)
        return cleantext

    def remove_URL(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'',text)

    
    def my_tokenizer(self, s):
        s = s.lower() # downcase
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
        tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
        tokens = [t for t in tokens if t not in string.punctuation] # remove punctuation        
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove any digits, i.e. "3rd edition"
        return tokens

    def preprocess_text(self, text):
        temp_text = self.remove_URL(text)
        no_html = self.clean_html(temp_text)
        return self.my_tokenizer(no_html)
    def Word_map_generation(self, df):
        #put clean Text into an array:
        contents = df["article_content"].values
        word_index_map = {}
        current_index = 0
        all_tokens = []
        all_content = []
        index_word_map = []
        error_count = 0
        for content in contents:
            try:
                content = content.encode('ascii', 'ignore').decode('utf-8') # this will throw exception if bad characters
                all_content.append(content)
                tokens = self.preprocess_text(content)
                all_tokens.append(tokens)
                for token in tokens:
                    if token not in word_index_map:
                        word_index_map[token] = current_index
                        current_index += 1
                        index_word_map.append(token)
            except Exception as e:
                print(e)
                print(title)
                error_count += 1
        return word_index_map, index_word_map,all_tokens, contents, error_count, self.stopwords_final


    def sklearn_Countvectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
                
        tf_vectorizer = CountVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                        token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                        tokenizer= self.preprocess_text)

        dtm_tf = tf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return tf_vectorizer, dtm_tf ## This is already a sparse feature matrix that could be fed into lda or SVD.
    def sklearn_TfidfVectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
        
        Tfidf_vectorizer = TfidfVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                           token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                           tokenizer= self.preprocess_text)

        dtm_Tfidf = Tfidf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return Tfidf_vectorizer, dtm_Tfidf ## This is already a sparse feature matrix that could be fed into lda or SVD.
            
    
    
#%% Feature generation:
#Goes to Util
class Feature_design(object):
    def __init__(self, word_index_map, all_tokens):
        self.word_index_map = word_index_map
        self.all_tokens = all_tokens
    # now let's create our input matrices - just indicator variables for this example - works better than proportions
    def tokens_to_vector(self, tokens):
        x = np.zeros(len(self.word_index_map))
        for t in tokens:
            i = self.word_index_map[t]
            x[i] = 1
        return x
    
    def Feature_matrix_generation(self):
        N = len(self.all_tokens)
        D = len(self.word_index_map)
        X = np.zeros((D, N)) # terms will go along rows, documents along columns
        i = 0
        for tokens in all_tokens:
            X[:,i] = self.tokens_to_vector(tokens)
            i += 1
        return X
    def Genism_Feature_matrix_generation(self, saving_dir):
        #analysis_dir= '/home/brain/AiTalents/Notebooks/Genism/'
        try:
            os.makedirs(saving_dir)
        except OSError:
            if not os.path.isdir(saving_dir):
                raise

        #os.chdir(analysis_dir)
        dictionary = corpora.Dictionary(self.all_tokens)
        corpus = [dictionary.doc2bow(text) for text in self.all_tokens]
        pickle.dump(corpus, open(os.path.join(saving_dir,'corpus.pkl'), 'wb'))
        dictionary.save(os.path.join(saving_dir,'dictionary.gensim'))
        return corpus, dictionary

#%% Topic models generation:
#Goes to Util
class Topic_modeling(object):
    def __init__(self, num_topics, saving_dir):
        
        
        self.num_topics = num_topics
        self.saving_dir = saving_dir
        try:
            os.makedirs(self.saving_dir)
        except OSError:
            if not os.path.isdir(self.saving_dir):
                raise
    def Genism_LDA(self, corpus, dictionary):
        #saving_dir= '/home/brain/AiTalents/Notebooks/Genism/'
            
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = self.num_topics, id2word=dictionary, passes=15)
        ldamodel.save(os.path.join(self.saving_dir,'LDA_model' + str(self.num_topics)+ '.gensim'))
        return ldamodel
    def Genism_LSI(self, corpus, dictionary):
        lsimodel = gensim.models.lsimodel.LsiModel(corpus, num_topics = self.num_topics, id2word=dictionary)
        lsimodel.save(os.path.join(self.saving_dir,'LSI_model' + str(self.num_topics)+ '.gensim'))
        return lsimodel
    def Sklearn_LDA(self, Feature_X):
        ldamodel = LatentDirichletAllocation()
        ldamodel.fit(Feature_X)
        Z = ldamodel.transform(Feature_X) # size: #term X #Component    >> For word Cloud
        # How about the graph connections?   componennts_ : size: #comps X #Articles  >> each column gives weights.
        U = ldamodel.components_    
        return ldamodel, Z, U 

            
#topics = ldamodel.print_topics(num_words=5)
#for topic in topics:
#    print(topic)

#%% Figures:

#Goes to Util
class Genism_plotting(object):
    def __init__(self,Figure_saving_dir):
        self.Figure_saving_dir = Figure_saving_dir
        try:
            os.makedirs(self.Figure_saving_dir)
        except OSError:
            if not os.path.isdir(self.Figure_saving_dir):
                raise
    def Convert(self, tup, di): 
        di = dict(tup) 
        return di

    def Reverse(self, tuples): 
        new_tup = () 
        for k in reversed(tuples): 
            new_tup = new_tup + (k,) 
        return new_tup 

    def pyLDA_Dashboard(self,ldamodel, corpus, dictionary,html_name, sort_topics=False):
        
        lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=sort_topics)
        pyLDAvis.save_html(lda_display, os.path.join(self.Figure_saving_dir,html_name + '_Dashboard.html'))
        #pyLDAvis.display(lda_display)
        return lda_display
    def Genism_word_cloud(self,html_name, ldamodel):
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

        cloud = WordCloud(background_color='#D8D8D8',
                          width=2500,
                          height=1800,
                          max_words=10,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)
        
        topic_word_box = ldamodel.show_topics(num_topics= ldamodel.num_topics, num_words= np.int(ldamodel.num_terms/100), formatted=False)
        # For each topic:
        topic_word_box_R= []
        for i in np.arange(ldamodel.num_topics):
            temp=[]
            temp = list(self.Reverse(topic_word_box[i][1]))
            topic_word_box_R.append(temp)
        
        D = ldamodel.num_topics
        CL = round(math.sqrt(D))
        RW = CL
        while RW * CL < D:
            RW += 1
        fig, axes = plt.subplots(RW, CL, figsize=(10,10), sharex=True, sharey=True)
        plt.style.context('grayscale')
        fig.patch.set_facecolor('#D8D8D8')

        Topic_names = ['Reporting', 'Investment', 'Risk_Management','Accounting', 'Payments_and_Settlement']
        #['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']
        #
        for i, ax in enumerate(axes.flatten()):
            if i >= D:
                axes.flatten()[i].axis('off')
            else:
            
                fig.add_subplot(ax)
                #topic_words = dict(topics[i][1])
                topic_words={}
                print(i)
                topic_words = self.Convert(topic_word_box_R[i], topic_words)
                cloud.generate_from_frequencies(topic_words, max_font_size=300)
                plt.gca().imshow(cloud)
                
                plt.gca().set_title(Topic_names[i] , fontdict=dict(size=16))
                #Topic_names.append('Topic_' + str(i +1))
                plt.gca().axis('off')


        plt.subplots_adjust(wspace=0, hspace=0)
        
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        
        #fig.patch.set_alpha(0.7)
        #plt.gca().set_facecolor('tab:gray')
        #cloud.to_file(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'))
        fig.savefig(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'), transparent=True)
        
        return 
                
class Genism_Graph(object):
    def __init__(self, Figure_saving_dir='', verbose = True):
        
        self.Figure_saving_dir = Figure_saving_dir
        self.verbose = verbose
        
    def Genism_networkX_graph(self, df, ldamodel):
        Article_names = df['article_title'].values
        
        Topic_names = ['Topic_'+ str(i) for i in range(1, len(ldamodel.show_topics())+1)]
        nx_graph = nx.Graph()
        Nodes = Topic_names + list(Article_names)
        # Add node for each Topic
        for T in Nodes:
            nx_graph.add_node(T)

        i = 0   
        # add edges
        for a in range(0,len(Article_names)):
            temp = ldamodel.get_document_topics(corpus[a])
            for c in range(0, len(temp)):
                if self.verbose:
                    
                    print('C: ', c)
                    print(len(ldamodel.get_document_topics(corpus[a])))

                T = Topic_names[temp[c][0]]
                if self.verbose:
                    print('T, A: ',T, Article_names[a])
                i += 1
                if self.verbose:
                    print('i:', i)
                # Only add edge if the count is positive

                nx_graph.add_edge(Article_names[a],T, weight = temp[c][1])
                if self.verbose:
                    print('Done_edge')
        return nx_graph
    def plot_Genism_networkX_graph(self,df, ldamodel, to_bundel=True ):
        nx_graph = self.Genism_networkX_graph(df, ldamodel)
        #Create the graph using networkX data:
        act_names = []
        kwargs = dict(width=800, height=800, xaxis=None, yaxis=None)
        opts.defaults(opts.Nodes(**kwargs), opts.Graph(**kwargs))
        
        #colors = ['#000000']+hv.Cycle('Category20').values
        topic_graph = hv.Graph.from_networkx(nx_graph, nx.layout.fruchterman_reingold_layout, k=1)
        topic_graph.nodes.data['act'] = topic_graph.nodes.data['index']
        for i in np.arange(topic_graph.nodes.data.shape[0]):
            if topic_graph.nodes.data.loc[i, 'act'] not in ['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']:
                topic_graph.nodes.data.loc[i, 'act'] = str(topic_graph.nodes.data.loc[i, 'act'].split('_')[1])
                
        
        tmp2= topic_graph.nodes.data[~topic_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        #colors = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*np.arange(len(tmp)), 20+2*np.arange(len(tmp)))]
        
        if to_bundel:

            bundled = bundle_graph(topic_graph)
            #.opts(opts.Nodes(color='act', size=10, width=1000, cmap=colors, legend_position='right'))
#opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            #
            return bundled, act_names, article_names
        else:
            return topic_graph, act_names, article_names
        #.opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            
    
    def subselect_Genism_single_Topic_graph(self, Topic_id, topic_graph):
        
        Sub_graph = topic_graph.select(start=Topic_id).opts(node_fill_color='blue', edge_line_width=hv.dim('weight'), edge_line_color='red')
        tmp2= Sub_graph.nodes.data[~Sub_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        
        return Sub_graph, act_names, article_names
    def subselect_Genism_single_Article_graph(self, Article_title, topic_graph, circular_layout=True):
        
        D = topic_graph.select(end=Article_title).opts(node_fill_color='white', edge_line_width=-1*np.log10(hv.dim('weight')))
        
        if circular_layout:
            
            F = layout_nodes(D)
            labels = hv.Labels(F.nodes, ['x', 'y'], 'index')

            return (F * labels.opts(text_font_size='10pt', text_color='black', bgcolor='white'))
        else: 
            
            return D#Goes to Util
class Text_parsing(object):
    def __init__(self):
        
        return 

    def parse_article(self, article, section_title, section_subtitle, act_name):
        article_title = article.get("TITLE")
        article_subtitle = article.get("SUBTITLE")
        article_content = article.get("CONTENT")
        assert article_title is not None
        assert article_subtitle is not None
        assert article_content is not None
        article = {"act_name":act_name,
                   "section_title": section_title,
                   "section_subtitle": section_subtitle,
                   "article_title": article_title + '_' + act_name,# + article_subtitle,
                   "article_subtitle":article_subtitle,
                   "article_content": article_content,
                  }
        return article

    def parse_section(self, section, collection=None, title=None, subtitle=None, act_name = ''):
        articles = [] if collection is None else collection
        if isinstance(section, list):
            for sec in section:
                articles = self.parse_section(sec, collection=articles,
                                         title=title, subtitle=subtitle, act_name=act_name)
            return articles
        section_title = section.get("TITLE") if title is None else f"{title}_{section.get('TITLE')}"
        section_subtitle = section.get("SUBTITLE") if title is None else f"{subtitle}_{section.get('SUBTITLE')}"

        # Only iterate if it contains a list of articles
        if "ARTICLE" in section:
            for art in section.get("ARTICLE",[]):
                article_entry = self.parse_article(art, section_title=section_title, section_subtitle=section_subtitle, act_name=act_name)
                articles.append(article_entry)
            return articles
        return self.parse_section(section["SECTION"], collection=articles,
                                  title=section_title, subtitle=section_subtitle, act_name=act_name)


    def parse_articles(self, json_data, act_name=''):
        articles = []
        sections = json_data['DOCUMENT']['SECTION']
        for i, section in enumerate(sections):
            if isinstance(section, list):
                for sec in section:
                    section_data = self.parse_section(sec,act_name=act_name)
                    articles.extend(section_data)
            else:
                section_data = self.parse_section(section,act_name=act_name)
                articles.extend(section_data)
        return pd.DataFrame(articles)

    def Read_in_Json_from_text(self, Orig_dir ='', all_jason_list_path =''):
        with open(all_jason_list_path) as fp: 
            Lines = fp.readlines() 
            files =[]
            acts =[]
            for line in Lines: 
                with open(os.path.join(Orig_dir, line.strip()), "r") as f:
                    data_1 = json.load(f)
                files.append(data_1)
                acts.append(re.sub('\.json$', '', line.strip()))
        return files, acts
    
    def merge_replicated_entries(self, df, matching_coloumn_name='', merging_coloumn_name=''):
        K = df[df.duplicated(subset=[matching_coloumn_name], keep=False)]
        df = df[~df.duplicated(subset=[matching_coloumn_name], keep=False)]

        q = K.article_title.unique()
        j =1
        for i in q:
            P =K.loc[K[matching_coloumn_name] == q[0]].reset_index(drop=True)
            temp_df = P.iloc[[0]]
            for j in range(1, P.shape[0]):
                temp_df.loc[0, merging_coloumn_name] = temp_df.loc[0, merging_coloumn_name]+ P.loc[j, merging_coloumn_name]
            df = df.append(temp_df, ignore_index=True, sort=False)

        return df
#%% preprocess text
# Goes to Util:
class Text_cleaning(object):
    def __init__(self, additional_stopwords_FullPath):
        stop_words_nltk = set(stopwords.words('english'))
        stopwords_final = set(w.rstrip() for w in open(additional_stopwords_FullPath))#'/home/brain/AiTalents/Notebooks/stopwords.txt'))
        stopwords_final = set([s.replace('\n', '') for s in stopwords_final])
        stopwords_final = stopwords_final.union(stop_words_nltk)
        self.stopwords_final = stopwords_final
    def clean_html(self, raw_text):
        cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        cleantext = re.sub(cleanr, '', raw_text)
        return cleantext

    def remove_URL(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'',text)

    
    def my_tokenizer(self, s):
        s = s.lower() # downcase
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
        tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
        tokens = [t for t in tokens if t not in string.punctuation] # remove punctuation        
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove any digits, i.e. "3rd edition"
        return tokens

    def preprocess_text(self, text):
        temp_text = self.remove_URL(text)
        no_html = self.clean_html(temp_text)
        return self.my_tokenizer(no_html)
    def Word_map_generation(self, df):
        #put clean Text into an array:
        contents = df["article_content"].values
        word_index_map = {}
        current_index = 0
        all_tokens = []
        all_content = []
        index_word_map = []
        error_count = 0
        for content in contents:
            try:
                content = content.encode('ascii', 'ignore').decode('utf-8') # this will throw exception if bad characters
                all_content.append(content)
                tokens = self.preprocess_text(content)
                all_tokens.append(tokens)
                for token in tokens:
                    if token not in word_index_map:
                        word_index_map[token] = current_index
                        current_index += 1
                        index_word_map.append(token)
            except Exception as e:
                print(e)
                print(title)
                error_count += 1
        return word_index_map, index_word_map,all_tokens, contents, error_count, self.stopwords_final


    def sklearn_Countvectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
                
        tf_vectorizer = CountVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                        token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                        tokenizer= self.preprocess_text)

        dtm_tf = tf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return tf_vectorizer, dtm_tf ## This is already a sparse feature matrix that could be fed into lda or SVD.
    def sklearn_TfidfVectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
        
        Tfidf_vectorizer = TfidfVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                           token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                           tokenizer= self.preprocess_text)

        dtm_Tfidf = Tfidf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return Tfidf_vectorizer, dtm_Tfidf ## This is already a sparse feature matrix that could be fed into lda or SVD.
            
    
    
#%% Feature generation:
#Goes to Util
class Feature_design(object):
    def __init__(self, word_index_map, all_tokens):
        self.word_index_map = word_index_map
        self.all_tokens = all_tokens
    # now let's create our input matrices - just indicator variables for this example - works better than proportions
    def tokens_to_vector(self, tokens):
        x = np.zeros(len(self.word_index_map))
        for t in tokens:
            i = self.word_index_map[t]
            x[i] = 1
        return x
    
    def Feature_matrix_generation(self):
        N = len(self.all_tokens)
        D = len(self.word_index_map)
        X = np.zeros((D, N)) # terms will go along rows, documents along columns
        i = 0
        for tokens in all_tokens:
            X[:,i] = self.tokens_to_vector(tokens)
            i += 1
        return X
    def Genism_Feature_matrix_generation(self, saving_dir):
        #analysis_dir= '/home/brain/AiTalents/Notebooks/Genism/'
        try:
            os.makedirs(saving_dir)
        except OSError:
            if not os.path.isdir(saving_dir):
                raise

        #os.chdir(analysis_dir)
        dictionary = corpora.Dictionary(self.all_tokens)
        corpus = [dictionary.doc2bow(text) for text in self.all_tokens]
        pickle.dump(corpus, open(os.path.join(saving_dir,'corpus.pkl'), 'wb'))
        dictionary.save(os.path.join(saving_dir,'dictionary.gensim'))
        return corpus, dictionary

#%% Topic models generation:
#Goes to Util
class Topic_modeling(object):
    def __init__(self, num_topics, saving_dir):
        
        
        self.num_topics = num_topics
        self.saving_dir = saving_dir
        try:
            os.makedirs(self.saving_dir)
        except OSError:
            if not os.path.isdir(self.saving_dir):
                raise
    def Genism_LDA(self, corpus, dictionary):
        #saving_dir= '/home/brain/AiTalents/Notebooks/Genism/'
            
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = self.num_topics, id2word=dictionary, passes=15)
        ldamodel.save(os.path.join(self.saving_dir,'LDA_model' + str(self.num_topics)+ '.gensim'))
        return ldamodel
    def Genism_LSI(self, corpus, dictionary):
        lsimodel = gensim.models.lsimodel.LsiModel(corpus, num_topics = self.num_topics, id2word=dictionary)
        lsimodel.save(os.path.join(self.saving_dir,'LSI_model' + str(self.num_topics)+ '.gensim'))
        return lsimodel
    def Sklearn_LDA(self, Feature_X):
        ldamodel = LatentDirichletAllocation()
        ldamodel.fit(Feature_X)
        Z = ldamodel.transform(Feature_X) # size: #term X #Component    >> For word Cloud
        # How about the graph connections?   componennts_ : size: #comps X #Articles  >> each column gives weights.
        U = ldamodel.components_    
        return ldamodel, Z, U 

            
#topics = ldamodel.print_topics(num_words=5)
#for topic in topics:
#    print(topic)

#%% Figures:

#Goes to Util
class Genism_plotting(object):
    def __init__(self,Figure_saving_dir):
        self.Figure_saving_dir = Figure_saving_dir
        try:
            os.makedirs(self.Figure_saving_dir)
        except OSError:
            if not os.path.isdir(self.Figure_saving_dir):
                raise
    def Convert(self, tup, di): 
        di = dict(tup) 
        return di

    def Reverse(self, tuples): 
        new_tup = () 
        for k in reversed(tuples): 
            new_tup = new_tup + (k,) 
        return new_tup 

    def pyLDA_Dashboard(self,ldamodel, corpus, dictionary,html_name, sort_topics=False):
        
        lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=sort_topics)
        pyLDAvis.save_html(lda_display, os.path.join(self.Figure_saving_dir,html_name + '_Dashboard.html'))
        #pyLDAvis.display(lda_display)
        return lda_display
    def Genism_word_cloud(self,html_name, ldamodel):
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

        cloud = WordCloud(background_color='#D8D8D8',
                          width=2500,
                          height=1800,
                          max_words=10,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)
        
        topic_word_box = ldamodel.show_topics(num_topics= ldamodel.num_topics, num_words= np.int(ldamodel.num_terms/100), formatted=False)
        # For each topic:
        topic_word_box_R= []
        for i in np.arange(ldamodel.num_topics):
            temp=[]
            temp = list(self.Reverse(topic_word_box[i][1]))
            topic_word_box_R.append(temp)
        
        D = ldamodel.num_topics
        CL = round(math.sqrt(D))
        RW = CL
        while RW * CL < D:
            RW += 1
        fig, axes = plt.subplots(RW, CL, figsize=(10,10), sharex=True, sharey=True)
        plt.style.context('grayscale')
        fig.patch.set_facecolor('#D8D8D8')

        Topic_names = ['Reporting', 'Investment', 'Risk_Management','Accounting', 'Payments_and_Settlement']
        #['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']
        #
        for i, ax in enumerate(axes.flatten()):
            if i >= D:
                axes.flatten()[i].axis('off')
            else:
            
                fig.add_subplot(ax)
                #topic_words = dict(topics[i][1])
                topic_words={}
                print(i)
                topic_words = self.Convert(topic_word_box_R[i], topic_words)
                cloud.generate_from_frequencies(topic_words, max_font_size=300)
                plt.gca().imshow(cloud)
                
                plt.gca().set_title(Topic_names[i] , fontdict=dict(size=16))
                #Topic_names.append('Topic_' + str(i +1))
                plt.gca().axis('off')


        plt.subplots_adjust(wspace=0, hspace=0)
        
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        
        #fig.patch.set_alpha(0.7)
        #plt.gca().set_facecolor('tab:gray')
        #cloud.to_file(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'))
        fig.savefig(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'), transparent=True)
        
        return 
                
class Genism_Graph(object):
    def __init__(self, Figure_saving_dir='', verbose = True):
        
        self.Figure_saving_dir = Figure_saving_dir
        self.verbose = verbose
        
    def Genism_networkX_graph(self, df, ldamodel):
        Article_names = df['article_title'].values
        
        Topic_names = ['Topic_'+ str(i) for i in range(1, len(ldamodel.show_topics())+1)]
        nx_graph = nx.Graph()
        Nodes = Topic_names + list(Article_names)
        # Add node for each Topic
        for T in Nodes:
            nx_graph.add_node(T)

        i = 0   
        # add edges
        for a in range(0,len(Article_names)):
            temp = ldamodel.get_document_topics(corpus[a])
            for c in range(0, len(temp)):
                if self.verbose:
                    
                    print('C: ', c)
                    print(len(ldamodel.get_document_topics(corpus[a])))

                T = Topic_names[temp[c][0]]
                if self.verbose:
                    print('T, A: ',T, Article_names[a])
                i += 1
                if self.verbose:
                    print('i:', i)
                # Only add edge if the count is positive

                nx_graph.add_edge(Article_names[a],T, weight = temp[c][1])
                if self.verbose:
                    print('Done_edge')
        return nx_graph
    def plot_Genism_networkX_graph(self,df, ldamodel, to_bundel=True ):
        nx_graph = self.Genism_networkX_graph(df, ldamodel)
        #Create the graph using networkX data:
        act_names = []
        kwargs = dict(width=800, height=800, xaxis=None, yaxis=None)
        opts.defaults(opts.Nodes(**kwargs), opts.Graph(**kwargs))
        
        #colors = ['#000000']+hv.Cycle('Category20').values
        topic_graph = hv.Graph.from_networkx(nx_graph, nx.layout.fruchterman_reingold_layout, k=1)
        topic_graph.nodes.data['act'] = topic_graph.nodes.data['index']
        for i in np.arange(topic_graph.nodes.data.shape[0]):
            if topic_graph.nodes.data.loc[i, 'act'] not in ['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']:
                topic_graph.nodes.data.loc[i, 'act'] = str(topic_graph.nodes.data.loc[i, 'act'].split('_')[1])
                
        
        tmp2= topic_graph.nodes.data[~topic_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        #colors = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*np.arange(len(tmp)), 20+2*np.arange(len(tmp)))]
        
        if to_bundel:

            bundled = bundle_graph(topic_graph)
            #.opts(opts.Nodes(color='act', size=10, width=1000, cmap=colors, legend_position='right'))
#opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            #
            return bundled, act_names, article_names
        else:
            return topic_graph, act_names, article_names
        #.opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            
    
    def subselect_Genism_single_Topic_graph(self, Topic_id, topic_graph):
        
        Sub_graph = topic_graph.select(start=Topic_id).opts(node_fill_color='blue', edge_line_width=hv.dim('weight'), edge_line_color='red')
        tmp2= Sub_graph.nodes.data[~Sub_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        
        return Sub_graph, act_names, article_names
    def subselect_Genism_single_Article_graph(self, Article_title, topic_graph, circular_layout=True):
        
        D = topic_graph.select(end=Article_title).opts(node_fill_color='white', edge_line_width=-1*np.log10(hv.dim('weight')))
        
        if circular_layout:
            
            F = layout_nodes(D)
            labels = hv.Labels(F.nodes, ['x', 'y'], 'index')

            return (F * labels.opts(text_font_size='10pt', text_color='black', bgcolor='white'))
        else: 
            
            return D#Goes to Util
class Text_parsing(object):
    def __init__(self):
        
        return 

    def parse_article(self, article, section_title, section_subtitle, act_name):
        article_title = article.get("TITLE")
        article_subtitle = article.get("SUBTITLE")
        article_content = article.get("CONTENT")
        assert article_title is not None
        assert article_subtitle is not None
        assert article_content is not None
        article = {"act_name":act_name,
                   "section_title": section_title,
                   "section_subtitle": section_subtitle,
                   "article_title": article_title + '_' + act_name,# + article_subtitle,
                   "article_subtitle":article_subtitle,
                   "article_content": article_content,
                  }
        return article

    def parse_section(self, section, collection=None, title=None, subtitle=None, act_name = ''):
        articles = [] if collection is None else collection
        if isinstance(section, list):
            for sec in section:
                articles = self.parse_section(sec, collection=articles,
                                         title=title, subtitle=subtitle, act_name=act_name)
            return articles
        section_title = section.get("TITLE") if title is None else f"{title}_{section.get('TITLE')}"
        section_subtitle = section.get("SUBTITLE") if title is None else f"{subtitle}_{section.get('SUBTITLE')}"

        # Only iterate if it contains a list of articles
        if "ARTICLE" in section:
            for art in section.get("ARTICLE",[]):
                article_entry = self.parse_article(art, section_title=section_title, section_subtitle=section_subtitle, act_name=act_name)
                articles.append(article_entry)
            return articles
        return self.parse_section(section["SECTION"], collection=articles,
                                  title=section_title, subtitle=section_subtitle, act_name=act_name)


    def parse_articles(self, json_data, act_name=''):
        articles = []
        sections = json_data['DOCUMENT']['SECTION']
        for i, section in enumerate(sections):
            if isinstance(section, list):
                for sec in section:
                    section_data = self.parse_section(sec,act_name=act_name)
                    articles.extend(section_data)
            else:
                section_data = self.parse_section(section,act_name=act_name)
                articles.extend(section_data)
        return pd.DataFrame(articles)

    def Read_in_Json_from_text(self, Orig_dir ='', all_jason_list_path =''):
        with open(all_jason_list_path) as fp: 
            Lines = fp.readlines() 
            files =[]
            acts =[]
            for line in Lines: 
                with open(os.path.join(Orig_dir, line.strip()), "r") as f:
                    data_1 = json.load(f)
                files.append(data_1)
                acts.append(re.sub('\.json$', '', line.strip()))
        return files, acts
    
    def merge_replicated_entries(self, df, matching_coloumn_name='', merging_coloumn_name=''):
        K = df[df.duplicated(subset=[matching_coloumn_name], keep=False)]
        df = df[~df.duplicated(subset=[matching_coloumn_name], keep=False)]

        q = K.article_title.unique()
        j =1
        for i in q:
            P =K.loc[K[matching_coloumn_name] == q[0]].reset_index(drop=True)
            temp_df = P.iloc[[0]]
            for j in range(1, P.shape[0]):
                temp_df.loc[0, merging_coloumn_name] = temp_df.loc[0, merging_coloumn_name]+ P.loc[j, merging_coloumn_name]
            df = df.append(temp_df, ignore_index=True, sort=False)

        return df
#%% preprocess text
# Goes to Util:
class Text_cleaning(object):
    def __init__(self, additional_stopwords_FullPath):
        stop_words_nltk = set(stopwords.words('english'))
        stopwords_final = set(w.rstrip() for w in open(additional_stopwords_FullPath))#'/home/brain/AiTalents/Notebooks/stopwords.txt'))
        stopwords_final = set([s.replace('\n', '') for s in stopwords_final])
        stopwords_final = stopwords_final.union(stop_words_nltk)
        self.stopwords_final = stopwords_final
    def clean_html(self, raw_text):
        cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        cleantext = re.sub(cleanr, '', raw_text)
        return cleantext

    def remove_URL(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'',text)

    
    def my_tokenizer(self, s):
        s = s.lower() # downcase
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
        tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
        tokens = [t for t in tokens if t not in string.punctuation] # remove punctuation        
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove any digits, i.e. "3rd edition"
        return tokens

    def preprocess_text(self, text):
        temp_text = self.remove_URL(text)
        no_html = self.clean_html(temp_text)
        return self.my_tokenizer(no_html)
    def Word_map_generation(self, df):
        #put clean Text into an array:
        contents = df["article_content"].values
        word_index_map = {}
        current_index = 0
        all_tokens = []
        all_content = []
        index_word_map = []
        error_count = 0
        for content in contents:
            try:
                content = content.encode('ascii', 'ignore').decode('utf-8') # this will throw exception if bad characters
                all_content.append(content)
                tokens = self.preprocess_text(content)
                all_tokens.append(tokens)
                for token in tokens:
                    if token not in word_index_map:
                        word_index_map[token] = current_index
                        current_index += 1
                        index_word_map.append(token)
            except Exception as e:
                print(e)
                print(title)
                error_count += 1
        return word_index_map, index_word_map,all_tokens, contents, error_count, self.stopwords_final


    def sklearn_Countvectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
                
        tf_vectorizer = CountVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                        token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                        tokenizer= self.preprocess_text)

        dtm_tf = tf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return tf_vectorizer, dtm_tf ## This is already a sparse feature matrix that could be fed into lda or SVD.
    def sklearn_TfidfVectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
        
        Tfidf_vectorizer = TfidfVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                           token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                           tokenizer= self.preprocess_text)

        dtm_Tfidf = Tfidf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return Tfidf_vectorizer, dtm_Tfidf ## This is already a sparse feature matrix that could be fed into lda or SVD.
            
    
    
#%% Feature generation:
#Goes to Util
class Feature_design(object):
    def __init__(self, word_index_map, all_tokens):
        self.word_index_map = word_index_map
        self.all_tokens = all_tokens
    # now let's create our input matrices - just indicator variables for this example - works better than proportions
    def tokens_to_vector(self, tokens):
        x = np.zeros(len(self.word_index_map))
        for t in tokens:
            i = self.word_index_map[t]
            x[i] = 1
        return x
    
    def Feature_matrix_generation(self):
        N = len(self.all_tokens)
        D = len(self.word_index_map)
        X = np.zeros((D, N)) # terms will go along rows, documents along columns
        i = 0
        for tokens in all_tokens:
            X[:,i] = self.tokens_to_vector(tokens)
            i += 1
        return X
    def Genism_Feature_matrix_generation(self, saving_dir):
        #analysis_dir= '/home/brain/AiTalents/Notebooks/Genism/'
        try:
            os.makedirs(saving_dir)
        except OSError:
            if not os.path.isdir(saving_dir):
                raise

        #os.chdir(analysis_dir)
        dictionary = corpora.Dictionary(self.all_tokens)
        corpus = [dictionary.doc2bow(text) for text in self.all_tokens]
        pickle.dump(corpus, open(os.path.join(saving_dir,'corpus.pkl'), 'wb'))
        dictionary.save(os.path.join(saving_dir,'dictionary.gensim'))
        return corpus, dictionary

#%% Topic models generation:
#Goes to Util
class Topic_modeling(object):
    def __init__(self, num_topics, saving_dir):
        
        
        self.num_topics = num_topics
        self.saving_dir = saving_dir
        try:
            os.makedirs(self.saving_dir)
        except OSError:
            if not os.path.isdir(self.saving_dir):
                raise
    def Genism_LDA(self, corpus, dictionary):
        #saving_dir= '/home/brain/AiTalents/Notebooks/Genism/'
            
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = self.num_topics, id2word=dictionary, passes=15)
        ldamodel.save(os.path.join(self.saving_dir,'LDA_model' + str(self.num_topics)+ '.gensim'))
        return ldamodel
    def Genism_LSI(self, corpus, dictionary):
        lsimodel = gensim.models.lsimodel.LsiModel(corpus, num_topics = self.num_topics, id2word=dictionary)
        lsimodel.save(os.path.join(self.saving_dir,'LSI_model' + str(self.num_topics)+ '.gensim'))
        return lsimodel
    def Sklearn_LDA(self, Feature_X):
        ldamodel = LatentDirichletAllocation()
        ldamodel.fit(Feature_X)
        Z = ldamodel.transform(Feature_X) # size: #term X #Component    >> For word Cloud
        # How about the graph connections?   componennts_ : size: #comps X #Articles  >> each column gives weights.
        U = ldamodel.components_    
        return ldamodel, Z, U 

            
#topics = ldamodel.print_topics(num_words=5)
#for topic in topics:
#    print(topic)

#%% Figures:

#Goes to Util
class Genism_plotting(object):
    def __init__(self,Figure_saving_dir):
        self.Figure_saving_dir = Figure_saving_dir
        try:
            os.makedirs(self.Figure_saving_dir)
        except OSError:
            if not os.path.isdir(self.Figure_saving_dir):
                raise
    def Convert(self, tup, di): 
        di = dict(tup) 
        return di

    def Reverse(self, tuples): 
        new_tup = () 
        for k in reversed(tuples): 
            new_tup = new_tup + (k,) 
        return new_tup 

    def pyLDA_Dashboard(self,ldamodel, corpus, dictionary,html_name, sort_topics=False):
        
        lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=sort_topics)
        pyLDAvis.save_html(lda_display, os.path.join(self.Figure_saving_dir,html_name + '_Dashboard.html'))
        #pyLDAvis.display(lda_display)
        return lda_display
    def Genism_word_cloud(self,html_name, ldamodel):
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

        cloud = WordCloud(background_color='#D8D8D8',
                          width=2500,
                          height=1800,
                          max_words=10,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)
        
        topic_word_box = ldamodel.show_topics(num_topics= ldamodel.num_topics, num_words= np.int(ldamodel.num_terms/100), formatted=False)
        # For each topic:
        topic_word_box_R= []
        for i in np.arange(ldamodel.num_topics):
            temp=[]
            temp = list(self.Reverse(topic_word_box[i][1]))
            topic_word_box_R.append(temp)
        
        D = ldamodel.num_topics
        CL = round(math.sqrt(D))
        RW = CL
        while RW * CL < D:
            RW += 1
        fig, axes = plt.subplots(RW, CL, figsize=(10,10), sharex=True, sharey=True)
        plt.style.context('grayscale')
        fig.patch.set_facecolor('#D8D8D8')

        Topic_names = ['Reporting', 'Investment', 'Risk_Management','Accounting', 'Payments_and_Settlement']
        #['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']
        #
        for i, ax in enumerate(axes.flatten()):
            if i >= D:
                axes.flatten()[i].axis('off')
            else:
            
                fig.add_subplot(ax)
                #topic_words = dict(topics[i][1])
                topic_words={}
                print(i)
                topic_words = self.Convert(topic_word_box_R[i], topic_words)
                cloud.generate_from_frequencies(topic_words, max_font_size=300)
                plt.gca().imshow(cloud)
                
                plt.gca().set_title(Topic_names[i] , fontdict=dict(size=16))
                #Topic_names.append('Topic_' + str(i +1))
                plt.gca().axis('off')


        plt.subplots_adjust(wspace=0, hspace=0)
        
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        
        #fig.patch.set_alpha(0.7)
        #plt.gca().set_facecolor('tab:gray')
        #cloud.to_file(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'))
        fig.savefig(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'), transparent=True)
        
        return 
                
class Genism_Graph(object):
    def __init__(self, Figure_saving_dir='', verbose = True):
        
        self.Figure_saving_dir = Figure_saving_dir
        self.verbose = verbose
        
    def Genism_networkX_graph(self, df, ldamodel):
        Article_names = df['article_title'].values
        
        Topic_names = ['Topic_'+ str(i) for i in range(1, len(ldamodel.show_topics())+1)]
        nx_graph = nx.Graph()
        Nodes = Topic_names + list(Article_names)
        # Add node for each Topic
        for T in Nodes:
            nx_graph.add_node(T)

        i = 0   
        # add edges
        for a in range(0,len(Article_names)):
            temp = ldamodel.get_document_topics(corpus[a])
            for c in range(0, len(temp)):
                if self.verbose:
                    
                    print('C: ', c)
                    print(len(ldamodel.get_document_topics(corpus[a])))

                T = Topic_names[temp[c][0]]
                if self.verbose:
                    print('T, A: ',T, Article_names[a])
                i += 1
                if self.verbose:
                    print('i:', i)
                # Only add edge if the count is positive

                nx_graph.add_edge(Article_names[a],T, weight = temp[c][1])
                if self.verbose:
                    print('Done_edge')
        return nx_graph
    def plot_Genism_networkX_graph(self,df, ldamodel, to_bundel=True ):
        nx_graph = self.Genism_networkX_graph(df, ldamodel)
        #Create the graph using networkX data:
        act_names = []
        kwargs = dict(width=800, height=800, xaxis=None, yaxis=None)
        opts.defaults(opts.Nodes(**kwargs), opts.Graph(**kwargs))
        
        #colors = ['#000000']+hv.Cycle('Category20').values
        topic_graph = hv.Graph.from_networkx(nx_graph, nx.layout.fruchterman_reingold_layout, k=1)
        topic_graph.nodes.data['act'] = topic_graph.nodes.data['index']
        for i in np.arange(topic_graph.nodes.data.shape[0]):
            if topic_graph.nodes.data.loc[i, 'act'] not in ['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']:
                topic_graph.nodes.data.loc[i, 'act'] = str(topic_graph.nodes.data.loc[i, 'act'].split('_')[1])
                
        
        tmp2= topic_graph.nodes.data[~topic_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        #colors = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*np.arange(len(tmp)), 20+2*np.arange(len(tmp)))]
        
        if to_bundel:

            bundled = bundle_graph(topic_graph)
            #.opts(opts.Nodes(color='act', size=10, width=1000, cmap=colors, legend_position='right'))
#opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            #
            return bundled, act_names, article_names
        else:
            return topic_graph, act_names, article_names
        #.opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            
    
    def subselect_Genism_single_Topic_graph(self, Topic_id, topic_graph):
        
        Sub_graph = topic_graph.select(start=Topic_id).opts(node_fill_color='blue', edge_line_width=hv.dim('weight'), edge_line_color='red')
        tmp2= Sub_graph.nodes.data[~Sub_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        
        return Sub_graph, act_names, article_names
    def subselect_Genism_single_Article_graph(self, Article_title, topic_graph, circular_layout=True):
        
        D = topic_graph.select(end=Article_title).opts(node_fill_color='white', edge_line_width=-1*np.log10(hv.dim('weight')))
        
        if circular_layout:
            
            F = layout_nodes(D)
            labels = hv.Labels(F.nodes, ['x', 'y'], 'index')

            return (F * labels.opts(text_font_size='10pt', text_color='black', bgcolor='white'))
        else: 
            
            return D#Goes to Util
class Text_parsing(object):
    def __init__(self):
        
        return 

    def parse_article(self, article, section_title, section_subtitle, act_name):
        article_title = article.get("TITLE")
        article_subtitle = article.get("SUBTITLE")
        article_content = article.get("CONTENT")
        assert article_title is not None
        assert article_subtitle is not None
        assert article_content is not None
        article = {"act_name":act_name,
                   "section_title": section_title,
                   "section_subtitle": section_subtitle,
                   "article_title": article_title + '_' + act_name,# + article_subtitle,
                   "article_subtitle":article_subtitle,
                   "article_content": article_content,
                  }
        return article

    def parse_section(self, section, collection=None, title=None, subtitle=None, act_name = ''):
        articles = [] if collection is None else collection
        if isinstance(section, list):
            for sec in section:
                articles = self.parse_section(sec, collection=articles,
                                         title=title, subtitle=subtitle, act_name=act_name)
            return articles
        section_title = section.get("TITLE") if title is None else f"{title}_{section.get('TITLE')}"
        section_subtitle = section.get("SUBTITLE") if title is None else f"{subtitle}_{section.get('SUBTITLE')}"

        # Only iterate if it contains a list of articles
        if "ARTICLE" in section:
            for art in section.get("ARTICLE",[]):
                article_entry = self.parse_article(art, section_title=section_title, section_subtitle=section_subtitle, act_name=act_name)
                articles.append(article_entry)
            return articles
        return self.parse_section(section["SECTION"], collection=articles,
                                  title=section_title, subtitle=section_subtitle, act_name=act_name)


    def parse_articles(self, json_data, act_name=''):
        articles = []
        sections = json_data['DOCUMENT']['SECTION']
        for i, section in enumerate(sections):
            if isinstance(section, list):
                for sec in section:
                    section_data = self.parse_section(sec,act_name=act_name)
                    articles.extend(section_data)
            else:
                section_data = self.parse_section(section,act_name=act_name)
                articles.extend(section_data)
        return pd.DataFrame(articles)

    def Read_in_Json_from_text(self, Orig_dir ='', all_jason_list_path =''):
        with open(all_jason_list_path) as fp: 
            Lines = fp.readlines() 
            files =[]
            acts =[]
            for line in Lines: 
                with open(os.path.join(Orig_dir, line.strip()), "r") as f:
                    data_1 = json.load(f)
                files.append(data_1)
                acts.append(re.sub('\.json$', '', line.strip()))
        return files, acts
    
    def merge_replicated_entries(self, df, matching_coloumn_name='', merging_coloumn_name=''):
        K = df[df.duplicated(subset=[matching_coloumn_name], keep=False)]
        df = df[~df.duplicated(subset=[matching_coloumn_name], keep=False)]

        q = K.article_title.unique()
        j =1
        for i in q:
            P =K.loc[K[matching_coloumn_name] == q[0]].reset_index(drop=True)
            temp_df = P.iloc[[0]]
            for j in range(1, P.shape[0]):
                temp_df.loc[0, merging_coloumn_name] = temp_df.loc[0, merging_coloumn_name]+ P.loc[j, merging_coloumn_name]
            df = df.append(temp_df, ignore_index=True, sort=False)

        return df
#%% preprocess text
# Goes to Util:
class Text_cleaning(object):
    def __init__(self, additional_stopwords_FullPath):
        stop_words_nltk = set(stopwords.words('english'))
        stopwords_final = set(w.rstrip() for w in open(additional_stopwords_FullPath))#'/home/brain/AiTalents/Notebooks/stopwords.txt'))
        stopwords_final = set([s.replace('\n', '') for s in stopwords_final])
        stopwords_final = stopwords_final.union(stop_words_nltk)
        self.stopwords_final = stopwords_final
    def clean_html(self, raw_text):
        cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        cleantext = re.sub(cleanr, '', raw_text)
        return cleantext

    def remove_URL(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'',text)

    
    def my_tokenizer(self, s):
        s = s.lower() # downcase
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
        tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
        tokens = [t for t in tokens if t not in string.punctuation] # remove punctuation        
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove any digits, i.e. "3rd edition"
        return tokens

    def preprocess_text(self, text):
        temp_text = self.remove_URL(text)
        no_html = self.clean_html(temp_text)
        return self.my_tokenizer(no_html)
    def Word_map_generation(self, df):
        #put clean Text into an array:
        contents = df["article_content"].values
        word_index_map = {}
        current_index = 0
        all_tokens = []
        all_content = []
        index_word_map = []
        error_count = 0
        for content in contents:
            try:
                content = content.encode('ascii', 'ignore').decode('utf-8') # this will throw exception if bad characters
                all_content.append(content)
                tokens = self.preprocess_text(content)
                all_tokens.append(tokens)
                for token in tokens:
                    if token not in word_index_map:
                        word_index_map[token] = current_index
                        current_index += 1
                        index_word_map.append(token)
            except Exception as e:
                print(e)
                print(title)
                error_count += 1
        return word_index_map, index_word_map,all_tokens, contents, error_count, self.stopwords_final


    def sklearn_Countvectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
                
        tf_vectorizer = CountVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                        token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                        tokenizer= self.preprocess_text)

        dtm_tf = tf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return tf_vectorizer, dtm_tf ## This is already a sparse feature matrix that could be fed into lda or SVD.
    def sklearn_TfidfVectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
        
        Tfidf_vectorizer = TfidfVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                           token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                           tokenizer= self.preprocess_text)

        dtm_Tfidf = Tfidf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return Tfidf_vectorizer, dtm_Tfidf ## This is already a sparse feature matrix that could be fed into lda or SVD.
            
    
    
#%% Feature generation:
#Goes to Util
class Feature_design(object):
    def __init__(self, word_index_map, all_tokens):
        self.word_index_map = word_index_map
        self.all_tokens = all_tokens
    # now let's create our input matrices - just indicator variables for this example - works better than proportions
    def tokens_to_vector(self, tokens):
        x = np.zeros(len(self.word_index_map))
        for t in tokens:
            i = self.word_index_map[t]
            x[i] = 1
        return x
    
    def Feature_matrix_generation(self):
        N = len(self.all_tokens)
        D = len(self.word_index_map)
        X = np.zeros((D, N)) # terms will go along rows, documents along columns
        i = 0
        for tokens in all_tokens:
            X[:,i] = self.tokens_to_vector(tokens)
            i += 1
        return X
    def Genism_Feature_matrix_generation(self, saving_dir):
        #analysis_dir= '/home/brain/AiTalents/Notebooks/Genism/'
        try:
            os.makedirs(saving_dir)
        except OSError:
            if not os.path.isdir(saving_dir):
                raise

        #os.chdir(analysis_dir)
        dictionary = corpora.Dictionary(self.all_tokens)
        corpus = [dictionary.doc2bow(text) for text in self.all_tokens]
        pickle.dump(corpus, open(os.path.join(saving_dir,'corpus.pkl'), 'wb'))
        dictionary.save(os.path.join(saving_dir,'dictionary.gensim'))
        return corpus, dictionary

#%% Topic models generation:
#Goes to Util
class Topic_modeling(object):
    def __init__(self, num_topics, saving_dir):
        
        
        self.num_topics = num_topics
        self.saving_dir = saving_dir
        try:
            os.makedirs(self.saving_dir)
        except OSError:
            if not os.path.isdir(self.saving_dir):
                raise
    def Genism_LDA(self, corpus, dictionary):
        #saving_dir= '/home/brain/AiTalents/Notebooks/Genism/'
            
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = self.num_topics, id2word=dictionary, passes=15)
        ldamodel.save(os.path.join(self.saving_dir,'LDA_model' + str(self.num_topics)+ '.gensim'))
        return ldamodel
    def Genism_LSI(self, corpus, dictionary):
        lsimodel = gensim.models.lsimodel.LsiModel(corpus, num_topics = self.num_topics, id2word=dictionary)
        lsimodel.save(os.path.join(self.saving_dir,'LSI_model' + str(self.num_topics)+ '.gensim'))
        return lsimodel
    def Sklearn_LDA(self, Feature_X):
        ldamodel = LatentDirichletAllocation()
        ldamodel.fit(Feature_X)
        Z = ldamodel.transform(Feature_X) # size: #term X #Component    >> For word Cloud
        # How about the graph connections?   componennts_ : size: #comps X #Articles  >> each column gives weights.
        U = ldamodel.components_    
        return ldamodel, Z, U 

            
#topics = ldamodel.print_topics(num_words=5)
#for topic in topics:
#    print(topic)

#%% Figures:

#Goes to Util
class Genism_plotting(object):
    def __init__(self,Figure_saving_dir):
        self.Figure_saving_dir = Figure_saving_dir
        try:
            os.makedirs(self.Figure_saving_dir)
        except OSError:
            if not os.path.isdir(self.Figure_saving_dir):
                raise
    def Convert(self, tup, di): 
        di = dict(tup) 
        return di

    def Reverse(self, tuples): 
        new_tup = () 
        for k in reversed(tuples): 
            new_tup = new_tup + (k,) 
        return new_tup 

    def pyLDA_Dashboard(self,ldamodel, corpus, dictionary,html_name, sort_topics=False):
        
        lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=sort_topics)
        pyLDAvis.save_html(lda_display, os.path.join(self.Figure_saving_dir,html_name + '_Dashboard.html'))
        #pyLDAvis.display(lda_display)
        return lda_display
    def Genism_word_cloud(self,html_name, ldamodel):
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

        cloud = WordCloud(background_color='#D8D8D8',
                          width=2500,
                          height=1800,
                          max_words=10,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)
        
        topic_word_box = ldamodel.show_topics(num_topics= ldamodel.num_topics, num_words= np.int(ldamodel.num_terms/100), formatted=False)
        # For each topic:
        topic_word_box_R= []
        for i in np.arange(ldamodel.num_topics):
            temp=[]
            temp = list(self.Reverse(topic_word_box[i][1]))
            topic_word_box_R.append(temp)
        
        D = ldamodel.num_topics
        CL = round(math.sqrt(D))
        RW = CL
        while RW * CL < D:
            RW += 1
        fig, axes = plt.subplots(RW, CL, figsize=(10,10), sharex=True, sharey=True)
        plt.style.context('grayscale')
        fig.patch.set_facecolor('#D8D8D8')

        Topic_names = ['Reporting', 'Investment', 'Risk_Management','Accounting', 'Payments_and_Settlement']
        #['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']
        #
        for i, ax in enumerate(axes.flatten()):
            if i >= D:
                axes.flatten()[i].axis('off')
            else:
            
                fig.add_subplot(ax)
                #topic_words = dict(topics[i][1])
                topic_words={}
                print(i)
                topic_words = self.Convert(topic_word_box_R[i], topic_words)
                cloud.generate_from_frequencies(topic_words, max_font_size=300)
                plt.gca().imshow(cloud)
                
                plt.gca().set_title(Topic_names[i] , fontdict=dict(size=16))
                #Topic_names.append('Topic_' + str(i +1))
                plt.gca().axis('off')


        plt.subplots_adjust(wspace=0, hspace=0)
        
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        
        #fig.patch.set_alpha(0.7)
        #plt.gca().set_facecolor('tab:gray')
        #cloud.to_file(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'))
        fig.savefig(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'), transparent=True)
        
        return 
                
class Genism_Graph(object):
    def __init__(self, Figure_saving_dir='', verbose = True):
        
        self.Figure_saving_dir = Figure_saving_dir
        self.verbose = verbose
        
    def Genism_networkX_graph(self, df, ldamodel):
        Article_names = df['article_title'].values
        
        Topic_names = ['Topic_'+ str(i) for i in range(1, len(ldamodel.show_topics())+1)]
        nx_graph = nx.Graph()
        Nodes = Topic_names + list(Article_names)
        # Add node for each Topic
        for T in Nodes:
            nx_graph.add_node(T)

        i = 0   
        # add edges
        for a in range(0,len(Article_names)):
            temp = ldamodel.get_document_topics(corpus[a])
            for c in range(0, len(temp)):
                if self.verbose:
                    
                    print('C: ', c)
                    print(len(ldamodel.get_document_topics(corpus[a])))

                T = Topic_names[temp[c][0]]
                if self.verbose:
                    print('T, A: ',T, Article_names[a])
                i += 1
                if self.verbose:
                    print('i:', i)
                # Only add edge if the count is positive

                nx_graph.add_edge(Article_names[a],T, weight = temp[c][1])
                if self.verbose:
                    print('Done_edge')
        return nx_graph
    def plot_Genism_networkX_graph(self,df, ldamodel, to_bundel=True ):
        nx_graph = self.Genism_networkX_graph(df, ldamodel)
        #Create the graph using networkX data:
        act_names = []
        kwargs = dict(width=800, height=800, xaxis=None, yaxis=None)
        opts.defaults(opts.Nodes(**kwargs), opts.Graph(**kwargs))
        
        #colors = ['#000000']+hv.Cycle('Category20').values
        topic_graph = hv.Graph.from_networkx(nx_graph, nx.layout.fruchterman_reingold_layout, k=1)
        topic_graph.nodes.data['act'] = topic_graph.nodes.data['index']
        for i in np.arange(topic_graph.nodes.data.shape[0]):
            if topic_graph.nodes.data.loc[i, 'act'] not in ['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']:
                topic_graph.nodes.data.loc[i, 'act'] = str(topic_graph.nodes.data.loc[i, 'act'].split('_')[1])
                
        
        tmp2= topic_graph.nodes.data[~topic_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        #colors = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*np.arange(len(tmp)), 20+2*np.arange(len(tmp)))]
        
        if to_bundel:

            bundled = bundle_graph(topic_graph)
            #.opts(opts.Nodes(color='act', size=10, width=1000, cmap=colors, legend_position='right'))
#opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            #
            return bundled, act_names, article_names
        else:
            return topic_graph, act_names, article_names
        #.opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            
    
    def subselect_Genism_single_Topic_graph(self, Topic_id, topic_graph):
        
        Sub_graph = topic_graph.select(start=Topic_id).opts(node_fill_color='blue', edge_line_width=hv.dim('weight'), edge_line_color='red')
        tmp2= Sub_graph.nodes.data[~Sub_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        
        return Sub_graph, act_names, article_names
    def subselect_Genism_single_Article_graph(self, Article_title, topic_graph, circular_layout=True):
        
        D = topic_graph.select(end=Article_title).opts(node_fill_color='white', edge_line_width=-1*np.log10(hv.dim('weight')))
        
        if circular_layout:
            
            F = layout_nodes(D)
            labels = hv.Labels(F.nodes, ['x', 'y'], 'index')

            return (F * labels.opts(text_font_size='10pt', text_color='black', bgcolor='white'))
        else: 
            
            return D#Goes to Util
class Text_parsing(object):
    def __init__(self):
        
        return 

    def parse_article(self, article, section_title, section_subtitle, act_name):
        article_title = article.get("TITLE")
        article_subtitle = article.get("SUBTITLE")
        article_content = article.get("CONTENT")
        assert article_title is not None
        assert article_subtitle is not None
        assert article_content is not None
        article = {"act_name":act_name,
                   "section_title": section_title,
                   "section_subtitle": section_subtitle,
                   "article_title": article_title + '_' + act_name,# + article_subtitle,
                   "article_subtitle":article_subtitle,
                   "article_content": article_content,
                  }
        return article

    def parse_section(self, section, collection=None, title=None, subtitle=None, act_name = ''):
        articles = [] if collection is None else collection
        if isinstance(section, list):
            for sec in section:
                articles = self.parse_section(sec, collection=articles,
                                         title=title, subtitle=subtitle, act_name=act_name)
            return articles
        section_title = section.get("TITLE") if title is None else f"{title}_{section.get('TITLE')}"
        section_subtitle = section.get("SUBTITLE") if title is None else f"{subtitle}_{section.get('SUBTITLE')}"

        # Only iterate if it contains a list of articles
        if "ARTICLE" in section:
            for art in section.get("ARTICLE",[]):
                article_entry = self.parse_article(art, section_title=section_title, section_subtitle=section_subtitle, act_name=act_name)
                articles.append(article_entry)
            return articles
        return self.parse_section(section["SECTION"], collection=articles,
                                  title=section_title, subtitle=section_subtitle, act_name=act_name)


    def parse_articles(self, json_data, act_name=''):
        articles = []
        sections = json_data['DOCUMENT']['SECTION']
        for i, section in enumerate(sections):
            if isinstance(section, list):
                for sec in section:
                    section_data = self.parse_section(sec,act_name=act_name)
                    articles.extend(section_data)
            else:
                section_data = self.parse_section(section,act_name=act_name)
                articles.extend(section_data)
        return pd.DataFrame(articles)

    def Read_in_Json_from_text(self, Orig_dir ='', all_jason_list_path =''):
        with open(all_jason_list_path) as fp: 
            Lines = fp.readlines() 
            files =[]
            acts =[]
            for line in Lines: 
                with open(os.path.join(Orig_dir, line.strip()), "r") as f:
                    data_1 = json.load(f)
                files.append(data_1)
                acts.append(re.sub('\.json$', '', line.strip()))
        return files, acts
    
    def merge_replicated_entries(self, df, matching_coloumn_name='', merging_coloumn_name=''):
        K = df[df.duplicated(subset=[matching_coloumn_name], keep=False)]
        df = df[~df.duplicated(subset=[matching_coloumn_name], keep=False)]

        q = K.article_title.unique()
        j =1
        for i in q:
            P =K.loc[K[matching_coloumn_name] == q[0]].reset_index(drop=True)
            temp_df = P.iloc[[0]]
            for j in range(1, P.shape[0]):
                temp_df.loc[0, merging_coloumn_name] = temp_df.loc[0, merging_coloumn_name]+ P.loc[j, merging_coloumn_name]
            df = df.append(temp_df, ignore_index=True, sort=False)

        return df
#%% preprocess text
# Goes to Util:
class Text_cleaning(object):
    def __init__(self, additional_stopwords_FullPath):
        stop_words_nltk = set(stopwords.words('english'))
        stopwords_final = set(w.rstrip() for w in open(additional_stopwords_FullPath))#'/home/brain/AiTalents/Notebooks/stopwords.txt'))
        stopwords_final = set([s.replace('\n', '') for s in stopwords_final])
        stopwords_final = stopwords_final.union(stop_words_nltk)
        self.stopwords_final = stopwords_final
    def clean_html(self, raw_text):
        cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        cleantext = re.sub(cleanr, '', raw_text)
        return cleantext

    def remove_URL(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'',text)

    
    def my_tokenizer(self, s):
        s = s.lower() # downcase
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
        tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
        tokens = [t for t in tokens if t not in string.punctuation] # remove punctuation        
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove any digits, i.e. "3rd edition"
        return tokens

    def preprocess_text(self, text):
        temp_text = self.remove_URL(text)
        no_html = self.clean_html(temp_text)
        return self.my_tokenizer(no_html)
    def Word_map_generation(self, df):
        #put clean Text into an array:
        contents = df["article_content"].values
        word_index_map = {}
        current_index = 0
        all_tokens = []
        all_content = []
        index_word_map = []
        error_count = 0
        for content in contents:
            try:
                content = content.encode('ascii', 'ignore').decode('utf-8') # this will throw exception if bad characters
                all_content.append(content)
                tokens = self.preprocess_text(content)
                all_tokens.append(tokens)
                for token in tokens:
                    if token not in word_index_map:
                        word_index_map[token] = current_index
                        current_index += 1
                        index_word_map.append(token)
            except Exception as e:
                print(e)
                print(title)
                error_count += 1
        return word_index_map, index_word_map,all_tokens, contents, error_count, self.stopwords_final


    def sklearn_Countvectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
                
        tf_vectorizer = CountVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                        token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                        tokenizer= self.preprocess_text)

        dtm_tf = tf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return tf_vectorizer, dtm_tf ## This is already a sparse feature matrix that could be fed into lda or SVD.
    def sklearn_TfidfVectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
        
        Tfidf_vectorizer = TfidfVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                           token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                           tokenizer= self.preprocess_text)

        dtm_Tfidf = Tfidf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return Tfidf_vectorizer, dtm_Tfidf ## This is already a sparse feature matrix that could be fed into lda or SVD.
            
    
    
#%% Feature generation:
#Goes to Util
class Feature_design(object):
    def __init__(self, word_index_map, all_tokens):
        self.word_index_map = word_index_map
        self.all_tokens = all_tokens
    # now let's create our input matrices - just indicator variables for this example - works better than proportions
    def tokens_to_vector(self, tokens):
        x = np.zeros(len(self.word_index_map))
        for t in tokens:
            i = self.word_index_map[t]
            x[i] = 1
        return x
    
    def Feature_matrix_generation(self):
        N = len(self.all_tokens)
        D = len(self.word_index_map)
        X = np.zeros((D, N)) # terms will go along rows, documents along columns
        i = 0
        for tokens in all_tokens:
            X[:,i] = self.tokens_to_vector(tokens)
            i += 1
        return X
    def Genism_Feature_matrix_generation(self, saving_dir):
        #analysis_dir= '/home/brain/AiTalents/Notebooks/Genism/'
        try:
            os.makedirs(saving_dir)
        except OSError:
            if not os.path.isdir(saving_dir):
                raise

        #os.chdir(analysis_dir)
        dictionary = corpora.Dictionary(self.all_tokens)
        corpus = [dictionary.doc2bow(text) for text in self.all_tokens]
        pickle.dump(corpus, open(os.path.join(saving_dir,'corpus.pkl'), 'wb'))
        dictionary.save(os.path.join(saving_dir,'dictionary.gensim'))
        return corpus, dictionary

#%% Topic models generation:
#Goes to Util
class Topic_modeling(object):
    def __init__(self, num_topics, saving_dir):
        
        
        self.num_topics = num_topics
        self.saving_dir = saving_dir
        try:
            os.makedirs(self.saving_dir)
        except OSError:
            if not os.path.isdir(self.saving_dir):
                raise
    def Genism_LDA(self, corpus, dictionary):
        #saving_dir= '/home/brain/AiTalents/Notebooks/Genism/'
            
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = self.num_topics, id2word=dictionary, passes=15)
        ldamodel.save(os.path.join(self.saving_dir,'LDA_model' + str(self.num_topics)+ '.gensim'))
        return ldamodel
    def Genism_LSI(self, corpus, dictionary):
        lsimodel = gensim.models.lsimodel.LsiModel(corpus, num_topics = self.num_topics, id2word=dictionary)
        lsimodel.save(os.path.join(self.saving_dir,'LSI_model' + str(self.num_topics)+ '.gensim'))
        return lsimodel
    def Sklearn_LDA(self, Feature_X):
        ldamodel = LatentDirichletAllocation()
        ldamodel.fit(Feature_X)
        Z = ldamodel.transform(Feature_X) # size: #term X #Component    >> For word Cloud
        # How about the graph connections?   componennts_ : size: #comps X #Articles  >> each column gives weights.
        U = ldamodel.components_    
        return ldamodel, Z, U 

            
#topics = ldamodel.print_topics(num_words=5)
#for topic in topics:
#    print(topic)

#%% Figures:

#Goes to Util
class Genism_plotting(object):
    def __init__(self,Figure_saving_dir):
        self.Figure_saving_dir = Figure_saving_dir
        try:
            os.makedirs(self.Figure_saving_dir)
        except OSError:
            if not os.path.isdir(self.Figure_saving_dir):
                raise
    def Convert(self, tup, di): 
        di = dict(tup) 
        return di

    def Reverse(self, tuples): 
        new_tup = () 
        for k in reversed(tuples): 
            new_tup = new_tup + (k,) 
        return new_tup 

    def pyLDA_Dashboard(self,ldamodel, corpus, dictionary,html_name, sort_topics=False):
        
        lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=sort_topics)
        pyLDAvis.save_html(lda_display, os.path.join(self.Figure_saving_dir,html_name + '_Dashboard.html'))
        #pyLDAvis.display(lda_display)
        return lda_display
    def Genism_word_cloud(self,html_name, ldamodel):
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

        cloud = WordCloud(background_color='#D8D8D8',
                          width=2500,
                          height=1800,
                          max_words=10,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)
        
        topic_word_box = ldamodel.show_topics(num_topics= ldamodel.num_topics, num_words= np.int(ldamodel.num_terms/100), formatted=False)
        # For each topic:
        topic_word_box_R= []
        for i in np.arange(ldamodel.num_topics):
            temp=[]
            temp = list(self.Reverse(topic_word_box[i][1]))
            topic_word_box_R.append(temp)
        
        D = ldamodel.num_topics
        CL = round(math.sqrt(D))
        RW = CL
        while RW * CL < D:
            RW += 1
        fig, axes = plt.subplots(RW, CL, figsize=(10,10), sharex=True, sharey=True)
        plt.style.context('grayscale')
        fig.patch.set_facecolor('#D8D8D8')

        Topic_names = ['Reporting', 'Investment', 'Risk_Management','Accounting', 'Payments_and_Settlement']
        #['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']
        #
        for i, ax in enumerate(axes.flatten()):
            if i >= D:
                axes.flatten()[i].axis('off')
            else:
            
                fig.add_subplot(ax)
                #topic_words = dict(topics[i][1])
                topic_words={}
                print(i)
                topic_words = self.Convert(topic_word_box_R[i], topic_words)
                cloud.generate_from_frequencies(topic_words, max_font_size=300)
                plt.gca().imshow(cloud)
                
                plt.gca().set_title(Topic_names[i] , fontdict=dict(size=16))
                #Topic_names.append('Topic_' + str(i +1))
                plt.gca().axis('off')


        plt.subplots_adjust(wspace=0, hspace=0)
        
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        
        #fig.patch.set_alpha(0.7)
        #plt.gca().set_facecolor('tab:gray')
        #cloud.to_file(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'))
        fig.savefig(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'), transparent=True)
        
        return 
                
class Genism_Graph(object):
    def __init__(self, Figure_saving_dir='', verbose = True):
        
        self.Figure_saving_dir = Figure_saving_dir
        self.verbose = verbose
        
    def Genism_networkX_graph(self, df, ldamodel):
        Article_names = df['article_title'].values
        
        Topic_names = ['Topic_'+ str(i) for i in range(1, len(ldamodel.show_topics())+1)]
        nx_graph = nx.Graph()
        Nodes = Topic_names + list(Article_names)
        # Add node for each Topic
        for T in Nodes:
            nx_graph.add_node(T)

        i = 0   
        # add edges
        for a in range(0,len(Article_names)):
            temp = ldamodel.get_document_topics(corpus[a])
            for c in range(0, len(temp)):
                if self.verbose:
                    
                    print('C: ', c)
                    print(len(ldamodel.get_document_topics(corpus[a])))

                T = Topic_names[temp[c][0]]
                if self.verbose:
                    print('T, A: ',T, Article_names[a])
                i += 1
                if self.verbose:
                    print('i:', i)
                # Only add edge if the count is positive

                nx_graph.add_edge(Article_names[a],T, weight = temp[c][1])
                if self.verbose:
                    print('Done_edge')
        return nx_graph
    def plot_Genism_networkX_graph(self,df, ldamodel, to_bundel=True ):
        nx_graph = self.Genism_networkX_graph(df, ldamodel)
        #Create the graph using networkX data:
        act_names = []
        kwargs = dict(width=800, height=800, xaxis=None, yaxis=None)
        opts.defaults(opts.Nodes(**kwargs), opts.Graph(**kwargs))
        
        #colors = ['#000000']+hv.Cycle('Category20').values
        topic_graph = hv.Graph.from_networkx(nx_graph, nx.layout.fruchterman_reingold_layout, k=1)
        topic_graph.nodes.data['act'] = topic_graph.nodes.data['index']
        for i in np.arange(topic_graph.nodes.data.shape[0]):
            if topic_graph.nodes.data.loc[i, 'act'] not in ['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']:
                topic_graph.nodes.data.loc[i, 'act'] = str(topic_graph.nodes.data.loc[i, 'act'].split('_')[1])
                
        
        tmp2= topic_graph.nodes.data[~topic_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        #colors = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*np.arange(len(tmp)), 20+2*np.arange(len(tmp)))]
        
        if to_bundel:

            bundled = bundle_graph(topic_graph)
            #.opts(opts.Nodes(color='act', size=10, width=1000, cmap=colors, legend_position='right'))
#opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            #
            return bundled, act_names, article_names
        else:
            return topic_graph, act_names, article_names
        #.opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            
    
    def subselect_Genism_single_Topic_graph(self, Topic_id, topic_graph):
        
        Sub_graph = topic_graph.select(start=Topic_id).opts(node_fill_color='blue', edge_line_width=hv.dim('weight'), edge_line_color='red')
        tmp2= Sub_graph.nodes.data[~Sub_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        
        return Sub_graph, act_names, article_names
    def subselect_Genism_single_Article_graph(self, Article_title, topic_graph, circular_layout=True):
        
        D = topic_graph.select(end=Article_title).opts(node_fill_color='white', edge_line_width=-1*np.log10(hv.dim('weight')))
        
        if circular_layout:
            
            F = layout_nodes(D)
            labels = hv.Labels(F.nodes, ['x', 'y'], 'index')

            return (F * labels.opts(text_font_size='10pt', text_color='black', bgcolor='white'))
        else: 
            
            return D#Goes to Util
class Text_parsing(object):
    def __init__(self):
        
        return 

    def parse_article(self, article, section_title, section_subtitle, act_name):
        article_title = article.get("TITLE")
        article_subtitle = article.get("SUBTITLE")
        article_content = article.get("CONTENT")
        assert article_title is not None
        assert article_subtitle is not None
        assert article_content is not None
        article = {"act_name":act_name,
                   "section_title": section_title,
                   "section_subtitle": section_subtitle,
                   "article_title": article_title + '_' + act_name,# + article_subtitle,
                   "article_subtitle":article_subtitle,
                   "article_content": article_content,
                  }
        return article

    def parse_section(self, section, collection=None, title=None, subtitle=None, act_name = ''):
        articles = [] if collection is None else collection
        if isinstance(section, list):
            for sec in section:
                articles = self.parse_section(sec, collection=articles,
                                         title=title, subtitle=subtitle, act_name=act_name)
            return articles
        section_title = section.get("TITLE") if title is None else f"{title}_{section.get('TITLE')}"
        section_subtitle = section.get("SUBTITLE") if title is None else f"{subtitle}_{section.get('SUBTITLE')}"

        # Only iterate if it contains a list of articles
        if "ARTICLE" in section:
            for art in section.get("ARTICLE",[]):
                article_entry = self.parse_article(art, section_title=section_title, section_subtitle=section_subtitle, act_name=act_name)
                articles.append(article_entry)
            return articles
        return self.parse_section(section["SECTION"], collection=articles,
                                  title=section_title, subtitle=section_subtitle, act_name=act_name)


    def parse_articles(self, json_data, act_name=''):
        articles = []
        sections = json_data['DOCUMENT']['SECTION']
        for i, section in enumerate(sections):
            if isinstance(section, list):
                for sec in section:
                    section_data = self.parse_section(sec,act_name=act_name)
                    articles.extend(section_data)
            else:
                section_data = self.parse_section(section,act_name=act_name)
                articles.extend(section_data)
        return pd.DataFrame(articles)

    def Read_in_Json_from_text(self, Orig_dir ='', all_jason_list_path =''):
        with open(all_jason_list_path) as fp: 
            Lines = fp.readlines() 
            files =[]
            acts =[]
            for line in Lines: 
                with open(os.path.join(Orig_dir, line.strip()), "r") as f:
                    data_1 = json.load(f)
                files.append(data_1)
                acts.append(re.sub('\.json$', '', line.strip()))
        return files, acts
    
    def merge_replicated_entries(self, df, matching_coloumn_name='', merging_coloumn_name=''):
        K = df[df.duplicated(subset=[matching_coloumn_name], keep=False)]
        df = df[~df.duplicated(subset=[matching_coloumn_name], keep=False)]

        q = K.article_title.unique()
        j =1
        for i in q:
            P =K.loc[K[matching_coloumn_name] == q[0]].reset_index(drop=True)
            temp_df = P.iloc[[0]]
            for j in range(1, P.shape[0]):
                temp_df.loc[0, merging_coloumn_name] = temp_df.loc[0, merging_coloumn_name]+ P.loc[j, merging_coloumn_name]
            df = df.append(temp_df, ignore_index=True, sort=False)

        return df
#%% preprocess text
# Goes to Util:
class Text_cleaning(object):
    def __init__(self, additional_stopwords_FullPath):
        stop_words_nltk = set(stopwords.words('english'))
        stopwords_final = set(w.rstrip() for w in open(additional_stopwords_FullPath))#'/home/brain/AiTalents/Notebooks/stopwords.txt'))
        stopwords_final = set([s.replace('\n', '') for s in stopwords_final])
        stopwords_final = stopwords_final.union(stop_words_nltk)
        self.stopwords_final = stopwords_final
    def clean_html(self, raw_text):
        cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        cleantext = re.sub(cleanr, '', raw_text)
        return cleantext

    def remove_URL(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'',text)

    
    def my_tokenizer(self, s):
        s = s.lower() # downcase
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
        tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
        tokens = [t for t in tokens if t not in string.punctuation] # remove punctuation        
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove any digits, i.e. "3rd edition"
        return tokens

    def preprocess_text(self, text):
        temp_text = self.remove_URL(text)
        no_html = self.clean_html(temp_text)
        return self.my_tokenizer(no_html)
    def Word_map_generation(self, df):
        #put clean Text into an array:
        contents = df["article_content"].values
        word_index_map = {}
        current_index = 0
        all_tokens = []
        all_content = []
        index_word_map = []
        error_count = 0
        for content in contents:
            try:
                content = content.encode('ascii', 'ignore').decode('utf-8') # this will throw exception if bad characters
                all_content.append(content)
                tokens = self.preprocess_text(content)
                all_tokens.append(tokens)
                for token in tokens:
                    if token not in word_index_map:
                        word_index_map[token] = current_index
                        current_index += 1
                        index_word_map.append(token)
            except Exception as e:
                print(e)
                print(title)
                error_count += 1
        return word_index_map, index_word_map,all_tokens, contents, error_count, self.stopwords_final


    def sklearn_Countvectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
                
        tf_vectorizer = CountVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                        token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                        tokenizer= self.preprocess_text)

        dtm_tf = tf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return tf_vectorizer, dtm_tf ## This is already a sparse feature matrix that could be fed into lda or SVD.
    def sklearn_TfidfVectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
        
        Tfidf_vectorizer = TfidfVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                           token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                           tokenizer= self.preprocess_text)

        dtm_Tfidf = Tfidf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return Tfidf_vectorizer, dtm_Tfidf ## This is already a sparse feature matrix that could be fed into lda or SVD.
            
    
    
#%% Feature generation:
#Goes to Util
class Feature_design(object):
    def __init__(self, word_index_map, all_tokens):
        self.word_index_map = word_index_map
        self.all_tokens = all_tokens
    # now let's create our input matrices - just indicator variables for this example - works better than proportions
    def tokens_to_vector(self, tokens):
        x = np.zeros(len(self.word_index_map))
        for t in tokens:
            i = self.word_index_map[t]
            x[i] = 1
        return x
    
    def Feature_matrix_generation(self):
        N = len(self.all_tokens)
        D = len(self.word_index_map)
        X = np.zeros((D, N)) # terms will go along rows, documents along columns
        i = 0
        for tokens in all_tokens:
            X[:,i] = self.tokens_to_vector(tokens)
            i += 1
        return X
    def Genism_Feature_matrix_generation(self, saving_dir):
        #analysis_dir= '/home/brain/AiTalents/Notebooks/Genism/'
        try:
            os.makedirs(saving_dir)
        except OSError:
            if not os.path.isdir(saving_dir):
                raise

        #os.chdir(analysis_dir)
        dictionary = corpora.Dictionary(self.all_tokens)
        corpus = [dictionary.doc2bow(text) for text in self.all_tokens]
        pickle.dump(corpus, open(os.path.join(saving_dir,'corpus.pkl'), 'wb'))
        dictionary.save(os.path.join(saving_dir,'dictionary.gensim'))
        return corpus, dictionary

#%% Topic models generation:
#Goes to Util
class Topic_modeling(object):
    def __init__(self, num_topics, saving_dir):
        
        
        self.num_topics = num_topics
        self.saving_dir = saving_dir
        try:
            os.makedirs(self.saving_dir)
        except OSError:
            if not os.path.isdir(self.saving_dir):
                raise
    def Genism_LDA(self, corpus, dictionary):
        #saving_dir= '/home/brain/AiTalents/Notebooks/Genism/'
            
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = self.num_topics, id2word=dictionary, passes=15)
        ldamodel.save(os.path.join(self.saving_dir,'LDA_model' + str(self.num_topics)+ '.gensim'))
        return ldamodel
    def Genism_LSI(self, corpus, dictionary):
        lsimodel = gensim.models.lsimodel.LsiModel(corpus, num_topics = self.num_topics, id2word=dictionary)
        lsimodel.save(os.path.join(self.saving_dir,'LSI_model' + str(self.num_topics)+ '.gensim'))
        return lsimodel
    def Sklearn_LDA(self, Feature_X):
        ldamodel = LatentDirichletAllocation()
        ldamodel.fit(Feature_X)
        Z = ldamodel.transform(Feature_X) # size: #term X #Component    >> For word Cloud
        # How about the graph connections?   componennts_ : size: #comps X #Articles  >> each column gives weights.
        U = ldamodel.components_    
        return ldamodel, Z, U 

            
#topics = ldamodel.print_topics(num_words=5)
#for topic in topics:
#    print(topic)

#%% Figures:

#Goes to Util
class Genism_plotting(object):
    def __init__(self,Figure_saving_dir):
        self.Figure_saving_dir = Figure_saving_dir
        try:
            os.makedirs(self.Figure_saving_dir)
        except OSError:
            if not os.path.isdir(self.Figure_saving_dir):
                raise
    def Convert(self, tup, di): 
        di = dict(tup) 
        return di

    def Reverse(self, tuples): 
        new_tup = () 
        for k in reversed(tuples): 
            new_tup = new_tup + (k,) 
        return new_tup 

    def pyLDA_Dashboard(self,ldamodel, corpus, dictionary,html_name, sort_topics=False):
        
        lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=sort_topics)
        pyLDAvis.save_html(lda_display, os.path.join(self.Figure_saving_dir,html_name + '_Dashboard.html'))
        #pyLDAvis.display(lda_display)
        return lda_display
    def Genism_word_cloud(self,html_name, ldamodel):
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

        cloud = WordCloud(background_color='#D8D8D8',
                          width=2500,
                          height=1800,
                          max_words=10,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)
        
        topic_word_box = ldamodel.show_topics(num_topics= ldamodel.num_topics, num_words= np.int(ldamodel.num_terms/100), formatted=False)
        # For each topic:
        topic_word_box_R= []
        for i in np.arange(ldamodel.num_topics):
            temp=[]
            temp = list(self.Reverse(topic_word_box[i][1]))
            topic_word_box_R.append(temp)
        
        D = ldamodel.num_topics
        CL = round(math.sqrt(D))
        RW = CL
        while RW * CL < D:
            RW += 1
        fig, axes = plt.subplots(RW, CL, figsize=(10,10), sharex=True, sharey=True)
        plt.style.context('grayscale')
        fig.patch.set_facecolor('#D8D8D8')

        Topic_names = ['Reporting', 'Investment', 'Risk_Management','Accounting', 'Payments_and_Settlement']
        #['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']
        #
        for i, ax in enumerate(axes.flatten()):
            if i >= D:
                axes.flatten()[i].axis('off')
            else:
            
                fig.add_subplot(ax)
                #topic_words = dict(topics[i][1])
                topic_words={}
                print(i)
                topic_words = self.Convert(topic_word_box_R[i], topic_words)
                cloud.generate_from_frequencies(topic_words, max_font_size=300)
                plt.gca().imshow(cloud)
                
                plt.gca().set_title(Topic_names[i] , fontdict=dict(size=16))
                #Topic_names.append('Topic_' + str(i +1))
                plt.gca().axis('off')


        plt.subplots_adjust(wspace=0, hspace=0)
        
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        
        #fig.patch.set_alpha(0.7)
        #plt.gca().set_facecolor('tab:gray')
        #cloud.to_file(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'))
        fig.savefig(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'), transparent=True)
        
        return 
                
class Genism_Graph(object):
    def __init__(self, Figure_saving_dir='', verbose = True):
        
        self.Figure_saving_dir = Figure_saving_dir
        self.verbose = verbose
        
    def Genism_networkX_graph(self, df, ldamodel):
        Article_names = df['article_title'].values
        
        Topic_names = ['Topic_'+ str(i) for i in range(1, len(ldamodel.show_topics())+1)]
        nx_graph = nx.Graph()
        Nodes = Topic_names + list(Article_names)
        # Add node for each Topic
        for T in Nodes:
            nx_graph.add_node(T)

        i = 0   
        # add edges
        for a in range(0,len(Article_names)):
            temp = ldamodel.get_document_topics(corpus[a])
            for c in range(0, len(temp)):
                if self.verbose:
                    
                    print('C: ', c)
                    print(len(ldamodel.get_document_topics(corpus[a])))

                T = Topic_names[temp[c][0]]
                if self.verbose:
                    print('T, A: ',T, Article_names[a])
                i += 1
                if self.verbose:
                    print('i:', i)
                # Only add edge if the count is positive

                nx_graph.add_edge(Article_names[a],T, weight = temp[c][1])
                if self.verbose:
                    print('Done_edge')
        return nx_graph
    def plot_Genism_networkX_graph(self,df, ldamodel, to_bundel=True ):
        nx_graph = self.Genism_networkX_graph(df, ldamodel)
        #Create the graph using networkX data:
        act_names = []
        kwargs = dict(width=800, height=800, xaxis=None, yaxis=None)
        opts.defaults(opts.Nodes(**kwargs), opts.Graph(**kwargs))
        
        #colors = ['#000000']+hv.Cycle('Category20').values
        topic_graph = hv.Graph.from_networkx(nx_graph, nx.layout.fruchterman_reingold_layout, k=1)
        topic_graph.nodes.data['act'] = topic_graph.nodes.data['index']
        for i in np.arange(topic_graph.nodes.data.shape[0]):
            if topic_graph.nodes.data.loc[i, 'act'] not in ['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']:
                topic_graph.nodes.data.loc[i, 'act'] = str(topic_graph.nodes.data.loc[i, 'act'].split('_')[1])
                
        
        tmp2= topic_graph.nodes.data[~topic_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        #colors = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*np.arange(len(tmp)), 20+2*np.arange(len(tmp)))]
        
        if to_bundel:

            bundled = bundle_graph(topic_graph)
            #.opts(opts.Nodes(color='act', size=10, width=1000, cmap=colors, legend_position='right'))
#opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            #
            return bundled, act_names, article_names
        else:
            return topic_graph, act_names, article_names
        #.opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            
    
    def subselect_Genism_single_Topic_graph(self, Topic_id, topic_graph):
        
        Sub_graph = topic_graph.select(start=Topic_id).opts(node_fill_color='blue', edge_line_width=hv.dim('weight'), edge_line_color='red')
        tmp2= Sub_graph.nodes.data[~Sub_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        
        return Sub_graph, act_names, article_names
    def subselect_Genism_single_Article_graph(self, Article_title, topic_graph, circular_layout=True):
        
        D = topic_graph.select(end=Article_title).opts(node_fill_color='white', edge_line_width=-1*np.log10(hv.dim('weight')))
        
        if circular_layout:
            
            F = layout_nodes(D)
            labels = hv.Labels(F.nodes, ['x', 'y'], 'index')

            return (F * labels.opts(text_font_size='10pt', text_color='black', bgcolor='white'))
        else: 
            
            return D#Goes to Util
class Text_parsing(object):
    def __init__(self):
        
        return 

    def parse_article(self, article, section_title, section_subtitle, act_name):
        article_title = article.get("TITLE")
        article_subtitle = article.get("SUBTITLE")
        article_content = article.get("CONTENT")
        assert article_title is not None
        assert article_subtitle is not None
        assert article_content is not None
        article = {"act_name":act_name,
                   "section_title": section_title,
                   "section_subtitle": section_subtitle,
                   "article_title": article_title + '_' + act_name,# + article_subtitle,
                   "article_subtitle":article_subtitle,
                   "article_content": article_content,
                  }
        return article

    def parse_section(self, section, collection=None, title=None, subtitle=None, act_name = ''):
        articles = [] if collection is None else collection
        if isinstance(section, list):
            for sec in section:
                articles = self.parse_section(sec, collection=articles,
                                         title=title, subtitle=subtitle, act_name=act_name)
            return articles
        section_title = section.get("TITLE") if title is None else f"{title}_{section.get('TITLE')}"
        section_subtitle = section.get("SUBTITLE") if title is None else f"{subtitle}_{section.get('SUBTITLE')}"

        # Only iterate if it contains a list of articles
        if "ARTICLE" in section:
            for art in section.get("ARTICLE",[]):
                article_entry = self.parse_article(art, section_title=section_title, section_subtitle=section_subtitle, act_name=act_name)
                articles.append(article_entry)
            return articles
        return self.parse_section(section["SECTION"], collection=articles,
                                  title=section_title, subtitle=section_subtitle, act_name=act_name)


    def parse_articles(self, json_data, act_name=''):
        articles = []
        sections = json_data['DOCUMENT']['SECTION']
        for i, section in enumerate(sections):
            if isinstance(section, list):
                for sec in section:
                    section_data = self.parse_section(sec,act_name=act_name)
                    articles.extend(section_data)
            else:
                section_data = self.parse_section(section,act_name=act_name)
                articles.extend(section_data)
        return pd.DataFrame(articles)

    def Read_in_Json_from_text(self, Orig_dir ='', all_jason_list_path =''):
        with open(all_jason_list_path) as fp: 
            Lines = fp.readlines() 
            files =[]
            acts =[]
            for line in Lines: 
                with open(os.path.join(Orig_dir, line.strip()), "r") as f:
                    data_1 = json.load(f)
                files.append(data_1)
                acts.append(re.sub('\.json$', '', line.strip()))
        return files, acts
    
    def merge_replicated_entries(self, df, matching_coloumn_name='', merging_coloumn_name=''):
        K = df[df.duplicated(subset=[matching_coloumn_name], keep=False)]
        df = df[~df.duplicated(subset=[matching_coloumn_name], keep=False)]

        q = K.article_title.unique()
        j =1
        for i in q:
            P =K.loc[K[matching_coloumn_name] == q[0]].reset_index(drop=True)
            temp_df = P.iloc[[0]]
            for j in range(1, P.shape[0]):
                temp_df.loc[0, merging_coloumn_name] = temp_df.loc[0, merging_coloumn_name]+ P.loc[j, merging_coloumn_name]
            df = df.append(temp_df, ignore_index=True, sort=False)

        return df
#%% preprocess text
# Goes to Util:
class Text_cleaning(object):
    def __init__(self, additional_stopwords_FullPath):
        stop_words_nltk = set(stopwords.words('english'))
        stopwords_final = set(w.rstrip() for w in open(additional_stopwords_FullPath))#'/home/brain/AiTalents/Notebooks/stopwords.txt'))
        stopwords_final = set([s.replace('\n', '') for s in stopwords_final])
        stopwords_final = stopwords_final.union(stop_words_nltk)
        self.stopwords_final = stopwords_final
    def clean_html(self, raw_text):
        cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        cleantext = re.sub(cleanr, '', raw_text)
        return cleantext

    def remove_URL(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'',text)

    
    def my_tokenizer(self, s):
        s = s.lower() # downcase
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
        tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
        tokens = [t for t in tokens if t not in string.punctuation] # remove punctuation        
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove any digits, i.e. "3rd edition"
        return tokens

    def preprocess_text(self, text):
        temp_text = self.remove_URL(text)
        no_html = self.clean_html(temp_text)
        return self.my_tokenizer(no_html)
    def Word_map_generation(self, df):
        #put clean Text into an array:
        contents = df["article_content"].values
        word_index_map = {}
        current_index = 0
        all_tokens = []
        all_content = []
        index_word_map = []
        error_count = 0
        for content in contents:
            try:
                content = content.encode('ascii', 'ignore').decode('utf-8') # this will throw exception if bad characters
                all_content.append(content)
                tokens = self.preprocess_text(content)
                all_tokens.append(tokens)
                for token in tokens:
                    if token not in word_index_map:
                        word_index_map[token] = current_index
                        current_index += 1
                        index_word_map.append(token)
            except Exception as e:
                print(e)
                print(title)
                error_count += 1
        return word_index_map, index_word_map,all_tokens, contents, error_count, self.stopwords_final


    def sklearn_Countvectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
                
        tf_vectorizer = CountVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                        token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                        tokenizer= self.preprocess_text)

        dtm_tf = tf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return tf_vectorizer, dtm_tf ## This is already a sparse feature matrix that could be fed into lda or SVD.
    def sklearn_TfidfVectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
        
        Tfidf_vectorizer = TfidfVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                           token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                           tokenizer= self.preprocess_text)

        dtm_Tfidf = Tfidf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return Tfidf_vectorizer, dtm_Tfidf ## This is already a sparse feature matrix that could be fed into lda or SVD.
            
    
    
#%% Feature generation:
#Goes to Util
class Feature_design(object):
    def __init__(self, word_index_map, all_tokens):
        self.word_index_map = word_index_map
        self.all_tokens = all_tokens
    # now let's create our input matrices - just indicator variables for this example - works better than proportions
    def tokens_to_vector(self, tokens):
        x = np.zeros(len(self.word_index_map))
        for t in tokens:
            i = self.word_index_map[t]
            x[i] = 1
        return x
    
    def Feature_matrix_generation(self):
        N = len(self.all_tokens)
        D = len(self.word_index_map)
        X = np.zeros((D, N)) # terms will go along rows, documents along columns
        i = 0
        for tokens in all_tokens:
            X[:,i] = self.tokens_to_vector(tokens)
            i += 1
        return X
    def Genism_Feature_matrix_generation(self, saving_dir):
        #analysis_dir= '/home/brain/AiTalents/Notebooks/Genism/'
        try:
            os.makedirs(saving_dir)
        except OSError:
            if not os.path.isdir(saving_dir):
                raise

        #os.chdir(analysis_dir)
        dictionary = corpora.Dictionary(self.all_tokens)
        corpus = [dictionary.doc2bow(text) for text in self.all_tokens]
        pickle.dump(corpus, open(os.path.join(saving_dir,'corpus.pkl'), 'wb'))
        dictionary.save(os.path.join(saving_dir,'dictionary.gensim'))
        return corpus, dictionary

#%% Topic models generation:
#Goes to Util
class Topic_modeling(object):
    def __init__(self, num_topics, saving_dir):
        
        
        self.num_topics = num_topics
        self.saving_dir = saving_dir
        try:
            os.makedirs(self.saving_dir)
        except OSError:
            if not os.path.isdir(self.saving_dir):
                raise
    def Genism_LDA(self, corpus, dictionary):
        #saving_dir= '/home/brain/AiTalents/Notebooks/Genism/'
            
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = self.num_topics, id2word=dictionary, passes=15)
        ldamodel.save(os.path.join(self.saving_dir,'LDA_model' + str(self.num_topics)+ '.gensim'))
        return ldamodel
    def Genism_LSI(self, corpus, dictionary):
        lsimodel = gensim.models.lsimodel.LsiModel(corpus, num_topics = self.num_topics, id2word=dictionary)
        lsimodel.save(os.path.join(self.saving_dir,'LSI_model' + str(self.num_topics)+ '.gensim'))
        return lsimodel
    def Sklearn_LDA(self, Feature_X):
        ldamodel = LatentDirichletAllocation()
        ldamodel.fit(Feature_X)
        Z = ldamodel.transform(Feature_X) # size: #term X #Component    >> For word Cloud
        # How about the graph connections?   componennts_ : size: #comps X #Articles  >> each column gives weights.
        U = ldamodel.components_    
        return ldamodel, Z, U 

            
#topics = ldamodel.print_topics(num_words=5)
#for topic in topics:
#    print(topic)

#%% Figures:

#Goes to Util
class Genism_plotting(object):
    def __init__(self,Figure_saving_dir):
        self.Figure_saving_dir = Figure_saving_dir
        try:
            os.makedirs(self.Figure_saving_dir)
        except OSError:
            if not os.path.isdir(self.Figure_saving_dir):
                raise
    def Convert(self, tup, di): 
        di = dict(tup) 
        return di

    def Reverse(self, tuples): 
        new_tup = () 
        for k in reversed(tuples): 
            new_tup = new_tup + (k,) 
        return new_tup 

    def pyLDA_Dashboard(self,ldamodel, corpus, dictionary,html_name, sort_topics=False):
        
        lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=sort_topics)
        pyLDAvis.save_html(lda_display, os.path.join(self.Figure_saving_dir,html_name + '_Dashboard.html'))
        #pyLDAvis.display(lda_display)
        return lda_display
    def Genism_word_cloud(self,html_name, ldamodel):
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

        cloud = WordCloud(background_color='#D8D8D8',
                          width=2500,
                          height=1800,
                          max_words=10,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)
        
        topic_word_box = ldamodel.show_topics(num_topics= ldamodel.num_topics, num_words= np.int(ldamodel.num_terms/100), formatted=False)
        # For each topic:
        topic_word_box_R= []
        for i in np.arange(ldamodel.num_topics):
            temp=[]
            temp = list(self.Reverse(topic_word_box[i][1]))
            topic_word_box_R.append(temp)
        
        D = ldamodel.num_topics
        CL = round(math.sqrt(D))
        RW = CL
        while RW * CL < D:
            RW += 1
        fig, axes = plt.subplots(RW, CL, figsize=(10,10), sharex=True, sharey=True)
        plt.style.context('grayscale')
        fig.patch.set_facecolor('#D8D8D8')

        Topic_names = ['Reporting', 'Investment', 'Risk_Management','Accounting', 'Payments_and_Settlement']
        #['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']
        #
        for i, ax in enumerate(axes.flatten()):
            if i >= D:
                axes.flatten()[i].axis('off')
            else:
            
                fig.add_subplot(ax)
                #topic_words = dict(topics[i][1])
                topic_words={}
                print(i)
                topic_words = self.Convert(topic_word_box_R[i], topic_words)
                cloud.generate_from_frequencies(topic_words, max_font_size=300)
                plt.gca().imshow(cloud)
                
                plt.gca().set_title(Topic_names[i] , fontdict=dict(size=16))
                #Topic_names.append('Topic_' + str(i +1))
                plt.gca().axis('off')


        plt.subplots_adjust(wspace=0, hspace=0)
        
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        
        #fig.patch.set_alpha(0.7)
        #plt.gca().set_facecolor('tab:gray')
        #cloud.to_file(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'))
        fig.savefig(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'), transparent=True)
        
        return 
                
class Genism_Graph(object):
    def __init__(self, Figure_saving_dir='', verbose = True):
        
        self.Figure_saving_dir = Figure_saving_dir
        self.verbose = verbose
        
    def Genism_networkX_graph(self, df, ldamodel):
        Article_names = df['article_title'].values
        
        Topic_names = ['Topic_'+ str(i) for i in range(1, len(ldamodel.show_topics())+1)]
        nx_graph = nx.Graph()
        Nodes = Topic_names + list(Article_names)
        # Add node for each Topic
        for T in Nodes:
            nx_graph.add_node(T)

        i = 0   
        # add edges
        for a in range(0,len(Article_names)):
            temp = ldamodel.get_document_topics(corpus[a])
            for c in range(0, len(temp)):
                if self.verbose:
                    
                    print('C: ', c)
                    print(len(ldamodel.get_document_topics(corpus[a])))

                T = Topic_names[temp[c][0]]
                if self.verbose:
                    print('T, A: ',T, Article_names[a])
                i += 1
                if self.verbose:
                    print('i:', i)
                # Only add edge if the count is positive

                nx_graph.add_edge(Article_names[a],T, weight = temp[c][1])
                if self.verbose:
                    print('Done_edge')
        return nx_graph
    def plot_Genism_networkX_graph(self,df, ldamodel, to_bundel=True ):
        nx_graph = self.Genism_networkX_graph(df, ldamodel)
        #Create the graph using networkX data:
        act_names = []
        kwargs = dict(width=800, height=800, xaxis=None, yaxis=None)
        opts.defaults(opts.Nodes(**kwargs), opts.Graph(**kwargs))
        
        #colors = ['#000000']+hv.Cycle('Category20').values
        topic_graph = hv.Graph.from_networkx(nx_graph, nx.layout.fruchterman_reingold_layout, k=1)
        topic_graph.nodes.data['act'] = topic_graph.nodes.data['index']
        for i in np.arange(topic_graph.nodes.data.shape[0]):
            if topic_graph.nodes.data.loc[i, 'act'] not in ['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']:
                topic_graph.nodes.data.loc[i, 'act'] = str(topic_graph.nodes.data.loc[i, 'act'].split('_')[1])
                
        
        tmp2= topic_graph.nodes.data[~topic_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        #colors = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*np.arange(len(tmp)), 20+2*np.arange(len(tmp)))]
        
        if to_bundel:

            bundled = bundle_graph(topic_graph)
            #.opts(opts.Nodes(color='act', size=10, width=1000, cmap=colors, legend_position='right'))
#opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            #
            return bundled, act_names, article_names
        else:
            return topic_graph, act_names, article_names
        #.opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            
    
    def subselect_Genism_single_Topic_graph(self, Topic_id, topic_graph):
        
        Sub_graph = topic_graph.select(start=Topic_id).opts(node_fill_color='blue', edge_line_width=hv.dim('weight'), edge_line_color='red')
        tmp2= Sub_graph.nodes.data[~Sub_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        
        return Sub_graph, act_names, article_names
    def subselect_Genism_single_Article_graph(self, Article_title, topic_graph, circular_layout=True):
        
        D = topic_graph.select(end=Article_title).opts(node_fill_color='white', edge_line_width=-1*np.log10(hv.dim('weight')))
        
        if circular_layout:
            
            F = layout_nodes(D)
            labels = hv.Labels(F.nodes, ['x', 'y'], 'index')

            return (F * labels.opts(text_font_size='10pt', text_color='black', bgcolor='white'))
        else: 
            
            return D#Goes to Util
class Text_parsing(object):
    def __init__(self):
        
        return 

    def parse_article(self, article, section_title, section_subtitle, act_name):
        article_title = article.get("TITLE")
        article_subtitle = article.get("SUBTITLE")
        article_content = article.get("CONTENT")
        assert article_title is not None
        assert article_subtitle is not None
        assert article_content is not None
        article = {"act_name":act_name,
                   "section_title": section_title,
                   "section_subtitle": section_subtitle,
                   "article_title": article_title + '_' + act_name,# + article_subtitle,
                   "article_subtitle":article_subtitle,
                   "article_content": article_content,
                  }
        return article

    def parse_section(self, section, collection=None, title=None, subtitle=None, act_name = ''):
        articles = [] if collection is None else collection
        if isinstance(section, list):
            for sec in section:
                articles = self.parse_section(sec, collection=articles,
                                         title=title, subtitle=subtitle, act_name=act_name)
            return articles
        section_title = section.get("TITLE") if title is None else f"{title}_{section.get('TITLE')}"
        section_subtitle = section.get("SUBTITLE") if title is None else f"{subtitle}_{section.get('SUBTITLE')}"

        # Only iterate if it contains a list of articles
        if "ARTICLE" in section:
            for art in section.get("ARTICLE",[]):
                article_entry = self.parse_article(art, section_title=section_title, section_subtitle=section_subtitle, act_name=act_name)
                articles.append(article_entry)
            return articles
        return self.parse_section(section["SECTION"], collection=articles,
                                  title=section_title, subtitle=section_subtitle, act_name=act_name)


    def parse_articles(self, json_data, act_name=''):
        articles = []
        sections = json_data['DOCUMENT']['SECTION']
        for i, section in enumerate(sections):
            if isinstance(section, list):
                for sec in section:
                    section_data = self.parse_section(sec,act_name=act_name)
                    articles.extend(section_data)
            else:
                section_data = self.parse_section(section,act_name=act_name)
                articles.extend(section_data)
        return pd.DataFrame(articles)

    def Read_in_Json_from_text(self, Orig_dir ='', all_jason_list_path =''):
        with open(all_jason_list_path) as fp: 
            Lines = fp.readlines() 
            files =[]
            acts =[]
            for line in Lines: 
                with open(os.path.join(Orig_dir, line.strip()), "r") as f:
                    data_1 = json.load(f)
                files.append(data_1)
                acts.append(re.sub('\.json$', '', line.strip()))
        return files, acts
    
    def merge_replicated_entries(self, df, matching_coloumn_name='', merging_coloumn_name=''):
        K = df[df.duplicated(subset=[matching_coloumn_name], keep=False)]
        df = df[~df.duplicated(subset=[matching_coloumn_name], keep=False)]

        q = K.article_title.unique()
        j =1
        for i in q:
            P =K.loc[K[matching_coloumn_name] == q[0]].reset_index(drop=True)
            temp_df = P.iloc[[0]]
            for j in range(1, P.shape[0]):
                temp_df.loc[0, merging_coloumn_name] = temp_df.loc[0, merging_coloumn_name]+ P.loc[j, merging_coloumn_name]
            df = df.append(temp_df, ignore_index=True, sort=False)

        return df
#%% preprocess text
# Goes to Util:
class Text_cleaning(object):
    def __init__(self, additional_stopwords_FullPath):
        stop_words_nltk = set(stopwords.words('english'))
        stopwords_final = set(w.rstrip() for w in open(additional_stopwords_FullPath))#'/home/brain/AiTalents/Notebooks/stopwords.txt'))
        stopwords_final = set([s.replace('\n', '') for s in stopwords_final])
        stopwords_final = stopwords_final.union(stop_words_nltk)
        self.stopwords_final = stopwords_final
    def clean_html(self, raw_text):
        cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        cleantext = re.sub(cleanr, '', raw_text)
        return cleantext

    def remove_URL(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'',text)

    
    def my_tokenizer(self, s):
        s = s.lower() # downcase
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
        tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
        tokens = [t for t in tokens if t not in string.punctuation] # remove punctuation        
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove any digits, i.e. "3rd edition"
        return tokens

    def preprocess_text(self, text):
        temp_text = self.remove_URL(text)
        no_html = self.clean_html(temp_text)
        return self.my_tokenizer(no_html)
    def Word_map_generation(self, df):
        #put clean Text into an array:
        contents = df["article_content"].values
        word_index_map = {}
        current_index = 0
        all_tokens = []
        all_content = []
        index_word_map = []
        error_count = 0
        for content in contents:
            try:
                content = content.encode('ascii', 'ignore').decode('utf-8') # this will throw exception if bad characters
                all_content.append(content)
                tokens = self.preprocess_text(content)
                all_tokens.append(tokens)
                for token in tokens:
                    if token not in word_index_map:
                        word_index_map[token] = current_index
                        current_index += 1
                        index_word_map.append(token)
            except Exception as e:
                print(e)
                print(title)
                error_count += 1
        return word_index_map, index_word_map,all_tokens, contents, error_count, self.stopwords_final


    def sklearn_Countvectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
                
        tf_vectorizer = CountVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                        token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                        tokenizer= self.preprocess_text)

        dtm_tf = tf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return tf_vectorizer, dtm_tf ## This is already a sparse feature matrix that could be fed into lda or SVD.
    def sklearn_TfidfVectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
        
        Tfidf_vectorizer = TfidfVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                           token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                           tokenizer= self.preprocess_text)

        dtm_Tfidf = Tfidf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return Tfidf_vectorizer, dtm_Tfidf ## This is already a sparse feature matrix that could be fed into lda or SVD.
            
    
    
#%% Feature generation:
#Goes to Util
class Feature_design(object):
    def __init__(self, word_index_map, all_tokens):
        self.word_index_map = word_index_map
        self.all_tokens = all_tokens
    # now let's create our input matrices - just indicator variables for this example - works better than proportions
    def tokens_to_vector(self, tokens):
        x = np.zeros(len(self.word_index_map))
        for t in tokens:
            i = self.word_index_map[t]
            x[i] = 1
        return x
    
    def Feature_matrix_generation(self):
        N = len(self.all_tokens)
        D = len(self.word_index_map)
        X = np.zeros((D, N)) # terms will go along rows, documents along columns
        i = 0
        for tokens in all_tokens:
            X[:,i] = self.tokens_to_vector(tokens)
            i += 1
        return X
    def Genism_Feature_matrix_generation(self, saving_dir):
        #analysis_dir= '/home/brain/AiTalents/Notebooks/Genism/'
        try:
            os.makedirs(saving_dir)
        except OSError:
            if not os.path.isdir(saving_dir):
                raise

        #os.chdir(analysis_dir)
        dictionary = corpora.Dictionary(self.all_tokens)
        corpus = [dictionary.doc2bow(text) for text in self.all_tokens]
        pickle.dump(corpus, open(os.path.join(saving_dir,'corpus.pkl'), 'wb'))
        dictionary.save(os.path.join(saving_dir,'dictionary.gensim'))
        return corpus, dictionary

#%% Topic models generation:
#Goes to Util
class Topic_modeling(object):
    def __init__(self, num_topics, saving_dir):
        
        
        self.num_topics = num_topics
        self.saving_dir = saving_dir
        try:
            os.makedirs(self.saving_dir)
        except OSError:
            if not os.path.isdir(self.saving_dir):
                raise
    def Genism_LDA(self, corpus, dictionary):
        #saving_dir= '/home/brain/AiTalents/Notebooks/Genism/'
            
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = self.num_topics, id2word=dictionary, passes=15)
        ldamodel.save(os.path.join(self.saving_dir,'LDA_model' + str(self.num_topics)+ '.gensim'))
        return ldamodel
    def Genism_LSI(self, corpus, dictionary):
        lsimodel = gensim.models.lsimodel.LsiModel(corpus, num_topics = self.num_topics, id2word=dictionary)
        lsimodel.save(os.path.join(self.saving_dir,'LSI_model' + str(self.num_topics)+ '.gensim'))
        return lsimodel
    def Sklearn_LDA(self, Feature_X):
        ldamodel = LatentDirichletAllocation()
        ldamodel.fit(Feature_X)
        Z = ldamodel.transform(Feature_X) # size: #term X #Component    >> For word Cloud
        # How about the graph connections?   componennts_ : size: #comps X #Articles  >> each column gives weights.
        U = ldamodel.components_    
        return ldamodel, Z, U 

            
#topics = ldamodel.print_topics(num_words=5)
#for topic in topics:
#    print(topic)

#%% Figures:

#Goes to Util
class Genism_plotting(object):
    def __init__(self,Figure_saving_dir):
        self.Figure_saving_dir = Figure_saving_dir
        try:
            os.makedirs(self.Figure_saving_dir)
        except OSError:
            if not os.path.isdir(self.Figure_saving_dir):
                raise
    def Convert(self, tup, di): 
        di = dict(tup) 
        return di

    def Reverse(self, tuples): 
        new_tup = () 
        for k in reversed(tuples): 
            new_tup = new_tup + (k,) 
        return new_tup 

    def pyLDA_Dashboard(self,ldamodel, corpus, dictionary,html_name, sort_topics=False):
        
        lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=sort_topics)
        pyLDAvis.save_html(lda_display, os.path.join(self.Figure_saving_dir,html_name + '_Dashboard.html'))
        #pyLDAvis.display(lda_display)
        return lda_display
    def Genism_word_cloud(self,html_name, ldamodel):
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

        cloud = WordCloud(background_color='#D8D8D8',
                          width=2500,
                          height=1800,
                          max_words=10,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)
        
        topic_word_box = ldamodel.show_topics(num_topics= ldamodel.num_topics, num_words= np.int(ldamodel.num_terms/100), formatted=False)
        # For each topic:
        topic_word_box_R= []
        for i in np.arange(ldamodel.num_topics):
            temp=[]
            temp = list(self.Reverse(topic_word_box[i][1]))
            topic_word_box_R.append(temp)
        
        D = ldamodel.num_topics
        CL = round(math.sqrt(D))
        RW = CL
        while RW * CL < D:
            RW += 1
        fig, axes = plt.subplots(RW, CL, figsize=(10,10), sharex=True, sharey=True)
        plt.style.context('grayscale')
        fig.patch.set_facecolor('#D8D8D8')

        Topic_names = ['Reporting', 'Investment', 'Risk_Management','Accounting', 'Payments_and_Settlement']
        #['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']
        #
        for i, ax in enumerate(axes.flatten()):
            if i >= D:
                axes.flatten()[i].axis('off')
            else:
            
                fig.add_subplot(ax)
                #topic_words = dict(topics[i][1])
                topic_words={}
                print(i)
                topic_words = self.Convert(topic_word_box_R[i], topic_words)
                cloud.generate_from_frequencies(topic_words, max_font_size=300)
                plt.gca().imshow(cloud)
                
                plt.gca().set_title(Topic_names[i] , fontdict=dict(size=16))
                #Topic_names.append('Topic_' + str(i +1))
                plt.gca().axis('off')


        plt.subplots_adjust(wspace=0, hspace=0)
        
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        
        #fig.patch.set_alpha(0.7)
        #plt.gca().set_facecolor('tab:gray')
        #cloud.to_file(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'))
        fig.savefig(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'), transparent=True)
        
        return 
                
class Genism_Graph(object):
    def __init__(self, Figure_saving_dir='', verbose = True):
        
        self.Figure_saving_dir = Figure_saving_dir
        self.verbose = verbose
        
    def Genism_networkX_graph(self, df, ldamodel):
        Article_names = df['article_title'].values
        
        Topic_names = ['Topic_'+ str(i) for i in range(1, len(ldamodel.show_topics())+1)]
        nx_graph = nx.Graph()
        Nodes = Topic_names + list(Article_names)
        # Add node for each Topic
        for T in Nodes:
            nx_graph.add_node(T)

        i = 0   
        # add edges
        for a in range(0,len(Article_names)):
            temp = ldamodel.get_document_topics(corpus[a])
            for c in range(0, len(temp)):
                if self.verbose:
                    
                    print('C: ', c)
                    print(len(ldamodel.get_document_topics(corpus[a])))

                T = Topic_names[temp[c][0]]
                if self.verbose:
                    print('T, A: ',T, Article_names[a])
                i += 1
                if self.verbose:
                    print('i:', i)
                # Only add edge if the count is positive

                nx_graph.add_edge(Article_names[a],T, weight = temp[c][1])
                if self.verbose:
                    print('Done_edge')
        return nx_graph
    def plot_Genism_networkX_graph(self,df, ldamodel, to_bundel=True ):
        nx_graph = self.Genism_networkX_graph(df, ldamodel)
        #Create the graph using networkX data:
        act_names = []
        kwargs = dict(width=800, height=800, xaxis=None, yaxis=None)
        opts.defaults(opts.Nodes(**kwargs), opts.Graph(**kwargs))
        
        #colors = ['#000000']+hv.Cycle('Category20').values
        topic_graph = hv.Graph.from_networkx(nx_graph, nx.layout.fruchterman_reingold_layout, k=1)
        topic_graph.nodes.data['act'] = topic_graph.nodes.data['index']
        for i in np.arange(topic_graph.nodes.data.shape[0]):
            if topic_graph.nodes.data.loc[i, 'act'] not in ['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']:
                topic_graph.nodes.data.loc[i, 'act'] = str(topic_graph.nodes.data.loc[i, 'act'].split('_')[1])
                
        
        tmp2= topic_graph.nodes.data[~topic_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        #colors = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*np.arange(len(tmp)), 20+2*np.arange(len(tmp)))]
        
        if to_bundel:

            bundled = bundle_graph(topic_graph)
            #.opts(opts.Nodes(color='act', size=10, width=1000, cmap=colors, legend_position='right'))
#opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            #
            return bundled, act_names, article_names
        else:
            return topic_graph, act_names, article_names
        #.opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            
    
    def subselect_Genism_single_Topic_graph(self, Topic_id, topic_graph):
        
        Sub_graph = topic_graph.select(start=Topic_id).opts(node_fill_color='blue', edge_line_width=hv.dim('weight'), edge_line_color='red')
        tmp2= Sub_graph.nodes.data[~Sub_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        
        return Sub_graph, act_names, article_names
    def subselect_Genism_single_Article_graph(self, Article_title, topic_graph, circular_layout=True):
        
        D = topic_graph.select(end=Article_title).opts(node_fill_color='white', edge_line_width=-1*np.log10(hv.dim('weight')))
        
        if circular_layout:
            
            F = layout_nodes(D)
            labels = hv.Labels(F.nodes, ['x', 'y'], 'index')

            return (F * labels.opts(text_font_size='10pt', text_color='black', bgcolor='white'))
        else: 
            
            return D#Goes to Util
class Text_parsing(object):
    def __init__(self):
        
        return 

    def parse_article(self, article, section_title, section_subtitle, act_name):
        article_title = article.get("TITLE")
        article_subtitle = article.get("SUBTITLE")
        article_content = article.get("CONTENT")
        assert article_title is not None
        assert article_subtitle is not None
        assert article_content is not None
        article = {"act_name":act_name,
                   "section_title": section_title,
                   "section_subtitle": section_subtitle,
                   "article_title": article_title + '_' + act_name,# + article_subtitle,
                   "article_subtitle":article_subtitle,
                   "article_content": article_content,
                  }
        return article

    def parse_section(self, section, collection=None, title=None, subtitle=None, act_name = ''):
        articles = [] if collection is None else collection
        if isinstance(section, list):
            for sec in section:
                articles = self.parse_section(sec, collection=articles,
                                         title=title, subtitle=subtitle, act_name=act_name)
            return articles
        section_title = section.get("TITLE") if title is None else f"{title}_{section.get('TITLE')}"
        section_subtitle = section.get("SUBTITLE") if title is None else f"{subtitle}_{section.get('SUBTITLE')}"

        # Only iterate if it contains a list of articles
        if "ARTICLE" in section:
            for art in section.get("ARTICLE",[]):
                article_entry = self.parse_article(art, section_title=section_title, section_subtitle=section_subtitle, act_name=act_name)
                articles.append(article_entry)
            return articles
        return self.parse_section(section["SECTION"], collection=articles,
                                  title=section_title, subtitle=section_subtitle, act_name=act_name)


    def parse_articles(self, json_data, act_name=''):
        articles = []
        sections = json_data['DOCUMENT']['SECTION']
        for i, section in enumerate(sections):
            if isinstance(section, list):
                for sec in section:
                    section_data = self.parse_section(sec,act_name=act_name)
                    articles.extend(section_data)
            else:
                section_data = self.parse_section(section,act_name=act_name)
                articles.extend(section_data)
        return pd.DataFrame(articles)

    def Read_in_Json_from_text(self, Orig_dir ='', all_jason_list_path =''):
        with open(all_jason_list_path) as fp: 
            Lines = fp.readlines() 
            files =[]
            acts =[]
            for line in Lines: 
                with open(os.path.join(Orig_dir, line.strip()), "r") as f:
                    data_1 = json.load(f)
                files.append(data_1)
                acts.append(re.sub('\.json$', '', line.strip()))
        return files, acts
    
    def merge_replicated_entries(self, df, matching_coloumn_name='', merging_coloumn_name=''):
        K = df[df.duplicated(subset=[matching_coloumn_name], keep=False)]
        df = df[~df.duplicated(subset=[matching_coloumn_name], keep=False)]

        q = K.article_title.unique()
        j =1
        for i in q:
            P =K.loc[K[matching_coloumn_name] == q[0]].reset_index(drop=True)
            temp_df = P.iloc[[0]]
            for j in range(1, P.shape[0]):
                temp_df.loc[0, merging_coloumn_name] = temp_df.loc[0, merging_coloumn_name]+ P.loc[j, merging_coloumn_name]
            df = df.append(temp_df, ignore_index=True, sort=False)

        return df
#%% preprocess text
# Goes to Util:
class Text_cleaning(object):
    def __init__(self, additional_stopwords_FullPath):
        stop_words_nltk = set(stopwords.words('english'))
        stopwords_final = set(w.rstrip() for w in open(additional_stopwords_FullPath))#'/home/brain/AiTalents/Notebooks/stopwords.txt'))
        stopwords_final = set([s.replace('\n', '') for s in stopwords_final])
        stopwords_final = stopwords_final.union(stop_words_nltk)
        self.stopwords_final = stopwords_final
    def clean_html(self, raw_text):
        cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        cleantext = re.sub(cleanr, '', raw_text)
        return cleantext

    def remove_URL(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'',text)

    
    def my_tokenizer(self, s):
        s = s.lower() # downcase
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
        tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
        tokens = [t for t in tokens if t not in string.punctuation] # remove punctuation        
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove any digits, i.e. "3rd edition"
        return tokens

    def preprocess_text(self, text):
        temp_text = self.remove_URL(text)
        no_html = self.clean_html(temp_text)
        return self.my_tokenizer(no_html)
    def Word_map_generation(self, df):
        #put clean Text into an array:
        contents = df["article_content"].values
        word_index_map = {}
        current_index = 0
        all_tokens = []
        all_content = []
        index_word_map = []
        error_count = 0
        for content in contents:
            try:
                content = content.encode('ascii', 'ignore').decode('utf-8') # this will throw exception if bad characters
                all_content.append(content)
                tokens = self.preprocess_text(content)
                all_tokens.append(tokens)
                for token in tokens:
                    if token not in word_index_map:
                        word_index_map[token] = current_index
                        current_index += 1
                        index_word_map.append(token)
            except Exception as e:
                print(e)
                print(title)
                error_count += 1
        return word_index_map, index_word_map,all_tokens, contents, error_count, self.stopwords_final


    def sklearn_Countvectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
                
        tf_vectorizer = CountVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                        token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                        tokenizer= self.preprocess_text)

        dtm_tf = tf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return tf_vectorizer, dtm_tf ## This is already a sparse feature matrix that could be fed into lda or SVD.
    def sklearn_TfidfVectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
        
        Tfidf_vectorizer = TfidfVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                           token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                           tokenizer= self.preprocess_text)

        dtm_Tfidf = Tfidf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return Tfidf_vectorizer, dtm_Tfidf ## This is already a sparse feature matrix that could be fed into lda or SVD.
            
    
    
#%% Feature generation:
#Goes to Util
class Feature_design(object):
    def __init__(self, word_index_map, all_tokens):
        self.word_index_map = word_index_map
        self.all_tokens = all_tokens
    # now let's create our input matrices - just indicator variables for this example - works better than proportions
    def tokens_to_vector(self, tokens):
        x = np.zeros(len(self.word_index_map))
        for t in tokens:
            i = self.word_index_map[t]
            x[i] = 1
        return x
    
    def Feature_matrix_generation(self):
        N = len(self.all_tokens)
        D = len(self.word_index_map)
        X = np.zeros((D, N)) # terms will go along rows, documents along columns
        i = 0
        for tokens in all_tokens:
            X[:,i] = self.tokens_to_vector(tokens)
            i += 1
        return X
    def Genism_Feature_matrix_generation(self, saving_dir):
        #analysis_dir= '/home/brain/AiTalents/Notebooks/Genism/'
        try:
            os.makedirs(saving_dir)
        except OSError:
            if not os.path.isdir(saving_dir):
                raise

        #os.chdir(analysis_dir)
        dictionary = corpora.Dictionary(self.all_tokens)
        corpus = [dictionary.doc2bow(text) for text in self.all_tokens]
        pickle.dump(corpus, open(os.path.join(saving_dir,'corpus.pkl'), 'wb'))
        dictionary.save(os.path.join(saving_dir,'dictionary.gensim'))
        return corpus, dictionary

#%% Topic models generation:
#Goes to Util
class Topic_modeling(object):
    def __init__(self, num_topics, saving_dir):
        
        
        self.num_topics = num_topics
        self.saving_dir = saving_dir
        try:
            os.makedirs(self.saving_dir)
        except OSError:
            if not os.path.isdir(self.saving_dir):
                raise
    def Genism_LDA(self, corpus, dictionary):
        #saving_dir= '/home/brain/AiTalents/Notebooks/Genism/'
            
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = self.num_topics, id2word=dictionary, passes=15)
        ldamodel.save(os.path.join(self.saving_dir,'LDA_model' + str(self.num_topics)+ '.gensim'))
        return ldamodel
    def Genism_LSI(self, corpus, dictionary):
        lsimodel = gensim.models.lsimodel.LsiModel(corpus, num_topics = self.num_topics, id2word=dictionary)
        lsimodel.save(os.path.join(self.saving_dir,'LSI_model' + str(self.num_topics)+ '.gensim'))
        return lsimodel
    def Sklearn_LDA(self, Feature_X):
        ldamodel = LatentDirichletAllocation()
        ldamodel.fit(Feature_X)
        Z = ldamodel.transform(Feature_X) # size: #term X #Component    >> For word Cloud
        # How about the graph connections?   componennts_ : size: #comps X #Articles  >> each column gives weights.
        U = ldamodel.components_    
        return ldamodel, Z, U 

            
#topics = ldamodel.print_topics(num_words=5)
#for topic in topics:
#    print(topic)

#%% Figures:

#Goes to Util
class Genism_plotting(object):
    def __init__(self,Figure_saving_dir):
        self.Figure_saving_dir = Figure_saving_dir
        try:
            os.makedirs(self.Figure_saving_dir)
        except OSError:
            if not os.path.isdir(self.Figure_saving_dir):
                raise
    def Convert(self, tup, di): 
        di = dict(tup) 
        return di

    def Reverse(self, tuples): 
        new_tup = () 
        for k in reversed(tuples): 
            new_tup = new_tup + (k,) 
        return new_tup 

    def pyLDA_Dashboard(self,ldamodel, corpus, dictionary,html_name, sort_topics=False):
        
        lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=sort_topics)
        pyLDAvis.save_html(lda_display, os.path.join(self.Figure_saving_dir,html_name + '_Dashboard.html'))
        #pyLDAvis.display(lda_display)
        return lda_display
    def Genism_word_cloud(self,html_name, ldamodel):
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

        cloud = WordCloud(background_color='#D8D8D8',
                          width=2500,
                          height=1800,
                          max_words=10,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)
        
        topic_word_box = ldamodel.show_topics(num_topics= ldamodel.num_topics, num_words= np.int(ldamodel.num_terms/100), formatted=False)
        # For each topic:
        topic_word_box_R= []
        for i in np.arange(ldamodel.num_topics):
            temp=[]
            temp = list(self.Reverse(topic_word_box[i][1]))
            topic_word_box_R.append(temp)
        
        D = ldamodel.num_topics
        CL = round(math.sqrt(D))
        RW = CL
        while RW * CL < D:
            RW += 1
        fig, axes = plt.subplots(RW, CL, figsize=(10,10), sharex=True, sharey=True)
        plt.style.context('grayscale')
        fig.patch.set_facecolor('#D8D8D8')

        Topic_names = ['Reporting', 'Investment', 'Risk_Management','Accounting', 'Payments_and_Settlement']
        #['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']
        #
        for i, ax in enumerate(axes.flatten()):
            if i >= D:
                axes.flatten()[i].axis('off')
            else:
            
                fig.add_subplot(ax)
                #topic_words = dict(topics[i][1])
                topic_words={}
                print(i)
                topic_words = self.Convert(topic_word_box_R[i], topic_words)
                cloud.generate_from_frequencies(topic_words, max_font_size=300)
                plt.gca().imshow(cloud)
                
                plt.gca().set_title(Topic_names[i] , fontdict=dict(size=16))
                #Topic_names.append('Topic_' + str(i +1))
                plt.gca().axis('off')


        plt.subplots_adjust(wspace=0, hspace=0)
        
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        
        #fig.patch.set_alpha(0.7)
        #plt.gca().set_facecolor('tab:gray')
        #cloud.to_file(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'))
        fig.savefig(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'), transparent=True)
        
        return 
                
class Genism_Graph(object):
    def __init__(self, Figure_saving_dir='', verbose = True):
        
        self.Figure_saving_dir = Figure_saving_dir
        self.verbose = verbose
        
    def Genism_networkX_graph(self, df, ldamodel):
        Article_names = df['article_title'].values
        
        Topic_names = ['Topic_'+ str(i) for i in range(1, len(ldamodel.show_topics())+1)]
        nx_graph = nx.Graph()
        Nodes = Topic_names + list(Article_names)
        # Add node for each Topic
        for T in Nodes:
            nx_graph.add_node(T)

        i = 0   
        # add edges
        for a in range(0,len(Article_names)):
            temp = ldamodel.get_document_topics(corpus[a])
            for c in range(0, len(temp)):
                if self.verbose:
                    
                    print('C: ', c)
                    print(len(ldamodel.get_document_topics(corpus[a])))

                T = Topic_names[temp[c][0]]
                if self.verbose:
                    print('T, A: ',T, Article_names[a])
                i += 1
                if self.verbose:
                    print('i:', i)
                # Only add edge if the count is positive

                nx_graph.add_edge(Article_names[a],T, weight = temp[c][1])
                if self.verbose:
                    print('Done_edge')
        return nx_graph
    def plot_Genism_networkX_graph(self,df, ldamodel, to_bundel=True ):
        nx_graph = self.Genism_networkX_graph(df, ldamodel)
        #Create the graph using networkX data:
        act_names = []
        kwargs = dict(width=800, height=800, xaxis=None, yaxis=None)
        opts.defaults(opts.Nodes(**kwargs), opts.Graph(**kwargs))
        
        #colors = ['#000000']+hv.Cycle('Category20').values
        topic_graph = hv.Graph.from_networkx(nx_graph, nx.layout.fruchterman_reingold_layout, k=1)
        topic_graph.nodes.data['act'] = topic_graph.nodes.data['index']
        for i in np.arange(topic_graph.nodes.data.shape[0]):
            if topic_graph.nodes.data.loc[i, 'act'] not in ['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']:
                topic_graph.nodes.data.loc[i, 'act'] = str(topic_graph.nodes.data.loc[i, 'act'].split('_')[1])
                
        
        tmp2= topic_graph.nodes.data[~topic_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        #colors = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*np.arange(len(tmp)), 20+2*np.arange(len(tmp)))]
        
        if to_bundel:

            bundled = bundle_graph(topic_graph)
            #.opts(opts.Nodes(color='act', size=10, width=1000, cmap=colors, legend_position='right'))
#opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            #
            return bundled, act_names, article_names
        else:
            return topic_graph, act_names, article_names
        #.opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            
    
    def subselect_Genism_single_Topic_graph(self, Topic_id, topic_graph):
        
        Sub_graph = topic_graph.select(start=Topic_id).opts(node_fill_color='blue', edge_line_width=hv.dim('weight'), edge_line_color='red')
        tmp2= Sub_graph.nodes.data[~Sub_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        
        return Sub_graph, act_names, article_names
    def subselect_Genism_single_Article_graph(self, Article_title, topic_graph, circular_layout=True):
        
        D = topic_graph.select(end=Article_title).opts(node_fill_color='white', edge_line_width=-1*np.log10(hv.dim('weight')))
        
        if circular_layout:
            
            F = layout_nodes(D)
            labels = hv.Labels(F.nodes, ['x', 'y'], 'index')

            return (F * labels.opts(text_font_size='10pt', text_color='black', bgcolor='white'))
        else: 
            
            return D#Goes to Util
class Text_parsing(object):
    def __init__(self):
        
        return 

    def parse_article(self, article, section_title, section_subtitle, act_name):
        article_title = article.get("TITLE")
        article_subtitle = article.get("SUBTITLE")
        article_content = article.get("CONTENT")
        assert article_title is not None
        assert article_subtitle is not None
        assert article_content is not None
        article = {"act_name":act_name,
                   "section_title": section_title,
                   "section_subtitle": section_subtitle,
                   "article_title": article_title + '_' + act_name,# + article_subtitle,
                   "article_subtitle":article_subtitle,
                   "article_content": article_content,
                  }
        return article

    def parse_section(self, section, collection=None, title=None, subtitle=None, act_name = ''):
        articles = [] if collection is None else collection
        if isinstance(section, list):
            for sec in section:
                articles = self.parse_section(sec, collection=articles,
                                         title=title, subtitle=subtitle, act_name=act_name)
            return articles
        section_title = section.get("TITLE") if title is None else f"{title}_{section.get('TITLE')}"
        section_subtitle = section.get("SUBTITLE") if title is None else f"{subtitle}_{section.get('SUBTITLE')}"

        # Only iterate if it contains a list of articles
        if "ARTICLE" in section:
            for art in section.get("ARTICLE",[]):
                article_entry = self.parse_article(art, section_title=section_title, section_subtitle=section_subtitle, act_name=act_name)
                articles.append(article_entry)
            return articles
        return self.parse_section(section["SECTION"], collection=articles,
                                  title=section_title, subtitle=section_subtitle, act_name=act_name)


    def parse_articles(self, json_data, act_name=''):
        articles = []
        sections = json_data['DOCUMENT']['SECTION']
        for i, section in enumerate(sections):
            if isinstance(section, list):
                for sec in section:
                    section_data = self.parse_section(sec,act_name=act_name)
                    articles.extend(section_data)
            else:
                section_data = self.parse_section(section,act_name=act_name)
                articles.extend(section_data)
        return pd.DataFrame(articles)

    def Read_in_Json_from_text(self, Orig_dir ='', all_jason_list_path =''):
        with open(all_jason_list_path) as fp: 
            Lines = fp.readlines() 
            files =[]
            acts =[]
            for line in Lines: 
                with open(os.path.join(Orig_dir, line.strip()), "r") as f:
                    data_1 = json.load(f)
                files.append(data_1)
                acts.append(re.sub('\.json$', '', line.strip()))
        return files, acts
    
    def merge_replicated_entries(self, df, matching_coloumn_name='', merging_coloumn_name=''):
        K = df[df.duplicated(subset=[matching_coloumn_name], keep=False)]
        df = df[~df.duplicated(subset=[matching_coloumn_name], keep=False)]

        q = K.article_title.unique()
        j =1
        for i in q:
            P =K.loc[K[matching_coloumn_name] == q[0]].reset_index(drop=True)
            temp_df = P.iloc[[0]]
            for j in range(1, P.shape[0]):
                temp_df.loc[0, merging_coloumn_name] = temp_df.loc[0, merging_coloumn_name]+ P.loc[j, merging_coloumn_name]
            df = df.append(temp_df, ignore_index=True, sort=False)

        return df
#%% preprocess text
# Goes to Util:
class Text_cleaning(object):
    def __init__(self, additional_stopwords_FullPath):
        stop_words_nltk = set(stopwords.words('english'))
        stopwords_final = set(w.rstrip() for w in open(additional_stopwords_FullPath))#'/home/brain/AiTalents/Notebooks/stopwords.txt'))
        stopwords_final = set([s.replace('\n', '') for s in stopwords_final])
        stopwords_final = stopwords_final.union(stop_words_nltk)
        self.stopwords_final = stopwords_final
    def clean_html(self, raw_text):
        cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        cleantext = re.sub(cleanr, '', raw_text)
        return cleantext

    def remove_URL(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'',text)

    
    def my_tokenizer(self, s):
        s = s.lower() # downcase
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
        tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
        tokens = [t for t in tokens if t not in string.punctuation] # remove punctuation        
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove any digits, i.e. "3rd edition"
        return tokens

    def preprocess_text(self, text):
        temp_text = self.remove_URL(text)
        no_html = self.clean_html(temp_text)
        return self.my_tokenizer(no_html)
    def Word_map_generation(self, df):
        #put clean Text into an array:
        contents = df["article_content"].values
        word_index_map = {}
        current_index = 0
        all_tokens = []
        all_content = []
        index_word_map = []
        error_count = 0
        for content in contents:
            try:
                content = content.encode('ascii', 'ignore').decode('utf-8') # this will throw exception if bad characters
                all_content.append(content)
                tokens = self.preprocess_text(content)
                all_tokens.append(tokens)
                for token in tokens:
                    if token not in word_index_map:
                        word_index_map[token] = current_index
                        current_index += 1
                        index_word_map.append(token)
            except Exception as e:
                print(e)
                print(title)
                error_count += 1
        return word_index_map, index_word_map,all_tokens, contents, error_count, self.stopwords_final


    def sklearn_Countvectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
                
        tf_vectorizer = CountVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                        token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                        tokenizer= self.preprocess_text)

        dtm_tf = tf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return tf_vectorizer, dtm_tf ## This is already a sparse feature matrix that could be fed into lda or SVD.
    def sklearn_TfidfVectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
        
        Tfidf_vectorizer = TfidfVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                           token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                           tokenizer= self.preprocess_text)

        dtm_Tfidf = Tfidf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return Tfidf_vectorizer, dtm_Tfidf ## This is already a sparse feature matrix that could be fed into lda or SVD.
            
    
    
#%% Feature generation:
#Goes to Util
class Feature_design(object):
    def __init__(self, word_index_map, all_tokens):
        self.word_index_map = word_index_map
        self.all_tokens = all_tokens
    # now let's create our input matrices - just indicator variables for this example - works better than proportions
    def tokens_to_vector(self, tokens):
        x = np.zeros(len(self.word_index_map))
        for t in tokens:
            i = self.word_index_map[t]
            x[i] = 1
        return x
    
    def Feature_matrix_generation(self):
        N = len(self.all_tokens)
        D = len(self.word_index_map)
        X = np.zeros((D, N)) # terms will go along rows, documents along columns
        i = 0
        for tokens in all_tokens:
            X[:,i] = self.tokens_to_vector(tokens)
            i += 1
        return X
    def Genism_Feature_matrix_generation(self, saving_dir):
        #analysis_dir= '/home/brain/AiTalents/Notebooks/Genism/'
        try:
            os.makedirs(saving_dir)
        except OSError:
            if not os.path.isdir(saving_dir):
                raise

        #os.chdir(analysis_dir)
        dictionary = corpora.Dictionary(self.all_tokens)
        corpus = [dictionary.doc2bow(text) for text in self.all_tokens]
        pickle.dump(corpus, open(os.path.join(saving_dir,'corpus.pkl'), 'wb'))
        dictionary.save(os.path.join(saving_dir,'dictionary.gensim'))
        return corpus, dictionary

#%% Topic models generation:
#Goes to Util
class Topic_modeling(object):
    def __init__(self, num_topics, saving_dir):
        
        
        self.num_topics = num_topics
        self.saving_dir = saving_dir
        try:
            os.makedirs(self.saving_dir)
        except OSError:
            if not os.path.isdir(self.saving_dir):
                raise
    def Genism_LDA(self, corpus, dictionary):
        #saving_dir= '/home/brain/AiTalents/Notebooks/Genism/'
            
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = self.num_topics, id2word=dictionary, passes=15)
        ldamodel.save(os.path.join(self.saving_dir,'LDA_model' + str(self.num_topics)+ '.gensim'))
        return ldamodel
    def Genism_LSI(self, corpus, dictionary):
        lsimodel = gensim.models.lsimodel.LsiModel(corpus, num_topics = self.num_topics, id2word=dictionary)
        lsimodel.save(os.path.join(self.saving_dir,'LSI_model' + str(self.num_topics)+ '.gensim'))
        return lsimodel
    def Sklearn_LDA(self, Feature_X):
        ldamodel = LatentDirichletAllocation()
        ldamodel.fit(Feature_X)
        Z = ldamodel.transform(Feature_X) # size: #term X #Component    >> For word Cloud
        # How about the graph connections?   componennts_ : size: #comps X #Articles  >> each column gives weights.
        U = ldamodel.components_    
        return ldamodel, Z, U 

            
#topics = ldamodel.print_topics(num_words=5)
#for topic in topics:
#    print(topic)

#%% Figures:

#Goes to Util
class Genism_plotting(object):
    def __init__(self,Figure_saving_dir):
        self.Figure_saving_dir = Figure_saving_dir
        try:
            os.makedirs(self.Figure_saving_dir)
        except OSError:
            if not os.path.isdir(self.Figure_saving_dir):
                raise
    def Convert(self, tup, di): 
        di = dict(tup) 
        return di

    def Reverse(self, tuples): 
        new_tup = () 
        for k in reversed(tuples): 
            new_tup = new_tup + (k,) 
        return new_tup 

    def pyLDA_Dashboard(self,ldamodel, corpus, dictionary,html_name, sort_topics=False):
        
        lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=sort_topics)
        pyLDAvis.save_html(lda_display, os.path.join(self.Figure_saving_dir,html_name + '_Dashboard.html'))
        #pyLDAvis.display(lda_display)
        return lda_display
    def Genism_word_cloud(self,html_name, ldamodel):
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

        cloud = WordCloud(background_color='#D8D8D8',
                          width=2500,
                          height=1800,
                          max_words=10,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)
        
        topic_word_box = ldamodel.show_topics(num_topics= ldamodel.num_topics, num_words= np.int(ldamodel.num_terms/100), formatted=False)
        # For each topic:
        topic_word_box_R= []
        for i in np.arange(ldamodel.num_topics):
            temp=[]
            temp = list(self.Reverse(topic_word_box[i][1]))
            topic_word_box_R.append(temp)
        
        D = ldamodel.num_topics
        CL = round(math.sqrt(D))
        RW = CL
        while RW * CL < D:
            RW += 1
        fig, axes = plt.subplots(RW, CL, figsize=(10,10), sharex=True, sharey=True)
        plt.style.context('grayscale')
        fig.patch.set_facecolor('#D8D8D8')

        Topic_names = ['Reporting', 'Investment', 'Risk_Management','Accounting', 'Payments_and_Settlement']
        #['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']
        #
        for i, ax in enumerate(axes.flatten()):
            if i >= D:
                axes.flatten()[i].axis('off')
            else:
            
                fig.add_subplot(ax)
                #topic_words = dict(topics[i][1])
                topic_words={}
                print(i)
                topic_words = self.Convert(topic_word_box_R[i], topic_words)
                cloud.generate_from_frequencies(topic_words, max_font_size=300)
                plt.gca().imshow(cloud)
                
                plt.gca().set_title(Topic_names[i] , fontdict=dict(size=16))
                #Topic_names.append('Topic_' + str(i +1))
                plt.gca().axis('off')


        plt.subplots_adjust(wspace=0, hspace=0)
        
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        
        #fig.patch.set_alpha(0.7)
        #plt.gca().set_facecolor('tab:gray')
        #cloud.to_file(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'))
        fig.savefig(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'), transparent=True)
        
        return 
                
class Genism_Graph(object):
    def __init__(self, Figure_saving_dir='', verbose = True):
        
        self.Figure_saving_dir = Figure_saving_dir
        self.verbose = verbose
        
    def Genism_networkX_graph(self, df, ldamodel):
        Article_names = df['article_title'].values
        
        Topic_names = ['Topic_'+ str(i) for i in range(1, len(ldamodel.show_topics())+1)]
        nx_graph = nx.Graph()
        Nodes = Topic_names + list(Article_names)
        # Add node for each Topic
        for T in Nodes:
            nx_graph.add_node(T)

        i = 0   
        # add edges
        for a in range(0,len(Article_names)):
            temp = ldamodel.get_document_topics(corpus[a])
            for c in range(0, len(temp)):
                if self.verbose:
                    
                    print('C: ', c)
                    print(len(ldamodel.get_document_topics(corpus[a])))

                T = Topic_names[temp[c][0]]
                if self.verbose:
                    print('T, A: ',T, Article_names[a])
                i += 1
                if self.verbose:
                    print('i:', i)
                # Only add edge if the count is positive

                nx_graph.add_edge(Article_names[a],T, weight = temp[c][1])
                if self.verbose:
                    print('Done_edge')
        return nx_graph
    def plot_Genism_networkX_graph(self,df, ldamodel, to_bundel=True ):
        nx_graph = self.Genism_networkX_graph(df, ldamodel)
        #Create the graph using networkX data:
        act_names = []
        kwargs = dict(width=800, height=800, xaxis=None, yaxis=None)
        opts.defaults(opts.Nodes(**kwargs), opts.Graph(**kwargs))
        
        #colors = ['#000000']+hv.Cycle('Category20').values
        topic_graph = hv.Graph.from_networkx(nx_graph, nx.layout.fruchterman_reingold_layout, k=1)
        topic_graph.nodes.data['act'] = topic_graph.nodes.data['index']
        for i in np.arange(topic_graph.nodes.data.shape[0]):
            if topic_graph.nodes.data.loc[i, 'act'] not in ['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']:
                topic_graph.nodes.data.loc[i, 'act'] = str(topic_graph.nodes.data.loc[i, 'act'].split('_')[1])
                
        
        tmp2= topic_graph.nodes.data[~topic_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        #colors = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*np.arange(len(tmp)), 20+2*np.arange(len(tmp)))]
        
        if to_bundel:

            bundled = bundle_graph(topic_graph)
            #.opts(opts.Nodes(color='act', size=10, width=1000, cmap=colors, legend_position='right'))
#opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            #
            return bundled, act_names, article_names
        else:
            return topic_graph, act_names, article_names
        #.opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            
    
    def subselect_Genism_single_Topic_graph(self, Topic_id, topic_graph):
        
        Sub_graph = topic_graph.select(start=Topic_id).opts(node_fill_color='blue', edge_line_width=hv.dim('weight'), edge_line_color='red')
        tmp2= Sub_graph.nodes.data[~Sub_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        
        return Sub_graph, act_names, article_names
    def subselect_Genism_single_Article_graph(self, Article_title, topic_graph, circular_layout=True):
        
        D = topic_graph.select(end=Article_title).opts(node_fill_color='white', edge_line_width=-1*np.log10(hv.dim('weight')))
        
        if circular_layout:
            
            F = layout_nodes(D)
            labels = hv.Labels(F.nodes, ['x', 'y'], 'index')

            return (F * labels.opts(text_font_size='10pt', text_color='black', bgcolor='white'))
        else: 
            
            return D#Goes to Util
class Text_parsing(object):
    def __init__(self):
        
        return 

    def parse_article(self, article, section_title, section_subtitle, act_name):
        article_title = article.get("TITLE")
        article_subtitle = article.get("SUBTITLE")
        article_content = article.get("CONTENT")
        assert article_title is not None
        assert article_subtitle is not None
        assert article_content is not None
        article = {"act_name":act_name,
                   "section_title": section_title,
                   "section_subtitle": section_subtitle,
                   "article_title": article_title + '_' + act_name,# + article_subtitle,
                   "article_subtitle":article_subtitle,
                   "article_content": article_content,
                  }
        return article

    def parse_section(self, section, collection=None, title=None, subtitle=None, act_name = ''):
        articles = [] if collection is None else collection
        if isinstance(section, list):
            for sec in section:
                articles = self.parse_section(sec, collection=articles,
                                         title=title, subtitle=subtitle, act_name=act_name)
            return articles
        section_title = section.get("TITLE") if title is None else f"{title}_{section.get('TITLE')}"
        section_subtitle = section.get("SUBTITLE") if title is None else f"{subtitle}_{section.get('SUBTITLE')}"

        # Only iterate if it contains a list of articles
        if "ARTICLE" in section:
            for art in section.get("ARTICLE",[]):
                article_entry = self.parse_article(art, section_title=section_title, section_subtitle=section_subtitle, act_name=act_name)
                articles.append(article_entry)
            return articles
        return self.parse_section(section["SECTION"], collection=articles,
                                  title=section_title, subtitle=section_subtitle, act_name=act_name)


    def parse_articles(self, json_data, act_name=''):
        articles = []
        sections = json_data['DOCUMENT']['SECTION']
        for i, section in enumerate(sections):
            if isinstance(section, list):
                for sec in section:
                    section_data = self.parse_section(sec,act_name=act_name)
                    articles.extend(section_data)
            else:
                section_data = self.parse_section(section,act_name=act_name)
                articles.extend(section_data)
        return pd.DataFrame(articles)

    def Read_in_Json_from_text(self, Orig_dir ='', all_jason_list_path =''):
        with open(all_jason_list_path) as fp: 
            Lines = fp.readlines() 
            files =[]
            acts =[]
            for line in Lines: 
                with open(os.path.join(Orig_dir, line.strip()), "r") as f:
                    data_1 = json.load(f)
                files.append(data_1)
                acts.append(re.sub('\.json$', '', line.strip()))
        return files, acts
    
    def merge_replicated_entries(self, df, matching_coloumn_name='', merging_coloumn_name=''):
        K = df[df.duplicated(subset=[matching_coloumn_name], keep=False)]
        df = df[~df.duplicated(subset=[matching_coloumn_name], keep=False)]

        q = K.article_title.unique()
        j =1
        for i in q:
            P =K.loc[K[matching_coloumn_name] == q[0]].reset_index(drop=True)
            temp_df = P.iloc[[0]]
            for j in range(1, P.shape[0]):
                temp_df.loc[0, merging_coloumn_name] = temp_df.loc[0, merging_coloumn_name]+ P.loc[j, merging_coloumn_name]
            df = df.append(temp_df, ignore_index=True, sort=False)

        return df
#%% preprocess text
# Goes to Util:
class Text_cleaning(object):
    def __init__(self, additional_stopwords_FullPath):
        stop_words_nltk = set(stopwords.words('english'))
        stopwords_final = set(w.rstrip() for w in open(additional_stopwords_FullPath))#'/home/brain/AiTalents/Notebooks/stopwords.txt'))
        stopwords_final = set([s.replace('\n', '') for s in stopwords_final])
        stopwords_final = stopwords_final.union(stop_words_nltk)
        self.stopwords_final = stopwords_final
    def clean_html(self, raw_text):
        cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        cleantext = re.sub(cleanr, '', raw_text)
        return cleantext

    def remove_URL(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'',text)

    
    def my_tokenizer(self, s):
        s = s.lower() # downcase
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
        tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
        tokens = [t for t in tokens if t not in string.punctuation] # remove punctuation        
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove any digits, i.e. "3rd edition"
        return tokens

    def preprocess_text(self, text):
        temp_text = self.remove_URL(text)
        no_html = self.clean_html(temp_text)
        return self.my_tokenizer(no_html)
    def Word_map_generation(self, df):
        #put clean Text into an array:
        contents = df["article_content"].values
        word_index_map = {}
        current_index = 0
        all_tokens = []
        all_content = []
        index_word_map = []
        error_count = 0
        for content in contents:
            try:
                content = content.encode('ascii', 'ignore').decode('utf-8') # this will throw exception if bad characters
                all_content.append(content)
                tokens = self.preprocess_text(content)
                all_tokens.append(tokens)
                for token in tokens:
                    if token not in word_index_map:
                        word_index_map[token] = current_index
                        current_index += 1
                        index_word_map.append(token)
            except Exception as e:
                print(e)
                print(title)
                error_count += 1
        return word_index_map, index_word_map,all_tokens, contents, error_count, self.stopwords_final


    def sklearn_Countvectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
                
        tf_vectorizer = CountVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                        token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                        tokenizer= self.preprocess_text)

        dtm_tf = tf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return tf_vectorizer, dtm_tf ## This is already a sparse feature matrix that could be fed into lda or SVD.
    def sklearn_TfidfVectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
        
        Tfidf_vectorizer = TfidfVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                           token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                           tokenizer= self.preprocess_text)

        dtm_Tfidf = Tfidf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return Tfidf_vectorizer, dtm_Tfidf ## This is already a sparse feature matrix that could be fed into lda or SVD.
            
    
    
#%% Feature generation:
#Goes to Util
class Feature_design(object):
    def __init__(self, word_index_map, all_tokens):
        self.word_index_map = word_index_map
        self.all_tokens = all_tokens
    # now let's create our input matrices - just indicator variables for this example - works better than proportions
    def tokens_to_vector(self, tokens):
        x = np.zeros(len(self.word_index_map))
        for t in tokens:
            i = self.word_index_map[t]
            x[i] = 1
        return x
    
    def Feature_matrix_generation(self):
        N = len(self.all_tokens)
        D = len(self.word_index_map)
        X = np.zeros((D, N)) # terms will go along rows, documents along columns
        i = 0
        for tokens in all_tokens:
            X[:,i] = self.tokens_to_vector(tokens)
            i += 1
        return X
    def Genism_Feature_matrix_generation(self, saving_dir):
        #analysis_dir= '/home/brain/AiTalents/Notebooks/Genism/'
        try:
            os.makedirs(saving_dir)
        except OSError:
            if not os.path.isdir(saving_dir):
                raise

        #os.chdir(analysis_dir)
        dictionary = corpora.Dictionary(self.all_tokens)
        corpus = [dictionary.doc2bow(text) for text in self.all_tokens]
        pickle.dump(corpus, open(os.path.join(saving_dir,'corpus.pkl'), 'wb'))
        dictionary.save(os.path.join(saving_dir,'dictionary.gensim'))
        return corpus, dictionary

#%% Topic models generation:
#Goes to Util
class Topic_modeling(object):
    def __init__(self, num_topics, saving_dir):
        
        
        self.num_topics = num_topics
        self.saving_dir = saving_dir
        try:
            os.makedirs(self.saving_dir)
        except OSError:
            if not os.path.isdir(self.saving_dir):
                raise
    def Genism_LDA(self, corpus, dictionary):
        #saving_dir= '/home/brain/AiTalents/Notebooks/Genism/'
            
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = self.num_topics, id2word=dictionary, passes=15)
        ldamodel.save(os.path.join(self.saving_dir,'LDA_model' + str(self.num_topics)+ '.gensim'))
        return ldamodel
    def Genism_LSI(self, corpus, dictionary):
        lsimodel = gensim.models.lsimodel.LsiModel(corpus, num_topics = self.num_topics, id2word=dictionary)
        lsimodel.save(os.path.join(self.saving_dir,'LSI_model' + str(self.num_topics)+ '.gensim'))
        return lsimodel
    def Sklearn_LDA(self, Feature_X):
        ldamodel = LatentDirichletAllocation()
        ldamodel.fit(Feature_X)
        Z = ldamodel.transform(Feature_X) # size: #term X #Component    >> For word Cloud
        # How about the graph connections?   componennts_ : size: #comps X #Articles  >> each column gives weights.
        U = ldamodel.components_    
        return ldamodel, Z, U 

            
#topics = ldamodel.print_topics(num_words=5)
#for topic in topics:
#    print(topic)

#%% Figures:

#Goes to Util
class Genism_plotting(object):
    def __init__(self,Figure_saving_dir):
        self.Figure_saving_dir = Figure_saving_dir
        try:
            os.makedirs(self.Figure_saving_dir)
        except OSError:
            if not os.path.isdir(self.Figure_saving_dir):
                raise
    def Convert(self, tup, di): 
        di = dict(tup) 
        return di

    def Reverse(self, tuples): 
        new_tup = () 
        for k in reversed(tuples): 
            new_tup = new_tup + (k,) 
        return new_tup 

    def pyLDA_Dashboard(self,ldamodel, corpus, dictionary,html_name, sort_topics=False):
        
        lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=sort_topics)
        pyLDAvis.save_html(lda_display, os.path.join(self.Figure_saving_dir,html_name + '_Dashboard.html'))
        #pyLDAvis.display(lda_display)
        return lda_display
    def Genism_word_cloud(self,html_name, ldamodel):
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

        cloud = WordCloud(background_color='#D8D8D8',
                          width=2500,
                          height=1800,
                          max_words=10,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)
        
        topic_word_box = ldamodel.show_topics(num_topics= ldamodel.num_topics, num_words= np.int(ldamodel.num_terms/100), formatted=False)
        # For each topic:
        topic_word_box_R= []
        for i in np.arange(ldamodel.num_topics):
            temp=[]
            temp = list(self.Reverse(topic_word_box[i][1]))
            topic_word_box_R.append(temp)
        
        D = ldamodel.num_topics
        CL = round(math.sqrt(D))
        RW = CL
        while RW * CL < D:
            RW += 1
        fig, axes = plt.subplots(RW, CL, figsize=(10,10), sharex=True, sharey=True)
        plt.style.context('grayscale')
        fig.patch.set_facecolor('#D8D8D8')

        Topic_names = ['Reporting', 'Investment', 'Risk_Management','Accounting', 'Payments_and_Settlement']
        #['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']
        #
        for i, ax in enumerate(axes.flatten()):
            if i >= D:
                axes.flatten()[i].axis('off')
            else:
            
                fig.add_subplot(ax)
                #topic_words = dict(topics[i][1])
                topic_words={}
                print(i)
                topic_words = self.Convert(topic_word_box_R[i], topic_words)
                cloud.generate_from_frequencies(topic_words, max_font_size=300)
                plt.gca().imshow(cloud)
                
                plt.gca().set_title(Topic_names[i] , fontdict=dict(size=16))
                #Topic_names.append('Topic_' + str(i +1))
                plt.gca().axis('off')


        plt.subplots_adjust(wspace=0, hspace=0)
        
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        
        #fig.patch.set_alpha(0.7)
        #plt.gca().set_facecolor('tab:gray')
        #cloud.to_file(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'))
        fig.savefig(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'), transparent=True)
        
        return 
                
class Genism_Graph(object):
    def __init__(self, Figure_saving_dir='', verbose = True):
        
        self.Figure_saving_dir = Figure_saving_dir
        self.verbose = verbose
        
    def Genism_networkX_graph(self, df, ldamodel):
        Article_names = df['article_title'].values
        
        Topic_names = ['Topic_'+ str(i) for i in range(1, len(ldamodel.show_topics())+1)]
        nx_graph = nx.Graph()
        Nodes = Topic_names + list(Article_names)
        # Add node for each Topic
        for T in Nodes:
            nx_graph.add_node(T)

        i = 0   
        # add edges
        for a in range(0,len(Article_names)):
            temp = ldamodel.get_document_topics(corpus[a])
            for c in range(0, len(temp)):
                if self.verbose:
                    
                    print('C: ', c)
                    print(len(ldamodel.get_document_topics(corpus[a])))

                T = Topic_names[temp[c][0]]
                if self.verbose:
                    print('T, A: ',T, Article_names[a])
                i += 1
                if self.verbose:
                    print('i:', i)
                # Only add edge if the count is positive

                nx_graph.add_edge(Article_names[a],T, weight = temp[c][1])
                if self.verbose:
                    print('Done_edge')
        return nx_graph
    def plot_Genism_networkX_graph(self,df, ldamodel, to_bundel=True ):
        nx_graph = self.Genism_networkX_graph(df, ldamodel)
        #Create the graph using networkX data:
        act_names = []
        kwargs = dict(width=800, height=800, xaxis=None, yaxis=None)
        opts.defaults(opts.Nodes(**kwargs), opts.Graph(**kwargs))
        
        #colors = ['#000000']+hv.Cycle('Category20').values
        topic_graph = hv.Graph.from_networkx(nx_graph, nx.layout.fruchterman_reingold_layout, k=1)
        topic_graph.nodes.data['act'] = topic_graph.nodes.data['index']
        for i in np.arange(topic_graph.nodes.data.shape[0]):
            if topic_graph.nodes.data.loc[i, 'act'] not in ['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']:
                topic_graph.nodes.data.loc[i, 'act'] = str(topic_graph.nodes.data.loc[i, 'act'].split('_')[1])
                
        
        tmp2= topic_graph.nodes.data[~topic_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        #colors = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*np.arange(len(tmp)), 20+2*np.arange(len(tmp)))]
        
        if to_bundel:

            bundled = bundle_graph(topic_graph)
            #.opts(opts.Nodes(color='act', size=10, width=1000, cmap=colors, legend_position='right'))
#opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            #
            return bundled, act_names, article_names
        else:
            return topic_graph, act_names, article_names
        #.opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            
    
    def subselect_Genism_single_Topic_graph(self, Topic_id, topic_graph):
        
        Sub_graph = topic_graph.select(start=Topic_id).opts(node_fill_color='blue', edge_line_width=hv.dim('weight'), edge_line_color='red')
        tmp2= Sub_graph.nodes.data[~Sub_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        
        return Sub_graph, act_names, article_names
    def subselect_Genism_single_Article_graph(self, Article_title, topic_graph, circular_layout=True):
        
        D = topic_graph.select(end=Article_title).opts(node_fill_color='white', edge_line_width=-1*np.log10(hv.dim('weight')))
        
        if circular_layout:
            
            F = layout_nodes(D)
            labels = hv.Labels(F.nodes, ['x', 'y'], 'index')

            return (F * labels.opts(text_font_size='10pt', text_color='black', bgcolor='white'))
        else: 
            
            return D#Goes to Util
class Text_parsing(object):
    def __init__(self):
        
        return 

    def parse_article(self, article, section_title, section_subtitle, act_name):
        article_title = article.get("TITLE")
        article_subtitle = article.get("SUBTITLE")
        article_content = article.get("CONTENT")
        assert article_title is not None
        assert article_subtitle is not None
        assert article_content is not None
        article = {"act_name":act_name,
                   "section_title": section_title,
                   "section_subtitle": section_subtitle,
                   "article_title": article_title + '_' + act_name,# + article_subtitle,
                   "article_subtitle":article_subtitle,
                   "article_content": article_content,
                  }
        return article

    def parse_section(self, section, collection=None, title=None, subtitle=None, act_name = ''):
        articles = [] if collection is None else collection
        if isinstance(section, list):
            for sec in section:
                articles = self.parse_section(sec, collection=articles,
                                         title=title, subtitle=subtitle, act_name=act_name)
            return articles
        section_title = section.get("TITLE") if title is None else f"{title}_{section.get('TITLE')}"
        section_subtitle = section.get("SUBTITLE") if title is None else f"{subtitle}_{section.get('SUBTITLE')}"

        # Only iterate if it contains a list of articles
        if "ARTICLE" in section:
            for art in section.get("ARTICLE",[]):
                article_entry = self.parse_article(art, section_title=section_title, section_subtitle=section_subtitle, act_name=act_name)
                articles.append(article_entry)
            return articles
        return self.parse_section(section["SECTION"], collection=articles,
                                  title=section_title, subtitle=section_subtitle, act_name=act_name)


    def parse_articles(self, json_data, act_name=''):
        articles = []
        sections = json_data['DOCUMENT']['SECTION']
        for i, section in enumerate(sections):
            if isinstance(section, list):
                for sec in section:
                    section_data = self.parse_section(sec,act_name=act_name)
                    articles.extend(section_data)
            else:
                section_data = self.parse_section(section,act_name=act_name)
                articles.extend(section_data)
        return pd.DataFrame(articles)

    def Read_in_Json_from_text(self, Orig_dir ='', all_jason_list_path =''):
        with open(all_jason_list_path) as fp: 
            Lines = fp.readlines() 
            files =[]
            acts =[]
            for line in Lines: 
                with open(os.path.join(Orig_dir, line.strip()), "r") as f:
                    data_1 = json.load(f)
                files.append(data_1)
                acts.append(re.sub('\.json$', '', line.strip()))
        return files, acts
    
    def merge_replicated_entries(self, df, matching_coloumn_name='', merging_coloumn_name=''):
        K = df[df.duplicated(subset=[matching_coloumn_name], keep=False)]
        df = df[~df.duplicated(subset=[matching_coloumn_name], keep=False)]

        q = K.article_title.unique()
        j =1
        for i in q:
            P =K.loc[K[matching_coloumn_name] == q[0]].reset_index(drop=True)
            temp_df = P.iloc[[0]]
            for j in range(1, P.shape[0]):
                temp_df.loc[0, merging_coloumn_name] = temp_df.loc[0, merging_coloumn_name]+ P.loc[j, merging_coloumn_name]
            df = df.append(temp_df, ignore_index=True, sort=False)

        return df
#%% preprocess text
# Goes to Util:
class Text_cleaning(object):
    def __init__(self, additional_stopwords_FullPath):
        stop_words_nltk = set(stopwords.words('english'))
        stopwords_final = set(w.rstrip() for w in open(additional_stopwords_FullPath))#'/home/brain/AiTalents/Notebooks/stopwords.txt'))
        stopwords_final = set([s.replace('\n', '') for s in stopwords_final])
        stopwords_final = stopwords_final.union(stop_words_nltk)
        self.stopwords_final = stopwords_final
    def clean_html(self, raw_text):
        cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        cleantext = re.sub(cleanr, '', raw_text)
        return cleantext

    def remove_URL(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'',text)

    
    def my_tokenizer(self, s):
        s = s.lower() # downcase
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
        tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
        tokens = [t for t in tokens if t not in string.punctuation] # remove punctuation        
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove any digits, i.e. "3rd edition"
        return tokens

    def preprocess_text(self, text):
        temp_text = self.remove_URL(text)
        no_html = self.clean_html(temp_text)
        return self.my_tokenizer(no_html)
    def Word_map_generation(self, df):
        #put clean Text into an array:
        contents = df["article_content"].values
        word_index_map = {}
        current_index = 0
        all_tokens = []
        all_content = []
        index_word_map = []
        error_count = 0
        for content in contents:
            try:
                content = content.encode('ascii', 'ignore').decode('utf-8') # this will throw exception if bad characters
                all_content.append(content)
                tokens = self.preprocess_text(content)
                all_tokens.append(tokens)
                for token in tokens:
                    if token not in word_index_map:
                        word_index_map[token] = current_index
                        current_index += 1
                        index_word_map.append(token)
            except Exception as e:
                print(e)
                print(title)
                error_count += 1
        return word_index_map, index_word_map,all_tokens, contents, error_count, self.stopwords_final


    def sklearn_Countvectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
                
        tf_vectorizer = CountVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                        token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                        tokenizer= self.preprocess_text)

        dtm_tf = tf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return tf_vectorizer, dtm_tf ## This is already a sparse feature matrix that could be fed into lda or SVD.
    def sklearn_TfidfVectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
        
        Tfidf_vectorizer = TfidfVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                           token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                           tokenizer= self.preprocess_text)

        dtm_Tfidf = Tfidf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return Tfidf_vectorizer, dtm_Tfidf ## This is already a sparse feature matrix that could be fed into lda or SVD.
            
    
    
#%% Feature generation:
#Goes to Util
class Feature_design(object):
    def __init__(self, word_index_map, all_tokens):
        self.word_index_map = word_index_map
        self.all_tokens = all_tokens
    # now let's create our input matrices - just indicator variables for this example - works better than proportions
    def tokens_to_vector(self, tokens):
        x = np.zeros(len(self.word_index_map))
        for t in tokens:
            i = self.word_index_map[t]
            x[i] = 1
        return x
    
    def Feature_matrix_generation(self):
        N = len(self.all_tokens)
        D = len(self.word_index_map)
        X = np.zeros((D, N)) # terms will go along rows, documents along columns
        i = 0
        for tokens in all_tokens:
            X[:,i] = self.tokens_to_vector(tokens)
            i += 1
        return X
    def Genism_Feature_matrix_generation(self, saving_dir):
        #analysis_dir= '/home/brain/AiTalents/Notebooks/Genism/'
        try:
            os.makedirs(saving_dir)
        except OSError:
            if not os.path.isdir(saving_dir):
                raise

        #os.chdir(analysis_dir)
        dictionary = corpora.Dictionary(self.all_tokens)
        corpus = [dictionary.doc2bow(text) for text in self.all_tokens]
        pickle.dump(corpus, open(os.path.join(saving_dir,'corpus.pkl'), 'wb'))
        dictionary.save(os.path.join(saving_dir,'dictionary.gensim'))
        return corpus, dictionary

#%% Topic models generation:
#Goes to Util
class Topic_modeling(object):
    def __init__(self, num_topics, saving_dir):
        
        
        self.num_topics = num_topics
        self.saving_dir = saving_dir
        try:
            os.makedirs(self.saving_dir)
        except OSError:
            if not os.path.isdir(self.saving_dir):
                raise
    def Genism_LDA(self, corpus, dictionary):
        #saving_dir= '/home/brain/AiTalents/Notebooks/Genism/'
            
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = self.num_topics, id2word=dictionary, passes=15)
        ldamodel.save(os.path.join(self.saving_dir,'LDA_model' + str(self.num_topics)+ '.gensim'))
        return ldamodel
    def Genism_LSI(self, corpus, dictionary):
        lsimodel = gensim.models.lsimodel.LsiModel(corpus, num_topics = self.num_topics, id2word=dictionary)
        lsimodel.save(os.path.join(self.saving_dir,'LSI_model' + str(self.num_topics)+ '.gensim'))
        return lsimodel
    def Sklearn_LDA(self, Feature_X):
        ldamodel = LatentDirichletAllocation()
        ldamodel.fit(Feature_X)
        Z = ldamodel.transform(Feature_X) # size: #term X #Component    >> For word Cloud
        # How about the graph connections?   componennts_ : size: #comps X #Articles  >> each column gives weights.
        U = ldamodel.components_    
        return ldamodel, Z, U 

            
#topics = ldamodel.print_topics(num_words=5)
#for topic in topics:
#    print(topic)

#%% Figures:

#Goes to Util
class Genism_plotting(object):
    def __init__(self,Figure_saving_dir):
        self.Figure_saving_dir = Figure_saving_dir
        try:
            os.makedirs(self.Figure_saving_dir)
        except OSError:
            if not os.path.isdir(self.Figure_saving_dir):
                raise
    def Convert(self, tup, di): 
        di = dict(tup) 
        return di

    def Reverse(self, tuples): 
        new_tup = () 
        for k in reversed(tuples): 
            new_tup = new_tup + (k,) 
        return new_tup 

    def pyLDA_Dashboard(self,ldamodel, corpus, dictionary,html_name, sort_topics=False):
        
        lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=sort_topics)
        pyLDAvis.save_html(lda_display, os.path.join(self.Figure_saving_dir,html_name + '_Dashboard.html'))
        #pyLDAvis.display(lda_display)
        return lda_display
    def Genism_word_cloud(self,html_name, ldamodel):
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

        cloud = WordCloud(background_color='#D8D8D8',
                          width=2500,
                          height=1800,
                          max_words=10,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)
        
        topic_word_box = ldamodel.show_topics(num_topics= ldamodel.num_topics, num_words= np.int(ldamodel.num_terms/100), formatted=False)
        # For each topic:
        topic_word_box_R= []
        for i in np.arange(ldamodel.num_topics):
            temp=[]
            temp = list(self.Reverse(topic_word_box[i][1]))
            topic_word_box_R.append(temp)
        
        D = ldamodel.num_topics
        CL = round(math.sqrt(D))
        RW = CL
        while RW * CL < D:
            RW += 1
        fig, axes = plt.subplots(RW, CL, figsize=(10,10), sharex=True, sharey=True)
        plt.style.context('grayscale')
        fig.patch.set_facecolor('#D8D8D8')

        Topic_names = ['Reporting', 'Investment', 'Risk_Management','Accounting', 'Payments_and_Settlement']
        #['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']
        #
        for i, ax in enumerate(axes.flatten()):
            if i >= D:
                axes.flatten()[i].axis('off')
            else:
            
                fig.add_subplot(ax)
                #topic_words = dict(topics[i][1])
                topic_words={}
                print(i)
                topic_words = self.Convert(topic_word_box_R[i], topic_words)
                cloud.generate_from_frequencies(topic_words, max_font_size=300)
                plt.gca().imshow(cloud)
                
                plt.gca().set_title(Topic_names[i] , fontdict=dict(size=16))
                #Topic_names.append('Topic_' + str(i +1))
                plt.gca().axis('off')


        plt.subplots_adjust(wspace=0, hspace=0)
        
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        
        #fig.patch.set_alpha(0.7)
        #plt.gca().set_facecolor('tab:gray')
        #cloud.to_file(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'))
        fig.savefig(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'), transparent=True)
        
        return 
                
class Genism_Graph(object):
    def __init__(self, Figure_saving_dir='', verbose = True):
        
        self.Figure_saving_dir = Figure_saving_dir
        self.verbose = verbose
        
    def Genism_networkX_graph(self, df, ldamodel):
        Article_names = df['article_title'].values
        
        Topic_names = ['Topic_'+ str(i) for i in range(1, len(ldamodel.show_topics())+1)]
        nx_graph = nx.Graph()
        Nodes = Topic_names + list(Article_names)
        # Add node for each Topic
        for T in Nodes:
            nx_graph.add_node(T)

        i = 0   
        # add edges
        for a in range(0,len(Article_names)):
            temp = ldamodel.get_document_topics(corpus[a])
            for c in range(0, len(temp)):
                if self.verbose:
                    
                    print('C: ', c)
                    print(len(ldamodel.get_document_topics(corpus[a])))

                T = Topic_names[temp[c][0]]
                if self.verbose:
                    print('T, A: ',T, Article_names[a])
                i += 1
                if self.verbose:
                    print('i:', i)
                # Only add edge if the count is positive

                nx_graph.add_edge(Article_names[a],T, weight = temp[c][1])
                if self.verbose:
                    print('Done_edge')
        return nx_graph
    def plot_Genism_networkX_graph(self,df, ldamodel, to_bundel=True ):
        nx_graph = self.Genism_networkX_graph(df, ldamodel)
        #Create the graph using networkX data:
        act_names = []
        kwargs = dict(width=800, height=800, xaxis=None, yaxis=None)
        opts.defaults(opts.Nodes(**kwargs), opts.Graph(**kwargs))
        
        #colors = ['#000000']+hv.Cycle('Category20').values
        topic_graph = hv.Graph.from_networkx(nx_graph, nx.layout.fruchterman_reingold_layout, k=1)
        topic_graph.nodes.data['act'] = topic_graph.nodes.data['index']
        for i in np.arange(topic_graph.nodes.data.shape[0]):
            if topic_graph.nodes.data.loc[i, 'act'] not in ['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']:
                topic_graph.nodes.data.loc[i, 'act'] = str(topic_graph.nodes.data.loc[i, 'act'].split('_')[1])
                
        
        tmp2= topic_graph.nodes.data[~topic_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        #colors = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*np.arange(len(tmp)), 20+2*np.arange(len(tmp)))]
        
        if to_bundel:

            bundled = bundle_graph(topic_graph)
            #.opts(opts.Nodes(color='act', size=10, width=1000, cmap=colors, legend_position='right'))
#opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            #
            return bundled, act_names, article_names
        else:
            return topic_graph, act_names, article_names
        #.opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            
    
    def subselect_Genism_single_Topic_graph(self, Topic_id, topic_graph):
        
        Sub_graph = topic_graph.select(start=Topic_id).opts(node_fill_color='blue', edge_line_width=hv.dim('weight'), edge_line_color='red')
        tmp2= Sub_graph.nodes.data[~Sub_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        
        return Sub_graph, act_names, article_names
    def subselect_Genism_single_Article_graph(self, Article_title, topic_graph, circular_layout=True):
        
        D = topic_graph.select(end=Article_title).opts(node_fill_color='white', edge_line_width=-1*np.log10(hv.dim('weight')))
        
        if circular_layout:
            
            F = layout_nodes(D)
            labels = hv.Labels(F.nodes, ['x', 'y'], 'index')

            return (F * labels.opts(text_font_size='10pt', text_color='black', bgcolor='white'))
        else: 
            
            return D#Goes to Util
class Text_parsing(object):
    def __init__(self):
        
        return 

    def parse_article(self, article, section_title, section_subtitle, act_name):
        article_title = article.get("TITLE")
        article_subtitle = article.get("SUBTITLE")
        article_content = article.get("CONTENT")
        assert article_title is not None
        assert article_subtitle is not None
        assert article_content is not None
        article = {"act_name":act_name,
                   "section_title": section_title,
                   "section_subtitle": section_subtitle,
                   "article_title": article_title + '_' + act_name,# + article_subtitle,
                   "article_subtitle":article_subtitle,
                   "article_content": article_content,
                  }
        return article

    def parse_section(self, section, collection=None, title=None, subtitle=None, act_name = ''):
        articles = [] if collection is None else collection
        if isinstance(section, list):
            for sec in section:
                articles = self.parse_section(sec, collection=articles,
                                         title=title, subtitle=subtitle, act_name=act_name)
            return articles
        section_title = section.get("TITLE") if title is None else f"{title}_{section.get('TITLE')}"
        section_subtitle = section.get("SUBTITLE") if title is None else f"{subtitle}_{section.get('SUBTITLE')}"

        # Only iterate if it contains a list of articles
        if "ARTICLE" in section:
            for art in section.get("ARTICLE",[]):
                article_entry = self.parse_article(art, section_title=section_title, section_subtitle=section_subtitle, act_name=act_name)
                articles.append(article_entry)
            return articles
        return self.parse_section(section["SECTION"], collection=articles,
                                  title=section_title, subtitle=section_subtitle, act_name=act_name)


    def parse_articles(self, json_data, act_name=''):
        articles = []
        sections = json_data['DOCUMENT']['SECTION']
        for i, section in enumerate(sections):
            if isinstance(section, list):
                for sec in section:
                    section_data = self.parse_section(sec,act_name=act_name)
                    articles.extend(section_data)
            else:
                section_data = self.parse_section(section,act_name=act_name)
                articles.extend(section_data)
        return pd.DataFrame(articles)

    def Read_in_Json_from_text(self, Orig_dir ='', all_jason_list_path =''):
        with open(all_jason_list_path) as fp: 
            Lines = fp.readlines() 
            files =[]
            acts =[]
            for line in Lines: 
                with open(os.path.join(Orig_dir, line.strip()), "r") as f:
                    data_1 = json.load(f)
                files.append(data_1)
                acts.append(re.sub('\.json$', '', line.strip()))
        return files, acts
    
    def merge_replicated_entries(self, df, matching_coloumn_name='', merging_coloumn_name=''):
        K = df[df.duplicated(subset=[matching_coloumn_name], keep=False)]
        df = df[~df.duplicated(subset=[matching_coloumn_name], keep=False)]

        q = K.article_title.unique()
        j =1
        for i in q:
            P =K.loc[K[matching_coloumn_name] == q[0]].reset_index(drop=True)
            temp_df = P.iloc[[0]]
            for j in range(1, P.shape[0]):
                temp_df.loc[0, merging_coloumn_name] = temp_df.loc[0, merging_coloumn_name]+ P.loc[j, merging_coloumn_name]
            df = df.append(temp_df, ignore_index=True, sort=False)

        return df
#%% preprocess text
# Goes to Util:
class Text_cleaning(object):
    def __init__(self, additional_stopwords_FullPath):
        stop_words_nltk = set(stopwords.words('english'))
        stopwords_final = set(w.rstrip() for w in open(additional_stopwords_FullPath))#'/home/brain/AiTalents/Notebooks/stopwords.txt'))
        stopwords_final = set([s.replace('\n', '') for s in stopwords_final])
        stopwords_final = stopwords_final.union(stop_words_nltk)
        self.stopwords_final = stopwords_final
    def clean_html(self, raw_text):
        cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        cleantext = re.sub(cleanr, '', raw_text)
        return cleantext

    def remove_URL(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'',text)

    
    def my_tokenizer(self, s):
        s = s.lower() # downcase
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
        tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
        tokens = [t for t in tokens if t not in string.punctuation] # remove punctuation        
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove any digits, i.e. "3rd edition"
        return tokens

    def preprocess_text(self, text):
        temp_text = self.remove_URL(text)
        no_html = self.clean_html(temp_text)
        return self.my_tokenizer(no_html)
    def Word_map_generation(self, df):
        #put clean Text into an array:
        contents = df["article_content"].values
        word_index_map = {}
        current_index = 0
        all_tokens = []
        all_content = []
        index_word_map = []
        error_count = 0
        for content in contents:
            try:
                content = content.encode('ascii', 'ignore').decode('utf-8') # this will throw exception if bad characters
                all_content.append(content)
                tokens = self.preprocess_text(content)
                all_tokens.append(tokens)
                for token in tokens:
                    if token not in word_index_map:
                        word_index_map[token] = current_index
                        current_index += 1
                        index_word_map.append(token)
            except Exception as e:
                print(e)
                print(title)
                error_count += 1
        return word_index_map, index_word_map,all_tokens, contents, error_count, self.stopwords_final


    def sklearn_Countvectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
                
        tf_vectorizer = CountVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                        token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                        tokenizer= self.preprocess_text)

        dtm_tf = tf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return tf_vectorizer, dtm_tf ## This is already a sparse feature matrix that could be fed into lda or SVD.
    def sklearn_TfidfVectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
        
        Tfidf_vectorizer = TfidfVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                           token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                           tokenizer= self.preprocess_text)

        dtm_Tfidf = Tfidf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return Tfidf_vectorizer, dtm_Tfidf ## This is already a sparse feature matrix that could be fed into lda or SVD.
            
    
    
#%% Feature generation:
#Goes to Util
class Feature_design(object):
    def __init__(self, word_index_map, all_tokens):
        self.word_index_map = word_index_map
        self.all_tokens = all_tokens
    # now let's create our input matrices - just indicator variables for this example - works better than proportions
    def tokens_to_vector(self, tokens):
        x = np.zeros(len(self.word_index_map))
        for t in tokens:
            i = self.word_index_map[t]
            x[i] = 1
        return x
    
    def Feature_matrix_generation(self):
        N = len(self.all_tokens)
        D = len(self.word_index_map)
        X = np.zeros((D, N)) # terms will go along rows, documents along columns
        i = 0
        for tokens in all_tokens:
            X[:,i] = self.tokens_to_vector(tokens)
            i += 1
        return X
    def Genism_Feature_matrix_generation(self, saving_dir):
        #analysis_dir= '/home/brain/AiTalents/Notebooks/Genism/'
        try:
            os.makedirs(saving_dir)
        except OSError:
            if not os.path.isdir(saving_dir):
                raise

        #os.chdir(analysis_dir)
        dictionary = corpora.Dictionary(self.all_tokens)
        corpus = [dictionary.doc2bow(text) for text in self.all_tokens]
        pickle.dump(corpus, open(os.path.join(saving_dir,'corpus.pkl'), 'wb'))
        dictionary.save(os.path.join(saving_dir,'dictionary.gensim'))
        return corpus, dictionary

#%% Topic models generation:
#Goes to Util
class Topic_modeling(object):
    def __init__(self, num_topics, saving_dir):
        
        
        self.num_topics = num_topics
        self.saving_dir = saving_dir
        try:
            os.makedirs(self.saving_dir)
        except OSError:
            if not os.path.isdir(self.saving_dir):
                raise
    def Genism_LDA(self, corpus, dictionary):
        #saving_dir= '/home/brain/AiTalents/Notebooks/Genism/'
            
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = self.num_topics, id2word=dictionary, passes=15)
        ldamodel.save(os.path.join(self.saving_dir,'LDA_model' + str(self.num_topics)+ '.gensim'))
        return ldamodel
    def Genism_LSI(self, corpus, dictionary):
        lsimodel = gensim.models.lsimodel.LsiModel(corpus, num_topics = self.num_topics, id2word=dictionary)
        lsimodel.save(os.path.join(self.saving_dir,'LSI_model' + str(self.num_topics)+ '.gensim'))
        return lsimodel
    def Sklearn_LDA(self, Feature_X):
        ldamodel = LatentDirichletAllocation()
        ldamodel.fit(Feature_X)
        Z = ldamodel.transform(Feature_X) # size: #term X #Component    >> For word Cloud
        # How about the graph connections?   componennts_ : size: #comps X #Articles  >> each column gives weights.
        U = ldamodel.components_    
        return ldamodel, Z, U 

            
#topics = ldamodel.print_topics(num_words=5)
#for topic in topics:
#    print(topic)

#%% Figures:

#Goes to Util
class Genism_plotting(object):
    def __init__(self,Figure_saving_dir):
        self.Figure_saving_dir = Figure_saving_dir
        try:
            os.makedirs(self.Figure_saving_dir)
        except OSError:
            if not os.path.isdir(self.Figure_saving_dir):
                raise
    def Convert(self, tup, di): 
        di = dict(tup) 
        return di

    def Reverse(self, tuples): 
        new_tup = () 
        for k in reversed(tuples): 
            new_tup = new_tup + (k,) 
        return new_tup 

    def pyLDA_Dashboard(self,ldamodel, corpus, dictionary,html_name, sort_topics=False):
        
        lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=sort_topics)
        pyLDAvis.save_html(lda_display, os.path.join(self.Figure_saving_dir,html_name + '_Dashboard.html'))
        #pyLDAvis.display(lda_display)
        return lda_display
    def Genism_word_cloud(self,html_name, ldamodel):
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

        cloud = WordCloud(background_color='#D8D8D8',
                          width=2500,
                          height=1800,
                          max_words=10,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)
        
        topic_word_box = ldamodel.show_topics(num_topics= ldamodel.num_topics, num_words= np.int(ldamodel.num_terms/100), formatted=False)
        # For each topic:
        topic_word_box_R= []
        for i in np.arange(ldamodel.num_topics):
            temp=[]
            temp = list(self.Reverse(topic_word_box[i][1]))
            topic_word_box_R.append(temp)
        
        D = ldamodel.num_topics
        CL = round(math.sqrt(D))
        RW = CL
        while RW * CL < D:
            RW += 1
        fig, axes = plt.subplots(RW, CL, figsize=(10,10), sharex=True, sharey=True)
        plt.style.context('grayscale')
        fig.patch.set_facecolor('#D8D8D8')

        Topic_names = ['Reporting', 'Investment', 'Risk_Management','Accounting', 'Payments_and_Settlement']
        #['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']
        #
        for i, ax in enumerate(axes.flatten()):
            if i >= D:
                axes.flatten()[i].axis('off')
            else:
            
                fig.add_subplot(ax)
                #topic_words = dict(topics[i][1])
                topic_words={}
                print(i)
                topic_words = self.Convert(topic_word_box_R[i], topic_words)
                cloud.generate_from_frequencies(topic_words, max_font_size=300)
                plt.gca().imshow(cloud)
                
                plt.gca().set_title(Topic_names[i] , fontdict=dict(size=16))
                #Topic_names.append('Topic_' + str(i +1))
                plt.gca().axis('off')


        plt.subplots_adjust(wspace=0, hspace=0)
        
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        
        #fig.patch.set_alpha(0.7)
        #plt.gca().set_facecolor('tab:gray')
        #cloud.to_file(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'))
        fig.savefig(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'), transparent=True)
        
        return 
                
class Genism_Graph(object):
    def __init__(self, Figure_saving_dir='', verbose = True):
        
        self.Figure_saving_dir = Figure_saving_dir
        self.verbose = verbose
        
    def Genism_networkX_graph(self, df, ldamodel):
        Article_names = df['article_title'].values
        
        Topic_names = ['Topic_'+ str(i) for i in range(1, len(ldamodel.show_topics())+1)]
        nx_graph = nx.Graph()
        Nodes = Topic_names + list(Article_names)
        # Add node for each Topic
        for T in Nodes:
            nx_graph.add_node(T)

        i = 0   
        # add edges
        for a in range(0,len(Article_names)):
            temp = ldamodel.get_document_topics(corpus[a])
            for c in range(0, len(temp)):
                if self.verbose:
                    
                    print('C: ', c)
                    print(len(ldamodel.get_document_topics(corpus[a])))

                T = Topic_names[temp[c][0]]
                if self.verbose:
                    print('T, A: ',T, Article_names[a])
                i += 1
                if self.verbose:
                    print('i:', i)
                # Only add edge if the count is positive

                nx_graph.add_edge(Article_names[a],T, weight = temp[c][1])
                if self.verbose:
                    print('Done_edge')
        return nx_graph
    def plot_Genism_networkX_graph(self,df, ldamodel, to_bundel=True ):
        nx_graph = self.Genism_networkX_graph(df, ldamodel)
        #Create the graph using networkX data:
        act_names = []
        kwargs = dict(width=800, height=800, xaxis=None, yaxis=None)
        opts.defaults(opts.Nodes(**kwargs), opts.Graph(**kwargs))
        
        #colors = ['#000000']+hv.Cycle('Category20').values
        topic_graph = hv.Graph.from_networkx(nx_graph, nx.layout.fruchterman_reingold_layout, k=1)
        topic_graph.nodes.data['act'] = topic_graph.nodes.data['index']
        for i in np.arange(topic_graph.nodes.data.shape[0]):
            if topic_graph.nodes.data.loc[i, 'act'] not in ['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']:
                topic_graph.nodes.data.loc[i, 'act'] = str(topic_graph.nodes.data.loc[i, 'act'].split('_')[1])
                
        
        tmp2= topic_graph.nodes.data[~topic_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        #colors = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*np.arange(len(tmp)), 20+2*np.arange(len(tmp)))]
        
        if to_bundel:

            bundled = bundle_graph(topic_graph)
            #.opts(opts.Nodes(color='act', size=10, width=1000, cmap=colors, legend_position='right'))
#opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            #
            return bundled, act_names, article_names
        else:
            return topic_graph, act_names, article_names
        #.opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            
    
    def subselect_Genism_single_Topic_graph(self, Topic_id, topic_graph):
        
        Sub_graph = topic_graph.select(start=Topic_id).opts(node_fill_color='blue', edge_line_width=hv.dim('weight'), edge_line_color='red')
        tmp2= Sub_graph.nodes.data[~Sub_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        
        return Sub_graph, act_names, article_names
    def subselect_Genism_single_Article_graph(self, Article_title, topic_graph, circular_layout=True):
        
        D = topic_graph.select(end=Article_title).opts(node_fill_color='white', edge_line_width=-1*np.log10(hv.dim('weight')))
        
        if circular_layout:
            
            F = layout_nodes(D)
            labels = hv.Labels(F.nodes, ['x', 'y'], 'index')

            return (F * labels.opts(text_font_size='10pt', text_color='black', bgcolor='white'))
        else: 
            
            return D#Goes to Util
class Text_parsing(object):
    def __init__(self):
        
        return 

    def parse_article(self, article, section_title, section_subtitle, act_name):
        article_title = article.get("TITLE")
        article_subtitle = article.get("SUBTITLE")
        article_content = article.get("CONTENT")
        assert article_title is not None
        assert article_subtitle is not None
        assert article_content is not None
        article = {"act_name":act_name,
                   "section_title": section_title,
                   "section_subtitle": section_subtitle,
                   "article_title": article_title + '_' + act_name,# + article_subtitle,
                   "article_subtitle":article_subtitle,
                   "article_content": article_content,
                  }
        return article

    def parse_section(self, section, collection=None, title=None, subtitle=None, act_name = ''):
        articles = [] if collection is None else collection
        if isinstance(section, list):
            for sec in section:
                articles = self.parse_section(sec, collection=articles,
                                         title=title, subtitle=subtitle, act_name=act_name)
            return articles
        section_title = section.get("TITLE") if title is None else f"{title}_{section.get('TITLE')}"
        section_subtitle = section.get("SUBTITLE") if title is None else f"{subtitle}_{section.get('SUBTITLE')}"

        # Only iterate if it contains a list of articles
        if "ARTICLE" in section:
            for art in section.get("ARTICLE",[]):
                article_entry = self.parse_article(art, section_title=section_title, section_subtitle=section_subtitle, act_name=act_name)
                articles.append(article_entry)
            return articles
        return self.parse_section(section["SECTION"], collection=articles,
                                  title=section_title, subtitle=section_subtitle, act_name=act_name)


    def parse_articles(self, json_data, act_name=''):
        articles = []
        sections = json_data['DOCUMENT']['SECTION']
        for i, section in enumerate(sections):
            if isinstance(section, list):
                for sec in section:
                    section_data = self.parse_section(sec,act_name=act_name)
                    articles.extend(section_data)
            else:
                section_data = self.parse_section(section,act_name=act_name)
                articles.extend(section_data)
        return pd.DataFrame(articles)

    def Read_in_Json_from_text(self, Orig_dir ='', all_jason_list_path =''):
        with open(all_jason_list_path) as fp: 
            Lines = fp.readlines() 
            files =[]
            acts =[]
            for line in Lines: 
                with open(os.path.join(Orig_dir, line.strip()), "r") as f:
                    data_1 = json.load(f)
                files.append(data_1)
                acts.append(re.sub('\.json$', '', line.strip()))
        return files, acts
    
    def merge_replicated_entries(self, df, matching_coloumn_name='', merging_coloumn_name=''):
        K = df[df.duplicated(subset=[matching_coloumn_name], keep=False)]
        df = df[~df.duplicated(subset=[matching_coloumn_name], keep=False)]

        q = K.article_title.unique()
        j =1
        for i in q:
            P =K.loc[K[matching_coloumn_name] == q[0]].reset_index(drop=True)
            temp_df = P.iloc[[0]]
            for j in range(1, P.shape[0]):
                temp_df.loc[0, merging_coloumn_name] = temp_df.loc[0, merging_coloumn_name]+ P.loc[j, merging_coloumn_name]
            df = df.append(temp_df, ignore_index=True, sort=False)

        return df
#%% preprocess text
# Goes to Util:
class Text_cleaning(object):
    def __init__(self, additional_stopwords_FullPath):
        stop_words_nltk = set(stopwords.words('english'))
        stopwords_final = set(w.rstrip() for w in open(additional_stopwords_FullPath))#'/home/brain/AiTalents/Notebooks/stopwords.txt'))
        stopwords_final = set([s.replace('\n', '') for s in stopwords_final])
        stopwords_final = stopwords_final.union(stop_words_nltk)
        self.stopwords_final = stopwords_final
    def clean_html(self, raw_text):
        cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        cleantext = re.sub(cleanr, '', raw_text)
        return cleantext

    def remove_URL(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'',text)

    
    def my_tokenizer(self, s):
        s = s.lower() # downcase
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
        tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
        tokens = [t for t in tokens if t not in string.punctuation] # remove punctuation        
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove any digits, i.e. "3rd edition"
        return tokens

    def preprocess_text(self, text):
        temp_text = self.remove_URL(text)
        no_html = self.clean_html(temp_text)
        return self.my_tokenizer(no_html)
    def Word_map_generation(self, df):
        #put clean Text into an array:
        contents = df["article_content"].values
        word_index_map = {}
        current_index = 0
        all_tokens = []
        all_content = []
        index_word_map = []
        error_count = 0
        for content in contents:
            try:
                content = content.encode('ascii', 'ignore').decode('utf-8') # this will throw exception if bad characters
                all_content.append(content)
                tokens = self.preprocess_text(content)
                all_tokens.append(tokens)
                for token in tokens:
                    if token not in word_index_map:
                        word_index_map[token] = current_index
                        current_index += 1
                        index_word_map.append(token)
            except Exception as e:
                print(e)
                print(title)
                error_count += 1
        return word_index_map, index_word_map,all_tokens, contents, error_count, self.stopwords_final


    def sklearn_Countvectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
                
        tf_vectorizer = CountVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                        token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                        tokenizer= self.preprocess_text)

        dtm_tf = tf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return tf_vectorizer, dtm_tf ## This is already a sparse feature matrix that could be fed into lda or SVD.
    def sklearn_TfidfVectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
        
        Tfidf_vectorizer = TfidfVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                           token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                           tokenizer= self.preprocess_text)

        dtm_Tfidf = Tfidf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return Tfidf_vectorizer, dtm_Tfidf ## This is already a sparse feature matrix that could be fed into lda or SVD.
            
    
    
#%% Feature generation:
#Goes to Util
class Feature_design(object):
    def __init__(self, word_index_map, all_tokens):
        self.word_index_map = word_index_map
        self.all_tokens = all_tokens
    # now let's create our input matrices - just indicator variables for this example - works better than proportions
    def tokens_to_vector(self, tokens):
        x = np.zeros(len(self.word_index_map))
        for t in tokens:
            i = self.word_index_map[t]
            x[i] = 1
        return x
    
    def Feature_matrix_generation(self):
        N = len(self.all_tokens)
        D = len(self.word_index_map)
        X = np.zeros((D, N)) # terms will go along rows, documents along columns
        i = 0
        for tokens in all_tokens:
            X[:,i] = self.tokens_to_vector(tokens)
            i += 1
        return X
    def Genism_Feature_matrix_generation(self, saving_dir):
        #analysis_dir= '/home/brain/AiTalents/Notebooks/Genism/'
        try:
            os.makedirs(saving_dir)
        except OSError:
            if not os.path.isdir(saving_dir):
                raise

        #os.chdir(analysis_dir)
        dictionary = corpora.Dictionary(self.all_tokens)
        corpus = [dictionary.doc2bow(text) for text in self.all_tokens]
        pickle.dump(corpus, open(os.path.join(saving_dir,'corpus.pkl'), 'wb'))
        dictionary.save(os.path.join(saving_dir,'dictionary.gensim'))
        return corpus, dictionary

#%% Topic models generation:
#Goes to Util
class Topic_modeling(object):
    def __init__(self, num_topics, saving_dir):
        
        
        self.num_topics = num_topics
        self.saving_dir = saving_dir
        try:
            os.makedirs(self.saving_dir)
        except OSError:
            if not os.path.isdir(self.saving_dir):
                raise
    def Genism_LDA(self, corpus, dictionary):
        #saving_dir= '/home/brain/AiTalents/Notebooks/Genism/'
            
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = self.num_topics, id2word=dictionary, passes=15)
        ldamodel.save(os.path.join(self.saving_dir,'LDA_model' + str(self.num_topics)+ '.gensim'))
        return ldamodel
    def Genism_LSI(self, corpus, dictionary):
        lsimodel = gensim.models.lsimodel.LsiModel(corpus, num_topics = self.num_topics, id2word=dictionary)
        lsimodel.save(os.path.join(self.saving_dir,'LSI_model' + str(self.num_topics)+ '.gensim'))
        return lsimodel
    def Sklearn_LDA(self, Feature_X):
        ldamodel = LatentDirichletAllocation()
        ldamodel.fit(Feature_X)
        Z = ldamodel.transform(Feature_X) # size: #term X #Component    >> For word Cloud
        # How about the graph connections?   componennts_ : size: #comps X #Articles  >> each column gives weights.
        U = ldamodel.components_    
        return ldamodel, Z, U 

            
#topics = ldamodel.print_topics(num_words=5)
#for topic in topics:
#    print(topic)

#%% Figures:

#Goes to Util
class Genism_plotting(object):
    def __init__(self,Figure_saving_dir):
        self.Figure_saving_dir = Figure_saving_dir
        try:
            os.makedirs(self.Figure_saving_dir)
        except OSError:
            if not os.path.isdir(self.Figure_saving_dir):
                raise
    def Convert(self, tup, di): 
        di = dict(tup) 
        return di

    def Reverse(self, tuples): 
        new_tup = () 
        for k in reversed(tuples): 
            new_tup = new_tup + (k,) 
        return new_tup 

    def pyLDA_Dashboard(self,ldamodel, corpus, dictionary,html_name, sort_topics=False):
        
        lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=sort_topics)
        pyLDAvis.save_html(lda_display, os.path.join(self.Figure_saving_dir,html_name + '_Dashboard.html'))
        #pyLDAvis.display(lda_display)
        return lda_display
    def Genism_word_cloud(self,html_name, ldamodel):
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

        cloud = WordCloud(background_color='#D8D8D8',
                          width=2500,
                          height=1800,
                          max_words=10,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)
        
        topic_word_box = ldamodel.show_topics(num_topics= ldamodel.num_topics, num_words= np.int(ldamodel.num_terms/100), formatted=False)
        # For each topic:
        topic_word_box_R= []
        for i in np.arange(ldamodel.num_topics):
            temp=[]
            temp = list(self.Reverse(topic_word_box[i][1]))
            topic_word_box_R.append(temp)
        
        D = ldamodel.num_topics
        CL = round(math.sqrt(D))
        RW = CL
        while RW * CL < D:
            RW += 1
        fig, axes = plt.subplots(RW, CL, figsize=(10,10), sharex=True, sharey=True)
        plt.style.context('grayscale')
        fig.patch.set_facecolor('#D8D8D8')

        Topic_names = ['Reporting', 'Investment', 'Risk_Management','Accounting', 'Payments_and_Settlement']
        #['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']
        #
        for i, ax in enumerate(axes.flatten()):
            if i >= D:
                axes.flatten()[i].axis('off')
            else:
            
                fig.add_subplot(ax)
                #topic_words = dict(topics[i][1])
                topic_words={}
                print(i)
                topic_words = self.Convert(topic_word_box_R[i], topic_words)
                cloud.generate_from_frequencies(topic_words, max_font_size=300)
                plt.gca().imshow(cloud)
                
                plt.gca().set_title(Topic_names[i] , fontdict=dict(size=16))
                #Topic_names.append('Topic_' + str(i +1))
                plt.gca().axis('off')


        plt.subplots_adjust(wspace=0, hspace=0)
        
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        
        #fig.patch.set_alpha(0.7)
        #plt.gca().set_facecolor('tab:gray')
        #cloud.to_file(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'))
        fig.savefig(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'), transparent=True)
        
        return 
                
class Genism_Graph(object):
    def __init__(self, Figure_saving_dir='', verbose = True):
        
        self.Figure_saving_dir = Figure_saving_dir
        self.verbose = verbose
        
    def Genism_networkX_graph(self, df, ldamodel):
        Article_names = df['article_title'].values
        
        Topic_names = ['Topic_'+ str(i) for i in range(1, len(ldamodel.show_topics())+1)]
        nx_graph = nx.Graph()
        Nodes = Topic_names + list(Article_names)
        # Add node for each Topic
        for T in Nodes:
            nx_graph.add_node(T)

        i = 0   
        # add edges
        for a in range(0,len(Article_names)):
            temp = ldamodel.get_document_topics(corpus[a])
            for c in range(0, len(temp)):
                if self.verbose:
                    
                    print('C: ', c)
                    print(len(ldamodel.get_document_topics(corpus[a])))

                T = Topic_names[temp[c][0]]
                if self.verbose:
                    print('T, A: ',T, Article_names[a])
                i += 1
                if self.verbose:
                    print('i:', i)
                # Only add edge if the count is positive

                nx_graph.add_edge(Article_names[a],T, weight = temp[c][1])
                if self.verbose:
                    print('Done_edge')
        return nx_graph
    def plot_Genism_networkX_graph(self,df, ldamodel, to_bundel=True ):
        nx_graph = self.Genism_networkX_graph(df, ldamodel)
        #Create the graph using networkX data:
        act_names = []
        kwargs = dict(width=800, height=800, xaxis=None, yaxis=None)
        opts.defaults(opts.Nodes(**kwargs), opts.Graph(**kwargs))
        
        #colors = ['#000000']+hv.Cycle('Category20').values
        topic_graph = hv.Graph.from_networkx(nx_graph, nx.layout.fruchterman_reingold_layout, k=1)
        topic_graph.nodes.data['act'] = topic_graph.nodes.data['index']
        for i in np.arange(topic_graph.nodes.data.shape[0]):
            if topic_graph.nodes.data.loc[i, 'act'] not in ['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']:
                topic_graph.nodes.data.loc[i, 'act'] = str(topic_graph.nodes.data.loc[i, 'act'].split('_')[1])
                
        
        tmp2= topic_graph.nodes.data[~topic_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        #colors = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*np.arange(len(tmp)), 20+2*np.arange(len(tmp)))]
        
        if to_bundel:

            bundled = bundle_graph(topic_graph)
            #.opts(opts.Nodes(color='act', size=10, width=1000, cmap=colors, legend_position='right'))
#opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            #
            return bundled, act_names, article_names
        else:
            return topic_graph, act_names, article_names
        #.opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            
    
    def subselect_Genism_single_Topic_graph(self, Topic_id, topic_graph):
        
        Sub_graph = topic_graph.select(start=Topic_id).opts(node_fill_color='blue', edge_line_width=hv.dim('weight'), edge_line_color='red')
        tmp2= Sub_graph.nodes.data[~Sub_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        
        return Sub_graph, act_names, article_names
    def subselect_Genism_single_Article_graph(self, Article_title, topic_graph, circular_layout=True):
        
        D = topic_graph.select(end=Article_title).opts(node_fill_color='white', edge_line_width=-1*np.log10(hv.dim('weight')))
        
        if circular_layout:
            
            F = layout_nodes(D)
            labels = hv.Labels(F.nodes, ['x', 'y'], 'index')

            return (F * labels.opts(text_font_size='10pt', text_color='black', bgcolor='white'))
        else: 
            
            return D#Goes to Util
class Text_parsing(object):
    def __init__(self):
        
        return 

    def parse_article(self, article, section_title, section_subtitle, act_name):
        article_title = article.get("TITLE")
        article_subtitle = article.get("SUBTITLE")
        article_content = article.get("CONTENT")
        assert article_title is not None
        assert article_subtitle is not None
        assert article_content is not None
        article = {"act_name":act_name,
                   "section_title": section_title,
                   "section_subtitle": section_subtitle,
                   "article_title": article_title + '_' + act_name,# + article_subtitle,
                   "article_subtitle":article_subtitle,
                   "article_content": article_content,
                  }
        return article

    def parse_section(self, section, collection=None, title=None, subtitle=None, act_name = ''):
        articles = [] if collection is None else collection
        if isinstance(section, list):
            for sec in section:
                articles = self.parse_section(sec, collection=articles,
                                         title=title, subtitle=subtitle, act_name=act_name)
            return articles
        section_title = section.get("TITLE") if title is None else f"{title}_{section.get('TITLE')}"
        section_subtitle = section.get("SUBTITLE") if title is None else f"{subtitle}_{section.get('SUBTITLE')}"

        # Only iterate if it contains a list of articles
        if "ARTICLE" in section:
            for art in section.get("ARTICLE",[]):
                article_entry = self.parse_article(art, section_title=section_title, section_subtitle=section_subtitle, act_name=act_name)
                articles.append(article_entry)
            return articles
        return self.parse_section(section["SECTION"], collection=articles,
                                  title=section_title, subtitle=section_subtitle, act_name=act_name)


    def parse_articles(self, json_data, act_name=''):
        articles = []
        sections = json_data['DOCUMENT']['SECTION']
        for i, section in enumerate(sections):
            if isinstance(section, list):
                for sec in section:
                    section_data = self.parse_section(sec,act_name=act_name)
                    articles.extend(section_data)
            else:
                section_data = self.parse_section(section,act_name=act_name)
                articles.extend(section_data)
        return pd.DataFrame(articles)

    def Read_in_Json_from_text(self, Orig_dir ='', all_jason_list_path =''):
        with open(all_jason_list_path) as fp: 
            Lines = fp.readlines() 
            files =[]
            acts =[]
            for line in Lines: 
                with open(os.path.join(Orig_dir, line.strip()), "r") as f:
                    data_1 = json.load(f)
                files.append(data_1)
                acts.append(re.sub('\.json$', '', line.strip()))
        return files, acts
    
    def merge_replicated_entries(self, df, matching_coloumn_name='', merging_coloumn_name=''):
        K = df[df.duplicated(subset=[matching_coloumn_name], keep=False)]
        df = df[~df.duplicated(subset=[matching_coloumn_name], keep=False)]

        q = K.article_title.unique()
        j =1
        for i in q:
            P =K.loc[K[matching_coloumn_name] == q[0]].reset_index(drop=True)
            temp_df = P.iloc[[0]]
            for j in range(1, P.shape[0]):
                temp_df.loc[0, merging_coloumn_name] = temp_df.loc[0, merging_coloumn_name]+ P.loc[j, merging_coloumn_name]
            df = df.append(temp_df, ignore_index=True, sort=False)

        return df
#%% preprocess text
# Goes to Util:
class Text_cleaning(object):
    def __init__(self, additional_stopwords_FullPath):
        stop_words_nltk = set(stopwords.words('english'))
        stopwords_final = set(w.rstrip() for w in open(additional_stopwords_FullPath))#'/home/brain/AiTalents/Notebooks/stopwords.txt'))
        stopwords_final = set([s.replace('\n', '') for s in stopwords_final])
        stopwords_final = stopwords_final.union(stop_words_nltk)
        self.stopwords_final = stopwords_final
    def clean_html(self, raw_text):
        cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        cleantext = re.sub(cleanr, '', raw_text)
        return cleantext

    def remove_URL(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'',text)

    
    def my_tokenizer(self, s):
        s = s.lower() # downcase
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
        tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
        tokens = [t for t in tokens if t not in string.punctuation] # remove punctuation        
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove any digits, i.e. "3rd edition"
        return tokens

    def preprocess_text(self, text):
        temp_text = self.remove_URL(text)
        no_html = self.clean_html(temp_text)
        return self.my_tokenizer(no_html)
    def Word_map_generation(self, df):
        #put clean Text into an array:
        contents = df["article_content"].values
        word_index_map = {}
        current_index = 0
        all_tokens = []
        all_content = []
        index_word_map = []
        error_count = 0
        for content in contents:
            try:
                content = content.encode('ascii', 'ignore').decode('utf-8') # this will throw exception if bad characters
                all_content.append(content)
                tokens = self.preprocess_text(content)
                all_tokens.append(tokens)
                for token in tokens:
                    if token not in word_index_map:
                        word_index_map[token] = current_index
                        current_index += 1
                        index_word_map.append(token)
            except Exception as e:
                print(e)
                print(title)
                error_count += 1
        return word_index_map, index_word_map,all_tokens, contents, error_count, self.stopwords_final


    def sklearn_Countvectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
                
        tf_vectorizer = CountVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                        token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                        tokenizer= self.preprocess_text)

        dtm_tf = tf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return tf_vectorizer, dtm_tf ## This is already a sparse feature matrix that could be fed into lda or SVD.
    def sklearn_TfidfVectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
        
        Tfidf_vectorizer = TfidfVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                           token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                           tokenizer= self.preprocess_text)

        dtm_Tfidf = Tfidf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return Tfidf_vectorizer, dtm_Tfidf ## This is already a sparse feature matrix that could be fed into lda or SVD.
            
    
    
#%% Feature generation:
#Goes to Util
class Feature_design(object):
    def __init__(self, word_index_map, all_tokens):
        self.word_index_map = word_index_map
        self.all_tokens = all_tokens
    # now let's create our input matrices - just indicator variables for this example - works better than proportions
    def tokens_to_vector(self, tokens):
        x = np.zeros(len(self.word_index_map))
        for t in tokens:
            i = self.word_index_map[t]
            x[i] = 1
        return x
    
    def Feature_matrix_generation(self):
        N = len(self.all_tokens)
        D = len(self.word_index_map)
        X = np.zeros((D, N)) # terms will go along rows, documents along columns
        i = 0
        for tokens in all_tokens:
            X[:,i] = self.tokens_to_vector(tokens)
            i += 1
        return X
    def Genism_Feature_matrix_generation(self, saving_dir):
        #analysis_dir= '/home/brain/AiTalents/Notebooks/Genism/'
        try:
            os.makedirs(saving_dir)
        except OSError:
            if not os.path.isdir(saving_dir):
                raise

        #os.chdir(analysis_dir)
        dictionary = corpora.Dictionary(self.all_tokens)
        corpus = [dictionary.doc2bow(text) for text in self.all_tokens]
        pickle.dump(corpus, open(os.path.join(saving_dir,'corpus.pkl'), 'wb'))
        dictionary.save(os.path.join(saving_dir,'dictionary.gensim'))
        return corpus, dictionary

#%% Topic models generation:
#Goes to Util
class Topic_modeling(object):
    def __init__(self, num_topics, saving_dir):
        
        
        self.num_topics = num_topics
        self.saving_dir = saving_dir
        try:
            os.makedirs(self.saving_dir)
        except OSError:
            if not os.path.isdir(self.saving_dir):
                raise
    def Genism_LDA(self, corpus, dictionary):
        #saving_dir= '/home/brain/AiTalents/Notebooks/Genism/'
            
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = self.num_topics, id2word=dictionary, passes=15)
        ldamodel.save(os.path.join(self.saving_dir,'LDA_model' + str(self.num_topics)+ '.gensim'))
        return ldamodel
    def Genism_LSI(self, corpus, dictionary):
        lsimodel = gensim.models.lsimodel.LsiModel(corpus, num_topics = self.num_topics, id2word=dictionary)
        lsimodel.save(os.path.join(self.saving_dir,'LSI_model' + str(self.num_topics)+ '.gensim'))
        return lsimodel
    def Sklearn_LDA(self, Feature_X):
        ldamodel = LatentDirichletAllocation()
        ldamodel.fit(Feature_X)
        Z = ldamodel.transform(Feature_X) # size: #term X #Component    >> For word Cloud
        # How about the graph connections?   componennts_ : size: #comps X #Articles  >> each column gives weights.
        U = ldamodel.components_    
        return ldamodel, Z, U 

            
#topics = ldamodel.print_topics(num_words=5)
#for topic in topics:
#    print(topic)

#%% Figures:

#Goes to Util
class Genism_plotting(object):
    def __init__(self,Figure_saving_dir):
        self.Figure_saving_dir = Figure_saving_dir
        try:
            os.makedirs(self.Figure_saving_dir)
        except OSError:
            if not os.path.isdir(self.Figure_saving_dir):
                raise
    def Convert(self, tup, di): 
        di = dict(tup) 
        return di

    def Reverse(self, tuples): 
        new_tup = () 
        for k in reversed(tuples): 
            new_tup = new_tup + (k,) 
        return new_tup 

    def pyLDA_Dashboard(self,ldamodel, corpus, dictionary,html_name, sort_topics=False):
        
        lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=sort_topics)
        pyLDAvis.save_html(lda_display, os.path.join(self.Figure_saving_dir,html_name + '_Dashboard.html'))
        #pyLDAvis.display(lda_display)
        return lda_display
    def Genism_word_cloud(self,html_name, ldamodel):
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

        cloud = WordCloud(background_color='#D8D8D8',
                          width=2500,
                          height=1800,
                          max_words=10,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)
        
        topic_word_box = ldamodel.show_topics(num_topics= ldamodel.num_topics, num_words= np.int(ldamodel.num_terms/100), formatted=False)
        # For each topic:
        topic_word_box_R= []
        for i in np.arange(ldamodel.num_topics):
            temp=[]
            temp = list(self.Reverse(topic_word_box[i][1]))
            topic_word_box_R.append(temp)
        
        D = ldamodel.num_topics
        CL = round(math.sqrt(D))
        RW = CL
        while RW * CL < D:
            RW += 1
        fig, axes = plt.subplots(RW, CL, figsize=(10,10), sharex=True, sharey=True)
        plt.style.context('grayscale')
        fig.patch.set_facecolor('#D8D8D8')

        Topic_names = ['Reporting', 'Investment', 'Risk_Management','Accounting', 'Payments_and_Settlement']
        #['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']
        #
        for i, ax in enumerate(axes.flatten()):
            if i >= D:
                axes.flatten()[i].axis('off')
            else:
            
                fig.add_subplot(ax)
                #topic_words = dict(topics[i][1])
                topic_words={}
                print(i)
                topic_words = self.Convert(topic_word_box_R[i], topic_words)
                cloud.generate_from_frequencies(topic_words, max_font_size=300)
                plt.gca().imshow(cloud)
                
                plt.gca().set_title(Topic_names[i] , fontdict=dict(size=16))
                #Topic_names.append('Topic_' + str(i +1))
                plt.gca().axis('off')


        plt.subplots_adjust(wspace=0, hspace=0)
        
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        
        #fig.patch.set_alpha(0.7)
        #plt.gca().set_facecolor('tab:gray')
        #cloud.to_file(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'))
        fig.savefig(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'), transparent=True)
        
        return 
                
class Genism_Graph(object):
    def __init__(self, Figure_saving_dir='', verbose = True):
        
        self.Figure_saving_dir = Figure_saving_dir
        self.verbose = verbose
        
    def Genism_networkX_graph(self, df, ldamodel):
        Article_names = df['article_title'].values
        
        Topic_names = ['Topic_'+ str(i) for i in range(1, len(ldamodel.show_topics())+1)]
        nx_graph = nx.Graph()
        Nodes = Topic_names + list(Article_names)
        # Add node for each Topic
        for T in Nodes:
            nx_graph.add_node(T)

        i = 0   
        # add edges
        for a in range(0,len(Article_names)):
            temp = ldamodel.get_document_topics(corpus[a])
            for c in range(0, len(temp)):
                if self.verbose:
                    
                    print('C: ', c)
                    print(len(ldamodel.get_document_topics(corpus[a])))

                T = Topic_names[temp[c][0]]
                if self.verbose:
                    print('T, A: ',T, Article_names[a])
                i += 1
                if self.verbose:
                    print('i:', i)
                # Only add edge if the count is positive

                nx_graph.add_edge(Article_names[a],T, weight = temp[c][1])
                if self.verbose:
                    print('Done_edge')
        return nx_graph
    def plot_Genism_networkX_graph(self,df, ldamodel, to_bundel=True ):
        nx_graph = self.Genism_networkX_graph(df, ldamodel)
        #Create the graph using networkX data:
        act_names = []
        kwargs = dict(width=800, height=800, xaxis=None, yaxis=None)
        opts.defaults(opts.Nodes(**kwargs), opts.Graph(**kwargs))
        
        #colors = ['#000000']+hv.Cycle('Category20').values
        topic_graph = hv.Graph.from_networkx(nx_graph, nx.layout.fruchterman_reingold_layout, k=1)
        topic_graph.nodes.data['act'] = topic_graph.nodes.data['index']
        for i in np.arange(topic_graph.nodes.data.shape[0]):
            if topic_graph.nodes.data.loc[i, 'act'] not in ['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']:
                topic_graph.nodes.data.loc[i, 'act'] = str(topic_graph.nodes.data.loc[i, 'act'].split('_')[1])
                
        
        tmp2= topic_graph.nodes.data[~topic_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        #colors = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*np.arange(len(tmp)), 20+2*np.arange(len(tmp)))]
        
        if to_bundel:

            bundled = bundle_graph(topic_graph)
            #.opts(opts.Nodes(color='act', size=10, width=1000, cmap=colors, legend_position='right'))
#opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            #
            return bundled, act_names, article_names
        else:
            return topic_graph, act_names, article_names
        #.opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            
    
    def subselect_Genism_single_Topic_graph(self, Topic_id, topic_graph):
        
        Sub_graph = topic_graph.select(start=Topic_id).opts(node_fill_color='blue', edge_line_width=hv.dim('weight'), edge_line_color='red')
        tmp2= Sub_graph.nodes.data[~Sub_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        
        return Sub_graph, act_names, article_names
    def subselect_Genism_single_Article_graph(self, Article_title, topic_graph, circular_layout=True):
        
        D = topic_graph.select(end=Article_title).opts(node_fill_color='white', edge_line_width=-1*np.log10(hv.dim('weight')))
        
        if circular_layout:
            
            F = layout_nodes(D)
            labels = hv.Labels(F.nodes, ['x', 'y'], 'index')

            return (F * labels.opts(text_font_size='10pt', text_color='black', bgcolor='white'))
        else: 
            
            return D#Goes to Util
class Text_parsing(object):
    def __init__(self):
        
        return 

    def parse_article(self, article, section_title, section_subtitle, act_name):
        article_title = article.get("TITLE")
        article_subtitle = article.get("SUBTITLE")
        article_content = article.get("CONTENT")
        assert article_title is not None
        assert article_subtitle is not None
        assert article_content is not None
        article = {"act_name":act_name,
                   "section_title": section_title,
                   "section_subtitle": section_subtitle,
                   "article_title": article_title + '_' + act_name,# + article_subtitle,
                   "article_subtitle":article_subtitle,
                   "article_content": article_content,
                  }
        return article

    def parse_section(self, section, collection=None, title=None, subtitle=None, act_name = ''):
        articles = [] if collection is None else collection
        if isinstance(section, list):
            for sec in section:
                articles = self.parse_section(sec, collection=articles,
                                         title=title, subtitle=subtitle, act_name=act_name)
            return articles
        section_title = section.get("TITLE") if title is None else f"{title}_{section.get('TITLE')}"
        section_subtitle = section.get("SUBTITLE") if title is None else f"{subtitle}_{section.get('SUBTITLE')}"

        # Only iterate if it contains a list of articles
        if "ARTICLE" in section:
            for art in section.get("ARTICLE",[]):
                article_entry = self.parse_article(art, section_title=section_title, section_subtitle=section_subtitle, act_name=act_name)
                articles.append(article_entry)
            return articles
        return self.parse_section(section["SECTION"], collection=articles,
                                  title=section_title, subtitle=section_subtitle, act_name=act_name)


    def parse_articles(self, json_data, act_name=''):
        articles = []
        sections = json_data['DOCUMENT']['SECTION']
        for i, section in enumerate(sections):
            if isinstance(section, list):
                for sec in section:
                    section_data = self.parse_section(sec,act_name=act_name)
                    articles.extend(section_data)
            else:
                section_data = self.parse_section(section,act_name=act_name)
                articles.extend(section_data)
        return pd.DataFrame(articles)

    def Read_in_Json_from_text(self, Orig_dir ='', all_jason_list_path =''):
        with open(all_jason_list_path) as fp: 
            Lines = fp.readlines() 
            files =[]
            acts =[]
            for line in Lines: 
                with open(os.path.join(Orig_dir, line.strip()), "r") as f:
                    data_1 = json.load(f)
                files.append(data_1)
                acts.append(re.sub('\.json$', '', line.strip()))
        return files, acts
    
    def merge_replicated_entries(self, df, matching_coloumn_name='', merging_coloumn_name=''):
        K = df[df.duplicated(subset=[matching_coloumn_name], keep=False)]
        df = df[~df.duplicated(subset=[matching_coloumn_name], keep=False)]

        q = K.article_title.unique()
        j =1
        for i in q:
            P =K.loc[K[matching_coloumn_name] == q[0]].reset_index(drop=True)
            temp_df = P.iloc[[0]]
            for j in range(1, P.shape[0]):
                temp_df.loc[0, merging_coloumn_name] = temp_df.loc[0, merging_coloumn_name]+ P.loc[j, merging_coloumn_name]
            df = df.append(temp_df, ignore_index=True, sort=False)

        return df
#%% preprocess text
# Goes to Util:
class Text_cleaning(object):
    def __init__(self, additional_stopwords_FullPath):
        stop_words_nltk = set(stopwords.words('english'))
        stopwords_final = set(w.rstrip() for w in open(additional_stopwords_FullPath))#'/home/brain/AiTalents/Notebooks/stopwords.txt'))
        stopwords_final = set([s.replace('\n', '') for s in stopwords_final])
        stopwords_final = stopwords_final.union(stop_words_nltk)
        self.stopwords_final = stopwords_final
    def clean_html(self, raw_text):
        cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        cleantext = re.sub(cleanr, '', raw_text)
        return cleantext

    def remove_URL(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'',text)

    
    def my_tokenizer(self, s):
        s = s.lower() # downcase
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
        tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
        tokens = [t for t in tokens if t not in string.punctuation] # remove punctuation        
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove any digits, i.e. "3rd edition"
        return tokens

    def preprocess_text(self, text):
        temp_text = self.remove_URL(text)
        no_html = self.clean_html(temp_text)
        return self.my_tokenizer(no_html)
    def Word_map_generation(self, df):
        #put clean Text into an array:
        contents = df["article_content"].values
        word_index_map = {}
        current_index = 0
        all_tokens = []
        all_content = []
        index_word_map = []
        error_count = 0
        for content in contents:
            try:
                content = content.encode('ascii', 'ignore').decode('utf-8') # this will throw exception if bad characters
                all_content.append(content)
                tokens = self.preprocess_text(content)
                all_tokens.append(tokens)
                for token in tokens:
                    if token not in word_index_map:
                        word_index_map[token] = current_index
                        current_index += 1
                        index_word_map.append(token)
            except Exception as e:
                print(e)
                print(title)
                error_count += 1
        return word_index_map, index_word_map,all_tokens, contents, error_count, self.stopwords_final


    def sklearn_Countvectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
                
        tf_vectorizer = CountVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                        token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                        tokenizer= self.preprocess_text)

        dtm_tf = tf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return tf_vectorizer, dtm_tf ## This is already a sparse feature matrix that could be fed into lda or SVD.
    def sklearn_TfidfVectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
        
        Tfidf_vectorizer = TfidfVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                           token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                           tokenizer= self.preprocess_text)

        dtm_Tfidf = Tfidf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return Tfidf_vectorizer, dtm_Tfidf ## This is already a sparse feature matrix that could be fed into lda or SVD.
            
    
    
#%% Feature generation:
#Goes to Util
class Feature_design(object):
    def __init__(self, word_index_map, all_tokens):
        self.word_index_map = word_index_map
        self.all_tokens = all_tokens
    # now let's create our input matrices - just indicator variables for this example - works better than proportions
    def tokens_to_vector(self, tokens):
        x = np.zeros(len(self.word_index_map))
        for t in tokens:
            i = self.word_index_map[t]
            x[i] = 1
        return x
    
    def Feature_matrix_generation(self):
        N = len(self.all_tokens)
        D = len(self.word_index_map)
        X = np.zeros((D, N)) # terms will go along rows, documents along columns
        i = 0
        for tokens in all_tokens:
            X[:,i] = self.tokens_to_vector(tokens)
            i += 1
        return X
    def Genism_Feature_matrix_generation(self, saving_dir):
        #analysis_dir= '/home/brain/AiTalents/Notebooks/Genism/'
        try:
            os.makedirs(saving_dir)
        except OSError:
            if not os.path.isdir(saving_dir):
                raise

        #os.chdir(analysis_dir)
        dictionary = corpora.Dictionary(self.all_tokens)
        corpus = [dictionary.doc2bow(text) for text in self.all_tokens]
        pickle.dump(corpus, open(os.path.join(saving_dir,'corpus.pkl'), 'wb'))
        dictionary.save(os.path.join(saving_dir,'dictionary.gensim'))
        return corpus, dictionary

#%% Topic models generation:
#Goes to Util
class Topic_modeling(object):
    def __init__(self, num_topics, saving_dir):
        
        
        self.num_topics = num_topics
        self.saving_dir = saving_dir
        try:
            os.makedirs(self.saving_dir)
        except OSError:
            if not os.path.isdir(self.saving_dir):
                raise
    def Genism_LDA(self, corpus, dictionary):
        #saving_dir= '/home/brain/AiTalents/Notebooks/Genism/'
            
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = self.num_topics, id2word=dictionary, passes=15)
        ldamodel.save(os.path.join(self.saving_dir,'LDA_model' + str(self.num_topics)+ '.gensim'))
        return ldamodel
    def Genism_LSI(self, corpus, dictionary):
        lsimodel = gensim.models.lsimodel.LsiModel(corpus, num_topics = self.num_topics, id2word=dictionary)
        lsimodel.save(os.path.join(self.saving_dir,'LSI_model' + str(self.num_topics)+ '.gensim'))
        return lsimodel
    def Sklearn_LDA(self, Feature_X):
        ldamodel = LatentDirichletAllocation()
        ldamodel.fit(Feature_X)
        Z = ldamodel.transform(Feature_X) # size: #term X #Component    >> For word Cloud
        # How about the graph connections?   componennts_ : size: #comps X #Articles  >> each column gives weights.
        U = ldamodel.components_    
        return ldamodel, Z, U 

            
#topics = ldamodel.print_topics(num_words=5)
#for topic in topics:
#    print(topic)

#%% Figures:

#Goes to Util
class Genism_plotting(object):
    def __init__(self,Figure_saving_dir):
        self.Figure_saving_dir = Figure_saving_dir
        try:
            os.makedirs(self.Figure_saving_dir)
        except OSError:
            if not os.path.isdir(self.Figure_saving_dir):
                raise
    def Convert(self, tup, di): 
        di = dict(tup) 
        return di

    def Reverse(self, tuples): 
        new_tup = () 
        for k in reversed(tuples): 
            new_tup = new_tup + (k,) 
        return new_tup 

    def pyLDA_Dashboard(self,ldamodel, corpus, dictionary,html_name, sort_topics=False):
        
        lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=sort_topics)
        pyLDAvis.save_html(lda_display, os.path.join(self.Figure_saving_dir,html_name + '_Dashboard.html'))
        #pyLDAvis.display(lda_display)
        return lda_display
    def Genism_word_cloud(self,html_name, ldamodel):
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

        cloud = WordCloud(background_color='#D8D8D8',
                          width=2500,
                          height=1800,
                          max_words=10,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)
        
        topic_word_box = ldamodel.show_topics(num_topics= ldamodel.num_topics, num_words= np.int(ldamodel.num_terms/100), formatted=False)
        # For each topic:
        topic_word_box_R= []
        for i in np.arange(ldamodel.num_topics):
            temp=[]
            temp = list(self.Reverse(topic_word_box[i][1]))
            topic_word_box_R.append(temp)
        
        D = ldamodel.num_topics
        CL = round(math.sqrt(D))
        RW = CL
        while RW * CL < D:
            RW += 1
        fig, axes = plt.subplots(RW, CL, figsize=(10,10), sharex=True, sharey=True)
        plt.style.context('grayscale')
        fig.patch.set_facecolor('#D8D8D8')

        Topic_names = ['Reporting', 'Investment', 'Risk_Management','Accounting', 'Payments_and_Settlement']
        #['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']
        #
        for i, ax in enumerate(axes.flatten()):
            if i >= D:
                axes.flatten()[i].axis('off')
            else:
            
                fig.add_subplot(ax)
                #topic_words = dict(topics[i][1])
                topic_words={}
                print(i)
                topic_words = self.Convert(topic_word_box_R[i], topic_words)
                cloud.generate_from_frequencies(topic_words, max_font_size=300)
                plt.gca().imshow(cloud)
                
                plt.gca().set_title(Topic_names[i] , fontdict=dict(size=16))
                #Topic_names.append('Topic_' + str(i +1))
                plt.gca().axis('off')


        plt.subplots_adjust(wspace=0, hspace=0)
        
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        
        #fig.patch.set_alpha(0.7)
        #plt.gca().set_facecolor('tab:gray')
        #cloud.to_file(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'))
        fig.savefig(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'), transparent=True)
        
        return 
                
class Genism_Graph(object):
    def __init__(self, Figure_saving_dir='', verbose = True):
        
        self.Figure_saving_dir = Figure_saving_dir
        self.verbose = verbose
        
    def Genism_networkX_graph(self, df, ldamodel):
        Article_names = df['article_title'].values
        
        Topic_names = ['Topic_'+ str(i) for i in range(1, len(ldamodel.show_topics())+1)]
        nx_graph = nx.Graph()
        Nodes = Topic_names + list(Article_names)
        # Add node for each Topic
        for T in Nodes:
            nx_graph.add_node(T)

        i = 0   
        # add edges
        for a in range(0,len(Article_names)):
            temp = ldamodel.get_document_topics(corpus[a])
            for c in range(0, len(temp)):
                if self.verbose:
                    
                    print('C: ', c)
                    print(len(ldamodel.get_document_topics(corpus[a])))

                T = Topic_names[temp[c][0]]
                if self.verbose:
                    print('T, A: ',T, Article_names[a])
                i += 1
                if self.verbose:
                    print('i:', i)
                # Only add edge if the count is positive

                nx_graph.add_edge(Article_names[a],T, weight = temp[c][1])
                if self.verbose:
                    print('Done_edge')
        return nx_graph
    def plot_Genism_networkX_graph(self,df, ldamodel, to_bundel=True ):
        nx_graph = self.Genism_networkX_graph(df, ldamodel)
        #Create the graph using networkX data:
        act_names = []
        kwargs = dict(width=800, height=800, xaxis=None, yaxis=None)
        opts.defaults(opts.Nodes(**kwargs), opts.Graph(**kwargs))
        
        #colors = ['#000000']+hv.Cycle('Category20').values
        topic_graph = hv.Graph.from_networkx(nx_graph, nx.layout.fruchterman_reingold_layout, k=1)
        topic_graph.nodes.data['act'] = topic_graph.nodes.data['index']
        for i in np.arange(topic_graph.nodes.data.shape[0]):
            if topic_graph.nodes.data.loc[i, 'act'] not in ['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']:
                topic_graph.nodes.data.loc[i, 'act'] = str(topic_graph.nodes.data.loc[i, 'act'].split('_')[1])
                
        
        tmp2= topic_graph.nodes.data[~topic_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        #colors = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*np.arange(len(tmp)), 20+2*np.arange(len(tmp)))]
        
        if to_bundel:

            bundled = bundle_graph(topic_graph)
            #.opts(opts.Nodes(color='act', size=10, width=1000, cmap=colors, legend_position='right'))
#opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            #
            return bundled, act_names, article_names
        else:
            return topic_graph, act_names, article_names
        #.opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            
    
    def subselect_Genism_single_Topic_graph(self, Topic_id, topic_graph):
        
        Sub_graph = topic_graph.select(start=Topic_id).opts(node_fill_color='blue', edge_line_width=hv.dim('weight'), edge_line_color='red')
        tmp2= Sub_graph.nodes.data[~Sub_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        
        return Sub_graph, act_names, article_names
    def subselect_Genism_single_Article_graph(self, Article_title, topic_graph, circular_layout=True):
        
        D = topic_graph.select(end=Article_title).opts(node_fill_color='white', edge_line_width=-1*np.log10(hv.dim('weight')))
        
        if circular_layout:
            
            F = layout_nodes(D)
            labels = hv.Labels(F.nodes, ['x', 'y'], 'index')

            return (F * labels.opts(text_font_size='10pt', text_color='black', bgcolor='white'))
        else: 
            
            return D#Goes to Util
class Text_parsing(object):
    def __init__(self):
        
        return 

    def parse_article(self, article, section_title, section_subtitle, act_name):
        article_title = article.get("TITLE")
        article_subtitle = article.get("SUBTITLE")
        article_content = article.get("CONTENT")
        assert article_title is not None
        assert article_subtitle is not None
        assert article_content is not None
        article = {"act_name":act_name,
                   "section_title": section_title,
                   "section_subtitle": section_subtitle,
                   "article_title": article_title + '_' + act_name,# + article_subtitle,
                   "article_subtitle":article_subtitle,
                   "article_content": article_content,
                  }
        return article

    def parse_section(self, section, collection=None, title=None, subtitle=None, act_name = ''):
        articles = [] if collection is None else collection
        if isinstance(section, list):
            for sec in section:
                articles = self.parse_section(sec, collection=articles,
                                         title=title, subtitle=subtitle, act_name=act_name)
            return articles
        section_title = section.get("TITLE") if title is None else f"{title}_{section.get('TITLE')}"
        section_subtitle = section.get("SUBTITLE") if title is None else f"{subtitle}_{section.get('SUBTITLE')}"

        # Only iterate if it contains a list of articles
        if "ARTICLE" in section:
            for art in section.get("ARTICLE",[]):
                article_entry = self.parse_article(art, section_title=section_title, section_subtitle=section_subtitle, act_name=act_name)
                articles.append(article_entry)
            return articles
        return self.parse_section(section["SECTION"], collection=articles,
                                  title=section_title, subtitle=section_subtitle, act_name=act_name)


    def parse_articles(self, json_data, act_name=''):
        articles = []
        sections = json_data['DOCUMENT']['SECTION']
        for i, section in enumerate(sections):
            if isinstance(section, list):
                for sec in section:
                    section_data = self.parse_section(sec,act_name=act_name)
                    articles.extend(section_data)
            else:
                section_data = self.parse_section(section,act_name=act_name)
                articles.extend(section_data)
        return pd.DataFrame(articles)

    def Read_in_Json_from_text(self, Orig_dir ='', all_jason_list_path =''):
        with open(all_jason_list_path) as fp: 
            Lines = fp.readlines() 
            files =[]
            acts =[]
            for line in Lines: 
                with open(os.path.join(Orig_dir, line.strip()), "r") as f:
                    data_1 = json.load(f)
                files.append(data_1)
                acts.append(re.sub('\.json$', '', line.strip()))
        return files, acts
    
    def merge_replicated_entries(self, df, matching_coloumn_name='', merging_coloumn_name=''):
        K = df[df.duplicated(subset=[matching_coloumn_name], keep=False)]
        df = df[~df.duplicated(subset=[matching_coloumn_name], keep=False)]

        q = K.article_title.unique()
        j =1
        for i in q:
            P =K.loc[K[matching_coloumn_name] == q[0]].reset_index(drop=True)
            temp_df = P.iloc[[0]]
            for j in range(1, P.shape[0]):
                temp_df.loc[0, merging_coloumn_name] = temp_df.loc[0, merging_coloumn_name]+ P.loc[j, merging_coloumn_name]
            df = df.append(temp_df, ignore_index=True, sort=False)

        return df
#%% preprocess text
# Goes to Util:
class Text_cleaning(object):
    def __init__(self, additional_stopwords_FullPath):
        stop_words_nltk = set(stopwords.words('english'))
        stopwords_final = set(w.rstrip() for w in open(additional_stopwords_FullPath))#'/home/brain/AiTalents/Notebooks/stopwords.txt'))
        stopwords_final = set([s.replace('\n', '') for s in stopwords_final])
        stopwords_final = stopwords_final.union(stop_words_nltk)
        self.stopwords_final = stopwords_final
    def clean_html(self, raw_text):
        cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        cleantext = re.sub(cleanr, '', raw_text)
        return cleantext

    def remove_URL(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'',text)

    
    def my_tokenizer(self, s):
        s = s.lower() # downcase
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
        tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
        tokens = [t for t in tokens if t not in string.punctuation] # remove punctuation        
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove any digits, i.e. "3rd edition"
        return tokens

    def preprocess_text(self, text):
        temp_text = self.remove_URL(text)
        no_html = self.clean_html(temp_text)
        return self.my_tokenizer(no_html)
    def Word_map_generation(self, df):
        #put clean Text into an array:
        contents = df["article_content"].values
        word_index_map = {}
        current_index = 0
        all_tokens = []
        all_content = []
        index_word_map = []
        error_count = 0
        for content in contents:
            try:
                content = content.encode('ascii', 'ignore').decode('utf-8') # this will throw exception if bad characters
                all_content.append(content)
                tokens = self.preprocess_text(content)
                all_tokens.append(tokens)
                for token in tokens:
                    if token not in word_index_map:
                        word_index_map[token] = current_index
                        current_index += 1
                        index_word_map.append(token)
            except Exception as e:
                print(e)
                print(title)
                error_count += 1
        return word_index_map, index_word_map,all_tokens, contents, error_count, self.stopwords_final


    def sklearn_Countvectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
                
        tf_vectorizer = CountVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                        token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                        tokenizer= self.preprocess_text)

        dtm_tf = tf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return tf_vectorizer, dtm_tf ## This is already a sparse feature matrix that could be fed into lda or SVD.
    def sklearn_TfidfVectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
        
        Tfidf_vectorizer = TfidfVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                           token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                           tokenizer= self.preprocess_text)

        dtm_Tfidf = Tfidf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return Tfidf_vectorizer, dtm_Tfidf ## This is already a sparse feature matrix that could be fed into lda or SVD.
            
    
    
#%% Feature generation:
#Goes to Util
class Feature_design(object):
    def __init__(self, word_index_map, all_tokens):
        self.word_index_map = word_index_map
        self.all_tokens = all_tokens
    # now let's create our input matrices - just indicator variables for this example - works better than proportions
    def tokens_to_vector(self, tokens):
        x = np.zeros(len(self.word_index_map))
        for t in tokens:
            i = self.word_index_map[t]
            x[i] = 1
        return x
    
    def Feature_matrix_generation(self):
        N = len(self.all_tokens)
        D = len(self.word_index_map)
        X = np.zeros((D, N)) # terms will go along rows, documents along columns
        i = 0
        for tokens in all_tokens:
            X[:,i] = self.tokens_to_vector(tokens)
            i += 1
        return X
    def Genism_Feature_matrix_generation(self, saving_dir):
        #analysis_dir= '/home/brain/AiTalents/Notebooks/Genism/'
        try:
            os.makedirs(saving_dir)
        except OSError:
            if not os.path.isdir(saving_dir):
                raise

        #os.chdir(analysis_dir)
        dictionary = corpora.Dictionary(self.all_tokens)
        corpus = [dictionary.doc2bow(text) for text in self.all_tokens]
        pickle.dump(corpus, open(os.path.join(saving_dir,'corpus.pkl'), 'wb'))
        dictionary.save(os.path.join(saving_dir,'dictionary.gensim'))
        return corpus, dictionary

#%% Topic models generation:
#Goes to Util
class Topic_modeling(object):
    def __init__(self, num_topics, saving_dir):
        
        
        self.num_topics = num_topics
        self.saving_dir = saving_dir
        try:
            os.makedirs(self.saving_dir)
        except OSError:
            if not os.path.isdir(self.saving_dir):
                raise
    def Genism_LDA(self, corpus, dictionary):
        #saving_dir= '/home/brain/AiTalents/Notebooks/Genism/'
            
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = self.num_topics, id2word=dictionary, passes=15)
        ldamodel.save(os.path.join(self.saving_dir,'LDA_model' + str(self.num_topics)+ '.gensim'))
        return ldamodel
    def Genism_LSI(self, corpus, dictionary):
        lsimodel = gensim.models.lsimodel.LsiModel(corpus, num_topics = self.num_topics, id2word=dictionary)
        lsimodel.save(os.path.join(self.saving_dir,'LSI_model' + str(self.num_topics)+ '.gensim'))
        return lsimodel
    def Sklearn_LDA(self, Feature_X):
        ldamodel = LatentDirichletAllocation()
        ldamodel.fit(Feature_X)
        Z = ldamodel.transform(Feature_X) # size: #term X #Component    >> For word Cloud
        # How about the graph connections?   componennts_ : size: #comps X #Articles  >> each column gives weights.
        U = ldamodel.components_    
        return ldamodel, Z, U 

            
#topics = ldamodel.print_topics(num_words=5)
#for topic in topics:
#    print(topic)

#%% Figures:

#Goes to Util
class Genism_plotting(object):
    def __init__(self,Figure_saving_dir):
        self.Figure_saving_dir = Figure_saving_dir
        try:
            os.makedirs(self.Figure_saving_dir)
        except OSError:
            if not os.path.isdir(self.Figure_saving_dir):
                raise
    def Convert(self, tup, di): 
        di = dict(tup) 
        return di

    def Reverse(self, tuples): 
        new_tup = () 
        for k in reversed(tuples): 
            new_tup = new_tup + (k,) 
        return new_tup 

    def pyLDA_Dashboard(self,ldamodel, corpus, dictionary,html_name, sort_topics=False):
        
        lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=sort_topics)
        pyLDAvis.save_html(lda_display, os.path.join(self.Figure_saving_dir,html_name + '_Dashboard.html'))
        #pyLDAvis.display(lda_display)
        return lda_display
    def Genism_word_cloud(self,html_name, ldamodel):
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

        cloud = WordCloud(background_color='#D8D8D8',
                          width=2500,
                          height=1800,
                          max_words=10,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)
        
        topic_word_box = ldamodel.show_topics(num_topics= ldamodel.num_topics, num_words= np.int(ldamodel.num_terms/100), formatted=False)
        # For each topic:
        topic_word_box_R= []
        for i in np.arange(ldamodel.num_topics):
            temp=[]
            temp = list(self.Reverse(topic_word_box[i][1]))
            topic_word_box_R.append(temp)
        
        D = ldamodel.num_topics
        CL = round(math.sqrt(D))
        RW = CL
        while RW * CL < D:
            RW += 1
        fig, axes = plt.subplots(RW, CL, figsize=(10,10), sharex=True, sharey=True)
        plt.style.context('grayscale')
        fig.patch.set_facecolor('#D8D8D8')

        Topic_names = ['Reporting', 'Investment', 'Risk_Management','Accounting', 'Payments_and_Settlement']
        #['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']
        #
        for i, ax in enumerate(axes.flatten()):
            if i >= D:
                axes.flatten()[i].axis('off')
            else:
            
                fig.add_subplot(ax)
                #topic_words = dict(topics[i][1])
                topic_words={}
                print(i)
                topic_words = self.Convert(topic_word_box_R[i], topic_words)
                cloud.generate_from_frequencies(topic_words, max_font_size=300)
                plt.gca().imshow(cloud)
                
                plt.gca().set_title(Topic_names[i] , fontdict=dict(size=16))
                #Topic_names.append('Topic_' + str(i +1))
                plt.gca().axis('off')


        plt.subplots_adjust(wspace=0, hspace=0)
        
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        
        #fig.patch.set_alpha(0.7)
        #plt.gca().set_facecolor('tab:gray')
        #cloud.to_file(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'))
        fig.savefig(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'), transparent=True)
        
        return 
                
class Genism_Graph(object):
    def __init__(self, Figure_saving_dir='', verbose = True):
        
        self.Figure_saving_dir = Figure_saving_dir
        self.verbose = verbose
        
    def Genism_networkX_graph(self, df, ldamodel):
        Article_names = df['article_title'].values
        
        Topic_names = ['Topic_'+ str(i) for i in range(1, len(ldamodel.show_topics())+1)]
        nx_graph = nx.Graph()
        Nodes = Topic_names + list(Article_names)
        # Add node for each Topic
        for T in Nodes:
            nx_graph.add_node(T)

        i = 0   
        # add edges
        for a in range(0,len(Article_names)):
            temp = ldamodel.get_document_topics(corpus[a])
            for c in range(0, len(temp)):
                if self.verbose:
                    
                    print('C: ', c)
                    print(len(ldamodel.get_document_topics(corpus[a])))

                T = Topic_names[temp[c][0]]
                if self.verbose:
                    print('T, A: ',T, Article_names[a])
                i += 1
                if self.verbose:
                    print('i:', i)
                # Only add edge if the count is positive

                nx_graph.add_edge(Article_names[a],T, weight = temp[c][1])
                if self.verbose:
                    print('Done_edge')
        return nx_graph
    def plot_Genism_networkX_graph(self,df, ldamodel, to_bundel=True ):
        nx_graph = self.Genism_networkX_graph(df, ldamodel)
        #Create the graph using networkX data:
        act_names = []
        kwargs = dict(width=800, height=800, xaxis=None, yaxis=None)
        opts.defaults(opts.Nodes(**kwargs), opts.Graph(**kwargs))
        
        #colors = ['#000000']+hv.Cycle('Category20').values
        topic_graph = hv.Graph.from_networkx(nx_graph, nx.layout.fruchterman_reingold_layout, k=1)
        topic_graph.nodes.data['act'] = topic_graph.nodes.data['index']
        for i in np.arange(topic_graph.nodes.data.shape[0]):
            if topic_graph.nodes.data.loc[i, 'act'] not in ['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']:
                topic_graph.nodes.data.loc[i, 'act'] = str(topic_graph.nodes.data.loc[i, 'act'].split('_')[1])
                
        
        tmp2= topic_graph.nodes.data[~topic_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        #colors = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*np.arange(len(tmp)), 20+2*np.arange(len(tmp)))]
        
        if to_bundel:

            bundled = bundle_graph(topic_graph)
            #.opts(opts.Nodes(color='act', size=10, width=1000, cmap=colors, legend_position='right'))
#opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            #
            return bundled, act_names, article_names
        else:
            return topic_graph, act_names, article_names
        #.opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            
    
    def subselect_Genism_single_Topic_graph(self, Topic_id, topic_graph):
        
        Sub_graph = topic_graph.select(start=Topic_id).opts(node_fill_color='blue', edge_line_width=hv.dim('weight'), edge_line_color='red')
        tmp2= Sub_graph.nodes.data[~Sub_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        
        return Sub_graph, act_names, article_names
    def subselect_Genism_single_Article_graph(self, Article_title, topic_graph, circular_layout=True):
        
        D = topic_graph.select(end=Article_title).opts(node_fill_color='white', edge_line_width=-1*np.log10(hv.dim('weight')))
        
        if circular_layout:
            
            F = layout_nodes(D)
            labels = hv.Labels(F.nodes, ['x', 'y'], 'index')

            return (F * labels.opts(text_font_size='10pt', text_color='black', bgcolor='white'))
        else: 
            
            return D#Goes to Util
class Text_parsing(object):
    def __init__(self):
        
        return 

    def parse_article(self, article, section_title, section_subtitle, act_name):
        article_title = article.get("TITLE")
        article_subtitle = article.get("SUBTITLE")
        article_content = article.get("CONTENT")
        assert article_title is not None
        assert article_subtitle is not None
        assert article_content is not None
        article = {"act_name":act_name,
                   "section_title": section_title,
                   "section_subtitle": section_subtitle,
                   "article_title": article_title + '_' + act_name,# + article_subtitle,
                   "article_subtitle":article_subtitle,
                   "article_content": article_content,
                  }
        return article

    def parse_section(self, section, collection=None, title=None, subtitle=None, act_name = ''):
        articles = [] if collection is None else collection
        if isinstance(section, list):
            for sec in section:
                articles = self.parse_section(sec, collection=articles,
                                         title=title, subtitle=subtitle, act_name=act_name)
            return articles
        section_title = section.get("TITLE") if title is None else f"{title}_{section.get('TITLE')}"
        section_subtitle = section.get("SUBTITLE") if title is None else f"{subtitle}_{section.get('SUBTITLE')}"

        # Only iterate if it contains a list of articles
        if "ARTICLE" in section:
            for art in section.get("ARTICLE",[]):
                article_entry = self.parse_article(art, section_title=section_title, section_subtitle=section_subtitle, act_name=act_name)
                articles.append(article_entry)
            return articles
        return self.parse_section(section["SECTION"], collection=articles,
                                  title=section_title, subtitle=section_subtitle, act_name=act_name)


    def parse_articles(self, json_data, act_name=''):
        articles = []
        sections = json_data['DOCUMENT']['SECTION']
        for i, section in enumerate(sections):
            if isinstance(section, list):
                for sec in section:
                    section_data = self.parse_section(sec,act_name=act_name)
                    articles.extend(section_data)
            else:
                section_data = self.parse_section(section,act_name=act_name)
                articles.extend(section_data)
        return pd.DataFrame(articles)

    def Read_in_Json_from_text(self, Orig_dir ='', all_jason_list_path =''):
        with open(all_jason_list_path) as fp: 
            Lines = fp.readlines() 
            files =[]
            acts =[]
            for line in Lines: 
                with open(os.path.join(Orig_dir, line.strip()), "r") as f:
                    data_1 = json.load(f)
                files.append(data_1)
                acts.append(re.sub('\.json$', '', line.strip()))
        return files, acts
    
    def merge_replicated_entries(self, df, matching_coloumn_name='', merging_coloumn_name=''):
        K = df[df.duplicated(subset=[matching_coloumn_name], keep=False)]
        df = df[~df.duplicated(subset=[matching_coloumn_name], keep=False)]

        q = K.article_title.unique()
        j =1
        for i in q:
            P =K.loc[K[matching_coloumn_name] == q[0]].reset_index(drop=True)
            temp_df = P.iloc[[0]]
            for j in range(1, P.shape[0]):
                temp_df.loc[0, merging_coloumn_name] = temp_df.loc[0, merging_coloumn_name]+ P.loc[j, merging_coloumn_name]
            df = df.append(temp_df, ignore_index=True, sort=False)

        return df
#%% preprocess text
# Goes to Util:
class Text_cleaning(object):
    def __init__(self, additional_stopwords_FullPath):
        stop_words_nltk = set(stopwords.words('english'))
        stopwords_final = set(w.rstrip() for w in open(additional_stopwords_FullPath))#'/home/brain/AiTalents/Notebooks/stopwords.txt'))
        stopwords_final = set([s.replace('\n', '') for s in stopwords_final])
        stopwords_final = stopwords_final.union(stop_words_nltk)
        self.stopwords_final = stopwords_final
    def clean_html(self, raw_text):
        cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        cleantext = re.sub(cleanr, '', raw_text)
        return cleantext

    def remove_URL(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'',text)

    
    def my_tokenizer(self, s):
        s = s.lower() # downcase
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
        tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
        tokens = [t for t in tokens if t not in string.punctuation] # remove punctuation        
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove any digits, i.e. "3rd edition"
        return tokens

    def preprocess_text(self, text):
        temp_text = self.remove_URL(text)
        no_html = self.clean_html(temp_text)
        return self.my_tokenizer(no_html)
    def Word_map_generation(self, df):
        #put clean Text into an array:
        contents = df["article_content"].values
        word_index_map = {}
        current_index = 0
        all_tokens = []
        all_content = []
        index_word_map = []
        error_count = 0
        for content in contents:
            try:
                content = content.encode('ascii', 'ignore').decode('utf-8') # this will throw exception if bad characters
                all_content.append(content)
                tokens = self.preprocess_text(content)
                all_tokens.append(tokens)
                for token in tokens:
                    if token not in word_index_map:
                        word_index_map[token] = current_index
                        current_index += 1
                        index_word_map.append(token)
            except Exception as e:
                print(e)
                print(title)
                error_count += 1
        return word_index_map, index_word_map,all_tokens, contents, error_count, self.stopwords_final


    def sklearn_Countvectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
                
        tf_vectorizer = CountVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                        token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                        tokenizer= self.preprocess_text)

        dtm_tf = tf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return tf_vectorizer, dtm_tf ## This is already a sparse feature matrix that could be fed into lda or SVD.
    def sklearn_TfidfVectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
        
        Tfidf_vectorizer = TfidfVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                           token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                           tokenizer= self.preprocess_text)

        dtm_Tfidf = Tfidf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return Tfidf_vectorizer, dtm_Tfidf ## This is already a sparse feature matrix that could be fed into lda or SVD.
            
    
    
#%% Feature generation:
#Goes to Util
class Feature_design(object):
    def __init__(self, word_index_map, all_tokens):
        self.word_index_map = word_index_map
        self.all_tokens = all_tokens
    # now let's create our input matrices - just indicator variables for this example - works better than proportions
    def tokens_to_vector(self, tokens):
        x = np.zeros(len(self.word_index_map))
        for t in tokens:
            i = self.word_index_map[t]
            x[i] = 1
        return x
    
    def Feature_matrix_generation(self):
        N = len(self.all_tokens)
        D = len(self.word_index_map)
        X = np.zeros((D, N)) # terms will go along rows, documents along columns
        i = 0
        for tokens in all_tokens:
            X[:,i] = self.tokens_to_vector(tokens)
            i += 1
        return X
    def Genism_Feature_matrix_generation(self, saving_dir):
        #analysis_dir= '/home/brain/AiTalents/Notebooks/Genism/'
        try:
            os.makedirs(saving_dir)
        except OSError:
            if not os.path.isdir(saving_dir):
                raise

        #os.chdir(analysis_dir)
        dictionary = corpora.Dictionary(self.all_tokens)
        corpus = [dictionary.doc2bow(text) for text in self.all_tokens]
        pickle.dump(corpus, open(os.path.join(saving_dir,'corpus.pkl'), 'wb'))
        dictionary.save(os.path.join(saving_dir,'dictionary.gensim'))
        return corpus, dictionary

#%% Topic models generation:
#Goes to Util
class Topic_modeling(object):
    def __init__(self, num_topics, saving_dir):
        
        
        self.num_topics = num_topics
        self.saving_dir = saving_dir
        try:
            os.makedirs(self.saving_dir)
        except OSError:
            if not os.path.isdir(self.saving_dir):
                raise
    def Genism_LDA(self, corpus, dictionary):
        #saving_dir= '/home/brain/AiTalents/Notebooks/Genism/'
            
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = self.num_topics, id2word=dictionary, passes=15)
        ldamodel.save(os.path.join(self.saving_dir,'LDA_model' + str(self.num_topics)+ '.gensim'))
        return ldamodel
    def Genism_LSI(self, corpus, dictionary):
        lsimodel = gensim.models.lsimodel.LsiModel(corpus, num_topics = self.num_topics, id2word=dictionary)
        lsimodel.save(os.path.join(self.saving_dir,'LSI_model' + str(self.num_topics)+ '.gensim'))
        return lsimodel
    def Sklearn_LDA(self, Feature_X):
        ldamodel = LatentDirichletAllocation()
        ldamodel.fit(Feature_X)
        Z = ldamodel.transform(Feature_X) # size: #term X #Component    >> For word Cloud
        # How about the graph connections?   componennts_ : size: #comps X #Articles  >> each column gives weights.
        U = ldamodel.components_    
        return ldamodel, Z, U 

            
#topics = ldamodel.print_topics(num_words=5)
#for topic in topics:
#    print(topic)

#%% Figures:

#Goes to Util
class Genism_plotting(object):
    def __init__(self,Figure_saving_dir):
        self.Figure_saving_dir = Figure_saving_dir
        try:
            os.makedirs(self.Figure_saving_dir)
        except OSError:
            if not os.path.isdir(self.Figure_saving_dir):
                raise
    def Convert(self, tup, di): 
        di = dict(tup) 
        return di

    def Reverse(self, tuples): 
        new_tup = () 
        for k in reversed(tuples): 
            new_tup = new_tup + (k,) 
        return new_tup 

    def pyLDA_Dashboard(self,ldamodel, corpus, dictionary,html_name, sort_topics=False):
        
        lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=sort_topics)
        pyLDAvis.save_html(lda_display, os.path.join(self.Figure_saving_dir,html_name + '_Dashboard.html'))
        #pyLDAvis.display(lda_display)
        return lda_display
    def Genism_word_cloud(self,html_name, ldamodel):
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

        cloud = WordCloud(background_color='#D8D8D8',
                          width=2500,
                          height=1800,
                          max_words=10,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)
        
        topic_word_box = ldamodel.show_topics(num_topics= ldamodel.num_topics, num_words= np.int(ldamodel.num_terms/100), formatted=False)
        # For each topic:
        topic_word_box_R= []
        for i in np.arange(ldamodel.num_topics):
            temp=[]
            temp = list(self.Reverse(topic_word_box[i][1]))
            topic_word_box_R.append(temp)
        
        D = ldamodel.num_topics
        CL = round(math.sqrt(D))
        RW = CL
        while RW * CL < D:
            RW += 1
        fig, axes = plt.subplots(RW, CL, figsize=(10,10), sharex=True, sharey=True)
        plt.style.context('grayscale')
        fig.patch.set_facecolor('#D8D8D8')

        Topic_names = ['Reporting', 'Investment', 'Risk_Management','Accounting', 'Payments_and_Settlement']
        #['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']
        #
        for i, ax in enumerate(axes.flatten()):
            if i >= D:
                axes.flatten()[i].axis('off')
            else:
            
                fig.add_subplot(ax)
                #topic_words = dict(topics[i][1])
                topic_words={}
                print(i)
                topic_words = self.Convert(topic_word_box_R[i], topic_words)
                cloud.generate_from_frequencies(topic_words, max_font_size=300)
                plt.gca().imshow(cloud)
                
                plt.gca().set_title(Topic_names[i] , fontdict=dict(size=16))
                #Topic_names.append('Topic_' + str(i +1))
                plt.gca().axis('off')


        plt.subplots_adjust(wspace=0, hspace=0)
        
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        
        #fig.patch.set_alpha(0.7)
        #plt.gca().set_facecolor('tab:gray')
        #cloud.to_file(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'))
        fig.savefig(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'), transparent=True)
        
        return 
                
class Genism_Graph(object):
    def __init__(self, Figure_saving_dir='', verbose = True):
        
        self.Figure_saving_dir = Figure_saving_dir
        self.verbose = verbose
        
    def Genism_networkX_graph(self, df, ldamodel):
        Article_names = df['article_title'].values
        
        Topic_names = ['Topic_'+ str(i) for i in range(1, len(ldamodel.show_topics())+1)]
        nx_graph = nx.Graph()
        Nodes = Topic_names + list(Article_names)
        # Add node for each Topic
        for T in Nodes:
            nx_graph.add_node(T)

        i = 0   
        # add edges
        for a in range(0,len(Article_names)):
            temp = ldamodel.get_document_topics(corpus[a])
            for c in range(0, len(temp)):
                if self.verbose:
                    
                    print('C: ', c)
                    print(len(ldamodel.get_document_topics(corpus[a])))

                T = Topic_names[temp[c][0]]
                if self.verbose:
                    print('T, A: ',T, Article_names[a])
                i += 1
                if self.verbose:
                    print('i:', i)
                # Only add edge if the count is positive

                nx_graph.add_edge(Article_names[a],T, weight = temp[c][1])
                if self.verbose:
                    print('Done_edge')
        return nx_graph
    def plot_Genism_networkX_graph(self,df, ldamodel, to_bundel=True ):
        nx_graph = self.Genism_networkX_graph(df, ldamodel)
        #Create the graph using networkX data:
        act_names = []
        kwargs = dict(width=800, height=800, xaxis=None, yaxis=None)
        opts.defaults(opts.Nodes(**kwargs), opts.Graph(**kwargs))
        
        #colors = ['#000000']+hv.Cycle('Category20').values
        topic_graph = hv.Graph.from_networkx(nx_graph, nx.layout.fruchterman_reingold_layout, k=1)
        topic_graph.nodes.data['act'] = topic_graph.nodes.data['index']
        for i in np.arange(topic_graph.nodes.data.shape[0]):
            if topic_graph.nodes.data.loc[i, 'act'] not in ['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']:
                topic_graph.nodes.data.loc[i, 'act'] = str(topic_graph.nodes.data.loc[i, 'act'].split('_')[1])
                
        
        tmp2= topic_graph.nodes.data[~topic_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        #colors = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*np.arange(len(tmp)), 20+2*np.arange(len(tmp)))]
        
        if to_bundel:

            bundled = bundle_graph(topic_graph)
            #.opts(opts.Nodes(color='act', size=10, width=1000, cmap=colors, legend_position='right'))
#opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            #
            return bundled, act_names, article_names
        else:
            return topic_graph, act_names, article_names
        #.opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            
    
    def subselect_Genism_single_Topic_graph(self, Topic_id, topic_graph):
        
        Sub_graph = topic_graph.select(start=Topic_id).opts(node_fill_color='blue', edge_line_width=hv.dim('weight'), edge_line_color='red')
        tmp2= Sub_graph.nodes.data[~Sub_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        
        return Sub_graph, act_names, article_names
    def subselect_Genism_single_Article_graph(self, Article_title, topic_graph, circular_layout=True):
        
        D = topic_graph.select(end=Article_title).opts(node_fill_color='white', edge_line_width=-1*np.log10(hv.dim('weight')))
        
        if circular_layout:
            
            F = layout_nodes(D)
            labels = hv.Labels(F.nodes, ['x', 'y'], 'index')

            return (F * labels.opts(text_font_size='10pt', text_color='black', bgcolor='white'))
        else: 
            
            return D#Goes to Util
class Text_parsing(object):
    def __init__(self):
        
        return 

    def parse_article(self, article, section_title, section_subtitle, act_name):
        article_title = article.get("TITLE")
        article_subtitle = article.get("SUBTITLE")
        article_content = article.get("CONTENT")
        assert article_title is not None
        assert article_subtitle is not None
        assert article_content is not None
        article = {"act_name":act_name,
                   "section_title": section_title,
                   "section_subtitle": section_subtitle,
                   "article_title": article_title + '_' + act_name,# + article_subtitle,
                   "article_subtitle":article_subtitle,
                   "article_content": article_content,
                  }
        return article

    def parse_section(self, section, collection=None, title=None, subtitle=None, act_name = ''):
        articles = [] if collection is None else collection
        if isinstance(section, list):
            for sec in section:
                articles = self.parse_section(sec, collection=articles,
                                         title=title, subtitle=subtitle, act_name=act_name)
            return articles
        section_title = section.get("TITLE") if title is None else f"{title}_{section.get('TITLE')}"
        section_subtitle = section.get("SUBTITLE") if title is None else f"{subtitle}_{section.get('SUBTITLE')}"

        # Only iterate if it contains a list of articles
        if "ARTICLE" in section:
            for art in section.get("ARTICLE",[]):
                article_entry = self.parse_article(art, section_title=section_title, section_subtitle=section_subtitle, act_name=act_name)
                articles.append(article_entry)
            return articles
        return self.parse_section(section["SECTION"], collection=articles,
                                  title=section_title, subtitle=section_subtitle, act_name=act_name)


    def parse_articles(self, json_data, act_name=''):
        articles = []
        sections = json_data['DOCUMENT']['SECTION']
        for i, section in enumerate(sections):
            if isinstance(section, list):
                for sec in section:
                    section_data = self.parse_section(sec,act_name=act_name)
                    articles.extend(section_data)
            else:
                section_data = self.parse_section(section,act_name=act_name)
                articles.extend(section_data)
        return pd.DataFrame(articles)

    def Read_in_Json_from_text(self, Orig_dir ='', all_jason_list_path =''):
        with open(all_jason_list_path) as fp: 
            Lines = fp.readlines() 
            files =[]
            acts =[]
            for line in Lines: 
                with open(os.path.join(Orig_dir, line.strip()), "r") as f:
                    data_1 = json.load(f)
                files.append(data_1)
                acts.append(re.sub('\.json$', '', line.strip()))
        return files, acts
    
    def merge_replicated_entries(self, df, matching_coloumn_name='', merging_coloumn_name=''):
        K = df[df.duplicated(subset=[matching_coloumn_name], keep=False)]
        df = df[~df.duplicated(subset=[matching_coloumn_name], keep=False)]

        q = K.article_title.unique()
        j =1
        for i in q:
            P =K.loc[K[matching_coloumn_name] == q[0]].reset_index(drop=True)
            temp_df = P.iloc[[0]]
            for j in range(1, P.shape[0]):
                temp_df.loc[0, merging_coloumn_name] = temp_df.loc[0, merging_coloumn_name]+ P.loc[j, merging_coloumn_name]
            df = df.append(temp_df, ignore_index=True, sort=False)

        return df
#%% preprocess text
# Goes to Util:
class Text_cleaning(object):
    def __init__(self, additional_stopwords_FullPath):
        stop_words_nltk = set(stopwords.words('english'))
        stopwords_final = set(w.rstrip() for w in open(additional_stopwords_FullPath))#'/home/brain/AiTalents/Notebooks/stopwords.txt'))
        stopwords_final = set([s.replace('\n', '') for s in stopwords_final])
        stopwords_final = stopwords_final.union(stop_words_nltk)
        self.stopwords_final = stopwords_final
    def clean_html(self, raw_text):
        cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        cleantext = re.sub(cleanr, '', raw_text)
        return cleantext

    def remove_URL(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'',text)

    
    def my_tokenizer(self, s):
        s = s.lower() # downcase
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
        tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
        tokens = [t for t in tokens if t not in string.punctuation] # remove punctuation        
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove any digits, i.e. "3rd edition"
        return tokens

    def preprocess_text(self, text):
        temp_text = self.remove_URL(text)
        no_html = self.clean_html(temp_text)
        return self.my_tokenizer(no_html)
    def Word_map_generation(self, df):
        #put clean Text into an array:
        contents = df["article_content"].values
        word_index_map = {}
        current_index = 0
        all_tokens = []
        all_content = []
        index_word_map = []
        error_count = 0
        for content in contents:
            try:
                content = content.encode('ascii', 'ignore').decode('utf-8') # this will throw exception if bad characters
                all_content.append(content)
                tokens = self.preprocess_text(content)
                all_tokens.append(tokens)
                for token in tokens:
                    if token not in word_index_map:
                        word_index_map[token] = current_index
                        current_index += 1
                        index_word_map.append(token)
            except Exception as e:
                print(e)
                print(title)
                error_count += 1
        return word_index_map, index_word_map,all_tokens, contents, error_count, self.stopwords_final


    def sklearn_Countvectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
                
        tf_vectorizer = CountVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                        token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                        tokenizer= self.preprocess_text)

        dtm_tf = tf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return tf_vectorizer, dtm_tf ## This is already a sparse feature matrix that could be fed into lda or SVD.
    def sklearn_TfidfVectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
        
        Tfidf_vectorizer = TfidfVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                           token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                           tokenizer= self.preprocess_text)

        dtm_Tfidf = Tfidf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return Tfidf_vectorizer, dtm_Tfidf ## This is already a sparse feature matrix that could be fed into lda or SVD.
            
    
    
#%% Feature generation:
#Goes to Util
class Feature_design(object):
    def __init__(self, word_index_map, all_tokens):
        self.word_index_map = word_index_map
        self.all_tokens = all_tokens
    # now let's create our input matrices - just indicator variables for this example - works better than proportions
    def tokens_to_vector(self, tokens):
        x = np.zeros(len(self.word_index_map))
        for t in tokens:
            i = self.word_index_map[t]
            x[i] = 1
        return x
    
    def Feature_matrix_generation(self):
        N = len(self.all_tokens)
        D = len(self.word_index_map)
        X = np.zeros((D, N)) # terms will go along rows, documents along columns
        i = 0
        for tokens in all_tokens:
            X[:,i] = self.tokens_to_vector(tokens)
            i += 1
        return X
    def Genism_Feature_matrix_generation(self, saving_dir):
        #analysis_dir= '/home/brain/AiTalents/Notebooks/Genism/'
        try:
            os.makedirs(saving_dir)
        except OSError:
            if not os.path.isdir(saving_dir):
                raise

        #os.chdir(analysis_dir)
        dictionary = corpora.Dictionary(self.all_tokens)
        corpus = [dictionary.doc2bow(text) for text in self.all_tokens]
        pickle.dump(corpus, open(os.path.join(saving_dir,'corpus.pkl'), 'wb'))
        dictionary.save(os.path.join(saving_dir,'dictionary.gensim'))
        return corpus, dictionary

#%% Topic models generation:
#Goes to Util
class Topic_modeling(object):
    def __init__(self, num_topics, saving_dir):
        
        
        self.num_topics = num_topics
        self.saving_dir = saving_dir
        try:
            os.makedirs(self.saving_dir)
        except OSError:
            if not os.path.isdir(self.saving_dir):
                raise
    def Genism_LDA(self, corpus, dictionary):
        #saving_dir= '/home/brain/AiTalents/Notebooks/Genism/'
            
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = self.num_topics, id2word=dictionary, passes=15)
        ldamodel.save(os.path.join(self.saving_dir,'LDA_model' + str(self.num_topics)+ '.gensim'))
        return ldamodel
    def Genism_LSI(self, corpus, dictionary):
        lsimodel = gensim.models.lsimodel.LsiModel(corpus, num_topics = self.num_topics, id2word=dictionary)
        lsimodel.save(os.path.join(self.saving_dir,'LSI_model' + str(self.num_topics)+ '.gensim'))
        return lsimodel
    def Sklearn_LDA(self, Feature_X):
        ldamodel = LatentDirichletAllocation()
        ldamodel.fit(Feature_X)
        Z = ldamodel.transform(Feature_X) # size: #term X #Component    >> For word Cloud
        # How about the graph connections?   componennts_ : size: #comps X #Articles  >> each column gives weights.
        U = ldamodel.components_    
        return ldamodel, Z, U 

            
#topics = ldamodel.print_topics(num_words=5)
#for topic in topics:
#    print(topic)

#%% Figures:

#Goes to Util
class Genism_plotting(object):
    def __init__(self,Figure_saving_dir):
        self.Figure_saving_dir = Figure_saving_dir
        try:
            os.makedirs(self.Figure_saving_dir)
        except OSError:
            if not os.path.isdir(self.Figure_saving_dir):
                raise
    def Convert(self, tup, di): 
        di = dict(tup) 
        return di

    def Reverse(self, tuples): 
        new_tup = () 
        for k in reversed(tuples): 
            new_tup = new_tup + (k,) 
        return new_tup 

    def pyLDA_Dashboard(self,ldamodel, corpus, dictionary,html_name, sort_topics=False):
        
        lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=sort_topics)
        pyLDAvis.save_html(lda_display, os.path.join(self.Figure_saving_dir,html_name + '_Dashboard.html'))
        #pyLDAvis.display(lda_display)
        return lda_display
    def Genism_word_cloud(self,html_name, ldamodel):
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

        cloud = WordCloud(background_color='#D8D8D8',
                          width=2500,
                          height=1800,
                          max_words=10,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)
        
        topic_word_box = ldamodel.show_topics(num_topics= ldamodel.num_topics, num_words= np.int(ldamodel.num_terms/100), formatted=False)
        # For each topic:
        topic_word_box_R= []
        for i in np.arange(ldamodel.num_topics):
            temp=[]
            temp = list(self.Reverse(topic_word_box[i][1]))
            topic_word_box_R.append(temp)
        
        D = ldamodel.num_topics
        CL = round(math.sqrt(D))
        RW = CL
        while RW * CL < D:
            RW += 1
        fig, axes = plt.subplots(RW, CL, figsize=(10,10), sharex=True, sharey=True)
        plt.style.context('grayscale')
        fig.patch.set_facecolor('#D8D8D8')

        Topic_names = ['Reporting', 'Investment', 'Risk_Management','Accounting', 'Payments_and_Settlement']
        #['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']
        #
        for i, ax in enumerate(axes.flatten()):
            if i >= D:
                axes.flatten()[i].axis('off')
            else:
            
                fig.add_subplot(ax)
                #topic_words = dict(topics[i][1])
                topic_words={}
                print(i)
                topic_words = self.Convert(topic_word_box_R[i], topic_words)
                cloud.generate_from_frequencies(topic_words, max_font_size=300)
                plt.gca().imshow(cloud)
                
                plt.gca().set_title(Topic_names[i] , fontdict=dict(size=16))
                #Topic_names.append('Topic_' + str(i +1))
                plt.gca().axis('off')


        plt.subplots_adjust(wspace=0, hspace=0)
        
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        
        #fig.patch.set_alpha(0.7)
        #plt.gca().set_facecolor('tab:gray')
        #cloud.to_file(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'))
        fig.savefig(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'), transparent=True)
        
        return 
                
class Genism_Graph(object):
    def __init__(self, Figure_saving_dir='', verbose = True):
        
        self.Figure_saving_dir = Figure_saving_dir
        self.verbose = verbose
        
    def Genism_networkX_graph(self, df, ldamodel):
        Article_names = df['article_title'].values
        
        Topic_names = ['Topic_'+ str(i) for i in range(1, len(ldamodel.show_topics())+1)]
        nx_graph = nx.Graph()
        Nodes = Topic_names + list(Article_names)
        # Add node for each Topic
        for T in Nodes:
            nx_graph.add_node(T)

        i = 0   
        # add edges
        for a in range(0,len(Article_names)):
            temp = ldamodel.get_document_topics(corpus[a])
            for c in range(0, len(temp)):
                if self.verbose:
                    
                    print('C: ', c)
                    print(len(ldamodel.get_document_topics(corpus[a])))

                T = Topic_names[temp[c][0]]
                if self.verbose:
                    print('T, A: ',T, Article_names[a])
                i += 1
                if self.verbose:
                    print('i:', i)
                # Only add edge if the count is positive

                nx_graph.add_edge(Article_names[a],T, weight = temp[c][1])
                if self.verbose:
                    print('Done_edge')
        return nx_graph
    def plot_Genism_networkX_graph(self,df, ldamodel, to_bundel=True ):
        nx_graph = self.Genism_networkX_graph(df, ldamodel)
        #Create the graph using networkX data:
        act_names = []
        kwargs = dict(width=800, height=800, xaxis=None, yaxis=None)
        opts.defaults(opts.Nodes(**kwargs), opts.Graph(**kwargs))
        
        #colors = ['#000000']+hv.Cycle('Category20').values
        topic_graph = hv.Graph.from_networkx(nx_graph, nx.layout.fruchterman_reingold_layout, k=1)
        topic_graph.nodes.data['act'] = topic_graph.nodes.data['index']
        for i in np.arange(topic_graph.nodes.data.shape[0]):
            if topic_graph.nodes.data.loc[i, 'act'] not in ['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']:
                topic_graph.nodes.data.loc[i, 'act'] = str(topic_graph.nodes.data.loc[i, 'act'].split('_')[1])
                
        
        tmp2= topic_graph.nodes.data[~topic_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        #colors = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*np.arange(len(tmp)), 20+2*np.arange(len(tmp)))]
        
        if to_bundel:

            bundled = bundle_graph(topic_graph)
            #.opts(opts.Nodes(color='act', size=10, width=1000, cmap=colors, legend_position='right'))
#opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            #
            return bundled, act_names, article_names
        else:
            return topic_graph, act_names, article_names
        #.opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            
    
    def subselect_Genism_single_Topic_graph(self, Topic_id, topic_graph):
        
        Sub_graph = topic_graph.select(start=Topic_id).opts(node_fill_color='blue', edge_line_width=hv.dim('weight'), edge_line_color='red')
        tmp2= Sub_graph.nodes.data[~Sub_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        
        return Sub_graph, act_names, article_names
    def subselect_Genism_single_Article_graph(self, Article_title, topic_graph, circular_layout=True):
        
        D = topic_graph.select(end=Article_title).opts(node_fill_color='white', edge_line_width=-1*np.log10(hv.dim('weight')))
        
        if circular_layout:
            
            F = layout_nodes(D)
            labels = hv.Labels(F.nodes, ['x', 'y'], 'index')

            return (F * labels.opts(text_font_size='10pt', text_color='black', bgcolor='white'))
        else: 
            
            return D#Goes to Util
class Text_parsing(object):
    def __init__(self):
        
        return 

    def parse_article(self, article, section_title, section_subtitle, act_name):
        article_title = article.get("TITLE")
        article_subtitle = article.get("SUBTITLE")
        article_content = article.get("CONTENT")
        assert article_title is not None
        assert article_subtitle is not None
        assert article_content is not None
        article = {"act_name":act_name,
                   "section_title": section_title,
                   "section_subtitle": section_subtitle,
                   "article_title": article_title + '_' + act_name,# + article_subtitle,
                   "article_subtitle":article_subtitle,
                   "article_content": article_content,
                  }
        return article

    def parse_section(self, section, collection=None, title=None, subtitle=None, act_name = ''):
        articles = [] if collection is None else collection
        if isinstance(section, list):
            for sec in section:
                articles = self.parse_section(sec, collection=articles,
                                         title=title, subtitle=subtitle, act_name=act_name)
            return articles
        section_title = section.get("TITLE") if title is None else f"{title}_{section.get('TITLE')}"
        section_subtitle = section.get("SUBTITLE") if title is None else f"{subtitle}_{section.get('SUBTITLE')}"

        # Only iterate if it contains a list of articles
        if "ARTICLE" in section:
            for art in section.get("ARTICLE",[]):
                article_entry = self.parse_article(art, section_title=section_title, section_subtitle=section_subtitle, act_name=act_name)
                articles.append(article_entry)
            return articles
        return self.parse_section(section["SECTION"], collection=articles,
                                  title=section_title, subtitle=section_subtitle, act_name=act_name)


    def parse_articles(self, json_data, act_name=''):
        articles = []
        sections = json_data['DOCUMENT']['SECTION']
        for i, section in enumerate(sections):
            if isinstance(section, list):
                for sec in section:
                    section_data = self.parse_section(sec,act_name=act_name)
                    articles.extend(section_data)
            else:
                section_data = self.parse_section(section,act_name=act_name)
                articles.extend(section_data)
        return pd.DataFrame(articles)

    def Read_in_Json_from_text(self, Orig_dir ='', all_jason_list_path =''):
        with open(all_jason_list_path) as fp: 
            Lines = fp.readlines() 
            files =[]
            acts =[]
            for line in Lines: 
                with open(os.path.join(Orig_dir, line.strip()), "r") as f:
                    data_1 = json.load(f)
                files.append(data_1)
                acts.append(re.sub('\.json$', '', line.strip()))
        return files, acts
    
    def merge_replicated_entries(self, df, matching_coloumn_name='', merging_coloumn_name=''):
        K = df[df.duplicated(subset=[matching_coloumn_name], keep=False)]
        df = df[~df.duplicated(subset=[matching_coloumn_name], keep=False)]

        q = K.article_title.unique()
        j =1
        for i in q:
            P =K.loc[K[matching_coloumn_name] == q[0]].reset_index(drop=True)
            temp_df = P.iloc[[0]]
            for j in range(1, P.shape[0]):
                temp_df.loc[0, merging_coloumn_name] = temp_df.loc[0, merging_coloumn_name]+ P.loc[j, merging_coloumn_name]
            df = df.append(temp_df, ignore_index=True, sort=False)

        return df
#%% preprocess text
# Goes to Util:
class Text_cleaning(object):
    def __init__(self, additional_stopwords_FullPath):
        stop_words_nltk = set(stopwords.words('english'))
        stopwords_final = set(w.rstrip() for w in open(additional_stopwords_FullPath))#'/home/brain/AiTalents/Notebooks/stopwords.txt'))
        stopwords_final = set([s.replace('\n', '') for s in stopwords_final])
        stopwords_final = stopwords_final.union(stop_words_nltk)
        self.stopwords_final = stopwords_final
    def clean_html(self, raw_text):
        cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        cleantext = re.sub(cleanr, '', raw_text)
        return cleantext

    def remove_URL(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'',text)

    
    def my_tokenizer(self, s):
        s = s.lower() # downcase
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
        tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
        tokens = [t for t in tokens if t not in string.punctuation] # remove punctuation        
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
        tokens = [t for t in tokens if t not in self.stopwords_final] # remove stopwords      
        tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove any digits, i.e. "3rd edition"
        return tokens

    def preprocess_text(self, text):
        temp_text = self.remove_URL(text)
        no_html = self.clean_html(temp_text)
        return self.my_tokenizer(no_html)
    def Word_map_generation(self, df):
        #put clean Text into an array:
        contents = df["article_content"].values
        word_index_map = {}
        current_index = 0
        all_tokens = []
        all_content = []
        index_word_map = []
        error_count = 0
        for content in contents:
            try:
                content = content.encode('ascii', 'ignore').decode('utf-8') # this will throw exception if bad characters
                all_content.append(content)
                tokens = self.preprocess_text(content)
                all_tokens.append(tokens)
                for token in tokens:
                    if token not in word_index_map:
                        word_index_map[token] = current_index
                        current_index += 1
                        index_word_map.append(token)
            except Exception as e:
                print(e)
                print(title)
                error_count += 1
        return word_index_map, index_word_map,all_tokens, contents, error_count, self.stopwords_final


    def sklearn_Countvectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
                
        tf_vectorizer = CountVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                        token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                        tokenizer= self.preprocess_text)

        dtm_tf = tf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return tf_vectorizer, dtm_tf ## This is already a sparse feature matrix that could be fed into lda or SVD.
    def sklearn_TfidfVectorizer(self, df):  
        '''To see the tokens: tf_vectorizer.get_feature_names()'''
        contents = df["article_content"].values
        no_html=[]
        for content in contents:
            try:
                
                no_html.append(self.clean_html(content))
            except:
                 pass   
        
        Tfidf_vectorizer = TfidfVectorizer(encoding='ascii', strip_accents='ascii', stop_words=self.stopwords_final,\
                                           token_pattern = r'\b[a-zA-Z]{3,}\b', ngram_range = (1,1), analyzer='word',\
                                           tokenizer= self.preprocess_text)

        dtm_Tfidf = Tfidf_vectorizer.fit_transform(no_html)
        #print(dtm_tf.shape)
        return Tfidf_vectorizer, dtm_Tfidf ## This is already a sparse feature matrix that could be fed into lda or SVD.
            
    
    
#%% Feature generation:
#Goes to Util
class Feature_design(object):
    def __init__(self, word_index_map, all_tokens):
        self.word_index_map = word_index_map
        self.all_tokens = all_tokens
    # now let's create our input matrices - just indicator variables for this example - works better than proportions
    def tokens_to_vector(self, tokens):
        x = np.zeros(len(self.word_index_map))
        for t in tokens:
            i = self.word_index_map[t]
            x[i] = 1
        return x
    
    def Feature_matrix_generation(self):
        N = len(self.all_tokens)
        D = len(self.word_index_map)
        X = np.zeros((D, N)) # terms will go along rows, documents along columns
        i = 0
        for tokens in all_tokens:
            X[:,i] = self.tokens_to_vector(tokens)
            i += 1
        return X
    def Genism_Feature_matrix_generation(self, saving_dir):
        #analysis_dir= '/home/brain/AiTalents/Notebooks/Genism/'
        try:
            os.makedirs(saving_dir)
        except OSError:
            if not os.path.isdir(saving_dir):
                raise

        #os.chdir(analysis_dir)
        dictionary = corpora.Dictionary(self.all_tokens)
        corpus = [dictionary.doc2bow(text) for text in self.all_tokens]
        pickle.dump(corpus, open(os.path.join(saving_dir,'corpus.pkl'), 'wb'))
        dictionary.save(os.path.join(saving_dir,'dictionary.gensim'))
        return corpus, dictionary

#%% Topic models generation:
#Goes to Util
class Topic_modeling(object):
    def __init__(self, num_topics, saving_dir):
        
        
        self.num_topics = num_topics
        self.saving_dir = saving_dir
        try:
            os.makedirs(self.saving_dir)
        except OSError:
            if not os.path.isdir(self.saving_dir):
                raise
    def Genism_LDA(self, corpus, dictionary):
        #saving_dir= '/home/brain/AiTalents/Notebooks/Genism/'
            
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = self.num_topics, id2word=dictionary, passes=15)
        ldamodel.save(os.path.join(self.saving_dir,'LDA_model' + str(self.num_topics)+ '.gensim'))
        return ldamodel
    def Genism_LSI(self, corpus, dictionary):
        lsimodel = gensim.models.lsimodel.LsiModel(corpus, num_topics = self.num_topics, id2word=dictionary)
        lsimodel.save(os.path.join(self.saving_dir,'LSI_model' + str(self.num_topics)+ '.gensim'))
        return lsimodel
    def Sklearn_LDA(self, Feature_X):
        ldamodel = LatentDirichletAllocation()
        ldamodel.fit(Feature_X)
        Z = ldamodel.transform(Feature_X) # size: #term X #Component    >> For word Cloud
        # How about the graph connections?   componennts_ : size: #comps X #Articles  >> each column gives weights.
        U = ldamodel.components_    
        return ldamodel, Z, U 

            
#topics = ldamodel.print_topics(num_words=5)
#for topic in topics:
#    print(topic)

#%% Figures:

#Goes to Util
class Genism_plotting(object):
    def __init__(self,Figure_saving_dir):
        self.Figure_saving_dir = Figure_saving_dir
        try:
            os.makedirs(self.Figure_saving_dir)
        except OSError:
            if not os.path.isdir(self.Figure_saving_dir):
                raise
    def Convert(self, tup, di): 
        di = dict(tup) 
        return di

    def Reverse(self, tuples): 
        new_tup = () 
        for k in reversed(tuples): 
            new_tup = new_tup + (k,) 
        return new_tup 

    def pyLDA_Dashboard(self,ldamodel, corpus, dictionary,html_name, sort_topics=False):
        
        lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=sort_topics)
        pyLDAvis.save_html(lda_display, os.path.join(self.Figure_saving_dir,html_name + '_Dashboard.html'))
        #pyLDAvis.display(lda_display)
        return lda_display
    def Genism_word_cloud(self,html_name, ldamodel):
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

        cloud = WordCloud(background_color='#D8D8D8',
                          width=2500,
                          height=1800,
                          max_words=10,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)
        
        topic_word_box = ldamodel.show_topics(num_topics= ldamodel.num_topics, num_words= np.int(ldamodel.num_terms/100), formatted=False)
        # For each topic:
        topic_word_box_R= []
        for i in np.arange(ldamodel.num_topics):
            temp=[]
            temp = list(self.Reverse(topic_word_box[i][1]))
            topic_word_box_R.append(temp)
        
        D = ldamodel.num_topics
        CL = round(math.sqrt(D))
        RW = CL
        while RW * CL < D:
            RW += 1
        fig, axes = plt.subplots(RW, CL, figsize=(10,10), sharex=True, sharey=True)
        plt.style.context('grayscale')
        fig.patch.set_facecolor('#D8D8D8')

        Topic_names = ['Reporting', 'Investment', 'Risk_Management','Accounting', 'Payments_and_Settlement']
        #['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']
        #
        for i, ax in enumerate(axes.flatten()):
            if i >= D:
                axes.flatten()[i].axis('off')
            else:
            
                fig.add_subplot(ax)
                #topic_words = dict(topics[i][1])
                topic_words={}
                print(i)
                topic_words = self.Convert(topic_word_box_R[i], topic_words)
                cloud.generate_from_frequencies(topic_words, max_font_size=300)
                plt.gca().imshow(cloud)
                
                plt.gca().set_title(Topic_names[i] , fontdict=dict(size=16))
                #Topic_names.append('Topic_' + str(i +1))
                plt.gca().axis('off')


        plt.subplots_adjust(wspace=0, hspace=0)
        
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        
        #fig.patch.set_alpha(0.7)
        #plt.gca().set_facecolor('tab:gray')
        #cloud.to_file(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'))
        fig.savefig(os.path.join(self.Figure_saving_dir, html_name + '_Word_cloud.png'), transparent=True)
        
        return 
                
class Genism_Graph(object):
    def __init__(self, Figure_saving_dir='', verbose = True):
        
        self.Figure_saving_dir = Figure_saving_dir
        self.verbose = verbose
        
    def Genism_networkX_graph(self, df, ldamodel):
        Article_names = df['article_title'].values
        
        Topic_names = ['Topic_'+ str(i) for i in range(1, len(ldamodel.show_topics())+1)]
        nx_graph = nx.Graph()
        Nodes = Topic_names + list(Article_names)
        # Add node for each Topic
        for T in Nodes:
            nx_graph.add_node(T)

        i = 0   
        # add edges
        for a in range(0,len(Article_names)):
            temp = ldamodel.get_document_topics(corpus[a])
            for c in range(0, len(temp)):
                if self.verbose:
                    
                    print('C: ', c)
                    print(len(ldamodel.get_document_topics(corpus[a])))

                T = Topic_names[temp[c][0]]
                if self.verbose:
                    print('T, A: ',T, Article_names[a])
                i += 1
                if self.verbose:
                    print('i:', i)
                # Only add edge if the count is positive

                nx_graph.add_edge(Article_names[a],T, weight = temp[c][1])
                if self.verbose:
                    print('Done_edge')
        return nx_graph
    def plot_Genism_networkX_graph(self,df, ldamodel, to_bundel=True ):
        nx_graph = self.Genism_networkX_graph(df, ldamodel)
        #Create the graph using networkX data:
        act_names = []
        kwargs = dict(width=800, height=800, xaxis=None, yaxis=None)
        opts.defaults(opts.Nodes(**kwargs), opts.Graph(**kwargs))
        
        #colors = ['#000000']+hv.Cycle('Category20').values
        topic_graph = hv.Graph.from_networkx(nx_graph, nx.layout.fruchterman_reingold_layout, k=1)
        topic_graph.nodes.data['act'] = topic_graph.nodes.data['index']
        for i in np.arange(topic_graph.nodes.data.shape[0]):
            if topic_graph.nodes.data.loc[i, 'act'] not in ['Topic_1', 'Topic_2', 'Topic_3', 'Topic_4', 'Topic_5']:
                topic_graph.nodes.data.loc[i, 'act'] = str(topic_graph.nodes.data.loc[i, 'act'].split('_')[1])
                
        
        tmp2= topic_graph.nodes.data[~topic_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        #colors = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*np.arange(len(tmp)), 20+2*np.arange(len(tmp)))]
        
        if to_bundel:

            bundled = bundle_graph(topic_graph)
            #.opts(opts.Nodes(color='act', size=10, width=1000, cmap=colors, legend_position='right'))
#opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            #
            return bundled, act_names, article_names
        else:
            return topic_graph, act_names, article_names
        #.opts(cmap=colors, node_size=10, edge_line_width=1,node_line_color='gray', node_color='act', legend_position='right')
            
    
    def subselect_Genism_single_Topic_graph(self, Topic_id, topic_graph):
        
        Sub_graph = topic_graph.select(start=Topic_id).opts(node_fill_color='blue', edge_line_width=hv.dim('weight'), edge_line_color='red')
        tmp2= Sub_graph.nodes.data[~Sub_graph.nodes.data.act.str.contains("Topic")]
        act_names = list(tmp2.act.unique())
        article_names = set(list(tmp2['index'].values))
        
        return Sub_graph, act_names, article_names
    def subselect_Genism_single_Article_graph(self, Article_title, topic_graph, circular_layout=True):
        
        D = topic_graph.select(end=Article_title).opts(node_fill_color='white', edge_line_width=-1*np.log10(hv.dim('weight')))
        
        if circular_layout:
            
            F = layout_nodes(D)
            labels = hv.Labels(F.nodes, ['x', 'y'], 'index')

            return (F * labels.opts(text_font_size='10pt', text_color='black', bgcolor='white'))
        else: 
            
            return D
