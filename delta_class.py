import text_process_classes as txt
from collections import Counter
from operator import itemgetter
from random import shuffle
import random
from nltk import FreqDist
import numpy
from scipy import stats

class DeltaCorpus(object):
    def __init__(self, candidate_1, candidate_2, disputed, markov=False, culling=0, mfw=50, bootstrap=False, bootstrap_samples=1000, bootstrap_sample_size=2000, chunk_size=False, chunks=10, manual_chunks=False):
        # everything should be lowercase and punctuation stripped
        
        #config variables
        self.culling = culling
        self.mfw = mfw
        self.bootstrap = bootstrap
        self.bootstrap_samples = bootstrap_samples
        self.bootstrap_sample_size = bootstrap_sample_size
        self.markov = markov
        self.chunks = chunks
        self.manual_chunks = manual_chunks
        self.chunk_size = chunk_size
        self.candidate_1 = candidate_1
        self.candidate_2 = candidate_2
        self.disputed = disputed
        #culling dictionary, words and score based upon where they occur
        self.culling_dict = {}
        #list of tuples sorted by number of occurrences, begins as an empty list
        self.combined_tup_sorted = []
        #list of words only, begins as an empty list
        self.top_words_corpus = []
        self.candidate1_subsets_freq = []
        self.candidate2_subsets_freq = []
        self.disputed_freq =[]
        self.top_results = []
        self.z_columns = []
        self.z_matrix = []
        self.z_avgs = []
        self.z_diffs = []
        self.c_1_result = 0.0 #float
        self.c_2_result = 0.0 #float
        
    def process_all(self):
        if self.manual_chunks is True:
            #convert lists_of_lists or lists of dicts into lists of counters
            self.candidate_1_subsets = [Counter(x).items() for x in self.candidate_1]
            self.candidate_2_subsets = [Counter(x).items() for x in self.candidate_2] 
            #print self.candidate_1_subsets, self.candidate_2_subsets 
            if self.markov is True:
                #combine into repeating list
                self.repeating_list_1 = self.markov_chunks_to_repeating_list(self.candidate_1)
                self.repeating_list_2 = self.markov_chunks_to_repeating_list(self.candidate_2)
                

            #combine subset dictionaries to one dictionary per candidate
            self.candidate_1_counter = Counter()
            for i in self.candidate_1:
                self.candidate_1_counter += Counter(i)
            self.candidate_2_counter = Counter()
            for i in self.candidate_2:
                self.candidate_2_counter += Counter(i)
            self.candidate_1 = self.candidate_1_counter
            self.candidate_2 = self.candidate_2_counter
            
            #print self.candidate_1, self.candidate_2
            if self.markov is False:
                #make repeating lists from dicts
                self.repeating_list_1 = self.repeating_list_from_dict(self.candidate_1) #a list of dictionaries, each item represents a text chunk
                self.repeating_list_2 = self.repeating_list_from_dict(self.candidate_2) #a list of dictionaries, each item represents a text chunk
        
        if self.manual_chunks is False: #if Markov and manual are False
            if self.markov is True: #if markov is True and manual chunks is False
                self.repeating_list_1 = self.candidate_1
                self.repeating_list_2 = self.candidate_2
            if self.markov is False:
                self.repeating_list_1 = self.repeating_list_from_dict(self.candidate_1) #a list of dictionaries, each item represents a text chunk
                self.repeating_list_2 = self.repeating_list_from_dict(self.candidate_2) #a list of dictionaries, each item represents a text chunk
        
        # all texts begin as frequency table dictionaries and become Counter dictionaries
        self.candidate_1 = Counter(self.candidate_1) #freqdist dictionary
        self.candidate_2 = Counter(self.candidate_2) #freqdist dictionary
        self.disputed = Counter(self.disputed) #freqdist dictionary
        
        #Counter dictionary expressing freqdist for all three
        self.combined = self.candidate_1 + self.candidate_2 + self.disputed
        
        self.count_culls()
        
        #this method sets self.combined_tup_sorted
        self.set_tup()
        
        #this method sets the top words list
        self.top_words()
            
        #subsets    
        if self.manual_chunks is False:
            self.candidate_1_subsets = self.subset_creator(self.repeating_list_1) 
            self.candidate_2_subsets = self.subset_creator(self.repeating_list_2) 
        
        self.word_counts_to_freqs()
        self.freqs_to_top_words() #sets the top_results variable
        self.freqs_to_zscores()
        self.z_scores_to_z_avg()
        self.z_avg_to_z_diffs()
        self.z_scores_to_final_delta()
        
    def get_raw_data(self):
        self.raw_data = ""
        headers = ('chunk_id', ';', 'word_id', ';', 'word', ';', 'frequency', ';', 'zscore', '\n' )
        for i in headers:
            self.raw_data += unicode(i) 
        
        numpy.set_printoptions(threshold='nan')
        w = self.z_matrix.tolist()
          
        chunk_count = 1
        for x, y in enumerate(self.top_results):
            word_id = 1
            for h, i in enumerate(y):
                row = (chunk_count, ";", word_id, ";", i[0], ";", i[1], ";", w[x][h], "\n")
                for j in row:
                    self.raw_data += unicode(j)
                word_id +=1
            chunk_count +=1
        return self.raw_data
    
    def get_config(self):
        return { 'chunk_size':self.chunk_size, 'chunks': self.chunks, 'culling': self.culling, 'mfw': self.mfw, 'words_compared':len(self.top_words_corpus), 'markov': self.markov, 'bootstrap':self.bootstrap, 'bootstrap_samples':self.bootstrap_samples, 'bootstrap_sample_size':self.bootstrap_sample_size }
    
    def count_culls(self):
        #loop through all words
        for i, j in self.combined.items():
            self.culling_dict[i] = 0
            #check if they occur in each subset
            #score 1, 2 or 3 based upon occurrence
            if i in self.candidate_1:
                self.culling_dict[i] = self.culling_dict[i] +1
            if i in self.candidate_2:
                self.culling_dict[i] = self.culling_dict[i] +1
            if i in self.disputed:
                self.culling_dict[i] = self.culling_dict[i] +1

    def set_tup(self):
        self.combined_tup_sorted = sorted(self.combined.items(), key=itemgetter(1), reverse=True)    
    
    def top_words(self):
        #print self.combined_tup_sorted
        words = [i[0] for i in self.combined_tup_sorted]
        words_cull = []
        for i in words:
            if self.culling_dict[i]*34> self.culling:
                words_cull.append(i)        
        words = words_cull
        #print words
        self.top_words_corpus = words[:self.mfw] 
    
    def repeating_list_from_dict(self, counter_dict):
        #explode dict to list with repetition to simulate frequency
        repeating_list = []
        for i, k in counter_dict.items():
            for t in range(k):
                repeating_list.append(i)
        return repeating_list
    
    def subset_creator(self, list_of_words):
        if self.markov is False:
            #shuffle the list
            shuffle(list_of_words)
        
        if self.bootstrap is False:
            if self.chunk_size is False:#random chunking, N subsets
                list_of_subsets = self.chunkify(list_of_words, self.chunks)
            else:
                list_of_subsets = self.chunkify_by_size(list_of_words, self.chunk_size)
        else:
            #bootstrap instead of chunkify
            list_of_sets = []
            for n in self.bootstrap_samples:
                #generate a list, same length as repeating_list, random values, repetition is fine
                proc_list = []
                sample_max_length = self.bootstrap_sample_size
                #sample_max_length = len(list_of_words)*self.bootstrap_sample_size/100
                while len(proc_list)<=sample_max_length:
                    #random item from list_of_words
                    random_item = random.choice(list_of_words)
                    proc_list.append(list_of_words)
                list_of_sets.append(proc_list)
        
        # convert back to freqdist dictionary
        subset_tuples= []
        
        #list of tuples
        for i in list_of_subsets:
            f = FreqDist(i)
            subset_tuples.append(f.items())
        
        return subset_tuples
    
    def word_counts_to_freqs(self):
        #word counts to frequencies of word in the set
        
        for i in self.candidate_1_subsets:
            #each i is a list of tuples
            total_words = sum([x[1] for x in i]) #total of all counts in i 
                
            processing_list=[]
            for k in i:
                #print k
                if k[0] in self.top_words_corpus: #this is just a list of words, in order of most frequent               
                    word_freq =  float(k[1])/float(total_words)
                    candidate1_subsets_freq_tuple = (k[0], word_freq)
                else:
                    word_freq = 0
                    candidate1_subsets_freq_tuple = (k[0], word_freq)
                #append to list of tuples
                processing_list.append(candidate1_subsets_freq_tuple)
            self.candidate1_subsets_freq.append(processing_list)
        
        for i in self.candidate_2_subsets:
            #each i is a dictionary
            total_words = sum([x[1] for x in i]) #total of all counts in i 
            processing_list = []
            for k in i:
                #print k
                if k[0] in self.top_words_corpus: #this is just a list of words, in order of most frequent               
                    word_freq =  float(k[1])/float(total_words)
                    candidate2_subsets_freq_tuple = (k[0], word_freq)
                else:
                    word_freq = 0
                    candidate2_subsets_freq_tuple = (k[0], word_freq)
                #append to list of tuples
                processing_list.append(candidate2_subsets_freq_tuple)
            self.candidate2_subsets_freq.append(processing_list)
        
        total_words = sum(self.disputed.values())
        for i in self.disputed.items():
            
            if i[0] in self.top_words_corpus:
                word_freq =  float(i[1])/float(total_words)    
                disputed_freq_tuple = (i[0], word_freq)
                #print disputed_freq_tuple
                self.disputed_freq.append(disputed_freq_tuple)
        
    def freqs_to_top_words(self):
        top_results = []
        top_words_scores = []
        for i in self.top_words_corpus: #i is just a word
            processing_tuple = (i, 0) # now we have the right tuple, but all values are zero
            top_words_scores.append(processing_tuple) 
            
        for j in self.candidate1_subsets_freq: #j is a list of tuples    
            top_dict = dict(top_words_scores) #a dictionary of top words, all values currently zero
            g = dict(j)
            for k, m in g.items():
                #k is words, m is values(in each j)
                if k in self.top_words_corpus:
                    #if word in tuple matches word in top list
                    #update dict accordingly
                    top_dict[k] = m    
            #top_dict back to list of tuples, append to result list
            top_tuple= top_dict.items()
            top_results.append(top_tuple)      
        
        for j in self.candidate2_subsets_freq: #j is a list of tuples    
            top_dict = dict(top_words_scores) #a dictionary of top words, all values currently zero
            g = dict(j)
            for k, m in g.items():
                #k is words, m is values(in each j)
                if k in self.top_words_corpus:
                    #if word in tuple matches word in top list
                    #update dict accordingly
                    top_dict[k] = m    
            #top_dict back to list of tuples, append to result list
            top_tuple= top_dict.items()
            top_results.append(top_tuple)
        
        top_dict = dict(top_words_scores) #a dictionary of top words, all values currently zero
        g = dict(self.disputed_freq)
        for k, m in g.items():
                #k is words, m is values(in each j)
            if k in self.top_words_corpus:
                #if word in tuple matches word in top list
                #update dict accordingly
                top_dict[k] = m    
        #top_dict back to list of tuples, append to result list
        top_tuple= top_dict.items()
        top_results.append(top_tuple)
        self.top_results = top_results
    
    def freqs_to_zscores(self):
        self.matrix_list = []
        for i in self.top_results:
            self.matrix_list.append([x[1] for x in i])
            
        #convert list to numpy matrix
        self.matrix = numpy.matrix(self.matrix_list)  
        #print self.matrix
        
        #number of columns
        self.columns, self.rows = self.matrix.shape[1], self.matrix.shape[0]
        
        
        #execute scipy zscore function on each column
        for i in range(self.columns):
            z_column = stats.zscore(self.matrix[:,i])
            self.z_columns.append(z_column)
            self.z_matrix = numpy.column_stack(self.z_columns)
        #print self.z_matrix
            
    def z_scores_to_z_avg(self):
        for i in range(self.rows):
            avg = numpy.mean(self.z_matrix[i,:])
            self.z_avgs.append(avg) #returns one z avg per text, mean of all z scores for all words
    
    def z_avg_to_z_diffs(self):
        for i in self.z_avgs:
            set_count = len(self.z_avgs)-1
            diff = abs(i - self.z_avgs[set_count]) #for last item, should always be zero
            self.z_diffs.append(diff)
        
    def z_scores_to_final_delta(self):
        # rows 1-10 are set 1, rows 11-20 are set 2, row 21 (position 20) is disputed
        #make dynamic by removing last item, dividing the rest into two parts ... del lst[-1]
        even_set_count = len(self.z_avgs)
        a = (even_set_count/2)
        b = (even_set_count/2)-1
        self.c_1_result = numpy.mean(self.z_diffs[0:a])  
        self.c_2_result = numpy.mean(self.z_diffs[b:even_set_count]) 
    
    def chunkify(self, lst, n):
        return [ lst[i::n] for i in xrange(n) ]
    
    def chunkify_by_size(self, lst, n):
        chunks=[lst[x:x+n] for x in xrange(0, len(lst), n)]
        del chunks[-1]
        return chunks
    
    def markov_chunks_to_repeating_list(self, list_of_lists):
        #combine into repeating list
        repeating_list = []
        for i in list_of_lists:
            repeating_list.extend(i)
        return repeating_list
    
def path_to_instance(var_name):
    handle = raw_input("Enter path to .txt file to serve as " + str(var_name) + ": ")
    with open(handle) as h:
        raw_text = h.read()
        text_instance = txt.ProcessedText(raw_text)
    return text_instance

if __name__ == '__main__':
    """
    a = {"as":4, "good":5, "you":13, "yg": 5, "nev":4, "a":7}
    b = {"as":111, "good":111, "you":18, "yg": 116, "nev":112, "qs":114}
    c = {"as":111, "good":112, "you":18, "yg": 115, "nev":113, "qs":115}
   
    
    d = DeltaCorpus(a,b,c, culling=0, mfw=125, bootstrap=False)
    
    
    #print d.disputed
    #print d.candidate_1_subsets #, d.candidate_2_subsets
    #print d.z_diffs
    print len(d.candidate_1_subsets)
    print len(d.candidate_2_subsets)
    print a
    print b
    print c
    print d.c_1_result, d.c_2_result

    """
    
    """
    a = cand1.dist
    b = cand2.dist
    c = disp.dist
    
    
    a = cand1.tokenized_text
    b = cand2.tokenized_text
    c = disp.tokenized_text
    
    #print a,b,c
    
    #test manual chunk, markov
    a = [['the', 'that', 'the', 'a', 'am'], ['the', 'that', 'the', 'a', 'am'], ['the', 'that', 'the', 'a', 'am'], ['the', 'they', 'the', 'art', 'am']]
    b = [['the', 'some', 'the', 'so', 'am'], ['the', 'that', 'they', 'a', 'am'], ['them', 'that', 'the', 'as', 'am'], ['think', 'they', 'the', 'art', 'am']]
    c = ['the', 'some', 'they', 'art', 'true', 'me', 'my', 'i', 'we']
    """
    
    """
    #test manual chunk, no markov
    a = [{'the':5, 'that':7, 'a':12, 'am':2}, {'the':2, 'that':5, 'a':3, 'am':1}, {'the':3, 'that':4, 'a':11, 'am':4}]
    b = [{'the':6, 'some':9, 'so':4, 'am':2}, {'the':4, 'that':8, 'they':8, 'a':5, 'am':2}, {'them':5, 'that':3, 'the':3, 'as':8, 'am':4}]
    c = {'the':7, 'some':10, 'they':14, 'art':1, 'true':4, 'me':4, 'my':8, 'i':1, 'we':3}
    """
    
    cand1 = path_to_instance('#Candidate 1')
    cand2 = path_to_instance('#Candidate 2')
    disp = path_to_instance('#Disputed Text')
    
    a = cand1.tokenized_text
    b = cand2.tokenized_text
    c = disp.tokenized_text
    
    d = DeltaCorpus(a,b,c)
    attrs = [('culling', 0), ('mfw', 50), ('bootstrap', False), ('markov', True), ('manual_chunks', False), ('chunks', 10)]
    for j, k in attrs:
        setattr(d, j, k)
    d.process_all()
    
    #d = DeltaCorpus(a,b,c, culling=100, mfw=30, bootstrap=False)
    
    print d.top_words_corpus
    print "Found ", len(d.top_words_corpus), " words to compare with culling set to ", d.culling
    print d.c_1_result, d.c_2_result
    print d.get_config()
            
    