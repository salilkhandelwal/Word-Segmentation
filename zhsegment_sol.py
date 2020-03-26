import re, string, random, glob, operator, heapq, codecs, sys, optparse, os, logging, math
from functools import reduce
from collections import defaultdict
from math import log10

class Entry:
    def __init__(self, word, log_probability, start_position, back_pointer):
        self.log_probability = log_probability
        self.word = word
        self.start_position = start_position
        self.back_pointer = back_pointer
    def __lt__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return self.log_probability >= other.log_probability


class Segment:
    def __init__(self, Pw):
        self.Pw = Pw
    def segment(self, text):
        "Return a list of words that is the best segmentation of text."
#         print(text)
#         utf8line = unicode(line.strip(), 'utf-8')
        output = [w for w in text]  # segmentation is one word per character in the input
        maxlen = len(output)
        heap_var = []
        chart = [None] * maxlen
        for x in range(maxlen): #(min(maxlen, 20)):
            var = ''.join(output[0: x+1])
            var_prob = self.Pw(var)
            if var_prob != None:
                heapq.heappush(heap_var, Entry(var, math.log(var_prob, 10), 0, None))
        
        while(len(heap_var) > 0):
            entry = heapq.heappop(heap_var)
            endindex = entry.start_position + len(entry.word) - 1
            if chart[endindex] != None:
                if(entry.log_probability > chart[endindex].log_probability):
                    chart[endindex] = entry
                if(entry.log_probability <= chart[endindex].log_probability):
                    continue
            else:
                chart[endindex] = entry
            for i in range(endindex+1, maxlen): #min(maxlen, endindex+20)):
                tmp = ''.join(output[endindex+1:i + 1])
                tmp_prob = self.Pw(tmp)
                if tmp_prob != None:
                    heapq.heappush(heap_var, Entry(tmp, entry.log_probability + math.log(tmp_prob, 10), endindex+1, entry))
        
        final_var = chart[maxlen-1]
        final_seq = []
        while(final_var != None):
            final_seq.append(final_var.word)
            final_var = final_var.back_pointer
        final_seq.reverse()
        return final_seq
    
    

#### Support functions (p. 224)


FACTOR = 450 

def penalize_long_words(word, N):
    return float(FACTOR) / (N * FACTOR ** len(word))




from collections import Counter

class Pdist(dict):
    "A probability distribution estimated from counts in datafile."
    def __init__(self, data=[], N=None, missingfn=None):
        tokens = []
        counts = []
        bk = {}
        for key, count in data:
            tokens.append(key)
            bk[key] = bk.get(key, 0) + int(count)
            
        self.N = float(N or sum(bk.values()))
        
        C = Counter(bk)
        Nc = Counter(list(C.values()))
        
        assert(self.N == sum([k * v for k, v in Nc.items()]))
        
        default_value = Nc[1] / self.N
        
        self.missingfn = missingfn or (lambda *args: default_value)
        
        types = C.keys()
        
        for _type in types:
            c = C[_type]
            c_star = ((c+1) * (Nc[c+1] or 1)) / (Nc[c])
            self[_type] = c_star / self.N
        
    def __call__(self, key):
        if key in self: 
            return self[key]
        elif len(key) <= 2:
            return self.missingfn(key, self.N)
        else: 
            return None
def datafile(name, sep='\t'):
    "Read key,value pairs from file."
    with open(name) as fh:
        for line in fh:
            (key, value) = line.split(sep)
            yield (key, value)


if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts [default: data/count_1w.txt]")
    optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts [default: data/count_2w.txt]")
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="file to segment")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    Pw = Pdist(data=datafile(opts.counts1w), missingfn=penalize_long_words)
    segmenter = Segment(Pw)
    with open(opts.input) as f:
        for line in f:
            print(" ".join(segmenter.segment(line.strip())))


