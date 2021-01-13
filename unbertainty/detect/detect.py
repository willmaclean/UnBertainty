import spacy
nlp = spacy.load('en_core_web_sm')
import os
from clean import TextPreprocessor
from spacy.pipeline import SentenceSegmenter
from spacy.tokenizer import Tokenizer
from spacy.lookups import Lookups
from spacy.lemmatizer import Lemmatizer
from typing import List
from collections import defaultdict
import spacy
from spacy.pipeline import merge_entities
from spacy.matcher import Matcher, DependencyMatcher

def lemma_text(word):
    
    return nlp.vocab[word.lemma].text

def default_rules(doc):
    
    #these are the default uncertainty patterns to spot
    
    questionable_lemmas = ['possible','possibly','presumably','probably','questionable','suspect','suspected','suspicious']
    questionable_hashes = [nlp.vocab[word] for word in questionable_lemmas]
    aux = ['may', 'would', 'could']
    matches=[]
    
    for word in doc:
        
        if word.lemma in questionable_hashes:
    
        #putting the pattern and the matching text  together
            matches.append((nlp.vocab[word.lemma].text, word.text)) 

        elif (word.pos_ == 'AUX') & (word.lemma in aux):

            matches.append((nlp.vocab[word.lemma].text, word.text)) 

        elif (lemma_text(word) in ['question','suggestion']) & (word.dep_ == 'attr'):

            matches.append((lemma_text(word), word.text)) 

        elif (lemma_text(word)  in ['suspect','favour','suggest','suggesting','question','consider'])\
        and ('no' not in [w.text for w in word.children]):

            matches.append((lemma_text(word), word.text))

        elif (lemma_text(word) in ['concern','suspicion'])\
        & ('no' not in [w.text for w in word.children]):

            matches.append((lemma_text(word), word.text))

        elif (lemma_text(word)  in ['suspected'])\
        & ('nsubjpass' in [w.dep_ for w in word.children]) & ('no' not in [w.text for w in word.children]):

            matches.append((lemma_text(word), word.text))

        elif (lemma_text(word) == 'possible') & \
        ('no' not in [w.text for w in word.children]):

            matches.append((lemma_text(word), word.text)) 

        elif (lemma_text(word) in ['maybe','perhaps']):

            matches.append((lemma_text(word), word.text)) 

        elif (lemma_text(word) in ['reflect','represent','indicate','include']):
            
            s = set(['may','could','would','might'])
            p = set([w.text for w in word.children])
            
            if s.intersection(p):
                
                matches.append((lemma_text(word), word.text)) 

        elif (lemma_text(word) in ['can','could','may','possibly','would']):
            
            ancestor = [a for a in word.ancestors][0]

            for w in ancestor.children:

                if lemma_text(w) in ['down','due', 'thanks','because']:

                    matches.append((lemma_text, (word.text, w.text)))

        elif (lemma_text(word) in ['could','can','may','would','possibly']):

            for w in word.children:

                if lemma_text(w) == 'relate':

                    matches.append((lemma_text(word), 'could relate to'))

        elif (lemma_text(word) in ['could','may','would']):
            
            print(lemma_text(word), 't print')
            
            ancestor = [a for a in word.ancestors]
            
            print(ancestor)

            for w in ancestor.children:

                if lemma_text(w) == 'compatible':

                    matches.append(lemma_text(word), 'could be compatible with')

        elif lemma_text(word) in ['exclude', 'rule'] and 'not' in [lemma_text(w) for w in word.children]:

            matches.append((lemma_text(word), 'cannot exclude'))

        elif lemma_text(word) == 'be':
            
            s = set(['may','could','would','might'])
            p = set([w.text for w in word.children])
            
            if s.intersection(p):
                
                matches.append((lemma_text(word), 'may be'))

        elif (lemma_text(word) =='suggestive') & ('of' in [w.text for w in word.children]):

            matches.append((lemma_text(word), 'may be'))
    
    return matches
        

class Match:
    #need an object to store all the matches
    
    def __init__(self, default_rules=True):
        
        self.matched_doc = None
        self.lemmas_ = None
        self.rules_ = defaultdict()
        self.default_rules_ = True
        self.custom_rules_ = False
        self.matches_ = None
    
    def make_match(self, doc):
            
            if default_rules:
                
                self.matches_ = default_rules(doc)
                
                if self.custom_rules_:
                    
                    self.matches_.extend(self.custom_match(doc))
                
            else: 
                
                self.matches_ = self.custom_match(doc)
            
            return self.matches_
                
            
    def add_rule(self, pattern):
        
        """
        Rules should be of the form:
        
        {'name': to id the pattern
        ROOT': [lemmas, to, match],
        'CHILDREN: [lemmas to match]}
        """
        
        self.rules_[pattern['name']]['ROOT'] = pattern['ROOT']
        self.rules_[pattern['name']]['CHILDREN'] = pattern['CHILDREN']
        self.custom_rules = True
        
    def custom_match(self, doc):
        
        matches = []
        
        for word in doc:
            
            for rule_name in self.rules_.keys:
                
                if lemma_text(word) in self.rules_[rule_name]['ROOT'] & \
                (set(self.rules_[rule_name]['CHILDREN']).intersection(set([w.text for w in word.children]))):

                    matches.append((lemma_text(word), (self.rules_[rule_name]['ROOT'], self.rules_[rule_name]['CHILDREN'])))

        return matches
        
         
    