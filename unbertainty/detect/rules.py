import spacy
nlp = spacy.load('en_core_web_sm')
import os
from clean import TextPreprocessor
from spacy.pipeline import SentenceSegmenter
from spacy.tokenizer import Tokenizer
from spacy.lemmatizer import Lemmatizer
from typing import List
from collections import defaultdict
from spacy.pipeline import merge_entities
from spacy.matcher import Matcher, DependencyMatcher
from spacy.vocab import Vocab

def lemma_text(word):
    
    return nlp.vocab[word.lemma].text

def default_rules(doc):

    nlp = spacy.load('en_core_web_sm')
    
    #these are the default uncertainty patterns to spot
    
    questionable_lemmas = ['possible','possibly','presumably','probably','questionable','suspect','suspected','suspicious']
    questionable_hashes = [nlp.vocab[word] for word in questionable_lemmas]
    aux = ['may', 'would', 'could']
    matches=[]
    
matches=[]
    
    for word in doc:
        
#         iterates through each token in the spacy-tokenised doc and 
#         checks if there is a match with a word or with a dependency pattern
        
    #single adjectives and adverbs which denote uncertainty of observation
        if lemma_text(word) in ['possible','possibly','presumably','probably','questionable',\
                                'suspect','suspected','suspicious', 'probable', 'potential']:

            matches.append((lemma_text(word), 'single words showing questionability: possible, probably, etc.')) 
        
        #checking for modality
        elif (word.pos_ == 'AUX') & (lemma_text(word) in ['may', 'would', 'could']):

            matches.append((lemma_text(word), 'modality')) 

        #question or suggestion in attributive dependency relation
        elif (lemma_text(word) in ['question','suggestion']) & (word.dep_ == 'attr'):

            matches.append((lemma_text(word), 'is a question/suggestion of ')) 

            #matching verbs which denote an uncertain proposal/observation, also making sure 'no' is not in the parse
        elif (lemma_text(word)  in ['suspect','favour','suggest','suggesting','question','consider'])\
        and ('no' not in [w.text for w in word.children]):

            matches.append((lemma_text(word), 'I suspect/suggest'))
    
    #there is concern/suspicion
        elif (lemma_text(word) in ['concern','suspicion'])\
        & ('no' not in [w.text for w in word.children]):

            matches.append((lemma_text(word), 'there is concern/suspicion'))

        #suspected
        elif (lemma_text(word)  in ['suspected'])\
        & ('nsubjpass' in [w.dep_ for w in word.children]) & ('no' not in [w.text for w in word.children]):

            matches.append((lemma_text(word), 'X is suspected'))

                    #possible
        elif (lemma_text(word) == 'possible') & \
        ('no' not in [w.text for w in word.children]):

            matches.append((lemma_text(word), 'possible')) 

        #maybe/perhaps
        elif (lemma_text(word) in ['maybe','perhaps']):

            matches.append((lemma_text(word), 'maybe/perhaps')) 

            #soft observation verbs with modality
        elif (lemma_text(word) in ['reflect','represent','indicate','include']):
            
            s = set(['may','could','would','might'])
            p = set([w.text for w in word.children])
            
            if s.intersection(p):
                
                matches.append((lemma_text(word), 'soft verbs w modality')) 

                #modality and consequence -- could be because
        elif (lemma_text(word) in ['can','could','may','possibly','would']):
            
            try:
                ancestor = [a for a in word.ancestors][0]

                for w in ancestor.children:

                    if lemma_text(w) in ['down','due', 'thanks','because']:

                        matches.append((lemma_text(w), 'modality and consequence'))
            except:
                
                pass
            
            
        # modality and relation
        elif (lemma_text(word) in ['could','can','may','would','possibly']):

            for w in word.children:

                if lemma_text(w) == 'relate':

                    matches.append((lemma_text(word), 'modality and relation'))

        #could be compatible
        elif (lemma_text(word) in ['could','may','would']):
            
            ancestor = [a for a in word.ancestors] 
            
            for w in ancestor.children:

                if lemma_text(w) in ['compatible', 'representative']:

                    matches.append(lemma_text(word), 'could be compatible with')
        
        #cannot exclude
        elif lemma_text(word) in ['exclude', 'rule'] and 'not' in [lemma_text(w) for w in word.children]:

            matches.append((lemma_text(word), 'cannot exclude'))

        #modality and copula search
        elif lemma_text(word) == 'be':
            
            s = set(['may','could','would','might'])
            p = set([w.text for w in word.children])
            
            if s.intersection(p):
                
                matches.append((lemma_text(word), 'modality and copularity'))

        elif (lemma_text(word) =='suggestive') & ('of' in [w.text for w in word.children]):

            matches.append((lemma_text(word), 'suggestive of'))
    
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
        
         
    