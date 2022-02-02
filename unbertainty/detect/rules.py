import spacy
import os
from clean import TextPreprocessor
from spacy.pipeline import SentenceSegmenter
from spacy.tokenizer import Tokenizer
from spacy.lemmatizer import Lemmatizer
from typing import List, Dict
from collections import defaultdict
from spacy.pipeline import merge_entities
from spacy.matcher import Matcher, DependencyMatcher
from spacy.vocab import Vocab

nlp = spacy.load('en_core_web_sm')


QUESTIONABLE_LEMMAS = ['possible','possibly','presumably','probably','questionable','suspect','suspected','suspicious']


def lemma_text(word):
    
    return nlp.vocab[word.lemma].text

class Rules(doc):
    
    questionable_hashes = [nlp.vocab[word] for word in questionable_lemmas]
    aux = ['may', 'would', 'could']
    matches=[]

    def __init__(
                    self, 
                    model: str = 'en_core_web_sm',
                    override: Dict = None
                ):
        self.nlp = spacy.load(model)

    
    def _questionable_single_words(self, word):
        match_lemmas = [
                        'possible',
                        'possibly',
                        'presumably',
                        'probably',
                        'questionable',
                        'suspect',
                        'suspected',
                        'suspicious', 
                        'probable', 
                        'potential'
                      ]

        return dict(code = 'QSW',   
                    rule = (if lemma_text(word) in match_lemmas),
                    description = 'single words showing questionability',
                    match_lemmas=match_lemmas
                    )
                

    def _modality(self, word):
        match_lemmas = ['may', 'would', 'could']
        return dict(
                    code = 'MOD',
                    rule = ((if lemma(text) in match_words) & (word.pos_ == 'AUX')),
                    description = 'Modality'
                    )

    def _question_suggestion(self, word):
        match_lemmas = ['question', 'suggestion']
        return dict(
            code = 'QSS',
            rule = ((lemma_text(word) in match_lemmas) & (word.dep_ == 'attr')),,
            description = 'is a question of suggestion of '
            )

    def _suspect_suggest(self, word):
        match_lemmas = ['suspect','favour','suggest','suggesting','question','consider']
        return dict(
            code = 'SSS',
            rule = (lemma_text(word) in match_lemmas and ('no' not in [w.text for w in word.children])),
            description = 'Suspecting or suggesting'
            )

    def _concern_suspicion(self, word):
        match_lemmas = ['concern','suspicion']
        return dict(
            code = 'CSN',
            rule = (lemma_text(word) in match_lemmas) & ('no' not in [w.text for w in word.children]),
            description = 'There is concern or suspicion'
            )

    def _suspected(self, word):
        match_lemmas = ['suspected']
        return dict(
            code = 'SPC',
            rule = (lemma_text(word)  in match_lemmas)
                     & ('nsubjpass' in [w.dep_ for w in word.children]) & 
                     ('no' not in [w.text for w in word.children]),
            description = 'X is suspected')


    def _possible(self, word):
        return dict(
            code = 'POS',
            rule = (lemma_text(word) == 'possible') & ('no' not in [w.text for w in word.children]),
            description = 'Possible')

    def _maybe_perhaps(self, word):
        match_lemmas = ['maybe','perhaps']
        return dict(
            code = 'MPS',
            rule = (lemma_text(word) in match_lemmas),
            description = 'Maybe or perhaps')

    def _soft_modality(self, word):
        match_lemmas = ['reflect','represent','indicate','include']
        return dict(
            code = 'SMO',
            rule = (lemma_text(word) in match_lemmas),)
    


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
        
         
    