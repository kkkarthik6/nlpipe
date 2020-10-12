

'''Data Preprocessing'''
import spacy
import numpy as np
import pandas as pd

class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class PipelineError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class SpacyBulk():
    def __init__(self, ner=1, parser=1, tagger=1):
        self.disable = []
        self.ner = ner
        self.parser = parser
        self.tagger = tagger

        if self.ner == 0:
            self.disable.append('ner')
        if self.parser == 0:
            self.disable.append('parser')
        if self.tagger == 0:
            self.disable.append('tagger')
        if self.disable:
            self.nlp = spacy.load("en_core_web_sm", disable=self.disable)
        else:
            self.nlp = spacy.load('en_core_web_sm')
        self.pos_tag = []
        self.tokens = []
        self.tokenCnt_lst = []
        self.poss = []
        self.dep = []
        self.nerl = []
        self.text_corpus = []

    def get_DF(self, text_corpus, preprocess_out):
        self.text_corpus = text_corpus
        if len(self.text_corpus) == 0:
            raise PipelineError('Input text cannot be None',
                                'This object extract sentences, parts of speech, named entity recognition, dependencies')
        sentences = []
        tokens = []
        dep = []
        tag = []
        ID = []
        for i, text in enumerate(self.text_corpus):
            #print(i)
            self.dep = []
            self.pos_tag = []
            self.doc = self.nlp(str(text))
            sents = [i.text for i in self.doc.sents]
            ID.extend([i] * len(sents))
            sentences.extend(sents)
            tokens.extend(self.get_tokens())
            if self.parser:
                dep_vecs = self.get_dependencies()
                dep_lst = []
                # print(dep_vecs)
                for i in dep_vecs:
                    dep_lst.append(' '.join(i))
                dep.extend(dep_lst)
            else:
                dep.extend(['disabled'] * len(sents))
            if self.tagger:
                tag_vecs = self.get_pos_tag()
                tag_lst = []
                for i in tag_vecs:
                    tag_lst.append(' '.join(i))
                tag.extend(tag_lst)
            else:
                tag.extend(['disabled'] * len(sents))
            del self.doc

        # print(len(ID),len(sentences),len(dep),len(tag))
        self.df = pd.DataFrame(
            {'textID': ID, 'sentences': sentences, 'Dependency': dep, 'POS Tags': tag, 'tokens': tokens})
        self.df.to_csv(preprocess_out)
        return self.df

    def get_sentences_lst(self):
        return [i.text for i in self.doc.sents]

    def get_sentences(self):
        return list(self.doc.sents)

    def get_tokens(self):
        self.sentences = self.get_sentences()
        tokens = []
        for i in self.sentences:
            token = []
            for tokenz in i:
                if tokenz.is_alpha:
                    token.append(tokenz.text)
            tokens.append(token)
        return tokens

    def get_pos_tag(self):
        if self.tagger:
            self.sentences = self.get_sentences()

            for i in self.sentences:
                token = []
                for tokenz in i:
                    token.append((tokenz.tag_))
                self.pos_tag.append(token)
            return list(self.pos_tag)
        else:
            raise PipelineError('"tagger" disabled', 'enable to use this feature')

    def get_pos(self):
        if self.tagger:
            self.sentences = self.get_sentences()

            for i in self.sentences:
                token = []
                for tokenz in i:
                    token.append((tokenz.pos_))
                self.poss.append(token)
            return list(self.poss)
        else:
            raise PipelineError('"tagger" disabled', 'enable to use this feature')

    def get_dependencies(self):
        if self.parser:
            self.sentences = self.get_sentences()

            for i in self.sentences:
                token = []
                for tokenz in i:
                    token.append(tokenz.dep_)
                self.dep.append(token)
            return list(self.dep)
        else:
            raise PipelineError('"dep" disabled', 'enable to use this feature')

    def get_ner(self):
        if self.ner:
            # self.sentences.ents
            for ent in self.doc.ents:
                token = (ent.text, ent.label_)
                self.nerl.append(token)
            return list(self.nerl)
        else:
            raise PipelineError('"ner" disabled', 'enable to use this feature')


class Spacyize:
    def __init__(self, ner=1, parser=1, tagger=1, text=None):
        self.disable = []
        self.ner = ner
        self.parser = parser
        self.tagger = tagger
        self.text = text
        if self.text == None:
            raise PipelineError('Input text cannot be None',
                                'This object extract sentences, parts of speech, named entity recognition, dependencies')
        if self.ner == 0:
            self.disable.append('ner')
        if self.parser == 0:
            self.disable.append('parser')
        if self.tagger == 0:
            self.disable.append('tagger')
        if self.disable:
            self.nlp = spacy.load("en_core_web_sm", disable=self.disable)
        else:
            self.nlp = spacy.load('en_core_web_sm')
        self.doc = self.nlp(text)
        self.pos_tag = []
        self.tokens = []
        self.tokenCnt_lst = []
        self.poss = []
        self.dep = []
        self.nerl = []

    def get_sentences_lst(self):
        return [i.text for i in self.doc.sents]

    def get_sentences(self):
        return list(self.doc.sents)

    def count_sentences(self):
        return len(list(self.doc.sents))

    def get_tokens(self):
        self.sentences = self.get_sentences()
        # self.tokens = []
        for i in self.sentences:
            token = []
            for tokenz in i:
                if tokenz.is_alpha:
                    token.append(tokenz)
            self.tokens.append(token)
        return list(self.tokens)

    def average_tokens(self):
        if not self.tokens:
            self.get_tokens()

        for i in self.tokens:
            self.tokenCnt_lst.append(len(i))
        return np.mean(self.tokenCnt_lst)

    def get_pos_tag(self):
        if self.tagger:
            self.sentences = self.get_sentences()

            for i in self.sentences:
                token = []
                for tokenz in i:
                    token.append((tokenz.tag_))
                self.pos_tag.append(token)
            return list(self.pos_tag)
        else:
            raise PipelineError('"tagger" disabled', 'enable to use this feature')

    def get_pos(self):
        if self.tagger:
            self.sentences = self.get_sentences()

            for i in self.sentences:
                token = []
                for tokenz in i:
                    token.append((tokenz.pos_))
                self.poss.append(token)
            return list(self.poss)
        else:
            raise PipelineError('"tagger" disabled', 'enable to use this feature')

    def get_dependencies(self):
        if self.parser:
            self.sentences = self.get_sentences()

            for i in self.sentences:
                token = []
                for tokenz in i:
                    token.append(tokenz.dep_)
                self.dep.append(token)
            return list(self.dep)
        else:
            raise PipelineError('"dep" disabled', 'enable to use this feature')

    def get_ner(self):
        if self.ner:
            # self.sentences.ents
            for ent in self.doc.ents:
                token = (ent.text, ent.label_)
                self.nerl.append(token)
            return list(self.nerl)
        else:
            raise PipelineError('"ner" disabled', 'enable to use this feature')


class CustomTokenParser(Spacyize):
    def __init__(self, ner, parser, tagger, text, split):
        super().__init__(ner, parser, tagger, text)
        self.split = split
        self.nlp.add_pipe(self.set_custom_boundaries, before='parser')
        self.doc = self.nlp(text)

    def set_custom_boundaries(self, doc1):
        for token in doc1[:-1]:
            if token.text == self.split:
                doc1[token.i + 1].is_sent_start = True
        return doc1

def read_preprocessed(preprocess_out=''):
    df = pd.read_csv(preprocess_out)
    return df