from gensim.summarization.summarizer import summarize

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

class Summarizer:
    def __init__(self, ratio=0.3,word_count=None,split=False):
        self.ratio=ratio
        self.word_count=word_count
        self.split=split
    def summarize_corpus(self,text_corpus):
        text_corpus = text_corpus
        summaries = []
        try:
            for i in range(len(text_corpus)):
                summaries.append(summarize(text_corpus[i], ratio=self.ratio, word_count=self.word_count, split=self.split))
        except ValueError:
            raise PipelineError('Passed an empty text corpus', 'Please pass list of text data')
        return summaries