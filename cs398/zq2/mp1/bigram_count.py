import re
from mrjob.job import MRJob

WORD_REGEX = re.compile(r"[\w']+")

class BigramCount(MRJob):
    '''
        Input: List of lines containing sentences (possibly many sentences per line)
        Output: A generated list of key/value tuples:
            Key: A bigram separated by a comma (i.e. "the,cat")
            Value: The number of occurences of that bigram (integer)
    '''

    def mapper(self, key, val):
        word_list = WORD_REGEX.findall(val);
        for k in range(len(word_list)):
            if((k + 1) < len(word_list)):
                yield (','.join((word_list[k], word_list[k+1])), 1)

    def reducer(self, key, vals):
        total_sum = 0
        for v in vals:
            total_sum += 1
        yield key, total_sum



if __name__ == '__main__':
    BigramCount.SORT_VALUES = True
    BigramCount.run()
