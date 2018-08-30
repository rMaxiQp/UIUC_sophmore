import re
from mrjob.job import MRJob
from string import punctuation
r = re.compile(r'@[\d\w_]+'.format(re.escape(punctuation)))
class TwitterAtMentions(MRJob):
    '''
        Input: List of lines containing tab-separated tweets with following format:
            POST_DATETIME <tab> TWITTER_USER_URL <tab> TWEET_TEXT

        Output: A generated list of key/value tuples:
            Key: Twitter user handle (including '@' prefix)
            Value: Number of @-mentions received
    '''

    def mapper(self, key, val):
        appear = []
        for word in r.findall(val):
            size = len(word)
            if size > 3 and size < 17:
                if word not in appear:
                    appear.append(word)
                    yield word, 1

    def reducer(self, key, vails):
        count = 0;
        for v in vails:
            count += 1;
        yield key, count


if __name__ == '__main__':
    TwitterAtMentions.run()
