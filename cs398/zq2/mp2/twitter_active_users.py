from mrjob.job import MRJob

class TwitterActiveUsers(MRJob):
    '''
        Input: List of lines containing tab-separated tweets with following format:
            POST_DATETIME <tab> TWITTER_USER_URL <tab> TWEET_TEXT

        Output: A generated list of key/value tuples:
            Key: Day in `YYYY-MM-DD` format
            Value: Twitter user handle of the user with the most tweets on this day
                (including '@' prefix)
    '''

    def mapper(self, key, val):
        word_list = val.split('\t')
        yield (word_list[0].split()[0], word_list[1].split('/')[-1]), 1

    def combiner(self, key, vals):
        count = 0
        for v in vals:
            count += 1
        yield key[0], ('@' + key[1], count)

    def reducer(self, key, vals):
        most = ""
        value = 0
        for v in vals:
            if (v[1] > value) or (v[1] == value and v[0] < most):
                most = v[0]
                value = v[1]
        yield key, most

if __name__ == '__main__':
    TwitterActiveUsers.SORT_VALUES = True
    TwitterActiveUsers.run()
