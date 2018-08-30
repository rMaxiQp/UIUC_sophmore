from mrjob.job import MRJob

class WikipediaLinks(MRJob):
    '''
        Input: List of lines, each containing a user session.
            - Articles are separated by ';' characters
            - Given a session 'page_a;page_b':
                this indicates there is a link from the article page_a to page_b
            - A '<' character indicates that a user has clicked 'back' on their
                browser and has returned to the previous page they were on
        Output: The number of unique inbound links to each article
            Key: Article name (str)
            Value: Number of unique inbound links (int)
    '''

    def mapper(self, key, val):
        word_list = val.split(';')
        valid = [word_list[0]]
        for word in range(1, len(word_list)):
            if word_list[word] != '<':
                valid.append(word_list[word])
                yield word_list[word],valid[-2]
            else:
                valid.pop()

    def reducer(self, key, vals):
        ll = []
        total_sum = 0
        for v in vals:
            if v not in ll:
                total_sum += 1
                ll.append(v)
        yield key, total_sum


if __name__ == '__main__':
    WikipediaLinks.SORT_VALUES = True
    WikipediaLinks.run()
