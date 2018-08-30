import re
from mrjob.job import MRJob
from mrjob.step import MRStep
YYYY = re.compile(r"[\w']+")

class LibraryCheckoutCount(MRJob):
    '''
        Input: Records containing either checkout data or inventory data
        Output: A generated list of key/value tuples:
            Key: A book title and year, joined by a "|" character
            Value: The number of times that book was checked out in the given year
    '''

    def mapper1(self, key, val):
        word = val.split(',')
        if word[1].isnumeric():
            yield (word[0], YYYY.findall(word[-1])[2])
        else:
            yield (word[0], word[1])

    def reducer1(self, key, vals):
        new_key = None
        before_list = []
        for v in vals:
            if not v.isnumeric():
                new_key = v
            else:
                before_list.append(v)

        if new_key != None:
            for b in before_list:
                yield (new_key, b)

    def mapper2(self, key, val):
        yield (''.join([key, '|' ,val]), 1)

    def reducer2(self, key, vals):
        count = 0
        for v in vals:
            count += 1
        yield (key, count)

    def steps(self):
        return [
            MRStep(mapper=self.mapper1, reducer=self.reducer1),
            MRStep(mapper=self.mapper2, reducer=self.reducer2)
            ]


if __name__ == '__main__':
    LibraryCheckoutCount.run()
