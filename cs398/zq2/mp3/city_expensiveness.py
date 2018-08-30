from pyspark import SparkContext, SparkConf
import argparse
import json

def search(list_l):
    city = list_l.get('city')
    state = list_l.get('state')
    attributes = list_l.get('attributes')
    if city == None or state == None or attributes == None:
        return None
    for i in attributes:
        if 'RestaurantsPriceRange2' in i:
            return ((city + ', ' + state), (int(i.split(':')[1]),1))
    return None

def combine(x, y):
    return (x[0] + y[0], x[1] + y[1])

def find_city_expensiveness(sc, business_filename):
    '''
    Args:
        sc: The Spark Context
        business_filename: Filename of the Yelp businesses JSON file to use, where each line represents a business
    Returns:
        An RDD of tuples in the following format:
            (CITY_STATE, AVERAGE_PRICE)
            - CITY_STATE is in the format "CITY, STATE". i.e. "Urbana, IL"
            - AVERAGE_PRICE should be a float rounded to 2 decimal places
    '''
    distFile = sc.textFile(business_filename)
    source = distFile.map(json.loads) #return list of dict

    source = source.map(search)
    source = source.filter(lambda x: x != None)
    source = source.reduceByKey(combine)
    result = source.map(lambda x: (x[0], round(x[1][0] / x[1][1], 2)))
    return result.map(lambda x: (x[0], x[1]))

if __name__ == '__main__':
    # Get input/output files from user
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='File to load Yelp business data from')
    parser.add_argument('output', help='File to save RDD to')
    args = parser.parse_args()

    # Setup Spark
    conf = SparkConf().setAppName("city_expensiveness")
    sc = SparkContext(conf=conf)

    results = find_city_expensiveness(sc, args.input)
    results.saveAsTextFile(args.output)
