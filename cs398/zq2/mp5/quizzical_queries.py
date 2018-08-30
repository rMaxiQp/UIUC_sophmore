from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
import argparse

def setup_table(sc, sqlContext, reviews_filename):
    '''
    Args:
        sc: The Spark Context
        sqlContext: The Spark SQL context
        reviews_filename: Filename of the Amazon reviews file to use, where each line represents a review
    Parse the reviews file and register it as a table in Spark SQL in this function
    '''

    df = sqlContext.read.csv(reviews_filename, header=True, inferSchema=True)
    sqlContext.registerDataFrameAsTable(df, 'table')

def query_1(sc, sqlContext):
    '''
    Args:
        sc: The Spark Context
        sqlContext: The Spark SQL context
        An string: the review text of the review with id `22010`
    '''
    return ''.join(sqlContext.sql('SELECT Text '
                          'FROM table '
                          'WHERE Id = 22010 ').collect()[0])

def query_2(sc, sqlContext):
    '''
    Args:
        sc: The Spark Context
        sqlContext: The Spark SQL context
    Returns:
        An int: the number of 5-star ratings the product `B000E5C1YE` has
    '''
    list_l = sqlContext.sql("SELECT Score "
                          "FROM table "
                          "WHERE ProductId='B000E5C1YE' "
                          "AND Score=5").collect()
    return len(list_l)
def query_3(sc, sqlContext):
    '''
    Args:
        sc: The Spark Context
        sqlContext: The Spark SQL context
    Returns:
        An int: the number unique (distinct) users that have written reviews
    '''
    list_l = sqlContext.sql("SELECT DISTINCT UserId "
                   "FROM table "
                   "WHERE Text IS NOT NULL").collect()
    return len(list_l)

if __name__ == '__main__':
    # Get input/output files from user
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='File to load Amazon review data from')
    args = parser.parse_args()

    # Setup Spark
    conf = SparkConf().setAppName("quizzical_queries")
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)

    setup_table(sc, sqlContext, args.input)

    result_1 = query_1(sc, sqlContext)
    result_2 = query_2(sc, sqlContext)
    result_3 = query_3(sc, sqlContext)

    print("-" * 15 + " OUTPUT " + "-" * 15)
    print("Query 1: {}".format(result_1))
    print("Query 2: {}".format(result_2))
    print("Query 3: {}".format(result_3))
    print("-" * 30)
