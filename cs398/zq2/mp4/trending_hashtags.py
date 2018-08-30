from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
import argparse
import sys
import json

def parse(x):
    try:
        k = json.loads(x)
        result = []
        for i in k['text'].split():
            if i[0] == '#':
                result.append(i)
        return result
    except:
        return []

def topN(x):
    v_list = x.sortBy(lambda k : k[1]).take(10)
    x.filter(lambda x: x in v_list)
    return x

def find_trending_hashtags(sc, input_dstream):
    '''
    Args:
        sc: the SparkContext
        input_dstream: The discretized stream (DStream) of input
    Returns:
        A DStream containing the top 10 trending hashtags, and their usage count
    '''
    # format: "text": "<tweet body>"
    # YOUR CODE HERE
    input_list = input_dstream.flatMap(parse)
    input_window = input_list.countByValueAndWindow(60, 10)
    return input_window.transform(topN)


if __name__ == '__main__':
    # Get input/output files from user
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_stream', help='Stream URL to pull data from')
    parser.add_argument('--input_stream_port', help='Stream port to pull data from')

    parser.add_argument('output', help='Directory to save DStream results to')
    args = parser.parse_args()

    # Setup Spark
    conf = SparkConf().setAppName("trending_hashtags")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 10)
    ssc.checkpoint('streaming_checkpoints')

    if args.input_stream and args.input_stream_port:
        dstream = ssc.socketTextStream(args.input_stream,
                                       int(args.input_stream_port))
        results = find_trending_hashtags(sc, dstream)
    else:
        print("Need input_stream and input_stream_port")
        sys.exit(1)

    results.saveAsTextFiles(args.output)
    results.pprint()

    ssc.start()
    ssc.awaitTermination()
