from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
import argparse
import sys
import json
import re

REG = re.compile(r"[\w']+")

stopwords_str = 'i,me,my,myself,we,our,ours,ourselves,you,your,yours,yourself,yourselves,he,him,his,himself,she,her,hers,herself,it,its,itself,they,them,their,theirs,themselves,what,which,who,whom,this,that,these,those,am,is,are,was,were,be,been,being,have,has,had,having,do,does,did,doing,a,an,the,and,but,if,or,because,as,until,while,of,at,by,for,with,about,against,between,into,through,during,before,after,above,below,to,from,up,down,in,out,on,off,over,under,again,further,then,once,here,there,when,where,why,how,all,any,both,each,few,more,most,other,some,such,no,nor,not,only,own,same,so,than,too,very,s,t,can,will,just,don,should,now,d,ll,m,o,re,ve,y,ain,aren,couldn,didn,doesn,hadn,hasn,haven,isn,ma,mightn,mustn,needn,shan,shouldn,wasn,weren,won,wouldn,dont,cant'
stopwords = set(stopwords_str.split(','))

def group(x):
    try:
        k = json.loads(x)
    except:
        return None
    return (k['subreddit'], k['text'])

def sum_all(x):
    count_num = {}
    for i in REG.findall(x[1].lower()):
        if count_num.get(i) != None:
            count_num[i] += 1
        elif i not in stopwords:
            count_num[i] = 1

    ret = []
    val = []
    if(len(count_num) <= 10):
        for i in count_num:
            ret.append(i)
        return (x[0], tuple(ret))

    for i in count_num:
        ret.append((i, count_num[i]))
    ret = sorted(ret, key=lambda x: -x[1])[:10]
    for i in ret:
        val.append(i[0])
    return (x[0], tuple(val))

def find_subreddit_topics(sc, input_dstream):
    '''
    Args:
        sc: the SparkContext
        input_dstream: The discretized stream (DStream) of input
    Returns:
        A DStream containing the list of common subreddit words
    '''

    # YOUR CODE HERE
    input_list = input_dstream.map(group).filter(lambda x : x != None)
    input_window = input_list.reduceByKeyAndWindow(lambda x,y : x + y, lambda x,y : x - y, 900, 10)
    result = input_window.map(sum_all)
    return result

if __name__ == '__main__':
    # Get input/output files from user
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_stream', help='Stream URL to pull data from')
    parser.add_argument('--input_stream_port', help='Stream port to pull data from')

    parser.add_argument('output', help='Directory to save DStream results to')
    args = parser.parse_args()

    # Setup Spark
    conf = SparkConf().setAppName("subreddit_topics")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 10)
    ssc.checkpoint('streaming_checkpoints')

    if args.input_stream and args.input_stream_port:
        dstream = ssc.socketTextStream(args.input_stream,
                                       int(args.input_stream_port))
        results = find_subreddit_topics(sc, dstream)
    else:
        print("Need input_stream and input_stream_port")
        sys.exit(1)

    results.saveAsTextFiles(args.output)
    results.pprint()

    ssc.start()
    ssc.awaitTermination()
