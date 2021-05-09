from pyspark.sql import SparkSession
from pyspark import SparkContext
from graphframes import *
import time
from itertools import combinations
import sys
import os

os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11")


if __name__=="__main__":
    st = time.time()
    sc = SparkContext("local[*]")
    spark = SparkSession(sc)
    sc.setLogLevel("ERROR")

    filter_threshold = int(sys.argv[1])
    output_file = sys.argv[3]

    users = sc.textFile(sys.argv[2]).map(lambda x: x.split(",")).filter(lambda x: x[0] != "user_id")\
        .groupByKey().map(lambda x: (x[0], list(set(x[1])))).collectAsMap()

    user_pairs = sorted(combinations(list(users.keys()), 2))

    edge_pairs = []
    vertex = set()
    for user in user_pairs:
        if len(set(users.get(user[0])) & set(users.get(user[1]))) >= filter_threshold:
            edge_pairs.append(tuple(user))
            edge_pairs.append((user[1], user[0]))
            vertex.add(user[0])
            vertex.add(user[1])

    vertices = sc.parallelize(list(vertex)).map(lambda x: (x,)).toDF(["id"])
    edges = sc.parallelize(edge_pairs).toDF(["src", "dst"])

    communities = GraphFrame(vertices, edges).labelPropagation(maxIter=5)

    com_res = communities.rdd.coalesce(1).map(lambda x: (x[1], x[0])).groupByKey()\
        .map(lambda x: sorted(list(x[1]))).sortBy(lambda x: (len(x))).collect()

    with open(output_file, "w") as f:
        for i in com_res:
            f.write(str(i).strip("[]") + "\n")
        f.close()

    print("Duration: ", time.time()-st)

