
from KDT import *
def time_test():
    for i in range(10):
        print(rand_id())

    dim = 5
    pwd = path.dirname(__file__)
    tmppth = path.join(pwd, '%d' % dim)

    def rand_vec(dim):
        return [random.random() for i in range(dim)]

    a = KDT(tmppth, max_cluster=50)
    for i in range(100):
        v = rand_vec(dim)
        a._add_vec(v)
        # print(a._contains(v))

    def stmt():
        v = rand_vec(dim)
        a._get_nn(v, 10)
    v = rand_vec(dim)

    print(a._get_nn(v, 1), v in a)
    a._add_vec(v)
    print(a._get_nn(v, 1), v in a)
    import timeit
    print(timeit.timeit(stmt=stmt, number=10000))
def enmiao():
    from os import path
    import time
    pwd = path.dirname(__file__)
    tmppth = path.join(pwd, '%d'%time.time())
    a=KDT(tmppth,max_cluster = 2)
    for x in [0,2]:
        for y in [0,2]:
            a._add_vec((x,y))
    print(a._get_nn((1,1),4))
if(__name__=='__main__'):
    enmiao()