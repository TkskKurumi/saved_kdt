from math import sqrt
import json
import random,heapq,os
from glob import glob
from os import path
_inf=float('inf')
eps=1e-6
def base32(x,length=13):
    ch='01234567890abcdefghijklmnopqrstuvwxyz'
    mask=0b11111
    ret=[]
    for i in range(length):
        ret.append(ch[x&mask])
        x>>=5
    return ''.join(ret[::-1])
def rand_id():
    x=0
    for i in range(8):
        x=(x<<8)|random.randrange(256)
    return base32(x)
def load_json(pth):
    with open(pth,'r') as f:
        ret=json.load(f)
    return ret
def save_json(pth,j):
    if(not path.exists(path.dirname(pth))):
        os.makedirs(path.dirname(pth))
    with open(pth,'w') as f:
        json.dump(j,f)
    return pth
def dist(vec1,vec2):
    ret=0
    for idx,i in enumerate(vec1):
        ret+=(i-vec2[idx])**2
    return sqrt(ret)
def mean(vec):
    return sum(vec)/len(vec)
def variant(vec):
    ret=0
    mea=mean(vec)
    for i in vec:
        ret+=(i-mea)**2
    return sqrt(ret/len(vec))
class node:
    _saved_value=set([
    'left_child',
    'right_child',
    'vecs',
    'split_dim',
    'split_value',
    'name',
    'save_path'
    ])
    def construct_none():
        ret=dict()
        for i in node._saved_value:
            ret[i]=None
        return ret
    def __setattr__(self,name,value):
        self.__dict__[name]=value
        if(name in node._saved_value):
            if(getattr(self,'save_path',None)):
                self.save(self.save_path)
        
    def __init__(self,left_child,right_child,vecs,split_dim,split_value,name,save_path=None):
        
        #
        self.left_child=left_child
        self.right_child=right_child
        self.vecs=vecs
        self.split_dim=split_dim
        self.split_value=split_value
        self.name=name
        self.save_path=save_path
        #print(getattr(self,'save_path'))
    def as_dict(self):
        ret=dict()
        for i in node._saved_value:
            ret[i]=self.__getattribute__(i)
        return ret
    def from_json(pth):
        return node.from_dict(load_json(pth))
    def from_dict(dict):
        return node(**dict)
    def save(self,pth=None):
        if(pth is None):
            pth=getattr(self,'save_path',None)
            if(pth is None):
                return
        d=self.as_dict()
        save_json(pth,d)
    def calc_branch(self,vec):
        spl_val=self.split_value
        vec_val=vec[self.split_dim]
        a=(vec_val-spl_val,self.left_child)
        b=(spl_val-vec_val,self.right_child)
        return (a,b)
    def calc_dists(self,vec):
        ret=[]
        for v in self.vecs:
            ret.append((dist(v,vec),v))
        return ret
    def is_leaf(self):
        return not(self.vecs is None)
    def split(self):
        n_dim=len(self.vecs[0])
        var_dim=[]
        for dim in range(n_dim):
            values=[vec[dim] for vec in self.vecs]
            var_dim.append((variant(values),dim))
        _,dim=max(var_dim)
        values=[vec[dim] for vec in self.vecs]
        spl_val=mean(values)
        
        l_vecs=[]
        r_vecs=[]
        for v in self.vecs:
            if(v[dim]<spl_val):
                l_vecs.append(v)
            else:
                r_vecs.append(v)
        lc=node.construct_none()
        lc['name']=rand_id()
        lc['vecs']=l_vecs
        
        rc=node.construct_none()
        rc['name']=rand_id()
        rc['vecs']=r_vecs
        
        lc=node(**lc)
        rc=node(**rc)
        self.left_child=lc.name
        self.right_child=rc.name
        self.split_value=spl_val
        self.split_dim=dim
        self.vecs=None
        return lc,rc
class KDT:
    def __init__(self,save_dir,max_cluster=10):
        self.nodes={}
        self.save_dir=save_dir
        self.max_cluster=max_cluster
    def _empty(self):
        return not path.exists(self._node_pth('root'))
    def _node_pth(self,uid):
        return path.join(self.save_dir,uid+'.json')
    def _get_node(self,uid):
        if(uid in self.nodes):
            return self.nodes[uid]
        pth=self._node_pth(uid)
        if(path.exists(pth)):
            self.nodes[uid]=node.from_json(pth)
            self.nodes[uid].save_path=pth
            return self.nodes[uid]
        else:
            raise KeyError(uid)
    def _get_nn(self,vec,n,search_k=64):
        root=self._get_node('root')
        
        nodes=[(0,'root')]
        rets=[]
        def push_node(recall_dist,node):
            nonlocal nodes
            heapq.heappush(nodes,(recall_dist,node))
        def pop_node():
            nonlocal nodes
            return heapq.heappop(nodes)
        def add_ret(ret):
            nonlocal rets
            heapq.heappush(rets,ret)
            if(len(rets)>n):
                heapq.heappop(rets)
        def worst_ret():
            nonlocal rets
            if(len(rets)<n):
                return _inf
            return rets[0][0]
        while(nodes):
            recall_dist,node=pop_node()
            if(worst_ret()!=_inf):
                if(recall_dist>worst_ret()):
                    break
                if(recall_dist>0):
                    if(search_k<=0):
                        break
                    search_k-=1
            node=self._get_node(node)
            if(node.is_leaf()):
                node_rets=node.calc_dists(vec)
                for d,v in node_rets:
                    add_ret((-d,v,node.name))
            else:
                for recall_dist,child in node.calc_branch(vec):
                    push_node(recall_dist,child)
        return [(-d,v,n) for d,v,n in rets]
    def _construct_root(self,vec):
        d={}
        for i in node._saved_value:
            d[i]=None
        d['vecs']=[vec]
        d['name']='root'
        d['save_path']=self._node_pth('root')
        return node(**d)
    def _add_vec(self,vec):
        if(self._empty()):
            self.nodes['root']=self._construct_root(vec)
        else:
            u=self._get_node('root')
            while(not u.is_leaf()):
                _,child=min(u.calc_branch(vec))
                u=self._get_node(child)
            u.vecs.append(vec)
            u.save()
            if(len(u.vecs)>self.max_cluster):
                lc,rc=u.split()
                lc.save_path=self._node_pth(lc.name)
                rc.save_path=self._node_pth(rc.name)
    def _contains(self,vec):
        if(self._empty()):
            return False
        else:
            u=self._get_node('root')
            while(not u.is_leaf()):
                _,child=min(u.calc_branch(vec))
                u=self._get_node(child)
            node_rets=u.calc_dists(vec)
            dist=min(node_rets)[0]
            return dist<eps
               
if(__name__=='__main__'):
    for i in range(10):
        print(rand_id())
    
    dim=5
    pwd=path.dirname(__file__)
    tmppth=path.join(pwd,'%d'%dim)
    
    def rand_vec(dim):
        return [random.random() for i in range(dim)]
    
    a=KDT(tmppth,max_cluster=50)
    for i in range(100):
        v=rand_vec(dim)
        a._add_vec(v)
        #print(a._contains(v))
    def stmt():
        v=rand_vec(dim)
        a._get_nn(v,10)
    import timeit
    print(timeit.timeit(stmt=stmt,number=10000))