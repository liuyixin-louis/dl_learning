# Definition for a binary tree node.
class Tree():
    def __init__(self):
        self.root = TreeNode(2147483647)
        # self.root.left = TreeNode(5)
        # self.root.right = TreeNode(15)
        # self.root.left.left = TreeNode(5)

class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

count = {} # global count for val

def dfs(root):
    global count
    if root:
        count[root.val] = 1 if root.val not in count else count[root.val]+1
        dfs(root.left)
        dfs(root.right)

open = [] # open set for bfs
def bfs(root):
    global count,open
    if root:
        count[root.val] = 1 if root.val not in count else count[root.val]+1
        # if root.left:
        open.append(root.left)
        # if root.right:
        open.append(root.right)
        # if len(open) != 0:
        bfs(open.pop(0))


# copy from other# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


import collections
class Solution:
    def findMode(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        dicts = {}
        deque = collections.deque()
        deque.append(root)
        while len(deque)>0:
            currVal = deque.popleft()
            if currVal.val not in dicts:
                dicts[currVal.val] = 1
            else:
                dicts[currVal.val]+=1
            if currVal.left:
                deque.append(currVal.left)
            if currVal.right:
                deque.append(currVal.right)
        maxVal =  max(dicts.values())
        res = []
        for item,val in dicts.items():
            if val>=maxVal:
                res.append(item)
        return  res


def inorder_morris(root):
    ans = []
    nums = 0         #记录 当前正在记录的 数字
    nlen = 0         #记录当前数字长度
    mlen = 0         #记录最大长度
    def innode(root):
        nonlocal nlen 
        nonlocal mlen
        nonlocal nums
        nonlocal ans
        #先遍历右边子树
        if root.right:
            innode(root.right)
        #将遍历到最下面的数作为初始节点
        if nlen == 0:
            nums = root.val     
            nlen = 1            
            mlen = nlen         
            ans = []
            ans.append(nums)    
        #判断当前数字和正在记录数字一样的情况
        elif root.val == nums:
            nlen += 1
            if nlen>mlen:
                ans = []
                ans.append(nums)
                mlen = nlen
            elif nlen==mlen and not nums in ans:
                ans.append(nums)   
        #判断当前数字和正在记录的数字不一样的情况
        else:
            nums = root.val
            nlen = 1
            if nlen == mlen:
                ans.append(nums)
        #再遍历左边子树
        if root.left:
            innode(root.left)
    if root:
        innode(root)
    else:
        return []
    return ans


def test_dfs(root):
    dfs(root)
    mode_count = max(count.values())
    count2val = {v:k for k,v in count.items()}
    mode = count2val[mode_count]
    return mode

def test_bfs(root):
    bfs(root)
    mode_count = max(count.values())
    # count2val = {v:k for k,v in count.items()}
    # mode = count2val[mode_count]
    # return mode
    res = []
    for k,v in count.items():
        if v == mode_count:
            res.append(k)
    return res

def test_inorder_morris(root):
    return inorder_morris(root)

if __name__ == '__main__':
    tree = Tree()
    root = tree.root
    # res = test_dfs(root)
    res = test_bfs(root)
    # res = test_inorder_morris(root)
    print(res)
    