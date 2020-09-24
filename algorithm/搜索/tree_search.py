ls=[]
def inorder(node):
    if node:
        inorder(node.left)
        ls.append(node.val)
        inorder(node.right)