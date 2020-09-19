
"""
嵌套的checkbox可以看做是一棵树
"""


class Node:
    def __init__(self,id,father,children):
        self.id = id
        self.father = father
        self.children = children
        self.checked = False

    def _print_id(self):
        print (self.id)

class CheckBox:
    def __init__(self, init_map):
      self.map = init_map
      self.nodes = {}
      self._create_tree()

    def _create_tree(self):
        for k,v in self.map.items():
            node = Node(k,v['father'],v['children'])
            self.nodes[k] = node
    
    def check_node(self,ni):
        assert self.nodes
        ni  = str(ni) if type(ni) != str else ni
        node = self.nodes[ni]
        node.checked = True # check this
        father = node.father
        if father==None or self._check_upward_same_layer_not_all_checked_condition(node):
            pass
        else:
            self._upward(node)
        
        if node.children != None:
            self._pass_down(node)
        
        
    def _upward(self,node):
        # father node
        assert node.father
        father_node = self.nodes[node.father]
        father_node.checked = True
        if father_node.father == None or self._check_upward_same_layer_not_all_checked_condition(father_node):
            pass
        else:
            _upward(self,father_node.father)

    def _pass_down(self,node):
        pass
    
    def _check_upward_same_layer_not_all_checked_condition(self,node):
        father_id = node.father # not None
        assert  father_id
        children_id_list = self.nodes[father_id].children
        for child in children_id_list:
            nodei = self.nodes[child]
            if nodei.checked == False:
                return True
        return False
    
    def _reset(self):
        for k,v in self.nodes.items():
            v.checked = False

if __name__ == "__main__":
    demo_tree =  {
    "1":{'father':None,'children':['2','3']},
    "2":{'father':'1','children':['4','5']},
    "3":{'father':'1','children':None},
    "4":{'father':'2','children':None},
    "5":{'father':'2','children':['6']},
    "4":{'father':'2','children':None},
    "6":{'father':'5','children':None}
    }

    checkbox = CheckBox(demo_tree)
    print (1)

    # test case
    checkbox.check_node(6)
    checkbox.check_node(4)
    
    assert checkbox.nodes['1'].checked == False and checkbox.nodes['2'].checked == True
    checkbox._reset()

