#https://blog.csdn.net/qq_31244453/article/details/104657532?utm_medium=distribute.pc_relevant.none-task-blog-title-1&spm=1001.2101.3001.4242

class farther(object):
    def __init__(self):
        self.x = '这是属性'
        
    def fun(self):
        print(self.x)
        print('这是方法')
        
class son(farther):
    def __init__(self):
        super(son,self).__init__()
        print('实例化执行')
            
test = son()
test.fun()
test.x

