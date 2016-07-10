X = 3
class SyntaxTest:
    def __init__(self, a):
        global X
        print 'initing'
        self.a = 3 * a
        X = 5
    def hello(self):
        print 'hello'
