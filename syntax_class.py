from syntax_test import SyntaxTest
from syntax_test import X
class SyntaxClass:
    def __init__(self, a):
        self.obj = SyntaxTest(a)
    def say(self):
        global X
        self.obj.hello()
        print X
        print self.obj.a
        X = 9
