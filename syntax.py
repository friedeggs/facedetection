class A:
    def helpMe(self, b):
        b.id = 5
        b.fine()
        return b

class B:
    def __init__(self, val):
        self.id = val
    def fine(self):
        print "fine", self.id
    def hahaNo(self):
        a = A()
        a.helpMe(self)

class C:
    def __init__(self, settings):
        for attr, val in settings.iteritems():
            setattr(self, attr, val)

class Tree:
    def __init__(self, node, leftTree=None, rightTree=None):
        self.node = node
        self.leftTree = leftTree
        self.rightTree = rightTree
    def isLeaf(self):
        return self.leftTree is None and self.rightTree is None
    def splits(self):
        if self.isLeaf():
            return []
        leftSplits = self.leftTree.splits() if self.leftTree else []
        rightSplits = self.rightTree.splits() if self.rightTree else []
        return [self.node] + leftSplits + rightSplits
        # return [self.node] + self.leftTree.splits() + self.rightTree.splits()

def output(str):
    print str, 123

if __name__ == '__main__':
    # output("ASdfs %d" % 423)
    log = "Fitting weak regressor %d of %d" % (2, 3)
    print '{:<{}s}'.format(log, 30)

    # t = Tree('a', Tree(1), Tree('b', Tree(2), Tree('c', Tree(4), Tree(5))))
    # print t.isLeaf(), t.leftTree.isLeaf()
    # print t.splits()

    # b0 = B(1)
    # b0.hahaNo()
    # print b0.id

    # settings = {
    #     "P": 400,
    #     "S": 20
    # }
    # c = C(settings)
    # print c, c.P, c.S