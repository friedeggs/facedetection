import random

def foo():
    maxa = 0
    for i in range(3):
        b = random.randint(0, 5)
        b = 0
        if b > maxa:
            maxa = b
            a = b+1
    print maxa, a

if __name__ == '__main__':
    foo()
