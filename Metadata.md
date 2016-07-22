face_detector:


face_detector_2:
K = 10
N = 20
F = 4
T = 10

face_detector_3:
T = 5
K = 100
F = 5
P = 100
S = 20
n = 100
R = 4

10-15s:
lr = 0.1
T = 1
K = 50
F = 5
P = 400
S = 20
n = 100
R = 10 # Use 1 initialization instead? >:(
N = n*R
lmbda = 0.02

20s:

60_0.2:
lr = 0.5
T = 1
K = 60
F = 5
P = 400
S = 20
n = 20
R = 4 # Use 1 initialization instead? >:(
N = n*R
lmbda = 0.03

100:
lr = 0.2
T = 1
K = 100
F = 5
P = 400
S = 20
n = 200
R = 5 # Use 1 initialization instead? >:(
N = n*R
lmbda = 0.05
