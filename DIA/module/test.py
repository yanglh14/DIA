N = int(input())
M = [[0]*N for _ in range(N)]

pre = [0]*N
for i in range(N):
    pre[i] = (i+1)*(i+2)/2

for i in range(N):
    print(pre[:N-i])
    now = [0]*N
    for j in range(N-i):
        now[j] = pre[i+1]-1
    pre = now[:]


