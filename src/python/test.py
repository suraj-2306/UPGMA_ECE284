def initMat(n):
    return [[0.0 for _ in range(n)] for _ in range(n)]

def mirrorMat(d):
    n = len(d)
    for i in range(n):
        for j in range(i):
            d[j][i] = d[i][j]
    return d

def matInt2Float(d):
    return [[float(x) for x in row] for row in d]

def printMat (d):
    for i in d:
        for j in i:
            print("%4.1f" % (j), end=" ")
        print()
        

def matMin(d):
    smallest = None
    smallest_index = None
    
    for i in range(len(d)):
        for j in range(len(d[i])):
            if d[i][j] > 0:
                if smallest is None or d[i][j] < smallest:
                    smallest = d[i][j]
                    smallest_index = (i, j)
    
    return smallest, smallest_index

def matUpdate(d):
    print()

def initClusterLst(n):
    lst = []
    for i in range(n):
        lst.append({})
    return lst

def printClusterLst(lst):
    for i in lst:
        print(i)


d = [[0,0,0,0,0,0,0],
     [19,0,0,0,0,0,0],
     [27,31,0,0,0,0,0],
     [8,18,26,0,0,0,0],
     [33,36,41,31,0,0,0],
     [18,1,32,17,35,0,0],
     [13,13,29,14,28,12,0]]



d = mirrorMat(d)
d_red = d

for z in range (3):
    min_r, min_c = matMin(d_red)[1]
    comb_idx = min([min_r, min_c])
    rem_idx = max([min_r, min_c])
    for i in range (len(d)):
        for j in range (len(d)):
            if i == rem_idx or j == rem_idx:
                d_red[i][j] = 0
            else:
                if i == comb_idx:
                    #for k in range (i):
                    print("row update " + str(d[comb_idx][j]) + " " + str(d[rem_idx][j]) + " " + str((d[comb_idx][j] + d[rem_idx][j])/2))
                    d_red[i][j] = (d[comb_idx][j] + d[rem_idx][j])/2
                elif j == comb_idx:
                    #for k in range (comb_idx+1, len(d)):
                    print("col update " + str(d[i][comb_idx]) + " " + str(d[i][rem_idx]) + " " + str((d[i][comb_idx] + d[i][rem_idx])/2))
                    d_red[i][j] = (d[i][comb_idx] + d[i][rem_idx])/2
                else:
                    d_red[i][j] = d[i][j]

printMat(d_red)
            


