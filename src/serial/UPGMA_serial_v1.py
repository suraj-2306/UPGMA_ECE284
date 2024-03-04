import copy

######### Distance matrix management #########
def initMat(n):
    return [[0.0 for _ in range(n)] for _ in range(n)]

def mirrorMat(d):
    n = len(d)
    for i in range(n):
        for j in range(i):
            d[j][i] = d[i][j]
    return d

def printMat(d):
    for i in d:
        for j in i:
            print("%4.1f" % (j), end=" ")
        print()
    print()        

def matMinLoc(d):
    smallest = None
    smallest_index = None
    
    for i in range(len(d)):
        for j in range(len(d[i])):
            if d[i][j] > 0:
                if smallest is None or d[i][j] < smallest:
                    smallest = d[i][j]
                    smallest_index = (i, j)
    
    return smallest, smallest_index

def matEntryUpdate(lst, d_ref, d_red, ele1, ele2):
    #print("here1 ", end=' ')
    #print(lst, ele1, ele2)
    ele1_grp, ele2_grp = findCluster(lst, ele1, ele2)
    #print("here2")
    distSum = 0
    distCount = 0
    #print(lst[ele1_grp], end="")
    #print("x", end="")
    #print(lst[ele2_grp], end="  ")
    for i in lst[ele1_grp]:
        for j in lst[ele2_grp]:
            #print(str(d_ref[i][j]) + " + ", end="")
            distSum += d_ref[i][j]
            distCount +=1
    tempRes = distSum/distCount
    #print("Update " + str(ele1) + "x" + str(ele2) + " with " + str(distSum) + "/" + str(distCount) + " = " + str(tempRes))
    d_red[ele1][ele2] = tempRes

######### Cluster management ##############
def initClusterLst(n):    
    lst = []
    for i in range(n):
        lst.append([i])
    return lst

def printClusterLst(lst):
    for i in lst:
        print(i)
    print()

def findCluster(lst, ele1, ele2):    
    ele1_grp = -1
    ele2_grp = -1
    find_count = 0
    for i in range(len(lst)):
        for j in range(len(lst[i])):
            if lst[i][j] == ele1:
                ele1_grp = i
                find_count += 1
            elif lst[i][j] == ele2:
                ele2_grp = i
                find_count += 1
            if find_count == 2:
                return ele1_grp, ele2_grp

def grpLeaf(lst, ele1, ele2):
    ele1, ele2 = min(ele1, ele2), max(ele1, ele2)    
    ele1_grp, ele2_grp = findCluster(lst, ele1, ele2)
    if ele1_grp != ele2_grp:
        for i in lst.pop(ele2_grp):
            lst[ele1_grp].append(i)

######## Main ############################
d = [[0,0,0,0,0,0,0],
     [19,0,0,0,0,0,0],
     [27,31,0,0,0,0,0],
     [8,18,26,0,0,0,0],
     [33,36,41,31,0,0,0],
     [18,1,32,17,35,0,0],
     [13,13,29,14,28,12,0]]
d = mirrorMat(d)
print("Original Distance Matrix")
printMat(d)
cl = initClusterLst(len(d))
cl_hist = []
cl_hist.append((copy.deepcopy(cl)))
d_red = copy.deepcopy(d)
for loop in range(5):
    minLoc = matMinLoc(d_red)[1]
    greaterCombLoc = max(minLoc)
    #print("Greater comb location = " + str(greaterCombLoc))
    grpLeaf(cl, minLoc[0], minLoc[1])
    cl_hist.append(copy.deepcopy(cl))
    for i in range(len(d_red)):
        for j in range(len(d_red[0])):
            if d_red[i][j] != 0:
                if i in minLoc or j in minLoc:
                    if i == greaterCombLoc or j == greaterCombLoc:
                        #print("Reduce " + str(i) + "x" + str(j) + " to 0")
                        d_red[i][j] = 0
                    else:
                        if i != j:
                            matEntryUpdate(cl, d, d_red, i, j)

    #printClusterLst(cl)
    printMat(d_red)
    
for i in cl_hist:
    print(i, end="\n\n")

