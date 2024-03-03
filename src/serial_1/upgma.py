import pandas as pd
def excel_to_2d_list(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    return df.values.tolist()

def isValidRow(row,label_rows):
    if(label_rows[row][1] == 1):
        return True
    return False

def isValidCols(col,label_cols):
    if(label_cols[col][1] == 1):
        return True
    return False

def getLastValidCol(label_cols):
    index = 0
    get_index = 0
    for items in label_cols:
        if(items[1] == 1):
            get_index = index
        index = index + 1
    return get_index

def print_matrix(row_title, col_title, mylist, size_row, size_col):
    print(end="\t")
    for col in range(size_col):
        if( isValidCols(col,col_title) ):
            print(col_title[col][0],end="\t")
    print()
    for row in range(size_row):
        if( isValidRow(row,row_title) ):
            print(row_title[row][0],end="\t")
            for col in range(size_col):
                if( isValidCols(col,col_title) ):
                    if(row > col):
                        print(mylist[row][col],end="\t")
                    else:
                        print(end="\t")
            print()
    print()

def min_list(mylist, size_row,size_col, label_rows, label_cols):
    min = mylist[1][0]
    r = 1
    c = 0
    flag = 0
    for rows in range(size_row):
        if(isValidRow(rows,label_rows)):
            for cols in range(size_col):
                if(isValidCols(cols,label_cols)):
                    if(rows > cols):
                        if(flag == 0):
                            min = mylist[rows][cols]
                            r = rows
                            c = cols
                            flag = 1
                        if( mylist[rows][cols] < min ):
                            min = mylist[rows][cols]
                            r = rows
                            c = cols
    return min,r,c

file_path = 'data.xlsx'
sheet_name = 'Sheet1'
two_d_list = excel_to_2d_list(file_path, sheet_name)

print(len(two_d_list))

# exit(0)
label_cols = []
label_rows = []
SIZE_MAT = len(two_d_list)
ch = ord('A')
for i in range(SIZE_MAT):
    label_cols.append([chr(ch),1])
    label_rows.append([chr(ch),1])
    ch = ch + 1

# label_cols = [['A',1],['B',1],['C',1],['D',1],['E',1],['F',1],['G',1]]
# label_rows = [['A',1],['B',1],['C',1],['D',1],['E',1],['F',1],['G',1]]
distance_matrix = two_d_list
# print_matrix(label_rows,label_cols,distance_matrix,SIZE_MAT,SIZE_MAT)

label_rows[0][1] = 0
label_cols[SIZE_MAT-1][1] = 0

def mult_factor(row,col,label_rows,label_cols):
    return len(label_rows[row][0])*len(label_cols[col][0])

valid_rows = SIZE_MAT
valid_cols = SIZE_MAT
size_mat = SIZE_MAT

nodes = []
# answer = [[],[],[],[],[],[],[],[],[],[],[],[]]
answer = []

while size_mat > 1 :
    min_val,r,c = min_list(distance_matrix,SIZE_MAT,SIZE_MAT, label_rows,label_cols)
    col1 = min(r,c) #Lexicographically smaller col
    col2 = r+c-col1 
    row1 = col1 #Lexicographically smaller row
    row2 = col2

    #Go through all rows of a column
    for i in range(SIZE_MAT):
        #column will be valid becasue min function takes care of it
        if(isValidRow(i,label_rows)):
            second_coord = (row2,i)
            if(row2 <= i):
                second_coord = (i,row2)
            mf1 = mult_factor(i,col1,label_rows,label_cols)
            mf2 = mult_factor(second_coord[0],second_coord[1],label_rows,label_cols)
            if(i>col1):
                distance_matrix[i][col1] = round((distance_matrix[i][col1]*mf1 + distance_matrix[second_coord[0]][second_coord[1]]*mf2)/(mf1+mf2),2)
    
    #Go through all columns of a row
    for i in range(SIZE_MAT):
        if(isValidCols(i,label_cols)):
            second_coord = (i,col2)
            if( i <= col2 ):
                second_coord = (col2,i)
            mf1 = mult_factor(row1,i,label_rows,label_cols)
            mf2 = mult_factor(second_coord[0],second_coord[1],label_rows,label_cols)
            if(row1 > i):
                distance_matrix[row1][i] = round((distance_matrix[row1][i]*mf1 + distance_matrix[second_coord[0]][second_coord[1]]*mf2)/(mf1+mf2),2)

    # Change Labels
    old_label = label_cols[col1][0]
    new_label = label_rows[row2][0]
    label_cols[col1][0] = old_label + new_label
    label_rows[row1][0] = label_rows[row1][0] + label_cols[col2][0]
    nodes.append(label_cols[col1][0])

    # answer[len(old_label)].append(old_label)
    # answer[len(new_label)].append(new_label)

    answer.append([old_label,new_label,label_cols[col1][0]])

    #Change Valid Status
    if(label_cols[col2][1] == 0):
        label_cols[getLastValidCol(label_cols)][1] = 0
    else:
        label_cols[col2][1] = 0

    label_rows[row2][1] = 0
    
    #print_matrix(label_rows,label_cols,distance_matrix,SIZE_MAT,SIZE_MAT)
    size_mat = size_mat - 1

# print(answer)
# for items in answer:
#     for items_1 in items:
#         print(items_1,end="\t")
#     print()

SPACER = "|\t"
new_answer = list(reversed(answer))
# print(new_answer)
def recursive_print(tree_list, spacer):
    # print(tree_list)
    if(len(tree_list) == 0 ):
        return
    else:
        print(spacer+"|----"+tree_list[0][2])
        left = tree_list[0][0]
        if(len(left) == 1):
            print(spacer+SPACER+"|----"+left)
        index = 0
        for items in tree_list:
            if(left == items[2]):
                recursive_print(tree_list[index:],spacer+SPACER)
                break
            index=index+1
        right = tree_list[0][1]
        if(len(right) == 1):
            print(spacer+SPACER+"|----"+right)
        index = 0
        for items in tree_list:
            if(right == items[2]):
                recursive_print(tree_list[index:],spacer+SPACER)
                break
            index=index+1
        # print(spacer+left)
recursive_print(new_answer,"")

