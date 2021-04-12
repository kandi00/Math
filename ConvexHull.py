from math import atan
from math import pi
from math import degrees
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import random
from functools import reduce
import operator


#GIFT WRAPPING Algorithm

def gift_wrapping():

     #Random pontok generalasa
     S = []
     x, y = 0, 0
     for i in range(20):
          x = random.randint(0,100)
          y = random.randint(0,100)
          S.append((x, y))

     #S =[(0,0),(0,5),(10,5),(3,3),(10,0)]
     Szelsopontok = []
     i0 = legalacsonyabb_pont_indexe(S)
     i = i0
     h = 0
     while True:
          teta = degrees(2*pi)
          if h == 0:
               elozo= (S[i0][0]-1, S[i0][1])
               h = 1
          for j in range(len(S)):
               if j!=i:
                    szog1 = szog(elozo, (S[i][0], S[i][1]))
                    szog2 = szog((S[i][0], S[i][1]), (S[j][0], S[j][1]))
                    #print("szog2: ",szog2,"szog1:", szog1)
                    if szog2 - szog1 < 0:
                         szog2 += 360
                    if (szog2 - szog1) < teta:
                         teta = szog2 - szog1
                         k = j
                    print(S[i][0], S[i][1], S[k][0], S[k][1])    
          h += 1
          elozo = [S[i][0], S[i][1]]
          Szelsopontok += [elozo]
          i = k
          #print("K", k)
          #print("teta:\n ",teta)
          if i == i0: break
     Szelsopontok += [(S[i][0], S[i][1])]
     #print(Szelsopontok)

     #Abrazolas
     print(Szelsopontok)
     Szelsopontok = np.array(Szelsopontok)
     S = np.array(S)
     plt.plot(S[:,0], S[:,1], 'o')
     plt.plot(Szelsopontok[:,0], Szelsopontok[:,1])
     plt.grid()
     plt.axis('equal')
     plt.show()
     

     return
     
def legalacsonyabb_pont_indexe(S):
     n = len(S)
     mini = S[0][1]
     index = 0
     for i in range(1, n):
          if S[i][1] < mini:
               mini = S[i][1]
               index = i
     return index


def szog(O, P):
##     O = (0,0)
##     P = (0,1)
##     P = (-1,1)
##     P = (-1,0)
##     P = (-1,-1)
##     P = (0, -1)
##     P = (1,-1)
##     P = (0,0)

     Ox, Oy = O
     Px, Py = P
     Px = Px - Ox
     Py = Py - Oy
     Ox = 0
     Oy = 0
     
     if Px == 0:
          if Py>0:
               return degrees(pi/2)
          else:
               return degrees(3*pi/2)
          
     if Px>0:
          if Py>=0:
               teta = atan(Py/Px)
          else:
               teta = 2*pi + atan(Py/Px)
     elif Px<0:
          if Py<0:
               teta = pi + atan(Py/Px)
          else:
               teta = pi + atan(Py/Px)
     return degrees(teta)









#QUICK HULL Algorithm

def hull():

     #Random pontok generalasa
     S = []
     x, y = 0, 0
     for i in range(20):
          x = random.randint(0,100)
          y = random.randint(0,100)
          S.append((x, y))

     #A halmaz ket szelsopontjanak meghatarozasa
     i, j = szelsopontok(S)  # p, q pontok indexe
     k = max_tav(i, j, S)


     #Elkulonitjuk az egyenes folotti es alatti reszeket
     S1 = [ S[i] ]
     S2 = [ S[i] ]
     for elem in S:
          if elem != S[i] and elem!= S[j]:
               if same_side(S[i], S[j], S[k], elem):
                    S1 += [elem]
               else:
                    S2 += [elem]
     S1 += [ S[j] ]
     S2 += [ S[j] ]

     #Meghivjuk mindket reszre a quick_hull fuggvenyt
     S1 = quick_hull(0,len(S1)-1,S1)
     S2 = quick_hull(0,len(S2)-1,S2)

     #Abrazolas
     S = np.array(S)
     plt.plot(S[:,0], S[:,1], 'ro')

     S1.insert(0, S[i])
     S1.append(S[j])
     S1 = np.array(S1)
     plt.plot(S1[:,0], S1[:,1],'bo')
     plt.plot(S1[:,0], S1[:,1])


     S2.insert(0, S[i])
     S2.append(S[j])
     S2 = np.array(S2)
     plt.plot(S2[:,0], S2[:,1], 'bo')
     plt.plot(S2[:,0], S2[:,1])
     
     plt.plot()     
     plt.grid()
     plt.axis('equal')
     plt.show()
     
     return S1, S2


def quick_hull(i,j,S):
     if len(S) < 3:
          return []
     else:
          k = max_tav(i,j, S) # r pont indexe
          A, B = halmazok(S, i, j, k)   # qr tol balra  # rp-tol jobbr
          return quick_hull(0, len(A)-1 , A) + [S[k]] + quick_hull(0, len(B)-1, B)


def szelsopontok(S):
     p = (S[0][0], S[0][1])
     q = (S[0][0], S[0][1])
     k = 0
     i = 0
     j = 0
     for elem in S:
          if elem[0] > q[0] :
               q = elem
               j = k
          if elem[0] < p[0] :
               p = elem
               i = k
          k+=1
     return i, j


def max_tav(i, j, S):
     maxindex = 0
     tavolsag = 0
     a = S[i][1] - S[j][1]
     b = S[j][0] - S[i][0]
     c = S[i][0]*S[j][1] - S[j][0]*S[i][1]

     for l in range(len(S)):
          if l!=i and l!=j:
               if (abs(a*S[l][0] + b*S[l][1]+ c)/sqrt(a*a + b*b)) > tavolsag:
                    maxindex =  l
                    tavolsag = abs(a*S[l][0] + b*S[l][1]+ c)/sqrt(a*a + b*b)                    
     return maxindex
     

#same_side((-1, 1), (1, -1), (2,3), (-4,-5))
def same_side(a, b, p1, p2):
     v = (b[0] - a[0], b[1] - a[1])
     ap1 = (p1[0] - a[0], p1[1] - a[1])
     ap2 = (p2[0] - a[0], p2[1] - a[1])
     v1 = np.cross(v, ap1)
     v2 = np.cross(v, ap2)
     s = np.dot(v1, v2)
     if s<0:
          return False
     else:
          return True


def halmazok(S, i, j, k):
     A = [ S[i] ]
     B = [ S[k] ]
     if len(S) == 1:
          return A, B
     for elem in S:
          if  (not same_side(S[i], S[k], S[j], elem) and same_side(S[k], S[j], S[i], elem)) or(same_side(S[i], S[k], S[j], elem) and  not same_side(S[k], S[j], S[i], elem)):
               if elem[0] > S[k][0]:
                    B += [elem]
               else:
                    A += [elem]
     A += [S[k]]
     B += [S[j]]
     return A, B











#Graham Scan algorithm

def graham_scan():

    S = []
    x, y = 0, 0
    for i in range(20):
         x = random.randint(0,100)
         y = random.randint(0,100)
         S.append((x, y))
    
    if len(S) < 3: 
        return
  
    l = baloldali_pont(S)
    
    S[0], S[l] = S[l], S[0]
    
    p0 = S[0]

    #Rendezes
    X = True
    while(X):
        X = False
        for i in range(1, len(S)-1):
            if compare(S[i], S[i + 1], p0) > 0 :
                var = S[i]
                S[i] = S[i + 1]
                S[i + 1] = var
                
                X = True

    #A p0-hoz legkozelebbi pontot kitoroljuk, ha vannak kollinearis pontok           
    m = 1
    for i in range(1, len(S)):
        while i < len(S)-1 and orientation(p0, S[i], S[i+1] )== 0:
            i += 1
        S[m] = S[i]
        m += 1

    if m < 3:#Kevesebb mint harom elembol nem alkodhato konvex burkolo.
        return

    hull = []
    for i in range(3):
        hull += [S[i]]
    
    for i in range(3, m):
    #Ha orajalassal azonos fordulas jon, akkor kitoroljuk az utolso pontot.
        while orientation(hull[-2], hull[-1], S[i]) != -1:
            hull = hull[:-1]
        hull += [S[i]]

    hull.append(hull[0])
    hull = np.array(hull)
    plt.plot(hull[:,0], hull[:,1], 'bo')
    plt.plot(hull[:,0], hull[:,1])

    plt.plot()     
    plt.grid()
    plt.axis('equal')
    plt.show()

#Meghatarozza a legbaloldalibb pontot. 
def baloldali_pont(S):
    
    left_index = 0
    
    for i in range(1, len(S)): 
        if S[i][0] < S[left_index][0]: 
            left_index = i 
        elif S[i][0] == S[left_index][0]: 
            if S[i][1] > S[left_index][1]: 
                left_index = i 
    return left_index


def orientation(P, Q, R): 
    cr = (Q[1] - P[1]) * (R[0] - Q[0]) - (Q[0] - P[0]) * (R[1] - Q[1]) 
    if cr == 0:  
        return 0 #P,Q,R kollinearis
    elif cr > 0: 
        return 1  #oramutato jarasaval azonos forgas
    else: 
        return -1 #az oramutato jarasaval ellentetes forgas


#oramutatoval ellentetes rendezes
def compare(p1, p2, p0):
    orient = orientation(p0, p1, p2)
    if orient == 0:
        if distSquare(p0, p2) >= distSquare(p0, p1):
            return -1
        else: return 1
    elif orient == -1:
        return -1
    else:
        return 1

def distSquare(P, Q):
    return (P[0]- Q[0])*(P[0]- Q[0])-(P[1]- Q[1])*(P[1]- Q[1])




               
