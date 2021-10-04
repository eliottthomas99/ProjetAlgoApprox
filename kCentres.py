from random import *
import numpy as np


k = 2

P = [ (random()*100, random()*100) for m in range(5) ]
print(f"liste des points")
print(np.array(P))


def distance(p1,p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )


def getCentres(P,dj):
    """
    Renvoie des centres possibles à partir d'une distance
    """
    C=[]
    while(len(P)>0 ): # attention à itérer sur la liste en même temps qu'on la modifie
        indice=randint(0,len(P)-1 )
        p = P[indice]
        R = [ q for q in P if distance(p,q) <= dj  ]
        C.append(p)
        for q in R:
            P.remove(q)
    return C

#print(getCentres(P,30))


def getDistances(P):
    """
    Renvoie les distances d1 , ..., dm entre les n points il y en a 2 parmis n.
    Les renvoyer par ordre croissant.
    """
    distances = []
    for first in range(len(P)):
        for second in range(first+1, len(P)):
            distances.append(distance(P[first],P[second]))

    return sorted(distances)

# print(getDistances(P))

