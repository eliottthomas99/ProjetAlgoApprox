from random import *
import matplotlib.pyplot as plt
from numpy.random.mtrand import poisson
from sklearn.datasets import make_blobs
import numpy as np
import itertools



centers = [(10, 80), (20, 20),(80, 20), (80, 60)]
cluster_std = [8, 10,5,7]

k = len(centers)
P, y = make_blobs(n_samples=10, cluster_std=cluster_std, centers=centers, n_features=k, random_state=42)
colors=["red", "blue", "green", "purple"]



#P = [ (random()*100, random()*100) for m in range(10) ]
print(f"liste des points")
P=P.tolist() # pour ne pas avoir de pb avec le array
#print(type(P))
#print(P)


def distance(p1,p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )


def getCentres(P,dj):
    """
    Renvoie des centres possibles à partir d'une distance
    """
    C=[]
    Pcopy = P[:]
    while(len(Pcopy)>0 ): # attention à itérer sur la liste en même temps qu'on la modifie
        indice=randint(0,len(Pcopy)-1 )
        p = Pcopy[indice]
        R = [ q for q in Pcopy if distance(p,q) <= dj  ]
        C.append(p)
        for q in R:
            Pcopy.remove(q)
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

def kCentresApprox(P,k):
    D = getDistances(P)
    for i in range(0,len(D)-1):
        C = getCentres(P, 2*D[i])
        if(len(C) <=k ): 
            return (C, 2*D[i])
    


def kCentresBrutForce(P,k):

    #iterer sur tous les k-uplets de centres possibles

    k_uplets = list(itertools.combinations(P, k))
    

    best_dist = float('inf')
    dict_sol=dict()

    for brut_centres in k_uplets: #brut_centres representent une combinaison de centres de taille k. 
        #dist_max_centre = float('inf')

        dict_sol_temporaire=dict() #pour stocker l'association points centres temporaire

        for point in P: #on itere sur tous les points
            dist_point_min = float('inf')
            
            for b_centre in brut_centres: #on itere sur tous les centres de la combinaison en cours
                dist = distance(point,b_centre)
                if(dist<dist_point_min ):
                    dist_point_min = dist
                    p_centre = b_centre 

            dict_sol_temporaire[tuple(point)] = (p_centre,dist_point_min) #on assigne le meilleur centre pour chaque point
        
        distances = [d[1] for d in dict_sol_temporaire.values() ]

        dist_point_max=0
        for point_max_temp in dict_sol_temporaire.keys():
            #print("dicttemp",dict_sol_temporaire[point_max_temp][1])
            #print(dist_point_max)

            if(dict_sol_temporaire[point_max_temp][1] > dist_point_max ):
                dist_point_max = dict_sol_temporaire[point_max_temp][1]
                point_max = point_max_temp
        
        if(dist_point_max<best_dist):
            best_dist = dist_point_max
            dict_sol = dict_sol_temporaire
            best_sol = brut_centres
            point_far = point_max

    return (best_sol,best_dist,point_far,dict_sol)








print("result")
result = kCentresApprox(P,k)
deux_di = result[1]
centres = result[0] 

resultBrut = kCentresBrutForce(P,k)
centres_brut = resultBrut[0]
distance_brut = resultBrut[1]
point_far_brut = resultBrut[2]
dict_brut = resultBrut[3]
print("CENTRES")
print(centres_brut)

print("AFFICHAGE")

#Xs = [P[k][0] for k in range(len(P))]
#print(P)
#Ys = [P[k][1] for k in range(len(P))]

fig, ax = plt.subplots()

for centre_aff in range(k):
    plt.scatter(np.array(P)[y == centre_aff, 0], np.array(P)[y == centre_aff, 1], color=colors[centre_aff], s=10 )

#Centres
for centre in range(len(centres)):
    circle = plt.Circle(( centres[centre][0] , centres[centre][1] ), deux_di, color="r", fill=False ) 
    ax.add_artist(circle)


Xs=[point_far_brut[0],(dict_brut[point_far_brut])[0][0]]
Ys=[point_far_brut[1],(dict_brut[point_far_brut])[0][1]]
plt.plot(Xs,Ys)

ax.axis("equal")
plt.xlim(0,100)
plt.ylim(0,100)

plt.title('K-centres implementation', fontsize=8)

plt.show()




plt.show()


