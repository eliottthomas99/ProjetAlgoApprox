from random import *
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean
from numpy.random.mtrand import poisson
from sklearn.datasets import make_blobs
import numpy as np
import itertools
import time 




def distance(p1,p2):
    """
    Renvoie la distance euclidienne entre 2 points
    """
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )


def getCentres(P,dj):
    """
    Renvoie des centres possibles à partir d'une distance
    """
    C=[] # la liste des centres
    Pcopy = P[:] # Pcopy copie les valeurs de P sans le référencer
    while(len(Pcopy)>0 ): 
        indice=randint(0,len(Pcopy)-1 ) # On selectionne ainsi un point au hasard
        p = Pcopy[indice]
        R = [ q for q in Pcopy if distance(p,q) <= dj  ] # la liste des points qui ont été marqués centre ou voisinage de centre
        C.append(p) # on ajoute p à la liste des centres
        for q in R:
            Pcopy.remove(q)
    return C


def adjMatrix(P):
    """
    On fabrique une matrice d'adjacence à partir des coordaonnées des points pour simuler un graphe
    Mij correspond à la distance de i à j. 
    """
    M=[]
    for i in range(len(P)):
        M.append([])
        for j in range(len(P)):
            M[-1].append(distance(P[i],P[j]))
    return M


def sortedEdges(P,M):
    """
    Renvoie la liste des arêtes triée par poids des arêtes : ici la distance entre 2 points.  
    """
    E = []
    for i in range(len(P)):
        for j in range(i):
            E.append( (M[i][j], (i,j) ))
    
    sortedE  = sorted(E, key = lambda dist: dist[0])
    return sortedE



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


def kCentresApprox(P,k):
    """
    2-approximation vue en cours
    """
    D = getDistances(P) # cf ci-dessus. 
    for i in range(0,len(D)-1): # on itère sur toutes les distances possibles. 
        C = getCentres(P, 2*D[i])
        if(len(C) <=k ): # si la solution proposée comporte au maximum k éléments on la renvoie. 
            return (C, 2*D[i])

def getMaxDist(P,C):
    """
    Renvoie toutes les informations necessaires sur la distance maximum à un centre
    """

    dict_sol=dict() #pour stocker l'association points centres 
    for point in P: #on itere sur tous les points
        dist_point_min = float('inf')
        
        for b_centre in C: #on itere sur tous les centres de la combinaison en cours
            dist = distance(point,b_centre)
            if(dist<dist_point_min ):
                dist_point_min = dist
                p_centre = b_centre 

        dict_sol[tuple(point)] = (p_centre,dist_point_min) # on doit transformer "point" en tuple car les listes ne sont pas hashables
    
    dist_point_max=0
    for point_max_temp in dict_sol.keys():
            
            if(dict_sol[point_max_temp][1] > dist_point_max ):
                dist_point_max = dict_sol[point_max_temp][1]
                point_max = point_max_temp
        
    return (point_max,dist_point_max,dict_sol) # on renvoie le point le plus loin, la distance maximum au centre le plus proche et le dictionaire des relations points-centres.

def kCentresBrutForce(P,k):
    """
    Renvoie la meilleure solution de k-centres pour le problème considéré.
    Attention, c'est une solution par force brute donc elle est très rapidement dépasssée.
    """


    #iterer sur tous les k-uplets de centres possibles
    k_uplets = list(itertools.combinations(P, k))
    
    best_dist = float('inf')
    dict_sol=dict()

    for brut_centres in k_uplets: #brut_centres representent une combinaison de centres de taille k. 

        dict_sol_temporaire=dict() #pour stocker l'association points centres temporaire

        for point in P: #on itere sur tous les points
            dist_point_min = float('inf')
            
            for b_centre in brut_centres: #on itere sur tous les centres de la combinaison en cours
                dist = distance(point,b_centre)
                if(dist<dist_point_min ):
                    dist_point_min = dist
                    p_centre = b_centre 

            dict_sol_temporaire[tuple(point)] = (p_centre,dist_point_min) #on assigne le meilleur centre pour chaque point
        
        dist_point_max=0
        for point_max_temp in dict_sol_temporaire.keys():
            
            # on cherche la pire distance de la solution en cours
            if(dict_sol_temporaire[point_max_temp][1] > dist_point_max ):
                dist_point_max = dict_sol_temporaire[point_max_temp][1]
                point_max = point_max_temp
        
        # si cette solution est meilleure que la précédente on met à jour les données
        if(dist_point_max<best_dist):
            best_dist = dist_point_max
            dict_sol = dict_sol_temporaire
            best_sol = brut_centres
            point_far = point_max

    return (best_sol,best_dist,point_far,dict_sol)




def adjacences(Adj,x,we):
    """
    Renvoie les indices des points accessibles depuis x avec un coût inférieur ou égal à we.
    """
    res  = []
    for v in range(len(Adj[x])): # v représente les points adjacents à x
        
        if(Adj[x][v] <= we ):
            res.append(v)
    return res



def kCenterBestHeuristic(P,k):
    """
    2-aproximation par Dorit S HAUCHBAUM et David B SHMOYS
    """

    if(k==len(P)): # Si l'on souhaite autant de centres qu'il y a de points, on renvoie tous les points
        return P
    
    Adj = adjMatrix(P) # la matrice d'adjacence des points (basée sur les indices dans la liste P)
    se = sortedEdges(P,Adj) # la liste des distances triées par ordre croissant
    m = len(se) # le nombre d'arêtes
    low = 1 # si on renvoie tous les points
    high = m # si on ne renvoie qu'un point
    while(high != low+1): # On va effectuer une recherche dichotomique
        mid  = (low+high)//2 
        wmid = se[mid][0] # le poids de l'arête du milieu
        S = [] # la solution temporaire
        T = [indice for indice in range(len(P)) ] # on stoque dans T les indices des points de P
        while(len(T) > 0): # Tant qu'il reste des points à classer
            rdIndice = randint(0,len(T)-1) # on choisis un point au hasard
            x  = T[rdIndice] # x représente un point disponible, stoqué sous forme d'indice
            S.append(x) 
            adjx = adjacences(Adj,x,wmid) # représente tous les points accessibles depuis x avec un coût inférieur à wmid 
            for v in adjx: 
                adjv = adjacences(Adj,v,wmid) 
                for voisinV in adjv:
                    if(voisinV in T):
                        T.remove(voisinV)
        if(len(S)<=k): #Si on a k centres ou moins, on peut tenter d'avoir une meilleur solution en gardant moins d'arêtes disponibles.
            high=mid
            Sfin = S
        else: # Si la solution proposée comporte plus de k centres, c'est qu'il faut s'autoriser plus d'arêtes, donc augmenter le poids mid.
            low=mid
    return Sfin       


# TESTS

# TESTS SUR DE NOMBREUSES INSTANCES
"""
k =4
ratiosAPP = [] # listes des ratios d'aproximation obtenus par la 2-approximation du cours
ratiosAPPHEU = [] # listes des ratios d'aproximation obtenus par la 2-approximation "Best Heuristic"
t1 = time.perf_counter() # Pour mesurer le temps passé à effectueer les tests, en partiulier pour jauger du nombre de points par instance
for _ in range(100): # on test sur un maximum d'instances possibles
    
    centers = [(randint(1,100), randint(1,100)) for _ in range(k)] # les centres changent à chaque fois
    cluster_std = [randint(1,20) for _ in range(k)] # l'"éparpillement" de chaque cluster change à chaque fois

    P, y = make_blobs(n_samples=10, cluster_std=cluster_std, centers=centers, n_features=k) #  random_state=42
    P=P.tolist()

    resultHEU = kCenterBestHeuristic(P,k) # solution renvoyée par "Best heuristic"
    centers_heu =[]
    for i in resultHEU:
        centers_heu.append(P[i])

    maxDistHEU = getMaxDist(P,centers_heu)
    distAPPHEU  =maxDistHEU[1]



    result = kCentresApprox(P,k) # solution renvoyée par l'algo du cours
    centres = result[0] 
    maxDist = getMaxDist(P,centres)
    distAPP = maxDist[1]

    resultBrut = kCentresBrutForce(P,k) # solution optimale
    distOPT = resultBrut[1]

    ratiosAPP.append(distAPP/distOPT)
    ratiosAPPHEU.append(distAPPHEU/distOPT)


print(f"max ratioAPP {max(ratiosAPP)}")
print(f"avg ratioAPP {mean(ratiosAPP)}")

print(f"max ratioAPPHEU {max(ratiosAPPHEU)}")
print(f"avg ratioAPPHEU {mean(ratiosAPPHEU)}")
t2=time.perf_counter()
print(f"temps écoulé :{t2-t1} secondes")
"""




# AFFICHAGE 

k = 4
centers = [(10, 80), (20, 20),(80, 20), (80, 60)] # les coordonnées des centres des blobs
cluster_std = [8, 10,5,7] # les écarts type de chaque blob (plus d'écart = plus d'eclatement du blob)

P, y = make_blobs(n_samples=10, cluster_std=cluster_std, centers=centers, n_features=k,random_state=42)   
colors=["red", "blue", "green", "purple"] # pour pouvoir différencier les blobs

P=P.tolist() # pour transformer l'array en list. 



solHeu = kCenterBestHeuristic(P,k) # solution de "Best Heuristic"
print("centersHEU")
centers_heu =[]
for i in solHeu:
    print(P[i])
    centers_heu.append(P[i])
print("--------------------")
print("HEURISTIC")
maxDistHEU = getMaxDist(P,centers_heu)
pointAPPHEU = maxDistHEU[0] # le point le plus loin dasn APPHEU
dicoAPPHEU = maxDistHEU[2] # pour faire le lien entre les points et leur centre
print(f"APPHEU : {maxDistHEU[1]}")


print("--------------------")
print("APPROX")
result = kCentresApprox(P,k)
centres = result[0] 
deux_di = result[1]
maxDist = getMaxDist(P,centres)
pointAPP = maxDist[0] # le point le plus loin dans APP
dicoAPP = maxDist[2] # pour faire le lien entre les points et leur centre
print(f"APP : {maxDist[1]}")

print("--------------------")
print("OPT")
resultBrut = kCentresBrutForce(P,k)
centres_brut = resultBrut[0] 
distance_brut = resultBrut[1]
point_far_brut = resultBrut[2] # le point le plus loin dans OPT
dict_brut = resultBrut[3] # pour faire le lien entre les points et leur centre
print(f"OPT : {distance_brut}")
print("CENTRES BRUT")
print(centres_brut)


fig, ax = plt.subplots()

for centre_aff in range(k):
    plt.scatter(np.array(P)[y == centre_aff, 0], np.array(P)[y == centre_aff, 1], color=colors[centre_aff], s=10 )

#Centres APP
for centre in range(len(centres)): # on affiche les centres trouvés par APP et les cercles couvrant associés
    circle = plt.Circle(( centres[centre][0] , centres[centre][1] ), deux_di, color="r", fill=False ) 
    ax.add_artist(circle)


#APPHEU
XsAPPHEU=[pointAPPHEU[0],(dicoAPPHEU[pointAPPHEU])[0][0]]
YsAPPHEU=[pointAPPHEU[1],(dicoAPPHEU[pointAPPHEU])[0][1]]
plt.plot(XsAPPHEU,YsAPPHEU,"g--") # La distance max trouvée par APPHEU

#APP
XsAPP=[pointAPP[0],(dicoAPP[pointAPP])[0][0]]
YsAPP=[pointAPP[1],(dicoAPP[pointAPP])[0][1]]
plt.plot(XsAPP,YsAPP,"r:") # La distance max trouvée par APP

#OPT
Xs=[point_far_brut[0],(dict_brut[point_far_brut])[0][0]]
Ys=[point_far_brut[1],(dict_brut[point_far_brut])[0][1]]
plt.plot(Xs,Ys) # la distance max optimale




ax.axis("equal")
plt.xlim(0,100)
plt.ylim(0,100)

plt.title('K-centres implementation', fontsize=8)


plt.show()


