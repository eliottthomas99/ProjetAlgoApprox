# ProjetAlgoApprox

* 1. Implementer L'algo K-centre vu en cours. 
* 2. Comparer l'efficacité de cet algo en pratique et en théorie. -> tester l'optimal avec un  brutforce, ok si pas trop de nodes ? 
-> avantage de l'algo d'approx par rapport au brutForce -> prend moins de centres si necessaire


* 3. Comprendre l'algo alternatif et en faire une explication breve mais précise. 
* 4. Mettre en place l'algo alternatif. -> done
* 5. Comparer le nouvel algo à l'autre en pratique et à la théorie.  

### Principe de Best heuristic : 

Presentation des notions essentielles 
- independant set : no two verticises adjacents
- strong stable set :  independant + pour chaque x pas dans S, au maximum un voisin dans S
- dominating set : chaque x pas dans D adjacent à au moins un élément de D. 
- G(W) = (V,Ew) avec Ew, les arêtes tq we <= W.
- It can be easily verified that finding the solution to the k-center problem is equivalent to finding a minimum value of W such that the graph G( W) has a dominating set of size not exceeding k.


### reste à faire ? 

- faire un vrai README pour le prof
- remettre le code avec les tests en masse
- relire le rapport une dernière fois
- faire un zip avec le code, le readme et le rapport
- envoyer le mail au prof. 
