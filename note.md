# Note PJI
## Idées
- prise en compte de tous les formats video
- faire des exceptions
- faire documentation sur github
- faire comparaison entre les modèles
## 13/05
Demander si possibilité d'utiliser YOLOv8 (car déjà outil dévellopé dessus) + demander si AGPL-3.0 Licence pose problème
Instalation de https://github.com/derronqi/yolov8-face -> pas fonctionné
Réalisation du programme pouvant flouter une vidéo

## 14/05
Nouvelle façon : d'abord on passe la vidéo dans le modèle, et on ne passe que dans les frames les plus intérressantes. Avantage : + rapide pour des vidéos longues avec peu de têtes. Inconvéninant : dépendance + grande de mémoire
La nouvelle façon s'est révéllée être beaucoup trop couteuse
Prise en compte de plusieurs modèles

## 15/05
Nouvelle façon : on échantillonne sur une intervalle régulière des frames, on regarde si ces frames sont intérresasntes, et si on trouve une frame intérressante, alors on regarde ses voisins dans la vidéo originale. Avantage : +++ rapide pour des vidéos longues avec peu de têtes, pas de dépendance mémoire. Inconvénient : si un visage se trouve entre deux échantillonnage, alors il sera pas pris en compte. Bug : les dernières frames ne sont pas prises en compte

## 16/05
Création de fonction permettant de prendre un fichier .bag (que ROS1 pour l'instant) et extraire la vidéo. 
    - Vérifier/demander si `frequency` == fps (même longueur vidéo et différence de timestamp donc ça devrait le faire)

## 17/05
Correction de la méthode sur les dernières frames.
Création du programme utilisable comme ligne de commande (voir `blur_bag.py`)

