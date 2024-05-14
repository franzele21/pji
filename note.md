# Note PJI
## Idées
- prise en compte de tous les formats video
- faire des exceptions
- ligne de commande
- prise en comtpe de la taille de la vidéo pour pas de crash (avec ou sans `stream` en argument)
- vérifier seulement chaque x frames, et si la frame y sélectionnée est interressante, alors on vérfie aussi les voisins (y-x+1, y+x-1). Si x=10, on peut diviser par presque 10 le temps passé.
## 13/05
Demander si possibilité d'utiliser YOLOv8 (car déjà outil dévellopé dessus) + demander si AGPL-3.0 Licence pose problème
Instalation de https://github.com/derronqi/yolov8-face -> pas fonctionné
Réalisation du programme pouvant flouter une vidéo

## 14/05
Nouvelle façon : d'abord on passe la vidéo dans le modèle, et on ne passe que dans les frames les plus intérressantes. Avantage : + rapide pour des vidéos longues avec peu de têtes. Inconvéninant : dépendance + grande de mémoire
La nouvelle façon s'est révéllée être beaucoup trop couteuse
Prise en compte de plusieurs modèles


