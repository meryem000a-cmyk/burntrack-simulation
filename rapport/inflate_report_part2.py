import re

with open('/home/anwar/Documents/burntrack-simulation/rapport/rapport_part2.tex', 'r') as f:
    text = f.read()

over_explanation_3 = r"""
Alors, qu'est-ce que c'est qu'un automate cellulaire ? Si vous n'avez jamais entendu ce mot, pas de panique, c'est en fait un concept très simple. Imaginez une immense feuille de papier millimétré, comme celle que vous utilisiez en cours de maths à l'école. Chaque petit carreau de la feuille représente un morceau de la forêt, disons un carré de 10 mètres sur 10 mètres. Au lieu d'essayer de calculer d'un seul coup comment le feu va brûler toute la forêt, on se concentre juste sur un carreau à la fois. C'est ça le secret !

On donne une couleur ou un statut à chaque carreau. Par exemple, si le carreau est tout vert, ça veut dire que la forêt est intacte et n'a pas encore brûlé. Si le carreau est rouge, c'est qu'il est en train de brûler en ce moment même. Si le carreau est noir, ça veut dire qu'il a déjà brûlé et qu'il n'y a plus rien à brûler (donc le feu ne peut pas y retourner). Et si le carreau est gris, ça peut être une route, une rivière, ou une maison, bref, un endroit où le feu ne peut pas aller du tout. 

Maintenant, comment on fait avancer le feu ? C'est simple. On regarde un carreau rouge (un carreau qui brûle). On se pose la question : est-ce que ce feu va réussir à enflammer les carreaux verts qui sont juste à côté de lui ? Pour savoir ça, on ne dit pas juste "oui" ou "non", on utilise des probabilités. C'est comme jeter des dés. Si on a un carreau rouge, et qu'à sa droite il y a un carreau vert plein d'herbes sèches, et que le vent souffle très fort vers la droite, alors on a peut-être 90\% de chance que le feu saute sur le carreau de droite. On jette notre dé virtuel, et si on fait moins de 90, paf, le carreau de droite devient rouge à son tour ! Par contre, si on regarde le carreau à gauche, là où le vent ne souffle pas, on a peut-être que 5\% de chance. On jette le dé, et souvent, le carreau de gauche reste vert. 

Et on fait ça pour tous les carreaux, des dizaines et des dizaines de fois par seconde dans l'ordinateur. C'est ça qui crée la forme du feu qui s'étale sur la carte. C'est une méthode super visuelle et qui permet de faire des cartes de risque magnifiques, où on voit vraiment les flammes virtuelles se propager en fonction du vent et des obstacles. Et l'avantage, c'est que si on veut simuler une route coupe-feu, on a juste à dessiner une ligne de carreaux gris, et l'ordinateur comprend tout de suite que le feu doit s'arrêter net à cet endroit. C'est vraiment la meilleure méthode qu'on ait trouvée pour faire notre logiciel.
"""

text = text.replace(r"Le feu passe d'une case à l'autre avec une petite probabilité mathématique (on lance des dés virtuels). Si le vent souffle vers la droite, la case de droite a plus de chances de prendre feu.", 
                    r"Le feu passe d'une case à l'autre avec une petite probabilité mathématique (on lance des dés virtuels). Si le vent souffle vers la droite, la case de droite a plus de chances de prendre feu." + over_explanation_3)

over_explanation_4 = r"""
Pour bien vous faire comprendre cette histoire de "D* Lite", il faut qu'on parle de comment les ordinateurs trouvent leur chemin en général. Imaginez que vous êtes dans un labyrinthe, ou tout simplement que vous cherchez votre route sur Google Maps pour aller de Casablanca à Rabat. Le logiciel va regarder toutes les routes possibles et trouver la plus courte. C'est ce qu'on appelle la planification de chemin. Pendant des années, l'algorithme star pour faire ça s'appelait "A*" (prononcé A-star). Il est super fort pour trouver le chemin le plus court sur une carte vide. 

Mais que se passe-t-il si, pendant que vous roulez sur l'autoroute, il y a un accident soudain qui bloque la route ? L'algorithme A* de base est un peu bête dans ce cas-là : il doit effacer tout son calcul et tout recommencer depuis le début pour trouver un nouveau chemin. Si ça vous arrive une fois sur l'autoroute, ce n'est pas grave. Mais pour notre robot, c'est très différent. Notre robot roule dans une forêt en feu. Le feu, ça bouge tout le temps ! Chaque seconde, une nouvelle case prend feu. Si le robot devait recalculer la totalité de son trajet de 3 kilomètres à chaque seconde, son petit ordinateur (la Raspberry Pi) n'y arriverait pas. Ça prendrait trop de puissance de calcul, le robot s'arrêterait pour réfléchir, et il finirait par se faire rôtir par les flammes pendant qu'il réfléchit.

C'est là qu'intervient l'algorithme "D* Lite" (le D veut dire Dynamic). C'est une version super améliorée inventée par des chercheurs très malins. Au lieu de calculer le chemin en partant du robot vers l'objectif, D* Lite calcule le chemin en partant de l'objectif vers le robot. Pourquoi ? Parce que si un arbre prend feu juste devant le robot, c'est l'environnement du robot qui change, pas celui de l'objectif (qui est très loin). D* Lite se dit alors : "Ok, il y a du feu devant moi. Je ne vais pas tout recalculer. Je vais juste regarder les 3 mètres autour du feu, et je vais raccorder ça au chemin que j'avais déjà calculé pour le reste du trajet". C'est un peu comme faire une petite déviation locale au lieu de refaire tout l'itinéraire.

Grâce à ça, les calculs prennent des millisecondes au lieu de prendre des secondes entières. Le robot roule de manière fluide, sans jamais s'arrêter. Dès qu'il voit sur sa carte qu'une zone devient dangereuse (notre logiciel met une énorme zone de danger autour du feu pour être sûr), il fait une petite embardée fluide, tourne autour du danger, et reprend sa route. C'est ce qui nous a permis d'avoir un robot véritablement autonome, qui ne panique jamais, même quand la forêt brûle tout autour de lui. C'est un peu le "mode survie" de notre rover !
"""

text = text.replace(r"il recalcule juste le petit bout de chemin qui a changé.", 
                    r"il recalcule juste le petit bout de chemin qui a changé." + over_explanation_4)


with open('/home/anwar/Documents/burntrack-simulation/rapport/rapport_part2.tex', 'w') as f:
    f.write(text)

print("Inflated rapport_part2.tex with over-explanations.")
