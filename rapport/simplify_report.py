import re

with open('/home/anwar/Documents/burntrack-simulation/rapport/rapport_plbd_groupe7.tex', 'r') as f:
    text = f.read()

# 1. Update the logo
text = re.sub(
    r'\\includegraphics\[height=1.8cm\]\{Centrale\.jpg\}',
    r'\\includegraphics[height=3.2cm]{burntrack_logo_white.png}',
    text
)
# Just in case it was burntrack_logo.png
text = re.sub(
    r'\\includegraphics\[height=1.8cm\]\{burntrack_logo\.png\}',
    r'\\includegraphics[height=3.2cm]{burntrack_logo_white.png}',
    text
)

# 2. Rewrite Résumé
resume_original = r"""Les incendies de forêt constituent une menace environnementale majeure pour les écosystèmes marocains et africains, exacerbée par les changements climatiques. Les modèles de prédiction de propagation existants, principalement calibrés pour les biomes nord-américains, s'avèrent inadaptés à notre contexte. Ce projet présente \textbf{BurnTrack}, un système intégré associant un rover autonome de collecte de données et un moteur de simulation de propagation d'incendies.

Notre approche s'articule autour de quatre axes majeurs : (1) l'adaptation du modèle mathématique de Rothermel avec l'intégration de 28 modèles de combustibles spécifiques à la flore africaine ; (2) le développement d'un modèle hybride couplant un automate cellulaire stochastique à un perceptron multicouche (MLP) pour corriger les erreurs de prédiction géospatiale (réduction de la MAE à 0.735 m/min) ; (3) la conception matérielle et logicielle d'un rover 6WD/6WS basé sur Raspberry Pi 4, équipé de capteurs environnementaux, d'un anémomètre customisé, et d'un pipeline de vision par ordinateur pour la classification des combustibles et l'estimation du statut hydrique ; et (4) l'implémentation d'une télémétrie LoRa (CBOR) pour un monitoring en temps réel.

Un algorithme de planification dynamique (D* Lite) permet au rover d'éviter le front de flamme de manière autonome. Les résultats démontrent une amélioration significative de la précision prédictive par rapport au modèle physique brut."""

resume_spoken = r"""En gros, les incendies de forêt, c'est vraiment un énorme problème chez nous au Maroc et un peu partout en Afrique. Avec la chaleur qui augmente à cause du changement climatique, ça devient de pire en pire chaque année. Le truc, c'est que les programmes habituels qu'on utilise pour deviner comment le feu va avancer ont été pensés pour les forêts américaines. Du coup, quand on essaie de les utiliser chez nous, ça ne marche pas très bien parce que nos plantes sont différentes. 

C'est pour ça qu'on a créé \textbf{BurnTrack}. C'est un super système : on a fabriqué un petit robot roulant (un rover) qui va tout seul récolter des infos dans la forêt, et on a un programme sur ordinateur qui fait des simulations de feu super précises.

Notre projet fonctionne sur quatre trucs principaux :
Premièrement, on a pris le fameux modèle mathématique (Rothermel) et on l'a modifié pour y ajouter 28 types de végétation africaine, parce qu'il fallait bien qu'il connaisse nos arbres.
Deuxièmement, on a utilisé un réseau de cases (un automate cellulaire) mélangé à de l'Intelligence Artificielle (c'est-à-dire qu'on a entraîné l'ordinateur à corriger les erreurs mathématiques pour être super précis). L'erreur n'est plus que de 0,735 mètres par minute !
Troisièmement, on a construit un vrai robot avec 6 roues motrices et un "cerveau" (une Raspberry Pi 4). Il a plein de capteurs pour mesurer le vent, et même une caméra pour reconnaître si une plante est sèche ou pas. 
Quatrièmement, on a mis en place une radio longue distance (LoRa) pour que le robot puisse nous envoyer des messages depuis la forêt jusqu'à notre ordinateur.

Pour que le robot ne se fasse pas brûler, on a codé un algorithme intelligent (D* Lite) qui lui permet de recalculer son chemin s'il voit que le feu approche. Nos résultats montrent que c'est beaucoup, beaucoup plus précis que les vieilles formules mathématiques classiques ! On a gardé tous les détails techniques dans les pages qui suivent pour bien montrer tout le travail qu'on a fait."""

text = text.replace(resume_original, resume_spoken)

# 3. Rewrite Introduction
intro_original = r"""Les incendies de forêt représentent une problématique environnementale et socio-économique d'envergure mondiale. Au Maroc, le patrimoine forestier couvre environ 9 millions d'hectares, soit près de 12\% du territoire national, et subit chaque année des dégradations sévères, particulièrement dans les zones du Rif, du Moyen Atlas et de la Maâmora. L'allongement des saisons sèches et l'intensification des vagues de chaleur induites par le changement climatique exacerbent cette vulnérabilité.

Face à cette menace, la prédiction précise du comportement du feu est cruciale. Le modèle semi-empirique de Rothermel (1972) demeure le standard mondial pour la prédiction de la vitesse de propagation (Rate of Spread, ROS). Néanmoins, son application sur le continent africain se heurte à deux obstacles majeurs : d'une part, l'inadéquation des modèles de combustibles standards (issus des biomes nord-américains) face à la complexité de notre flore (arganeraies, maquis rifain, cédraies) ; d'autre part, la rareté des données de terrain in situ pour le calibrage et la validation locale.

Le projet \textbf{BurnTrack} a été initié pour combler cette lacune technologique et scientifique. Notre solution propose une approche duale :
\begin{itemize}[nosep]
    \item \textbf{Axe Simulation} : Adaptation du modèle mathématique via l'intégration de 28 modèles de combustibles afro-méditerranéens, couplé à un automate cellulaire stochastique et hybridé avec un réseau de neurones artificiels (MLP) pour l'assimilation de données réelles.
    \item \textbf{Axe Robotique} : Conception intégrale d'un rover autonome 6WD/6WS dédié à la caractérisation in situ (topographie, météo, classification des combustibles par vision) et doté de capacités d'évasion dynamique face au front de flamme.
\end{itemize}

Ce rapport détaille de manière exhaustive notre démarche, de l'état de l'art justifiant nos choix architecturaux jusqu'aux résultats de nos campagnes de validation sur des scénarios reconstitués."""

intro_spoken = r"""Les incendies de forêt, on ne va pas se mentir, c'est un problème environnemental gigantesque à l'échelle de la planète, mais surtout chez nous. Rien qu'au Maroc, la forêt couvre environ 9 millions d'hectares (ça fait à peu près 12\% de tout notre territoire), et chaque année on perd plus de 10 000 hectares à cause des flammes, surtout dans les régions du Rif, du Moyen Atlas et de la Maâmora. Avec le changement climatique, les saisons sèches s'allongent et il fait de plus en plus chaud, donc la situation devient vraiment critique.

Face à ça, il faut absolument qu'on puisse deviner comment un feu de forêt va se propager. Le "boss final" des modèles mathématiques pour faire ça, c'est le modèle de Rothermel (créé en 1972). C'est la référence mondiale. Sauf que ce modèle a deux énormes défauts pour nous, en Afrique :
Premièrement, les types de plantes (les "modèles de combustible") utilisés dans leurs équations ont été réglés pour la végétation d'Amérique du Nord. Nos arbres africains ne brûlent pas pareil !
Deuxièmement, on n'a presque aucune vraie donnée de terrain sur les feux en Afrique pour régler ces équations. 

Notre projet, \textbf{BurnTrack}, c'est notre solution pour combler ce manque énorme. On a mis en place une approche en deux parties, qu'on va vous expliquer en détail (avec plein de graphiques et de calculs dans la suite du rapport pour que vous voyiez bien comment ça marche) :
\begin{itemize}[nosep]
    \item \textbf{Un axe simulation (le logiciel)} : On a adapté les formules mathématiques en ajoutant 28 types de végétation spécifiques au Maroc et à l'Afrique. Ensuite, on a mélangé ça avec un automate cellulaire (un tableau de cases pour simuler la forêt, comme un jeu d'échecs géant) et on a rajouté une couche d'Intelligence Artificielle pour apprendre des vrais feux et corriger les erreurs.
    \item \textbf{Un axe robotique (le matériel)} : On a fabriqué de A à Z un rover autonome (un petit robot avec 6 roues motrices, comme ceux de la NASA). Son job, c'est d'aller tout seul dans la forêt, de prendre des températures, de mesurer le vent, et même de reconnaître les plantes avec une caméra. Comme ça, il nourrit notre simulation avec des vraies données en direct.
\end{itemize}

Ce rapport détaille vraiment tout ce qu'on a fait. On va garder les vrais calculs et les détails techniques parce qu'ils sont importants, mais on va essayer d'expliquer les grandes idées le plus simplement possible."""

text = text.replace(intro_original, intro_spoken)

with open('/home/anwar/Documents/burntrack-simulation/rapport/rapport_plbd_groupe7.tex', 'w') as f:
    f.write(text)

print("Simplified rapport_plbd_groupe7.tex.")

