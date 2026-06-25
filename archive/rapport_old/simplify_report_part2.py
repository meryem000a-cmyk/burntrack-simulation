import re

with open('/home/anwar/Documents/burntrack-simulation/rapport/rapport_part2.tex', 'r') as f:
    text = f.read()

conc_original = r"""\section{Conclusion générale et perspectives}

Le projet BurnTrack s'est achevé sur la conception fonctionnelle d'un écosystème intégré pour l'analyse prédictive et la surveillance des incendies de forêt, comblant ainsi le vide laissé par l'inadéquation des modèles standards sur le continent africain. 

\subsection*{Synthèse des contributions}
Les apports majeurs de ce travail peuvent être résumés en cinq points cardinaux :
\begin{enumerate}
    \item \textbf{Modélisation afro-méditerranéenne} : Élaboration de la première base de données consolidée de 28 modèles de combustibles couvrant les biomes d'Afrique du Nord, subsahariens et austraux.
    \item \textbf{Correction neuronale des équations physiques} : L'hybridation du moteur de Rothermel par un MLP a permis d'assimiler les variables latentes (VPD, dynamiques convectives locales) avec une réduction de la MAE prédictive de 81\%.
    \item \textbf{Moteur de propagation discret} : Implémentation d'un automate cellulaire stochastique garantissant la conservation de la masse et obéissant aux conditions CFL pour une résolution spatio-temporelle fluide.
    \item \textbf{Robotique autonome de terrain} : Conception intégrale d'un rover 6WD à suspension rocker-bogie, instrumenté pour la classification botanique locale via vision artificielle (CNN/ONNX).
    \item \textbf{Évasion et navigation temps réel} : Validation de l'algorithme D* Lite couplé à des zones d'exclusion dynamiques, prouvant la résilience du rover face à un front de flamme stochastique.
\end{enumerate}

\subsection*{Difficultés rencontrées}
L'intégration d'un tel système ne s'est pas faite sans obstacles. La principale contrainte logicielle fut la distorsion spatiale inhérente à la topologie de Moore dans l'automate cellulaire, résolue ultérieurement par une pondération euclidienne. Sur le plan matériel, la puissance des servomoteurs de direction (MG90S) s'est avérée limitante pour le braquage à l'arrêt du châssis lourd, imposant une modification logicielle pour un braquage uniquement en mouvement. Enfin, l'absence de bases de données empiriques exhaustives sur les feux de brousse a nécessité une recherche bibliographique minutieuse pour compiler les 1840 points d'entraînement de l'IA.

\subsection*{Perspectives futures}
Dans une optique de poursuite du projet, plusieurs axes de développement sont envisagés :
\begin{itemize}
    \item \textbf{Déploiement grandeur nature} : Réalisation de campagnes d'essais in situ dans la forêt de Bouskoura, avec validation croisée par des drones de reconnaissance (UAV).
    \item \textbf{Couplage satellite} : Intégration d'API satellitaires (Sentinel-2) pour la mise à jour quotidienne du NDVI et du stress hydrique de la flore avant déploiement du rover.
    \item \textbf{Essaim de rovers} : Évolution vers une flotte de micro-rovers communiquant en Mesh LoRa pour encercler la zone d'étude et fournir un maillage météorologique haute résolution.
\end{itemize}"""

conc_spoken = r"""\section{Conclusion générale et perspectives}

Le projet BurnTrack s'est achevé sur la conception fonctionnelle d'un système génial pour prévoir et surveiller les incendies de forêt, ce qui comble vraiment un manque énorme en Afrique où les anciens modèles ne marchaient pas du tout.

\subsection*{Synthèse des contributions (Ce qu'on a réussi à faire)}
Pour résumer, ce projet a vraiment cartonné sur plusieurs points cruciaux :
\begin{enumerate}
    \item \textbf{Des modèles pour NOS plantes} : On a créé la toute première base de données consolidée avec 28 modèles de plantes d'Afrique du Nord et du Sud.
    \item \textbf{Une Intelligence Artificielle magique} : En branchant notre petit réseau de neurones sur les formules classiques, on a corrigé les grosses erreurs de calcul. L'erreur a chuté de 81\%, c'est énorme !
    \item \textbf{Une simulation sous forme de grille} : On a codé un automate cellulaire (comme une grande feuille à carreaux où le feu saute de case en case) qui marche super bien.
    \item \textbf{Un robot fabriqué maison} : On a monté de toutes pièces un rover à 6 roues qui peut reconnaître les plantes tout seul grâce à sa caméra.
    \item \textbf{Un GPS de survie} : On a réussi à programmer le robot pour qu'il calcule son chemin en évitant le feu, et il s'en sort super bien même quand le feu va vite.
\end{enumerate}

\subsection*{Difficultés rencontrées (Nos grosses galères)}
Bien sûr, tout n'était pas rose. 
- Au début, les calculs sur les cases diagonales faisaient avancer le feu deux fois trop vite (un bug classique). On a dû bidouiller plein de formules pour régler ça.
- Côté mécanique, nos petits moteurs de direction (les MG90S) galéraient à tourner les grosses roues quand le robot était à l'arrêt, donc on a modifié le code pour qu'il ne tourne qu'en roulant.
- Et franchement, trouver des vraies données sur les feux en Afrique pour entraîner l'IA, c'était l'enfer ! On a dû fouiller partout dans les archives pour trouver nos 1840 points de données.

\subsection*{Perspectives futures (La suite !)}
Si on pouvait continuer le projet, on adorerait faire ces trucs :
\begin{itemize}
    \item \textbf{Tests réels en forêt} : Aller faire rouler le robot pour de vrai dans la forêt de Bouskoura, avec pourquoi pas un petit drone qui vole au-dessus pour filmer.
    \item \textbf{Brancher des satellites} : Connecter notre programme directement aux images satellites (Sentinel-2) pour savoir tous les jours si la forêt est sèche, avant même d'envoyer le robot.
    \item \textbf{Une armée de robots} : Envoyer plein de petits rovers qui se parlent entre eux avec la radio LoRa pour encercler la zone et avoir la météo exacte partout en même temps.
\end{itemize}"""

if conc_original in text:
    text = text.replace(conc_original, conc_spoken)
    with open('/home/anwar/Documents/burntrack-simulation/rapport/rapport_part2.tex', 'w') as f:
        f.write(text)
    print("Simplified rapport_part2.tex.")
else:
    print("Could not find the exact original conclusion block in rapport_part2.tex.")

