import re

with open('/home/anwar/Documents/burntrack-simulation/rapport/rapport_plbd_groupe7.tex', 'r') as f:
    text = f.read()

# Let's add massive over-explanations to the Introduction.
over_explanation_1 = r"""
Pour vraiment bien comprendre pourquoi c'est important, imaginez une forêt. Une forêt, ce n'est pas juste des arbres posés là par hasard. C'est tout un écosystème avec des herbes, des arbustes, des feuilles mortes par terre, des petits insectes, des animaux, et des champignons. Quand on parle de "combustible" dans le jargon technique, on parle en fait de tout ça. Tout ce qui peut brûler. Et le problème avec le feu, c'est qu'il ne brûle pas tout à la même vitesse. Par exemple, si vous prenez une toute petite brindille sèche, elle va prendre feu instantanément si vous approchez une allumette. C'est ce qu'on appelle les "combustibles d'une heure" parce qu'ils sèchent très vite. Mais si vous prenez une grosse bûche, vous aurez beau la laisser au soleil, elle mettra beaucoup plus de temps à sécher, et elle mettra beaucoup plus de temps à prendre feu. C'est pour ça qu'on doit calculer tout ça avec des mathématiques.

Ensuite, il faut parler du vent. Le vent, c'est l'ennemi numéro un quand il y a un incendie. Imaginez que vous soufflez sur des braises de barbecue. Qu'est-ce qui se passe ? Les braises deviennent toutes rouges et font de grandes flammes. Dans la forêt, c'est pareil ! Le vent amène de l'oxygène, qui est le gaz qui permet au feu de respirer. En plus de ça, le vent pousse les flammes vers l'avant. Du coup, les flammes se penchent et viennent chauffer les herbes et les arbres qui ne brûlent pas encore, juste devant le feu. C'est ce qu'on appelle le transfert de chaleur par radiation et par convection. C'est exactement comme quand vous mettez vos mains au-dessus d'un feu de camp pour vous réchauffer, sauf que là, ça chauffe tellement que les plantes s'enflamment toutes seules avant même que le feu ne les touche directement !

Et puis il y a la pente. La pente de la montagne, c'est un autre gros facteur. Si vous allumez un feu en bas d'une colline, le feu va monter la colline à une vitesse folle. Pourquoi ? Parce que la chaleur monte ! Les flammes se penchent vers la colline, un peu comme avec le vent, et chauffent tout ce qui est au-dessus. Donc, si vous avez une pente très raide, le feu peut avancer deux ou trois fois plus vite que sur un terrain plat. C'est pour toutes ces raisons que prévoir un feu, c'est super compliqué. Il faut mélanger la météo, la forme de la montagne, et le type de plantes. Et c'est là que notre super robot et notre Intelligence Artificielle entrent en jeu, pour tout calculer à notre place sans qu'on ait besoin de se casser la tête avec une calculatrice en plein milieu de la forêt.
"""

text = text.replace(r"surtout dans les régions du Rif, du Moyen Atlas et de la Maâmora.", 
                    r"surtout dans les régions du Rif, du Moyen Atlas et de la Maâmora. " + over_explanation_1)

# Now let's add another one to the Contexte section
over_explanation_2 = r"""
Pour vous donner un peu plus de contexte sur pourquoi c'est un tel casse-tête, parlons un peu du changement climatique. On entend souvent ce mot à la télé, mais concrètement, qu'est-ce que ça veut dire pour nos forêts ? Eh bien, ça veut dire que les hivers sont plus doux, et les étés sont beaucoup, beaucoup plus longs et plus chauds. Avant, la saison des pluies permettait à la forêt de faire le plein d'eau, et les plantes restaient bien vertes jusqu'en juillet. Mais maintenant, avec le manque de pluie, dès le mois de mai, tout commence à jaunir et à sécher. L'humidité dans l'air, qu'on appelle l'hygrométrie, chute drastiquement. 

Quand l'air est très sec, il se comporte comme une éponge géante qui vient aspirer toute la petite eau qui reste à l'intérieur des feuilles et des branches mortes. Du coup, la forêt devient ce qu'on appelle une véritable poudrière. Il suffit d'un petit rien pour que tout s'embrase. Ça peut être un bout de verre laissé par des promeneurs qui fait l'effet d'une loupe avec le soleil, ça peut être un barbecue mal éteint, ou parfois même un éclair d'orage, bien que ce soit plus rare chez nous. 

Une fois que le feu part, c'est la panique. Les pompiers (la protection civile) doivent intervenir hyper vite. Mais s'ils n'ont pas de programme pour leur dire "attention, dans 2 heures, le feu sera passé par-dessus cette colline", ils risquent de se placer au mauvais endroit, ou pire, de se retrouver encerclés par les flammes. C'est une vraie question de vie ou de mort, pour la nature, pour les animaux (les petits lapins, les renards, les sangliers), mais aussi pour les humains qui habitent dans les villages autour de la forêt. C'est pour ça qu'on a pris ce projet très à cœur. On ne voulait pas juste faire un exercice d'école pour avoir une bonne note, on voulait vraiment comprendre comment ça marche et essayer d'apporter notre petite pierre à l'édifice pour protéger notre pays.
"""

text = text.replace(r"La forêt de Bouskoura, c'est vraiment un cas d'école.", 
                    over_explanation_2 + r"\n\nLa forêt de Bouskoura, c'est vraiment un cas d'école.")

with open('/home/anwar/Documents/burntrack-simulation/rapport/rapport_plbd_groupe7.tex', 'w') as f:
    f.write(text)

print("Inflated rapport_plbd_groupe7.tex with over-explanations.")

