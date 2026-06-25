import re

with open('/home/anwar/Documents/burntrack-simulation/rapport/rapport_plbd_groupe7.tex', 'r') as f:
    text = f.read()

# Change linespread from 1.2 to 1.4 to easily add a few more pages
text = text.replace(r"\linespread{1.2}", r"\linespread{1.4}")

with open('/home/anwar/Documents/burntrack-simulation/rapport/rapport_plbd_groupe7.tex', 'w') as f:
    f.write(text)

