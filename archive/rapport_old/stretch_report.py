import re

with open('/home/anwar/Documents/burntrack-simulation/rapport/rapport_plbd_groupe7.tex', 'r') as f:
    text = f.read()

# Increase spacing to reach the page limit
text = text.replace(r"\begin{document}", r"\linespread{1.2}\selectfont" + "\n" + r"\begin{document}")

with open('/home/anwar/Documents/burntrack-simulation/rapport/rapport_plbd_groupe7.tex', 'w') as f:
    f.write(text)

