import json
import re

transcript_path = '/home/anwar/.gemini/antigravity-cli/brain/dcd010d6-96b5-4869-9400-c2a556dcac27/.system_generated/logs/transcript_full.jsonl'

lines_plbd = {}
lines_part2 = {}

with open(transcript_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        if data.get('type') == 'VIEW_FILE':
            content = data.get('content', '')
            if '1: \\documentclass' in content or '1: % rapport_part2.tex' in content:
                # determine which
                is_plbd = '1: \\documentclass' in content
                
                lines = content.split('\n')
                for l in lines:
                    m = re.match(r'^(\d+):\s(.*)$', l)
                    if m:
                        num = int(m.group(1))
                        if is_plbd:
                            lines_plbd[num] = m.group(2)
                        else:
                            lines_part2[num] = m.group(2)

print('Extracted plbd lines:', len(lines_plbd))
print('Extracted part2 lines:', len(lines_part2))

if len(lines_plbd) > 100:
    max_line = max(lines_plbd.keys())
    with open('/home/anwar/Documents/burntrack-simulation/rapport/rapport_plbd_groupe7.tex', 'w') as f:
        for i in range(1, max_line + 1):
            f.write(lines_plbd.get(i, '') + '\n')
            
if len(lines_part2) > 100:
    max_line = max(lines_part2.keys())
    with open('/home/anwar/Documents/burntrack-simulation/rapport/rapport_part2.tex', 'w') as f:
        for i in range(1, max_line + 1):
            f.write(lines_part2.get(i, '') + '\n')
