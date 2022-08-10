# coding: utf-8
import sys

from remake.util import sysrun

mass_url = sys.argv[1]

output = sysrun(f'moo ls -l {mass_url}').stdout
lines = [l for l in output.split('\n') if l]
sizes = [int(l.split()[4]) for l in lines]

print(f'Total size under {mass_url}: {sum(sizes) / 1e9:.1f}G')
