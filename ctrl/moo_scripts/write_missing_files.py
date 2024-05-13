from pathlib import Path

suite = Path('suite.sh').read_text().split('=')[1].strip()
print(suite)

local_pp_names = set([p.name for p in Path.cwd().glob('*.pp')])
moose_urls = Path(f'{suite}_ls_files.txt').read_text().split()
moose_paths = [Path(l.split(':')[-1]) for l in moose_urls]
moose_pp_names = [p.name for p in moose_paths]

moose_name_to_url = {
    name: url
    for name, url in zip(moose_pp_names, moose_urls)
}

missing_names = set(moose_pp_names) - local_pp_names
missing_files_txt = Path(f'{suite}_missing_files.txt')

missing_urls = [moose_name_to_url[n] for n in sorted(missing_names)]
for u in missing_urls:
    print(f'  {u}')
missing_files_txt.write_text('\n'.join(missing_urls))

