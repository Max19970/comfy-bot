import os
import re

ru_pattern = re.compile(r'[А-Яа-яЁё]')

for root, _, files in os.walk('.'):
    if '.venv' in root or '.git' in root or '__pycache__' in root or '.mypy_cache' in root:
        continue
    for file in files:
        if not file.endswith('.py'):
            continue
        path = os.path.join(root, file)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if ru_pattern.search(line):
                        print(f"{path}:{i+1}: {line.strip()}")
        except Exception as e:
            pass
