import ast
import os
import re

ru_pattern = re.compile(r'[А-Яа-яЁё]')

class StringVisitor(ast.NodeVisitor):
    def __init__(self, filename):
        self.filename = filename
        self.strings = []

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            if ru_pattern.search(node.value):
                # Check if it's a docstring
                if not (isinstance(self.current_parent(), (ast.FunctionDef, ast.ClassDef, ast.Module)) and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)):
                    self.strings.append((node.lineno, node.value))
        self.generic_visit(node)
        
    def current_parent(self):
        # We'd need an ast wrapper to track parents, simpler approach below:
        pass

def extract_strings(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        tree = ast.parse(content)
    except SyntaxError:
        return []
    
    strings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if ru_pattern.search(node.value):
                strings.append((node.lineno, node.value))
    return strings

total_strings = 0
for root, _, files in os.walk('.'):
    if '.venv' in root or '.git' in root or '__pycache__' in root or '.mypy_cache' in root:
        continue
    for file in files:
        if not file.endswith('.py'):
            continue
        path = os.path.join(root, file)
        extracted = extract_strings(path)
        for lineno, val in extracted:
            print(f"{path}:{lineno}: {repr(val)}")
            total_strings += 1

print(f"Total: {total_strings}")
