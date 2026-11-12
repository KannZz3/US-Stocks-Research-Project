"""Extract computational stages; notebooks are references, never runtime state."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'dashboard/ml/stages'
MODELS = {'elastic':'Elastic Net', 'xgb':'Xgb', 'cat':'Cat', 'light':'Light'}

class Computation(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        return None if node.module == 'IPython.display' else node
    def visit_Import(self, node):
        return None if any(a.name.startswith('matplotlib') for a in node.names) else node
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id in ('print','display'):
            return ast.Pass()
        return self.generic_visit(node)

def clean(source):
    tree = Computation().visit(ast.parse(source))
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'

def extract():
    manifest = {}
    sources = {}
    for key, name in MODELS.items():
        path = ROOT / 'pipeline' / f'Data_{name}_Tech.ipynb'
        cells = [''.join(c['source']) for c in json.loads(path.read_text(encoding='utf-8'))['cells'] if c['cell_type']=='code']
        cells = [s for s in cells if s.strip() and not s.startswith('%')][:9]
        sources[key] = cells
        manifest[key] = {'source':str(path.relative_to(ROOT)), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    for index, name in [(1,'folds'),(2,'scaling'),(5,'tier3'),(6,'signals'),(7,'evaluation')]:
        versions = [clean(c[index]) for c in sources.values()]
        assert len(set(versions)) == 1, name
        (OUT / f'{name}.py').write_text(versions[0],encoding='utf-8')
    for key,cells in sources.items():
        (OUT / f'fit_{key}.py').write_text(clean(cells[3]),encoding='utf-8')
        tree = ast.parse(clean(cells[4]))
        fit = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name.startswith('c5_') and n.name.endswith('_fit'))
        old=fit.name;fit.name='c5_model_fit'
        (OUT / f'screen_fit_{key}.py').write_text(ast.unparse(fit)+'\n',encoding='utf-8')
        rest=ast.unparse(ast.Module(body=[n for n in tree.body if n is not fit],type_ignores=[])).replace(old,'c5_model_fit')
        # Model imports are provided by the fit stage; common screening logic only.
        rest_tree=ast.parse(rest)
        rest_tree.body=[n for n in rest_tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
        rest=ast.unparse(rest_tree)
        if key=='elastic': common=rest
        else: assert rest==common, key
    (OUT/'screening.py').write_text(common+'\n',encoding='utf-8')
    defaults={}
    for key,cells in sources.items():
        source=cells[8].split('# 7. Two CSV files')[0]
        tree=ast.parse(source)
        parameter_end_line=source[:source.index('# Frozen')].count('\n')+1
        defaults[key]={}
        body=[]
        for n in tree.body:
            if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name):
                name=n.targets[0].id
                if name.startswith('C8_') and n.lineno < parameter_end_line:
                    # Only literal user parameters; derived/frozen settings stay in source.
                    if name in ('C8_OUTPUT_DIR',): continue
                    try: value=ast.literal_eval(n.value)
                    except (ValueError,TypeError): value=None
                    if value is not None:
                        defaults[key][name]=value; continue
            if isinstance(n,ast.Expr) and 'C8_OUTPUT_DIR.mkdir' in ast.unparse(n): continue
            if isinstance(n,ast.Assign) and ast.unparse(n).startswith('C8_OUTPUT_DIR ='): continue
            body.append(n)
        code=clean(ast.unparse(ast.Module(body=body,type_ignores=[])))
        if key=='elastic': c8=code
        else: assert c8==code
    # Disabled directions have zero score columns; avoid multiplying by active weights.
    c8=c8.replace('priority = scores * hits @ C8_FROZEN_WEIGHTS[side]', 'priority = scores * hits @ C8_FROZEN_WEIGHTS[side] if fs else np.zeros(len(oo))')
    (OUT/'execution.py').write_text(c8,encoding='utf-8')
    (OUT.parent/'defaults.json').write_text(json.dumps(defaults,indent=2),encoding='utf-8')
    (OUT.parent/'provenance.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

if __name__=='__main__': extract()
