"""Regenerate the retained 31 feature formulas; never execute or alter notebooks."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
class Computation(ast.NodeTransformer):
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id in ('print','display'):
            return ast.Pass()
        return self.generic_visit(node)
    def visit_Assert(self,node):
        return None if 'df.shape' in ast.unparse(node) else node

def extract():
    path=ROOT/'data/data_processing/data_engineering.ipynb'
    cells=json.loads(path.read_text(encoding='utf-8'))['cells']
    body=[]
    for index in [2,3,4,5,6]:
        body+=Computation().visit(ast.parse(''.join(cells[index]['source']))).body
    body+=ast.parse(''.join(cells[9]['source'])).body[:3]
    # Cell 9 starts with the market-cap conversion, its positivity assertion,
    # and LOG_MCAP. The later 40-feature summaries reference excluded groups.
    body+=ast.parse('return df').body
    function=ast.FunctionDef(name='technical_features',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='df')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=body,decorator_list=[])
    code='"""Retained formulas from data_engineering.ipynb Cells 2–6 and 9.\nOnly display statements and fixed-size assertions removed. Caller supplies a copy.\n"""\nimport numpy as np\nimport pandas as pd\n\n'+ast.unparse(ast.fix_missing_locations(function))+'\n'
    (ROOT/'dashboard/features.py').write_text(code,encoding='utf-8')
    files=[path,ROOT/'data/data_processing/data_processing.ipynb']
    provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (ROOT/'dashboard/processing_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')

if __name__=='__main__': extract()
