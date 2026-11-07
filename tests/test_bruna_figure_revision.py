import ast
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]

class FigureRevisionContractTests(unittest.TestCase):
    def test_layouts_do_not_use_absolute_figure_titles_or_arrows(self):
        tree=ast.parse((ROOT/'scripts/bruna_figure_layouts_v4.py').read_text(encoding='utf-8'))
        functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name.startswith('build_')]
        self.assertEqual(len(functions),2)
        for node in functions:
            code=ast.unparse(node)
            self.assertNotIn('fig.text(',code)
            self.assertNotIn('FancyArrowPatch',code)
            self.assertIn('subfigures',code)
            self.assertIn('fig.canvas.draw()',code)

    def test_notebook_is_unexecuted_and_preserves_reference_data(self):
        nb=json.loads((ROOT/'notebooks/10_cremad_figuras_corrigidas_v4_colab.ipynb').read_text(encoding='utf-8'))
        self.assertFalse(nb['metadata']['bruna_revision']['executed'])
        joined='\n'.join(''.join(c['source']) for c in nb['cells'])
        self.assertIn('20260809_223452_c0613d3c',joined)
        self.assertIn('cremad_landmarks_T100_semantic.npz',joined)
        self.assertIn('paper_figures_v4_revisao_20261107',joined)
        self.assertIn('validate_figure_layout(fig)',joined)
        self.assertNotIn('fig.text(0.285, 0.485',joined)
        for i,cell in enumerate(nb['cells']):
            if cell['cell_type']=='code':
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'],[])
                source='\n'.join('pass' if s.lstrip().startswith('%') else s for s in ''.join(cell['source']).splitlines())
                compile(source,f'cell-{i}','exec')

if __name__=='__main__':unittest.main()
