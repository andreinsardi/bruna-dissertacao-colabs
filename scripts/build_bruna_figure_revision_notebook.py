"""Gera uma revisão do notebook09, sem executar/alterar a execução histórica."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'notebooks/09_cremad_figuras_cientificas_v3_colab.ipynb'
LAYOUT=ROOT/'scripts/bruna_figure_layouts_v4.py'
DEST=ROOT/'notebooks/10_cremad_figuras_corrigidas_v4_colab.ipynb'
def replace_function(source,name,new):
    tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    lines=source.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+new+'\n\n'+''.join(lines[node.end_lineno:])
def build():
    nb=json.loads(SOURCE.read_text(encoding='utf-8'))
    layout=LAYOUT.read_text(encoding='utf-8');tree=ast.parse(layout)
    funcs={n.name:ast.get_source_segment(layout,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
    found=set()
    for i,cell in enumerate(nb['cells']):
        text=''.join(cell['source'])
        text=text.replace('paper_figures_v3','paper_figures_v4_revisao_20261107').replace('figures_v3_cache','figures_v4_cache')
        if cell['cell_type']=='markdown':
            text=text.replace('figuras científicas V3','figuras científicas V4 - revisão de layout')
            if i==0:text+='\n\nRevisão de 07/11/2026: títulos locais com SubFigure; colorbar própria da atenção; remoção de setas dependentes de posições anteriores ao layout. Os dados e resultados canônicos não são alterados. As figuras são salvas em nova pasta e não substituem automaticamente o manuscrito.\n'
        else:
            for name in ('build_experimental_figure','build_architecture_figure'):
                if f'def {name}(' in text:text=replace_function(text,name,funcs[name]);found.add(name)
            if 'def save_figure(' in text:
                text=funcs['validate_figure_layout']+'\n\n'+text
                text=text.replace('    target_dir = PT_DIR if language == "pt" else EN_DIR','    layout_audit = validate_figure_layout(fig) if basename in {"visao_experimental_comparativa_v2.png", "arquitetura_emot_facial_v2.png"} else {"manual_visual_review_required": True}\n    target_dir = PT_DIR if language == "pt" else EN_DIR')
                text=text.replace('"source_note": source_note,','"source_note": source_note,\n            "layout_version": "v4-20261107",\n            "layout_audit": json.dumps(layout_audit),')
            if 'FORCE_FIGURES = set()' in text:text=text.replace('FORCE_FIGURES = set()','FORCE_FIGURES = {"experimental", "architecture"}')
            # Instalação sem upgrade global que poderia mudar NumPy do Colab.
            text=text.replace('%pip install -q --upgrade opencv-python-headless seaborn','%pip install -q opencv-python-headless seaborn')
            cell['execution_count']=None;cell['outputs']=[]
            python_text='\n'.join('pass # Colab magic' if line.lstrip().startswith('%') else line for line in text.splitlines())
            compile(python_text,f'figura-v4-cell-{i}','exec')
        cell['source']=text.splitlines(keepends=True)
    assert found=={'build_experimental_figure','build_architecture_figure'},found
    nb['metadata']['bruna_revision']={'date':'2026-11-07','source_notebook_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'layout_module_sha256':hashlib.sha256(LAYOUT.read_bytes()).hexdigest(),'source_run_id':'20260809_223452_c0613d3c','executed':False}
    DEST.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(json.dumps({'notebook':str(DEST),'cells':len(nb['cells']),'executed':False,'sha256':hashlib.sha256(DEST.read_bytes()).hexdigest()},ensure_ascii=False))
    return DEST
if __name__=='__main__':build()
