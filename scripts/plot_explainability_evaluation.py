"""Render regional Integrated Gradients and temporal deletion as vector figures."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--region-scores", type=Path, required=True)
parser.add_argument("--deletion-scores", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(args.region_scores)
labels = {'lips':'Lábios', 'face_oval':'Contorno', 'nose':'Nariz',
          'left_eye':'Olho esquerdo', 'right_eye':'Olho direito',
          'left_eyebrow':'Sobrancelha esquerda', 'right_eyebrow':'Sobrancelha direita'}
groups = ['African American','Asian','Caucasian']
colors = ['#28609A','#AF5C2A','#2B795F']
fig, ax = plt.subplots(figsize=(9.0,4.1), layout='constrained')
pos = np.arange(len(labels))
for offset, group, color in zip((-0.24,0,0.24),groups,colors):
    part = df[df.Race.eq(group)].set_index('region').loc[list(labels)]
    ax.barh(pos+offset, part.mean_importance, height=.23, color=color,
            label=f'{group} (n={int(part.n.iloc[0])})')
ax.set_yticks(pos, list(labels.values()))
ax.invert_yaxis()
ax.set_xlim(0,.30)
ax.set_xlabel('Importância média normalizada')
ax.set_title('Gradientes Integrados por região facial')
ax.legend(loc='lower right', frameon=True)
ax.grid(axis='x',alpha=.2)
ax.set_axisbelow(True)
fig.savefig(args.output_dir/'importancia_regioes_por_ig.pdf')
plt.close(fig)

deletion = pd.read_csv(args.deletion_scores)
fig, axes = plt.subplots(1,3,figsize=(10,3.5),sharex=True,sharey=True,layout='constrained')
for ax, group in zip(axes, groups):
    part = deletion[deletion.Race.eq(group)].groupby('fraction_deleted')[['masked_prob','random_masked_prob']].mean()
    ax.plot(part.index*100, part.masked_prob, color='#28609A', marker='o', markersize=3, label='Atenção')
    ax.plot(part.index*100, part.random_masked_prob, color='#AF5C2A', linestyle='--', marker='s', markersize=3, label='Ordem aleatória')
    ax.set_title(group, fontsize=10)
    ax.set_ylim(0,1)
    ax.set_xticks([0,25,50,75,100])
    ax.grid(alpha=.2)
axes[0].set_ylabel('Probabilidade média da classe explicada')
axes[-1].legend(loc='upper right', fontsize=8)
fig.supxlabel('Instantes mascarados (%)')
fig.savefig(args.output_dir/'delecao_atencao_aleatoria.pdf')
