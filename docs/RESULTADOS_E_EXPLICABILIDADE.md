# Resultados e explicabilidade na dissertação

Referência: edição de 11/10/2026, execução `20260809_223452_c0613d3c`. O notebook 08 executado contém as saídas dos métodos e o 09 produz composições científicas a partir dos mesmos artefatos. Os valores abaixo descrevem essa execução, sem novo treinamento neste repositório.

| Método ou verificação | Evidência no trabalho | Figuras da dissertação |
|---|---|---|
| Gradientes Integrados | Importância de marcos, regiões e blocos; aproximadamente 69% da atribuição nos três componentes dinâmicos | 5.10, 5.11 e 5.12 |
| Gradiente × Entrada | Atribuição complementar por atributo, marco e região no notebook 08 | Método descrito na seção 4.8; sem figura própria dedicada na edição |
| Attention Rollout | Relevância temporal obtida pela composição das matrizes de atenção | 5.9 e 5.12 |
| Oclusão regional | Queda de acurácia de 0,439, 0,537 e 0,422 após zerar o núcleo expressivo nos grupos African American, Asian e Caucasian | 5.12 e 5.13 |
| Deleção temporal | Redução da probabilidade da classe explicada após remover quadros relevantes | 5.14 |
| Conferência com Captum | 64 amostras; cosseno médio 0,9991 e Spearman médio 0,9983 entre implementações de Gradientes Integrados | Painel F da figura 5.12 |

Captum é a biblioteca utilizada na conferência numérica, não um método de explicação adicional. ADI, TBD e Jaccard comparam explicações entre grupos; não são modelos preditivos. A importância foi calculada na representação geométrica do modelo, não como inferência direta de estados psicológicos.

No protocolo fixo, o EmoT apresentou acurácia 0,714 e macro-F1 0,710. Nas cinco partições externas por ator, o macro-F1 médio foi 0,658 ± 0,046. As predições e métricas são condicionadas à modalidade visual, aos rótulos de atuação e às partições documentadas.

A figura 5.12 mistura explicitamente um exemplo local (quadro, marcos e atribuições de um vídeo) com resultados agregados de oclusão e conferência Captum. Os painéis agregados não devem ser apresentados como se pertencessem somente ao vídeo ilustrado. Os três componentes dinâmicos são diferenças no tempo reamostrado; a terceira ordem não possui unidades de jerk físico.

O texto explicativo está nas seções 4.8 (métodos), 5.8 (explicações espaço-temporais), 5.9 (oclusão e fidelidade) e 6.4 (discussão de explicabilidade e equidade explicativa).
