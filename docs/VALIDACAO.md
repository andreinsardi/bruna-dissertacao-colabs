# Validação do acervo — 30/10/2026

Foram conferidos os 13 notebooks incluídos no manifesto, sem regravá-los. A conferência abrangeu o formato Jupyter, os hashes integrais e do código, a sintaxe de 304 células de código e de 49 módulos ou etapas embutidos e a presença das saídas preservadas. Nenhum dos notebooks contém saída Jupyter do tipo `error`.

As cópias limpas e executadas dos notebooks 08 e 09 possuem código idêntico. As versões em inglês possuem a mesma árvore sintática Python das versões em português, após a transformação dos comandos IPython. A diferença de comentários e de texto explicativo não alterou o código executável nessa comparação.

O arquivo executado do notebook 08 preserva o SHA-256 citado na dissertação e o marcador `EXECUÇÃO CONCLUÍDA E PERSISTIDA: 20260809_223452_c0613d3c`. O relatório detalhado está em [VALIDACAO_NOTEBOOKS.json](VALIDACAO_NOTEBOOKS.json).

Os 11 testes dos módulos de recuperação e análise passaram. Eles cobrem persistência de estado e checkpoints, recuperação de falhas, reutilização condicionada à presença de artefatos, histórico de treinamento, cobertura das partições externas, ausência de sobreposição de atores e agregação das métricas e matrizes de confusão. Uma exceção `boom` é deliberadamente provocada e capturada pelo teste de recuperação; a mensagem impressa pelo módulo não representa falha da suíte.

Também foram conferidos os links locais dos arquivos Markdown e a leitura dos metadados de citação. A busca por padrões reconhecíveis de tokens e chaves privadas nos arquivos a publicar não identificou correspondências; ela não é uma prova formal de ausência de toda informação sensível.

Esta verificação não executou os notebooks completos nem repetiu treinamento, download de dados ou reconstrução integral das figuras. A evidência experimental continua sendo a execução preservada e identificada no manifesto.

Comandos de conferência:

```bash
python -m pip install -r requirements-validation.txt
python scripts/validate_notebooks.py
python -m unittest discover -s tests -v
```
