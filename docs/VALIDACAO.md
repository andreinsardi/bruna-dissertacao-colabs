# Validação do acervo — 07/11/2026

Foram conferidos os 16 notebooks incluídos no manifesto, sem regravá-los. A conferência abrangeu o formato Jupyter, os hashes integrais e do código, a sintaxe de 329 células de código e de 49 módulos ou etapas embutidos e a presença das saídas preservadas. Nenhum dos notebooks contém saída Jupyter do tipo `error`. Os treze arquivos históricos permanecem com seus hashes anteriores.

As cópias limpas e executadas dos notebooks 08 e 09 possuem código idêntico. As versões em inglês possuem a mesma árvore sintática Python das versões em português, após a transformação dos comandos IPython. A diferença de comentários e de texto explicativo não alterou o código executável nessa comparação.

O arquivo executado do notebook 08 preserva o SHA-256 citado na dissertação e o marcador `EXECUÇÃO CONCLUÍDA E PERSISTIDA: 20260809_223452_c0613d3c`. O relatório detalhado está em [VALIDACAO_NOTEBOOKS.json](VALIDACAO_NOTEBOOKS.json).

Os 49 testes dos módulos de recuperação, análise, figuras e avaliação externa passaram. Eles cobrem persistência e recuperação, integridade das partições e das predições, agregação por ator, permutação em blocos, ajuste de Holm, identidade AST das arquiteturas e do extrator, alinhamento tensor/manifesto, recorte de classes RAVDESS, verificação de downloads e mapas agregados. Uma exceção `boom` é deliberadamente provocada e capturada pelo teste de recuperação; arquivos corrompidos também são deliberadamente rejeitados no teste de download. Essas mensagens não representam falha da suíte.

Também foram conferidos os links locais dos arquivos Markdown e a leitura dos metadados de citação. A busca por padrões reconhecíveis de tokens e chaves privadas nos arquivos a publicar não identificou correspondências; ela não é uma prova formal de ausência de toda informação sensível.

Esta verificação local do acervo não executa os notebooks completos nem substitui a auditoria dos resultados. Separadamente, os Colabs 10 e 11 foram executados no ambiente dos autores; o estado da validação externa do Colab 12 está registrado em [REVISAO_20261107.md](REVISAO_20261107.md). As cópias históricas executadas e seus hashes permanecem identificados no manifesto.

Comandos de conferência:

```bash
python -m pip install -r requirements-validation.txt
python scripts/validate_notebooks.py
python -m unittest discover -s tests -v
```
