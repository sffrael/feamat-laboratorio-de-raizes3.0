# FEAMAT — Laboratório de Raízes

Aplicação educacional em Streamlit para explorar os métodos numéricos da Bisseção e de Newton-Raphson.

## Funcionalidades
- Funções predefinidas com preenchimento automático de `f(x)`, `a`, `b` e `x₀`.
- Entrada de função personalizada.
- Análise explícita dos sinais de `f(a)`, `f(b)` e `f(m)`.
- Tabela de iterações da Bisseção com a decisão do intervalo mantido.
- Tabela de iterações de Newton-Raphson.
- Gráfico da função e marcação das aproximações.
- Exportação das tabelas em CSV.

## Executar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicar no Streamlit Community Cloud
1. Envie `app.py`, `requirements.txt` e `README.md` para a raiz do repositório no GitHub.
2. Acesse https://share.streamlit.io/
3. Crie um aplicativo selecionando o repositório, a branch `main` e o arquivo `app.py`.
4. Clique em Deploy.

## Sintaxe das funções
Use `x` como variável. Potências podem ser escritas com `^` ou `**`.
Funções aceitas: `sin`, `sen`, `cos`, `tan`, `tg`, `exp`, `sqrt`, `log`, `ln`, `abs`.
Constantes aceitas: `pi`, `e`.

Exemplos: `x^2 - 2`, `cos(x) - x`, `exp(x) - 3`, `log(x) - 2`.
