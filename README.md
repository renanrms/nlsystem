# nlsystem

Pacote para análise de sistemas não necessariamente lineares.

## Desenvolvimento

Para desenvolver é essencial criar o ambiente necessário para rodar o módulo e os testes. Para isto, siga os passos abaixo, a partir da pasta raiz do reporitório.

1. Inicie um ambiente virtual

Crie o ambiente para desenvolvimento com `python3 -m venv .venv`.
Ative com `source ./.venv/bin/activate`. Quando quiser desativar use o comando `deactivate`.

2. Faça uma instalação editável do pacote com `pip install -e .` (Instalando desta maneira o pacote sempre conterá o código atual).

3. Rode os testes com `pytest`.