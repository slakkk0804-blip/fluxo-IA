# FluxoIA — Projeto completo (SENAC UC9 / Grupo 6)

Este é o projeto inteiro na **versão de 3 abas**, incluindo a **Supervisão inteligente** com YOLO. Os dados para treinamento vêm da base pública real Metro Interstate Traffic Volume, UCI. O painel não depende de API em tempo real.

## Arquivos incluídos

- `app.py`: interface Streamlit com 3 abas — **Previsões e validação (1–2)**, **Planejamento e dados (3–4)** e **Supervisão inteligente (5)**.
- `supervisao.py`: detecção/rastreamento de veículos em MP4, contagem por categoria, linha virtual e alertas de paradas prolongadas (não confirma acidentes).
- `preparar.py`: baixa a base da UCI **uma única vez**, trata dados, treina e compara modelos; gera os artefatos locais e a apresentação HTML.
- `modelo_utils.py`: transformação das datas em variáveis usadas pelo modelo.
- `gerar_apresentacao.py`: gera `APRESENTACAO_OFFLINE.html` após o treinamento.
- `testar.py`: testa inferência e métricas offline.
- `requirements.txt` e `requirements_supervisao.txt`: dependências.
- `data/`: pasta do CSV original após a primeira preparação.
- `artefatos/`: pasta dos modelos e resultados gerados no treinamento.

**Importante:** o pacote contém o código completo, mas não inclui o arquivo treinado do computador anterior nem o CSV original. Ambos são recriados na primeira execução de `preparar.py` com acesso à internet. Depois disso, o aplicativo funciona offline, exceto se o YOLO ainda precisar baixar seus pesos na primeira análise de vídeo.

## Como usar no Windows (VS Code / PowerShell)

Abra no VS Code a pasta `fluxoia` (a que contém `app.py`). Execute, uma linha de cada vez:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements_supervisao.txt
.\.venv\Scripts\python.exe preparar.py
.\.venv\Scripts\python.exe testar.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

O programa informará um endereço como **http://localhost:8501**. Abra no navegador. Se os arquivos de `artefatos/` já existirem (em outra cópia sua), pode copiá-los para esta pasta e pular `preparar.py`.

Na primeira análise de um vídeo, o YOLO pode precisar baixar os próprios pesos. Para uma apresentação sem internet, faça essa primeira análise antes.

## O que demonstra

1. **Previsões e validação** — previsão livre por dia/hora e comparação de previsões versus valores realmente observados na base histórica.
2. **Planejamento e dados** — simulação *hipotética* de exposição ao tráfego em manutenção e gráficos descritivos por hora/dia.
3. **Supervisão inteligente** — upload de MP4, opção de mostrar/ocultar vídeo, detecção de carros/motos/ônibus/caminhões, contagem de objetos distintos e de travessias de linha, e alertas de veículos aparentemente parados por muito tempo. **Alertas não comprovam acidentes.**

## Fonte e limites

**Dataset:** Metro Interstate Traffic Volume, UCI Machine Learning Repository, autoria John Hogue, https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume (licença CC BY 4.0). Registros históricos de um trecho da I-94, nos Estados Unidos, entre 2012 e 2018. **O modelo não foi treinado para o trânsito de Cuiabá.**

O módulo de previsão usa dados históricos de calendário e não tem acesso ao tráfego atual. A previsão livre para 2026 é projeção fora do período de treinamento, sem validação específica para 2026. A visão computacional usa modelo YOLO separado do Random Forest. A contagem pode falhar com oclusão, sombras, ângulo da câmera e rastreamento instável.

## Se o download da UCI falhar

Baixe o dataset manualmente na página da UCI, extraia `Metro_Interstate_Traffic_Volume.csv` e copie-o para a pasta `data/`, mantendo esse nome. Execute `preparar.py` novamente.
