"""Obtém dataset público apenas nesta fase, treina/valida e salva tudo para execução offline.

Uso: python preparar.py
Se não puder baixar automaticamente, coloque o CSV da UCI em data/Metro_Interstate_Traffic_Volume.csv.
"""
from __future__ import annotations
import gzip
import io
import json
from pathlib import Path
import urllib.request
import zipfile
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import permutation_importance
from modelo_utils import preparar_calendario, obter_grupo_horario, FEATURES

BASE = Path(__file__).resolve().parent
DADOS = BASE / 'data'
ART = BASE / 'artefatos'
CSV = DADOS / 'Metro_Interstate_Traffic_Volume.csv'
URLS = [
    'https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip',
    'https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz',
    'https://raw.githubusercontent.com/marcopeix/TimeSeriesForecastingInPython/master/data/Metro_Interstate_Traffic_Volume.csv',
]


def obter_dataset():
    DADOS.mkdir(parents=True, exist_ok=True)
    if CSV.exists() and CSV.stat().st_size > 10000:
        print('Usando CSV local:', CSV)
        return CSV
    falhas=[]
    for url in URLS:
        try:
            print('Baixando a base de dados (uma única vez):', url)
            pedido=urllib.request.Request(url,headers={'User-Agent':'FluxoIA-UC9-educacional/1.0 (dataset publico UCI)'})
            with urllib.request.urlopen(pedido,timeout=35) as resposta:
                b=resposta.read()
            if url.endswith('.zip'):
                with zipfile.ZipFile(io.BytesIO(b)) as z:
                    nomes=[n for n in z.namelist() if n.lower().endswith('.csv') or n.lower().endswith('.csv.gz')]
                    if not nomes: raise RuntimeError('ZIP não contém CSV')
                    b=z.read(nomes[0]); b=gzip.decompress(b) if nomes[0].endswith('.gz') else b
            elif url.endswith('.gz'):
                b=gzip.decompress(b)
            if not b.startswith(b'holiday,temp,rain_1h'):
                raise ValueError('Cabeçalho inesperado; download inválido')
            CSV.write_bytes(b)
            print('CSV salvo:', len(b), 'bytes')
            return CSV
        except Exception as e:
            falhas.append(f'{url}: {type(e).__name__}: {e}')
    raise RuntimeError('Não foi possível obter a base automaticamente.\n'
                       'Baixe o arquivo pelo site da UCI e coloque o CSV em data/.\n'+'\n'.join(falhas))


def metricas(y, pred):
    return {'mae':round(float(mean_absolute_error(y,pred)),2),
            'rmse':round(float(np.sqrt(mean_squared_error(y,pred))),2),
            'r2':round(float(r2_score(y,pred)),4)}


def preparar_serie(raw):
    cols={'holiday','temp','rain_1h','snow_1h','clouds_all','weather_main','weather_description','date_time','traffic_volume'}
    if not cols.issubset(raw.columns):raise ValueError('Dataset UCI com colunas ausentes')
    raw['date_time']=pd.to_datetime(raw['date_time'],errors='coerce')
    raw['traffic_volume']=pd.to_numeric(raw['traffic_volume'],errors='coerce')
    raw=raw.dropna(subset=['date_time','traffic_volume']).copy()
    # UCI possui horários repetidos por haver mais de uma observação meteorológica.
    # Unificar por horário evita que a mesma hora apareça múltiplas vezes no treino.
    serie=(raw.groupby('date_time',as_index=False).agg(
        traffic_volume=('traffic_volume','mean'),
        temp=('temp','mean'), rain_1h=('rain_1h','mean'),
        clouds_all=('clouds_all','mean'),
    )).sort_values('date_time').reset_index(drop=True)
    return serie


def main():
    ART.mkdir(parents=True,exist_ok=True)
    caminho=obter_dataset()
    raw=pd.read_csv(caminho,keep_default_na=False)
    serie=preparar_serie(raw)
    if len(serie)<1000:raise ValueError('Base insuficiente para o projeto')
    X=preparar_calendario(serie['date_time'])
    y=serie['traffic_volume'].to_numpy(float)
    a=int(.70*len(serie)); b=int(.85*len(serie))
    xtr,xva,xte=X.iloc[:a],X.iloc[a:b],X.iloc[b:]
    ytr,yva,yte=y[:a],y[a:b],y[b:]
    # Baseline: média histórica por hora e dia da semana (calculada SOMENTE no treino).
    grupo=serie.iloc[:a].copy()
    grupo['hora']=grupo['date_time'].dt.hour
    grupo['dia_semana']=grupo['date_time'].dt.dayofweek
    media_grupo=grupo.groupby(['dia_semana','hora'])['traffic_volume'].mean()
    media_hora=grupo.groupby('hora')['traffic_volume'].mean()
    geral=float(grupo['traffic_volume'].mean())
    def baseline(df):
        return np.asarray([media_grupo.get((int(d),int(h)),media_hora.get(int(h),geral))
                           for d,h in zip(df['dia_semana'],df['hora'])])
    resultados={}
    pred_val={'Baseline por dia/hora':baseline(xva)}
    pred_test={'Baseline por dia/hora':baseline(xte)}
    resultados['Baseline por dia/hora']={'validacao':metricas(yva,pred_val['Baseline por dia/hora']),
                                         'teste':metricas(yte,pred_test['Baseline por dia/hora'])}
    candidatos={
        'Regressão Ridge': Ridge(alpha=100.0),
        'Árvore de decisão': DecisionTreeRegressor(max_depth=11,min_samples_leaf=25,random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=85,max_depth=14,min_samples_leaf=5,
                                               n_jobs=-1,random_state=42),
    }
    modelos={}
    for nome,modelo in candidatos.items():
        print('Treinando:',nome,flush=True)
        modelo.fit(xtr,ytr)
        modelos[nome]=modelo
        pred_val[nome]=modelo.predict(xva)
        pred_test[nome]=modelo.predict(xte)
        resultados[nome]={'validacao':metricas(yva,pred_val[nome]),'teste':metricas(yte,pred_test[nome])}
    # Ajuste efetivo de hiperparâmetros: profundidade validada em período posterior ao treino.
    teste_ajuste={}
    for profundidade in (7,11,15):
        t=DecisionTreeRegressor(max_depth=profundidade,min_samples_leaf=25,random_state=42)
        t.fit(xtr,ytr)
        teste_ajuste[str(profundidade)]=float(mean_absolute_error(yva,t.predict(xva)))
    escolha_arvore=min(teste_ajuste,key=teste_ajuste.get)
    if int(escolha_arvore)!=11:
        t=DecisionTreeRegressor(max_depth=int(escolha_arvore),min_samples_leaf=25,random_state=42).fit(xtr,ytr)
        modelos['Árvore de decisão']=t
        pred_val['Árvore de decisão']=t.predict(xva)
        pred_test['Árvore de decisão']=t.predict(xte)
        resultados['Árvore de decisão']={'validacao':metricas(yva,pred_val['Árvore de decisão']),
                                           'teste':metricas(yte,pred_test['Árvore de decisão'])}
    escolhido=min(modelos,key=lambda nome:resultados[nome]['validacao']['mae'])
    vencedor=modelos[escolhido]
    print('Selecionado por MAE de validação:',escolhido)
    # Explicabilidade via importância por permutação em amostra fixa da validação.
    amostra=np.linspace(0,len(xva)-1,min(1200,len(xva)),dtype=int)
    perm=permutation_importance(vencedor,xva.iloc[amostra],yva[amostra],
                                scoring='neg_mean_absolute_error',n_repeats=3,random_state=42,n_jobs=1)
    importancia=[{'variavel':str(col),'aumento_mae':round(float(valor),2)}
                  for col,valor in sorted(zip(FEATURES,perm.importances_mean),key=lambda kv:kv[1],reverse=True)]
    # Viés operacional por grupos temporais, não viés demográfico.
    testes=serie.iloc[b:].copy()
    testes['pred_modelo']=pred_test[escolhido]
    testes['pred_baseline']=pred_test['Baseline por dia/hora']
    testes['erro_abs_modelo']=np.abs(testes['traffic_volume']-testes['pred_modelo'])
    testes['periodo']=obter_grupo_horario(testes['date_time'])
    testes['tipo_dia']=np.where(testes['date_time'].dt.dayofweek>=5,'Fim de semana','Dia útil')
    viés={campo:[{'grupo':str(k),'n':int(len(p)),'mae':round(float(p['erro_abs_modelo'].mean()),2)}
                  for k,p in testes.groupby(campo)] for campo in ['periodo','tipo_dia']}
    # Simulação em veículos expostos por interrupção em um horário: cenário puramente ilustrativo.
    # Projeta horas de menor demanda por dia no teste, comparando à escolha fixa das 16h.
    testes['data']=testes['date_time'].dt.date
    expostos=[]
    for data, df in testes.groupby('data'):
        observado_16=df[df['date_time'].dt.hour==16]
        candidatos=df[df['date_time'].dt.hour.between(9,18)]
        if observado_16.empty or len(candidatos)<7:continue
        menor=candidatos.loc[candidatos['pred_modelo'].idxmin()]
        expostos.append({'dia':str(data),'horario_sugerido':str(menor['date_time']),
                         'real_hora_sugerida':float(menor['traffic_volume']),
                         'real_16h':float(observado_16.iloc[0]['traffic_volume'])})
    resumo_impacto={
        'dias_avaliados':len(expostos),
        'media_veiculos_16h':round(float(np.mean([r['real_16h'] for r in expostos])),1) if expostos else None,
        'media_veiculos_hora_sugerida':round(float(np.mean([r['real_hora_sugerida'] for r in expostos])),1) if expostos else None,
        'aviso':'Simulação retrospectiva em rodovia estrangeira; não mede economia, congestionamento ou benefício real.'}
    pacote={'modelo':vencedor,'colunas':FEATURES,'nome':escolhido,
            'baseline_grupo':media_grupo.to_dict(),
            'baseline_hora':media_hora.to_dict(),'baseline_geral':geral}
    joblib.dump(pacote,ART/'modelo_fluxo.joblib',compress=3)
    # Guarda APENAS os registros de teste para demonstração offline com valores observados.
    testes[['date_time','traffic_volume','pred_modelo','pred_baseline','periodo','tipo_dia']].to_csv(
        ART/'demonstracao_teste.csv',index=False,float_format='%.2f')
    # Gráficos descritivos reais, sem depender de dataset no app.
    serie['hora']=serie['date_time'].dt.hour
    serie['dia_semana']=serie['date_time'].dt.dayofweek
    eda={
        'media_por_hora':[{ 'hora':int(h),'volume':round(float(v),1)} for h,v in serie.groupby('hora')['traffic_volume'].mean().items()],
        'media_por_dia_semana':[{ 'dia':int(h),'volume':round(float(v),1)} for h,v in serie.groupby('dia_semana')['traffic_volume'].mean().items()],
        'volume_medio':round(float(y.mean()),2),
        'volume_minimo':round(float(y.min()),2),'volume_maximo':round(float(y.max()),2),
        'percentual_horarios_duplicados':round(100*(len(raw)-len(serie))/len(raw),2),
    }
    met={
        'fonte':'UCI Metro Interstate Traffic Volume (John Hogue, 2019)',
        'dataset_url':'https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume',
        'licenca':'CC BY 4.0','localidade':'I-94 sentido oeste, entre Minneapolis e St. Paul, Minnesota, EUA',
        'origem_linhas':int(len(raw)),'horas_unicas':int(len(serie)),
        'variaveis_originais':list(raw.columns),'atributos_modelo':FEATURES,
        'periodo_de':str(serie['date_time'].min()),'periodo_ate':str(serie['date_time'].max()),
        'treino_n':a,'validacao_n':b-a,'teste_n':len(serie)-b,
        'treino_fim':str(serie['date_time'].iloc[a-1]),
        'validacao_fim':str(serie['date_time'].iloc[b-1]),
        'metodo':'Divisão cronológica 70% treino, 15% validação, 15% teste; sem embaralhar.',
        'selecionado':escolhido,'resultados':resultados,'arvore_profundidade_mae_validacao':teste_ajuste,
        'arvore_profundidade_selecionada':int(escolha_arvore),
        'importancia_permutacao':importancia,'comparacao_grupos':viés,
        'simulacao_retroativa':resumo_impacto,'eda':eda,
        'limitacoes':[
            'Medições reais de trecho de rodovia dos Estados Unidos, não representam Cuiabá.',
            'Sem dados de velocidade, volume de veículos não prova congestionamento.',
            'Previsão usa apenas calendário; não antecipa acidentes, obras ou condições climáticas específicas.',
            'Demonstrativo é histórico fora da amostra de treino, não previsão do trânsito ao vivo.',
        ],
    }
    (ART/'metricas.json').write_text(json.dumps(met,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'selecionado':escolhido,'resultado_teste':resultados[escolhido]['teste'],
                       'baseline_teste':resultados['Baseline por dia/hora']['teste'],
                       'amostras':len(serie)},indent=2,ensure_ascii=False))
    from gerar_apresentacao import gerar
    gerar()
    print('PRONTO. Os arquivos em artefatos/ bastam para executar o aplicativo sem internet.')

if __name__=='__main__':main()
