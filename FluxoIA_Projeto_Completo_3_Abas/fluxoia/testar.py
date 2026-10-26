"""Verifica que o modelo e a demonstração podem executar totalmente offline."""
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
from modelo_utils import preparar_calendario

p=Path(__file__).resolve().parent/'artefatos'
pacote=joblib.load(p/'modelo_fluxo.joblib')
metricas=json.loads((p/'metricas.json').read_text(encoding='utf-8'))
teste=pd.read_csv(p/'demonstracao_teste.csv')
X=preparar_calendario(pd.to_datetime(teste['date_time'].head(8)))
preds=np.asarray(pacote['modelo'].predict(X))
assert np.isfinite(preds).all()
assert len(preds)==8
assert len(teste)==metricas['teste_n']
assert 'Baseline por dia/hora' in metricas['resultados']
print('OK: modelo, métricas, teste histórico e inferência offline funcionando.')
print('Modelo selecionado:',metricas['selecionado'])
print('MAE no teste:',metricas['resultados'][metricas['selecionado']]['teste']['mae'])
