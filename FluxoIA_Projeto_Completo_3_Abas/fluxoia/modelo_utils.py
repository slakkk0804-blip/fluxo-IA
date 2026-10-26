"""Transformações exclusivamente de calendário: nenhuma previsão meteorológica é exigida na inferência.
Usar exatamente as mesmas funções no treino e na interface evita descompasso de variáveis.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

FEATURES = ["hora", "dia_semana", "mes", "dia_ano", "fim_semana", "hora_sin", "hora_cos", "semana_sin", "semana_cos", "ano_sin", "ano_cos"]


def preparar_calendario(datas):
    datas = pd.Series(pd.to_datetime(datas, errors="raise"))
    hora = datas.dt.hour.astype(int)
    semana = datas.dt.dayofweek.astype(int)
    mes = datas.dt.month.astype(int)
    dia_ano = datas.dt.dayofyear.astype(int)
    x = pd.DataFrame({
        "hora": hora, "dia_semana": semana, "mes": mes,
        "dia_ano": dia_ano, "fim_semana": (semana >= 5).astype(int),
        "hora_sin": np.sin(2*np.pi*hora/24),
        "hora_cos": np.cos(2*np.pi*hora/24),
        "semana_sin": np.sin(2*np.pi*semana/7),
        "semana_cos": np.cos(2*np.pi*semana/7),
        "ano_sin": np.sin(2*np.pi*dia_ano/365.25),
        "ano_cos": np.cos(2*np.pi*dia_ano/365.25),
    })
    return x[FEATURES]


def obter_grupo_horario(datas):
    ds = pd.Series(pd.to_datetime(datas))
    h = ds.dt.hour
    return np.select([h.between(0,5),h.between(6,11),h.between(12,17)],
                     ["Madrugada", "Manhã", "Tarde"], default="Noite")
