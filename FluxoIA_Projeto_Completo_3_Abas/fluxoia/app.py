"""FluxoIA: painel demonstrativo totalmente offline após a etapa preparar.py."""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from modelo_utils import preparar_calendario

BASE=Path(__file__).resolve().parent
ARTEFATOS=BASE/'artefatos'

st.set_page_config(page_title='FluxoIA | Volume de Tráfego',page_icon='🚗',layout='wide')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"],button,input{font-family:Inter,Arial,sans-serif!important}
.stApp{background:#f6f7fc;color:#18233c}
.block-container{max-width:1320px;padding-top:2rem;padding-bottom:4rem}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid #eaecf3}
[data-testid="stSidebar"] *{color:#293852}
h1,h2,h3,h4,h5,h6,p,div,span{color:inherit}
.hero{padding:4px 0 15px}
.hero strong{font-size:1.95rem;font-weight:800;color:#1d2945;letter-spacing:-.06rem}
.hero p{color:#52627d;font-size:.92rem;margin-top:5px}
.mini{color:#62708b!important;font-size:.82rem}
.cblock{background:#fff;border:1px solid #e9edf5;border-radius:18px;padding:22px;box-shadow:0 8px 24px rgba(40,52,88,.035)}
.stMetric{background:#fff;border:1px solid #e9edf5;border-radius:18px;padding:18px 21px;box-shadow:0 8px 24px rgba(40,52,88,.035)}
[data-testid="stMetricLabel"]{font-size:.87rem!important;color:#42516d!important}
[data-testid="stMetricValue"]{color:#162743!important}
div[data-testid="stAlert"] p{color:#24334c!important}
.stSelectbox label,.stSlider label{color:#253954!important}
[data-testid="stExpander"]{background:#fff;border:1px solid #e8ebf2;border-radius:14px;color:#192b48}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{color:#42516d!important}
.pill{display:inline-block;background:#e8edf9;color:#4358a8!important;padding:6px 10px;border-radius:20px;font-weight:700;font-size:.74rem;margin-right:6px}
.banner{background:#e7eefc;border:1px solid #d7e2f5;border-radius:12px;padding:13px 16px;color:#2d487e;font-size:.88rem;line-height:1.5}
small{color:#64708a}

/* Destaque discreto para a navegação principal. */
button[data-baseweb="tab"]{font-weight:700!important;min-height:48px!important}
button[data-baseweb="tab"][aria-selected="true"]{color:#4059b8!important}
[data-testid="stTabs"] [data-baseweb="tab-panel"]{padding-top:16px}
</style>''',unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def carregar_modelo():
    return joblib.load(ARTEFATOS/'modelo_fluxo.joblib')

@st.cache_data(show_spinner=False)
def carregar_dados():
    resumo=json.loads((ARTEFATOS/'metricas.json').read_text(encoding='utf-8'))
    teste=pd.read_csv(ARTEFATOS/'demonstracao_teste.csv',parse_dates=['date_time'])
    return resumo,teste


def pt(valor,casas=0):
    return f'{valor:,.{casas}f}'.replace(',','_').replace('.',',').replace('_','.')


def prever_data(modelo,datas):
    entrada=preparar_calendario(pd.Series(pd.to_datetime(datas)))
    return np.maximum(0,modelo['modelo'].predict(entrada))


def grafico_padrao(fig,height=310):
    fig.update_layout(height=height,margin=dict(l=8,r=10,t=14,b=12),
                      plot_bgcolor='#ffffff',paper_bgcolor='#ffffff',
                      font=dict(color='#40506c',size=12),showlegend=True,
                      legend=dict(orientation='h',y=-.2,x=0),
                      hoverlabel=dict(bgcolor='#ffffff',font_color='#1d2d47'))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor='#eef1f7',zeroline=False)
    return fig


# Barra lateral apenas com a identificação do projeto.
with st.sidebar:
    st.markdown('### 🚗 FluxoIA')
    st.caption('SENAC · UC9 · Grupo 6')

st.markdown('<div class="hero"><strong>FluxoIA</strong></div>',unsafe_allow_html=True)

faltando=[p for p in ['modelo_fluxo.joblib','metricas.json','demonstracao_teste.csv'] if not (ARTEFATOS/p).exists()]
if faltando:
    st.warning('O modelo ainda não foi treinado neste computador. Nenhum resultado foi inventado.')
    st.code('python preparar.py\nstreamlit run app.py',language='bash')
    st.info('A preparação obtém a base pública da UCI uma vez, treina e compara os modelos, salva as métricas e os exemplos. Depois o painel funciona offline.')
    st.stop()

try:
    pacote=carregar_modelo(); meta,dados=carregar_dados()
except Exception as e:
    st.error(f'Erro ao abrir os arquivos treinados: {e}')
    st.stop()

selecionado=meta['selecionado']
met=meta['resultados'][selecionado]['teste']
base=meta['resultados']['Baseline por dia/hora']['teste']

# Navegação: todas as funcionalidades ficam organizadas em três abas.
aba_previsoes, aba_planejamento, aba_supervisao = st.tabs([
    '📈 Previsões e validação',
    '🛠️ Planejamento e dados',
    '📹 Supervisão inteligente',
])

with aba_previsoes:
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Modelo selecionado',selecionado)
    c2.metric('MAE no teste',f"{pt(met['mae'])} veículos")
    c3.metric('Baseline MAE',f"{pt(base['mae'])} veículos")
    c4.metric('Registros históricos',pt(meta['origem_linhas']))
    st.write('')

    # PREVISÃO LIVRE: usa somente atributos conhecidos do calendário, sem API e sem novo treino.
    st.subheader('1. Previsão livre — qualquer data e horário')
    hoje = datetime.now()
    cl1, cl2 = st.columns([1, 1], gap='medium')
    with cl1:
        data_livre = st.date_input('Data da previsão', value=hoje.date(), key='data_livre', format='DD/MM/YYYY')
    with cl2:
        hora_livre = st.selectbox('Hora da previsão', options=list(range(24)), index=hoje.hour,
                                 format_func=lambda h: f'{h:02d}:00', key='hora_livre')

    # Calcula o ponto escolhido e uma curva de 24 h com as mesmas variáveis do treinamento.
    instante = pd.Timestamp(data_livre) + pd.Timedelta(hours=int(hora_livre))
    horas_livres = pd.date_range(pd.Timestamp(data_livre), periods=24, freq='h')
    previsoes_livres = prever_data(pacote, horas_livres)
    prev_livre = float(previsoes_livres[int(hora_livre)])
    # Baseline não depende de rede: médias históricas salvas durante a etapa preparar.py.
    grupo_base = pacote.get('baseline_grupo', {})
    hora_base = pacote.get('baseline_hora', {})
    geral_base = float(pacote.get('baseline_geral', 0))
    def media_simples_calendario(dia_semana, hora):
        return float(grupo_base.get((int(dia_semana), int(hora)),
                                    hora_base.get(int(hora), geral_base)))
    comparacao_livre = [media_simples_calendario(dia_l.weekday(), dia_l.hour)
                        for dia_l in horas_livres]

    m1,m2,m3 = st.columns(3)
    m1.metric('Previsão do Random Forest', f'{pt(prev_livre)} veículos/h')
    m2.metric('Média histórica de referência', f'{pt(comparacao_livre[hora_livre])} veículos/h')
    m3.metric('Data e horário escolhidos', instante.strftime('%d/%m/%Y · %Hh'))

    fig_livre = go.Figure()
    fig_livre.add_trace(go.Scatter(x=horas_livres, y=previsoes_livres, mode='lines+markers',
                                  name='Previsão do Random Forest', line=dict(color='#5468c8', width=3)))
    fig_livre.add_trace(go.Scatter(x=horas_livres, y=comparacao_livre, mode='lines',
                                  name='Baseline histórico', line=dict(color='#37b4c0', width=2, dash='dash')))
    fig_livre.add_vline(x=instante, line_dash='dot', line_color='#d98a40', line_width=2)
    fig_livre.update_yaxes(title_text='Veículos por hora')
    st.plotly_chart(grafico_padrao(fig_livre, 315), use_container_width=True)

    opcoes_operacao = range(9, 19)
    mais_tranquilo = min(opcoes_operacao, key=lambda h: float(previsoes_livres[h]))

    st.divider()
    st.subheader('2. Demonstração histórica — previsão x realidade')
    dados['data']=dados['date_time'].dt.date
    datas=dados['data'].drop_duplicates().to_list()
    if not datas:st.error('Conjunto de teste vazio.');st.stop()

    esq,dir=st.columns([1,2],gap='large')
    with esq:
        data=st.selectbox('Dia observado (teste)',datas,index=min(len(datas)-1,max(0,len(datas)//3)),format_func=lambda d:pd.Timestamp(d).strftime('%d/%m/%Y'))
        desse_dia=dados[dados['data']==data].copy().sort_values('date_time')
        hora=st.selectbox('Horário',list(desse_dia['date_time'].dt.strftime('%H:%M').values),index=min(10,len(desse_dia)-1))
        registro=desse_dia.loc[desse_dia['date_time'].dt.strftime('%H:%M')==hora].iloc[0]
        st.metric('Previsão do modelo',f"{pt(registro['pred_modelo'])} veículos/h")
        st.metric('Volume real observado',f"{pt(registro['traffic_volume'])} veículos/h")
        st.caption(f"Erro absoluto neste exemplo: {pt(abs(registro['pred_modelo']-registro['traffic_volume']))} veículos/h.")
        st.write('**Para que serve?** Um gestor pode comparar períodos e avaliar quando uma intervenção teria menor exposição ao tráfego.')
    with dir:
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=desse_dia['date_time'],y=desse_dia['traffic_volume'],name='Volume observado',mode='lines+markers',line=dict(color='#6777d6',width=3)))
        fig.add_trace(go.Scatter(x=desse_dia['date_time'],y=desse_dia['pred_modelo'],name='Previsão IA',mode='lines+markers',line=dict(color='#39b5d4',width=2,dash='dash')))
        fig.update_yaxes(title_text='Veículos por hora')
        st.plotly_chart(grafico_padrao(fig,330),use_container_width=True)
        st.caption('Os dois valores correspondem a horários históricos; não se trata de previsão em tempo real.')


with aba_planejamento:
    st.subheader('3. Planejamento de manutenção — cenário demonstrativo')
    candidatos=desse_dia[desse_dia['date_time'].dt.hour.between(9,18)].copy()
    if not candidatos.empty:
        escolhido=candidatos.loc[candidatos['pred_modelo'].idxmin()]
        fixo=candidatos[candidatos['date_time'].dt.hour==16]
        if not fixo.empty:
            custo_por_carro=st.slider('Custo hipotético por veículo exposto à intervenção (R$)',0.0,1.0,0.10,step=0.05)
            baseline_obs=float(fixo.iloc[0]['traffic_volume']);sugerido_obs=float(escolhido['traffic_volume'])
            a,b,c=st.columns(3)
            a.metric('Horário com menor previsão',escolhido['date_time'].strftime('%Hh'))
            b.metric('Volume real nesse horário',f'{pt(sugerido_obs)} veículos')
            c.metric('Diferença hipotética de custo',f'R$ {pt((baseline_obs-sugerido_obs)*custo_por_carro,2)}')
        else:
            st.info('Este dia não contém registro de 16h para comparação.')
    else:
        st.info('Não há horas entre 9h e 18h suficientes para esta demonstração.')

    st.subheader('4. O que os dados históricos mostram?')
    l,r=st.columns(2,gap='medium')
    eda=meta['eda']
    with l:
        fig=px.bar(pd.DataFrame(eda['media_por_hora']),x='hora',y='volume',labels={'hora':'Hora do dia','volume':'Veículos/h (média)'},color_discrete_sequence=['#8d80e1'])
        st.plotly_chart(grafico_padrao(fig,300),use_container_width=True)
    with r:
        df=pd.DataFrame(eda['media_por_dia_semana'])
        df['dia']=df['dia'].map({0:'Seg',1:'Ter',2:'Qua',3:'Qui',4:'Sex',5:'Sáb',6:'Dom'})
        fig=px.bar(df,x='dia',y='volume',labels={'dia':'Dia da semana','volume':'Veículos/h (média)'},color_discrete_sequence=['#37b9c8'])
        st.plotly_chart(grafico_padrao(fig,300),use_container_width=True)
        st.caption('Padrão semanal da rodovia monitorada nos Estados Unidos.')




with aba_supervisao:
    from supervisao import mostrar_supervisao
    mostrar_supervisao()

