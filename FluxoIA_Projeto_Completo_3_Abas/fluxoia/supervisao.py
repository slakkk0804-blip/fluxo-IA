"""Supervisão inteligente (MP4 local).

Módulo independente do Random Forest de previsão de tráfego.
Detecção COCO (car, motorcycle, bus, truck) e tracking YOLO11n.
Contagens aproximadas e alertas heurísticos de parada prolongada;
NÃO reconhece ou diagnostica acidentes de trânsito.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import subprocess
import tempfile
import time

import streamlit as st

CLASSES = {2: "Carros", 3: "Motocicletas", 5: "Ônibus", 7: "Caminhões"}
CORES = {
    2: (223, 129, 67), 3: (185, 144, 95), 5: (90, 174, 67), 7: (108, 106, 219)
}  # BGR, apenas para caixas no vídeo.


@dataclass
class Trilho:
    classe: int
    ultima_posicao: tuple[float, float]
    primeira_aparicao: float
    ultima_aparicao: float
    quadros: int = 1
    contado: bool = False
    lado_estavel: int = 0
    atravessou: bool = False
    parado_desde: float | None = None
    alerta_emitido: bool = False


def lado_da_linha(posicao: tuple[float, float], eixo: str, linha_px: int, faixa: int) -> int:
    valor = posicao[0] if eixo == "Vertical" else posicao[1]
    if valor < linha_px - faixa:
        return -1
    if valor > linha_px + faixa:
        return 1
    return 0


def atualizar_trilho(
    trilho: Trilho,
    classe: int,
    posicao: tuple[float, float],
    tempo_s: float,
    lado: int,
    deslocamento_min: float,
    segundos_parado: float,
) -> tuple[bool, bool, bool]:
    """Retorna (nova identificação, novo cruzamento, novo alerta de parada)."""
    nova_identificacao = False
    novo_cruzamento = False
    novo_alerta = False

    anterior = trilho.ultima_posicao
    delta = math.dist(anterior, posicao)
    trilho.quadros += 1
    trilho.ultima_posicao = posicao
    trilho.ultima_aparicao = tempo_s

    if trilho.quadros >= 2 and not trilho.contado:
        trilho.contado = True
        nova_identificacao = True

    if lado != 0:
        if trilho.lado_estavel and lado != trilho.lado_estavel and not trilho.atravessou:
            trilho.atravessou = True
            novo_cruzamento = True
        trilho.lado_estavel = lado

    # Evidência fraca: período sem movimento aparente no plano da imagem.
    if delta < deslocamento_min:
        if trilho.parado_desde is None:
            trilho.parado_desde = tempo_s
        if tempo_s - trilho.parado_desde >= segundos_parado and not trilho.alerta_emitido:
            trilho.alerta_emitido = True
            novo_alerta = True
    else:
        trilho.parado_desde = None

    return nova_identificacao, novo_cruzamento, novo_alerta


def converter_video_h264(video_intermediario: Path, video_final: Path) -> tuple[bool, str]:
    """Converte MP4V do OpenCV em H.264, normalmente compatível com navegadores."""
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        proc = subprocess.run(
            [ffmpeg, '-y', '-hide_banner', '-loglevel', 'error', '-i', str(video_intermediario),
             '-an', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '26',
             '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video_final)],
            capture_output=True, text=True, timeout=120, check=False,
        )
        if proc.returncode == 0 and video_final.exists() and video_final.stat().st_size > 0:
            return True, ''
        return False, proc.stderr[-400:] or 'Falha na conversão de vídeo.'
    except Exception as exc:
        return False, str(exc)


def analisar_video(
    caminho: Path,
    *,
    eixo: str,
    porcentagem_linha: int,
    fps_analise: int,
    confianca: float,
    progresso=None,
    quadro_preview=None,
) -> dict:
    """Processa o MP4 com o YOLO pré-treinado. Não acessa APIs durante a análise."""
    import cv2
    from ultralytics import YOLO

    cap = cv2.VideoCapture(str(caminho))
    if not cap.isOpened():
        raise ValueError('Não foi possível abrir o MP4. Tente outro vídeo em formato H.264.')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError('O vídeo não informou uma taxa de quadros válida (FPS).')
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if min(original_w, original_h) < 50 or total_frames <= 0:
        cap.release()
        raise ValueError('Vídeo inválido, curto demais ou sem quadros legíveis.')
    duracao = total_frames / fps
    if duracao > 90:
        cap.release()
        raise ValueError('Para esta demonstração, use um vídeo de até 90 segundos.')

    fator = min(1.0, 640.0 / original_w)
    width = max(2, int(original_w * fator) // 2 * 2)
    height = max(2, int(original_h * fator) // 2 * 2)
    passo = max(1, round(fps / fps_analise))
    output_fps = fps / passo
    linha_px = round((width if eixo == 'Vertical' else height) * porcentagem_linha / 100)
    faixa = max(5, round(min(width, height) * .018))

    # Cada vídeo recebe um tracker novo. persist=True só é seguro no MESMO vídeo.
    modelo = YOLO('yolo11n.pt')
    tracks: dict[int, Trilho] = {}
    vistos = Counter({categoria: 0 for categoria in CLASSES.values()})
    cruzamentos = Counter({categoria: 0 for categoria in CLASSES.values()})
    alertas: list[dict] = []
    processados = 0
    last_preview = 0
    iniciado = time.monotonic()

    with tempfile.TemporaryDirectory(prefix='fluxoia_yolo_saida_') as td:
        intermediario = Path(td) / 'analise_mp4v.mp4'
        final = Path(td) / 'analise_h264.mp4'
        writer = cv2.VideoWriter(str(intermediario), cv2.VideoWriter_fourcc(*'mp4v'), output_fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError('Não foi possível gerar o vídeo processado neste computador.')
        try:
            for indice in range(total_frames):
                sucesso, frame = cap.read()
                if not sucesso:
                    break
                if indice % passo != 0:
                    continue
                momento = indice / fps
                reduzido = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                resultado = modelo.track(reduzido, persist=True, tracker='bytetrack.yaml',
                                         conf=confianca, classes=list(CLASSES),
                                         imgsz=640, verbose=False)[0]
                processados += 1
                anotado = reduzido.copy()
                if eixo == 'Vertical':
                    cv2.line(anotado, (linha_px, 0), (linha_px, height), (32, 178, 245), 2)
                else:
                    cv2.line(anotado, (0, linha_px), (width, linha_px), (32, 178, 245), 2)

                if resultado.boxes is not None and resultado.boxes.id is not None:
                    ids = resultado.boxes.id.int().cpu().tolist()
                    cids = resultado.boxes.cls.int().cpu().tolist()
                    coords = resultado.boxes.xyxy.cpu().numpy()
                    for ident, classe, bbox in zip(ids, cids, coords):
                        if classe not in CLASSES:
                            continue
                        x1, y1, x2, y2 = (int(v) for v in bbox)
                        pos = ((x1+x2)/2.0, (y1+y2)/2.0)
                        lado = lado_da_linha(pos, eixo, linha_px, faixa)
                        if ident not in tracks:
                            tracks[ident] = Trilho(classe, pos, momento, momento,
                                                    lado_estavel=lado)
                        else:
                            track = tracks[ident]
                            # Trocas esporádicas de rótulo no mesmo ID são ignoradas.
                            novo, cruzou, parado = atualizar_trilho(
                                track, classe, pos, momento, lado,
                                deslocamento_min=max(2.0, min(width,height)*0.009),
                                segundos_parado=7.0,
                            )
                            if novo:
                                vistos[CLASSES[track.classe]] += 1
                            if cruzou:
                                cruzamentos[CLASSES[track.classe]] += 1
                            if parado:
                                alertas.append({'tempo': round(momento,1),
                                                'tipo': CLASSES[track.classe],
                                                'id': int(ident),
                                                'motivo': 'Parada prolongada aparente'})

                        cor = CORES[tracks[ident].classe]
                        cv2.rectangle(anotado, (x1, y1), (x2, y2), cor, 2)
                        cv2.putText(anotado, f'{CLASSES[tracks[ident].classe]} #{ident}',
                                    (max(2, x1), max(18, y1-6)), cv2.FONT_HERSHEY_SIMPLEX,
                                    .48, cor, 2, cv2.LINE_AA)
                cv2.rectangle(anotado, (0, 0), (width, 39), (23, 34, 52), -1)
                resumo = f"FluxoIA   Identificados: {sum(vistos.values())}   Cruzaram: {sum(cruzamentos.values())}"
                cv2.putText(anotado, resumo, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, .55,
                            (255,255,255), 2, cv2.LINE_AA)
                writer.write(anotado)
                if progresso is not None and processados % 5 == 0:
                    progresso.progress(min(1.0, (indice+1)/total_frames),
                                       text=f'Analisando vídeo... {round((indice+1)/total_frames*100)}%')
                if quadro_preview is not None and time.monotonic()-last_preview > 1.2:
                    quadro_preview.image(cv2.cvtColor(anotado, cv2.COLOR_BGR2RGB),
                                         caption='Processamento — detecções ao vivo', use_container_width=True)
                    last_preview = time.monotonic()
        finally:
            writer.release()
            cap.release()
        if processados == 0:
            raise ValueError('Nenhum quadro pôde ser analisado.')
        converteu, motivo = converter_video_h264(intermediario, final)
        video_saida = final if converteu else intermediario
        return {
            'identificados': dict(vistos),
            'cruzamentos': dict(cruzamentos),
            'alertas': alertas,
            'quadros': processados,
            'duracao': duracao,
            'fps_processado': output_fps,
            'segundos_processamento': round(time.monotonic()-iniciado,1),
            'video': video_saida.read_bytes(),
            'h264': converteu,
            'aviso_codec': motivo,
        }


def mostrar_supervisao() -> None:
    st.divider()
    st.subheader('5. Supervisão inteligente — veículos em vídeo')

    video = st.file_uploader('Selecione um vídeo MP4 de trânsito', type=['mp4'], key='supervisao_mp4')
    left,right,third = st.columns([1,1,1], gap='medium')
    with left:
        eixo = st.selectbox('Orientação da linha', ['Vertical','Horizontal'],
                            help='Vertical: veículos cruzam da esquerda para a direita (ou vice-versa). Horizontal: de cima para baixo.')
    with right:
        porcentagem = st.slider('Posição da linha (%)', 15,85,50,5)
    with third:
        velocidade = st.selectbox('Análises por segundo', [4,5,6,8], index=1,
                                  help='Menos análises por segundo deixam o processamento mais rápido, mas podem reduzir as contagens.')
    confianca = .25
    mostrar = st.toggle('Mostrar vídeo', value=True, key='supervisao_mostrar_video')
    if video is not None:
        if video.size > 85 * 1024 * 1024:
            st.error('Este MP4 excede 85 MB. Reduza a resolução ou escolha um vídeo menor.')
            return
        conteudo = video.getvalue()
        assinatura = hashlib.sha256(conteudo).hexdigest()
        parametros = (assinatura, eixo, porcentagem, velocidade, confianca)
        gravado = st.session_state.get('supervisao_resultado')
        valido = gravado is not None and gravado.get('parametros') == parametros
        if mostrar and not valido:
            st.video(conteudo, format='video/mp4')
            st.caption('Vídeo original. As caixas e as contagens aparecerão após a análise.')
        if st.button('▶ Analisar com IA', type='primary', key='supervisao_analisar'):
            try:
                import cv2  # noqa: F401
                from ultralytics import YOLO  # noqa: F401
                import imageio_ffmpeg  # noqa: F401
            except ImportError as exc:
                st.error('Falta instalar o módulo de visão computacional. No terminal do VS Code, execute:')
                st.code(r'.\.venv\Scripts\python.exe -m pip install -r requirements_supervisao.txt', language='powershell')
                st.caption(f'Detalhes técnicos: {exc}')
                return
            barra = st.progress(0, text='Preparando análise...')
            preview = st.empty() if mostrar else None
            try:
                with tempfile.TemporaryDirectory(prefix='fluxoia_entrada_') as td:
                    nome = Path(td) / 'entrada.mp4'
                    nome.write_bytes(conteudo)
                    resultado = analisar_video(nome, eixo=eixo, porcentagem_linha=porcentagem,
                                              fps_analise=velocidade, confianca=confianca,
                                              progresso=barra, quadro_preview=preview)
                st.session_state['supervisao_resultado'] = {**resultado, 'parametros': parametros}
                barra.progress(1.0, text='Análise concluída!')
                if preview is not None:
                    preview.empty()
                st.rerun()
            except Exception as exc:
                barra.empty()
                if preview is not None:
                    preview.empty()
                st.error(f'Falha na análise do vídeo: {exc}')
                st.caption('Verifique se o vídeo abre normalmente, se o YOLO conseguiu baixar seus pesos e se há espaço livre no computador.')
                return
        if valido:
            r = gravado
            if mostrar:
                st.video(r['video'], format='video/mp4')
                st.caption('Vídeo processado: caixas coloridas e linha virtual de contagem.')
                if not r['h264']:
                    st.warning('O vídeo processado foi salvo em MP4V e talvez não reproduza neste navegador. ' + r['aviso_codec'])
            else:
                st.caption('Vídeo oculto. Os resultados abaixo continuam disponíveis.')
            st.download_button('Baixar vídeo com detecções (.mp4)', data=r['video'],
                               file_name='FluxoIA_Supervisao_Analisada.mp4', mime='video/mp4')
            st.markdown('**Veículos identificados no vídeo**')
            cols = st.columns(4)
            for coluna, categoria in zip(cols, CLASSES.values()):
                coluna.metric(categoria, str(r['identificados'].get(categoria,0)))
            st.markdown('**Veículos que cruzaram a linha virtual**')
            cols2 = st.columns(4)
            for coluna, categoria in zip(cols2, CLASSES.values()):
                coluna.metric(categoria, str(r['cruzamentos'].get(categoria,0)))
            if sum(r['cruzamentos'].values()) == 0:
                st.caption('Nenhum cruzamento confirmado. Experimente mudar a orientação ou posição da linha.')
            st.markdown('**Verificação de possíveis incidentes**')
            a,b = st.columns(2)
            a.metric('Alertas de parada prolongada', str(len(r['alertas'])))
            b.metric('Acidente confirmado', 'Não avaliado')
            if r['alertas']:
                with st.expander('Consultar alertas de parada',expanded=False):
                    st.dataframe([{'Segundo do vídeo':item['tempo'],
                                   'Tipo':item['tipo'], 'ID':item['id'], 'Sinal':item['motivo']}
                                  for item in r['alertas']], hide_index=True, use_container_width=True)
            else:
                st.success('Nenhuma parada prolongada sinalizada pelos critérios desta demonstração.')
                st.caption('Isso não significa ausência de acidentes; o módulo não detecta colisões.')
    else:
        st.caption('Envie um MP4 para iniciar. O resto do FluxoIA funciona normalmente sem este recurso.')

