import os
import time
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np

@dataclass
class Furo:
    id: int
    x: int
    y: int
    raio: int

def abrir_camera(indice_ou_url, largura: int = 1280, altura: int = 720) -> cv2.VideoCapture:
    captura = cv2.VideoCapture(indice_ou_url)

    captura.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
    captura.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)

    if not captura.isOpened():
        raise RuntimeError(
            f"Não foi possível abrir a câmera '{indice_ou_url}'. "
            "Verifique se o dispositivo está conectado, se o índice/URL está "
            "correto e se nenhum outro processo está utilizando a câmera."
        )
    return captura

def capturar_frame(captura: cv2.VideoCapture) -> np.ndarray:
    ok, frame = captura.read()
    if not ok or frame is None:
        raise RuntimeError("Falha ao capturar frame da câmera (stream interrompido).")
    return frame

def capturar_frame_estabilizado(
    captura: cv2.VideoCapture, n_frames: int = 5
) -> np.ndarray:
    acumulador = None
    for _ in range(n_frames):
        frame = capturar_frame(captura)
        frame_float = frame.astype(np.float32)
        acumulador = frame_float if acumulador is None else acumulador + frame_float
    media = (acumulador / n_frames).astype(np.uint8)
    return media

def pre_processar_imagem(
    imagem: np.ndarray,
    blur_kernel: Tuple[int, int] = (9, 9),
) -> dict:
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    suavizada = cv2.GaussianBlur(cinza, blur_kernel, 2)

    _, binaria = cv2.threshold(
        suavizada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel, iterations=2)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations=1)

    bordas = cv2.Canny(suavizada, 50, 150)

    return {"cinza": cinza, "suavizada": suavizada, "binaria": binaria, "bordas": bordas}

def detectar_furos_hough(
    imagem_suavizada: np.ndarray,
    dp: float = 1.2,
    min_dist: int = 20,
    param1: int = 60,
    param2: int = 25,
    raio_min: int = 5,
    raio_max: int = 40,
) -> List[Tuple[int, int, int]]:
    circulos = cv2.HoughCircles(
        imagem_suavizada,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=raio_min,
        maxRadius=raio_max,
    )

    resultado = []
    if circulos is not None:
        circulos = np.round(circulos[0, :]).astype(int)
        for (x, y, r) in circulos:
            resultado.append((int(x), int(y), int(r)))
    return resultado

def detectar_furos_contornos(
    imagem_binaria: np.ndarray,
    area_min: float = 40.0,
    area_max: float = 5000.0,
    circularidade_min: float = 0.7,
) -> List[Tuple[int, int, int]]:
    contornos, _ = cv2.findContours(
        imagem_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    resultado = []
    for c in contornos:
        area = cv2.contourArea(c)
        if area < area_min or area > area_max:
            continue

        perimetro = cv2.arcLength(c, True)
        if perimetro == 0:
            continue

        circularidade = 4 * np.pi * (area / (perimetro ** 2))
        if circularidade < circularidade_min:
            continue

        (x, y), raio = cv2.minEnclosingCircle(c)
        resultado.append((int(round(x)), int(round(y)), int(round(raio))))

    return resultado

def ordenar_e_enumerar_furos(
    furos_brutos: List[Tuple[int, int, int]],
    tolerancia_linha: Optional[int] = None,
) -> List[Furo]:
    if not furos_brutos:
        return []

    if tolerancia_linha is None:
        raio_medio = np.mean([r for _, _, r in furos_brutos])
        tolerancia_linha = max(int(raio_medio * 1.5), 10)

    furos_ordenados_y = sorted(furos_brutos, key=lambda f: f[1])

    linhas: List[List[Tuple[int, int, int]]] = []
    for furo in furos_ordenados_y:
        _, y, _ = furo
        colocado = False
        for linha in linhas:
            y_medio_linha = np.mean([f[1] for f in linha])
            if abs(y - y_medio_linha) <= tolerancia_linha:
                linha.append(furo)
                colocado = True
                break
        if not colocado:
            linhas.append([furo])

    linhas.sort(key=lambda linha: np.mean([f[1] for f in linha]))
    for linha in linhas:
        linha.sort(key=lambda f: f[0])

    furos_enumerados: List[Furo] = []
    id_atual = 1
    for linha in linhas:
        for (x, y, r) in linha:
            furos_enumerados.append(Furo(id=id_atual, x=x, y=y, raio=r))
            id_atual += 1

    return furos_enumerados

def gerar_arquivo_sql(
    furos: List[Furo],
    caminho_saida: str = "furosEvaporador.sql",
    nome_tabela: str = "furosEvaporador",
) -> str:
    linhas_sql = []
    linhas_sql.append("-- Script gerado automaticamente a partir de captura ao vivo da câmera")
    linhas_sql.append(f"-- Total de furos detectados: {len(furos)}")
    linhas_sql.append("")
    linhas_sql.append(f"DROP TABLE IF EXISTS {nome_tabela};")
    linhas_sql.append("")
    linhas_sql.append(f"CREATE TABLE {nome_tabela} (")
    linhas_sql.append("    id INTEGER PRIMARY KEY,")
    linhas_sql.append("    posicao_x INTEGER NOT NULL,")
    linhas_sql.append("    posicao_y INTEGER NOT NULL,")
    linhas_sql.append("    status_limpeza INTEGER NOT NULL DEFAULT 0")
    linhas_sql.append(");")
    linhas_sql.append("")

    for furo in furos:
        linhas_sql.append(
            f"INSERT INTO {nome_tabela} (id, posicao_x, posicao_y, status_limpeza) "
            f"VALUES ({furo.id}, {furo.x}, {furo.y}, 0);"
        )

    conteudo = "\n".join(linhas_sql) + "\n"

    with open(caminho_saida, "w", encoding="utf-8") as arquivo:
        arquivo.write(conteudo)

    return caminho_saida

def desenhar_furos_anotados(frame: np.ndarray, furos: List[Furo]) -> np.ndarray:
    saida = frame.copy()

    for furo in furos:
        cv2.circle(saida, (furo.x, furo.y), furo.raio, (0, 255, 0), 2)
        cv2.circle(saida, (furo.x, furo.y), 3, (0, 0, 255), -1)
        cv2.putText(
            saida,
            str(furo.id),
            (furo.x + furo.raio + 3, furo.y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    for i in range(len(furos) - 1):
        p1 = (furos[i].x, furos[i].y)
        p2 = (furos[i + 1].x, furos[i + 1].y)
        cv2.line(saida, p1, p2, (0, 200, 255), 1, cv2.LINE_AA)

    cv2.putText(
        saida,
        f"Furos detectados: {len(furos)}  |  [s] salvar SQL  [q] sair",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return saida

def processar_frame(
    frame: np.ndarray, metodo: str = "hough"
) -> Tuple[List[Furo], np.ndarray]:
    etapas = pre_processar_imagem(frame)

    if metodo == "hough":
        furos_brutos = detectar_furos_hough(etapas["suavizada"])
        if not furos_brutos:
            furos_brutos = detectar_furos_contornos(etapas["binaria"])
    elif metodo == "contornos":
        furos_brutos = detectar_furos_contornos(etapas["binaria"])
    else:
        raise ValueError("metodo deve ser 'hough' ou 'contornos'")

    furos = ordenar_e_enumerar_furos(furos_brutos)
    frame_anotado = desenhar_furos_anotados(frame, furos)

    return furos, frame_anotado

def executar_deteccao_ao_vivo(
    fonte_camera=0,
    pasta_saida: str = "saida",
    metodo: str = "hough",
    n_frames_estabilizacao: int = 5,
):
    os.makedirs(pasta_saida, exist_ok=True)
    captura = abrir_camera(fonte_camera)

    print("Captura ao vivo iniciada. Janela de vídeo em foco:")
    print("  [s] = capturar frame estabilizado e exportar SQL")
    print("  [q] ou [ESC] = encerrar")

    try:
        while True:
            frame = capturar_frame(captura)
            furos, frame_anotado = processar_frame(frame, metodo=metodo)

            cv2.imshow("Deteccao de Furos - Evaporador (ao vivo)", frame_anotado)

            tecla = cv2.waitKey(1) & 0xFF

            if tecla in (ord("q"), 27):  
                break

            elif tecla == ord("s"):
                frame_estavel = capturar_frame_estabilizado(
                    captura, n_frames=n_frames_estabilizacao
                )
                furos_finais, frame_final_anotado = processar_frame(
                    frame_estavel, metodo=metodo
                )

                if not furos_finais:
                    print("Nenhum furo detectado no frame estabilizado. "
                          "Ajuste iluminação/posição e tente novamente.")
                    continue

                caminho_sql = gerar_arquivo_sql(
                    furos_finais,
                    caminho_saida=os.path.join(pasta_saida, "furosEvaporador.sql"),
                )
                caminho_imagem = os.path.join(pasta_saida, "furos_detectados.png")
                cv2.imwrite(caminho_imagem, frame_final_anotado)

                print(f"{len(furos_finais)} furo(s) capturado(s) e exportado(s).")
                print(f"  Imagem de validação: {caminho_imagem}")
                print(f"  Arquivo SQL:         {caminho_sql}")

    finally:
        captura.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(
        description="Detecção e mapeamento EM TEMPO REAL de furos de evaporador "
        "via câmera, para orientar um braço robótico de limpeza."
    )
    parser.add_argument(
        "--camera",
        default="0",
        help="Índice da câmera local (ex.: 0, 1) ou URL de câmera IP/RTSP",
    )
    parser.add_argument(
        "--saida", default="saida", help="Pasta onde salvar os resultados exportados"
    )
    parser.add_argument(
        "--metodo",
        default="hough",
        choices=["hough", "contornos"],
        help="Algoritmo de detecção a ser usado",
    )
    parser.add_argument(
        "--frames-estabilizacao",
        type=int,
        default=5,
        help="Número de frames usados na média ao pressionar 's' (reduz ruído)",
    )
    args = parser.parse_args()

    fonte = int(args.camera) if args.camera.isdigit() else args.camera

    executar_deteccao_ao_vivo(
        fonte_camera=fonte,
        pasta_saida=args.saida,
        metodo=args.metodo,
        n_frames_estabilizacao=args.frames_estabilizacao,
    )

if __name__ == "__main__":
    main()
