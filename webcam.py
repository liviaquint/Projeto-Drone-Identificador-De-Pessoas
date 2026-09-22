from ultralytics import YOLO
import cv2


# ==========================================================
# 1. CARREGAR O MODELO
# ==========================================================

# yolo11s = bom equilíbrio entre velocidade e precisão
model = YOLO("yolo11s.pt")


# ==========================================================
# 2. ABRIR O VÍDEO
# ==========================================================

video = cv2.VideoCapture("pessoas sesi 2.mp4")

if not video.isOpened():
    print("Erro: não foi possível abrir o vídeo.")
    exit()


# ==========================================================
# 3. CONFIGURAR JANELA
# ==========================================================

cv2.namedWindow("Detector", cv2.WINDOW_NORMAL)

cv2.setWindowProperty(
    "Detector",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)


# ==========================================================
# 4. CORES
# ==========================================================

BRANCO = (255, 255, 255)
VERDE = (80, 220, 120)
PRETO = (20, 20, 20)


# ==========================================================
# 5. LOOP PRINCIPAL
# ==========================================================

while True:

    sucesso, frame = video.read()

    if not sucesso:
        print("Vídeo finalizado.")
        break


    # ======================================================
    # 6. DETECÇÃO + RASTREAMENTO
    # ======================================================

    resultados = model.track(
        frame,

        # Somente pessoas
        classes=[0],

        # Mais sensível para pessoas parcialmente visíveis
        conf=0.15,

        # Menor = mais rápido
        imgsz=640,

        # Rastreamento rápido
        tracker="bytetrack.yaml",

        # Mantém os IDs
        persist=True,

        # Não mostrar informações no terminal
        verbose=False
    )


    resultado = resultados[0]


    # ======================================================
    # 7. CONTADOR
    # ======================================================

    quantidade_pessoas = 0


    # ======================================================
    # 8. PROCESSAR PESSOAS
    # ======================================================

    if resultado.boxes is not None:

        for caixa in resultado.boxes:

            classe = int(caixa.cls[0])

            # Garantir que seja pessoa
            if classe != 0:
                continue


            # ==================================================
            # COORDENADAS
            # ==================================================

            x1, y1, x2, y2 = map(
                int,
                caixa.xyxy[0]
            )


            # ==================================================
            # CONTAR
            # ==================================================

            quantidade_pessoas += 1


            # ==================================================
            # DESENHAR QUADRADO
            # ==================================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                VERDE,
                2
            )


            # ==================================================
            # TEXTO
            # ==================================================

            texto = "Pessoa Identificada"


            tamanho_texto, _ = cv2.getTextSize(
                texto,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                2
            )

            texto_largura = tamanho_texto[0]
            texto_altura = tamanho_texto[1]


            # ==================================================
            # POSIÇÃO
            # ==================================================

            texto_x = x1

            texto_y = max(
                y1 - 8,
                texto_altura + 8
            )


            # ==================================================
            # FUNDO DO TEXTO
            # ==================================================

            cv2.rectangle(
                frame,

                (
                    texto_x,
                    texto_y - texto_altura - 8
                ),

                (
                    texto_x + texto_largura + 8,
                    texto_y
                ),

                PRETO,
                -1
            )


            # ==================================================
            # ESCREVER "PESSOA"
            # ==================================================

            cv2.putText(
                frame,

                texto,

                (
                    texto_x + 4,
                    texto_y - 4
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.50,

                VERDE,

                2,

                cv2.LINE_AA
            )


    # ======================================================
    # 9. PAINEL
    # ======================================================

    painel_x = 10
    painel_y = 10

    painel_largura = 180
    painel_altura = 70


    # ======================================================
    # FUNDO DO PAINEL
    # ======================================================

    cv2.rectangle(
        frame,

        (
            painel_x,
            painel_y
        ),

        (
            painel_x + painel_largura,
            painel_y + painel_altura
        ),

        PRETO,
        -1
    )


    # ======================================================
    # BORDA
    # ======================================================

    cv2.rectangle(
        frame,

        (
            painel_x,
            painel_y
        ),

        (
            painel_x + painel_largura,
            painel_y + painel_altura
        ),

        VERDE,
        2
    )


    # ======================================================
    # TÍTULO
    # ======================================================

    cv2.putText(
        frame,

        "YOLO DETECTOR",

        (
            painel_x + 12,
            painel_y + 23
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.50,

        BRANCO,

        1,

        cv2.LINE_AA
    )


    # ======================================================
    # LINHA
    # ======================================================

    cv2.line(
        frame,

        (
            painel_x + 12,
            painel_y + 31
        ),

        (
            painel_x + painel_largura - 12,
            painel_y + 31
        ),

        VERDE,

        1
    )


    # ======================================================
    # QUANTIDADE
    # ======================================================

    cv2.putText(
        frame,

        f"Pessoas: {quantidade_pessoas}",

        (
            painel_x + 12,
            painel_y + 57
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        BRANCO,

        1,

        cv2.LINE_AA
    )


    # ======================================================
    # MOSTRAR VÍDEO
    # ======================================================

    cv2.imshow(
        "Detector",
        frame
    )


    # ======================================================
    # ESC PARA SAIR
    # ======================================================

    if cv2.waitKey(1) & 0xFF == 27:
        break


# ==========================================================
# 10. ENCERRAR
# ==========================================================

video.release()
cv2.destroyAllWindows()