import cv2
import numpy as np


def detectar_circulos_webcam(indice_camera=0):
    captura = cv2.VideoCapture(indice_camera)

    if not captura.isOpened():
        raise RuntimeError("Não foi possível acessar a câmera. Verifique se ela está conectada e não está sendo usada por outro programa.")

    print("Câmera iniciada. Pressione 'q' para sair ou 's' para salvar um print.")

    contPrints = 0 

    while True:
        ret, frame = captura.read()
        if not ret:
            print("Não foi possível ler o quadro da câmera.")
            break

        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cinza_suavizada = cv2.medianBlur(cinza, 5)

        circulos = cv2.HoughCircles(
            cinza_suavizada,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=100,
            param2=30,
            minRadius=5,
            maxRadius=0
        )

        totalCirculos = 0

        if circulos is not None:
            circulos = np.uint16(np.around(circulos))
            totalCirculos = circulos.shape[1]

            for (x, y, raio) in circulos[0, :]:
                cv2.circle(frame, (x, y), raio, (0, 255, 0), 2)   
                cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)      

        cv2.putText(
            frame,
            f"Circulos: {totalCirculos}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        cv2.imshow("Deteccao de Circulos - Camera ao Vivo", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):
            break
        elif tecla == ord('s'):
            contPrints += 1
            nome_arquivo = f"print_circulos_{contPrints}.png"
            cv2.imwrite(nome_arquivo, frame)
            print(f"Print salvo como: {nome_arquivo}")

    captura.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detectar_circulos_webcam()