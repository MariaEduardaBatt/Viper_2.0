import math
import sys
import cv2
import numpy as np

TRACKBAR_MAX = 300


def empty(_):
    pass


def _screen_size():
    try:
        import ctypes

        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        return 1920, 1080


def _initial_scale(img, screen):
    sw, sh = screen
    iw, ih = img.shape[1], img.shape[0]
    s = min(100.0, 90.0 * sw / iw, 90.0 * sh / ih)
    return int(max(10, min(300, s)))


def build_trackbars(window, has_alt, escala_inicial):
    cv2.createTrackbar("param1", window, 100, TRACKBAR_MAX, empty)
    cv2.createTrackbar("param2", window, 100, TRACKBAR_MAX, empty)
    cv2.createTrackbar("param2ALT", window, 50, 99, empty)
    cv2.createTrackbar("minDist", window, 60, TRACKBAR_MAX, empty)
    cv2.createTrackbar("minRadius", window, 10, TRACKBAR_MAX, empty)
    cv2.createTrackbar("maxRadius", window, 0, TRACKBAR_MAX, empty)
    cv2.createTrackbar("dp", window, 12, 30, empty)
    cv2.createTrackbar("blur", window, 5, 30, empty)
    cv2.createTrackbar("median", window, 0, 30, empty)
    cv2.createTrackbar("metodo", window, 1, 1 if has_alt else 0, empty)
    cv2.createTrackbar("escala", window, escala_inicial, 300, empty)


def get_params(window):
    p1 = cv2.getTrackbarPos("param1", window)
    p2 = cv2.getTrackbarPos("param2", window)
    p2alt = cv2.getTrackbarPos("param2ALT", window) / 100.0
    md = cv2.getTrackbarPos("minDist", window)
    mn = cv2.getTrackbarPos("minRadius", window)
    mx = cv2.getTrackbarPos("maxRadius", window)
    dp = cv2.getTrackbarPos("dp", window) / 10.0
    b = cv2.getTrackbarPos("blur", window)
    m = cv2.getTrackbarPos("median", window)
    metodo = cv2.getTrackbarPos("metodo", window)
    if md < 1:
        md = 1
    if dp < 0.1:
        dp = 0.1
    if b < 1:
        b = 1
    if b % 2 == 0:
        b += 1
    if m > 0 and m % 2 == 0:
        m += 1
    return p1, p2, p2alt, md, mn, mx, dp, b, m, metodo


def _intersection_area(x1, y1, r1, x2, y2, r2):
    d = math.hypot(x1 - x2, y1 - y2)
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    r1sq = r1 * r1
    r2sq = r2 * r2
    a = r1sq * math.acos((d * d + r1sq - r2sq) / (2 * d * r1))
    b = r2sq * math.acos((d * d + r2sq - r1sq) / (2 * d * r2))
    c = 0.5 * math.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    return a + b - c


def dedupe(circles, overlap_ratio=0.5):
    circles = sorted(circles, key=lambda c: -float(c[2]))
    kept = []
    for x, y, r in circles:
        x, y, r = float(x), float(y), float(r)
        duplicate = False
        for kx, ky, kr in kept:
            inter = _intersection_area(x, y, r, kx, ky, kr)
            if inter >= overlap_ratio * math.pi * min(r, kr) ** 2:
                duplicate = True
                break
        if not duplicate:
            kept.append((int(x), int(y), int(r)))
    return kept


def main():
    if len(sys.argv) < 2:
        print("Uso: python detect_circles.py <caminho/da/imagem.jpg>")
        sys.exit(1)

    path = sys.argv[1]
    img = cv2.imread(path)
    if img is None:
        print(f"Nao foi possivel carregar a imagem: {path}")
        sys.exit(1)

    has_alt = hasattr(cv2, "HOUGH_GRADIENT_ALT")
    window = "Detecao de circulos - HoughCircles"
    escala_inicial = _initial_scale(img, _screen_size())
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    build_trackbars(window, has_alt, escala_inicial)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    auto_max_radius = min(img.shape[0], img.shape[1]) // 2
    last_count = None
    last_win_size = None

    print("Ajuste os sliders para calibrar a deteccao.")
    print("ESC ou Q para sair. Use + e - para aumentar/diminuir a escala.")
    if not has_alt:
        print("Metodo ALT indisponivel nesta versao do OpenCV.")

    escala = escala_inicial

    while True:
        tb_escala = cv2.getTrackbarPos("escala", window)
        if tb_escala >= 10 and tb_escala != escala:
            escala = tb_escala

        p1, p2, p2alt, md, mn, mx, dp, b, m, metodo = get_params(window)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (b, b), 0)
        if m > 0:
            gray = cv2.medianBlur(gray, m)

        use_alt = has_alt and metodo == 1
        method = cv2.HOUGH_GRADIENT_ALT if use_alt else cv2.HOUGH_GRADIENT
        if use_alt:
            dp = 1.0
            param2 = p2alt
        else:
            param2 = p2

        max_radius = mx if mx > 0 else auto_max_radius
        circles = cv2.HoughCircles(
            gray,
            method,
            dp=dp,
            minDist=md,
            param1=p1,
            param2=param2,
            minRadius=mn if mn > 0 else 1,
            maxRadius=max_radius,
        )

        output = img.copy()
        n = 0
        if circles is not None:
            detected = dedupe(circles[0])
            for x, y, r in detected:
                cv2.circle(output, (x, y), r, (0, 255, 0), 2)
                cv2.circle(output, (x, y), 2, (0, 0, 255), 3)
            n = len(detected)
        if n != last_count:
            print(f"{n} circulo(s) detectado(s)")
            last_count = n

        modo = "ALT" if use_alt else "GRADIENT"
        cv2.putText(
            output,
            f"{n} circulos - {modo}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        s = escala / 100.0
        win_w = max(1, int(round(img.shape[1] * s)))
        win_h = max(1, int(round(img.shape[0] * s)))
        if (win_w, win_h) != last_win_size:
            cv2.resizeWindow(window, win_w, win_h)
            last_win_size = (win_w, win_h)
        cv2.imshow(window, output)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("+"), ord("=")):
            escala = min(300, escala + 25)
            cv2.setTrackbarPos("escala", window, escala)
        elif key in (ord("-"), ord("_")):
            escala = max(10, escala - 25)
            cv2.setTrackbarPos("escala", window, escala)
        elif key in (27, ord("q"), ord("Q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
