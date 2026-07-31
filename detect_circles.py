import math
import sys
import cv2
import numpy as np

TRACKBAR_MAX = 300
CIRCULARITY_MIN = 0.7
SOLIDITY_MIN = 0.6
SWEEP_POINTS = 4
SWEEP_MIN_RATIO = 0.4
NESTED_RADIUS_RATIO = 0.75
NESTED_CENTER_RATIO = 0.25


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
    cv2.createTrackbar("minDist", window, 20, TRACKBAR_MAX, empty)
    cv2.createTrackbar("minRadius", window, 1, TRACKBAR_MAX, empty)
    cv2.createTrackbar("maxRadius", window, 0, TRACKBAR_MAX, empty)
    cv2.createTrackbar("dp", window, 12, 30, empty)
    cv2.createTrackbar("blur", window, 5, 30, empty)
    cv2.createTrackbar("median", window, 0, 30, empty)
    cv2.createTrackbar("metodo", window, 1, 2, empty)
    cv2.createTrackbar("boost", window, 1, 1, empty)
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
    boost = cv2.getTrackbarPos("boost", window)
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
    return p1, p2, p2alt, md, mn, mx, dp, b, m, metodo, boost


def _param2_sweep(p2):
    low = max(0.1, p2 * SWEEP_MIN_RATIO)
    return [round(p2 - (p2 - low) * i / SWEEP_POINTS, 3) for i in range(SWEEP_POINTS + 1)]


def _detect_by_hough(gray, method, dp, min_dist, param1, param2, min_r, max_r):
    circles = cv2.HoughCircles(
        gray,
        method,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return []
    return circles[0].tolist()


def _hough_multi_pass(gray, method, dp, md, p1, p2, mn, mx):
    found = []
    for acc in _param2_sweep(p2):
        found.extend(_detect_by_hough(gray, method, dp, md, p1, acc, mn, mx))
    md2 = max(1, md // 2)
    if md2 != md:
        for acc in _param2_sweep(p2):
            found.extend(_detect_by_hough(gray, method, dp, md2, p1, acc, mn, mx))
    return found


def _detect_by_contours(gray, min_r, max_r):
    results = []
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for bin_img in (th, cv2.bitwise_not(th)):
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            perim = cv2.arcLength(c, True)
            if perim <= 0:
                continue
            circularity = 4.0 * math.pi * area / (perim * perim)
            if circularity < CIRCULARITY_MIN:
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            if r < min_r or r > max_r:
                continue
            if math.pi * r * r > 0 and area / (math.pi * r * r) < SOLIDITY_MIN:
                continue
            results.append((x, y, r))
    return results


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
            d = math.hypot(x - kx, y - ky)
            big = max(r, kr)
            small = min(r, kr)
            if big > 0 and d <= NESTED_CENTER_RATIO * big and small / big < NESTED_RADIUS_RATIO:
                continue
            inter = _intersection_area(x, y, r, kx, ky, kr)
            if inter >= overlap_ratio * math.pi * small ** 2:
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
    last_params = None
    gray_cache = None
    detected = []
    modo_texto = ""

    print("Ajuste os sliders para calibrar a deteccao.")
    print("metodo: 0=GRADIENT, 1=ALT, 2=HIBRIDO. boost=multi-passe.")
    print("ESC ou Q para sair. Use + e - para aumentar/diminuir a escala.")
    if not has_alt:
        print("Metodo ALT indisponivel nesta versao do OpenCV.")

    escala = escala_inicial

    while True:
        tb_escala = cv2.getTrackbarPos("escala", window)
        if tb_escala >= 10 and tb_escala != escala:
            escala = tb_escala

        params = get_params(window)
        p1, p2, p2alt, md, mn, mx, dp, b, m, metodo, boost = params

        if params != last_params or gray_cache is None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = clahe.apply(gray)
            gray = cv2.GaussianBlur(gray, (b, b), 0)
            if m > 0:
                gray = cv2.medianBlur(gray, m)
            gray_cache = gray

            use_alt = has_alt and metodo == 1
            method = cv2.HOUGH_GRADIENT_ALT if use_alt else cv2.HOUGH_GRADIENT
            if use_alt:
                dp_eff = 1.0
                param2 = p2alt
            else:
                dp_eff = dp
                param2 = p2

            max_radius = mx if mx > 0 else auto_max_radius
            min_radius = mn if mn > 0 else 1

            candidates = []
            if metodo == 2:
                candidates.extend(_detect_by_contours(gray_cache, min_radius, max_radius))
                candidates.extend(
                    _detect_by_hough(gray_cache, method, dp_eff, md, p1, param2, min_radius, max_radius)
                )
                modo_texto = "HIBRIDO"
            else:
                if boost:
                    candidates.extend(
                        _hough_multi_pass(gray_cache, method, dp_eff, md, p1, param2, min_radius, max_radius)
                    )
                    modo_texto = ("ALT" if use_alt else "GRADIENT") + "+BOOST"
                else:
                    candidates.extend(
                        _detect_by_hough(gray_cache, method, dp_eff, md, p1, param2, min_radius, max_radius)
                    )
                    modo_texto = "ALT" if use_alt else "GRADIENT"

            detected = dedupe(candidates)
            last_params = params

        output = img.copy()
        n = len(detected)
        if n != last_count:
            print(f"{n} circulo(s) detectado(s)")
            last_count = n

        cv2.putText(
            output,
            f"{n} circulos - {modo_texto}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        for x, y, r in detected:
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.circle(output, (x, y), 2, (0, 0, 255), 3)

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
