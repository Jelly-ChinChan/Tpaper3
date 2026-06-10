import streamlit as st
from PIL import Image
import numpy as np
from collections import deque


def otsu_threshold(arr: np.ndarray) -> float:
    values = arr.ravel().astype(np.float64)
    if values.size == 0:
        return 0.0

    v_min, v_max = values.min(), values.max()
    if v_max == v_min:
        return float(v_min)

    scaled = ((values - v_min) / (v_max - v_min) * 255).astype(np.uint8)

    hist = np.bincount(scaled, minlength=256).astype(np.float64)
    total = scaled.size
    sum_total = np.dot(np.arange(256), hist)

    sumB = 0.0
    wB = 0.0
    maximum = -1.0
    threshold_scaled = 128

    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue

        wF = total - wB
        if wF == 0:
            break

        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF

        between = wB * wF * (mB - mF) ** 2
        if between > maximum:
            maximum = between
            threshold_scaled = t

    return float(v_min + (threshold_scaled / 255) * (v_max - v_min))


def auto_find_paper_by_blue_edge(
    img_rgb: Image.Image,
    blue_b_min=70,
    blue_index_min=15,
    expand_ratio=1.03
):
    arr = np.array(img_rgb).astype(np.float32)
    h, w, _ = arr.shape

    R = arr[:, :, 0]
    G = arr[:, :, 1]
    B = arr[:, :, 2]

    blue_index = B - ((R + G) / 2)

    blue_edge_mask = (
        (B > blue_b_min) &
        (blue_index > blue_index_min)
    )

    ys, xs = np.where(blue_edge_mask)

    if len(xs) < 100:
        cx, cy = w // 2, h // 2
        r = min(w, h) // 2
    else:
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        cx = int((x0 + x1) / 2)
        cy = int((y0 + y1) / 2)

        diameter = max(x1 - x0, y1 - y0)
        r = int(diameter / 2 * expand_ratio)

    r = min(r, cx, cy, w - cx - 1, h - cy - 1)

    yy, xx = np.ogrid[:h, :w]
    circle_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2

    return circle_mask, blue_edge_mask, cx, cy, r


def largest_candidate_component(candidate_mask, cx, cy):
    h, w = candidate_mask.shape
    visited = np.zeros_like(candidate_mask, dtype=bool)

    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    best_component = np.zeros_like(candidate_mask, dtype=bool)
    best_score = -1

    ys_all, xs_all = np.where(candidate_mask)

    for sy, sx in zip(ys_all, xs_all):
        if visited[sy, sx]:
            continue

        q = deque()
        q.append((sy, sx))
        visited[sy, sx] = True

        pixels = []

        while q:
            y, x = q.popleft()
            pixels.append((y, x))

            for dy, dx in directions:
                ny, nx = y + dy, x + dx

                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue

                if visited[ny, nx]:
                    continue

                if candidate_mask[ny, nx]:
                    visited[ny, nx] = True
                    q.append((ny, nx))

        if len(pixels) == 0:
            continue

        py = np.array([p[0] for p in pixels])
        px = np.array([p[1] for p in pixels])

        dist_to_center = np.sqrt((px.mean() - cx) ** 2 + (py.mean() - cy) ** 2)

        # 面積越大越好，離中心越近越好
        score = len(pixels) - dist_to_center * 2

        if score > best_score:
            best_score = score
            best_component = np.zeros_like(candidate_mask, dtype=bool)
            best_component[py, px] = True

    return best_component


def radial_fill_from_boundary(boundary_mask, circle_mask, cx, cy, bins=720):
    """
    將中心到紅色邊界之間全部填滿。
    可避免紅色內部出現黃色破洞。
    """
    h, w = boundary_mask.shape

    ys, xs = np.where(boundary_mask)

    if len(xs) == 0:
        return np.zeros_like(boundary_mask, dtype=bool)

    dx = xs - cx
    dy = ys - cy

    angles = np.arctan2(dy, dx)
    angles = (angles + 2 * np.pi) % (2 * np.pi)

    radii = np.sqrt(dx ** 2 + dy ** 2)

    angle_bins = np.floor(angles / (2 * np.pi) * bins).astype(int)
    angle_bins = np.clip(angle_bins, 0, bins - 1)

    max_r = np.zeros(bins, dtype=np.float32)

    for b, r in zip(angle_bins, radii):
        if r > max_r[b]:
            max_r[b] = r

    # 補沒有資料的角度
    valid = max_r > 0
    if valid.sum() < 10:
        return boundary_mask

    for i in range(bins):
        if max_r[i] == 0:
            left = i
            right = i

            while max_r[left % bins] == 0:
                left -= 1

            while max_r[right % bins] == 0:
                right += 1

            max_r[i] = (max_r[left % bins] + max_r[right % bins]) / 2

    # 平滑邊界，避免鋸齒過強
    smooth = max_r.copy()
    window = 9

    for i in range(bins):
        vals = []
        for k in range(-window, window + 1):
            vals.append(max_r[(i + k) % bins])
        smooth[i] = np.mean(vals)

    yy, xx = np.ogrid[:h, :w]

    dx_all = xx - cx
    dy_all = yy - cy

    angle_all = np.arctan2(dy_all, dx_all)
    angle_all = (angle_all + 2 * np.pi) % (2 * np.pi)

    bin_all = np.floor(angle_all / (2 * np.pi) * bins).astype(int)
    bin_all = np.clip(bin_all, 0, bins - 1)

    radius_all = np.sqrt(dx_all ** 2 + dy_all ** 2)

    filled = (radius_all <= smooth[bin_all]) & circle_mask

    return filled


def analyze_faded_white_area(
    img_rgb: Image.Image,
    circle_mask: np.ndarray,
    cx: int,
    cy: int,
    manual_threshold=None,
    white_bias=0.0,
    min_candidate_ratio=0.02,
    radial_fill=True
):
    arr = np.array(img_rgb).astype(np.float32)

    R = arr[:, :, 0]
    G = arr[:, :, 1]
    B = arr[:, :, 2]

    gray = 0.299 * R + 0.587 * G + 0.114 * B
    blue_index = B - ((R + G) / 2)

    white_score = gray - blue_index + white_bias

    valid_white_score = white_score[circle_mask]

    if manual_threshold is None:
        threshold = otsu_threshold(valid_white_score)
    else:
        threshold = manual_threshold

    candidate_faded = (white_score > threshold) & circle_mask
    candidate_ratio = candidate_faded.sum() / circle_mask.sum()

    if candidate_ratio < min_candidate_ratio:
        faded_mask = np.zeros_like(circle_mask, dtype=bool)
    else:
        boundary_component = largest_candidate_component(
            candidate_faded,
            cx=cx,
            cy=cy
        )

        if radial_fill:
            faded_mask = radial_fill_from_boundary(
                boundary_component,
                circle_mask,
                cx=cx,
                cy=cy
            )
        else:
            faded_mask = boundary_component

        faded_mask = faded_mask & circle_mask

    blue_mask = circle_mask & (~faded_mask)

    faded_count = int(faded_mask.sum())
    blue_count = int(blue_mask.sum())
    total = faded_count + blue_count

    faded_ratio = faded_count / total if total else 0
    blue_ratio = blue_count / total if total else 0

    return {
        "gray": gray,
        "blue_index": blue_index,
        "white_score": white_score,
        "threshold": threshold,
        "candidate_faded": candidate_faded,
        "candidate_ratio": candidate_ratio,
        "faded_mask": faded_mask,
        "blue_mask": blue_mask,
        "faded_count": faded_count,
        "blue_count": blue_count,
        "total": total,
        "faded_ratio": faded_ratio,
        "blue_ratio": blue_ratio,
    }


def make_overlay(
    img_rgb,
    circle_mask,
    blue_edge_mask,
    blue_mask,
    faded_mask,
    show_candidate=False,
    candidate_faded=None
):
    arr = np.array(img_rgb).astype(np.float32)
    overlay = arr.copy()

    alpha_blue = 0.40
    alpha_faded = 0.45
    alpha_candidate = 0.25

    outside = ~circle_mask
    overlay[outside] = overlay[outside] * 0.30

    if show_candidate and candidate_faded is not None:
        overlay[candidate_faded] = (
            (1 - alpha_candidate) * overlay[candidate_faded]
            + alpha_candidate * np.array([180, 0, 255])
        )

    overlay[blue_mask] = (
        (1 - alpha_blue) * overlay[blue_mask]
        + alpha_blue * np.array([255, 230, 0])
    )

    overlay[faded_mask] = (
        (1 - alpha_faded) * overlay[faded_mask]
        + alpha_faded * np.array([255, 0, 0])
    )

    edge_show = blue_edge_mask & circle_mask
    overlay[edge_show] = (
        0.35 * overlay[edge_show]
        + 0.65 * np.array([255, 255, 0])
    )

    return overlay.clip(0, 255).astype(np.uint8)


# ================= Streamlit UI =================
st.title("🧪 圓形鉬藍試紙白色褪色面積分析")

st.write(
    "本程式會自動偵測圓形試紙，排除桌面背景，"
    "並將中心到褪色邊界之間填滿，避免紅色區內部出現黃色破洞。"
)

uploaded_file = st.file_uploader(
    "請選擇一張圖片",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    st.subheader("1) 原始照片")
    st.image(img, caption="原始照片", use_container_width=True)

    st.subheader("2) 自動偵測圓形試紙")

    with st.expander("進階設定：圓形試紙偵測", expanded=False):
        blue_b_min = st.slider("偵測外圈：B 通道最低值", 0, 255, 70)

        blue_index_min = st.slider(
            "偵測外圈：Blue Index 最低值",
            -50,
            150,
            15
        )

        expand_ratio = st.slider(
            "圓形遮罩放大比例",
            0.90,
            1.15,
            1.03,
            0.01
        )

    circle_mask, blue_edge_mask, cx, cy, r = auto_find_paper_by_blue_edge(
        img,
        blue_b_min=blue_b_min,
        blue_index_min=blue_index_min,
        expand_ratio=expand_ratio
    )

    st.info(f"自動偵測結果：圓心 = ({cx}, {cy})，半徑 = {r} px")

    st.subheader("3) 褪色區判斷")

    with st.expander("進階設定：褪色區分析", expanded=False):
        white_bias = st.slider(
            "白色判斷偏移值",
            -100.0,
            100.0,
            0.0,
            1.0
        )

        min_candidate_ratio = st.slider(
            "最小候選褪色比例",
            0.0,
            0.10,
            0.02,
            0.005
        )

        radial_fill = st.checkbox(
            "從中心填滿到褪色邊界",
            value=True
        )

        show_candidate = st.checkbox(
            "顯示候選褪色區",
            value=False
        )

    auto_result = analyze_faded_white_area(
        img,
        circle_mask,
        cx=cx,
        cy=cy,
        manual_threshold=None,
        white_bias=white_bias,
        min_candidate_ratio=min_candidate_ratio,
        radial_fill=radial_fill
    )

    with st.expander("進階設定：手動調整 White Score 閾值", expanded=False):
        use_manual = st.checkbox("使用手動閾值", value=False)

        manual_threshold = st.slider(
            "White Score 閾值",
            0.0,
            255.0,
            float(auto_result["threshold"]),
            1.0
        )

    if use_manual:
        result = analyze_faded_white_area(
            img,
            circle_mask,
            cx=cx,
            cy=cy,
            manual_threshold=manual_threshold,
            white_bias=white_bias,
            min_candidate_ratio=min_candidate_ratio,
            radial_fill=radial_fill
        )
    else:
        result = auto_result

    st.write(f"目前 White Score 閾值：**{result['threshold']:.2f}**")
    st.write(f"候選褪色區比例：**{result['candidate_ratio']:.2%}**")

    overlay = make_overlay(
        img,
        circle_mask,
        blue_edge_mask,
        result["blue_mask"],
        result["faded_mask"],
        show_candidate=show_candidate,
        candidate_faded=result["candidate_faded"]
    )

    st.subheader("4) AI 自動選區與分析結果")

    st.image(
        Image.fromarray(overlay),
        caption="黃色=剩餘藍色區，紅色=中心到褪色邊界的白色褪色區，紫色=候選褪色區，變暗區=桌面不計算",
        use_container_width=True
    )

    st.success(
        f"✅ 剩餘藍色面積比例：**{result['blue_ratio']:.2%}**　｜　"
        f"白色褪色面積比例：**{result['faded_ratio']:.2%}**"
    )

    st.subheader("5) 數值摘要")

    st.write(
        f"- Blue pixels：{result['blue_count']}\n"
        f"- Faded white pixels：{result['faded_count']}\n"
        f"- Total counted pixels：{result['total']}\n"
        f"- Blue ratio：{result['blue_ratio']:.4f}\n"
        f"- Faded white ratio：{result['faded_ratio']:.4f}\n"
        f"- White Score threshold：{result['threshold']:.2f}\n"
        f"- Candidate faded ratio：{result['candidate_ratio']:.4f}"
    )
