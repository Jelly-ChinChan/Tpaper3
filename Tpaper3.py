import streamlit as st
from PIL import Image
import numpy as np
from collections import deque


# ---------- Otsu 閾值 ----------
def otsu_threshold(arr: np.ndarray) -> float:
    values = arr.ravel().astype(np.float64)

    if values.size == 0:
        return 0.0

    v_min = values.min()
    v_max = values.max()

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

    threshold_original = v_min + (threshold_scaled / 255) * (v_max - v_min)

    return float(threshold_original)


# ---------- 自動偵測深藍色試紙外圈 ----------
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
        cx = w // 2
        cy = h // 2
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


# ---------- 保留中心連通的白色褪色區 ----------
def keep_center_connected_region(candidate_mask, cx, cy, search_radius=60):
    """
    從中心附近尋找 candidate_mask=True 的起點，
    然後只保留與該起點連通的區域。
    這可以避免外圈邊緣或雜訊被判斷成褪色區。
    """
    h, w = candidate_mask.shape

    cx = int(np.clip(cx, 0, w - 1))
    cy = int(np.clip(cy, 0, h - 1))

    # 先找中心附近是否有白色候選點
    yy, xx = np.ogrid[:h, :w]
    center_search_mask = (
        ((xx - cx) ** 2 + (yy - cy) ** 2 <= search_radius ** 2)
        & candidate_mask
    )

    ys, xs = np.where(center_search_mask)

    if len(xs) == 0:
        # 如果中心附近沒有找到，就回傳空遮罩
        return np.zeros_like(candidate_mask, dtype=bool)

    # 選擇距離中心最近的候選點作為起點
    distances = (xs - cx) ** 2 + (ys - cy) ** 2
    idx = int(np.argmin(distances))
    start_x = int(xs[idx])
    start_y = int(ys[idx])

    visited = np.zeros_like(candidate_mask, dtype=bool)
    connected = np.zeros_like(candidate_mask, dtype=bool)

    q = deque()
    q.append((start_y, start_x))
    visited[start_y, start_x] = True
    connected[start_y, start_x] = True

    # 8-connected flood fill
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    while q:
        y, x = q.popleft()

        for dy, dx in directions:
            ny = y + dy
            nx = x + dx

            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue

            if visited[ny, nx]:
                continue

            visited[ny, nx] = True

            if candidate_mask[ny, nx]:
                connected[ny, nx] = True
                q.append((ny, nx))

    return connected


# ---------- 分析白色褪色區 / 剩餘藍色區 ----------
def analyze_faded_white_area(
    img_rgb: Image.Image,
    circle_mask: np.ndarray,
    cx: int,
    cy: int,
    manual_threshold=None,
    white_bias=0.0,
    center_search_radius=80,
    min_candidate_ratio=0.02
):
    arr = np.array(img_rgb).astype(np.float32)

    R = arr[:, :, 0]
    G = arr[:, :, 1]
    B = arr[:, :, 2]

    # 亮度
    gray = 0.299 * R + 0.587 * G + 0.114 * B

    # 藍色指數
    blue_index = B - ((R + G) / 2)

    # 白色分數：
    # 白色區：亮度高、藍色指數低
    # 藍色區：亮度低、藍色指數高
    white_score = gray - blue_index + white_bias

    valid_white_score = white_score[circle_mask]

    if manual_threshold is None:
        threshold = otsu_threshold(valid_white_score)
    else:
        threshold = manual_threshold

    # 所有可能的白色區
    candidate_faded = (white_score > threshold) & circle_mask

    # 若候選白色區太少，避免誤判
    candidate_ratio = candidate_faded.sum() / circle_mask.sum()

    if candidate_ratio < min_candidate_ratio:
        faded_mask = np.zeros_like(circle_mask, dtype=bool)
    else:
        # 只保留「與中心相連」的那一片白色區
        faded_mask = keep_center_connected_region(
            candidate_faded,
            cx=cx,
            cy=cy,
            search_radius=center_search_radius
        )

    # 剩餘區域都視為藍色區
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


# ---------- 製作分析疊圖 ----------
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

    # 圓形外部：變暗，表示桌面不計算
    outside = ~circle_mask
    overlay[outside] = overlay[outside] * 0.30

    # 若要看候選褪色區，先用紫色淡淡標示
    if show_candidate and candidate_faded is not None:
        overlay[candidate_faded] = (
            (1 - alpha_candidate) * overlay[candidate_faded]
            + alpha_candidate * np.array([180, 0, 255])
        )

    # 剩餘藍色區：黃色標示
    overlay[blue_mask] = (
        (1 - alpha_blue) * overlay[blue_mask]
        + alpha_blue * np.array([255, 230, 0])
    )

    # 中心連通白色褪色區：紅色標示
    overlay[faded_mask] = (
        (1 - alpha_faded) * overlay[faded_mask]
        + alpha_faded * np.array([255, 0, 0])
    )

    # 自動偵測到的深藍外圈：亮黃色加強
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
    "並只保留與中心相連的不規則白色褪色區。"
)

st.info(
    "判斷原則：試劑從中心滴入，因此褪色區應該是從中心向外擴散的連通區。"
    "外圈邊緣若被白色分數誤判，因為沒有和中心相連，會被排除。"
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
        blue_b_min = st.slider(
            "偵測外圈：B 通道最低值",
            0, 255, 70
        )

        blue_index_min = st.slider(
            "偵測外圈：Blue Index 最低值",
            -50, 150, 15
        )

        expand_ratio = st.slider(
            "圓形遮罩放大比例",
            0.90, 1.15, 1.03, 0.01
        )

    circle_mask, blue_edge_mask, cx, cy, r = auto_find_paper_by_blue_edge(
        img,
        blue_b_min=blue_b_min,
        blue_index_min=blue_index_min,
        expand_ratio=expand_ratio
    )

    st.info(f"自動偵測結果：圓心 = ({cx}, {cy})，半徑 = {r} px")

    st.subheader("3) 中心連通褪色區判斷")

    with st.expander("進階設定：中心連通區判斷", expanded=False):
        center_search_radius = st.slider(
            "中心搜尋半徑",
            10,
            200,
            80,
            5,
            help="程式會在圓心附近尋找白色褪色區起點。若中心白色區沒有被抓到，可以調大。"
        )

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
            0.005,
            help="候選白色區太少時視為無明顯褪色，避免雜訊。"
        )

        show_candidate = st.checkbox(
            "顯示所有候選褪色區",
            value=False,
            help="紫色會顯示原本可能被判定為褪色的區域；最後紅色只保留與中心相連的區域。"
        )

    auto_result = analyze_faded_white_area(
        img,
        circle_mask,
        cx=cx,
        cy=cy,
        manual_threshold=None,
        white_bias=white_bias,
        center_search_radius=center_search_radius,
        min_candidate_ratio=min_candidate_ratio
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
            center_search_radius=center_search_radius,
            min_candidate_ratio=min_candidate_ratio
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
        caption="黃色=剩餘藍色區，紅色=中心連通白色褪色區，紫色=候選褪色區，變暗區=桌面不計算",
        use_container_width=True
    )

    st.success(
        f"✅ 剩餘藍色面積比例：**{result['blue_ratio']:.2%}**　｜　"
        f"中心白色褪色面積比例：**{result['faded_ratio']:.2%}**"
    )

    st.subheader("5) 數值摘要")

    st.write(
        f"- Blue pixels：{result['blue_count']}\n"
        f"- Center connected faded white pixels：{result['faded_count']}\n"
        f"- Total counted pixels：{result['total']}\n"
        f"- Blue ratio：{result['blue_ratio']:.4f}\n"
        f"- Center connected faded white ratio：{result['faded_ratio']:.4f}\n"
        f"- White Score threshold：{result['threshold']:.2f}\n"
        f"- Candidate faded ratio：{result['candidate_ratio']:.4f}"
    )
