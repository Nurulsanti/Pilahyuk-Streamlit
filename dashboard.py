import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from urllib.parse import quote

sns.set_theme(style="whitegrid")

DATASET_URL = (
    "https://raw.githubusercontent.com/DBSDicoding2026-UNTIRTA/Wranglingling/main"
    "/clean_sampah_metadata_updated.csv"
)
GITHUB_REPO_RAW_BASE = "https://raw.githubusercontent.com/DBSDicoding2026-UNTIRTA/Wranglingling/main"
LOCAL_DATASET_CANDIDATES = [
    "clean_sampah_metadata_updated.csv",
]
DATE_KEYWORDS = ("date", "tanggal", "time", "waktu")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def slugify_category(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value.lower()).strip("_")


def find_local_dataset_path() -> Path | None:
    for filename in LOCAL_DATASET_CANDIDATES:
        candidate = Path(filename)
        if candidate.exists():
            return candidate
    return None


def get_dataset_sources() -> list[str]:
    sources = [DATASET_URL]
    local_dataset_path = find_local_dataset_path()
    if local_dataset_path is not None:
        sources.append(str(local_dataset_path))
    return sources


@st.cache_data(show_spinner=False)
def load_and_preprocess_dataset(path_str: str) -> tuple[pd.DataFrame, list[str]]:
    raw_df = pd.read_csv(path_str)
    return preprocess_data(raw_df)


def find_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def find_dataset_image_roots() -> list[Path]:
    image_root = Path("images")
    return [image_root] if image_root.exists() and image_root.is_dir() else []


def build_remote_image_url(category: str, file_name: str) -> str:
    safe_category = quote(str(category).strip(), safe="")
    safe_file_name = quote(str(file_name).strip().lstrip("/"), safe="/")
    return f"{GITHUB_REPO_RAW_BASE}/images/{safe_category}/{safe_file_name}"


def find_category_sample_image(category: str, image_roots: list[Path]) -> Path | None:
    if not image_roots:
        return None

    slug = slugify_category(category)
    for image_root in image_roots:
        candidate_dirs = [image_root / category]
        if slug and slug != category:
            candidate_dirs.append(image_root / slug)

        for class_dir in candidate_dirs:
            if not class_dir.exists() or not class_dir.is_dir():
                continue

            for image_path in sorted(class_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    return image_path

    return None


def prioritize_categories(categories: list[str]) -> list[str]:
    preferred = ["Clothes", "Organik"]
    ordered = []
    seen = set()

    for item in preferred:
        if item in categories and item not in seen:
            ordered.append(item)
            seen.add(item)

    for item in categories:
        if item not in seen:
            ordered.append(item)
            seen.add(item)

    return ordered


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    data = df.copy()

    rename_map = {
        "class_label": "kategori",
        "file_name": "nama_file",
        "file_ext": "ekstensi_file",
        "file_size_kb": "ukuran_file_kb",
        "aspect_ratio": "rasio_aspek",
        "pixels": "jumlah_piksel",
        "color_mode": "mode_warna",
        "file_path": "lokasi_file",
        "width": "lebar",
        "height": "tinggi",
    }
    data = data.rename(columns={k: v for k, v in rename_map.items() if k in data.columns})

    # Remove file path column for privacy / display reasons
    if "lokasi_file" in data.columns:
        data = data.drop(columns=["lokasi_file"])

    date_cols: list[str] = []
    for col in data.columns:
        if any(keyword in col.lower() for keyword in DATE_KEYWORDS):
            converted = pd.to_datetime(data[col], errors="coerce")
            if converted.notna().any():
                data[col] = converted
                date_cols.append(col)

    data = data.drop_duplicates()

    for col in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[col]):
            continue

        if pd.api.types.is_numeric_dtype(data[col]):
            if data[col].isna().any():
                data[col] = data[col].fillna(data[col].median())
        else:
            if data[col].isna().any():
                mode_series = data[col].mode(dropna=True)
                fallback = mode_series.iloc[0] if not mode_series.empty else "Tidak diketahui"
                data[col] = data[col].fillna(fallback)

    return data, date_cols


@st.cache_data(show_spinner=False)
def count_images_in_root(image_root: Path) -> int:
    return sum(
        1
        for image_path in image_root.rglob("*")
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
    )


@st.cache_data(show_spinner=False)
def find_category_sample_image_cached(category: str, image_root_paths: tuple[str, ...]) -> str | None:
    image_roots = [Path(root) for root in image_root_paths]
    sample = find_category_sample_image(category, image_roots)
    return str(sample) if sample is not None else None


@st.cache_data(show_spinner=False)
def resolve_category_sample_image_cached(
    category: str,
    sample_file_name: str | None,
    image_root_paths: tuple[str, ...],
) -> str | None:
    local_sample = find_category_sample_image_cached(category, image_root_paths)
    if local_sample is not None:
        return local_sample

    if sample_file_name:
        return build_remote_image_url(category, sample_file_name)

    return None


def compute_iqr_outlier_rate(series: pd.Series) -> float:
    if series.empty:
        return 0.0

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0.0

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return float(((series < lower) | (series > upper)).mean() * 100)


def sidebar_filters(
    df: pd.DataFrame,
    category_col: str | None,
    date_cols: list[str],
) -> tuple[pd.DataFrame, str | None]:
    filtered = df.copy()
    selected_date_col: str | None = None

    st.sidebar.header("Filter Data")

    if category_col:
        category_options = sorted(filtered[category_col].dropna().astype(str).unique().tolist())
        selected_categories = st.sidebar.multiselect(
            "Pilih kategori",
            options=category_options,
            default=category_options,
        )

        if selected_categories:
            filtered = filtered[filtered[category_col].astype(str).isin(selected_categories)]
        else:
            filtered = filtered.iloc[0:0]

    if date_cols:
        selected_date_col = st.sidebar.selectbox("Pilih kolom tanggal", options=date_cols)
        valid_dates = filtered[selected_date_col].dropna()

        if not valid_dates.empty:
            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()
            date_input = st.sidebar.date_input(
                "Pilih rentang tanggal",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )

            if isinstance(date_input, tuple) and len(date_input) == 2:
                start_date, end_date = date_input
            else:
                start_date = date_input
                end_date = date_input

            start_ts = pd.to_datetime(start_date)
            end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            filtered = filtered[filtered[selected_date_col].between(start_ts, end_ts)]

    numeric_cols = filtered.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        selected_numeric_col = st.sidebar.selectbox("Pilih kolom numerik", options=numeric_cols)
        min_val = float(filtered[selected_numeric_col].min())
        max_val = float(filtered[selected_numeric_col].max())

        if min_val < max_val:
            value_range = st.sidebar.slider(
                "Pilih rentang nilai",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val),
            )
            filtered = filtered[filtered[selected_numeric_col].between(value_range[0], value_range[1])]
        else:
            st.sidebar.caption("Kolom numerik terpilih memiliki nilai konstan.")

    return filtered, selected_date_col


def build_insights(
    df: pd.DataFrame,
    category_col: str | None,
    size_col: str | None,
    pixel_col: str | None,
) -> list[str]:
    insights: list[str] = []

    if category_col and not df.empty:
        class_counts = df[category_col].value_counts()
        total = class_counts.sum()
        if total > 0 and not class_counts.empty:
            imbalance_ratio = class_counts.max() / max(class_counts.min(), 1)
            under_15 = class_counts[(class_counts / total * 100) < 15].index.tolist()
            insights.append(
                f"Distribusi kategori menunjukkan imbalance ratio sekitar {imbalance_ratio:.2f}x antara kelas terbesar dan terkecil."
            )
            if under_15:
                insights.append(
                    f"Kategori dengan porsi di bawah 15%: {', '.join(map(str, under_15))}. Kategori ini layak diprioritaskan untuk augmentasi data."
                )

    if category_col and size_col and not df.empty:
        grouped_size = df.groupby(category_col)[size_col].median().sort_values(ascending=False)
        overall_median = df[size_col].median()
        if pd.notna(overall_median) and overall_median != 0:
            deviating = grouped_size[
                ((grouped_size - overall_median).abs() / overall_median * 100) >= 20
            ].index.tolist()
            if deviating:
                insights.append(
                    "Terdapat kategori dengan deviasi median ukuran file >=20% terhadap median keseluruhan: "
                    + ", ".join(map(str, deviating))
                    + "."
                )

    if category_col and size_col and not df.empty:
        outlier_classes: list[str] = []
        for category, series in df.groupby(category_col)[size_col]:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_rate = ((series < lower) | (series > upper)).mean() * 100
            if outlier_rate >= 5:
                outlier_classes.append(f"{category} ({outlier_rate:.1f}%)")

        if outlier_classes:
            insights.append(
                "Outlier ukuran file (metode IQR) yang perlu quality check tambahan: "
                + ", ".join(outlier_classes)
                + "."
            )

    if pixel_col and size_col and not df.empty:
        corr_value = df[[pixel_col, size_col]].corr().iloc[0, 1]
        if pd.notna(corr_value):
            insights.append(
                f"Korelasi antara jumlah piksel dan ukuran file berada di sekitar {corr_value:.2f}, menunjukkan hubungan teknis antar fitur gambar."
            )

    if not insights:
        insights.append("Data hasil filter belum cukup untuk menghasilkan insight yang stabil.")

    return insights


def create_placeholder_image(label: str) -> Image.Image:
    w, h = 640, 420
    image = Image.new("RGB", (w, h), color=(240, 240, 240))
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    draw.text(
        ((w - text_width) / 2, (h - text_height) / 2),
        label,
        fill=(30, 30, 30),
        font=font,
    )
    return image


def build_gallery_image_source(
    row: pd.Series,
    category_col: str,
    file_name_col: str | None,
    search_roots: list[Path],
) -> str | None:
    category = str(row[category_col])
    sample_file_name = None

    if file_name_col and file_name_col in row and pd.notna(row[file_name_col]):
        sample_file_name = str(row[file_name_col])

    if sample_file_name:
        for root in search_roots:
            for category_dir_name in (category, slugify_category(category)):
                if not category_dir_name:
                    continue

                local_candidate = root / category_dir_name / sample_file_name
                if local_candidate.exists() and local_candidate.is_file():
                    return str(local_candidate)

        return build_remote_image_url(category, sample_file_name)

    return resolve_category_sample_image_cached(
        category,
        None,
        tuple(str(root) for root in search_roots),
    )


def render_category_summary_cards(
    df: pd.DataFrame,
    category_col: str,
    palette: list[tuple[str, str, str]],
) -> None:
    icon_map = {
        "clothes": "👕",
        "kaca": "🪟",
        "kardus": "📦",
        "kertas": "📄",
        "logam": "⚙️",
        "organik": "🌿",
        "plastik": "🧴",
        "residu": "🗑️",
    }
    counts = (
        df[category_col]
        .dropna()
        .astype(str)
        .value_counts()
        .reset_index()
    )
    counts.columns = ["kategori", "jumlah"]

    ordered_labels = prioritize_categories(counts["kategori"].tolist())[:8]
    ordered_counts = counts.set_index("kategori").loc[ordered_labels].reset_index()

    for row_offset in range(0, len(ordered_counts), 4):
        row_items = ordered_counts.iloc[row_offset : row_offset + 4]
        cols = st.columns(len(row_items))
        for item_index, (col, (_, row)) in enumerate(zip(cols, row_items.iterrows())):
            accent_color, bg_color, label_color = palette[(row_offset + item_index) % len(palette)]
            icon = icon_map.get(str(row["kategori"]).lower(), "♻️")
            with col:
                st.markdown(
                    f"""
                    <div class="category-card" style="--accent:{accent_color}; --card-bg:{bg_color}; --label:{label_color};">
                        <div class="category-card__icon">{icon}</div>
                        <div class="category-card__label">{row['kategori']}</div>
                        <div class="category-card__value">{int(row['jumlah']):,}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_image_gallery(
    filtered_df: pd.DataFrame,
    category_col: str,
    file_name_col: str | None,
    available_image_roots: list[Path],
) -> None:
    page_size = 12
    total_items = len(filtered_df)
    total_pages = max(1, (total_items + page_size - 1) // page_size)

    if "gallery_page" not in st.session_state:
        st.session_state.gallery_page = 1

    st.session_state.gallery_page = max(1, min(st.session_state.gallery_page, total_pages))

    nav_left, nav_center, nav_right = st.columns([1, 2, 1])
    with nav_left:
        prev_disabled = st.session_state.gallery_page <= 1
        if st.button("◀ Prev", disabled=prev_disabled, use_container_width=True):
            st.session_state.gallery_page -= 1
            st.rerun()
    with nav_center:
        st.markdown(
            f"<div class='gallery-counter'>Halaman {st.session_state.gallery_page} / {total_pages} · {total_items:,} gambar ditemukan</div>",
            unsafe_allow_html=True,
        )
    with nav_right:
        next_disabled = st.session_state.gallery_page >= total_pages
        if st.button("Next ▶", disabled=next_disabled, use_container_width=True):
            st.session_state.gallery_page += 1
            st.rerun()

    start_idx = (st.session_state.gallery_page - 1) * page_size
    end_idx = start_idx + page_size
    page_df = filtered_df.iloc[start_idx:end_idx]

    if page_df.empty:
        st.info("Tidak ada gambar untuk ditampilkan pada halaman ini.")
        return

    columns_per_row = 4
    rows = [page_df.iloc[i : i + columns_per_row] for i in range(0, len(page_df), columns_per_row)]
    gallery_found = 0

    for chunk in rows:
        cols = st.columns(columns_per_row)
        for idx, (_, row) in enumerate(chunk.iterrows()):
            with cols[idx]:
                category = str(row[category_col])
                img_source = build_gallery_image_source(row, category_col, file_name_col, available_image_roots)
                caption = str(row[file_name_col]) if file_name_col and file_name_col in row and pd.notna(row[file_name_col]) else category

                if img_source is not None:
                    try:
                        if img_source.startswith("http"):
                            st.image(img_source, use_container_width=True)
                        else:
                            st.image(Image.open(Path(img_source)), use_container_width=True)
                        gallery_found += 1
                    except Exception:
                        st.image(create_placeholder_image(category), use_container_width=True)
                else:
                    st.image(create_placeholder_image(category), use_container_width=True)

                st.markdown(
                    f"""
                    <div class="gallery-card">
                        <div class="gallery-card__badge">{category}</div>
                        <div class="gallery-card__caption">{caption}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown(
        f"**Insight:** {gallery_found} gambar pada halaman ini berhasil dimuat dari sumber dataset. "
        + (
            "Jika ada kartu placeholder, artinya file gambar yang cocok belum ditemukan di sumber lokal/remote."
            if gallery_found < len(page_df)
            else "Semua kartu pada halaman ini menampilkan gambar asli."
        )
    )


def render_page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-title">{title}</div>
            <p class="hero-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_gallery_page(
    filtered_df: pd.DataFrame,
    category_col: str | None,
    file_name_col: str | None,
    available_image_roots: list[Path],
    total_data_count: int,
) -> None:
    render_page_header(
        "🖼️ Galeri Gambar Sampah",
        "Jelajahi koleksi gambar sampah daur ulang per kategori dengan navigasi halaman yang rapi dan fokus ke visual.",
    )

    if not category_col or filtered_df.empty:
        st.info("Pilih filter kategori agar galeri bisa ditampilkan.")
        return

    category_palette = [
        ("#3d8bfd", "#eff7ff", "#2f6fe0"),
        ("#5aa9ff", "#f4faff", "#3f89eb"),
        ("#7cc4ff", "#f5fbff", "#5ea8eb"),
        ("#98d1ff", "#f8fcff", "#6fb7ee"),
        ("#6e9eff", "#eef5ff", "#4f7fe0"),
        ("#86bfff", "#f2f8ff", "#5f9ae8"),
        ("#4f8ff7", "#edf4ff", "#2f6fe0"),
        ("#b5dcff", "#fbfdff", "#7fb9f1"),
    ]
    render_category_summary_cards(filtered_df, category_col, category_palette)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    render_image_gallery(filtered_df, category_col, file_name_col, available_image_roots)


def render_analytics_page(
    filtered_df: pd.DataFrame,
    category_col: str | None,
    size_col: str | None,
    pixel_col: str | None,
    selected_date_col: str | None,
    total_data_count: int,
) -> None:
    render_page_header(
        "📊 Hasil Analisis",
        "Lihat ringkasan statistik, distribusi fitur, korelasi, dan insight preprocessing dari data yang sudah difilter.",
    )

    with st.container():
        st.subheader("Data")
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.dataframe(filtered_df, use_container_width=True, height=340)

        with col_right:
            st.markdown("### Ringkasan")
            st.metric("Jumlah Data", f"{total_data_count:,}")
            if size_col and not filtered_df.empty:
                st.metric("Rata-rata ukuran file (KB)", f"{filtered_df[size_col].mean():.2f}")
            if category_col and not filtered_df.empty:
                st.metric("Jumlah kategori", filtered_df[category_col].nunique())

    with st.container():
        st.subheader("Ringkasan Statistik")
        numeric_df = filtered_df.select_dtypes(include="number")
        if numeric_df.empty:
            st.info("Tidak ada kolom numerik untuk ditampilkan dalam ringkasan statistik.")
        else:
            st.dataframe(numeric_df.describe().T, use_container_width=True)

    with st.container():
        st.subheader("Visualisasi Data")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            if category_col and not filtered_df.empty:
                class_dist = (
                    filtered_df[category_col]
                    .value_counts()
                    .rename_axis("kategori")
                    .reset_index(name="jumlah_data")
                )
                class_dist["persen"] = class_dist["jumlah_data"] / class_dist["jumlah_data"].sum() * 100
                fig1, ax1 = plt.subplots(figsize=(8, 4.5))
                sns.barplot(data=class_dist, x="kategori", y="jumlah_data", palette="Blues", ax=ax1)
                ax1.set_title("Distribusi Jumlah Data per Kategori")
                ax1.set_xlabel("Kategori")
                ax1.set_ylabel("Jumlah Data")
                ax1.tick_params(axis="x", rotation=25)
                for p, pct in zip(ax1.patches, class_dist["persen"]):
                    height = p.get_height()
                    ax1.annotate(f"{pct:.1f}%", (p.get_x() + p.get_width() / 2, height), ha="center", va="bottom", fontsize=9)
                st.pyplot(fig1)

                if not class_dist.empty:
                    top_row = class_dist.iloc[0]
                    bottom_row = class_dist.iloc[-1]
                    imbalance_ratio = top_row["jumlah_data"] / max(bottom_row["jumlah_data"], 1)
                    low_classes = class_dist.loc[class_dist["persen"] < 15, "kategori"].tolist()
                    st.markdown(
                        f"**Insight:** Kategori dengan data terbanyak adalah **{top_row['kategori']}** dan yang paling sedikit **{bottom_row['kategori']}** dengan rasio sekitar **{imbalance_ratio:.2f}x**. "
                        + (
                            f"Kategori di bawah 15% total data: {', '.join(map(str, low_classes))}."
                            if low_classes
                            else "Tidak ada kategori yang berada di bawah 15% total data."
                        )
                    )
            else:
                st.info("Kolom kategori tidak ditemukan untuk membuat bar chart.")

        with chart_col2:
            fig2, ax2 = plt.subplots(figsize=(8, 4.5))
            line_chart_rendered = False

            if selected_date_col and not filtered_df.empty:
                trend_df = (
                    filtered_df.dropna(subset=[selected_date_col])
                    .set_index(selected_date_col)
                    .resample("D")
                    .size()
                    .rename("jumlah_data")
                    .reset_index()
                )
                if not trend_df.empty:
                    sns.lineplot(data=trend_df, x=selected_date_col, y="jumlah_data", marker="o", ax=ax2)
                    ax2.set_title("Tren Jumlah Data per Hari")
                    ax2.set_xlabel("Tanggal")
                    ax2.set_ylabel("Jumlah Data")
                    line_chart_rendered = True

            if not line_chart_rendered and category_col and size_col and not filtered_df.empty:
                trend_alt = (
                    filtered_df.groupby(category_col)[size_col]
                    .median()
                    .sort_values(ascending=False)
                    .reset_index()
                )
                sns.lineplot(data=trend_alt, x=category_col, y=size_col, marker="o", color="#2f6fe0", ax=ax2)
                ax2.set_title("Perbandingan Median Ukuran File per Kategori")
                ax2.set_xlabel("Kategori")
                ax2.set_ylabel("Median Ukuran File (KB)")
                ax2.tick_params(axis="x", rotation=25)
                line_chart_rendered = True

            if line_chart_rendered:
                st.pyplot(fig2)

                if selected_date_col and not trend_df.empty:
                    peak_row = trend_df.loc[trend_df["jumlah_data"].idxmax()]
                    st.markdown(
                        f"**Insight:** Aktivitas data paling padat terjadi pada **{peak_row[selected_date_col].date()}** dengan **{int(peak_row['jumlah_data'])}** data. "
                        f"Pola ini membantu melihat apakah ada lonjakan pengumpulan data pada periode tertentu."
                    )
                elif category_col and size_col and not filtered_df.empty:
                    highest_row = trend_alt.iloc[0]
                    lowest_row = trend_alt.iloc[-1]
                    st.markdown(
                        f"**Insight:** Median ukuran file tertinggi ada pada **{highest_row[category_col]}** dan terendah pada **{lowest_row[category_col]}**. "
                        "Ini menandakan karakteristik file antar kategori tidak seragam dan layak diperlakukan berbeda saat preprocessing."
                    )
            else:
                st.info("Line chart belum dapat ditampilkan karena kolom pendukung tidak tersedia.")

        numeric_for_corr = filtered_df.select_dtypes(include="number")
        if numeric_for_corr.shape[1] >= 2:
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            corr_matrix = numeric_for_corr.corr(numeric_only=True)
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Blues", ax=ax3)
            ax3.set_title("Heatmap Korelasi Fitur Numerik")
            st.pyplot(fig3)

            corr_no_diag = corr_matrix.abs().where(~pd.DataFrame(
                np.eye(corr_matrix.shape[0], dtype=bool),
                index=corr_matrix.index,
                columns=corr_matrix.columns,
            ))
            if corr_no_diag.notna().any().any():
                strongest_pair = corr_no_diag.stack().idxmax()
                strongest_value = corr_matrix.loc[strongest_pair[0], strongest_pair[1]]
                st.markdown(
                    f"**Insight:** Korelasi paling kuat terlihat antara **{strongest_pair[0]}** dan **{strongest_pair[1]}** dengan nilai **{strongest_value:.2f}**. "
                    "Pola ini penting untuk melihat fitur mana yang saling berkaitan dan berpotensi redundant."
                )
        else:
            st.info("Heatmap korelasi membutuhkan minimal dua kolom numerik.")

    with st.container():
        st.subheader("Distribusi Ukuran File")
        size_col1, size_col2 = st.columns(2)

        with size_col1:
            if size_col and not filtered_df.empty:
                fig_size_hist, ax_size_hist = plt.subplots(figsize=(8, 4.5))
                sns.histplot(filtered_df[size_col], bins=40, kde=True, color="#4f8ff7", ax=ax_size_hist)
                ax_size_hist.set_title("Histogram Ukuran File (KB)")
                ax_size_hist.set_xlabel("Ukuran File (KB)")
                ax_size_hist.set_ylabel("Jumlah Gambar")
                st.pyplot(fig_size_hist)

                skewness = float(filtered_df[size_col].skew())
                tail_note = "condong ke kanan" if skewness > 0.5 else "cukup seimbang"
                st.markdown(
                    f"**Insight:** Distribusi ukuran file {tail_note} (skewness {skewness:.2f}). "
                    "Ini membantu menentukan apakah perlu normalisasi atau kompresi tambahan."
                )
            else:
                st.info("Kolom ukuran file belum tersedia untuk histogram.")

        with size_col2:
            if category_col and size_col and not filtered_df.empty:
                fig_size_box, ax_size_box = plt.subplots(figsize=(8, 4.5))
                sns.boxplot(data=filtered_df, x=category_col, y=size_col, palette="Blues", ax=ax_size_box)
                ax_size_box.set_title("Boxplot Ukuran File per Kategori")
                ax_size_box.set_xlabel("Kategori")
                ax_size_box.set_ylabel("Ukuran File (KB)")
                ax_size_box.tick_params(axis="x", rotation=25)
                st.pyplot(fig_size_box)

                medians = filtered_df.groupby(category_col)[size_col].median().sort_values(ascending=False)
                if not medians.empty:
                    st.markdown(
                        f"**Insight:** Median ukuran file tertinggi berada pada **{medians.index[0]}** dan terendah pada **{medians.index[-1]}**, menandakan perbedaan karakteristik visual antar kategori."
                    )
            else:
                st.info("Boxplot ukuran file membutuhkan kolom kategori dan ukuran file.")

    with st.container():
        st.subheader("Deviasi dan Outlier Ukuran File")
        dev_col1, dev_col2 = st.columns(2)

        with dev_col1:
            if category_col and size_col and not filtered_df.empty:
                overall_median = filtered_df[size_col].median()
                median_by_class = filtered_df.groupby(category_col)[size_col].median()
                deviation_pct = ((median_by_class - overall_median).abs() / overall_median * 100).sort_values(ascending=False)

                dev_df = deviation_pct.reset_index()
                dev_df.columns = ["kategori", "deviasi_persen"]

                fig_dev, ax_dev = plt.subplots(figsize=(8, 4.5))
                sns.barplot(data=dev_df, x="kategori", y="deviasi_persen", palette="Blues", ax=ax_dev)
                ax_dev.set_title("Deviasi Median Ukuran File per Kategori")
                ax_dev.set_xlabel("Kategori")
                ax_dev.set_ylabel("Deviasi (%)")
                ax_dev.tick_params(axis="x", rotation=25)
                st.pyplot(fig_dev)

                top_dev = dev_df.head(2)["kategori"].tolist()
                if top_dev:
                    st.markdown(
                        f"**Insight:** Deviasi median terbesar berasal dari kategori {', '.join(top_dev)}. Kategori ini perlu perhatian khusus saat standarisasi ukuran input."
                    )
            else:
                st.info("Deviasi median membutuhkan kolom kategori dan ukuran file.")

        with dev_col2:
            if category_col and size_col and not filtered_df.empty:
                outlier_rates = (
                    filtered_df.groupby(category_col)[size_col]
                    .apply(compute_iqr_outlier_rate)
                    .sort_values(ascending=False)
                )
                outlier_df = outlier_rates.reset_index()
                outlier_df.columns = ["kategori", "outlier_rate"]

                fig_outlier, ax_outlier = plt.subplots(figsize=(8, 4.5))
                sns.barplot(data=outlier_df, x="kategori", y="outlier_rate", palette="Blues", ax=ax_outlier)
                ax_outlier.set_title("Outlier Rate Ukuran File (IQR)")
                ax_outlier.set_xlabel("Kategori")
                ax_outlier.set_ylabel("Outlier Rate (%)")
                ax_outlier.tick_params(axis="x", rotation=25)
                st.pyplot(fig_outlier)

                high_outliers = outlier_df[outlier_df["outlier_rate"] >= 5]["kategori"].tolist()
                if high_outliers:
                    st.markdown(
                        "**Insight:** Outlier ukuran file >=5% muncul pada kategori: "
                        + ", ".join(high_outliers)
                        + ". Ini menandakan perlunya QC tambahan atau trimming."
                    )
                else:
                    st.markdown("**Insight:** Tidak ada kategori dengan outlier rate di atas 5%.")
            else:
                st.info("Outlier rate membutuhkan kolom kategori dan ukuran file.")

    with st.container():
        st.subheader("Distribusi Jumlah Piksel")
        if category_col and pixel_col and not filtered_df.empty:
            fig_px, ax_px = plt.subplots(figsize=(10, 5))
            sns.violinplot(data=filtered_df, x=category_col, y=pixel_col, palette="Blues", ax=ax_px)
            ax_px.set_title("Violin Plot Jumlah Piksel per Kategori")
            ax_px.set_xlabel("Kategori")
            ax_px.set_ylabel("Jumlah Piksel")
            ax_px.tick_params(axis="x", rotation=25)
            st.pyplot(fig_px)

            pixel_medians = filtered_df.groupby(category_col)[pixel_col].median().sort_values(ascending=False)
            if not pixel_medians.empty:
                st.markdown(
                    f"**Insight:** Median jumlah piksel tertinggi berada pada **{pixel_medians.index[0]}**, mengindikasikan resolusi rata-rata lebih besar pada kategori tersebut."
                )
        else:
            st.info("Violin plot jumlah piksel membutuhkan kolom kategori dan jumlah piksel.")

    with st.container():
        st.subheader("Insight")
        for insight in build_insights(filtered_df, category_col, size_col, pixel_col):
            st.markdown(f"- {insight}")

    with st.container():
        st.subheader("Kesimpulan")
        st.markdown(
            "- Distribusi data antar kategori masih perlu dipantau untuk mencegah bias model pada kelas mayoritas.\n"
            "- Perbedaan karakteristik ukuran file dan resolusi antar kategori menguatkan kebutuhan preprocessing yang konsisten sekaligus adaptif.\n"
            "- Dashboard interaktif ini dapat digunakan untuk menentukan prioritas augmentasi data, class weight, dan quality control sebelum tahap modeling."
        )


def main() -> None:
    st.set_page_config(
        page_title="Dashboard Analisis Sampah Daur Ulang",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        :root {
            --bg: #eef6ff;
            --surface: rgba(255, 255, 255, 0.86);
            --surface-strong: rgba(255, 255, 255, 0.96);
            --border: rgba(79, 143, 247, 0.14);
            --primary-blue: #4f8ff7;
            --primary-blue-dark: #2f6fe0;
            --primary-blue-soft: #eaf3ff;
            --surface-blue: #f7fbff;
            --text-strong: #123056;
            --text-soft: #5f7896;
            --shadow: 0 18px 45px rgba(47, 111, 224, 0.10);
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(79, 143, 247, 0.10), transparent 35%),
                radial-gradient(circle at top right, rgba(122, 196, 255, 0.08), transparent 28%),
                linear-gradient(180deg, #f7fbff 0%, var(--bg) 100%);
            color: var(--text-strong);
        }
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
            max-width: 1520px;
        }
        .hero-banner {
            background: linear-gradient(135deg, #eff7ff 0%, #dfeeff 100%);
            border-radius: 28px;
            padding: 2rem 2.2rem;
            color: var(--text-strong);
            box-shadow: var(--shadow);
            margin-bottom: 1.25rem;
            border: 1px solid var(--border);
            position: relative;
            overflow: hidden;
        }
        .hero-banner::after {
            content: "";
            position: absolute;
            inset: auto -4rem -4rem auto;
            width: 11rem;
            height: 11rem;
            background: radial-gradient(circle, rgba(79, 143, 247, 0.22) 0%, rgba(79, 143, 247, 0) 70%);
            pointer-events: none;
        }
        .hero-title {
            font-size: clamp(1.7rem, 3vw, 2.4rem);
            font-weight: 800;
            margin: 0 0 0.45rem 0;
            letter-spacing: -0.03em;
        }
        .hero-subtitle {
            color: var(--text-soft);
            font-size: clamp(0.95rem, 1.5vw, 1.02rem);
            margin: 0;
            max-width: 62rem;
        }
        .section-divider {
            margin: 1rem 0 1.3rem 0;
        }
        .gallery-counter {
            text-align: center;
            color: var(--text-soft);
            font-weight: 600;
            padding-top: 0.6rem;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%);
            border-right: 1px solid rgba(79, 143, 247, 0.12);
        }
        section[data-testid="stSidebar"] > div {
            padding-top: 1rem;
        }
        .sidebar-brand {
            background: linear-gradient(135deg, #eff7ff 0%, #dfeeff 100%);
            border: 1px solid rgba(79, 143, 247, 0.16);
            border-radius: 22px;
            padding: 1rem 1rem 0.9rem 1rem;
            box-shadow: 0 12px 28px rgba(47, 111, 224, 0.08);
            margin-bottom: 1rem;
        }
        .sidebar-brand__eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #2f6fe0;
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(79, 143, 247, 0.14);
            border-radius: 999px;
            padding: 0.3rem 0.7rem;
            margin-bottom: 0.75rem;
        }
        .sidebar-brand__title {
            font-size: 1.15rem;
            font-weight: 800;
            color: var(--text-strong);
            margin: 0;
            line-height: 1.2;
        }
        .sidebar-brand__subtitle {
            color: var(--text-soft);
            font-size: 0.86rem;
            margin-top: 0.35rem;
            line-height: 1.4;
        }
        .sidebar-section-title {
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #5f7896;
            margin: 1rem 0 0.55rem 0;
        }
        .sidebar-note {
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(79, 143, 247, 0.12);
            border-radius: 18px;
            padding: 0.8rem 0.85rem;
            color: var(--text-soft);
            font-size: 0.85rem;
            line-height: 1.45;
            margin-bottom: 1rem;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding-top: 1rem;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
            padding-top: 0.5rem;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
            border-radius: 14px;
            margin-bottom: 0.35rem;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {
            background: rgba(79, 143, 247, 0.12);
        }
        section[data-testid="stSidebar"] .stRadio > div {
            gap: 0.2rem;
        }
        section[data-testid="stSidebar"] .stRadio label {
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(79, 143, 247, 0.12);
            border-radius: 16px;
            padding: 0.65rem 0.8rem;
            margin-bottom: 0.45rem;
            box-shadow: 0 8px 16px rgba(47, 111, 224, 0.04);
        }
        section[data-testid="stSidebar"] .stRadio label:hover {
            border-color: rgba(47, 111, 224, 0.22);
            background: rgba(239, 247, 255, 0.98);
        }
        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
            gap: 0.25rem;
        }
        section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] > div,
        section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div,
        section[data-testid="stSidebar"] .stDateInput input,
        section[data-testid="stSidebar"] .stNumberInput input,
        section[data-testid="stSidebar"] .stTextInput input {
            background: rgba(255,255,255,0.9);
            border-radius: 16px !important;
            border-color: rgba(79, 143, 247, 0.14) !important;
        }
        section[data-testid="stSidebar"] .stSlider {
            padding-top: 0.15rem;
        }
        .stButton > button {
            border-radius: 999px;
            border: 1px solid rgba(79, 143, 247, 0.16);
            background: linear-gradient(180deg, #ffffff 0%, #f3f8ff 100%);
            color: var(--text-strong);
            box-shadow: 0 10px 18px rgba(47, 111, 224, 0.08);
            transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            border-color: rgba(47, 111, 224, 0.28);
            box-shadow: 0 14px 24px rgba(47, 111, 224, 0.12);
        }
        .stMetric {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1rem 1rem 0.85rem 1rem;
            box-shadow: 0 10px 24px rgba(47, 111, 224, 0.05);
        }
        .stMetric label {
            color: var(--text-soft) !important;
        }
        [data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid var(--border);
            box-shadow: 0 10px 24px rgba(47, 111, 224, 0.05);
        }
        [data-testid="stSelectbox"], [data-testid="stMultiSelect"], [data-testid="stDateInput"], [data-testid="stSlider"] {
            border-radius: 16px;
        }
        .category-card {
            border-radius: 18px;
            padding: 1rem 0.85rem;
            text-align: center;
            background: linear-gradient(180deg, var(--card-bg), #ffffff);
            border: 1px solid color-mix(in srgb, var(--accent) 24%, white 76%);
            box-shadow: 0 10px 24px rgba(47, 111, 224, 0.07);
            margin-bottom: 0.5rem;
            min-height: 11rem;
        }
        .category-card__icon {
            font-size: clamp(1.5rem, 2vw, 1.95rem);
            line-height: 1;
            margin-bottom: 0.55rem;
        }
        .category-card__label {
            color: var(--label);
            font-weight: 800;
            font-size: clamp(0.88rem, 1.2vw, 0.98rem);
            margin-bottom: 0.35rem;
        }
        .category-card__value {
            color: var(--text-strong);
            font-size: clamp(1.15rem, 2vw, 1.45rem);
            font-weight: 800;
        }
        .gallery-card {
            margin-top: 0.45rem;
            margin-bottom: 1rem;
            padding: 0.7rem 0.8rem;
            border-radius: 14px;
            background: rgba(255,255,255,0.9);
            border: 1px solid rgba(79, 143, 247, 0.16);
            box-shadow: 0 8px 16px rgba(47, 111, 224, 0.05);
        }
        .gallery-card__badge {
            display: inline-block;
            font-size: 0.8rem;
            font-weight: 800;
            color: #245bb8;
            background: #e8f2ff;
            border-radius: 999px;
            padding: 0.24rem 0.7rem;
            margin-bottom: 0.45rem;
        }
        .gallery-card__caption {
            color: var(--text-soft);
            font-size: 0.84rem;
            word-break: break-word;
        }
        @media (max-width: 768px) {
            .hero-banner {
                padding: 1.4rem 1.2rem;
                border-radius: 22px;
            }
            .hero-title {
                font-size: 1.55rem;
            }
            .hero-subtitle {
                font-size: 0.95rem;
            }
            .category-card {
                min-height: 9.5rem;
                padding: 0.9rem 0.75rem;
            }
            .gallery-card {
                padding: 0.65rem 0.7rem;
            }
        }
        @media (max-width: 640px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            .stButton > button {
                width: 100%;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    clean_df = None
    date_cols: list[str] = []
    load_error: Exception | None = None

    for dataset_source in get_dataset_sources():
        try:
            clean_df, date_cols = load_and_preprocess_dataset(dataset_source)
            dataset_path = dataset_source
            break
        except Exception as exc:
            load_error = exc
    else:
        st.error(
            "Gagal memuat dataset dari URL GitHub maupun file lokal. "
            "Periksa koneksi internet dan pastikan file CSV tersedia."
        )
        if load_error is not None:
            st.exception(load_error)
        st.stop()

    assert clean_df is not None

    category_col = find_existing_column(clean_df, ["kategori", "class_label", "category"])
    file_name_col = find_existing_column(clean_df, ["nama_file", "file_name"])
    size_col = find_existing_column(clean_df, ["ukuran_file_kb", "file_size_kb"])
    pixel_col = find_existing_column(clean_df, ["jumlah_piksel", "pixels"])

    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand__eyebrow">♻️ Sampah Explorer</div>
            <div class="sidebar-brand__title">Dashboard Sampah Daur Ulang</div>
            <div class="sidebar-brand__subtitle">Navigasi cepat ke galeri dan analitik data gambar.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown('<div class="sidebar-section-title">Navigasi</div>', unsafe_allow_html=True)
    page = st.sidebar.radio(
        "Pilih halaman",
        options=["Gallery", "Analytics"],
        index=0,
        label_visibility="collapsed",
    )

    st.sidebar.markdown(
        f"""
        <div class="sidebar-note">
            <strong>Sumber aktif</strong><br>
            {dataset_path}<br>
            <span style="opacity:0.92;">Pilih halaman untuk fokus pada gambar atau hasil analisis.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown('<div class="sidebar-section-title">Filter</div>', unsafe_allow_html=True)
    filtered_df, selected_date_col = sidebar_filters(clean_df, category_col, date_cols)

    raw_images_root = Path("images")
    curated_images_root = Path("clean_images_filtered")
    available_image_roots = [
        root for root in (raw_images_root, curated_images_root)
        if root.exists() and root.is_dir()
    ]

    total_data_count = len(clean_df)

    st.caption(f"Sumber data aktif: {dataset_path}")

    if page == "Gallery":
        render_gallery_page(filtered_df, category_col, file_name_col, available_image_roots, total_data_count)
    else:
        render_analytics_page(filtered_df, category_col, size_col, pixel_col, selected_date_col, total_data_count)


if __name__ == "__main__":
    main()