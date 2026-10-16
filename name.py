import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
LAST_YEAR = 2025      # 수업 기준 기간의 끝
MIN_DAYS = 300        # 연도별 최소 관측일 수

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="wide")


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_yearly():
    df = pd.read_csv(URL, encoding="utf-8")
    df.columns = [str(c).strip() for c in df.columns]
    df["날짜"] = pd.to_datetime(df["날짜"].astype(str).str.strip(), errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜"])
    df["연도"] = df["날짜"].dt.year

    # 2025년까지만 사용
    df = df[df["연도"] <= LAST_YEAR]

    # 연도별 관측일 수(평균기온이 기록된 날)와 평균기온
    g = df.groupby("연도")["평균기온"].agg(관측일="count", 연평균기온="mean").reset_index()
    total_years = len(g)
    g = g[g["관측일"] >= MIN_DAYS].reset_index(drop=True)
    return g, total_years


st.title("🌡️ 기온 예측기")
st.caption("서울의 연평균기온 추세(회귀 직선)로 원하는 해의 기온을 예상해 봅니다.")

try:
    yearly, total_years = load_yearly()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

if len(yearly) < 2:
    st.error("회귀 직선을 만들기에 충분한 연도 데이터가 없습니다.")
    st.stop()

x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)

slope, intercept = np.polyfit(x, y, 1)
r = float(np.corrcoef(x, y)[0, 1])

n_years = len(yearly)
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())

# ---- 사용한 데이터 정보 ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("직선을 만든 해의 개수", f"{n_years}개")
c2.metric("시작 연도", f"{start_year}년")
c3.metric("끝 연도", f"{end_year}년")
c4.metric("상관계수 (r)", f"{r:.3f}")
st.caption(
    f"{LAST_YEAR}년 이후 데이터와 관측일이 {MIN_DAYS}일 미만인 해"
    f"({total_years - n_years}개 연도)는 제외했습니다."
)

st.divider()

# ---- 연도 선택 & 예측 ----
year = st.slider("예상 기온을 알고 싶은 연도", 1900, 2100, 2050, step=1)
pred = slope * year + intercept

st.markdown(
    f"""
    <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
        <div style="font-size:1.2rem; opacity:0.75;">{year}년의 예상 평균기온</div>
        <div style="font-size:4.5rem; font-weight:800; line-height:1.2;">{pred:.2f} °C</div>
    </div>
    """,
    unsafe_allow_html=True,
)
if year < start_year or year > end_year:
    st.info(
        f"{year}년은 직선을 만든 기간({start_year}~{end_year}년) 밖이므로 "
        "추세를 연장한 값입니다. 실제와 차이가 클 수 있어요."
    )

# ---- 그래프 ----
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(size=8, opacity=0.8),
        hovertemplate="%{x}년<br>%{y:.2f}°C<extra></extra>",
    )
)
line_x = np.array([1900, 2100])
fig.add_trace(
    go.Scatter(
        x=line_x,
        y=slope * line_x + intercept,
        mode="lines",
        name="회귀 직선",
        line=dict(width=3, color="red"),
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter(
        x=[year],
        y=[pred],
        mode="markers",
        name=f"{year}년 예상",
        marker=dict(size=16, symbol="star", color="gold", line=dict(width=1.5, color="black")),
        hovertemplate=f"{year}년 예상<br>{pred:.2f}°C<extra></extra>",
    )
)
fig.update_layout(
    title=f"서울 연평균기온과 회귀 직선 (상관계수 r = {r:.3f})",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    xaxis=dict(range=[1900, 2100]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=520,
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    f"**회귀식:** 평균기온 = {slope:.4f} × 연도 + ({intercept:.2f})  \n"
    f"→ 10년마다 약 **{slope * 10:+.2f} °C** 변화"
)

with st.expander("연도별 평균기온 데이터 보기"):
    st.dataframe(
        yearly.rename(columns={"관측일": "관측일 수"}).round({"연평균기온": 2}),
        use_container_width=True,
        hide_index=True,
    )
