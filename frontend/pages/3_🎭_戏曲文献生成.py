# frontend/pages/3_🎭_戏曲文献生成.py 戏曲文献生成页面
import streamlit as st
import sys
import os

# 确保能导入api_client
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from api_client import api_client

st.set_page_config(page_title="戏曲文献生成", page_icon="🎭", layout="wide")

st.title("🎭 戏曲文献生成")
st.markdown("一键生成戏曲学术文献，支持 **txt / pdf / md** 三种格式输出")

# ===================== 文献参数配置 =====================
st.subheader("📝 文献参数")

col1, col2, col3 = st.columns(3)

with col1:
    genre = st.selectbox(
        "戏曲种类",
        ["京剧", "豫剧", "越剧", "黄梅戏", "评剧", "昆曲", "川剧", "粤剧"],
    )

with col2:
    length = st.slider(
        "目标字数",
        min_value=200,
        max_value=3000,
        value=800,
        step=100,
        help="文献目标生成字数",
    )

with col3:
    formats = st.multiselect(
        "输出格式",
        ["txt", "pdf", "md"],
        default=["txt", "pdf", "md"],
        help="选择要生成的文献格式",
    )

theme = st.text_input(
    "文献主题",
    value="戏曲艺术特色与发展",
    placeholder="例如：京剧脸谱艺术研究、豫剧唱腔特色分析",
)

title_input = st.text_input(
    "自定义文件名（可选）",
    placeholder="留空则自动生成",
    help="生成的文件名（不含扩展名）",
)

# 格式校验
if not formats:
    st.warning("⚠️ 请至少选择一种输出格式")

# ===================== 生成按钮 =====================
col_a, col_b = st.columns([1, 5])
with col_a:
    gen_btn = st.button(
        "🎭 生成文献",
        type="primary",
        use_container_width=True,
        disabled=not formats,
    )

if gen_btn:
    with st.spinner("🎭 正在调用LLM生成戏曲文献，请稍候..."):
        resp = api_client.literature_generate(
            genre=genre,
            theme=theme,
            length=length,
            formats=formats,
            title=title_input if title_input.strip() else None,
        )

    if resp.get("code") == 200:
        st.success(f"✅ {resp['msg']}")

        # 展示生成的文件
        files = resp["data"]["files"]
        st.subheader("📄 生成的文件")

        for f in files:
            fmt_icon = {"txt": "📄", "pdf": "📕", "md": "📝"}.get(f["format"], "📄")
            download_url = api_client.literature_download_url(f["path"])
            with st.container():
                col1, col2, col3 = st.columns([1, 4, 2])
                col1.markdown(f"{fmt_icon} **.{f['format'].upper()}**")
                col2.markdown(f"`{f['filename']}`")
                col3.markdown(f"[⬇️ 下载 {f['format'].upper()}]({download_url})")
        st.markdown("")
        st.info(f"📁 文献已保存至服务器目录：`{resp['data']['root_dir']}`")
    else:
        st.error(f"❌ {resp.get('msg', '生成失败')}")

st.markdown("---")

# ===================== 已生成文献列表 =====================
st.subheader("📚 已生成文献")

# 刷新按钮
if st.button("🔄 刷新文献列表", use_container_width=False):
    st.rerun()

file_list = api_client.literature_list()
if file_list.get("code") == 200:
    data = file_list["data"]
    st.caption(f"文献根目录：`{data['root_dir']}`，共 {data['total']} 个文件")

    files_by_type = data["files"]

    # 按类型分列展示
    type_tabs = st.tabs(["📄 TXT", "📕 PDF", "📝 MD"])

    for idx, (fmt, items) in enumerate(files_by_type.items()):
        with type_tabs[idx]:
            if not items:
                st.info(f"暂无 {fmt.upper()} 格式文献")
                continue
            for item in items:
                col1, col2, col3 = st.columns([3, 2, 2])
                with col1:
                    st.markdown(f"**{item['filename']}**")
                with col2:
                    st.caption(f"大小：{item['size_kb']}KB\n修改：{item['modified_time']}")
                with col3:
                    download_url = api_client.literature_download_url(item["path"])
                    st.markdown(f"[⬇️ 下载]({download_url})")
                st.divider()
else:
    st.warning(file_list.get("msg", "无法获取文献列表"))