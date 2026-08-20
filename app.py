import streamlit as st
from science_agent import run_agent, LEVEL_OPTIONS
import knowledge_base as kb

st.set_page_config(page_title="科普智能体", page_icon="🔭", layout="centered")

st.markdown("""
<style>
.answer-card {
    background-color: #ffffff;
    border: 1px solid #e6e6e6;
    border-radius: 14px;
    padding: 24px 28px;
    margin-top: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}
.answer-card h2 {
    font-size: 1.15rem;
    margin-top: 1.2em;
    margin-bottom: 0.5em;
    padding-bottom: 6px;
    border-bottom: 2px solid #f0f2f6;
}
.question-card {
    background: linear-gradient(135deg, #fff8e6, #fff3d6);
    border: 1px solid #ffe4a3;
    border-radius: 14px;
    padding: 18px 22px;
    margin-top: 14px;
}
.question-card b { color: #b5762f; }

.practice-card {
    background: linear-gradient(135deg, #eaf3ff, #dcecff);
    border: 1px solid #b9d8ff;
    border-radius: 14px;
    padding: 18px 22px;
    margin-top: 14px;
    white-space: pre-wrap;
}
.practice-card b { color: #2b5fa8; }

.frontier-card {
    background: linear-gradient(135deg, #f3eaff, #ebdcff);
    border: 1px solid #d6b9ff;
    border-radius: 14px;
    padding: 18px 22px;
    margin-top: 14px;
}
.frontier-card b { color: #6a3fa0; }

.link-item {
    display: block;
    padding: 10px 14px;
    margin-top: 8px;
    background: #f7f8fa;
    border-radius: 10px;
    border: 1px solid #e6e6e6;
    text-decoration: none !important;
    color: #333 !important;
}
.link-item:hover { background: #eef1f5; }
</style>
""", unsafe_allow_html=True)

def _get_secret(key: str, default: str = "") -> str:
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    import os
    return os.environ.get(key, default)


# 简单的访问密码保护：部署到公网后，防止陌生人随意访问消耗你的API额度。
# 不设置 APP_PASSWORD 这个 secret/环境变量的话，就不会启用密码保护。
_app_password = _get_secret("APP_PASSWORD")
if _app_password:
    if "authed" not in st.session_state:
        st.session_state.authed = False
    if not st.session_state.authed:
        st.title("🔭 科普智能体")
        pwd = st.text_input("请输入访问密码", type="password")
        if pwd:
            if pwd == _app_password:
                st.session_state.authed = True
                st.rerun()
            else:
                st.error("密码不正确")
        st.stop()

st.title("🔭 科普智能体")
st.caption("兼顾课本知识、考点考纲与前沿拓展")

with st.sidebar:
    st.subheader("⚙️ 设置")
    level = st.selectbox("学段（不选则自动判断）", LEVEL_OPTIONS, index=0)
    want_practice = st.checkbox("附加巩固练习", value=True)
    want_frontier = st.checkbox("附加前沿挑战", value=True)
    want_links = st.checkbox("附加权威拓展资料", value=True)

    st.divider()
    st.subheader("📚 我的题库")
    st.caption("上传诊断卷/试卷，生成巩固练习时会优先从这里检索真实题目")

    uploaded_files = st.file_uploader(
        "上传卷子（支持 pdf / docx / txt，可多选）",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("添加到题库"):
        for f in uploaded_files:
            try:
                n = kb.add_document(f.name, f.getvalue())
                if n > 0:
                    st.success(f"「{f.name}」已解析，新增 {n} 道题")
                else:
                    st.warning(f"「{f.name}」没能提取出有效内容（如果是扫描版PDF，可能没有可选中的文字）")
            except Exception as e:
                st.error(f"「{f.name}」处理失败：{e}")

    docs = kb.list_documents()
    if docs:
        st.write(f"题库中共有 **{sum(d['count'] for d in docs)}** 道题，来自 {len(docs)} 份文件：")
        for d in docs:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"📄 {d['source']}（{d['count']}题）")
            with col2:
                if st.button("删除", key=f"del_{d['source']}"):
                    kb.remove_document(d['source'])
                    st.rerun()
        if st.button("🗑️ 清空整个题库"):
            kb.clear_kb()
            st.rerun()
    else:
        st.caption("题库目前是空的")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_extras(images, questions, practice, practice_source, frontier, links):
    if images:
        st.write("")
        cols = st.columns(min(len(images), 3))
        for i, url in enumerate(images):
            with cols[i % len(cols)]:
                try:
                    st.image(url, use_container_width=True)
                except Exception:
                    pass

    for q in questions:
        st.markdown(f'<div class="question-card">🤔 <b>拓展思考</b><br>{q}</div>', unsafe_allow_html=True)

    if practice:
        label = "来自你上传的题库" if practice_source == "kb" else "AI结合考点生成，非保证真实原题，仅供参考"
        st.markdown(
            f'<div class="practice-card">📝 <b>巩固练习</b>（{label}）<br><br>{practice}</div>',
            unsafe_allow_html=True,
        )

    if frontier:
        st.markdown(f'<div class="frontier-card">🧭 <b>前沿挑战</b><br>{frontier}</div>', unsafe_allow_html=True)

    if links:
        st.markdown("**🔗 拓展资料**")
        for link in links:
            st.markdown(
                f'<a class="link-item" href="{link["url"]}" target="_blank">🔎 {link["title"]}</a>',
                unsafe_allow_html=True,
            )


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(f'<div class="answer-card">\n\n{msg.get("answer", "")}\n\n</div>', unsafe_allow_html=True)
            render_extras(
                msg.get("images", []),
                msg.get("questions", []),
                msg.get("practice"),
                msg.get("practice_source", "ai"),
                msg.get("frontier"),
                msg.get("links", []),
            )
        else:
            st.write(msg["content"])

if prompt := st.chat_input("请输入你想了解的科普问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        status_box = st.status("思考中...", expanded=True)

        def on_step(text):
            status_box.write(text)

        try:
            result = run_agent(
                prompt,
                level=level,
                on_step=on_step,
                want_practice=want_practice,
                want_frontier=want_frontier,
                want_links=want_links,
            )
            error = None
        except Exception as e:
            result = {
                "answer": f"运行出错：{e}",
                "images": [], "questions": [], "practice": None,
                "practice_source": "none", "frontier": None, "links": [], "level": level,
            }
            error = str(e)

        status_box.update(
            label=f"完成（学段：{result['level']}）" if not error else "出错了",
            state="complete" if not error else "error",
            expanded=False,
        )

        st.markdown(f'<div class="answer-card">\n\n{result["answer"]}\n\n</div>', unsafe_allow_html=True)
        render_extras(
            result["images"], result["questions"],
            result["practice"], result["practice_source"],
            result["frontier"], result["links"],
        )

    st.session_state.messages.append({"role": "assistant", **result})
