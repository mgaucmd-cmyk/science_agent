import os
import re
from typing import Optional, List, Dict, Tuple

from openai import OpenAI
from tavily import TavilyClient

from knowledge_base import search_kb

# ============================================================
# 核心科普回答部分（保持原有 ReAct 循环）
# ============================================================

AGENT_SYSTEM_PROMPT = """
你是一个智能科普助手。你的科普对象涵盖中小学以及大学生，内容包括不限于天文、物理，不仅能巧妙结合课本知识，需要进行丰富的前沿知识以及跨学科知识拓展或进一步提问，拓宽知识广度和深度，科普过程中要有趣又专业。引导提问者分析思考，进一步解释剖析事物背后的原理。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
- `search_science(topic: str)`: 根据科普主题搜索资料，返回课本知识点、原理剖析、前沿进展、跨学科拓展内容。
- `generate_question(content: str)`: 根据已获取的科普内容，生成启发思考的拓展提问，引导用户分析思考。

# 输出格式要求:
你的每次回复必须严格遵循以下格式，包含一对Thought和Action：

Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action的格式必须是以下之一：
1. 调用工具：function_name(arg_name="arg_value")
2. 结束任务：Finish[最终答案]

# 最终答案排版要求（非常重要）:
当你使用 Finish[...] 结束任务时，方括号内的内容必须是排版清晰的 Markdown，严格按以下结构组织（没有对应内容的部分可以省略该标题，但至少要有课本知识点和原理剖析两部分）：

## 📘 课本知识点
（对应中学/大学课本的基础概念、公式、考点考纲，简明扼要）

## 🔬 原理剖析
（深入解释背后的物理/科学原理，可分小段）

## 🚀 前沿拓展
（现代科技应用、最新研究进展、跨学科联系）

## 🤔 想一想
（1-3个启发思考的问题，用有序列表）

段落之间要有换行，适当使用加粗突出关键词、公式，避免输出成一整段文字。

# 语言与公式格式要求（严格遵守）:
- 全部输出必须是简体中文。即使你检索到的参考资料是英文，也必须完全理解后用中文重新表达，绝不能直接把英文原文或翻译腔句子夹杂在回答里。
- 数学公式必须使用美元符号包裹：行内公式用单个美元符号，如 F浮=ρ液gV排 这种简单公式可以直接用文字加下标表示；如果确实需要用LaTeX，行内公式写成 $F_{浮}=\rho_{液}gV_{排}$ 这种单美元符号包裹的形式，块级公式用两个美元符号包裹并单独成行。禁止使用 \\( \\) 或 \\[ \\] 这种反斜杠加括号的LaTeX定界符，Streamlit渲染不了会显示成乱码。

# 重要提示:
- 每次只输出一对Thought‑Action
- Action必须在同一行，不要换行
- 当收集到足够信息可以回答用户问题时，必须使用 Action: Finish[最终答案] 格式结束
- 优先调用search_science获取专业素材，拿到素材后可以调用generate_question生成思考题，之后整合全部内容输出完整科普回答。
- 最多只有5次行动机会：如果已经调用过 search_science 和 generate_question 各一次，下一步必须直接 Finish，把已获得的素材整合成完整科普回答，不要再重复调用工具。
"""

_session_images = []
_session_questions = []


def search_science(topic: str) -> str:
    """调用Tavily搜索科普主题：课本知识、原理、前沿、跨学科拓展，顺带抓取相关图片"""
    api_key = _get_secret("TAVILY_API_KEY")
    if not api_key:
        return "错误：未配置TAVILY_API_KEY。"
    tavily = TavilyClient(api_key=api_key)
    query = f"{topic} 中小学大学科普 课本知识点 底层原理 前沿研究 跨学科拓展"
    try:
        response = tavily.search(
            query=query,
            search_depth="basic",
            include_answer=True,
            include_images=True,
            max_results=5,
        )
        for img in response.get("images", []) or []:
            url = img.get("url") if isinstance(img, dict) else img
            if url and url not in _session_images:
                _session_images.append(url)

        if response.get("answer"):
            return "（以下为检索到的参考资料，如含英文，请在最终科普内容中用中文重新表达）：\n" + response["answer"]
        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")
        if not formatted_results:
            return f"抱歉，没有找到关于【{topic}】的科普资料。"
        return "科普检索结果：\n" + "\n".join(formatted_results)
    except Exception as e:
        return f"错误：执行科普搜索时出现问题 - {e}"


def generate_question(content: str) -> str:
    """根据科普文本生成启发式思考问题（用自家LLM转述，保证输出为中文），同时单独保存一份供网页展示"""
    api_key = _get_secret("TAVILY_API_KEY")
    context = ""
    if api_key:
        try:
            tavily = TavilyClient(api_key=api_key)
            response = tavily.search(
                query=f"{content[:200]} 启发式思考问题 原理分析",
                search_depth="basic",
                max_results=5,
            )
            context = "\n".join(r.get("content", "") for r in response.get("results", []))
        except Exception:
            context = ""

    llm = _get_llm()
    prompt = (
        f"科普内容：{content[:1200]}\n\n"
        + (f"补充参考资料（可能包含英文，仅供参考，不要照抄或直译，需理解后融会贯通）：{context[:1500]}\n\n" if context else "")
        + "请基于以上内容，生成3-5个启发思考的问题，引导分析原理、拓展思考，不要出简单记忆题。直接输出问题列表，不要有多余说明。"
    )
    result = llm.generate(
        prompt,
        system_prompt="你是一名善于启发学生深入思考的老师。无论参考资料是什么语言，输出必须全部是简体中文。",
    )
    if result.startswith("错误："):
        return result
    result = result.strip()
    _session_questions.append(result)
    return "拓展思考题：\n" + result


available_tools = {
    "search_science": search_science,
    "generate_question": generate_question,
}


class OpenAICompatibleClient:
    """一个用于调用任何兼容OpenAI接口的LLM服务的客户端。"""

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"错误：调用语言模型服务时出错 - {e}"


def _get_secret(key: str) -> str:
    """
    优先从 Streamlit Cloud 的 secrets 读取（部署到云端时），
    读不到再退回本地环境变量（本地开发时使用）。
    """
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, "")


BASE_URL = "https://api-inference.modelscope.cn/v1"
MODEL_ID = "deepseek-ai/DeepSeek-V4-Pro-0813"

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        api_key = _get_secret("MODELSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("未配置 MODELSCOPE_API_KEY，请检查环境变量或 Streamlit Secrets。")
        _llm = OpenAICompatibleClient(model=MODEL_ID, api_key=api_key, base_url=BASE_URL)
    return _llm


# ============================================================
# 学段判断
# ============================================================

LEVEL_OPTIONS = ["自动判断", "小学", "初中(中考)", "高中(高考)", "大学"]

_LEVEL_KEYWORDS = {
    "小学": ["小学", "三年级", "四年级", "五年级", "六年级"],
    "初中(中考)": ["中考", "初一", "初二", "初三", "七年级", "八年级", "九年级", "初中"],
    "高中(高考)": ["高考", "高一", "高二", "高三", "选修", "必修", "高中"],
    "大学": ["大学", "考研", "期末", "线性代数", "普通物理", "大学物理", "高等数学", "本科", "研究生"],
}


def detect_level(user_prompt: str) -> str:
    """根据关键词粗略判断学段，判断不出时默认覆盖初高中通用难度"""
    for level, kws in _LEVEL_KEYWORDS.items():
        for kw in kws:
            if kw in user_prompt:
                return level
    return "初中(中考)~高中(高考)通用"


# ============================================================
# 巩固练习 / 前沿挑战 / 权威资料链接
# 三者均独立于ReAct主循环之外，由确定性代码控制，保证质量可控
# ============================================================

# 只从这些相对权威、内容质量有保障的域名里找拓展资料，其余一律不采用
TRUSTED_DOMAINS = [
    "bilibili.com",
    "zh.wikipedia.org",
    "wikipedia.org",
    "khanacademy.org",
    "arxiv.org",
    "nature.com",
    "icourse163.org",   # 中国大学MOOC
    "guokr.com",         # 果壳
    "cas.cn",             # 中科院
]

# Tavily 搜索结果的相关度低于这个分数就不采用，宁缺毋滥
LINK_SCORE_THRESHOLD = 0.35


def get_practice_problems(topic: str, level: str) -> Tuple[Optional[str], str]:
    """
    生成/检索1-2道贴合考点的练习题。
    优先从用户自己上传的题库中检索真实题目；检索不到高质量匹配时，
    才退化为让AI结合考点生成（并标注为AI生成，非保证真实原题）。

    返回 (题目文本 或 None, 来源标记)，来源标记为 "kb"（题库） / "ai"（AI生成） / "none"
    """
    kb_hits = search_kb(topic, top_k=2)
    if kb_hits:
        lines = []
        for h in kb_hits:
            lines.append(f"【来源：{h['source']}】\n\n{h['text']}")
        return "\n\n---\n\n".join(lines), "kb"

    llm = _get_llm()
    prompt = (
        f"请判断『{topic}』属于哪个学科方向（物理/化学/生物/天文/数学等），"
        f"针对『{level}』这一学段水平，出1-2道贴合该学段常见考点和题型的练习题。\n"
        f"每道题需包含：题目正文（如为选择题请给出选项）、正确答案、简要解析（2-3句即可）。\n"
        f"如果该主题本身不适合以练习题形式呈现（比如纯概念性、无法命题），请只回复：不适用。"
    )
    result = llm.generate(
        prompt,
        system_prompt="你是一名经验丰富、熟悉各学段考纲的学科教师，出题严谨、贴合考点，不出偏题怪题。",
    )
    if not result or "不适用" in result.strip()[:10] or result.startswith("错误："):
        return None, "none"
    return result.strip(), "ai"


def get_frontier_challenge(topic: str) -> Optional[str]:
    """
    介绍该方向目前尚未解决的前沿问题/技术瓶颈，激发深入思考。
    先用搜索获取参考资料，再统一交给自家LLM转述成中文（避免直接透传英文摘要）。
    """
    api_key = _get_secret("TAVILY_API_KEY")
    context = ""
    if api_key:
        try:
            tavily = TavilyClient(api_key=api_key)
            response = tavily.search(
                query=f"{topic} 领域未解决的问题 前沿难题 技术瓶颈",
                search_depth="basic",
                max_results=5,
            )
            context = "\n".join(r.get("content", "") for r in response.get("results", []))
        except Exception:
            context = ""

    llm = _get_llm()
    prompt = (
        f"主题：{topic}\n\n"
        + (f"检索到的参考资料（可能包含英文，请理解后完全用中文重新表达，不要保留英文原文或直译腔）：{context[:1500]}\n\n" if context else "")
        + "请简要介绍该方向目前科学界公认尚未解决的1-2个前沿问题或技术瓶颈，说明其意义与难点所在，激发读者深入思考。"
          "如果你不确定某个具体细节，请直接略过，不要编造。如果该主题实在没有明显的'未解决前沿问题'，请只回复：暂无。"
    )
    text = llm.generate(
        prompt,
        system_prompt="你是一名严谨的科学作家，只陈述你确信真实存在的前沿难题，绝不编造具体数据或事件。无论参考资料是什么语言，输出必须全部是简体中文。",
    )
    if not text or "暂无" in text.strip()[:10] or text.startswith("错误："):
        return None
    return text.strip()


def search_authoritative_links(topic: str) -> List[Dict]:
    """
    在可信域名范围内搜索权威资料（文章/文献/视频讲解等），按相关度过滤，
    质量不达标就返回空列表，不勉强凑数。
    """
    api_key = _get_secret("TAVILY_API_KEY")
    if not api_key:
        return []
    try:
        tavily = TavilyClient(api_key=api_key)
        response = tavily.search(
            query=f"{topic} 讲解 科普",
            search_depth="basic",
            max_results=8,
            include_domains=TRUSTED_DOMAINS,
        )
        candidates = []
        for r in response.get("results", []):
            score = r.get("score", 0) or 0
            if score < LINK_SCORE_THRESHOLD:
                continue
            candidates.append({
                "title": r.get("title", "").strip(),
                "url": r.get("url", ""),
                "score": score,
            })
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:3]
    except Exception:
        return []


# ============================================================
# 主入口
# ============================================================

def run_agent(
    user_prompt: str,
    level: str = "自动判断",
    max_loops: int = 5,
    on_step=None,
    want_practice: bool = True,
    want_frontier: bool = True,
    want_links: bool = True,
) -> dict:
    """
    运行一次完整流程，返回：
        {
            "answer": 核心科普内容(Markdown),
            "images": [图片URL, ...],
            "questions": [思考题文本, ...],
            "level": 实际使用的学段,
            "practice": 巩固练习文本 或 None,
            "frontier": 前沿挑战文本 或 None,
            "links": [{"title":..., "url":...}, ...],
        }
    """
    _session_images.clear()
    _session_questions.clear()

    resolved_level = level if level and level != "自动判断" else detect_level(user_prompt)

    llm = _get_llm()
    prompt_history = [f"用户请求: {user_prompt}"]
    final_answer = None
    practice: Optional[str] = None
    practice_source: str = "none"
    frontier: Optional[str] = None
    links: List[Dict] = []

    for i in range(max_loops):
        full_prompt = "\n".join(prompt_history)
        llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)

        if llm_output.strip().startswith("Sorry,") or "topup" in llm_output:
            final_answer = "调用大语言模型时遇到额度或限流问题，请检查账号余额或稍后重试。"
            break

        match = re.search(
            r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)',
            llm_output, re.DOTALL)
        if match:
            llm_output = match.group(1).strip()

        if on_step:
            on_step(f"**第{i + 1}轮**\n\n{llm_output}")
        prompt_history.append(llm_output)

        action_match = re.search(r"Action:\s*(.*)", llm_output, re.DOTALL)
        if not action_match:
            observation_str = "Observation: 错误: 未能解析到 Action 字段。请确保回复严格遵循 'Thought: ... Action: ...' 的格式。"
            prompt_history.append(observation_str)
            continue

        action_str = action_match.group(1).strip()

        if action_str.startswith("Finish"):
            finish_match = re.match(r"Finish\[(.*)\]", action_str, re.DOTALL)
            final_answer = finish_match.group(1) if finish_match else action_str
            break

        try:
            tool_name = re.search(r"(\w+)\(", action_str).group(1)
            args_str = re.search(r"\((.*)\)", action_str, re.DOTALL).group(1)
            kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))
        except AttributeError:
            observation_str = f"Observation: 错误：无法解析 Action 内容，请检查格式。原始内容: {action_str}"
            prompt_history.append(observation_str)
            continue

        if tool_name in available_tools:
            observation = available_tools[tool_name](**kwargs)
        else:
            observation = f"错误：未定义的工具 '{tool_name}'"

        observation_str = f"Observation: {observation}"
        if on_step:
            on_step(observation_str)
        prompt_history.append(observation_str)

    if final_answer is None:
        full_prompt = "\n".join(prompt_history) + \
            "\n\n请不要再调用任何工具，直接基于以上已获得的素材，按要求的Markdown格式输出 Action: Finish[完整科普回答]。"
        final_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
        finish_match = re.search(r"Finish\[(.*)\]", final_output, re.DOTALL)
        final_answer = finish_match.group(1) if finish_match else final_output

    if want_practice:
        if on_step:
            on_step(f"正在根据「{resolved_level}」学段查找/生成巩固练习...")
        practice, practice_source = get_practice_problems(user_prompt, resolved_level)

    if want_frontier:
        if on_step:
            on_step("正在梳理该方向的前沿挑战...")
        frontier = get_frontier_challenge(user_prompt)

    if want_links:
        if on_step:
            on_step("正在筛选权威拓展资料...")
        links = search_authoritative_links(user_prompt)

    return {
        "answer": final_answer,
        "images": list(_session_images)[:6],
        "questions": list(_session_questions),
        "level": resolved_level,
        "practice": practice,
        "practice_source": practice_source,
        "frontier": frontier,
        "links": links,
    }


if __name__ == "__main__":
    q = input("请输入你的问题: ")
    result = run_agent(q, on_step=print)
    print(result["answer"])
