import os
import operator
from typing import Literal, Annotated

import streamlit as st
from pydantic import BaseModel, Field, model_validator
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch


# ---------------------------------------------------------------------------
# 1. API KEYS (from Streamlit Cloud's secrets manager, not hardcoded)
# ---------------------------------------------------------------------------
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]


# ---------------------------------------------------------------------------
# 2. SCHEMAS (unchanged from the Colab build)
# ---------------------------------------------------------------------------
class Finding(BaseModel):
    sub_question: str
    answer: str = Field(description="A concise, well-supported answer to the sub-question, based only on the search results provided")
    sources: list[str] = Field(description="URLs of the sources that support the answer")
    is_grounded: bool = Field(default=True, description="Automatically set to False if no sources were found")

    @model_validator(mode="after")
    def check_grounding(self):
        if len(self.sources) == 0:
            self.is_grounded = False
        return self


class FindingVerification(BaseModel):
    sub_question: str
    is_well_supported: bool = Field(description="False if the answer is thin, unsupported by the given sources, or doesn't actually answer the sub-question")
    issue: str = Field(default="", description="Brief explanation of the problem, empty string if is_well_supported is True")


class CriticReport(BaseModel):
    verifications: list[FindingVerification]
    contradictions: list[str] = Field(default_factory=list, description="Descriptions of any direct contradictions found between different findings. Empty list if none.")


class ResearchState(BaseModel):
    user_input: str
    input_type: Literal["question", "statement", "clarification", "suggestion"] = "question"
    question: str = ""
    sub_questions: list[str] = Field(default_factory=list)
    findings: Annotated[list[Finding], operator.add] = Field(default_factory=list)
    critic_report: CriticReport | None = None
    report: str = ""


class SubQuestions(BaseModel):
    sub_questions: list[str] = Field(
        min_length=1,
        max_length=6,
        description="1 to 6 focused, non-overlapping sub-questions that together cover the original research question. Use 1 if the question is already narrow and doesn't need decomposition."
    )


# ---------------------------------------------------------------------------
# 3. HELPER: safely extract plain text from an LLM response
# ---------------------------------------------------------------------------
def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block["text"] if isinstance(block, dict) and "text" in block else str(block)
            for block in content
        )
    return str(content)


# ---------------------------------------------------------------------------
# 4. GRAPH CONSTRUCTION (built once, cached across Streamlit reruns)
# ---------------------------------------------------------------------------
@st.cache_resource
def build_graph():
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)
    llm_planner = llm.with_structured_output(SubQuestions)
    llm_researcher = llm.with_structured_output(Finding)
    llm_critic = llm.with_structured_output(CriticReport)
    search_tool = TavilySearch(max_results=5)

    def research_one(sub_question: str) -> Finding:
        search_results = search_tool.invoke(sub_question)
        prompt = (
            f"Sub-question: {sub_question}\n\n"
            f"Search results:\n{search_results}\n\n"
            f"Based only on these search results, provide a concise answer with source URLs."
        )
        return llm_researcher.invoke(prompt)

    def planner(state: ResearchState) -> dict:
        prompt = (
            f"Break the following research question into focused sub-questions.\n\n"
            f"Question: {state.question}"
        )
        result = llm_planner.invoke(prompt)
        return {"sub_questions": result.sub_questions}

    def fan_out_to_researchers(state: ResearchState):
        return [Send("researcher", {"sub_question": sq}) for sq in state.sub_questions]

    def researcher(payload: dict) -> dict:
        finding = research_one(payload["sub_question"])
        return {"findings": [finding]}

    def critic(state: ResearchState) -> dict:
        findings_text = "\n\n".join(
            f"Sub-question: {f.sub_question}\nAnswer: {f.answer}\nSources: {f.sources}\nis_grounded: {f.is_grounded}"
            for f in state.findings
        )
        prompt = (
            f"Review the following research findings. For each one, judge whether it is well-supported "
            f"and actually answers its sub-question, or if it's thin/unsupported. "
            f"IMPORTANT: if is_grounded is False, you MUST mark is_well_supported as False for that finding, "
            f"regardless of how complete the answer text sounds — an ungrounded finding has no real sources. "
            f"Also check whether any findings contradict each other.\n\n{findings_text}"
        )
        report = llm_critic.invoke(prompt)
        return {"critic_report": report}

    def enforce_grounding_rule(state: ResearchState) -> dict:
        report = state.critic_report
        findings_by_question = {f.sub_question: f for f in state.findings}
        corrected = []
        for v in report.verifications:
            finding = findings_by_question.get(v.sub_question)
            if finding is not None and not finding.is_grounded and v.is_well_supported:
                corrected.append(v.model_copy(update={
                    "is_well_supported": False,
                    "issue": "Overridden: finding has no sources (is_grounded=False), cannot be well-supported regardless of critic's text judgment."
                }))
            else:
                corrected.append(v)
        return {"critic_report": report.model_copy(update={"verifications": corrected})}

    def writer(state: ResearchState) -> dict:
        findings_text = "\n\n".join(
            f"Sub-question: {f.sub_question}\nAnswer: {f.answer}\nSources: {', '.join(f.sources) if f.sources else 'None'}"
            for f in state.findings
        )
        verification_text = "\n".join(
            f"- {v.sub_question}: {'OK' if v.is_well_supported else 'FLAGGED - ' + v.issue}"
            for v in state.critic_report.verifications
        )
        contradiction_text = (
            "\n".join(f"- {c}" for c in state.critic_report.contradictions)
            if state.critic_report.contradictions else "None found."
        )
        prompt = (
            f"Write a coherent, well-organized research report answering this question:\n{state.question}\n\n"
            f"Base the report on these findings:\n{findings_text}\n\n"
            f"Critic's verification results (per finding):\n{verification_text}\n\n"
            f"Critic's flagged contradictions between findings:\n{contradiction_text}\n\n"
            f"IMPORTANT: If any finding was flagged as not well-supported, or any contradiction was found, "
            f"you MUST transparently mention this in the report at the relevant point — do not silently omit, "
            f"hide, or resolve it yourself. Include source URLs where relevant. Write in clear, professional "
            f"prose with section headers."
        )
        report_text = extract_text(llm.invoke(prompt).content)
        return {"report": report_text}

    builder = StateGraph(ResearchState)
    builder.add_node("planner", planner)
    builder.add_node("researcher", researcher)
    builder.add_node("critic", critic)
    builder.add_node("enforce_grounding_rule", enforce_grounding_rule)
    builder.add_node("writer", writer)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", fan_out_to_researchers, ["researcher"])
    builder.add_edge("researcher", "critic")
    builder.add_edge("critic", "enforce_grounding_rule")
    builder.add_edge("enforce_grounding_rule", "writer")
    builder.add_edge("writer", END)

    return builder.compile()

# ---------------------------------------------------------------------------
# 5. STREAMLIT UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Multi-Agent Deep Research Assistant", layout="wide")
st.title("🔎 Multi-Agent Deep Research Assistant")
st.caption("Planner → Parallel Researchers → Critic → Writer, built with LangGraph")

if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "question_input" not in st.session_state:
    st.session_state.question_input = ""

question = st.text_area(
    "Enter a research question:",
    height=100,
    placeholder="e.g. Compare EV strategies of Tesla, BYD, and Tata Motors",
    key="question_input"
)

col1, col2 = st.columns([1, 5])
with col1:
    run_clicked = st.button("Run Research", type="primary")
with col2:
    if st.button("New Question"):
        st.session_state.last_result = None
        st.session_state.question_input = ""
        st.rerun()

if run_clicked:
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        graph = build_graph()
        with st.spinner("Planning, researching, verifying, and writing..."):
            try:
                result = graph.invoke({"user_input": question, "question": question})
                st.session_state.last_result = result
            except Exception as e:
                st.error(f"Something went wrong: {e}")

if st.session_state.last_result:
    result = st.session_state.last_result
    st.markdown(result["report"])
    with st.expander("See critic's verification details"):
        for v in result["critic_report"].verifications:
            status = "✅ OK" if v.is_well_supported else f"⚠️ FLAGGED — {v.issue}"
            st.write(f"**{v.sub_question}**: {status}")
        if result["critic_report"].contradictions:
            st.write("**Contradictions found:**")
            for c in result["critic_report"].contradictions:
                st.write(f"- {c}")
