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
# 1. API KEYS
# ---------------------------------------------------------------------------
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]


# ---------------------------------------------------------------------------
# 2. SCHEMAS
# ---------------------------------------------------------------------------
class Finding(BaseModel):
    sub_question: str
    answer: str = Field(
        description="A thorough, detailed answer to the sub-question, covering all relevant facts, "
                    "figures, and context found in the search results. Do not omit relevant details "
                    "for the sake of brevity."
    )
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


class AmbiguityCheck(BaseModel):
    is_ambiguous: bool = Field(description="True only if the input has multiple plausible, meaningfully different real-world interpretations, such that guessing wrong would produce a useless report")
    clarifying_question: str = Field(default="", description="If ambiguous, one short, specific question to resolve it. Empty string otherwise.")


class QuestionCheck(BaseModel):
    is_research_question: bool = Field(description="True if this is genuinely asking to research/investigate/compare something. False if it's a statement, instruction, suggestion, or greeting that isn't asking for research.")


# ---------------------------------------------------------------------------
# 3. HELPERS
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


@st.cache_resource
def get_question_checker():
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)
    return llm.with_structured_output(QuestionCheck)


def check_is_question(user_input: str) -> QuestionCheck:
    checker = get_question_checker()
    prompt = (
        f"A user typed this into a research assistant: \"{user_input}\"\n\n"
        f"Decide if this is genuinely a request to research, investigate, compare, or find information about "
        f"something. It is NOT a research question if it's a statement of fact/preference, an instruction "
        f"(e.g. 'skip that', 'focus on X instead'), a suggestion, a greeting, or general chat."
    )
    return checker.invoke(prompt)


@st.cache_resource
def get_ambiguity_checker():
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)
    return llm.with_structured_output(AmbiguityCheck)


def check_ambiguity(user_input: str) -> AmbiguityCheck:
    checker = get_ambiguity_checker()
    prompt = (
        f"A user submitted this research request: \"{user_input}\"\n\n"
        f"Decide if this is genuinely ambiguous — i.e. it contains a name, acronym, or term with multiple "
        f"plausible, meaningfully different real-world meanings, such that researching the wrong one would "
        f"produce a useless report. Only flag it if there's a real risk of misinterpretation, not for every "
        f"vague-sounding question. If ambiguous, propose one short, specific clarifying question."
    )
    return checker.invoke(prompt)


# ---------------------------------------------------------------------------
# 4. GRAPH CONSTRUCTION
# ---------------------------------------------------------------------------
@st.cache_resource
def build_graph():
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)
    llm_planner = llm.with_structured_output(SubQuestions)
    llm_researcher = llm.with_structured_output(Finding)
    llm_critic = llm.with_structured_output(CriticReport)
    search_tool = TavilySearch(max_results=8)

    def research_one(sub_question: str) -> Finding:
        search_results = search_tool.invoke(sub_question)
        prompt = (
            f"Sub-question: {sub_question}\n\n"
            f"Search results:\n{search_results}\n\n"
            f"Based only on these search results, provide a thorough, detailed answer covering all "
            f"relevant facts and figures, with source URLs. Do not compress or omit relevant details."
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
        grounded_findings = [f for f in state.findings if f.is_grounded]
        ungrounded_findings = [f for f in state.findings if not f.is_grounded]

        findings_text = "\n\n".join(
            f"Sub-question: {f.sub_question}\nAnswer: {f.answer}\nSources: {', '.join(f.sources) if f.sources else 'None'}"
            for f in grounded_findings
        )
        verification_text = "\n".join(
            f"- {v.sub_question}: {'OK' if v.is_well_supported else 'FLAGGED - ' + v.issue}"
            for v in state.critic_report.verifications
        )
        contradiction_text = (
            "\n".join(f"- {c}" for c in state.critic_report.contradictions)
            if state.critic_report.contradictions else "None found."
        )
        unresearched_text = (
            "\n".join(f"- {f.sub_question}" for f in ungrounded_findings)
            if ungrounded_findings else "None."
        )

        prompt = (
            f"Write a coherent, THOROUGH, in-depth research report answering this question:\n{state.question}\n\n"
            f"Base the report ONLY on these well-grounded findings, giving each its own detailed section "
            f"(aim for roughly 150-250 words per section, more if the finding supports it — do not summarize "
            f"away relevant facts, figures, or context):\n{findings_text}\n\n"
            f"Critic's verification results (per finding):\n{verification_text}\n\n"
            f"Critic's flagged contradictions between findings:\n{contradiction_text}\n\n"
            f"The following sub-questions could NOT be researched due to lack of usable search results:\n{unresearched_text}\n\n"
            f"IMPORTANT: Do not create a full section for any sub-question with no real findings. Instead, "
            f"end the report with a brief 'Areas Not Covered' section listing those sub-questions in one or two "
            f"lines each. If any INCLUDED finding was flagged as not well-supported, or a contradiction was found, "
            f"mention this transparently at the relevant point — do not silently omit, hide, or resolve it yourself. "
            f"If a finding was flagged because it answers a DIFFERENT or BROADER scope than its sub-question asked "
            f"for, do not present its content as if it directly answers that sub-question — instead, clearly "
            f"reframe that section as related context that was found instead of a direct answer, and briefly note "
            f"what the sub-question actually needed that wasn't found. "
            f"Include source URLs where relevant. Write in clear, professional prose with section headers."
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
if "input_version" not in st.session_state:
    st.session_state.input_version = 0
if "pending_clarification" not in st.session_state:
    st.session_state.pending_clarification = None

if st.session_state.pending_clarification:
    st.info(f"❓ {st.session_state.pending_clarification['clarifying_question']}")

question = st.text_area(
    "Enter a research question:" if not st.session_state.pending_clarification else "Your clarification:",
    height=100,
    placeholder="e.g. Compare EV strategies of Tesla, BYD, and Tata Motors",
    key=f"question_input_{st.session_state.input_version}"
)

col1, col2 = st.columns([1, 5])
with col1:
    run_clicked = st.button("Run Research", type="primary")
with col2:
    if st.button("New Question"):
        st.session_state.last_result = None
        st.session_state.pending_clarification = None
        st.session_state.input_version += 1
        st.rerun()

if run_clicked:
    if not question.strip():
        st.warning("Please enter something first.")
    elif st.session_state.pending_clarification:
        # This input answers a pending clarifying question — skip the guard/ambiguity checks
        # and go straight to research using the combined question.
        full_question = (
            f"{st.session_state.pending_clarification['original_question']} "
            f"(Clarification: {question.strip()})"
        )
        st.session_state.pending_clarification = None
        graph = build_graph()
        with st.spinner("Planning, researching, verifying, and writing..."):
            try:
                result = graph.invoke({"user_input": full_question, "question": full_question})
                st.session_state.last_result = result
                st.session_state.input_version += 1
            except Exception as e:
                st.error(f"Something went wrong: {e}")
        st.rerun()
    else:
        q_check = check_is_question(question.strip())
        if not q_check.is_research_question:
            st.warning(
                "This looks like a statement or instruction rather than a research question. "
                "Try rephrasing it as something to research — e.g. 'What is Tesla's EV strategy?'"
            )
        else:
            ambiguity = check_ambiguity(question.strip())
            if ambiguity.is_ambiguous:
                st.session_state.pending_clarification = {
                    "original_question": question.strip(),
                    "clarifying_question": ambiguity.clarifying_question,
                }
                st.session_state.input_version += 1
                st.rerun()
            else:
                graph = build_graph()
                with st.spinner("Planning, researching, verifying, and writing..."):
                    try:
                        result = graph.invoke({"user_input": question.strip(), "question": question.strip()})
                        st.session_state.last_result = result
                        st.session_state.input_version += 1
                    except Exception as e:
                        st.error(f"Something went wrong: {e}")
                st.rerun()

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
