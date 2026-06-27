"""
agents/eye_agents.py
Multi-agent system built with LangGraph.
Five specialist agents coordinated by an orchestrator.
"""

from __future__ import annotations
from typing import TypedDict, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from loguru import logger
from config import OPENAI_API_KEY, SEVERITY_MAP, CONFIDENCE_THRESHOLD

# ── Shared state that flows between agents ───────────────────────

class EyeAnalysisState(TypedDict):
    # Inputs
    patient_id:   str
    eye_side:     str
    patient_info: dict          # age, diabetic, hba1c, etc.

    # From vision model
    predicted_label: str
    confidence:      float
    all_scores:      dict
    quality_check:   dict

    # Agent outputs (accumulated)
    image_report:     str
    diagnosis_report: str
    trend_report:     str
    alert_triggered:  bool
    alert_message:    str
    patient_summary:  str

    # Final
    final_report: str
    messages:     Annotated[list, operator.add]


# ── LLM setup ────────────────────────────────────────────────────

def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        api_key=OPENAI_API_KEY,
    )


# ══════════════════════════════════════════════════════════════════
# AGENT 1 — Image Analysis Agent
# Interprets raw model output into clinical language
# ══════════════════════════════════════════════════════════════════

def image_analysis_agent(state: EyeAnalysisState) -> dict:
    logger.info("Agent 1: Image Analysis running...")
    llm = get_llm()

    scores_str = "\n".join(
        f"  {label}: {score*100:.1f}%"
        for label, score in state["all_scores"].items()
    )

    messages = [
        SystemMessage(content=(
            "You are a medical AI image analysis specialist. "
            "Interpret retinal scan AI results in clear clinical language. "
            "Be precise but avoid unnecessary jargon. Keep it to 3-4 sentences."
        )),
        HumanMessage(content=(
            f"Retinal scan result for {state['eye_side']} eye:\n"
            f"Primary diagnosis: {state['predicted_label']}\n"
            f"Confidence: {state['confidence']*100:.1f}%\n"
            f"All class probabilities:\n{scores_str}\n\n"
            f"Image quality: {state['quality_check'].get('reason', 'unknown')}\n\n"
            "Provide a concise clinical interpretation of these AI findings."
        )),
    ]

    response = llm.invoke(messages)
    return {"image_report": response.content,
            "messages": [{"agent": "image_analysis", "content": response.content}]}


# ══════════════════════════════════════════════════════════════════
# AGENT 2 — Diagnosis Agent
# Combines AI result with patient history
# ══════════════════════════════════════════════════════════════════

def diagnosis_agent(state: EyeAnalysisState) -> dict:
    logger.info("Agent 2: Diagnosis running...")
    llm = get_llm()
    info = state["patient_info"]
    severity = SEVERITY_MAP.get(state["predicted_label"], {})

    messages = [
        SystemMessage(content=(
            "You are a clinical diagnosis AI assistant specializing in diabetic eye disease. "
            "Combine AI scan results with patient history to provide a holistic diagnosis assessment. "
            "Be specific, clinical, and evidence-based. 4-5 sentences."
        )),
        HumanMessage(content=(
            f"Patient profile:\n"
            f"  Age: {info.get('age', 'unknown')}\n"
            f"  Diabetic: {info.get('diabetic', 'unknown')}\n"
            f"  HbA1c: {info.get('hba1c', 'unknown')}%\n"
            f"  Medical notes: {info.get('notes', 'none')}\n\n"
            f"AI scan finding: {state['predicted_label']} "
            f"(confidence {state['confidence']*100:.1f}%)\n"
            f"Severity level: {severity.get('level', '?')}/4\n"
            f"Recommended action: {severity.get('action', 'unknown')}\n\n"
            "Provide a combined diagnosis assessment considering both the scan and patient risk factors."
        )),
    ]

    response = llm.invoke(messages)
    return {"diagnosis_report": response.content,
            "messages": [{"agent": "diagnosis", "content": response.content}]}


# ══════════════════════════════════════════════════════════════════
# AGENT 3 — Trend / Monitoring Agent
# Analyzes progression over time
# ══════════════════════════════════════════════════════════════════

def trend_agent(state: EyeAnalysisState) -> dict:
    logger.info("Agent 3: Trend analysis running...")
    llm = get_llm()
    info = state["patient_info"]
    previous_scans = info.get("previous_scans", [])

    if previous_scans:
        history_str = "\n".join(
            f"  {s['date']}: {s['label']} (confidence {s['confidence']*100:.1f}%)"
            for s in previous_scans[-5:]   # Last 5 scans
        )
    else:
        history_str = "  No previous scans on record (first visit)"

    messages = [
        SystemMessage(content=(
            "You are a longitudinal disease monitoring AI. "
            "Analyze trends in diabetic retinopathy progression over time. "
            "Identify if the condition is stable, improving, or worsening. "
            "Give specific, actionable monitoring guidance. 3-4 sentences."
        )),
        HumanMessage(content=(
            f"Patient: {state['patient_id']}\n"
            f"Previous scan history:\n{history_str}\n\n"
            f"Current scan: {state['predicted_label']} "
            f"({state['confidence']*100:.1f}% confidence)\n\n"
            "Analyze the trend and provide monitoring recommendations."
        )),
    ]

    response = llm.invoke(messages)
    return {"trend_report": response.content,
            "messages": [{"agent": "trend", "content": response.content}]}


# ══════════════════════════════════════════════════════════════════
# AGENT 4 — Alert Agent
# Decides if urgent escalation is needed
# ══════════════════════════════════════════════════════════════════

def alert_agent(state: EyeAnalysisState) -> dict:
    logger.info("Agent 4: Alert check running...")
    severity = SEVERITY_MAP.get(state["predicted_label"], {})
    severity_level = severity.get("level", 0)
    low_confidence = state["confidence"] < CONFIDENCE_THRESHOLD

    alert_triggered = False
    alert_message   = ""

    # Rule-based alert triggers
    if severity_level >= 3:
        alert_triggered = True
        alert_message = (
            f"URGENT ALERT — Patient {state['patient_id']}: "
            f"{state['predicted_label']} detected in {state['eye_side']} eye "
            f"(confidence: {state['confidence']*100:.1f}%). "
            f"Immediate ophthalmologist referral required."
        )
    elif low_confidence and severity_level >= 2:
        alert_triggered = True
        alert_message = (
            f"REVIEW REQUIRED — Patient {state['patient_id']}: "
            f"Moderate DR suspected but confidence is low ({state['confidence']*100:.1f}%). "
            f"Manual review by clinician recommended."
        )
    elif not state["quality_check"].get("ok", True):
        alert_triggered = True
        alert_message = (
            f"IMAGE QUALITY ALERT — Patient {state['patient_id']}: "
            f"{state['quality_check'].get('reason', 'Quality issue')}. "
            f"Please re-take the scan."
        )

    if alert_triggered:
        logger.warning(f"ALERT: {alert_message}")

    return {"alert_triggered": alert_triggered, "alert_message": alert_message,
            "messages": [{"agent": "alert", "triggered": alert_triggered,
                          "message": alert_message}]}


# ══════════════════════════════════════════════════════════════════
# AGENT 5 — Report Agent
# Writes plain-English summary for patient + doctor
# ══════════════════════════════════════════════════════════════════

def report_agent(state: EyeAnalysisState) -> dict:
    logger.info("Agent 5: Report generation running...")
    llm = get_llm()
    severity = SEVERITY_MAP.get(state["predicted_label"], {})

    messages = [
        SystemMessage(content=(
            "You are a medical report writer. Create a clear, compassionate summary "
            "that both doctors and patients can understand. "
            "Structure: (1) What was found, (2) What it means, (3) What happens next. "
            "Use plain English. Avoid alarming language unless urgent action is needed. "
            "Keep it to 150-200 words."
        )),
        HumanMessage(content=(
            f"Compile a patient report from these agent findings:\n\n"
            f"IMAGE ANALYSIS:\n{state['image_report']}\n\n"
            f"DIAGNOSIS ASSESSMENT:\n{state['diagnosis_report']}\n\n"
            f"TREND ANALYSIS:\n{state['trend_report']}\n\n"
            f"RECOMMENDED ACTION: {severity.get('action', 'See your doctor')}\n"
            f"URGENT: {'Yes' if state['alert_triggered'] else 'No'}\n\n"
            "Write the final patient-facing report."
        )),
    ]

    response = llm.invoke(messages)

    # Build full structured report
    final_report = {
        "patient_id":       state["patient_id"],
        "eye_side":         state["eye_side"],
        "finding":          state["predicted_label"],
        "confidence":       f"{state['confidence']*100:.1f}%",
        "severity":         severity.get("level", 0),
        "action":           severity.get("action", ""),
        "urgent":           state["alert_triggered"],
        "alert_message":    state["alert_message"],
        "patient_summary":  response.content,
        "image_report":     state["image_report"],
        "diagnosis_report": state["diagnosis_report"],
        "trend_report":     state["trend_report"],
    }

    return {"patient_summary": response.content,
            "final_report": str(final_report),
            "messages": [{"agent": "report", "content": response.content}]}


# ══════════════════════════════════════════════════════════════════
# ORCHESTRATOR — LangGraph workflow
# ══════════════════════════════════════════════════════════════════

def build_eye_agent_graph() -> StateGraph:
    """Wire up all 5 agents into a LangGraph workflow."""
    graph = StateGraph(EyeAnalysisState)

    graph.add_node("image_analysis", image_analysis_agent)
    graph.add_node("diagnosis",      diagnosis_agent)
    graph.add_node("trend",          trend_agent)
    graph.add_node("alert",          alert_agent)
    graph.add_node("report",         report_agent)

    # Linear pipeline: image → diagnosis → trend → alert → report
    graph.set_entry_point("image_analysis")
    graph.add_edge("image_analysis", "diagnosis")
    graph.add_edge("diagnosis",      "trend")
    graph.add_edge("trend",          "alert")
    graph.add_edge("alert",          "report")
    graph.add_edge("report",         END)

    return graph.compile()


# ── Public API ───────────────────────────────────────────────────

async def run_eye_analysis(
    patient_id:    str,
    eye_side:      str,
    patient_info:  dict,
    model_result:  dict,
    quality_check: dict,
) -> dict:
    """
    Run the full 5-agent pipeline.
    Returns the compiled final report dict.
    """
    graph = build_eye_agent_graph()

    initial_state: EyeAnalysisState = {
        "patient_id":      patient_id,
        "eye_side":        eye_side,
        "patient_info":    patient_info,
        "predicted_label": model_result["label"],
        "confidence":      model_result["confidence"],
        "all_scores":      model_result["all_scores"],
        "quality_check":   quality_check,
        "image_report":    "",
        "diagnosis_report": "",
        "trend_report":    "",
        "alert_triggered": False,
        "alert_message":   "",
        "patient_summary": "",
        "final_report":    "",
        "messages":        [],
    }

    result = await graph.ainvoke(initial_state)
    return result
