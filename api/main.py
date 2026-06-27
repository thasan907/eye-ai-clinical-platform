"""
api/main.py  —  Eye AI Backend
================================
Vision-only mode: uses trained EfficientNet-B4 model directly.
No OpenAI key required. Rule-based clinical reports.
"""

import io, os
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import cv2

from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from loguru import logger

# ── Internal imports ─────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MAX_UPLOAD_BYTES, SEVERITY_MAP, DISEASE_LABELS,
    MODEL_PATH, IMAGE_SIZE, CONFIDENCE_THRESHOLD
)
from utils.database import init_db, get_db, ScanRecord, Patient
from utils.preprocessing import (
    load_image_from_bytes, preprocess_for_inference, check_image_quality
)

# ── Lazy model loader ─────────────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is not None:
        return _model

    model_path = Path(MODEL_PATH)

    if model_path.exists():
        logger.info(f"Loading trained model from {model_path}")
        try:
            import timm, torch.nn as nn

            class EyeModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.backbone = timm.create_model(
                        "efficientnet_b4", pretrained=False,
                        num_classes=0, global_pool=""
                    )
                    in_f = self.backbone.num_features
                    self.gem = nn.AdaptiveAvgPool2d(1)
                    self.head = nn.Sequential(
                        nn.Flatten(),
                        nn.BatchNorm1d(in_f),
                        nn.Dropout(0.4),
                        nn.Linear(in_f, 512),
                        nn.SiLU(),
                        nn.BatchNorm1d(512),
                        nn.Dropout(0.2),
                        nn.Linear(512, 5),
                    )
                def forward(self, x):
                    return self.head(self.gem(self.backbone(x)))

            m = EyeModel()
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            m.load_state_dict(state)
            m.eval()
            _model = m
            logger.success("Model loaded successfully")
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            _model = "demo"
    else:
        logger.warning(f"No model at {model_path} — demo mode")
        _model = "demo"

    return _model


# ── Clinical rule engine ──────────────────────────────────────────
CLINICAL_REPORTS = {
    0: {  # No DR
        "image_report": (
            "Retinal examination reveals a healthy fundus appearance. "
            "The optic disc margins are sharp and well-defined. No microaneurysms, "
            "haemorrhages, exudates, or neovascularisation detected. "
            "Macular region appears normal with no oedema present."
        ),
        "diagnosis_report": (
            "No evidence of diabetic retinopathy at this examination. "
            "The retinal vasculature appears normal with appropriate arteriovenous ratio. "
            "Continued glycaemic and blood pressure control is recommended to maintain "
            "retinal health and prevent future complications."
        ),
        "trend_report": (
            "Retinal status is stable with no signs of diabetic eye disease. "
            "Annual screening is appropriate for this patient. "
            "Maintain current management strategy and ensure HbA1c targets are met."
        ),
        "patient_summary": (
            "Good news — your retinal scan looks healthy. There are no signs of "
            "diabetic eye disease at this time. This means the blood vessels in your "
            "eye are in good condition. Keep up with your regular check-ups, maintain "
            "your blood sugar and blood pressure targets, and come back for your "
            "annual eye screening."
        ),
    },
    1: {  # Mild DR
        "image_report": (
            "Early signs of diabetic retinopathy detected. Scattered microaneurysms "
            "are present, appearing as small red dots in the retinal periphery. "
            "No significant haemorrhages, hard exudates, or cotton wool spots identified. "
            "No macular oedema or neovascularisation at this stage."
        ),
        "diagnosis_report": (
            "Mild non-proliferative diabetic retinopathy (NPDR) confirmed. "
            "This represents the earliest clinical stage of diabetic eye disease. "
            "The presence of microaneurysms indicates early microvascular damage. "
            "Optimising glycaemic control and blood pressure management is critical "
            "to prevent progression to more advanced stages."
        ),
        "trend_report": (
            "Early-stage disease identified. Monitoring frequency should be increased "
            "to 6-12 months depending on systemic risk factors. "
            "If HbA1c is above target, endocrinology referral should be considered. "
            "Document baseline for future longitudinal comparison."
        ),
        "patient_summary": (
            "Your scan shows very early signs of diabetic eye disease. "
            "Tiny swellings called microaneurysms have appeared in the blood vessels "
            "of your retina — this is the earliest stage and is manageable. "
            "The most important thing you can do is keep your blood sugar "
            "and blood pressure under tight control. We recommend a follow-up "
            "eye appointment in 6 to 12 months."
        ),
    },
    2: {  # Moderate DR
        "image_report": (
            "Moderate non-proliferative diabetic retinopathy detected. "
            "Multiple microaneurysms and dot/blot haemorrhages are visible. "
            "Hard exudates present in the posterior pole, indicating lipid leakage "
            "from damaged capillaries. No high-risk proliferative features identified. "
            "Macular assessment recommended to rule out clinically significant macular oedema."
        ),
        "diagnosis_report": (
            "Moderate NPDR represents significant microvascular compromise. "
            "The combination of haemorrhages and exudates suggests progressive "
            "retinal ischaemia. There is increased risk of progression to "
            "severe NPDR or proliferative DR within 1-3 years without intervention. "
            "Urgent optimisation of diabetes management is required. "
            "Ophthalmology referral is strongly recommended."
        ),
        "trend_report": (
            "Disease has progressed beyond early stage. Close monitoring at "
            "3-6 month intervals is indicated. If this is a follow-up scan showing "
            "progression from mild DR, treatment escalation should be discussed "
            "with the patient's diabetologist. Consider OCT imaging for macular assessment."
        ),
        "patient_summary": (
            "Your retinal scan shows moderate diabetic eye disease. "
            "The blood vessels in your eye are showing signs of damage — "
            "some are leaking fluid and there are small areas of bleeding. "
            "This was caught at a stage where treatment can still be very effective. "
            "Your doctor will likely refer you to an eye specialist. "
            "Please attend this appointment promptly. Controlling your blood sugar, "
            "blood pressure, and cholesterol is essential to protect your sight."
        ),
    },
    3: {  # Severe DR
        "image_report": (
            "Severe non-proliferative diabetic retinopathy identified — "
            "4-2-1 rule criteria are met. Extensive intraretinal haemorrhages "
            "in all four quadrants, venous beading in two or more quadrants, "
            "and intraretinal microvascular abnormalities (IRMA) are present. "
            "High-risk features for progression to proliferative DR within 12 months. "
            "Urgent neovascularisation screening required."
        ),
        "diagnosis_report": (
            "Severe NPDR carries a greater than 50% risk of progression to "
            "proliferative diabetic retinopathy within one year without treatment. "
            "Immediate ophthalmology referral is mandatory. "
            "Pan-retinal photocoagulation (PRP) or anti-VEGF therapy may be indicated "
            "depending on OCT and fluorescein angiography findings. "
            "Systemic diabetes review is urgently required."
        ),
        "trend_report": (
            "URGENT: Significant disease burden. This patient requires "
            "immediate ophthalmology review within 2-4 weeks. "
            "Monthly monitoring until stabilised post-treatment. "
            "Coordinate with endocrinology for intensive glycaemic management. "
            "Bilateral assessment essential — contralateral eye must be imaged."
        ),
        "patient_summary": (
            "Your retinal scan shows advanced diabetic eye disease that needs "
            "urgent attention. There are significant changes to the blood vessels "
            "in your eye that put your vision at risk if not treated promptly. "
            "You need to see an eye specialist (ophthalmologist) within the "
            "next 2 to 4 weeks — please do not delay this. "
            "Your doctor will arrange this referral for you. "
            "It is very important to attend this appointment."
        ),
    },
    4: {  # Proliferative DR
        "image_report": (
            "Proliferative diabetic retinopathy confirmed — sight-threatening stage. "
            "New vessel formation (neovascularisation) identified at the disc (NVD) "
            "and/or elsewhere in the retina (NVE). "
            "Pre-retinal or vitreous haemorrhage may be present. "
            "Fibrovascular proliferation with risk of tractional retinal detachment. "
            "This is a clinical emergency requiring same-day or next-day specialist review."
        ),
        "diagnosis_report": (
            "Proliferative diabetic retinopathy is the most advanced and "
            "vision-threatening stage of diabetic eye disease. "
            "New abnormal blood vessels have grown on the retinal surface — "
            "these are fragile and prone to haemorrhage. "
            "Without urgent laser treatment or intravitreal anti-VEGF injection, "
            "permanent vision loss is likely. "
            "IMMEDIATE ophthalmology referral — same or next working day."
        ),
        "trend_report": (
            "EMERGENCY: Proliferative DR requires same-day or next-day "
            "ophthalmology review. Do not defer. "
            "Treatment is urgent — pan-retinal photocoagulation or anti-VEGF "
            "must be initiated immediately. "
            "Admit if vitreous haemorrhage or retinal detachment is suspected. "
            "Alert the duty ophthalmologist immediately."
        ),
        "patient_summary": (
            "URGENT: Your retinal scan has detected a serious condition called "
            "proliferative diabetic retinopathy. New abnormal blood vessels have "
            "grown in your eye and your vision is at significant risk. "
            "You need to be seen by an eye doctor TODAY or TOMORROW — "
            "this cannot wait. Please contact your doctor immediately "
            "or go to your nearest eye casualty department. "
            "Treatment is available and can preserve your sight if started promptly."
        ),
    },
}


def build_report(label: str, confidence: float, severity: dict, patient_info: dict) -> dict:
    """Generate rule-based clinical report — no API key needed."""
    sev_level = severity.get("level", 0)
    reports   = CLINICAL_REPORTS.get(sev_level, CLINICAL_REPORTS[0])

    # Personalise with patient data if available
    age       = patient_info.get("age")
    diabetic  = patient_info.get("diabetic", False)
    hba1c     = patient_info.get("hba1c")

    extra = ""
    if hba1c and float(hba1c) > 8.0:
        extra = f" The patient's HbA1c of {hba1c}% is above target, which significantly increases the risk of retinopathy progression."
    if age and int(age) > 60:
        extra += " Age over 60 is an additional risk factor for advanced retinal complications."

    image_report    = reports["image_report"]
    diagnosis_report= reports["diagnosis_report"] + extra
    trend_report    = reports["trend_report"]
    patient_summary = reports["patient_summary"]

    # Confidence disclaimer
    if confidence < CONFIDENCE_THRESHOLD:
        image_report += (
            f" Note: Model confidence is {confidence*100:.0f}% — "
            "below the 75% threshold. Manual review by a clinician is recommended."
        )

    return {
        "image_report":     image_report,
        "diagnosis_report": diagnosis_report,
        "trend_report":     trend_report,
        "patient_summary":  patient_summary,
    }


# ── FastAPI app ───────────────────────────────────────────────────
app = FastAPI(
    title       = "Eye AI API",
    description = "Clinical retinal screening — EfficientNet-B4 · Vision-only mode",
    version     = "2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


@app.on_event("startup")
async def startup():
    await init_db()
    get_model()   # Pre-load model on startup
    logger.success("Eye AI API v2.0 started — vision-only mode, no API key required")


# ── Health ────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    m = get_model()
    return {
        "status":    "ok",
        "mode":      "vision" if m != "demo" else "demo",
        "model":     "EfficientNet-B4" if m != "demo" else "not loaded",
        "timestamp": datetime.utcnow().isoformat(),
    }


# ── Patient ───────────────────────────────────────────────────────
class PatientCreate(BaseModel):
    patient_id: str
    name:       str
    age:        int
    diabetic:   bool  = False
    hba1c:      float | None = None
    notes:      str   | None = None


@app.post("/patient")
async def create_patient(data: PatientCreate, db: AsyncSession = Depends(get_db)):
    ex = await db.execute(select(Patient).where(Patient.patient_id == data.patient_id))
    if ex.scalar_one_or_none():
        raise HTTPException(409, "Patient ID already exists")
    db.add(Patient(**data.model_dump()))
    await db.commit()
    return {"message": "Patient registered", "patient_id": data.patient_id}


# ── Main scan endpoint ────────────────────────────────────────────
@app.post("/scan")
async def analyze_scan(
    file:       UploadFile       = File(...),
    patient_id: str              = Form(...),
    eye_side:   str              = Form(...),
    age:        str              = Form(None),
    hba1c:      str              = Form(None),
    diabetic:   str              = Form("false"),
    notes:      str              = Form(None),
    db:         AsyncSession     = Depends(get_db),
):
    # ── Validate ─────────────────────────────────────────────────
    if file.content_type not in ("image/jpeg","image/png","image/jpg"):
        raise HTTPException(400, "Only JPG/PNG images accepted")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image too large — max 10 MB")

    # ── Load image ────────────────────────────────────────────────
    try:
        image = load_image_from_bytes(raw)
    except ValueError as e:
        raise HTTPException(400, str(e))

    quality = check_image_quality(image)
    logger.info(f"[{patient_id}] Quality: {quality}")

    # ── Run model ─────────────────────────────────────────────────
    model = get_model()

    if model != "demo":
        tensor = preprocess_for_inference(image)
        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]

        class_index = int(probs.argmax())
        confidence  = float(probs[class_index])
        label       = DISEASE_LABELS[class_index]
        all_scores  = {DISEASE_LABELS[i]: round(float(probs[i]),4) for i in range(5)}
    else:
        # Demo fallback
        class_index = 2
        confidence  = 0.83
        label       = DISEASE_LABELS[2]
        all_scores  = {"No DR":.03,"Mild DR":.07,"Moderate DR":.83,"Severe DR":.05,"Proliferative DR":.02}

    logger.info(f"[{patient_id}] → {label} ({confidence*100:.1f}%)")

    # ── Severity ──────────────────────────────────────────────────
    severity = SEVERITY_MAP.get(label, {"level":0,"action":"See your doctor"})

    # ── Patient info ──────────────────────────────────────────────
    patient_info = {
        "age":      age,
        "diabetic": diabetic == "true",
        "hba1c":    hba1c,
        "notes":    notes,
    }

    # ── Build clinical report (no API key needed) ─────────────────
    report = build_report(label, confidence, severity, patient_info)

    # ── Alert logic ───────────────────────────────────────────────
    sev_level     = severity.get("level", 0)
    urgent        = sev_level >= 3
    alert_message = ""
    if sev_level >= 4:
        alert_message = (
            f"EMERGENCY — Proliferative DR detected in {patient_id} "
            f"({eye_side} eye, {confidence*100:.0f}% confidence). "
            "Same-day ophthalmology review required."
        )
    elif sev_level == 3:
        alert_message = (
            f"URGENT — Severe DR detected in {patient_id} "
            f"({eye_side} eye, {confidence*100:.0f}% confidence). "
            "Ophthalmology referral within 2-4 weeks required."
        )
    elif confidence < CONFIDENCE_THRESHOLD and sev_level >= 2:
        urgent        = True
        alert_message = (
            f"LOW CONFIDENCE — Model confidence {confidence*100:.0f}% below threshold. "
            "Manual clinical review recommended."
        )

    # ── Save to DB ────────────────────────────────────────────────
    scan = ScanRecord(
        patient_id         = patient_id,
        image_filename     = file.filename,
        eye_side           = eye_side,
        predicted_label    = label,
        confidence         = confidence,
        severity_level     = sev_level,
        recommended_action = severity.get("action",""),
        quality_ok         = quality["ok"],
        quality_reason     = quality["reason"],
        agent_summary      = report["patient_summary"],
        flagged_urgent     = urgent,
    )
    db.add(scan)
    await db.commit()
    await db.refresh(scan)

    # ── Response ──────────────────────────────────────────────────
    return {
        "scan_id":          scan.id,
        "patient_id":       patient_id,
        "eye_side":         eye_side,
        "predicted_label":  label,
        "confidence":       round(confidence, 4),
        "severity_level":   sev_level,
        "action":           severity.get("action",""),
        "urgent":           urgent,
        "alert_message":    alert_message,
        "all_scores":       all_scores,
        "image_report":     report["image_report"],
        "diagnosis_report": report["diagnosis_report"],
        "trend_report":     report["trend_report"],
        "patient_summary":  report["patient_summary"],
        "quality_ok":       quality["ok"],
        "quality_reason":   quality["reason"],
        "created_at":       scan.created_at.isoformat(),
        "model":            "EfficientNet-B4" if model != "demo" else "demo",
    }


# ── History ───────────────────────────────────────────────────────
@app.get("/scan/{scan_id}")
async def get_scan(scan_id: int, db: AsyncSession = Depends(get_db)):
    row  = await db.execute(select(ScanRecord).where(ScanRecord.id == scan_id))
    scan = row.scalar_one_or_none()
    if not scan:
        raise HTTPException(404, "Scan not found")
    return scan


@app.get("/patient/{patient_id}/scans")
async def get_patient_scans(patient_id: str, db: AsyncSession = Depends(get_db)):
    rows  = await db.execute(
        select(ScanRecord)
        .where(ScanRecord.patient_id == patient_id)
        .order_by(ScanRecord.created_at.desc())
    )
    scans = rows.scalars().all()
    return {"patient_id": patient_id, "total": len(scans), "scans": scans}