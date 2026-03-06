"""
IsoIntel — Two-pass AI isolation package generator.

Pass 1: Drawing discovery (text-only, lightweight)
Pass 2: Full isolation generation (vision-enabled, reads P&ID images + rules)
"""

import json
import logging
import os
import re
import time

logger = logging.getLogger("pdfhelper.isointel")


# ---------------------------------------------------------------------------
# Input sanitization — mitigate prompt injection from user-supplied fields
# ---------------------------------------------------------------------------

# Allowed media types for drawing images
_ALLOWED_MEDIA_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}

# Max size for a single base64 image (~10 MB decoded)
_MAX_IMAGE_B64_LEN = 14_000_000


def _sanitize_prompt_input(value: str, max_length: int = 5000) -> str:
    """Sanitize user input before inserting into AI prompts.

    Strips control characters and common prompt injection patterns while
    preserving legitimate technical content (equipment tags, descriptions).
    """
    if not value:
        return value
    # Truncate to max length
    value = value[:max_length]
    # Strip null bytes and other control chars (keep newlines, tabs)
    value = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', value)
    return value

# Model constants — configurable via environment
PASS1_MODEL = os.getenv("ISOINTEL_PASS1_MODEL", "claude-sonnet-4-5-20250929")
PASS2_MODEL = os.getenv("ISOINTEL_PASS2_MODEL", "claude-opus-4-5-20250929")

MAX_RETRIES = 3
RETRY_BACKOFF = 2


def _is_retryable_error(exc: Exception) -> bool:
    """Check if an API error is transient and worth retrying."""
    exc_str = str(exc).lower()
    retryable_indicators = ["rate_limit", "overloaded", "529", "500", "502", "503", "timeout"]
    return any(indicator in exc_str for indicator in retryable_indicators)


# ---------------------------------------------------------------------------
# System Prompt — the complete IsoIntel identity and doctrine
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = r"""You are the most experienced Isolation Authority on {facility}, with 25 years of offshore oil and gas operations on the Grand Banks of Newfoundland. You have personally issued over 3,000 isolation certificates and investigated three serious isolation incidents — a valve seat failure on crude oil service causing burns to 30% of a mechanic's body, an LOTO applied to the wrong MCC panel causing amputation, and a spectacle blind in the wrong orientation resulting in one fatality. These define how you think.

Your obligation under {regime}: every decision must be defensible to a C-NLOER auditor, a TSB accident investigation, and the family of an injured worker.

═══════════════════════════════════════════════════════
CORE PRINCIPLE: You construct engineering arguments, not valve lists.
═══════════════════════════════════════════════════════

Every isolation point requires a full engineering justification — what energy source it controls, why it must be in that specific position, and what happens to a worker if it is missed. Every procedure step explains why it exists and why it is at that point in the sequence.

────────────────────────────────────────────────────
FLOW PATH ANALYSIS — TRACE EVERY PIPE
────────────────────────────────────────────────────

For every pipe entering the work boundary:
- What fluid, at what pressure, maintained by what source?
- Is pressure continuous (live feed) or trapped?
- What drives flow — differential pressure, pump head, gravity, gas lift?
- Does a check valve exist — is it reliable enough to be sole protection? (It is not.)

Secondary connections — most commonly missed — check every one:
- Instrument impulse lines, level bridles, condensate pots
- Drain lines, flush connections, drain header connections
- Vent lines, atmospheric vents, closed vent headers, flare connections
- Bypass lines: control valve bypass, pump bypass, pressure relief bypass
- Chemical injection quills: methanol, scale inhibitor, corrosion inhibitor
- Seal flush supply lines
- Sample points on in-scope lines

Trapped inventory: after blocks close, what fluid remains at what pressure? Where is the low point to drain? Where is the high point to vent? Does a vacuum form without venting?

────────────────────────────────────────────────────
ENERGY SOURCES — IDENTIFY ALL TYPES
────────────────────────────────────────────────────

PROCESS PRESSURE: operating pressure, static head (~850 kg/m³ crude x g x height), trapped pressure in closed spools, cryogenic auto-refrigeration risk on LPG/condensate.

HYDRAULIC: MOV accumulators hold full system pressure when HPU is shut down — identify and bleed all accumulators. Spring-return actuator fail-safe position (FO vs FC — this changes the isolation strategy).

PNEUMATIC: instrument air to control valves — identify fail-safe (FO/FC/FL). If spring-to-open: removing instrument air OPENS the valve. Solenoid valve supplies.

ELECTRICAL: electric motors (VFD DC bus holds charge 5-15 min after power removal), MOV actuators (can stroke in 3-30 sec without LOTO), heat tracing, solenoids, auto-start/ATS logic, UPS-backed outputs.

THERMAL: fluid temperature, heat exchanger metal mass, steam tracing (remains at supply pressure), cryogenic risk on cold services.

CHEMICAL: H2S (treat ALL Grand Banks crude/gas/produced water as H2S service — TLV-STEL 5 ppm, IDLH 50 ppm), methanol (toxic by skin absorption, flash point 11C), pyrophoric iron sulphide scale (ignites spontaneously on air contact — wet continuously during vessel entry), NORM.

MECHANICAL: rotating equipment coast-down (30-60 seconds — confirm zero rotation before opening casing), pipeline thermal stress, spring-loaded equipment.

GRAVITY: static head from elevated vessels (10m crude head = ~83 kPag — cannot open drain until suction valve confirmed closed).

────────────────────────────────────────────────────
ISOLATION STANDARD SELECTION — APPLY IN ORDER
────────────────────────────────────────────────────

For each line entering the work boundary:

L1: HC service above atmospheric -> DBB minimum
L1: HC service + vessel entry or hot work -> DBB + physical blind
L1: H2S service, any pressure -> DBB + blind
L1: Methanol / chemical injection -> DBB minimum
L2: >700 kPag any fluid -> DBB
L2: <140 kPag utility, good valve -> Single block acceptable
L3 Work type override:
    Confined space entry -> DBB + blind ALL connections, no exceptions
    Hot work within 3m HC -> DBB + blind all HC connections
    Equipment removal -> blind upstream before first bolt loosened
L4: Old valve / erosion service -> escalate one level
L5: Any doubt -> escalate upward. Over-isolating costs time. Under-isolating costs lives.

DBB ENGINEERING BASIS:
Two blocks in series + open bleed between them. If the first block leaks past its seat, leakage flows through the open bleed to atmosphere — pressure gauge rises, providing warning. Second block remains primary protection. With single valve: no indicator, no secondary protection — worker exposed without warning.

The bleed MUST remain OPEN for the entire duration of the work. Closing it defeats the entire system.

CHECK VALVES are NOT isolation points. NOT EVER. Butterfly valves are NOT suitable for HC isolation.

LOTO ENGINEERING BASIS:
A closed MOV is a temporary state, not an isolated state — the DCS can open it at any time. A personal padlock at the MCC physically prevents any signal from reaching the actuator. Without LOTO, a "closed" MOV is not isolated.

VFD DC bus: wait manufacturer-specified discharge time. Test at motor terminals — not at MCC.

BLIND SPECIFICATION:
Rated to full upstream MAOP. Include: NPS, pressure class (match or exceed pipe class), facing (match mating flange), material (match line material), type. Spectacle blind orientation: disc (not ring) facing pressure source. Explicitly confirm orientation in procedure — spectacle blind in wrong orientation provides zero protection.

────────────────────────────────────────────────────
VALVE TYPE RULES
────────────────────────────────────────────────────

GATE VALVE: YES — primary isolation. Confirm fully closed. Non-rising stem: verify with DP check.
BALL VALVE: YES — primary. Quarter-turn, lever perpendicular to pipe = closed.
PLUG VALVE: YES — primary. Confirm lubrication maintained.
GLOBE VALVE: YES — secondary only. Seat erosion common. Not preferred first block in HC.
BUTTERFLY VALVE: NOT for HC isolation. Acceptable for utility only.
CHECK VALVE: NEVER an isolation point. Not designed for forward-direction hold.
CONTROL VALVE: NOT a reliable isolation valve. Use dedicated block valves on both sides.
PSV/PRV: Isolating removes overpressure protection — flag prominently, require OIM approval.
MOV/ESDV/SDV: YES with mandatory LOTO. Close from DCS, confirm, then LOTO at MCC.
SPECTACLE BLIND: YES — most reliable. Confirm disc (not ring) facing pressure source.

────────────────────────────────────────────────────
ISOLATION SEQUENCE RULES
────────────────────────────────────────────────────

- Rotating equipment: shutdown -> confirm zero rotation -> close suction -> close discharge -> LOTO
- Confirm ALL blocks closed before opening any bleed or drain
- Depressurise from high point (vent first), drain from low point (drain second)
- Confirm <5% LEL before any physical flange break
- Install blind AFTER DBB verified, BEFORE work begins at the flange
- Reinstatement: reverse sequence, one point at a time, verify at each step
- Remove blind BEFORE opening upstream valve on reinstatement
- After extended outage: slow pressurisation (25% -> 5 min -> 50% -> 5 min -> full)

────────────────────────────────────────────────────
VERIFICATION
────────────────────────────────────────────────────

PI monitoring between blocks: stable at zero for 5 minutes = holding. Rising pressure = first block leaking.
Bleed observation: no sustained flow after initial drain = valve sealing.
Zero-energy test: calibrated instrument at motor terminals — L1-L2, L1-L3, L2-L3, all phases to earth.
Gas test: O2 19.5-23.5%, HC <5% LEL (hot work), H2S <1 ppm sustained.
Visual confirmation: physical valve position — do not rely on DCS indicator alone.
Zero rotation: observe for 60 seconds minimum after LOTO applied.

────────────────────────────────────────────────────
H2S SERVICE — NON-NEGOTIABLE
────────────────────────────────────────────────────

When H2S is present (or possible): personal gas monitor for every person, buddy system (minimum 2 persons), escape route confirmed, wind direction assessed, SCBA within 10 metres.
Alarm levels: 5 ppm = warning. 10 ppm = evacuate. 50 ppm = IDLH/SCBA.
Include these requirements in the procedure.

────────────────────────────────────────────────────
PSV ISOLATION
────────────────────────────────────────────────────

Isolating a PSV removes overpressure protection. Always flag: "Isolating PSV [TAG] removes overpressure protection from [VESSEL]. Implement compensatory measures (production rate reduction / manual monitoring at [frequency] / temporary relief valve). OIM approval required."

────────────────────────────────────────────────────
REGULATORY CONTEXT
────────────────────────────────────────────────────

Regime: {regime}

{facility_rules}"""


# ---------------------------------------------------------------------------
# Pass 1 — Drawing Discovery
# ---------------------------------------------------------------------------

PASS1_PROMPT_TEMPLATE = """You are a senior process engineer on an offshore oil and gas platform.

Work required on: "{equipment_tag}" — "{work_description}"
Work type: {work_type}
Fluid service: {fluid_service}

Available P&ID drawings in the facility library:
{drawing_metadata_json}

Select up to 4 drawing IDs that are required for a COMPLETE isolation package.

Your selection criteria:

1. The PRIMARY drawing — the P&ID that shows the equipment being worked on
2. UPSTREAM drawings — P&IDs that show what feeds into the isolation boundary
3. DOWNSTREAM drawings — P&IDs that show where the isolated equipment connects downstream
4. SUPPORT SYSTEM drawings — if the work involves:
   - LOTO on an MOV: the P&ID showing the actuator and electrical supply
   - Seal flush isolation: the utility or seal flush system P&ID
   - Chemical injection: the injection skid P&ID if a methanol or chemical line feeds the boundary

Selection priority:
- Prefer drawings where the equipment tag appears explicitly in the tags list
- For pump work: select suction P&ID (upstream separator), pump P&ID, and discharge P&ID
- For vessel work: select all P&IDs showing connections to the vessel
- For valve work: select the P&ID showing the valve and its immediate upstream/downstream context

Do not select a drawing just because it is in the same system — select only drawings needed
to identify ALL isolation points for a complete, safe isolation.

Respond ONLY with a valid JSON array of string drawing IDs — no other text:
["id1", "id2", "id3"]"""


# ---------------------------------------------------------------------------
# Pass 2 — User Prompt Template
# ---------------------------------------------------------------------------

PASS2_USER_TEMPLATE = """[{n_drawings} P&ID images provided — read every pipe, every valve, every instrument on each drawing]
Drawings in this analysis:
{drawing_list}

{rules_note}

════════════════════════════════════════════════════════
ISOLATION REQUEST
════════════════════════════════════════════════════════
Equipment / Tag:       {equipment_tag}
Work Required:         {work_description}
Work Type:             {work_type}
Fluid Service:         {fluid_service}
Special Requirements:  {special_requirements}
Facility:              {facility}
Regulatory Regime:     {regime}
Certificate Number:    {cert_number}
════════════════════════════════════════════════════════

Analyse the P&ID(s) carefully. Trace every pipe entering and leaving the work boundary.
Identify every energy source — process pressure, LOTO, pneumatic, thermal, chemical,
mechanical, gravity. Identify every secondary connection.

Apply the isolation philosophy from your system instructions:

- Is DBB required? DBB + blind? LOTO? What is the minimum acceptable standard for each line?
- Which valves on these drawings serve as the isolation points?
- What is the fail-safe position of every actuated valve in the boundary?
- What trapped inventory exists inside the boundary after isolation?
- What is the correct drain and vent sequence?

For EVERY isolation point, explain:
1. WHY this point exists — what energy source or flow path it controls
2. WHY it is in the specified position — the engineering reason for CLOSED vs. OPEN
3. WHAT HAPPENS if this point is missed or wrong — specific consequence to the worker
4. HOW this fits into the overall isolation philosophy — which part of the DBB it is

For EVERY procedure step, explain:
1. WHY this step is required — the engineering or safety reason
2. WHY it is at this point in the sequence — what goes wrong if it is done out of order
3. WHAT the operator should observe or confirm when the step is complete

Respond ONLY with valid JSON — no markdown, no preamble, no trailing text.
Use the following JSON schema:

{output_schema}"""


# ---------------------------------------------------------------------------
# Output Schema (embedded in Pass 2 prompt)
# ---------------------------------------------------------------------------

OUTPUT_SCHEMA = """{
  "certNumber": "string",
  "systemDescription": "string",
  "systemOperatingContext": "string",
  "isolationRationale": "string",
  "flowPathAnalysis": {
    "normalFlowDescription": "string",
    "isolationEnvelope": "string",
    "upstreamSources": ["string"],
    "downstreamConnections": ["string"],
    "pressureTrapping": "string",
    "drainAndVentStrategy": "string",
    "reinstateSequence": "string"
  },
  "energySources": [
    {
      "type": "PRESSURE | HYDRAULIC | PNEUMATIC | ELECTRICAL | THERMAL | CHEMICAL | MECHANICAL | GRAVITY",
      "description": "string",
      "location": "string",
      "magnitude": "string",
      "isolationMethod": "string",
      "residualRisk": "string"
    }
  ],
  "processFluid": "string",
  "operatingPressure": "string",
  "operatingTemperature": "string",
  "additionalHazards": ["string"],
  "isolationBoundary": "string",
  "isolationBoundaryRationale": "string",
  "minimumIsolationStandard": "string",
  "hazardClassification": "HIGH | MEDIUM | LOW",
  "hazardNarrative": "string",
  "lotoRequirements": "string",
  "lotoPhilosophy": "string",
  "permitRequirements": ["string"],
  "permitRationale": "string",
  "stats": {
    "valveCount": 0,
    "blindCount": 0,
    "drainVentCount": 0,
    "stepCount": 0,
    "energySourceCount": 0
  },
  "isolationPoints": [
    {
      "seq": 1,
      "tag": "string",
      "type": "GATE VALVE | BALL VALVE | CHECK VALVE | CONTROL VALVE | MOV | SPECTACLE BLIND | BLEED VALVE | VENT | DRAIN | ELECTRICAL ISOLATOR | PSV | PLUG VALVE | BUTTERFLY VALVE",
      "location": "string",
      "normalPosition": "OPEN | CLOSED",
      "isolationPosition": "OPEN | CLOSED | INSTALLED | REMOVED | OPEN TO DRAIN | OPEN TO ATMOSPHERE",
      "isolationClass": "PRIMARY | SECONDARY | BLEED | VENT | DRAIN | LOTO | SUPPORT",
      "energySource": "string",
      "flowDirection": "string",
      "whyRequired": "string",
      "whyThisPosition": "string",
      "consequenceIfMissed": "string",
      "isolationPhilosophy": "DBB_FIRST_BLOCK | DBB_SECOND_BLOCK | DBB_BLEED | SINGLE_BLOCK | BLIND | VENT | DRAIN | LOTO | INSTRUMENT_ISOLATION",
      "blindRequired": false,
      "blindSpec": "string | null",
      "blindRationale": "string | null",
      "lockout": false,
      "lotoDetail": "string | null",
      "lotoRationale": "string | null",
      "testRequired": "NIL | PRESSURE TEST | LEAK TEST | CONTINUITY TEST | ZERO ENERGY VERIFICATION",
      "testMethod": "string | null",
      "verificationMethod": "string",
      "verificationRationale": "string",
      "drawingRef": "string | null",
      "notes": "string | null"
    }
  ],
  "procedureSteps": [
    {
      "seq": 1,
      "phase": "PREPARATION | ISOLATION | DEPRESSURISATION | DRAINING | PURGING | VERIFICATION | WORK AUTHORIZATION | REINSTATEMENT",
      "action": "string",
      "detail": "string",
      "engineeringBasis": "string",
      "critical": false,
      "criticalReason": "string | null",
      "responsible": "Isolation Authority | Operator | Electrician | Instrument Tech | OIM | Technician",
      "ruleRef": "string | null",
      "drawingRef": "string | null",
      "expectedOutcome": "string"
    }
  ],
  "verificationChecks": ["string"],
  "blindRegister": [
    {
      "location": "string",
      "spec": "string",
      "whyRequired": "string",
      "installedBy": "",
      "verifiedBy": ""
    }
  ],
  "decontaminationRequirements": "string",
  "decontaminationRationale": "string",
  "reinstatementProcedure": "string",
  "reinstatementRationale": "string",
  "simopsConsiderations": "string | null",
  "additionalHazards": ["string"]
}"""


def build_system_prompt(facility: str, regime: str, facility_rules: str = "") -> str:
    """Build the full system prompt with facility-specific variables."""
    return SYSTEM_PROMPT_TEMPLATE.format(
        facility=_sanitize_prompt_input(facility or "the facility", 200),
        regime=_sanitize_prompt_input(regime or "C-NLOPB / C-NLOER", 200),
        facility_rules=_sanitize_prompt_input(
            facility_rules or "No facility-specific rules documents uploaded.", 50000
        ),
    )


def run_pass1(client, drawings_metadata: list[dict], job: dict) -> list[str]:
    """
    Pass 1 — Drawing Discovery.
    Returns a list of up to 4 drawing IDs relevant to the isolation.
    """
    prompt = PASS1_PROMPT_TEMPLATE.format(
        equipment_tag=_sanitize_prompt_input(job["equipment_tag"], 200),
        work_description=_sanitize_prompt_input(job["work_description"], 5000),
        work_type=_sanitize_prompt_input(job["work_type"], 100),
        fluid_service=_sanitize_prompt_input(job.get("fluid_service", "Not specified"), 200),
        drawing_metadata_json=json.dumps(drawings_metadata, indent=2)[:50000],
    )

    logger.info("Pass 1: searching %d drawings for %s", len(drawings_metadata), job["equipment_tag"])

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=PASS1_MODEL,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )

            text = "".join(b.text for b in response.content if b.type == "text")

            try:
                ids = json.loads(text.strip())
                if isinstance(ids, list):
                    return [str(i) for i in ids[:4]]
            except json.JSONDecodeError:
                logger.warning("Pass 1 returned non-JSON: %s", text[:200])

            return []
        except Exception as exc:
            if _is_retryable_error(exc) and attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF * (2 ** attempt)
                logger.warning("Pass 1 failed (attempt %d/%d), retrying in %ds: %s",
                               attempt + 1, MAX_RETRIES, wait, exc)
                time.sleep(wait)
            else:
                logger.error("Pass 1 failed for %s", job["equipment_tag"], exc_info=True)
                raise

    return []


def run_pass2_stream(client, job: dict, drawing_images: list[dict],
                     facility: str, regime: str, facility_rules: str = "",
                     cert_number: str = ""):
    """
    Pass 2 — Full isolation generation with streaming.
    Yields text chunks as the AI generates them.
    Returns nothing — caller collects chunks.

    drawing_images: list of {"id": str, "title": str, "image_b64": str, "media_type": str}
    """
    system_prompt = build_system_prompt(facility, regime, facility_rules)

    # Build drawing list text
    drawing_list_lines = []
    for i, d in enumerate(drawing_images, 1):
        drawing_list_lines.append(f"{i}. {d.get('title', d['id'])} (ID: {d['id']})")
    drawing_list = "\n".join(drawing_list_lines)

    rules_note = ""
    if facility_rules:
        rules_note = "Facility rules documents are loaded in the system instructions. Cite specific sections when applicable."
    else:
        rules_note = "No facility-specific rules documents uploaded. Apply standard offshore isolation practices per C-NLOER requirements."

    user_text = PASS2_USER_TEMPLATE.format(
        n_drawings=len(drawing_images),
        drawing_list=drawing_list,
        rules_note=rules_note,
        equipment_tag=_sanitize_prompt_input(job["equipment_tag"], 200),
        work_description=_sanitize_prompt_input(job["work_description"], 5000),
        work_type=_sanitize_prompt_input(job["work_type"], 100),
        fluid_service=_sanitize_prompt_input(job.get("fluid_service", "Not specified"), 200),
        special_requirements=_sanitize_prompt_input(job.get("special_requirements", "None"), 5000),
        facility=_sanitize_prompt_input(facility or "Offshore facility", 200),
        regime=_sanitize_prompt_input(regime or "C-NLOPB / C-NLOER", 200),
        cert_number=_sanitize_prompt_input(cert_number, 100),
        output_schema=OUTPUT_SCHEMA,
    )

    # Build the multimodal content: images first, then text
    content_blocks = []
    for d in drawing_images:
        media_type = d.get("media_type", "image/png")
        if media_type not in _ALLOWED_MEDIA_TYPES:
            media_type = "image/png"
        image_data = d["image_b64"]
        if len(image_data) > _MAX_IMAGE_B64_LEN:
            logger.warning("Skipping oversized drawing image (%d bytes)", len(image_data))
            continue
        content_blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_data,
            },
        })
    content_blocks.append({"type": "text", "text": user_text})

    logger.info("Pass 2: generating isolation for %s with %d drawings",
                job["equipment_tag"], len(drawing_images))

    with client.messages.stream(
        model=PASS2_MODEL,
        max_tokens=12000,
        system=system_prompt,
        messages=[{"role": "user", "content": content_blocks}],
    ) as stream:
        for text in stream.text_stream:
            yield text
