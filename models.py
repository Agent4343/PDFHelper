from typing import Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_]+$")
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    username: str = Field(max_length=50)
    password: str = Field(max_length=128)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
    api_key_required: bool = False
    has_users: bool = False
    warnings: list[str] = []


# ---------------------------------------------------------------------------
# Documents & Search
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    search_terms: list[str] = Field(default=[], max_length=50, description="Exact keywords to search")
    ai_query: str | None = Field(default=None, max_length=2000, description="AI concept search query")
    case_sensitive: bool = False


class AnalyzeRequest(BaseModel):
    compliance_context: str | None = Field(
        default=None,
        max_length=2000,
        description="Optional compliance standard to check against, e.g. 'OSHA 2024', 'HIPAA', 'FDA 21 CFR Part 11'",
    )
    search_terms: list[str] = Field(default=[], max_length=50, description="Optional keywords to search for")
    ai_query: str | None = Field(default=None, max_length=2000, description="Optional AI concept search query")


class MergeRequest(BaseModel):
    doc_ids: list[str] = Field(..., min_length=2, description="IDs of documents to merge (in order)")
    output_filename: str = Field(default="merged.pdf", max_length=255)


class SplitRequest(BaseModel):
    pages: list[int] = Field(..., min_length=1, description="Page numbers to extract (1-based)")
    output_filename: str = Field(default="split.pdf", max_length=255)


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(description="'user' or 'assistant'")
    content: str = Field(max_length=200000)


class ChatRequest(BaseModel):
    message: str = Field(max_length=50000, description="The user's message")
    doc_ids: list[str] = Field(default=[], max_length=100, description="Document IDs to use as context (empty = all)")
    conversation_history: list[ChatMessage] = Field(default=[], max_length=200, description="Previous messages for context")
    session_id: str | None = Field(default=None, max_length=100, description="Chat session ID to continue (omit to create new)")
    model: str = Field(default="", pattern=r"^(sonnet|haiku|)$", description="Model to use: 'sonnet' or 'haiku' (empty = default)")


class ExportChatRequest(BaseModel):
    session_id: str = Field(max_length=100, description="Chat session to export")
    format: str = Field(default="docx", pattern=r"^docx$", description="Export format (docx)")


class GenerateDocRequest(BaseModel):
    session_id: str = Field(max_length=100, description="Chat session for context")
    instructions: str = Field(max_length=10000, description="What the document should contain, e.g. 'Create a safety procedure for valve isolation'")
    doc_ids: list[str] = Field(default=[], max_length=100, description="Document IDs to reference")
    title: str = Field(default="Generated Document", max_length=255)
    include_vba: bool = Field(default=False, description="Include VBA macro code for Word formatting automation")


class MarkdownToDocxRequest(BaseModel):
    markdown: str = Field(max_length=200000)
    title: str = Field(default="Document", max_length=255)


class ImproveProcedureRequest(BaseModel):
    session_id: str = Field(max_length=100, description="Chat session for context")
    procedure_doc_id: str = Field(max_length=100, description="Document ID of the procedure to improve")
    reference_doc_ids: list[str] = Field(default=[], max_length=100, description="Document IDs of reference materials (standards, regs, other procedures)")
    focus_areas: str = Field(default="", max_length=5000, description="Specific areas to improve: clarity, safety steps, compliance, formatting, etc.")
    title: str = Field(default="Improved Procedure", max_length=255)
    include_vba: bool = Field(default=False, description="Include VBA macro code for Word formatting automation")


# ---------------------------------------------------------------------------
# Code Chat
# ---------------------------------------------------------------------------

class CodeChatRequest(BaseModel):
    message: str = Field(max_length=50000, description="The user's message")
    doc_ids: list[str] = Field(default=[], max_length=100, description="Document IDs for context")
    conversation_history: list[ChatMessage] = Field(default=[], max_length=200, description="Previous messages")
    session_id: str | None = Field(default=None, max_length=100, description="Code session ID to continue")
    model: str = Field(default="", pattern=r"^(sonnet|haiku|)$", description="Model to use")


# ---------------------------------------------------------------------------
# Drawings & Isolation
# ---------------------------------------------------------------------------

_VALID_WORK_TYPES = {
    "MAINTENANCE", "HOT WORK", "CONFINED SPACE ENTRY", "PRESSURE TEST",
    "INSPECTION", "EQUIPMENT REMOVAL", "ELECTRICAL ISOLATION", "INSTRUMENT MAINTENANCE",
}


class IsolationRequest(BaseModel):
    equipment_tag: str = Field(max_length=200, description="Equipment tag, e.g. HB-P-1001A")
    work_description: str = Field(max_length=5000, description="Description of work to be performed")
    work_type: str = Field(max_length=100, description="MAINTENANCE, HOT WORK, CONFINED SPACE ENTRY, PRESSURE TEST, INSPECTION, EQUIPMENT REMOVAL, ELECTRICAL ISOLATION, INSTRUMENT MAINTENANCE")
    fluid_service: str = Field(default="Not specified", max_length=200, description="Fluid service, e.g. Crude Oil, HC Gas, Produced Water")
    special_requirements: str = Field(default="None", max_length=5000, description="Any special requirements")
    facility: str = Field(default="Hebron", max_length=200, description="Facility name")
    regime: str = Field(default="C-NLOPB / C-NLOER", max_length=200, description="Regulatory regime")
    drawing_ids: list[str] = Field(default=[], max_length=20, description="Specific drawing IDs to use (empty = auto-select via Pass 1)")


# ---------------------------------------------------------------------------
# Doc Updater
# ---------------------------------------------------------------------------

class RegulationSearchRequest(BaseModel):
    query: str = Field(max_length=2000)
    doc_id: str | None = Field(default=None)
    context: str = Field(default="", max_length=5000)


class GenerateUpdatesRequest(BaseModel):
    doc_id: str = Field(max_length=100)
    regulation_text: str = Field(max_length=100000)
    additional_instructions: str = Field(default="", max_length=5000)


class ReviewSectionRequest(BaseModel):
    highlighted_text: str = Field(max_length=10000)
    context: str = Field(default="", max_length=5000)
    focus: str = Field(default="compliance,clarity,completeness", max_length=500)


class ApplyUpdatesRequest(BaseModel):
    updates_markdown: str = Field(max_length=200000, description="The full AI-generated updates text to apply")
    title: str = Field(default="Updated Document", max_length=255)
    include_vba: bool = Field(default=False)


class SaveSessionRequest(BaseModel):
    doc_id: str = Field(max_length=100)
    title: str = Field(default="", max_length=255)
    regulation_query: str = Field(default="", max_length=2000)
    regulation_results: str = Field(default="", max_length=200000)
    updates_json: str = Field(default="[]", max_length=200000)
    accepted_ids: list[str] = Field(default=[])


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class BulkAuditRequest(BaseModel):
    doc_ids: list[str] = Field(default=[], description="Document IDs to audit (empty = all)")
    focus_areas: str = Field(default="", max_length=2000)
    model: str = Field(default="haiku", pattern=r"^(sonnet|haiku)$")


class ComplianceAuditRequest(BaseModel):
    doc_id: str = Field(max_length=100)
    focus_areas: str = Field(default="", max_length=2000)
    model: str = Field(default="sonnet", pattern=r"^(sonnet|haiku)$")


class CompareDocsRequest(BaseModel):
    doc_id_1: str = Field(max_length=100)
    doc_id_2: str = Field(max_length=100)
    focus_areas: str = Field(default="", max_length=2000)
    model: str = Field(default="sonnet", pattern=r"^(sonnet|haiku)$")


class ProcedureWriterRequest(BaseModel):
    description: str = Field(max_length=5000)
    source_doc_id: str = Field(default="", description="Primary document to base the procedure on")
    reference_doc_ids: list[str] = Field(default=[])
    include_regulations: bool = Field(default=True)
    model: str = Field(default="sonnet", pattern=r"^(sonnet|haiku)$")


class CodeBuilderRequest(BaseModel):
    description: str = Field(max_length=50000)
    doc_ids: list[str] = Field(default=[])
    app_type: str = Field(default="dashboard", pattern=r"^(dashboard|form|tracker|checklist|report|custom)$")
    model: str = Field(default="sonnet", pattern=r"^(sonnet|haiku)$")


# ---------------------------------------------------------------------------
# Posters
# ---------------------------------------------------------------------------

class PosterCreateRequest(BaseModel):
    prompt: str = Field(max_length=30000, description="Describe the poster you want to create")
    size: str = Field(default="letter", pattern=r"^(letter|a4|a3|wide|square|banner)$")
    style: str = Field(default="", max_length=50, description="Optional style preset")
    model: str = Field(default="haiku", pattern=r"^(sonnet|haiku)$")


class PosterUpdateRequest(BaseModel):
    prompt: str = Field(max_length=30000, description="Describe what to change on the poster")
    model: str = Field(default="haiku", pattern=r"^(sonnet|haiku)$")


class PosterSaveHTMLRequest(BaseModel):
    html: str = Field(max_length=500000, description="Updated HTML content")
