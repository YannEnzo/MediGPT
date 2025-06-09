# models.py - Fixed Pydantic Models for MediGPT Agents
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

# Enums for structured data
class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"

class SeverityLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

class ConfidenceLevel(str, Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"

# Core Input Models
class VitalSigns(BaseModel):
    """Vital signs data structure"""
    heart_rate: Optional[int] = None
    systolic_bp: Optional[int] = None
    diastolic_bp: Optional[int] = None
    temperature: Optional[float] = None
    respiratory_rate: Optional[int] = None
    oxygen_saturation: Optional[float] = None

class LabResults(BaseModel):
    """Laboratory test results"""
    wbc: Optional[float] = None
    rbc: Optional[float] = None
    hemoglobin: Optional[float] = None
    hematocrit: Optional[float] = None
    platelets: Optional[float] = None
    glucose: Optional[float] = None
    creatinine: Optional[float] = None
    bun: Optional[float] = None
    sodium: Optional[float] = None
    potassium: Optional[float] = None
    crp: Optional[float] = None
    esr: Optional[float] = None
    other_labs: Optional[Dict[str, Union[float, str]]] = Field(default_factory=dict)

class EHRInput(BaseModel):
    """Electronic Health Record input model"""
    patient_id: str
    age: int
    gender: Gender
    
    # Clinical presentation
    chief_complaint: Optional[str] = None
    symptoms: List[str] = Field(default_factory=list)
    symptom_duration: Optional[str] = None
    
    # Objective data
    vitals: Optional[VitalSigns] = None
    labs: Optional[LabResults] = None
    
    # Medical history
    medical_history: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    
    # Social history
    smoking_status: Optional[str] = None
    alcohol_use: Optional[str] = None
    
    # Additional context
    notes: Optional[str] = None

# Response Models
class Diagnosis(BaseModel):
    """Individual diagnosis with metadata"""
    condition: str
    icd10_code: Optional[str] = None
    confidence: float
    confidence_level: ConfidenceLevel
    
    # Clinical reasoning
    supporting_evidence: List[str] = Field(default_factory=list)
    differential_notes: Optional[str] = None
    
    # Risk assessment
    severity: SeverityLevel
    urgency: SeverityLevel
    
    # Likelihood and prevalence
    prevalence_info: Optional[str] = None

class DiagnosisResponse(BaseModel):
    """Response model for diagnosis endpoint"""
    patient_id: str
    diagnoses: List[Diagnosis]
    
    # Metadata
    processed_at: datetime
    agent_version: str
    processing_time_ms: Optional[float] = None
    
    # Summary statistics
    total_diagnoses: int = Field(default=0)
    high_confidence_count: int = Field(default=0)
    
    def __init__(self, **data):
        super().__init__(**data)
        # Auto-calculate summary statistics
        if 'total_diagnoses' not in data:
            self.total_diagnoses = len(self.diagnoses)
        if 'high_confidence_count' not in data:
            self.high_confidence_count = sum(1 for d in self.diagnoses if d.confidence > 0.7)

class ValidationEvidence(BaseModel):
    """Evidence retrieved for diagnosis validation"""
    source: str
    title: str
    abstract: Optional[str] = None
    relevance_score: float
    publication_year: Optional[int] = None
    study_type: Optional[str] = None
    evidence_level: Optional[str] = None

class ValidationResponse(BaseModel):
    """Response model for validation endpoint"""
    diagnosis: str
    
    # Validation results
    is_supported: bool
    confidence_adjustment: float
    final_confidence: float
    
    # Evidence
    supporting_evidence: List[ValidationEvidence] = Field(default_factory=list)
    contradicting_evidence: List[ValidationEvidence] = Field(default_factory=list)
    
    # Analysis
    evidence_summary: str
    recommendations: List[str] = Field(default_factory=list)
    
    # Metadata
    validated_at: datetime = Field(default_factory=datetime.now)
    total_sources_reviewed: int = 0

class Medication(BaseModel):
    """Medication prescription details"""
    name: str
    generic_name: Optional[str] = None
    dosage: str
    frequency: str
    duration: str
    route: str = "oral"
    
    # Clinical details
    indication: str
    contraindications: List[str] = Field(default_factory=list)
    side_effects: List[str] = Field(default_factory=list)
    monitoring_required: List[str] = Field(default_factory=list)

class Treatment(BaseModel):
    """Treatment intervention details"""
    intervention_type: str
    description: str
    rationale: str
    
    # Timeline
    start_immediately: bool = True
    duration: Optional[str] = None
    follow_up_schedule: List[str] = Field(default_factory=list)
    
    # Outcomes
    expected_outcome: str
    success_probability: float
    
    # Risk assessment
    risk_level: SeverityLevel
    complications: List[str] = Field(default_factory=list)

class TreatmentResponse(BaseModel):
    """Response model for treatment simulation endpoint"""
    diagnosis: str
    patient_id: str
    
    # Treatment plan
    medications: List[Medication] = Field(default_factory=list)
    treatments: List[Treatment] = Field(default_factory=list)
    
    # Clinical plan
    immediate_actions: List[str] = Field(default_factory=list)
    short_term_plan: List[str] = Field(default_factory=list)
    long_term_plan: List[str] = Field(default_factory=list)
    
    # Prognosis
    expected_recovery_time: Optional[str] = None
    prognosis: str
    risk_factors: List[str] = Field(default_factory=list)
    
    # Patient instructions
    lifestyle_modifications: List[str] = Field(default_factory=list)
    warning_signs: List[str] = Field(default_factory=list)
    when_to_seek_care: List[str] = Field(default_factory=list)
    
    # Metadata
    generated_at: datetime = Field(default_factory=datetime.now)
    treatment_agent_version: str = "v1.0"
    personalization_factors: List[str] = Field(default_factory=list)