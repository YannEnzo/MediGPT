# main.py - FastAPI Application Entry Point
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import uvicorn
from datetime import datetime

# Import our models and agents
from models import (
    EHRInput, 
    DiagnosisResponse, 
    ValidationResponse, 
    TreatmentResponse,
    Diagnosis,
    Treatment,
    ValidationEvidence
)
from agents.diagnosis_agent import DiagnosisAgent
from agents.validation_agent import ValidationAgent
from agents.treatment_agent import TreatmentAgent

# Initialize FastAPI app
app = FastAPI(
    title="MediGPT Agents API",
    description="Multi-agent AI system for medical diagnosis, validation, and treatment simulation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agents
diagnosis_agent = DiagnosisAgent()
validation_agent = ValidationAgent()
treatment_agent = TreatmentAgent()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "MediGPT Agents API is running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "agents": ["diagnosis", "validation", "treatment"]
    }

@app.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose_patient(ehr_data: EHRInput):
    """
    Analyze patient EHR data and return ranked diagnoses
    
    Args:
        ehr_data: Electronic Health Record input with symptoms, vitals, labs
        
    Returns:
        DiagnosisResponse with ranked diagnoses, confidence scores, and explanations
    """
    start_time = datetime.now()
    
    try:
        # Process diagnosis using the diagnosis agent
        diagnoses = await diagnosis_agent.analyze_patient(ehr_data)
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time_ms = (end_time - start_time).total_seconds() * 1000
        
        return DiagnosisResponse(
            patient_id=ehr_data.patient_id,
            diagnoses=diagnoses,
            processed_at=end_time,
            agent_version="diagnosis_v1.0",
            processing_time_ms=processing_time_ms
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnosis processing failed: {str(e)}")

@app.get("/validate/{diagnosis}", response_model=ValidationResponse)
async def validate_diagnosis(diagnosis: str, patient_age: Optional[int] = None, patient_gender: Optional[str] = None):
    """
    Validate a diagnosis using RAG-based evidence retrieval
    
    Args:
        diagnosis: The diagnosis to validate
        patient_age: Optional patient age for context
        patient_gender: Optional patient gender for context
        
    Returns:
        ValidationResponse with evidence, confidence adjustments, and citations
    """
    try:
        # Create context for validation
        context = {
            "age": patient_age,
            "gender": patient_gender
        }
        
        # Validate using the validation agent
        validation_result = await validation_agent.validate_diagnosis(diagnosis, context)
        
        return validation_result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation processing failed: {str(e)}")

@app.post("/simulate", response_model=TreatmentResponse)
async def simulate_treatment(diagnosis: str, patient_data: EHRInput):
    """
    Simulate treatment plan and outcomes for a given diagnosis
    
    Args:
        diagnosis: Primary diagnosis for treatment
        patient_data: Patient EHR data for personalized treatment
        
    Returns:
        TreatmentResponse with treatment plan, medications, timeline, and risk assessment
    """
    try:
        # Generate treatment plan using the treatment agent
        treatment_plan = await treatment_agent.generate_treatment_plan(diagnosis, patient_data)
        
        return treatment_plan
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Treatment simulation failed: {str(e)}")

@app.get("/agents/status")
async def get_agents_status():
    """Get status and capabilities of all agents"""
    return {
        "diagnosis_agent": {
            "status": "active",
            "model": diagnosis_agent.get_model_info(),
            "capabilities": ["symptom_analysis", "differential_diagnosis", "confidence_scoring"]
        },
        "validation_agent": {
            "status": "active",
            "model": validation_agent.get_model_info(),
            "capabilities": ["evidence_retrieval", "citation_analysis", "confidence_adjustment"]
        },
        "treatment_agent": {
            "status": "active",
            "model": treatment_agent.get_model_info(),
            "capabilities": ["treatment_planning", "medication_selection", "outcome_prediction"]
        }
    }

# Additional utility endpoints
@app.post("/batch_diagnose")
async def batch_diagnose(patient_batch: List[EHRInput]):
    """Process multiple patients in batch"""
    results = []
    for patient in patient_batch:
        try:
            diagnoses = await diagnosis_agent.analyze_patient(patient)
            results.append({
                "patient_id": patient.patient_id,
                "status": "success",
                "diagnoses": diagnoses
            })
        except Exception as e:
            results.append({
                "patient_id": patient.patient_id,
                "status": "error",
                "error": str(e)
            })
    
    return {"batch_results": results, "total_processed": len(patient_batch)}

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )