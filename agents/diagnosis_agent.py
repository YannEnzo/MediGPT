# agents/diagnosis_agent.py - Diagnosis Agent for Medical Analysis
import asyncio
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from models import (
    EHRInput, Diagnosis, SeverityLevel, ConfidenceLevel,
    VitalSigns, LabResults
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiagnosisAgent:
    """
    Diagnosis Agent that analyzes patient EHR data and suggests ranked diagnoses.
    Currently uses rule-based logic with plans for LLM integration.
    """
    
    def __init__(self):
        self.agent_name = "DiagnosisAgent"
        self.version = "1.0.0"
        self.model_type = "rule_based"  # Will change to "gpt-4" or "med-bert" later
        
        # Initialize medical knowledge base (simplified for MVP)
        self._load_medical_knowledge()
        
        logger.info(f"Initialized {self.agent_name} v{self.version}")
    
    def _load_medical_knowledge(self):
        """Load medical knowledge base for diagnosis matching"""
        # Simplified medical knowledge - will be replaced with comprehensive database
        self.symptom_patterns = {
            # Respiratory conditions
            "pneumonia": {
                "symptoms": ["fever", "cough", "chest pain", "shortness of breath", "fatigue"],
                "vitals": {"temperature": (100.5, 104.0), "respiratory_rate": (20, 30)},
                "labs": {"wbc": (10.0, 25.0), "crp": (10.0, 100.0)},
                "icd10": "J18.9",
                "base_confidence": 0.8,  # Increased from 0.7
                "severity": SeverityLevel.MODERATE
            },
            "upper_respiratory_infection": {
                "symptoms": ["cough", "sore throat", "runny nose", "congestion", "mild fever"],
                "vitals": {"temperature": (99.0, 101.0)},
                "labs": {},
                "icd10": "J06.9",
                "base_confidence": 0.8,
                "severity": SeverityLevel.LOW
            },
            "bronchitis": {
                "symptoms": ["persistent cough", "chest discomfort", "fatigue", "mild fever"],
                "vitals": {},
                "labs": {},
                "icd10": "J40",
                "base_confidence": 0.6,
                "severity": SeverityLevel.LOW
            },
            
            # Cardiovascular conditions
            "myocardial_infarction": {
                "symptoms": ["chest pain", "shortness of breath", "nausea", "sweating", "arm pain"],
                "vitals": {"heart_rate": (60, 120), "systolic_bp": (90, 180)},
                "labs": {},
                "icd10": "I21.9",
                "base_confidence": 0.9,
                "severity": SeverityLevel.CRITICAL
            },
            "hypertension": {
                "symptoms": ["headache", "dizziness", "fatigue"],
                "vitals": {"systolic_bp": (140, 200), "diastolic_bp": (90, 120)},
                "labs": {},
                "icd10": "I10",
                "base_confidence": 0.8,
                "severity": SeverityLevel.MODERATE
            },
            
            # Infectious diseases
            "influenza": {
                "symptoms": ["fever", "body aches", "fatigue", "cough", "headache", "chills"],
                "vitals": {"temperature": (101.0, 104.0)},
                "labs": {"wbc": (3.0, 12.0)},
                "icd10": "J11.1",
                "base_confidence": 0.7,
                "severity": SeverityLevel.MODERATE
            },
            "gastroenteritis": {
                "symptoms": ["nausea", "vomiting", "diarrhea", "abdominal pain", "fever"],
                "vitals": {"temperature": (99.0, 102.0), "heart_rate": (80, 120)},
                "labs": {},
                "icd10": "K59.1",
                "base_confidence": 0.6,
                "severity": SeverityLevel.LOW
            },
            
            # Endocrine conditions
            "diabetes_mellitus": {
                "symptoms": ["excessive urination", "excessive thirst", "fatigue", "blurred vision"],
                "vitals": {},
                "labs": {"glucose": (126.0, 400.0)},
                "icd10": "E11.9",
                "base_confidence": 0.9,
                "severity": SeverityLevel.MODERATE
            }
        }
    
    async def analyze_patient(self, ehr_data: EHRInput) -> List[Diagnosis]:
        """
        Main method to analyze patient data and return ranked diagnoses
        
        Args:
            ehr_data: Patient EHR data
            
        Returns:
            List of Diagnosis objects ranked by confidence
        """
        logger.info(f"Analyzing patient {ehr_data.patient_id}")
        logger.info(f"Patient symptoms: {ehr_data.symptoms}")
        logger.info(f"Patient vitals: {ehr_data.vitals}")
        logger.info(f"Patient labs: {ehr_data.labs}")
        
        # Start analysis timer
        start_time = datetime.now()
        
        try:
            # Generate potential diagnoses
            potential_diagnoses = []
            
            for condition, knowledge in self.symptom_patterns.items():
                diagnosis = await self._evaluate_condition(ehr_data, condition, knowledge)
                if diagnosis:
                    potential_diagnoses.append(diagnosis)
                    logger.info(f"Generated diagnosis: {condition} with confidence {diagnosis.confidence:.2f}")
            
            # Rank diagnoses by confidence
            ranked_diagnoses = sorted(potential_diagnoses, key=lambda d: d.confidence, reverse=True)
            
            # Take top 5 most likely diagnoses
            final_diagnoses = ranked_diagnoses[:5]
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.info(f"Generated {len(final_diagnoses)} diagnoses for patient {ehr_data.patient_id} in {processing_time:.2f}ms")
            
            return final_diagnoses
            
        except Exception as e:
            logger.error(f"Error analyzing patient {ehr_data.patient_id}: {str(e)}")
            raise
    
    async def _evaluate_condition(self, ehr_data: EHRInput, condition: str, knowledge: Dict) -> Optional[Diagnosis]:
        """
        Evaluate a specific condition against patient data
        
        Args:
            ehr_data: Patient data
            condition: Condition name
            knowledge: Medical knowledge for this condition
            
        Returns:
            Diagnosis object if condition matches, None otherwise
        """
        # Calculate symptom match score
        symptom_score = self._calculate_symptom_match(ehr_data.symptoms, knowledge["symptoms"])
        
        # If symptom match is too low, skip this condition - LOWERED THRESHOLD
        if symptom_score < 0.2:  # Was 0.3, now 0.2 - more lenient
            return None
        
        # Calculate vital signs match
        vital_score = self._calculate_vital_match(ehr_data.vitals, knowledge.get("vitals", {}))
        
        # Calculate lab results match
        lab_score = self._calculate_lab_match(ehr_data.labs, knowledge.get("labs", {}))
        
        # Calculate overall confidence
        base_confidence = knowledge["base_confidence"]
        
        # Weight the scores
        symptom_weight = 0.5
        vital_weight = 0.3
        lab_weight = 0.2
        
        if not ehr_data.vitals:
            symptom_weight = 0.7
            lab_weight = 0.3
            vital_weight = 0.0
        
        if not ehr_data.labs:
            symptom_weight = 0.7
            vital_weight = 0.3
            lab_weight = 0.0
        
        # Calculate weighted confidence
        confidence = (
            symptom_score * symptom_weight +
            vital_score * vital_weight +
            lab_score * lab_weight
        ) * base_confidence
        
        # Ensure confidence is between 0 and 1
        confidence = max(0.0, min(1.0, confidence))
        
        # Generate supporting evidence
        supporting_evidence = self._generate_supporting_evidence(
            ehr_data, condition, symptom_score, vital_score, lab_score
        )
        
        # Determine confidence level
        confidence_level = self._get_confidence_level(confidence)
        
        # Adjust severity based on patient factors
        severity = self._adjust_severity_for_patient(knowledge["severity"], ehr_data)
        urgency = self._determine_urgency(severity, confidence)
        
        return Diagnosis(
            condition=condition.replace("_", " ").title(),
            icd10_code=knowledge["icd10"],
            confidence=confidence,
            confidence_level=confidence_level,
            supporting_evidence=supporting_evidence,
            differential_notes=self._generate_differential_notes(condition, ehr_data),
            severity=severity,
            urgency=urgency,
            prevalence_info=self._get_prevalence_info(condition, ehr_data)
        )
    
    def _calculate_symptom_match(self, patient_symptoms: List[str], condition_symptoms: List[str]) -> float:
        """Calculate symptom match score between patient and condition"""
        if not condition_symptoms:
            return 0.5  # Neutral score if no specific symptoms
        
        if not patient_symptoms:
            return 0.0
        
        # Normalize symptoms for comparison
        patient_symptoms_normalized = [s.lower().strip() for s in patient_symptoms]
        condition_symptoms_normalized = [s.lower().strip() for s in condition_symptoms]
        
        logger.info(f"Comparing patient symptoms {patient_symptoms_normalized} vs condition symptoms {condition_symptoms_normalized}")
        
        # Calculate matches (allowing partial matches)
        matches = 0
        matched_symptoms = []
        for condition_symptom in condition_symptoms_normalized:
            for patient_symptom in patient_symptoms_normalized:
                if self._symptoms_match(patient_symptom, condition_symptom):
                    matches += 1
                    matched_symptoms.append(f"{patient_symptom} -> {condition_symptom}")
                    break
        
        # Calculate score as percentage of condition symptoms matched
        score = matches / len(condition_symptoms)
        logger.info(f"Symptom matches: {matched_symptoms}, Score: {score:.2f}")
        return score
    
    def _symptoms_match(self, patient_symptom: str, condition_symptom: str) -> bool:
        """Check if patient symptom matches condition symptom (including partial matches)"""
        # Normalize both symptoms
        patient_symptom = patient_symptom.lower().strip()
        condition_symptom = condition_symptom.lower().strip()
        
        # Exact match
        if patient_symptom == condition_symptom:
            return True
        
        # Partial match (patient symptom contains condition symptom or vice versa)
        if condition_symptom in patient_symptom or patient_symptom in condition_symptom:
            return True
        
        # Check individual words
        patient_words = patient_symptom.split()
        condition_words = condition_symptom.split()
        
        # If any word from condition symptom is in patient symptom
        for condition_word in condition_words:
            if len(condition_word) > 2:  # Only check meaningful words
                for patient_word in patient_words:
                    if condition_word in patient_word or patient_word in condition_word:
                        return True
        
        # Synonym matching (expanded)
        synonyms = {
            "fever": ["high temperature", "pyrexia", "febrile", "hot", "temperature"],
            "shortness of breath": ["dyspnea", "difficulty breathing", "breathlessness", "sob", "breathing problems"],
            "chest pain": ["chest discomfort", "chest tightness", "chest pressure"],
            "fatigue": ["tiredness", "exhaustion", "weakness", "tired"],
            "nausea": ["feeling sick", "queasiness", "sick"],
            "headache": ["head pain", "cephalgia", "head ache"],
            "cough": ["coughing", "productive cough", "dry cough"]
        }
        
        # Check if both symptoms are in the same synonym group
        for main_symptom, synonym_list in synonyms.items():
            all_terms = [main_symptom] + synonym_list
            if patient_symptom in all_terms and condition_symptom in all_terms:
                return True
        
        return False
    
    def _calculate_vital_match(self, patient_vitals: Optional[VitalSigns], condition_vitals: Dict) -> float:
        """Calculate vital signs match score"""
        if not condition_vitals or not patient_vitals:
            return 0.5  # Neutral score if no vital requirements or data
        
        matches = 0
        total_checks = 0
        
        for vital_name, (min_val, max_val) in condition_vitals.items():
            patient_value = getattr(patient_vitals, vital_name, None)
            if patient_value is not None:
                total_checks += 1
                if min_val <= patient_value <= max_val:
                    matches += 1
        
        return matches / total_checks if total_checks > 0 else 0.5
    
    def _calculate_lab_match(self, patient_labs: Optional[LabResults], condition_labs: Dict) -> float:
        """Calculate lab results match score"""
        if not condition_labs or not patient_labs:
            return 0.5  # Neutral score if no lab requirements or data
        
        matches = 0
        total_checks = 0
        
        for lab_name, (min_val, max_val) in condition_labs.items():
            patient_value = getattr(patient_labs, lab_name, None)
            if patient_value is not None:
                total_checks += 1
                if min_val <= patient_value <= max_val:
                    matches += 1
        
        return matches / total_checks if total_checks > 0 else 0.5
    
    def _generate_supporting_evidence(self, ehr_data: EHRInput, condition: str, 
                                    symptom_score: float, vital_score: float, lab_score: float) -> List[str]:
        """Generate supporting evidence for the diagnosis"""
        evidence = []
        
        if symptom_score > 0.6:
            evidence.append(f"Patient symptoms are highly consistent with {condition.replace('_', ' ')}")
        elif symptom_score > 0.3:
            evidence.append(f"Patient symptoms partially match {condition.replace('_', ' ')}")
        
        if ehr_data.vitals and vital_score > 0.6:
            evidence.append("Vital signs support this diagnosis")
        
        if ehr_data.labs and lab_score > 0.6:
            evidence.append("Laboratory results are consistent with this condition")
        
        # Age-specific evidence
        if ehr_data.age > 65:
            evidence.append("Patient age increases risk for this condition")
        
        # Gender-specific evidence (simplified)
        if condition == "myocardial_infarction" and ehr_data.gender.value == "male":
            evidence.append("Male gender increases cardiovascular risk")
        
        return evidence
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert numeric confidence to categorical level"""
        if confidence >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.6:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.4:
            return ConfidenceLevel.MODERATE
        elif confidence >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _adjust_severity_for_patient(self, base_severity: SeverityLevel, ehr_data: EHRInput) -> SeverityLevel:
        """Adjust severity based on patient-specific factors"""
        # Age adjustments
        if ehr_data.age > 75:
            if base_severity == SeverityLevel.LOW:
                return SeverityLevel.MODERATE
            elif base_severity == SeverityLevel.MODERATE:
                return SeverityLevel.HIGH
        
        # Comorbidity adjustments
        high_risk_conditions = ["diabetes", "heart disease", "copd", "cancer"]
        if any(condition in " ".join(ehr_data.medical_history).lower() for condition in high_risk_conditions):
            if base_severity == SeverityLevel.LOW:
                return SeverityLevel.MODERATE
        
        return base_severity
    
    def _determine_urgency(self, severity: SeverityLevel, confidence: float) -> SeverityLevel:
        """Determine urgency based on severity and confidence"""
        if severity == SeverityLevel.CRITICAL:
            return SeverityLevel.CRITICAL
        elif severity == SeverityLevel.HIGH and confidence > 0.7:
            return SeverityLevel.HIGH
        elif severity == SeverityLevel.MODERATE and confidence > 0.8:
            return SeverityLevel.MODERATE
        else:
            return SeverityLevel.LOW
    
    def _generate_differential_notes(self, condition: str, ehr_data: EHRInput) -> str:
        """Generate differential diagnosis notes"""
        if condition == "pneumonia":
            return "Consider viral pneumonia, bacterial pneumonia, or atypical pneumonia. Rule out tuberculosis in high-risk patients."
        elif condition == "myocardial_infarction":
            return "Consider STEMI vs NSTEMI. Rule out unstable angina, aortic dissection, and pulmonary embolism."
        elif condition == "influenza":
            return "Consider other viral syndromes, bacterial infections, or COVID-19."
        else:
            return f"Consider alternative diagnoses with similar presentation to {condition.replace('_', ' ')}"
    
    def _get_prevalence_info(self, condition: str, ehr_data: EHRInput) -> str:
        """Get prevalence information for the condition"""
        prevalence_data = {
            "pneumonia": "Affects 5-6 per 1000 adults annually",
            "hypertension": "Affects ~45% of adults in the US",
            "diabetes_mellitus": "Affects ~10% of US population",
            "influenza": "Seasonal incidence varies, 5-20% population annually"
        }
        
        return prevalence_data.get(condition, "Prevalence data not available")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "name": self.agent_name,
            "version": self.version,
            "model_type": self.model_type,
            "capabilities": [
                "symptom_analysis",
                "vital_signs_interpretation", 
                "lab_results_analysis",
                "differential_diagnosis",
                "confidence_scoring",
                "severity_assessment"
            ],
            "conditions_supported": len(self.symptom_patterns)
        }
    
    async def update_medical_knowledge(self, new_patterns: Dict):
        """Update medical knowledge base (for future enhancement)"""
        self.symptom_patterns.update(new_patterns)
        logger.info(f"Updated medical knowledge base with {len(new_patterns)} new patterns")