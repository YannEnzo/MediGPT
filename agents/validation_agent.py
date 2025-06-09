# agents/validation_agent.py - Validation Agent for Evidence-Based Diagnosis Validation
import asyncio
import random
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import json

from models import (
    ValidationResponse, ValidationEvidence, ConfidenceLevel
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationAgent:
    """
    Validation Agent that uses RAG (Retrieval-Augmented Generation) to validate diagnoses
    against medical literature and evidence-based sources.
    """
    
    def __init__(self):
        self.agent_name = "ValidationAgent"
        self.version = "1.0.0"
        self.model_type = "rag_faiss"  # Will integrate with FAISS vector store
        
        # Initialize mock medical literature database
        self._initialize_mock_literature()
        
        # RAG configuration
        self.max_sources_to_review = 10
        self.relevance_threshold = 0.6
        self.confidence_adjustment_factor = 0.2
        
        logger.info(f"Initialized {self.agent_name} v{self.version}")
    
    def _initialize_mock_literature(self):
        """Initialize mock medical literature for validation (will be replaced with real RAG)"""
        # This simulates a medical literature database
        # In production, this would be replaced with FAISS vector store and actual PubMed data
        self.medical_literature = {
            "pneumonia": [
                {
                    "title": "Community-Acquired Pneumonia: Diagnosis and Management in Adults",
                    "abstract": "Community-acquired pneumonia remains a leading cause of morbidity and mortality. Clinical presentation typically includes fever, cough, and chest pain. Chest X-ray and laboratory studies support diagnosis.",
                    "source": "PMID:12345678",
                    "publication_year": 2023,
                    "study_type": "Clinical Guidelines",
                    "evidence_level": "A",
                    "supports_diagnosis": True,
                    "relevance_keywords": ["fever", "cough", "chest pain", "pneumonia", "respiratory"]
                },
                {
                    "title": "Biomarkers in Pneumonia: A Systematic Review",
                    "abstract": "Elevated C-reactive protein and white blood cell count are reliable indicators of bacterial pneumonia. Procalcitonin shows superior specificity for bacterial vs viral etiology.",
                    "source": "PMID:12345679",
                    "publication_year": 2022,
                    "study_type": "Systematic Review",
                    "evidence_level": "A",
                    "supports_diagnosis": True,
                    "relevance_keywords": ["CRP", "WBC", "biomarkers", "pneumonia", "bacterial"]
                }
            ],
            "myocardial infarction": [
                {
                    "title": "Acute Myocardial Infarction: Contemporary Diagnosis and Management",
                    "abstract": "Acute MI presents with chest pain, dyspnea, and diaphoresis. ECG changes and cardiac biomarkers confirm diagnosis. Early intervention improves outcomes significantly.",
                    "source": "PMID:23456789",
                    "publication_year": 2023,
                    "study_type": "Clinical Review",
                    "evidence_level": "A",
                    "supports_diagnosis": True,
                    "relevance_keywords": ["chest pain", "MI", "cardiac", "ECG", "troponin"]
                },
                {
                    "title": "Gender Differences in Myocardial Infarction Presentation",
                    "abstract": "Women with MI may present with atypical symptoms including nausea, fatigue, and back pain. Traditional chest pain is less common in women under 55.",
                    "source": "PMID:23456790",
                    "publication_year": 2021,
                    "study_type": "Observational Study",
                    "evidence_level": "B",
                    "supports_diagnosis": True,
                    "relevance_keywords": ["gender", "women", "atypical", "MI", "symptoms"]
                }
            ],
            "influenza": [
                {
                    "title": "Influenza: Diagnosis, Treatment, and Prevention Guidelines",
                    "abstract": "Influenza typically presents with acute onset fever, myalgia, and respiratory symptoms. Rapid antigen tests and PCR provide diagnostic confirmation.",
                    "source": "PMID:34567890",
                    "publication_year": 2023,
                    "study_type": "Clinical Guidelines",
                    "evidence_level": "A",
                    "supports_diagnosis": True,
                    "relevance_keywords": ["influenza", "fever", "myalgia", "respiratory", "viral"]
                }
            ],
            "hypertension": [
                {
                    "title": "Hypertension: Updated Guidelines for Diagnosis and Management",
                    "abstract": "Hypertension is defined as sustained BP ≥140/90 mmHg. Often asymptomatic, but may present with headache and dizziness. Lifestyle and pharmacological interventions are effective.",
                    "source": "PMID:45678901",
                    "publication_year": 2022,
                    "study_type": "Clinical Guidelines",
                    "evidence_level": "A",
                    "supports_diagnosis": True,
                    "relevance_keywords": ["hypertension", "blood pressure", "headache", "dizziness"]
                }
            ],
            "diabetes mellitus": [
                {
                    "title": "Type 2 Diabetes: Diagnostic Criteria and Clinical Management",
                    "abstract": "Type 2 diabetes diagnosis requires fasting glucose ≥126 mg/dL or HbA1c ≥6.5%. Classic symptoms include polyuria, polydipsia, and weight loss.",
                    "source": "PMID:56789012",
                    "publication_year": 2023,
                    "study_type": "Clinical Guidelines",
                    "evidence_level": "A",
                    "supports_diagnosis": True,
                    "relevance_keywords": ["diabetes", "glucose", "polyuria", "polydipsia", "HbA1c"]
                }
            ],
            "upper respiratory infection": [
                {
                    "title": "Upper Respiratory Tract Infections: Evidence-Based Management",
                    "abstract": "Most upper respiratory infections are viral and self-limiting. Symptoms include cough, congestion, and mild fever. Supportive care is usually sufficient.",
                    "source": "PMID:67890123",
                    "publication_year": 2022,
                    "study_type": "Clinical Review",
                    "evidence_level": "B",
                    "supports_diagnosis": True,
                    "relevance_keywords": ["upper respiratory", "viral", "cough", "congestion", "self-limiting"]
                }
            ]
        }
        
        # Mock contradictory evidence for realistic validation
        self.contradictory_evidence = {
            "pneumonia": [
                {
                    "title": "Overdiagnosis of Pneumonia in Emergency Departments",
                    "abstract": "Studies suggest pneumonia may be overdiagnosed, particularly in patients with non-specific respiratory symptoms. Viral infections often mimic bacterial pneumonia.",
                    "source": "PMID:78901234",
                    "publication_year": 2021,
                    "study_type": "Observational Study",
                    "evidence_level": "B",
                    "supports_diagnosis": False,
                    "relevance_keywords": ["overdiagnosis", "viral", "pneumonia", "respiratory"]
                }
            ]
        }
    
    async def validate_diagnosis(self, diagnosis: str, context: Dict[str, Any]) -> ValidationResponse:
        """
        Main method to validate a diagnosis using RAG-based evidence retrieval
        
        Args:
            diagnosis: The diagnosis to validate
            context: Additional context (age, gender, etc.)
            
        Returns:
            ValidationResponse with evidence and confidence adjustment
        """
        logger.info(f"Validating diagnosis: {diagnosis}")
        
        start_time = datetime.now()
        
        try:
            # Normalize diagnosis for lookup
            normalized_diagnosis = diagnosis.lower().strip()
            
            # Retrieve supporting evidence
            supporting_evidence = await self._retrieve_supporting_evidence(normalized_diagnosis, context)
            
            # Retrieve contradicting evidence
            contradicting_evidence = await self._retrieve_contradicting_evidence(normalized_diagnosis, context)
            
            # Analyze evidence and calculate confidence adjustment
            analysis_result = await self._analyze_evidence(
                normalized_diagnosis, supporting_evidence, contradicting_evidence, context
            )
            
            # Generate evidence summary
            evidence_summary = self._generate_evidence_summary(
                supporting_evidence, contradicting_evidence, analysis_result
            )
            
            # Generate clinical recommendations
            recommendations = self._generate_recommendations(
                normalized_diagnosis, analysis_result, context
            )
            
            # Calculate final confidence
            original_confidence = 0.7  # This would come from the diagnosis agent
            confidence_adjustment = analysis_result["confidence_adjustment"]
            final_confidence = max(0.0, min(1.0, original_confidence + confidence_adjustment))
            
            # Determine if diagnosis is supported
            is_supported = analysis_result["is_supported"]
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.info(f"Validated {diagnosis} in {processing_time:.2f}ms - Supported: {is_supported}")
            
            return ValidationResponse(
                diagnosis=diagnosis,
                is_supported=is_supported,
                confidence_adjustment=confidence_adjustment,
                final_confidence=final_confidence,
                supporting_evidence=supporting_evidence,
                contradicting_evidence=contradicting_evidence,
                evidence_summary=evidence_summary,
                recommendations=recommendations,
                validated_at=datetime.now(),
                total_sources_reviewed=len(supporting_evidence) + len(contradicting_evidence)
            )
            
        except Exception as e:
            logger.error(f"Error validating diagnosis {diagnosis}: {str(e)}")
            raise
    
    async def _retrieve_supporting_evidence(self, diagnosis: str, context: Dict) -> List[ValidationEvidence]:
        """Retrieve evidence that supports the diagnosis"""
        evidence_list = []
        
        # Look up evidence in mock literature database
        literature_entries = self.medical_literature.get(diagnosis, [])
        
        for entry in literature_entries:
            if entry["supports_diagnosis"]:
                # Calculate relevance score based on context matching
                relevance_score = self._calculate_relevance_score(entry, context)
                
                if relevance_score >= self.relevance_threshold:
                    evidence = ValidationEvidence(
                        source=entry["source"],
                        title=entry["title"],
                        abstract=entry["abstract"],
                        relevance_score=relevance_score,
                        publication_year=entry["publication_year"],
                        study_type=entry["study_type"],
                        evidence_level=entry["evidence_level"]
                    )
                    evidence_list.append(evidence)
        
        # Sort by relevance score
        evidence_list.sort(key=lambda e: e.relevance_score, reverse=True)
        
        # Return top evidence
        return evidence_list[:self.max_sources_to_review]
    
    async def _retrieve_contradicting_evidence(self, diagnosis: str, context: Dict) -> List[ValidationEvidence]:
        """Retrieve evidence that contradicts or questions the diagnosis"""
        evidence_list = []
        
        # Look up contradictory evidence
        contradictory_entries = self.contradictory_evidence.get(diagnosis, [])
        
        for entry in contradictory_entries:
            relevance_score = self._calculate_relevance_score(entry, context)
            
            if relevance_score >= self.relevance_threshold:
                evidence = ValidationEvidence(
                    source=entry["source"],
                    title=entry["title"],
                    abstract=entry["abstract"],
                    relevance_score=relevance_score,
                    publication_year=entry["publication_year"],
                    study_type=entry["study_type"],
                    evidence_level=entry["evidence_level"]
                )
                evidence_list.append(evidence)
        
        return evidence_list
    
    def _calculate_relevance_score(self, entry: Dict, context: Dict) -> float:
        """Calculate relevance score for an evidence entry"""
        base_score = 0.7  # Base relevance
        
        # Adjust for evidence level
        evidence_level_bonus = {
            "A": 0.2,
            "B": 0.1,
            "C": 0.0
        }
        base_score += evidence_level_bonus.get(entry.get("evidence_level", "C"), 0.0)
        
        # Adjust for study type
        study_type_bonus = {
            "Clinical Guidelines": 0.15,
            "Systematic Review": 0.12,
            "Meta-Analysis": 0.12,
            "Randomized Controlled Trial": 0.10,
            "Clinical Review": 0.08,
            "Observational Study": 0.05
        }
        base_score += study_type_bonus.get(entry.get("study_type", ""), 0.0)
        
        # Adjust for publication recency
        current_year = datetime.now().year
        publication_year = entry.get("publication_year", 2020)
        years_old = current_year - publication_year
        
        if years_old <= 2:
            base_score += 0.1
        elif years_old <= 5:
            base_score += 0.05
        else:
            base_score -= 0.05
        
        # Context-specific adjustments
        if context.get("age") and context["age"] > 65:
            if "elderly" in entry.get("relevance_keywords", []):
                base_score += 0.1
        
        if context.get("gender"):
            if context["gender"] in entry.get("relevance_keywords", []):
                base_score += 0.1
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, base_score))
    
    async def _analyze_evidence(self, diagnosis: str, supporting_evidence: List[ValidationEvidence], 
                               contradicting_evidence: List[ValidationEvidence], context: Dict) -> Dict[str, Any]:
        """Analyze evidence and determine confidence adjustment"""
        
        # Calculate evidence strength
        supporting_strength = sum(e.relevance_score for e in supporting_evidence)
        contradicting_strength = sum(e.relevance_score for e in contradicting_evidence)
        
        # Calculate confidence adjustment
        if supporting_strength > contradicting_strength:
            confidence_adjustment = min(0.2, (supporting_strength - contradicting_strength) * 0.1)
            is_supported = True
        elif contradicting_strength > supporting_strength:
            confidence_adjustment = max(-0.2, -(contradicting_strength - supporting_strength) * 0.1)
            is_supported = False
        else:
            confidence_adjustment = 0.0
            is_supported = len(supporting_evidence) >= len(contradicting_evidence)
        
        # Adjust based on evidence quality
        high_quality_supporting = sum(1 for e in supporting_evidence if e.evidence_level in ["A", "B"])
        high_quality_contradicting = sum(1 for e in contradicting_evidence if e.evidence_level in ["A", "B"])
        
        if high_quality_supporting > high_quality_contradicting:
            confidence_adjustment += 0.05
        elif high_quality_contradicting > high_quality_supporting:
            confidence_adjustment -= 0.05
        
        return {
            "confidence_adjustment": confidence_adjustment,
            "is_supported": is_supported,
            "supporting_strength": supporting_strength,
            "contradicting_strength": contradicting_strength,
            "high_quality_supporting": high_quality_supporting,
            "high_quality_contradicting": high_quality_contradicting
        }
    
    def _generate_evidence_summary(self, supporting_evidence: List[ValidationEvidence], 
                                 contradicting_evidence: List[ValidationEvidence], 
                                 analysis_result: Dict) -> str:
        """Generate a summary of the evidence analysis"""
        summary_parts = []
        
        # Overall assessment
        if analysis_result["is_supported"]:
            summary_parts.append(f"The diagnosis is supported by {len(supporting_evidence)} relevant sources.")
        else:
            summary_parts.append(f"The diagnosis has limited support with {len(contradicting_evidence)} sources raising concerns.")
        
        # Evidence quality
        if analysis_result["high_quality_supporting"] > 0:
            summary_parts.append(f"High-quality evidence (Level A/B) supports this diagnosis from {analysis_result['high_quality_supporting']} sources.")
        
        if analysis_result["high_quality_contradicting"] > 0:
            summary_parts.append(f"However, {analysis_result['high_quality_contradicting']} high-quality sources suggest caution.")
        
        # Specific evidence highlights
        if supporting_evidence:
            best_evidence = max(supporting_evidence, key=lambda e: e.relevance_score)
            summary_parts.append(f"Strongest supporting evidence comes from {best_evidence.study_type.lower()} research.")
        
        return " ".join(summary_parts)
    
    def _generate_recommendations(self, diagnosis: str, analysis_result: Dict, context: Dict) -> List[str]:
        """Generate clinical recommendations based on evidence analysis"""
        recommendations = []
        
        if analysis_result["is_supported"]:
            if analysis_result["confidence_adjustment"] > 0.1:
                recommendations.append("Strong evidence supports this diagnosis. Proceed with appropriate treatment.")
            else:
                recommendations.append("Moderate evidence supports this diagnosis. Consider additional confirmatory testing.")
        else:
            recommendations.append("Limited evidence for this diagnosis. Consider alternative diagnoses.")
            recommendations.append("Additional diagnostic workup may be warranted.")
        
        # Diagnosis-specific recommendations
        if diagnosis == "pneumonia":
            recommendations.append("Consider chest imaging and respiratory cultures if not already obtained.")
            recommendations.append("Monitor for complications such as pleural effusion or sepsis.")
        elif diagnosis == "myocardial infarction":
            recommendations.append("Urgent cardiology consultation and cardiac catheterization evaluation.")
            recommendations.append("Serial cardiac biomarkers and ECGs recommended.")
        elif diagnosis == "diabetes mellitus":
            recommendations.append("Confirm with repeat glucose testing or HbA1c if not diagnostic.")
            recommendations.append("Assess for diabetic complications and cardiovascular risk factors.")
        
        # Context-specific recommendations
        if context.get("age") and context["age"] > 65:
            recommendations.append("Consider age-related complications and comorbidities in management.")
        
        return recommendations
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the validation model"""
        return {
            "name": self.agent_name,
            "version": self.version,
            "model_type": self.model_type,
            "capabilities": [
                "evidence_retrieval",
                "literature_analysis",
                "confidence_adjustment",
                "citation_generation",
                "recommendation_synthesis"
            ],
            "literature_sources": sum(len(sources) for sources in self.medical_literature.values()),
            "max_sources_reviewed": self.max_sources_to_review
        }
    
    async def update_literature_database(self, new_literature: Dict):
        """Update the medical literature database (for future RAG integration)"""
        for diagnosis, papers in new_literature.items():
            if diagnosis in self.medical_literature:
                self.medical_literature[diagnosis].extend(papers)
            else:
                self.medical_literature[diagnosis] = papers
        
        logger.info(f"Updated literature database with {sum(len(papers) for papers in new_literature.values())} new papers")
    
    async def search_literature_by_keywords(self, keywords: List[str], max_results: int = 5) -> List[ValidationEvidence]:
        """Search literature by keywords (simulates vector similarity search)"""
        all_evidence = []
        
        for diagnosis, papers in self.medical_literature.items():
            for paper in papers:
                # Simple keyword matching (would be replaced with vector similarity)
                relevance_score = 0.0
                for keyword in keywords:
                    if keyword.lower() in paper.get("relevance_keywords", []):
                        relevance_score += 0.2
                    if keyword.lower() in paper["title"].lower():
                        relevance_score += 0.3
                    if keyword.lower() in paper["abstract"].lower():
                        relevance_score += 0.1
                
                if relevance_score > 0.0:
                    evidence = ValidationEvidence(
                        source=paper["source"],
                        title=paper["title"],
                        abstract=paper["abstract"],
                        relevance_score=min(1.0, relevance_score),
                        publication_year=paper["publication_year"],
                        study_type=paper["study_type"],
                        evidence_level=paper["evidence_level"]
                    )
                    all_evidence.append(evidence)
        
        # Sort by relevance and return top results
        all_evidence.sort(key=lambda e: e.relevance_score, reverse=True)
        return all_evidence[:max_results]