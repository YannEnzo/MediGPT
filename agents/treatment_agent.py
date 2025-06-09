# agents/treatment_agent.py - Treatment Agent for Clinical Treatment Planning and Simulation
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from models import (
    EHRInput, TreatmentResponse, Medication, Treatment, SeverityLevel
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TreatmentAgent:
    """
    Treatment Agent that generates personalized treatment plans and simulates outcomes
    based on diagnosis and patient-specific factors.
    """
    
    def __init__(self):
        self.agent_name = "TreatmentAgent"
        self.version = "1.0.0"
        self.model_type = "rule_based_treatment"  # Will evolve to include ML/RL
        
        # Initialize treatment knowledge base
        self._load_treatment_protocols()
        
        logger.info(f"Initialized {self.agent_name} v{self.version}")
    
    def _load_treatment_protocols(self):
        """Load treatment protocols and medication databases"""
        
        # Treatment protocols by condition
        self.treatment_protocols = {
            "pneumonia": {
                "medications": [
                    {
                        "name": "Amoxicillin-Clavulanate",
                        "generic_name": "Amoxicillin/Clavulanic Acid",
                        "dosage": "875 mg/125 mg",
                        "frequency": "twice daily",
                        "duration": "7-10 days",
                        "route": "oral",
                        "indication": "Community-acquired pneumonia",
                        "contraindications": ["penicillin allergy"],
                        "side_effects": ["diarrhea", "nausea", "skin rash"],
                        "monitoring": ["liver function", "renal function"]
                    },
                    {
                        "name": "Azithromycin", 
                        "generic_name": "Azithromycin",
                        "dosage": "500 mg day 1, then 250 mg",
                        "frequency": "once daily",
                        "duration": "5 days",
                        "route": "oral",
                        "indication": "Atypical pneumonia coverage",
                        "contraindications": ["macrolide allergy", "QT prolongation"],
                        "side_effects": ["GI upset", "QT prolongation"],
                        "monitoring": ["ECG if cardiac risk factors"]
                    }
                ],
                "non_drug_treatments": [
                    {
                        "intervention_type": "supportive care",
                        "description": "Rest, increased fluid intake, and symptomatic relief",
                        "rationale": "Supports immune system and prevents dehydration",
                        "duration": "throughout illness",
                        "expected_outcome": "Improved comfort and faster recovery",
                        "success_probability": 0.9
                    },
                    {
                        "intervention_type": "respiratory support",
                        "description": "Supplemental oxygen if SpO2 < 90%",
                        "rationale": "Prevents hypoxemia and organ dysfunction",
                        "duration": "until SpO2 > 92% on room air",
                        "expected_outcome": "Maintained tissue oxygenation",
                        "success_probability": 0.95
                    }
                ],
                "recovery_timeline": "7-14 days with appropriate treatment",
                "complications": ["pleural effusion", "sepsis", "respiratory failure"],
                "follow_up": ["chest X-ray in 6-8 weeks", "symptom resolution check in 1 week"]
            },
            
            "myocardial infarction": {
                "medications": [
                    {
                        "name": "Aspirin",
                        "generic_name": "Acetylsalicylic Acid",
                        "dosage": "81 mg",
                        "frequency": "once daily",
                        "duration": "lifelong",
                        "route": "oral",
                        "indication": "Secondary prevention of cardiovascular events",
                        "contraindications": ["active bleeding", "severe asthma"],
                        "side_effects": ["GI bleeding", "bruising"],
                        "monitoring": ["bleeding signs", "platelet count"]
                    },
                    {
                        "name": "Atorvastatin",
                        "generic_name": "Atorvastatin",
                        "dosage": "80 mg",
                        "frequency": "once daily",
                        "duration": "lifelong",
                        "route": "oral",
                        "indication": "Cholesterol management and plaque stabilization",
                        "contraindications": ["active liver disease", "pregnancy"],
                        "side_effects": ["muscle pain", "liver enzyme elevation"],
                        "monitoring": ["liver function", "CK levels", "lipid panel"]
                    },
                    {
                        "name": "Metoprolol",
                        "generic_name": "Metoprolol Tartrate",
                        "dosage": "25 mg",
                        "frequency": "twice daily",
                        "duration": "lifelong",
                        "route": "oral",
                        "indication": "Cardioprotection and blood pressure control",
                        "contraindications": ["severe bradycardia", "cardiogenic shock"],
                        "side_effects": ["fatigue", "bradycardia", "hypotension"],
                        "monitoring": ["heart rate", "blood pressure", "exercise tolerance"]
                    }
                ],
                "non_drug_treatments": [
                    {
                        "intervention_type": "emergency intervention",
                        "description": "Cardiac catheterization and PCI if indicated",
                        "rationale": "Restore coronary blood flow and minimize myocardial damage",
                        "duration": "acute intervention",
                        "expected_outcome": "Restored coronary perfusion",
                        "success_probability": 0.85
                    },
                    {
                        "intervention_type": "cardiac rehabilitation",
                        "description": "Supervised exercise and education program",
                        "rationale": "Improve cardiovascular fitness and reduce future risk",
                        "duration": "12 weeks",
                        "expected_outcome": "Improved exercise capacity and quality of life",
                        "success_probability": 0.8
                    }
                ],
                "recovery_timeline": "6-8 weeks for initial recovery, lifelong management",
                "complications": ["heart failure", "arrhythmias", "recurrent MI"],
                "follow_up": ["cardiology follow-up in 1-2 weeks", "echo in 3 months"]
            },
            
            "hypertension": {
                "medications": [
                    {
                        "name": "Lisinopril",
                        "generic_name": "Lisinopril",
                        "dosage": "10 mg",
                        "frequency": "once daily",
                        "duration": "lifelong",
                        "route": "oral",
                        "indication": "Blood pressure control",
                        "contraindications": ["pregnancy", "angioedema history"],
                        "side_effects": ["dry cough", "hyperkalemia", "angioedema"],
                        "monitoring": ["blood pressure", "creatinine", "potassium"]
                    },
                    {
                        "name": "Amlodipine",
                        "generic_name": "Amlodipine Besylate",
                        "dosage": "5 mg",
                        "frequency": "once daily",
                        "duration": "lifelong",
                        "route": "oral",
                        "indication": "Blood pressure control",
                        "contraindications": ["severe aortic stenosis"],
                        "side_effects": ["peripheral edema", "dizziness", "flushing"],
                        "monitoring": ["blood pressure", "edema", "heart rate"]
                    }
                ],
                "non_drug_treatments": [
                    {
                        "intervention_type": "lifestyle modification",
                        "description": "DASH diet, weight loss, regular exercise, sodium restriction",
                        "rationale": "Reduce cardiovascular risk and blood pressure naturally",
                        "duration": "lifelong",
                        "expected_outcome": "5-10 mmHg reduction in blood pressure",
                        "success_probability": 0.7
                    }
                ],
                "recovery_timeline": "Blood pressure control typically achieved in 4-6 weeks",
                "complications": ["stroke", "heart disease", "kidney disease"],
                "follow_up": ["blood pressure monitoring weekly initially", "follow-up in 4 weeks"]
            },
            
            "diabetes mellitus": {
                "medications": [
                    {
                        "name": "Metformin",
                        "generic_name": "Metformin Hydrochloride",
                        "dosage": "500 mg",
                        "frequency": "twice daily with meals",
                        "duration": "lifelong",
                        "route": "oral",
                        "indication": "Glucose control and insulin sensitivity",
                        "contraindications": ["severe kidney disease", "metabolic acidosis"],
                        "side_effects": ["GI upset", "lactic acidosis (rare)", "B12 deficiency"],
                        "monitoring": ["glucose", "HbA1c", "creatinine", "B12 levels"]
                    }
                ],
                "non_drug_treatments": [
                    {
                        "intervention_type": "diabetes education",
                        "description": "Comprehensive diabetes self-management education",
                        "rationale": "Empower patient for effective self-care",
                        "duration": "initial education plus ongoing support",
                        "expected_outcome": "Improved diabetes knowledge and self-care",
                        "success_probability": 0.8
                    },
                    {
                        "intervention_type": "nutritional counseling",
                        "description": "Medical nutrition therapy with registered dietitian",
                        "rationale": "Optimize dietary management of diabetes",
                        "duration": "initial sessions plus follow-up",
                        "expected_outcome": "Better glucose control through diet",
                        "success_probability": 0.75
                    }
                ],
                "recovery_timeline": "Lifelong management, glucose control typically achieved in 3-6 months",
                "complications": ["diabetic ketoacidosis", "neuropathy", "nephropathy", "retinopathy"],
                "follow_up": ["HbA1c every 3 months", "annual eye exam", "annual foot exam"]
            },
            
            "influenza": {
                "medications": [
                    {
                        "name": "Oseltamivir",
                        "generic_name": "Oseltamivir Phosphate",
                        "dosage": "75 mg",
                        "frequency": "twice daily",
                        "duration": "5 days",
                        "route": "oral",
                        "indication": "Antiviral treatment for influenza",
                        "contraindications": ["severe kidney disease"],
                        "side_effects": ["nausea", "vomiting", "headache"],
                        "monitoring": ["symptom improvement", "kidney function"]
                    }
                ],
                "non_drug_treatments": [
                    {
                        "intervention_type": "supportive care",
                        "description": "Rest, fluids, acetaminophen for fever/aches",
                        "rationale": "Symptom relief and recovery support",
                        "duration": "7-10 days",
                        "expected_outcome": "Symptomatic improvement",
                        "success_probability": 0.9
                    }
                ],
                "recovery_timeline": "7-10 days for most patients",
                "complications": ["pneumonia", "myocarditis", "encephalitis"],
                "follow_up": ["return if symptoms worsen", "routine follow-up typically not needed"]
            }
        }
        
        # Risk stratification factors
        self.risk_factors = {
            "age_high_risk": 65,
            "cardiovascular_conditions": ["heart disease", "hypertension", "diabetes"],
            "respiratory_conditions": ["asthma", "COPD", "lung disease"],
            "immunocompromising_conditions": ["cancer", "HIV", "immunosuppression"]
        }
    
    async def generate_treatment_plan(self, diagnosis: str, patient_data: EHRInput) -> TreatmentResponse:
        """
        Generate a comprehensive treatment plan for the given diagnosis and patient
        
        Args:
            diagnosis: Primary diagnosis
            patient_data: Patient EHR data for personalization
            
        Returns:
            TreatmentResponse with complete treatment plan
        """
        logger.info(f"Generating treatment plan for {diagnosis} - Patient {patient_data.patient_id}")
        
        start_time = datetime.now()
        
        try:
            # Normalize diagnosis
            normalized_diagnosis = diagnosis.lower().strip()
            
            # Get base treatment protocol
            protocol = self.treatment_protocols.get(normalized_diagnosis)
            if not protocol:
                # Return generic supportive care if no specific protocol
                return await self._generate_generic_treatment_plan(diagnosis, patient_data)
            
            # Personalize medications based on patient factors
            personalized_medications = await self._personalize_medications(
                protocol["medications"], patient_data
            )
            
            # Personalize non-drug treatments
            personalized_treatments = await self._personalize_treatments(
                protocol["non_drug_treatments"], patient_data
            )
            
            # Generate clinical plans
            immediate_actions = self._generate_immediate_actions(normalized_diagnosis, patient_data)
            short_term_plan = self._generate_short_term_plan(normalized_diagnosis, patient_data)
            long_term_plan = self._generate_long_term_plan(normalized_diagnosis, patient_data)
            
            # Assess prognosis and recovery timeline
            prognosis_info = self._assess_prognosis(normalized_diagnosis, patient_data)
            
            # Generate patient education and lifestyle recommendations
            lifestyle_modifications = self._generate_lifestyle_recommendations(normalized_diagnosis, patient_data)
            
            # Generate warning signs and follow-up instructions
            warning_signs = self._generate_warning_signs(normalized_diagnosis, patient_data)
            when_to_seek_care = self._generate_care_seeking_instructions(normalized_diagnosis, patient_data)
            
            # Identify personalization factors used
            personalization_factors = self._identify_personalization_factors(patient_data)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.info(f"Generated treatment plan for {diagnosis} in {processing_time:.2f}ms")
            
            return TreatmentResponse(
                diagnosis=diagnosis,
                patient_id=patient_data.patient_id,
                medications=personalized_medications,
                treatments=personalized_treatments,
                immediate_actions=immediate_actions,
                short_term_plan=short_term_plan,
                long_term_plan=long_term_plan,
                expected_recovery_time=prognosis_info["recovery_time"],
                prognosis=prognosis_info["prognosis"],
                risk_factors=prognosis_info["risk_factors"],
                lifestyle_modifications=lifestyle_modifications,
                warning_signs=warning_signs,
                when_to_seek_care=when_to_seek_care,
                generated_at=datetime.now(),
                treatment_agent_version=self.version,
                personalization_factors=personalization_factors
            )
            
        except Exception as e:
            logger.error(f"Error generating treatment plan for {diagnosis}: {str(e)}")
            raise
    
    async def _personalize_medications(self, base_medications: List[Dict], patient_data: EHRInput) -> List[Medication]:
        """Personalize medications based on patient factors"""
        personalized_meds = []
        
        for med_data in base_medications:
            # Check for contraindications
            if self._has_contraindications(med_data, patient_data):
                # Skip this medication or find alternative
                alternative = self._find_alternative_medication(med_data, patient_data)
                if alternative:
                    med_data = alternative
                else:
                    continue
            
            # Adjust dosing for age, weight, kidney function
            adjusted_dosage = self._adjust_dosage(med_data, patient_data)
            
            medication = Medication(
                name=med_data["name"],
                generic_name=med_data.get("generic_name"),
                dosage=adjusted_dosage,
                frequency=med_data["frequency"],
                duration=med_data["duration"],
                route=med_data["route"],
                indication=med_data["indication"],
                contraindications=med_data["contraindications"],
                side_effects=med_data["side_effects"],
                monitoring_required=med_data["monitoring"]
            )
            
            personalized_meds.append(medication)
        
        return personalized_meds
    
    async def _personalize_treatments(self, base_treatments: List[Dict], patient_data: EHRInput) -> List[Treatment]:
        """Personalize non-medication treatments"""
        personalized_treatments = []
        
        for treatment_data in base_treatments:
            # Adjust based on patient factors
            risk_level = self._assess_treatment_risk(treatment_data, patient_data)
            success_probability = self._adjust_success_probability(treatment_data, patient_data)
            
            treatment = Treatment(
                intervention_type=treatment_data["intervention_type"],
                description=treatment_data["description"],
                rationale=treatment_data["rationale"],
                start_immediately=treatment_data.get("start_immediately", True),
                duration=treatment_data.get("duration"),
                follow_up_schedule=treatment_data.get("follow_up_schedule", []),
                expected_outcome=treatment_data["expected_outcome"],
                success_probability=success_probability,
                risk_level=risk_level,
                complications=treatment_data.get("complications", [])
            )
            
            personalized_treatments.append(treatment)
        
        return personalized_treatments
    
    def _has_contraindications(self, med_data: Dict, patient_data: EHRInput) -> bool:
        """Check if patient has contraindications for this medication"""
        contraindications = med_data.get("contraindications", [])
        
        # Check allergies
        for allergy in patient_data.allergies:
            for contraindication in contraindications:
                if allergy.lower() in contraindication.lower():
                    return True
        
        # Check medical history
        for condition in patient_data.medical_history:
            for contraindication in contraindications:
                if condition.lower() in contraindication.lower():
                    return True
        
        # Age-specific contraindications
        if patient_data.age > 75 and "elderly caution" in contraindications:
            return True
        
        # Pregnancy contraindications
        if patient_data.gender.value == "female" and "pregnancy" in contraindications:
            # In real system, would need pregnancy status
            pass
        
        return False
    
    def _find_alternative_medication(self, original_med: Dict, patient_data: EHRInput) -> Optional[Dict]:
        """Find alternative medication if contraindication exists"""
        # Simplified alternative medication logic
        alternatives = {
            "Amoxicillin-Clavulanate": {
                "name": "Cephalexin",
                "generic_name": "Cephalexin",
                "dosage": "500 mg",
                "frequency": "four times daily",
                "duration": "7-10 days",
                "route": "oral",
                "indication": "Alternative antibiotic for beta-lactam allergic patients",
                "contraindications": ["cephalosporin allergy"],
                "side_effects": ["diarrhea", "nausea"],
                "monitoring": ["renal function"]
            }
        }
        
        return alternatives.get(original_med["name"])
    
    def _adjust_dosage(self, med_data: Dict, patient_data: EHRInput) -> str:
        """Adjust medication dosage based on patient factors"""
        base_dosage = med_data["dosage"]
        
        # Age-based adjustments
        if patient_data.age > 75:
            if "mg" in base_dosage:
                # Simplified dose reduction for elderly
                return f"Reduced dose: {base_dosage} (consider 50-75% of standard dose)"
        
        # Kidney function adjustments (would need actual lab values)
        if patient_data.labs and hasattr(patient_data.labs, 'creatinine') and patient_data.labs.creatinine:
            if patient_data.labs.creatinine > 1.5:
                return f"Renally adjusted: {base_dosage} (adjust for kidney function)"
        
        return base_dosage
    
    def _assess_treatment_risk(self, treatment_data: Dict, patient_data: EHRInput) -> SeverityLevel:
        """Assess risk level for a treatment based on patient factors"""
        base_risk = SeverityLevel.LOW
        
        # Age increases risk
        if patient_data.age > 75:
            if base_risk == SeverityLevel.LOW:
                base_risk = SeverityLevel.MODERATE
        
        # Comorbidities increase risk
        high_risk_conditions = ["heart failure", "kidney disease", "liver disease"]
        if any(condition in " ".join(patient_data.medical_history).lower() 
               for condition in high_risk_conditions):
            if base_risk == SeverityLevel.LOW:
                base_risk = SeverityLevel.MODERATE
            elif base_risk == SeverityLevel.MODERATE:
                base_risk = SeverityLevel.HIGH
        
        return base_risk
    
    def _adjust_success_probability(self, treatment_data: Dict, patient_data: EHRInput) -> float:
        """Adjust success probability based on patient factors"""
        base_probability = treatment_data.get("success_probability", 0.7)
        
        # Age adjustments
        if patient_data.age > 75:
            base_probability *= 0.9  # Slightly lower success in elderly
        
        # Comorbidity adjustments
        if len(patient_data.medical_history) > 3:
            base_probability *= 0.85  # Multiple comorbidities reduce success
        
        return max(0.0, min(1.0, base_probability))
    
    def _generate_immediate_actions(self, diagnosis: str, patient_data: EHRInput) -> List[str]:
        """Generate immediate actions based on diagnosis and severity"""
        actions = []
        
        if diagnosis == "myocardial infarction":
            actions.extend([
                "Administer aspirin 325mg immediately if not contraindicated",
                "Obtain 12-lead ECG immediately",
                "Start continuous cardiac monitoring",
                "Establish IV access and draw cardiac biomarkers",
                "Contact cardiology for urgent consultation"
            ])
        elif diagnosis == "pneumonia":
            actions.extend([
                "Obtain chest X-ray",
                "Draw blood cultures before antibiotics",
                "Start empiric antibiotic therapy",
                "Monitor oxygen saturation"
            ])
        elif diagnosis == "hypertension":
            actions.extend([
                "Confirm blood pressure with manual cuff",
                "Assess for target organ damage",
                "Review current medications"
            ])
        else:
            actions.append("Complete comprehensive assessment")
            actions.append("Initiate appropriate monitoring")
        
        return actions
    
    def _generate_short_term_plan(self, diagnosis: str, patient_data: EHRInput) -> List[str]:
        """Generate short-term management plan (days to weeks)"""
        plan = []
        
        if diagnosis == "pneumonia":
            plan.extend([
                "Complete antibiotic course as prescribed",
                "Monitor for symptom improvement",
                "Return if fever persists beyond 48-72 hours",
                "Gradual return to normal activities"
            ])
        elif diagnosis == "diabetes mellitus":
            plan.extend([
                "Begin glucose monitoring 4 times daily",
                "Start diabetes education classes",
                "Schedule nutrition counseling",
                "Establish follow-up with endocrinologist"
            ])
        
        return plan
    
    def _generate_long_term_plan(self, diagnosis: str, patient_data: EHRInput) -> List[str]:
        """Generate long-term management plan (months to years)"""
        plan = []
        
        if diagnosis == "diabetes mellitus":
            plan.extend([
                "HbA1c monitoring every 3 months",
                "Annual diabetic eye exam",
                "Annual foot examination",
                "Cardiovascular risk assessment annually"
            ])
        elif diagnosis == "hypertension":
            plan.extend([
                "Regular blood pressure monitoring",
                "Annual cardiovascular risk assessment",
                "Medication adherence monitoring",
                "Lifestyle modification support"
            ])
        
        return plan
    
    def _assess_prognosis(self, diagnosis: str, patient_data: EHRInput) -> Dict[str, Any]:
        """Assess prognosis and recovery timeline"""
        protocol = self.treatment_protocols.get(diagnosis, {})
        
        base_recovery_time = protocol.get("recovery_timeline", "Variable recovery time")
        base_complications = protocol.get("complications", [])
        
        # Adjust for patient factors
        risk_factors = []
        prognosis = "Good"
        
        if patient_data.age > 75:
            risk_factors.append("Advanced age")
            prognosis = "Fair to good with close monitoring"
        
        if len(patient_data.medical_history) > 2:
            risk_factors.append("Multiple comorbidities")
            if prognosis == "Good":
                prognosis = "Fair"
        
        # Diagnosis-specific adjustments
        if diagnosis == "myocardial infarction":
            if patient_data.age > 65:
                prognosis = "Guarded, requires intensive management"
        
        return {
            "recovery_time": base_recovery_time,
            "prognosis": prognosis,
            "risk_factors": risk_factors
        }
    
    def _generate_lifestyle_recommendations(self, diagnosis: str, patient_data: EHRInput) -> List[str]:
        """Generate lifestyle modification recommendations"""
        recommendations = []
        
        common_recommendations = [
            "Maintain regular sleep schedule (7-9 hours nightly)",
            "Stay adequately hydrated",
            "Avoid smoking and limit alcohol consumption"
        ]
        
        if diagnosis in ["hypertension", "diabetes mellitus", "myocardial infarction"]:
            recommendations.extend([
                "Follow heart-healthy diet (DASH or Mediterranean)",
                "Engage in regular moderate exercise (150 min/week)",
                "Maintain healthy weight (BMI 18.5-24.9)",
                "Manage stress through relaxation techniques"
            ])
        
        if diagnosis == "diabetes mellitus":
            recommendations.extend([
                "Monitor carbohydrate intake",
                "Regular foot care and inspection",
                "Maintain good dental hygiene"
            ])
        
        recommendations.extend(common_recommendations)
        return recommendations
    
    def _generate_warning_signs(self, diagnosis: str, patient_data: EHRInput) -> List[str]:
        """Generate warning signs to watch for"""
        warning_signs = []
        
        if diagnosis == "pneumonia":
            warning_signs.extend([
                "Worsening shortness of breath",
                "Chest pain",
                "High fever (>102°F) persisting after 48 hours of antibiotics",
                "Confusion or altered mental status"
            ])
        elif diagnosis == "myocardial infarction":
            warning_signs.extend([
                "Recurrent chest pain",
                "New or worsening shortness of breath",
                "Swelling in legs or ankles",
                "Irregular heartbeat"
            ])
        elif diagnosis == "diabetes mellitus":
            warning_signs.extend([
                "Blood glucose consistently >300 mg/dL",
                "Persistent nausea and vomiting",
                "Fruity breath odor",
                "Rapid breathing"
            ])
        
        return warning_signs
    
    def _generate_care_seeking_instructions(self, diagnosis: str, patient_data: EHRInput) -> List[str]:
        """Generate instructions for when to seek immediate care"""
        instructions = []
        
        emergency_signs = [
            "Severe chest pain",
            "Difficulty breathing or severe shortness of breath",
            "Loss of consciousness",
            "Severe allergic reaction to medications"
        ]
        
        if diagnosis == "myocardial infarction":
            instructions.extend([
                "Call 911 immediately for any chest pain lasting >5 minutes",
                "Seek immediate care for new or worsening heart failure symptoms"
            ])
        
        instructions.extend([
            f"Seek immediate emergency care for: {', '.join(emergency_signs)}",
            "Contact healthcare provider for medication side effects",
            "Schedule routine follow-up as recommended"
        ])
        
        return instructions
    
    def _identify_personalization_factors(self, patient_data: EHRInput) -> List[str]:
        """Identify factors used for treatment personalization"""
        factors = []
        
        factors.append(f"Age: {patient_data.age} years")
        factors.append(f"Gender: {patient_data.gender.value}")
        
        if patient_data.medical_history:
            factors.append(f"Medical history: {', '.join(patient_data.medical_history[:3])}")
        
        if patient_data.allergies:
            factors.append(f"Allergies: {', '.join(patient_data.allergies)}")
        
        if patient_data.medications:
            factors.append(f"Current medications: {len(patient_data.medications)} medications")
        
        return factors
    
    async def _generate_generic_treatment_plan(self, diagnosis: str, patient_data: EHRInput) -> TreatmentResponse:
        """Generate generic treatment plan when no specific protocol exists"""
        logger.warning(f"No specific protocol for {diagnosis}, generating generic plan")
        
        # Generic supportive care
        generic_treatment = Treatment(
            intervention_type="supportive care",
            description="Symptom management and supportive care",
            rationale="Provide comfort and support natural healing",
            start_immediately=True,
            duration="as needed",
            expected_outcome="Symptom relief and gradual improvement",
            success_probability=0.7,
            risk_level=SeverityLevel.LOW,
            complications=["symptom progression"]
        )
        
        return TreatmentResponse(
            diagnosis=diagnosis,
            patient_id=patient_data.patient_id,
            medications=[],
            treatments=[generic_treatment],
            immediate_actions=["Complete assessment", "Symptom monitoring"],
            short_term_plan=["Supportive care", "Monitor symptoms"],
            long_term_plan=["Follow-up as needed"],
            expected_recovery_time="Variable",
            prognosis="Depends on underlying condition",
            risk_factors=["Unknown etiology"],
            lifestyle_modifications=["Rest", "Adequate hydration", "Healthy diet"],
            warning_signs=["Worsening symptoms", "New concerning symptoms"],
            when_to_seek_care=["If symptoms worsen or persist"],
            generated_at=datetime.now(),
            treatment_agent_version=self.version,
            personalization_factors=[f"Age: {patient_data.age}", f"Gender: {patient_data.gender.value}"]
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the treatment model"""
        return {
            "name": self.agent_name,
            "version": self.version,
            "model_type": self.model_type,
            "capabilities": [
                "treatment_planning",
                "medication_selection",
                "dosage_adjustment",
                "personalization",
                "risk_assessment",
                "outcome_prediction",
                "patient_education"
            ],
            "supported_conditions": list(self.treatment_protocols.keys()),
            "personalization_factors": [
                "age", "gender", "medical_history", "allergies", 
                "current_medications", "lab_values"
            ]
        }
    
    async def add_treatment_protocol(self, condition: str, protocol: Dict):
        """Add new treatment protocol for a condition"""
        self.treatment_protocols[condition] = protocol
        logger.info(f"Added treatment protocol for {condition}")
    
    async def simulate_treatment_outcome(self, treatment_plan: TreatmentResponse, 
                                       days_forward: int = 30) -> Dict[str, Any]:
        """Simulate treatment outcome over specified time period (future enhancement)"""
        # This would integrate with outcome prediction models
        # For now, return simplified simulation
        
        return {
            "simulation_days": days_forward,
            "predicted_outcome": "improvement expected",
            "adherence_factors": ["medication complexity", "side effects"],
            "risk_events": [],
            "follow_up_recommended": True
        }