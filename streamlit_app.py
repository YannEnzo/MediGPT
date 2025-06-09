# streamlit_app.py - Streamlit Frontend for MediGPT Agents
import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, List, Any
import time

# Configure Streamlit page
st.set_page_config(
    page_title="MediGPT Agents",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_BASE_URL = "http://localhost:8000"

class MediGPTInterface:
    """Streamlit interface for MediGPT Agents system"""
    
    def __init__(self):
        self.api_base = API_BASE_URL
        
    def check_api_connection(self):
        """Check if API is running"""
        try:
            response = requests.get(f"{self.api_base}/", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def call_diagnose_api(self, patient_data: Dict) -> Dict:
        """Call the diagnose endpoint"""
        try:
            response = requests.post(
                f"{self.api_base}/diagnose",
                json=patient_data,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            return None
    
    def call_validate_api(self, diagnosis: str, age: int = None, gender: str = None) -> Dict:
        """Call the validate endpoint"""
        try:
            params = {}
            if age:
                params["patient_age"] = age
            if gender:
                params["patient_gender"] = gender
                
            response = requests.get(
                f"{self.api_base}/validate/{diagnosis}",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Validation API Error: {str(e)}")
            return None
    
    def call_treatment_api(self, diagnosis: str, patient_data: Dict) -> Dict:
        """Call the treatment simulation endpoint"""
        try:
            payload = {
                "diagnosis": diagnosis,
                "patient_data": patient_data
            }
            response = requests.post(
                f"{self.api_base}/simulate",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Treatment API Error: {str(e)}")
            return None

def main():
    """Main Streamlit application"""
    
    # Initialize interface
    interface = MediGPTInterface()
    
    # Header
    st.title("🏥 MediGPT Agents")
    st.markdown("### AI-Powered Clinical Decision Support System")
    
    # Check API connection
    if not interface.check_api_connection():
        st.error("⚠️ Cannot connect to MediGPT API. Please ensure the FastAPI server is running on http://localhost:8000")
        st.info("Run: `uvicorn main:app --reload` to start the API server")
        return
    
    st.success("✅ Connected to MediGPT API")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    mode = st.sidebar.selectbox(
        "Select Mode",
        ["Patient Analysis", "Diagnosis Validation", "Treatment Planning", "Batch Processing", "Agent Status"]
    )
    
    if mode == "Patient Analysis":
        render_patient_analysis(interface)
    elif mode == "Diagnosis Validation":
        render_diagnosis_validation(interface)
    elif mode == "Treatment Planning":
        render_treatment_planning(interface)
    elif mode == "Batch Processing":
        render_batch_processing(interface)
    elif mode == "Agent Status":
        render_agent_status(interface)

def render_patient_analysis(interface):
    """Render patient analysis interface"""
    st.header("📋 Patient Analysis")
    
    # Sample patient templates
    st.sidebar.subheader("Quick Templates")
    template = st.sidebar.selectbox(
        "Load Sample Patient",
        ["Custom", "Pneumonia Case", "Heart Attack Case", "Diabetes Case", "Hypertension Case", "Flu Case"]
    )
    
    # Get sample data
    sample_data = get_sample_patient_data(template)
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Patient Demographics")
        patient_id = st.text_input("Patient ID", value=sample_data.get("patient_id", "PAT001"))
        age = st.number_input("Age", min_value=0, max_value=120, value=sample_data.get("age", 35))
        gender = st.selectbox("Gender", ["male", "female", "other", "unknown"], 
                             index=["male", "female", "other", "unknown"].index(sample_data.get("gender", "female")))
        
        st.subheader("Clinical Presentation")
        chief_complaint = st.text_area("Chief Complaint", 
                                      value=sample_data.get("chief_complaint", ""))
        
        # Symptoms input
        symptoms_text = st.text_area(
            "Symptoms (one per line)", 
            value="\n".join(sample_data.get("symptoms", [])),
            height=100
        )
        symptoms = [s.strip() for s in symptoms_text.split("\n") if s.strip()]
        
        symptom_duration = st.text_input("Symptom Duration", 
                                        value=sample_data.get("symptom_duration", ""))
    
    with col2:
        st.subheader("Vital Signs")
        col2a, col2b = st.columns(2)
        
        with col2a:
            heart_rate = st.number_input("Heart Rate (BPM)", min_value=30, max_value=300, 
                                       value=sample_data.get("vitals", {}).get("heart_rate"))
            systolic_bp = st.number_input("Systolic BP", min_value=60, max_value=300,
                                        value=sample_data.get("vitals", {}).get("systolic_bp"))
            temperature = st.number_input("Temperature (°F)", min_value=90.0, max_value=115.0,
                                        value=sample_data.get("vitals", {}).get("temperature"))
        
        with col2b:
            respiratory_rate = st.number_input("Respiratory Rate", min_value=8, max_value=60,
                                             value=sample_data.get("vitals", {}).get("respiratory_rate"))
            diastolic_bp = st.number_input("Diastolic BP", min_value=30, max_value=200,
                                         value=sample_data.get("vitals", {}).get("diastolic_bp"))
            oxygen_saturation = st.number_input("O2 Saturation (%)", min_value=70.0, max_value=100.0,
                                               value=sample_data.get("vitals", {}).get("oxygen_saturation"))
        
        st.subheader("Laboratory Results")
        col2c, col2d = st.columns(2)
        
        with col2c:
            wbc = st.number_input("WBC (K/uL)", min_value=0.0, 
                                value=sample_data.get("labs", {}).get("wbc"))
            hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0,
                                       value=sample_data.get("labs", {}).get("hemoglobin"))
            glucose = st.number_input("Glucose (mg/dL)", min_value=0.0,
                                    value=sample_data.get("labs", {}).get("glucose"))
        
        with col2d:
            crp = st.number_input("CRP (mg/L)", min_value=0.0,
                                value=sample_data.get("labs", {}).get("crp"))
            creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.0,
                                       value=sample_data.get("labs", {}).get("creatinine"))
            platelets = st.number_input("Platelets (K/uL)", min_value=0.0,
                                      value=sample_data.get("labs", {}).get("platelets"))
    
    # Medical History
    st.subheader("Medical History")
    col3, col4 = st.columns(2)
    
    with col3:
        medical_history_text = st.text_area(
            "Past Medical History (one per line)",
            value="\n".join(sample_data.get("medical_history", [])),
            height=80
        )
        medical_history = [h.strip() for h in medical_history_text.split("\n") if h.strip()]
        
        medications_text = st.text_area(
            "Current Medications (one per line)",
            value="\n".join(sample_data.get("medications", [])),
            height=80
        )
        medications = [m.strip() for m in medications_text.split("\n") if m.strip()]
    
    with col4:
        allergies_text = st.text_area(
            "Allergies (one per line)",
            value="\n".join(sample_data.get("allergies", [])),
            height=80
        )
        allergies = [a.strip() for a in allergies_text.split("\n") if a.strip()]
        
        notes = st.text_area("Additional Notes", 
                           value=sample_data.get("notes", ""))
    
    # Analyze button
    if st.button("🔍 Analyze Patient", type="primary"):
        # Construct patient data
        patient_data = {
            "patient_id": patient_id,
            "age": age,
            "gender": gender,
            "chief_complaint": chief_complaint,
            "symptoms": symptoms,
            "symptom_duration": symptom_duration,
            "medical_history": medical_history,
            "medications": medications,
            "allergies": allergies,
            "notes": notes
        }
        
        # Add vitals if provided
        vitals = {}
        if heart_rate: vitals["heart_rate"] = heart_rate
        if systolic_bp: vitals["systolic_bp"] = systolic_bp
        if diastolic_bp: vitals["diastolic_bp"] = diastolic_bp
        if temperature: vitals["temperature"] = temperature
        if respiratory_rate: vitals["respiratory_rate"] = respiratory_rate
        if oxygen_saturation: vitals["oxygen_saturation"] = oxygen_saturation
        
        if vitals:
            patient_data["vitals"] = vitals
        
        # Add labs if provided
        labs = {}
        if wbc: labs["wbc"] = wbc
        if hemoglobin: labs["hemoglobin"] = hemoglobin
        if glucose: labs["glucose"] = glucose
        if crp: labs["crp"] = crp
        if creatinine: labs["creatinine"] = creatinine
        if platelets: labs["platelets"] = platelets
        
        if labs:
            patient_data["labs"] = labs
        
        # Call API
        with st.spinner("Analyzing patient data..."):
            result = interface.call_diagnose_api(patient_data)
        
        if result:
            display_diagnosis_results(result)

def display_diagnosis_results(result):
    """Display diagnosis results"""
    st.header("🔬 Diagnosis Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Patient ID", result["patient_id"])
    with col2:
        st.metric("Total Diagnoses", len(result["diagnoses"]))
    with col3:
        high_conf_count = sum(1 for d in result["diagnoses"] if d["confidence"] > 0.7)
        st.metric("High Confidence", high_conf_count)
    with col4:
        processing_time = result.get("processing_time_ms")
        if processing_time is not None:
            st.metric("Processing Time", f"{processing_time:.1f}ms")
        else:
            st.metric("Processing Time", "< 1ms")
    
    # Diagnosis table
    st.subheader("Differential Diagnosis")
    
    # Create DataFrame for better display
    diagnosis_data = []
    for d in result["diagnoses"]:
        diagnosis_data.append({
            "Condition": d["condition"],
            "ICD-10": d.get("icd10_code", "N/A"),
            "Confidence": f"{d['confidence']:.1%}",
            "Confidence Level": d["confidence_level"].replace("_", " ").title(),
            "Severity": d["severity"].title(),
            "Urgency": d["urgency"].title()
        })
    
    df = pd.DataFrame(diagnosis_data)
    st.dataframe(df, use_container_width=True)
    
    # Confidence visualization
    st.subheader("Confidence Distribution")
    
    # Create confidence chart
    conditions = [d["condition"] for d in result["diagnoses"]]
    confidences = [d["confidence"] for d in result["diagnoses"]]
    
    fig = px.bar(
        x=conditions,
        y=confidences,
        title="Diagnosis Confidence Scores",
        labels={"x": "Condition", "y": "Confidence"},
        color=confidences,
        color_continuous_scale="RdYlGn"
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed diagnosis information
    st.subheader("Detailed Analysis")
    
    for i, diagnosis in enumerate(result["diagnoses"]):
        with st.expander(f"{diagnosis['condition']} (Confidence: {diagnosis['confidence']:.1%})"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Supporting Evidence:**")
                for evidence in diagnosis.get("supporting_evidence", []):
                    st.write(f"• {evidence}")
                
                st.write(f"**Severity:** {diagnosis['severity'].title()}")
                st.write(f"**Urgency:** {diagnosis['urgency'].title()}")
            
            with col2:
                st.write("**Differential Notes:**")
                st.write(diagnosis.get("differential_notes", "No additional notes"))
                
                if diagnosis.get("prevalence_info"):
                    st.write("**Prevalence:**")
                    st.write(diagnosis["prevalence_info"])

def render_diagnosis_validation(interface):
    """Render diagnosis validation interface"""
    st.header("✅ Diagnosis Validation")
    
    # Input form
    col1, col2 = st.columns([2, 1])
    
    with col1:
        diagnosis = st.text_input("Diagnosis to Validate", 
                                value="pneumonia",
                                help="Enter the diagnosis you want to validate against medical literature")
    
    with col2:
        age = st.number_input("Patient Age (optional)", min_value=0, max_value=120, value=None)
        gender = st.selectbox("Patient Gender (optional)", 
                            ["", "male", "female", "other"])
    
    if st.button("🔍 Validate Diagnosis", type="primary"):
        if diagnosis:
            with st.spinner("Validating against medical literature..."):
                result = interface.call_validate_api(
                    diagnosis, 
                    age if age else None, 
                    gender if gender else None
                )
            
            if result:
                display_validation_results(result)
        else:
            st.error("Please enter a diagnosis to validate")

def display_validation_results(result):
    """Display validation results"""
    st.header("📚 Validation Results")
    
    # Summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_color = "green" if result["is_supported"] else "red"
        st.markdown(f"**Diagnosis:** {result['diagnosis']}")
        st.markdown(f"**Supported:** :{status_color}[{result['is_supported']}]")
    
    with col2:
        adjustment = result["confidence_adjustment"]
        adjustment_color = "green" if adjustment > 0 else "red" if adjustment < 0 else "gray"
        st.markdown(f"**Confidence Adjustment:** :{adjustment_color}[{adjustment:+.2f}]")
        st.markdown(f"**Final Confidence:** {result['final_confidence']:.1%}")
    
    with col3:
        st.markdown(f"**Sources Reviewed:** {result['total_sources_reviewed']}")
        st.markdown(f"**Validated:** {result['validated_at'][:19]}")
    
    # Evidence summary
    st.subheader("Evidence Summary")
    st.write(result["evidence_summary"])
    
    # Supporting evidence
    if result["supporting_evidence"]:
        st.subheader("📈 Supporting Evidence")
        for i, evidence in enumerate(result["supporting_evidence"]):
            with st.expander(f"{evidence['title']} (Relevance: {evidence['relevance_score']:.1%})"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Abstract:** {evidence['abstract']}")
                with col2:
                    st.write(f"**Source:** {evidence['source']}")
                    st.write(f"**Year:** {evidence.get('publication_year', 'N/A')}")
                    st.write(f"**Study Type:** {evidence.get('study_type', 'N/A')}")
                    st.write(f"**Evidence Level:** {evidence.get('evidence_level', 'N/A')}")
    
    # Contradicting evidence
    if result["contradicting_evidence"]:
        st.subheader("📉 Contradicting Evidence")
        for evidence in result["contradicting_evidence"]:
            with st.expander(f"{evidence['title']} (Relevance: {evidence['relevance_score']:.1%})"):
                st.write(f"**Abstract:** {evidence['abstract']}")
                st.write(f"**Source:** {evidence['source']}")
    
    # Recommendations
    if result["recommendations"]:
        st.subheader("💡 Clinical Recommendations")
        for rec in result["recommendations"]:
            st.write(f"• {rec}")

def render_treatment_planning(interface):
    """Render treatment planning interface"""
    st.header("💊 Treatment Planning")
    st.info("Generate personalized treatment plans based on diagnosis and patient factors")
    
    # This would be a simplified version for demo purposes
    diagnosis = st.text_input("Primary Diagnosis", value="pneumonia")
    
    # Quick patient data input for treatment
    col1, col2 = st.columns(2)
    with col1:
        patient_id = st.text_input("Patient ID", value="PAT001")
        age = st.number_input("Age", min_value=0, max_value=120, value=35)
        gender = st.selectbox("Gender", ["male", "female", "other", "unknown"])
    
    with col2:
        allergies = st.text_area("Allergies (one per line)", height=100)
        medical_history = st.text_area("Medical History (one per line)", height=100)
    
    if st.button("📋 Generate Treatment Plan", type="primary"):
        # Construct minimal patient data for treatment
        patient_data = {
            "patient_id": patient_id,
            "age": age,
            "gender": gender,
            "allergies": [a.strip() for a in allergies.split("\n") if a.strip()],
            "medical_history": [h.strip() for h in medical_history.split("\n") if h.strip()],
            "symptoms": [],  # Would come from previous diagnosis
            "vitals": None,
            "labs": None,
            "medications": [],
            "notes": ""
        }
        
        with st.spinner("Generating treatment plan..."):
            # Note: The API expects diagnosis as a URL parameter and patient_data in body
            # We'll need to adjust this
            result = interface.call_treatment_api(diagnosis, patient_data)
        
        if result:
            display_treatment_results(result)

def display_treatment_results(result):
    """Display treatment planning results"""
    st.header("💊 Treatment Plan")
    
    # Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Diagnosis", result["diagnosis"])
        st.metric("Patient", result["patient_id"])
    with col2:
        st.metric("Medications", len(result.get("medications", [])))
        st.metric("Treatments", len(result.get("treatments", [])))
    with col3:
        st.metric("Recovery Time", result.get("expected_recovery_time", "Variable"))
    
    # Medications
    if result.get("medications"):
        st.subheader("💊 Medications")
        for med in result["medications"]:
            with st.expander(f"{med['name']} ({med['dosage']})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Dosage:** {med['dosage']}")
                    st.write(f"**Frequency:** {med['frequency']}")
                    st.write(f"**Duration:** {med['duration']}")
                    st.write(f"**Route:** {med['route']}")
                with col2:
                    st.write(f"**Indication:** {med['indication']}")
                    if med.get('side_effects'):
                        st.write(f"**Side Effects:** {', '.join(med['side_effects'])}")
                    if med.get('monitoring_required'):
                        st.write(f"**Monitoring:** {', '.join(med['monitoring_required'])}")
    
    # Treatments
    if result.get("treatments"):
        st.subheader("🏥 Treatments")
        for treatment in result["treatments"]:
            with st.expander(f"{treatment['intervention_type'].title()}"):
                st.write(f"**Description:** {treatment['description']}")
                st.write(f"**Rationale:** {treatment['rationale']}")
                st.write(f"**Expected Outcome:** {treatment['expected_outcome']}")
                st.write(f"**Success Probability:** {treatment['success_probability']:.1%}")
    
    # Clinical Plans
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if result.get("immediate_actions"):
            st.subheader("🚨 Immediate Actions")
            for action in result["immediate_actions"]:
                st.write(f"• {action}")
    
    with col2:
        if result.get("short_term_plan"):
            st.subheader("📅 Short-term Plan")
            for item in result["short_term_plan"]:
                st.write(f"• {item}")
    
    with col3:
        if result.get("long_term_plan"):
            st.subheader("📈 Long-term Plan")
            for item in result["long_term_plan"]:
                st.write(f"• {item}")
    
    # Patient Education
    if result.get("lifestyle_modifications"):
        st.subheader("🏃 Lifestyle Modifications")
        for mod in result["lifestyle_modifications"]:
            st.write(f"• {mod}")
    
    # Warning Signs
    if result.get("warning_signs"):
        st.subheader("⚠️ Warning Signs")
        st.warning("Watch for these symptoms and seek immediate care:")
        for sign in result["warning_signs"]:
            st.write(f"• {sign}")

def render_batch_processing(interface):
    """Render batch processing interface"""
    st.header("📊 Batch Processing")
    st.info("Process multiple patients simultaneously for research or bulk analysis")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Patient Data (CSV or JSON)", 
        type=['csv', 'json'],
        help="Upload a file with multiple patient records"
    )
    
    if uploaded_file:
        # Display file preview
        st.subheader("File Preview")
        # Implementation would depend on file format
        st.info("Batch processing implementation would parse uploaded files and process multiple patients")
    
    # Sample data generator
    st.subheader("Generate Sample Data")
    num_patients = st.slider("Number of Sample Patients", 1, 10, 3)
    
    if st.button("Generate Sample Batch"):
        sample_patients = []
        for i in range(num_patients):
            sample_patients.append(get_sample_patient_data("Pneumonia Case"))
        
        st.json(sample_patients)

def render_agent_status(interface):
    """Render agent status and monitoring"""
    st.header("🤖 Agent Status")
    
    # Get agent status
    try:
        response = requests.get(f"{interface.api_base}/agents/status", timeout=10)
        if response.status_code == 200:
            status_data = response.json()
            
            # Display status for each agent
            col1, col2, col3 = st.columns(3)
            
            agents = ["diagnosis_agent", "validation_agent", "treatment_agent"]
            cols = [col1, col2, col3]
            
            for agent, col in zip(agents, cols):
                with col:
                    agent_info = status_data.get(agent, {})
                    st.subheader(agent.replace("_", " ").title())
                    
                    status = agent_info.get("status", "unknown")
                    status_color = "green" if status == "active" else "red"
                    st.markdown(f"**Status:** :{status_color}[{status}]")
                    
                    st.write(f"**Model:** {agent_info.get('model', 'N/A')}")
                    
                    capabilities = agent_info.get("capabilities", [])
                    if capabilities:
                        st.write("**Capabilities:**")
                        for cap in capabilities[:3]:  # Show first 3
                            st.write(f"• {cap}")
                        if len(capabilities) > 3:
                            st.write(f"... and {len(capabilities) - 3} more")
        else:
            st.error("Could not retrieve agent status")
    
    except Exception as e:
        st.error(f"Error retrieving agent status: {e}")
    
    # System metrics (mock data for demo)
    st.subheader("📈 System Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Patients Processed", "1,234", "+15%")
    with col2:
        st.metric("Avg Response Time", "245ms", "-12ms")
    with col3:
        st.metric("Accuracy Rate", "94.2%", "+2.1%")
    with col4:
        st.metric("Uptime", "99.9%", "0%")

def get_sample_patient_data(template: str) -> Dict:
    """Get sample patient data based on template"""
    templates = {
        "Pneumonia Case": {
            "patient_id": "PAT001",
            "age": 45,
            "gender": "female",
            "chief_complaint": "Cough and fever for 3 days",
            "symptoms": ["fever", "cough", "chest pain", "shortness of breath", "fatigue"],
            "symptom_duration": "3 days",
            "vitals": {
                "heart_rate": 100,
                "systolic_bp": 130,
                "diastolic_bp": 85,
                "temperature": 101.5,
                "respiratory_rate": 22,
                "oxygen_saturation": 94.0
            },
            "labs": {
                "wbc": 12.3,
                "crp": 18.0,
                "hemoglobin": 13.2
            },
            "medical_history": ["hypertension"],
            "medications": ["lisinopril"],
            "allergies": [],
            "notes": "Productive cough with yellow sputum"
        },
        "Heart Attack Case": {
            "patient_id": "PAT002",
            "age": 62,
            "gender": "male",
            "chief_complaint": "Severe chest pain",
            "symptoms": ["chest pain", "shortness of breath", "nausea", "sweating", "arm pain"],
            "symptom_duration": "2 hours",
            "vitals": {
                "heart_rate": 95,
                "systolic_bp": 140,
                "diastolic_bp": 90,
                "temperature": 98.6
            },
            "labs": {},
            "medical_history": ["diabetes", "hypertension", "smoking"],
            "medications": ["metformin", "lisinopril"],
            "allergies": [],
            "notes": "Crushing chest pain, radiating to left arm"
        },
        "Diabetes Case": {
            "patient_id": "PAT003",
            "age": 38,
            "gender": "male",
            "chief_complaint": "Excessive urination and thirst",
            "symptoms": ["excessive urination", "excessive thirst", "fatigue", "blurred vision", "weight loss"],
            "symptom_duration": "2 weeks",
            "vitals": {
                "heart_rate": 88,
                "systolic_bp": 125,
                "diastolic_bp": 80
            },
            "labs": {
                "glucose": 285.0,
                "hemoglobin": 14.1
            },
            "medical_history": [],
            "medications": [],
            "allergies": [],
            "notes": "Family history of diabetes"
        },
        "Hypertension Case": {
            "patient_id": "PAT004",
            "age": 55,
            "gender": "female",
            "chief_complaint": "Headache and dizziness",
            "symptoms": ["headache", "dizziness", "fatigue"],
            "symptom_duration": "1 week",
            "vitals": {
                "heart_rate": 78,
                "systolic_bp": 165,
                "diastolic_bp": 95
            },
            "labs": {},
            "medical_history": ["obesity"],
            "medications": [],
            "allergies": [],
            "notes": "No previous diagnosis of hypertension"
        },
        "Flu Case": {
            "patient_id": "PAT005",
            "age": 28,
            "gender": "female",
            "chief_complaint": "Flu-like symptoms",
            "symptoms": ["fever", "body aches", "fatigue", "cough", "headache", "chills"],
            "symptom_duration": "2 days",
            "vitals": {
                "heart_rate": 92,
                "temperature": 102.1
            },
            "labs": {},
            "medical_history": [],
            "medications": [],
            "allergies": [],
            "notes": "Seasonal flu outbreak in community"
        }
    }
    
    return templates.get(template, {})

if __name__ == "__main__":
    main()