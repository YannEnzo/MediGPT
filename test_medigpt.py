# test_medigpt.py - Test Script for MediGPT Agents System
import requests
import json
import time
from typing import Dict, List
import sys

# Configuration
API_BASE_URL = "http://localhost:8000"

class MediGPTTester:
    """Test suite for MediGPT Agents system"""
    
    def __init__(self, api_base: str = API_BASE_URL):
        self.api_base = api_base
        self.test_results = []
    
    def run_all_tests(self):
        """Run all tests and report results"""
        print("🧪 MediGPT Agents Test Suite")
        print("=" * 50)
        
        # Test API connection
        self.test_api_connection()
        
        # Test each endpoint
        self.test_diagnosis_endpoint()
        self.test_validation_endpoint()
        self.test_treatment_endpoint()
        self.test_agent_status()
        self.test_batch_processing()
        
        # Report results
        self.report_results()
    
    def test_api_connection(self):
        """Test basic API connectivity"""
        print("\n1. Testing API Connection...")
        
        try:
            response = requests.get(f"{self.api_base}/", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ API is running - Version: {data.get('version', 'Unknown')}")
                self.test_results.append(("API Connection", True, "Connected successfully"))
            else:
                print(f"   ❌ API returned status {response.status_code}")
                self.test_results.append(("API Connection", False, f"Status {response.status_code}"))
        except Exception as e:
            print(f"   ❌ Cannot connect to API: {e}")
            self.test_results.append(("API Connection", False, str(e)))
    
    def test_diagnosis_endpoint(self):
        """Test the diagnosis endpoint with sample data"""
        print("\n2. Testing Diagnosis Endpoint...")
        
        # Sample patient data
        test_patients = [
            {
                "name": "Pneumonia Case",
                "data": {
                    "patient_id": "TEST001",
                    "age": 45,
                    "gender": "female",
                    "symptoms": ["fever", "cough", "chest pain", "shortness of breath"],
                    "vitals": {
                        "heart_rate": 100,
                        "systolic_bp": 130,
                        "diastolic_bp": 85,
                        "temperature": 101.5,
                        "respiratory_rate": 22
                    },
                    "labs": {
                        "wbc": 12.3,
                        "crp": 18.0
                    },
                    "medical_history": ["hypertension"],
                    "medications": ["lisinopril"],
                    "allergies": []
                }
            },
            {
                "name": "Diabetes Case",
                "data": {
                    "patient_id": "TEST002",
                    "age": 38,
                    "gender": "male",
                    "symptoms": ["excessive urination", "excessive thirst", "fatigue", "blurred vision"],
                    "labs": {
                        "glucose": 285.0
                    },
                    "medical_history": [],
                    "medications": [],
                    "allergies": []
                }
            }
        ]
        
        for patient in test_patients:
            try:
                print(f"   Testing {patient['name']}...")
                response = requests.post(
                    f"{self.api_base}/diagnose",
                    json=patient["data"],
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    diagnoses_count = len(result.get("diagnoses", []))
                    top_diagnosis = result["diagnoses"][0]["condition"] if diagnoses_count > 0 else "None"
                    confidence = result["diagnoses"][0]["confidence"] if diagnoses_count > 0 else 0
                    
                    print(f"   ✅ {patient['name']}: {diagnoses_count} diagnoses, top: {top_diagnosis} ({confidence:.1%})")
                    self.test_results.append((f"Diagnosis - {patient['name']}", True, f"{diagnoses_count} diagnoses generated"))
                else:
                    print(f"   ❌ {patient['name']}: Status {response.status_code}")
                    self.test_results.append((f"Diagnosis - {patient['name']}", False, f"Status {response.status_code}"))
                    
            except Exception as e:
                print(f"   ❌ {patient['name']}: {e}")
                self.test_results.append((f"Diagnosis - {patient['name']}", False, str(e)))
    
    def test_validation_endpoint(self):
        """Test the validation endpoint"""
        print("\n3. Testing Validation Endpoint...")
        
        test_diagnoses = [
            {"diagnosis": "pneumonia", "age": 45, "gender": "female"},
            {"diagnosis": "diabetes mellitus", "age": 38, "gender": "male"},
            {"diagnosis": "hypertension", "age": 55, "gender": "female"}
        ]
        
        for test in test_diagnoses:
            try:
                print(f"   Testing validation for {test['diagnosis']}...")
                
                params = {
                    "patient_age": test["age"],
                    "patient_gender": test["gender"]
                }
                
                response = requests.get(
                    f"{self.api_base}/validate/{test['diagnosis']}",
                    params=params,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    is_supported = result.get("is_supported", False)
                    sources_count = result.get("total_sources_reviewed", 0)
                    
                    print(f"   ✅ {test['diagnosis']}: Supported={is_supported}, {sources_count} sources reviewed")
                    self.test_results.append((f"Validation - {test['diagnosis']}", True, f"Supported: {is_supported}"))
                else:
                    print(f"   ❌ {test['diagnosis']}: Status {response.status_code}")
                    self.test_results.append((f"Validation - {test['diagnosis']}", False, f"Status {response.status_code}"))
                    
            except Exception as e:
                print(f"   ❌ {test['diagnosis']}: {e}")
                self.test_results.append((f"Validation - {test['diagnosis']}", False, str(e)))
    
    def test_treatment_endpoint(self):
        """Test the treatment endpoint"""
        print("\n4. Testing Treatment Endpoint...")
        
        test_cases = [
            {
                "diagnosis": "pneumonia",
                "patient_data": {
                    "patient_id": "TEST003",
                    "age": 45,
                    "gender": "female",
                    "symptoms": ["fever", "cough"],
                    "medical_history": ["hypertension"],
                    "allergies": [],
                    "medications": []
                }
            },
            {
                "diagnosis": "diabetes mellitus",
                "patient_data": {
                    "patient_id": "TEST004",
                    "age": 38,
                    "gender": "male",
                    "symptoms": ["excessive urination", "excessive thirst"],
                    "medical_history": [],
                    "allergies": [],
                    "medications": []
                }
            }
        ]
        
        for test in test_cases:
            try:
                print(f"   Testing treatment for {test['diagnosis']}...")
                
                # Note: Adjusting payload structure based on actual API
                payload = {
                    "diagnosis": test["diagnosis"],
                    "patient_data": test["patient_data"]
                }
                
                response = requests.post(
                    f"{self.api_base}/simulate",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    medications_count = len(result.get("medications", []))
                    treatments_count = len(result.get("treatments", []))
                    
                    print(f"   ✅ {test['diagnosis']}: {medications_count} medications, {treatments_count} treatments")
                    self.test_results.append((f"Treatment - {test['diagnosis']}", True, f"{medications_count} meds, {treatments_count} treatments"))
                else:
                    print(f"   ❌ {test['diagnosis']}: Status {response.status_code}")
                    self.test_results.append((f"Treatment - {test['diagnosis']}", False, f"Status {response.status_code}"))
                    
            except Exception as e:
                print(f"   ❌ {test['diagnosis']}: {e}")
                self.test_results.append((f"Treatment - {test['diagnosis']}", False, str(e)))
    
    def test_agent_status(self):
        """Test agent status endpoint"""
        print("\n5. Testing Agent Status...")
        
        try:
            response = requests.get(f"{self.api_base}/agents/status", timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                agents = ["diagnosis_agent", "validation_agent", "treatment_agent"]
                
                for agent in agents:
                    if agent in result:
                        status = result[agent].get("status", "unknown")
                        print(f"   ✅ {agent}: {status}")
                        self.test_results.append((f"Agent Status - {agent}", True, status))
                    else:
                        print(f"   ❌ {agent}: Not found in response")
                        self.test_results.append((f"Agent Status - {agent}", False, "Not found"))
            else:
                print(f"   ❌ Agent status: Status {response.status_code}")
                self.test_results.append(("Agent Status", False, f"Status {response.status_code}"))
                
        except Exception as e:
            print(f"   ❌ Agent status: {e}")
            self.test_results.append(("Agent Status", False, str(e)))
    
    def test_batch_processing(self):
        """Test batch processing endpoint"""
        print("\n6. Testing Batch Processing...")
        
        # Sample batch data
        batch_patients = [
            {
                "patient_id": "BATCH001",
                "age": 30,
                "gender": "male",
                "symptoms": ["fever", "cough"],
                "medical_history": [],
                "medications": [],
                "allergies": []
            },
            {
                "patient_id": "BATCH002",
                "age": 50,
                "gender": "female",
                "symptoms": ["headache", "dizziness"],
                "medical_history": [],
                "medications": [],
                "allergies": []
            }
        ]
        
        try:
            response = requests.post(
                f"{self.api_base}/batch_diagnose",
                json=batch_patients,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                processed_count = result.get("total_processed", 0)
                print(f"   ✅ Batch processing: {processed_count} patients processed")
                self.test_results.append(("Batch Processing", True, f"{processed_count} patients"))
            else:
                print(f"   ❌ Batch processing: Status {response.status_code}")
                self.test_results.append(("Batch Processing", False, f"Status {response.status_code}"))
                
        except Exception as e:
            print(f"   ❌ Batch processing: {e}")
            self.test_results.append(("Batch Processing", False, str(e)))
    
    def report_results(self):
        """Report final test results"""
        print("\n" + "=" * 50)
        print("📊 Test Results Summary")
        print("=" * 50)
        
        passed = sum(1 for _, success, _ in self.test_results if success)
        total = len(self.test_results)
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success Rate: {passed/total*100:.1f}%" if total > 0 else "No tests run")
        
        print("\nDetailed Results:")
        for test_name, success, message in self.test_results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"  {status} {test_name}: {message}")
        
        if passed == total:
            print("\n🎉 All tests passed! MediGPT system is working correctly.")
        else:
            print(f"\n⚠️  {total - passed} test(s) failed. Please check the system configuration.")

def run_performance_test():
    """Run performance tests"""
    print("\n🚀 Performance Testing...")
    
    # Sample patient for performance testing
    patient_data = {
        "patient_id": "PERF001",
        "age": 45,
        "gender": "female",
        "symptoms": ["fever", "cough", "chest pain"],
        "vitals": {"heart_rate": 100, "temperature": 101.5},
        "labs": {"wbc": 12.3},
        "medical_history": [],
        "medications": [],
        "allergies": []
    }
    
    # Test multiple requests
    times = []
    for i in range(5):
        start_time = time.time()
        try:
            response = requests.post(
                f"{API_BASE_URL}/diagnose",
                json=patient_data,
                timeout=30
            )
            end_time = time.time()
            
            if response.status_code == 200:
                times.append(end_time - start_time)
                print(f"   Request {i+1}: {(end_time - start_time)*1000:.1f}ms")
        except Exception as e:
            print(f"   Request {i+1}: Failed - {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\nPerformance Summary:")
        print(f"   Average: {avg_time*1000:.1f}ms")
        print(f"   Min: {min_time*1000:.1f}ms")
        print(f"   Max: {max_time*1000:.1f}ms")

def main():
    """Main test function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--performance":
        run_performance_test()
    else:
        tester = MediGPTTester()
        tester.run_all_tests()

if __name__ == "__main__":
    main()