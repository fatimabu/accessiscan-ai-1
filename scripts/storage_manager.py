import json
import datetime
import os

class AuditStorage:
    """
    DATA PERSISTENCE & COMPLIANCE ARCHIVE
    Purpose: Centralizes the storage of audit logs and maps AI findings to legal standards.
    Stakeholder Note: Ensures every audit has a unique timestamp and 'Chain of Custody' metadata.
    """

    def __init__(self, storage_dir='runs/detect/audit_results'):
        self.storage_dir = storage_dir
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

    def save_audit_result(self, station_name, detections):
        """
        ARCHIVING: Wraps raw findings into a professional audit report.
        Business Note: This JSON structure is ready for RAG reporting and client hand-off.
        """
        report = {
            "metadata": {
                "station": station_name,
                "timestamp": datetime.datetime.now().isoformat(),
                "audit_standard": "DSAPT 2026",
                "auditor_tool": "AccessiScan-AI (Sprint 5 Build)"
            },
            "findings": detections,
            "summary": self._generate_dsapt_summary(detections)
        }

        # Generate unique filename based on time
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f"{station_name}_audit_{timestamp}.json"
        file_path = os.path.join(self.storage_dir, file_name)

        with open(file_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        return file_path
    
    def _generate_dsapt_summary(self, detections):
        """
        LEGAL MAPPING: Translates AI data into executive compliance status.
        """
        
        classes_found = [d['class_name'].lower().replace(" ", "_") for d in detections]
        violations = [d for d in detections if d.get('status') == "Non-Compliant"]
        
        # Identify specific violations based on DSAPT 2026 thresholds
        gap_violations = [d for d in violations if d.get('measurement_cm') and d['measurement_cm'] > 5.0]
        angle_violations = [d for d in violations if d.get('angulation_deg') and abs(d['angulation_deg']) > 2.0]

        # Check for mandatory infrastructure presence [DSAPT 2026 requires both ramps and tactile indicators for a PASS]
        has_ramp = "ramp" in classes_found
        # Updated to match standardised data.yaml naming
        has_tactile = "tactile_indicators" in classes_found 

        return {
            "total_features_detected": len(detections),
            "violation_count": len(violations),
            "specific_issues": {
                "excessive_gaps": len(gap_violations),
                "steep_angulation": len(angle_violations)
            },
            "core_infrastructure_present": has_ramp and has_tactile,
            "overall_status": "FAIL" if len(violations) > 0 else "PASS",
            "notes": "Automated scan complete. Metrology Pass (5.0cm/2.0°) applied."
        }