import os
import cv2
from scripts.detect import AccesiScanDetector
from scripts.storage_manager import AuditStorage

from scripts.system_utils import setup_environment
# Initialize environment before loading heavy ML dependencies
setup_environment()

class AccesiScanEngine:
    """
    EXECUTIVE PIPELINE CONTROLLER
    Purpose: Orchestrates the workflow between Computer Vision and Data Persistence.
    Stakeholder Note: The single entry point for the Mobile UI and Backend integration.
    """

    def __init__(self):
        # Initialize sub-modules (Specialist and Archivist)
        self.detector = AccesiScanDetector()
        self.storage = AuditStorage()

    def run_audit(self, image_path, station_name="General_Station"):
        """
        END-TO-END WORKFLOW: Raw Image -> AI Analysis -> Precision Metrology -> Archived Report.
        """
        if not os.path.exists(image_path):
            return {"status": "Error", "message": f"Input file not found at {image_path}"}

        try:
            # Step 1: Trigger Vision & Metrology (The 'Specialist')
            raw_findings, annotated_img = self.detector.run_detection(image_path)
            
            # Step 2: Save Visual Evidence (For the 'Client')
            img_name = f"{station_name}_evidence.jpg"
            img_path = os.path.join(self.storage.storage_dir, img_name)
            cv2.imwrite(img_path, annotated_img)

            # Step 3: Archive Technical Data (The 'Archivist')
            report_path = self.storage.save_audit_result(station_name, raw_findings)
            
            return {
                "status": "Success",
                "station": station_name,
                "visual_evidence": img_path,
                "data_report": report_path,
                "message": f"Audit complete. Findings archived to {report_path}"
            }

        except Exception as e:
            # Reliability: Captures any technical failures to prevent system crashes
            return {"status": "Error", "message": f"Pipeline Interruption: {str(e)}"}

if __name__ == "__main__":
    # --- BATCH AUDIT SIMULATION (Demo) ---
    # This section allows to see the system processing images at scale.
    from tqdm import tqdm
    
    engine = AccesiScanEngine()
    test_suite = ["data/images/test/your_actual_file_name.jpg"] # Update with real paths
    
    print("--- Starting AccessiScan-AI DSAPT Audit ---")
    for img in tqdm(test_suite, desc="Processing Station Photos", unit="img"):
        if os.path.exists(img):
            res = engine.run_audit(img, "Reservoir_Station")
            print(f"\nAudit Result for {img}: {res['status']}")
            print(f"Technical Report: {res['data_report']}")
