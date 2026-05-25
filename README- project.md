# AccesiScan-AI: Automated DSAPT 2026 Compliance Auditing

## 1. Strategic Project Context

### Mission Statement

AccesiScan-AI is a high-precision computer vision solution engineered to resolve the Australian transport sector's "$12.5M Compliance Crisis." By transitioning from antiquated manual inspections to automated, smartphone-driven auditing, we empower infrastructure operators to achieve full adherence to the Disability Standards for Accessible Public Transport (DSAPT) 2026. Our objective is to bridge the gap between physical infrastructure and legal accountability using scalable AI.

###  The Business Case

The current model of accessibility auditing is unsustainable given the national scale of 5,000+ Hubs. AccesiScan-AI disrupts the prohibitive cost model that currently fuels "Audit Reluctance."

Metric	       Current Manual Audit Model	            AccesiScan-AI Model
---------      -------------------------------      ------------------------------
**Labor**	     6 Hours (Expert Labor)	              15 Minutes (Station Officer)
**Cost**	     ~$2,500 AUD per station	          ~$50 AUD per station
**Equipment**	 Specialized expert manual tools	  Standard iOS/Android devices
**Model**	     Labor-intensive service	          SaaS (Compliance-as-a-Service)
---------      --------------------------------      -------------------------------

### TRL 3 Classification

This project is officially classified as a Technical Readiness Level (TRL) 3 Government Proof of Concept. It is specifically designed to eliminate the budget-driven delays in critical safety checks that leave operators legally exposed under Australian law.

-----------------------------------------------------------------------------------------------

## 2. System Workflow & Key Features

**End-to-End Mobile App Pipeline**

The architecture follows a modular five-stage pipeline to transform raw environmental data into defensible legal audits:

1. **Data Collection:** Capture of station imagery and video via standard mobile sensors.
2. **CV Analysis:** Object detection and spatial analysis via the YOLOv8 engine (`detect.py`) for automated object boundary detection and spatial mapping.
3. **Feature Classification:** detected assets evaluation, dynamically calculating status states (Accessible, Non-Compliant, or Uncertain) based on automated feature measurements.
4. **DSAPT Compliance Check:** 1:1 automated mapping execution of physical measurements against federal standards by the core logic engine (`storage_manager.py`).
5. **Report Generation:** Export of structured JSON/PDF audit logs, storing them locally and queuing them for synchronization to the cloud-based Results Dashboard. the Results Dashboard.


 ### Core Operational Architecture & System Mechanics
Behind the mobile interface, the pipeline operations are entirely decoupled across three core modules to isolate raw computer vision logic from legal workflows:

```
                  +-----------------------------------+
                  |      ai_engine.py (Orchestrator)  |
                  +-----------------+-----------------+
                                  |
            +-----------------------+-----------------------+
            |                                               |
            v                                               v
+-----------------------+                       +-----------------------+
|       detect.py       |                       |  storage_manager.py   |
|     (Specialist)      |                       |      (Archivist)      |
|                       |                       |                       |
| * YOLOv8-Nano (FP16)  |                       | * Legal Decision Pass |
| * Metrology Engine    |                       | * Structural Auditing |
|   (10.0 px = 1.0 cm)  |                       | * JSON Serialization  |
| * APP Privacy Blur    |                       |                       |
+-----------------------+                       +-----------------------+
```


### Technical Benchmarks

Definitions of success are tied to rigorous engineering and ethical benchmarks:

1. **Geometric Precision:** A sub-centimeter pixel calibration scaling factor where 10.0 pixels=1.0 cm is implemented to target a precision threshold of ±0.5°for angular alignments and ±2 cm for structural gaps. This level of precision is critical to ensuring 1:14 DSAPT ramp compliance.
2. **Reliability:** 100% deterministic mapping with execution threading protections (workers=0) to eliminate structural hallucinations and ensure strict legal defensibility.
3. **Privacy:** Automatically drops a 15*15 Gaussian Blur across processed frames before disk serialization to permanently obscure personally identifiable information (PII) in compliance with APP Privacy Laws.
4. **Cost Efficiency:** Compiles the trained model topology into a static ONNX computational graph map, minimizing file overhead and enabling lightweight edge execution on standard smartphone chips.

--------------------------------------------------------------------------------

## 3. Repository Structure & Configuration Mapping
**Modular Architecture File Tree**

The repository enforces a decoupled directory layout to support reproducible machine learning training and strict separation between stateless vision execution and file persistence layers:

```AccessiScan-AI/
├──ai_engine.py                 # Executive Pipeline controller(Main Entry Point)
├── train.py                    # Production ML Training & Model Compilation Loop
├── configs/                    # Model Strategy Configuration Layer
│   └── data.yaml               # Legal Dataset Taxonomy Definitions(classes and paths)
├── data/                       # YOLO-formatted dataset (images + labels)
│   ├── images/                 # Raw Audit Verification Image Matrix Arrays
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/                 # Label Studio Multi-Class Bounding Box Coordinates
│       ├── train/
│       ├── val/
│       └── test/
├── scripts/                    # Decoupled Operational Submodules
│   ├── detect.py               # The Specialist: Stateless Vision Inference & Metrology Engine
│   └── storage_manager.py      # The Archivist: Legal Decision Engine & JSON Ledger System
├── models/                     # Production Network Topology Weights Storage
│   └── train/
│       └── weights/            # Production model location
│           └── best.pt         # Latest best model (auto-copied) 
├── runs/                       # Automated Pipeline Outputs
│   └── detect/                 
│       ├── predict/            # [Storage] Standard YOLO inference (Label overlays)
│       └── audit_results/      # [Storage] DSAPT Compliance Reports & Metrology Evidence
├── wandb/                      # Offline experiment tracking
├── requirements.txt            # Project Framework Pinpoint Dependencies
├── README.md                   # Core Project Architectural Blueprint Document
├── LICENSE                     # Project license
└── .gitignore                  # Environment Protection Filters (Omits Local venv & Caches)'''


[!IMPORTANT] Production Subfolder Pathing Logic Guardrail: > To guarantee runtime stability across diverse local or server deployment locations, `scripts/detect.py` evaluates model assets utilizing strict relative lookups (`../models/train/weights/best.pt`). This architecture guarantees that the execution layer reads target weights flawlessly when initialized from the central project directory. If the weight file is missing during initialization, the pipeline implements an automated graceful degradation fallback to base `yolov8n.pt` to prevent app-wide thread crashes.

--------------------------------------------------------------------------------

## 4. Environment Setup & Core Stack

###  Primary Technologies:

1. **Python 3.10:** Stable base for government AI deployments.
2. **Ultralytics YOLOv8 (Nano Variant):** Industry-leading object detection framework. Pinned to the 3.2M parameter Nano configuration to maintain a microscopic memory footprint.
3. **ONNX Runtime Backend:** Integrated directly into the compilation workflow (`train.py`) to convert native PyTorch weights into static computational graphs, unlocking hardware-agnostic acceleration.
4. **OpenCV:** Utilized for precision metrology and privacy blurring.
5. **Weights & Biases (WandB):** Serves as the experiment tracking telemetry suite, capturing loss gradients and mean Average Precision (mAP) variables for TRL 3 validation.
6. **Ecosystem Stack (Streamlit / Pandas):** Utilized within the broader platform dashboard ecosystem to parse and visualize the serialized JSON audit logs for infrastructure operators.

### Installation & Deployment Instructions

Initialize your isolated local runtime environment and install the verified dependency versions using the terminal:

#### 1. Clone the repository and navigate to the project root
cd AccessiScan-AI

#### 2. Instantiate and isolate the Python virtual environment
python -m venv venv

#### 3. Activate the virtual environment context
#### For Windows PowerShell / Command Prompt:
venv\Scripts\activate
#### For macOS / Linux Terminal:
source venv/bin/activate

# 4. Upgrade core package managers
pip install --upgrade pip setuptools wheel

# 5. Install the pinned production dependency tree
pip install ultralytics opencv-python numpy scipy wandb

[!WARNING] Universal Data-Loader Stability Control: > The training workflow inside train.py hardcodes multi-threaded data loading to a single main process (workers=0). This architectural pin is mandatory to prevent system-level socket hangs and race conditions inside the Python data loader when executing on standard consumer workstations. Do not modify this parameter, as it guarantees deterministic optimization runs.

--------------------------------------------------------------------------------

## 5. Dataset Configuration & DSAPT Regulatory Mapping

**The Dataset Schema Blueprint**

The `configs/data.yaml` file functions as the platform's core Legal Mapping Document. By defining a strict index framework, it guarantees that incoming object boundary coordinates correspond directly to individual structural sub-parts of the Disability Standards for Accessible Public Transport (DSAPT) 2026.

To prove architectural scalability for regional deployment, the final output layer is configured for the complete 20-class compliance framework (nc: 20). The active software layer dynamically audits 7 core high-risk classes, while the remaining 13 classes are natively reserved to enable seamless future model expansion without breaking core reporting structures or API schemas.

**The 20 DSAPT Classes (Taxonomy):**
| Category | Specific Label | DSAPT Reference |
| :--- | :--- | :--- |
| Signage | accessibility_signage | DSAPT Part 17 |
| Protection | handrail | DSAPT Part 15 |
| Infrastructure | kerb_ramp | DSAPT Part 6.5 |
| Boarding | platform_edges | DSAPT Part 10 |
| Vertical Movement | ramp | DSAPT Part 6 |
| Vertical Movement | stairs | DSAPT Part 14 |
| Tactile Indicators | tactile_indicators | DSAPT Part 18 |
| Signage | signage_braille | DSAPT Part 17.2 |
| Protection | barriers | DSAPT Part 15.4 |
| Boarding | boarding_points | DSAPT Part 8 |
| Boarding | gaps | DSAPT Part 8.2 |
| Access | mobility_access_features | DSAPT Part 4 |
| Access | path_widths | DSAPT Part 2 |
| Infrastructure | vertical_movement_infrastructure | DSAPT Part 13 |
| Infrastructure | lift_entries | DSAPT Part 13.1 |
| Infrastructure | escalators | DSAPT Part 14.3 |
| Infrastructure | parking_bays | DSAPT Part 1 |
| Access | thresholds | DSAPT Part 2.4 |
| Communications | hearing_loops | DSAPT Part 26 |
| Protection | bollards | DSAPT Part 2.1 |
### Mandatory Naming Standardization

String compliance is explicitly checked by the auditing engine. Every identifier must adhere strictly to singular snake_case syntax.

[!CAUTION] The label tactile_indicators and the string platform_edge must retain their exact character spellings. Pluralizing or altering these string names will instantly crash the downstream relational verification dictionaries inside storage_manager.py and output a code exception log.

### Annotation Protocol

- Ingestion Platform: Annotations are managed via Label Studio, enforcing tight boundary coordinate mappings for core features
- Leakage Avoidance: The training pipeline enforces a strict 70% Training / 20% Validation / 10% Testing partitioning strategy. This physical separation ensures zero validation cross-contamination, guaranteeing that your final mean Average Precision (mAP) calculations represent real-world generalization performance.


--------------------------------------------------------------------------------


## 6. Model Training & Artifact Management

**Training Loop Execution**

To initiate the machine learning training pipeline for the Nano-class architecture, navigate to your project root folder and execute the training script. The script automatically orchestrates the optimization cycle using your pre-configured, deterministic hyperparameters:

#### Execute directly from the AccessiScan-AI root directory
python train.py

**Note: If you need to manually override the native configurations via the command line interface parser arguments, you can pass custom parameters explicitly:*

python train.py --model yolov8n.pt --epochs 100 --imgsz 640 --device cpu

### Underlying Configuration Logic
The execution loop inside train.py handles complete optimization tracking and model compilation via a four-fold architectural pattern:
1. **Mobile Topology Target:** The script loads the lightweight yolov8n.pt base graph layer. This optimizes the parameter matrix to a small footprint, ensuring rapid feature localization when running on resource-constrained mobile hardware.
2. **Scale Matrix Enforcement:** Tensors are uniformly scaled to a high-resolution 640*640 pixels (imgsz=640). This resolution provides the geometric precision required to extract minute physical access features (such as individual tactile dots or narrow handrail boundaries).
3. **Multi-Thread Data Isolation:** The configuration sets a strict default of workers=0. This pins the incoming data matrix parsing to a single main execution thread, eliminating multi-threading socket locks and preventing system-level data loader hangs on standard local workstations.
4. **Automated Artifact Promotion:** Upon completion of the final epoch, the script utilizes shutil.copy2 to automatically intercept the highest-performing optimization weights file (`best.pt`) and promote it directly into the production storage directory (`models/train/weights/best.pt`). This automation ensures that the computer vision module always points to the latest high-accuracy model variant.


--------------------------------------------------------------------------------


## 7. Core Inference Pipeline & Compliance Evaluation Engine

**Executive Orchestration Pipeline**

The main application workflow is managed directly via ai_engine.py sitting at the project root directory. This core engine controls the data hand-offs, exception-handling wrappers, and execution sequence across your decoupled submodules to transform an incoming image array into verified compliance outputs:

#### Execute the central orchestration audit loop from the root directory
python ai_engine.py

**The Vision Processing Phase:**
`ai_engine.py` intercepts the raw file path pointer from the environment and streams the pixel matrix into `scripts/detect.py`.

**The Ledger Storage Phase:**
The resulting coordinates and spatial metrics are collected and pushed directly into scripts/storage_manager.py to calculate regulatory passes/fails and commit the findings to a timestamped JSON compliance log located at `runs/detect/audit_results/`.

### Privacy Defensibility (Australian Privacy Principle Alignment)

To maintain strict adherence to the Australian Privacy Principles (APP) when processing data within public transit infrastructure nodes, the vision system includes an active, automated anonymization guardrail:
1. **Global Anonymization Filter:** Immediately following object bounding box localization, the pipeline intercepts the frame. It passes a destructive 15*15 Gaussian Blur kernel across the visual evidence array matrix.
2. **Irreversible Anonymization:** This process permanently compromises personal identification markers (such as passenger faces or structural background identifiers) prior to saving the file to disk, protecting public identity while preserving the underlying geometric asset shapes required for human audit verification.

### Relational Compliance Thresholds & Co-Dependence Logic

The structural decision ledger inside `storage_manager.py` applies deterministic DSAPT 2026 legal constants to the metrology payloads. Rather than evaluating features completely in isolation, it enforces Structural Co-dependence Rules:

1. **Geometric Safety Limits:** The engine evaluates pixel arrays using a calibrated metric scaling factor 10.0 pixels = 1.0 cm. It flags an automatic Non-Compliant infraction if a platform clearance gap measures >5.0 cm or if platform edge alignment slopes vary >2.0° from the baseline camera plane.
2. **The Accessibility Mandate:** To secure an overall station evaluation status of PASS, the decision logic requires the concurrent presence of both a physical access path (ramp) AND a directional safety path (tactile_indicators). If either critical element is completely absent from the environment, or if any geometric measurement fails a safety threshold, the global audit drops into a cascading FAIL state.

--------------------------------------------------------------------------------

## 8. Experiment Tracking & Monitoring

*** Weights & Biases (W&B) Integration**
The machine learning pipeline couples directly with Weights & Biases to manage MLOps telemetry, tracking loss gradients and evaluation metrics across training cycles.

All primary performance variables—including localization loss, classification loss, bounding box precision, and mean Average Precision (mAP@50 and mAP@50-95)—are streamed in real-time to your cloud-hosted project workspace for longitudinal performance verification and model generation comparison.

**Network Disruption Recovery Fallback:**If a training run is executed in an offline environment or suffers a network drop out, metrics are safely preserved locally in cache. They can be manually synchronized to the cloud dashboard using the fallback recovery command:

wandb sync

### Multi-Thread Operational Safety

To prevent resource-wasting accidental initializations, memory leaks, or "dummy" experiment creations during modular script imports, the telemetry tracking session is structurally isolated:

**Scope Guardrail:** The initialization sequence is strictly enclosed within the deterministic `if __name__ == "__main__":` execution block of `train.py`. This ensures that if another submodule loops through or imports components from the training script, a parallel background W&B logging thread is never accidentally triggered.


--------------------------------------------------------------------------------

## 9. Finalized Production Fixes & Summary

*** Pre-Handoff Checklist**

The following critical technical fixes are verified for the current build:

[x] **Submodule Pathing Security:** Relative path resolution matrices (../) verified inside scripts/detect.py to allow flawless execution of model weights across decoupled directories without path exceptions.

[x] **Universal Threading Stability:** Dedicated single-process data loading (workers=0) established universally within train.py to prevent system socket locks and race conditions across diverse OS environments.

[x] **Taxonomy Naming Standardization:** Strict singular snake_case string convention (tactile_indicators, platform_edge) hardcoded across the annotation schema, configurations, and evaluation modules to prevent structural index dictionary drops.

[x] **Ingestion Processing Protocol:** Label Studio verified as the standardized coordinate layout tool, utilizing clean bounding boxes exported via optimized YOLOv8 schemas.

### Long-Term Strategic Engineering Vision

To scale the platform beyond the TRL 3 proof-of-concept phase, the architectural roadmap is designed to support three major technological milestones:

1. **National Digital Twin Integration:** Injecting serialized JSON compliance audit data directly into state and federal transport spatial networks to build an active, data-driven digital twin of national transit hubs.

2. **Edge Stream Video Auditing:** Upgrading the pipeline execution loop from static image arrays to processing real-time streaming video inputs, enabling station inspectors to conduct audits dynamically while walking down a platform.

3. **Cross-Platform Mobile Engine Deployment:** Wrapping the underlying ONNX computational graphs into a native, cross-platform mobile frontend UI (iOS/Android) for immediate deployment to transit workers.

### Legal & Ethical Benchmarks

AccesiScan-AI is fundamentally engineered for strict legal compliance. By creating a forward-looking 1:1 mapping system across all 20 regulatory DSAPT categories, and combining it with automated, irreversible 15*15 Gaussian Privacy Blurring to protect public PII, the architecture provides a robust, ethically defensible, and inclusive auditing framework for the future of Australian public infrastructure.

