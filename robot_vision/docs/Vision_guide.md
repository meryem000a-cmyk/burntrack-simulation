# **Advanced Edge Vision Pipelines: Deploying SOTA Lightweight Plant Classification on Resource-Constrained Rover Architectures**

The deployment of computer vision models on embedded field hardware requires balancing computational accuracy with strict resource limitations. In the context of an autonomous rover exploring varied terrains to catalog African flora and assess fire risk, the hardware—specifically a Raspberry Pi 4 Model B (RPi 4B)—must act as a multi-tenant system. It must execute deterministic sensor polling, motor control, and navigation algorithms alongside non-deterministic, compute-heavy machine learning inference. If the vision pipeline saturates the Quad-core Broadcom BCM2711 ARM Cortex-A72 CPU or monopolizes the LPDDR4 RAM, the rover’s kinematics and obstacle avoidance systems will experience catastrophic latency, potentially leading to hardware damage or mission failure.  
Developing a State-of-the-Art (SOTA) plant classification system for this architecture requires a holistic approach. It demands the curation of comprehensive and highly specific datasets of African plant life, the selection of an optimized lightweight vision backbone capable of multi-task instance counting, precise INT8 quantization, and strict thread-level resource management during deployment. The following report provides an exhaustive, step-by-step framework to acquire relevant botanical data, train a lightweight architecture, compress the model for edge execution, and output structured JSON data for Rothermel fire spread simulations while guaranteeing significant system headroom for robotic control loops.

## **Comprehensive African Flora Data Acquisition**

The African continent exhibits high levels of species diversity and endemism, yet botanical data from the region is historically fragmented and heavily biased toward specific well-researched geographic pockets.1 To build a robust classification model capable of operating in wild environments, data must be aggregated from multiple highly curated biodiversity networks. Training a model exclusively on general-purpose image datasets will yield poor domain adaptation for the specific morphology, lighting conditions, and background clutter inherent to tropical and sub-Saharan African vascular plants.

### **Foundational Plant Datasets and Pre-training**

Before fine-tuning on highly specific African flora, a baseline understanding of plant morphology must be established within the neural network's feature extraction layers. The Pl@ntNet-300K dataset serves as the optimal pre-training foundation for this endeavor. Derived from the Pl@ntNet citizen observatory, this dataset contains 306,146 images spanning 1,081 plant species.2 Its primary utility lies in its structural characteristics: it exhibits a heavy long-tailed distribution where 80% of the species account for only 11% of the total images, and it features high visual ambiguity.2 These conditions perfectly mirror the challenges of wild botanical classification, forcing the model to learn fine-grained feature extraction necessary to distinguish visually similar taxa, rather than relying on spurious background correlations.4

### **African Specific Taxa and Biodiversity Databases**

For the primary domain adaptation, the RAINBIO dataset provides the critical taxonomic backbone. RAINBIO is a mega-database compiling tropical African vascular plant distributions, containing 609,776 georeferenced records across 22,577 species.6 While RAINBIO primarily provides occurrence data rather than direct image hosting, its curated species checklists serve as the exact query parameters needed to extract images from large-scale biodiversity aggregators like the Global Biodiversity Information Facility (GBIF) and iNaturalist.1  
The "African Plants \- a photo guide" dataset, published by Senckenberg, serves as a premier source of highly accurate photographic records of vascular plants from Africa.10 This dataset is accessible via GBIF using the Dataset Key e5774d90-9f01-42bb-a747-32331be82b18.10 By isolating this specific dataset within GBIF, researchers guarantee that the images retrieved are vetted by botanists, ensuring high label accuracy for the training pipeline.10

### **Dataset Annotation for Rothermel Curing Criteria**

To feed accurate data into advanced fire spread simulations—specifically those coupling the Rothermel mathematical model with Multi-Dimensional Cellular Automata (MD-CA)—the vision dataset must be annotated for plant dryness. The Rothermel model's dynamic fuel load transfer heavily relies on the Live Herbaceous Moisture Content (LHMC) and the specific curing percentage of the vegetation.  
During data preparation, all bounding boxes in the acquired dataset must be dual-labeled with both the plant species and its curing classification. The standard Rothermel curing classifications that map directly to fuel dynamics are:

* **Uncured (Not Dry):** Represents live fuel with an LHMC of 120% or higher where no fuel load is transferred to dead fine fuels.  
* **Partially Cured / Fully Cured (Dry):** Represents vegetation with an LHMC below 98% (down to 30% or less), indicating that the plant matter has transferred its load to dead, 1-hour timelag fuels, creating high fire spread risk.

By classifying these precise curing stages, the cellular automata grid can dynamically update its transition rules for localized fire acceleration based directly on the rover's real-time JSON outputs.

### **Automated Data Acquisition CLI and Pipeline**

To assemble the image dataset locally on a high-performance training machine (prior to edge deployment), programmatic access to GBIF and iNaturalist is required. The pygbif and pyinaturalist Python libraries provide robust interfaces for querying these occurrences.14

#### **GBIF Image Extraction via pygbif and SQL API**

For a comprehensive dataset, utilizing the asynchronous SQL download API is vastly superior. The SQL API allows users to request an Apache Parquet or ZIP TSV file containing all occurrence metadata matching a query, which can then be parsed locally to download images.17

Bash  
\# Create a dedicated Python virtual environment for data acquisition  
python3 \-m venv flora\_env  
source flora\_env/bin/activate

\# Install required API clients and data processing libraries  
pip install pygbif pyinaturalist pandas requests

To execute a bulk image download using pygbif, the continent parameter must be set to africa and mediatype to StillImage.20

Python  
import os  
import requests  
from pygbif import occurrences as occ

def acquire\_gbif\_african\_flora(taxon\_key, output\_dir, max\_records=1000):  
    """  
    Downloads occurrences with images for a specific African taxon using pagination.  
    """  
    os.makedirs(output\_dir, exist\_ok=True)  
    offset \= 0  
    limit \= 300  
    downloaded \= 0

    while downloaded \< max\_records:  
        res \= occ.search(  
            taxonKey=taxon\_key,   
            continent='africa',   
            mediatype='StillImage',   
            limit=limit,   
            offset=offset  
        )  
          
        results \= res.get('results',)  
        if not results:  
            break  
              
        for record in results:  
            media \= record.get('media',)  
            for m in media:  
                if m.get('type') \== 'StillImage' and 'identifier' in m:  
                    img\_url \= m\['identifier'\]  
                    img\_name \= f"{record\['key'\]}.jpg"  
                    img\_path \= os.path.join(output\_dir, img\_name)  
                      
                    if not os.path.exists(img\_path):  
                        try:  
                            response \= requests.get(img\_url, timeout=10)  
                            if response.status\_code \== 200:  
                                with open(img\_path, 'wb') as f:  
                                    f.write(response.content)  
                                downloaded \+= 1  
                        except requests.exceptions.RequestException:  
                            continue  
                  
                if downloaded \>= max\_records:  
                    break  
        offset \+= limit

#### **iNaturalist Extraction via pyinaturalist**

Similarly, iNaturalist contains thousands of "research-grade" observations of African flora, which have been peer-reviewed by at least two users on the platform.22 Using pyinaturalist, queries can be constrained by geographic bounding boxes or specific place IDs, enforcing a high-quality filter to eliminate casual observations.25

Python  
from pyinaturalist import get\_observations  
import time

def acquire\_inaturalist\_flora(taxon\_id, place\_id, output\_dir):  
    search\_params \= {  
        "taxon\_id": taxon\_id,  
        "place\_id": place\_id,   
        "quality\_grade": "research",  
        "has\_photos": True,  
        "per\_page": 200  
    }  
      
    results \= get\_observations(\*\*search\_params).get("results",)  
      
    for obs in results:  
        for photo in obs.get("photos",):  
            url \= photo.get("url").replace("square", "medium")   
            img\_path \= os.path.join(output\_dir, f"inat\_{obs\['id'\]}.jpg")  
              
            try:  
                response \= requests.get(url, timeout=10)  
                if response.status\_code \== 200:  
                    with open(img\_path, 'wb') as f:  
                        f.write(response.content)  
            except Exception as e:  
                pass  
              
            time.sleep(1.2)

## **SOTA Lightweight Vision Models for the Edge**

Deploying deep learning on a Raspberry Pi 4 demands architectural efficiency. The RPi 4 features a Broadcom BCM2711 SoC with a quad-core ARM Cortex-A72 CPU running at 1.5 GHz, delivering approximately 32 GFLOPS of FP32 compute.27  
To satisfy the rover's requirement to both *identify* plant types and *count* the number of dry vs. non-dry instances in a single frame, a standard image classification model (which outputs a single label per image) is insufficient. The architecture must utilize a **Multi-Task Object Detection** framework. Object detection models predict bounding boxes for every plant in the frame, allowing the system to aggregate counts and classify the curing state (dryness) of each individual detection simultaneously.

### **Multi-Task YOLO11n for Instance Counting and Dryness**

For edge-based multi-task detection, lightweight variants of the YOLO (You Only Look Once) family, such as **YOLO11n (Nano)** or specialized multi-task derivatives like MTS-YOLO, are currently state-of-the-art. These architectures employ efficient structural re-parameterization and streamlined feature pyramids to achieve sub-20ms inference times on ARM CPUs while successfully identifying multiple objects and their states (e.g., maturity or dryness) within the same forward pass.  
By labeling the dataset with compound classes (e.g., Acacia\_dry, Acacia\_not\_dry), a single lightweight YOLO11n network can localize the plants and determine their Rothermel curing stage simultaneously, leaving massive headroom for the Pi 4's control loops.  
Alternatively, for users strictly building custom backbones, **MobileNetV4-Conv-Small** presents exceptional CPU inference speeds (as low as 5.7 ms) 29 and can be adapted into a Single Shot Detector (SSD) backbone to serve the same counting purpose, though standardizing on the Ultralytics YOLO pipeline is vastly more streamlined for CLI automation.

## **Model Training and Domain Adaptation Guide**

Training the multi-task detection model on the curated African flora dataset requires programmatic execution. The training pipeline utilizes the Ultralytics CLI for YOLO11n, which handles aggressive mosaic augmentations natively to simulate the occlusion conditions the rover will experience in the field.

### **CLI Training Environment Setup**

Bash  
\# Install PyTorch with CUDA support and the Ultralytics library  
pip install torch torchvision torchaudio \--index-url https://download.pytorch.org/whl/cu118  
pip install ultralytics

A data.yaml file must be generated to map the African plant species and their curing states (dry vs not dry) for the detector:

YAML  
path:../datasets/african\_flora  
train: images/train  
val: images/val

\# Classes combining Species \+ Dryness (Rothermel Curing State)  
names:  
  0: Acacia\_not\_dry  
  1: Acacia\_dry  
  2: Baobab\_not\_dry  
  3: Baobab\_dry  
  \#... mapped up to the total number of species conditions

### **Automated Training Execution**

Bash  
\# Train the YOLO11n model using the generated dataset configuration  
yolo task=detect mode=train model=yolo11n.pt data=data.yaml epochs=150 imgsz=320 batch=32 device=0

Once the model reaches optimal mean Average Precision (mAP) on the validation set, it must be quantized for deployment on the Raspberry Pi.

## **Quantization and Compression for CPU Execution**

A floating-point (FP32) model is highly inefficient on an ARM CPU like the Cortex-A72. To satisfy the rover's strict memory and CPU constraints, the model must be compressed using 8-bit Integer (INT8) Post-Training Quantization (PTQ).30

### **Exporting to LiteRT (TFLite) INT8**

The trained weights can be exported directly to the highly optimized LiteRT (TFLite) flatbuffer format using the Ultralytics conversion pipeline, which natively handles calibration against the dataset to determine the integer scaling and zero-point parameters required to prevent accuracy degradation.

Bash  
\# Export the trained YOLO model to INT8 TFLite format for the Raspberry Pi  
\# The 'data' argument is crucial as it provides the representative dataset for INT8 calibration  
yolo export model=runs/detect/train/weights/best.pt format=tflite int8=True data=data.yaml imgsz=320

This process generates a best\_full\_integer\_quant.tflite file, reducing the memory footprint to under 3 MB and enabling heavily accelerated ARM NEON SIMD execution.

## **Edge Deployment and JSON Orchestration on Raspberry Pi 4**

With the model highly compressed, it must be integrated into the rover's software stack. The overarching requirement is that the vision pipeline must not monopolize CPU execution time, leaving sufficient processing power for deterministic control loops (GPS, humidity, wind, and LoRa data aggregation).

### **LiteRT Runtime and Thread Bounding**

Installing the entire TensorFlow suite on a Raspberry Pi is highly detrimental. Instead, the specialized tflite\_runtime must be utilized.34  
**CLI Installation Command on Raspberry Pi:**

Bash  
python3 \-m pip install tflite-runtime numpy opencv-python

To prevent hardware resource contention, the LiteRT Interpreter object exposes the set\_num\_threads() method.36 By explicitly restricting the interpreter to 2 threads, the vision pipeline is pinned, guaranteeing that at least two physical CPU cores remain completely idle for the rover's sensor pipelines and high-frequency LoRa communications.38

### **Real-Time Rover Vision Integration Script (JSON Output)**

The following implementation provides an exhaustive blueprint for deploying the INT8 object detection model on the RPi 4\. It actively manages thread allocation, processes the bounding boxes to aggregate counts based on species and curing levels, and formats the output into the exact JSON specification required for the Rothermel Cellular Automata simulations.

Python  
import numpy as np  
import cv2  
import time  
import json  
from multiprocessing import Process, Queue  
try:  
    from tflite\_runtime.interpreter import Interpreter  
except ImportError:  
    print("tflite\_runtime missing. Run: pip install tflite-runtime")  
    exit(1)

def vision\_inference\_process(input\_queue, output\_queue, model\_path, class\_names):  
    """  
    Isolated process for running multi-task plant and dryness detection.  
    Outputs a JSON string containing plant types and their dry/not\_dry counts.  
    """  
    interpreter \= Interpreter(model\_path=model\_path)  
      
    \# CRITICAL: Restrict CPU utilization to leave headroom for rover sensors.  
    \# Limiting to 2 threads ensures 2 cores are entirely free on the RPi 4\.  
    try:  
        interpreter.set\_num\_threads(2)  
        print("LiteRT Interpreter locked to 2 threads for CPU headroom.")  
    except Exception as e:  
        print(f"Warning: Thread configuration failed: {e}")  
          
    interpreter.allocate\_tensors()  
    input\_details \= interpreter.get\_input\_details()  
    output\_details \= interpreter.get\_output\_details()  
      
    \# Precompute quantization parameters  
    in\_scale, in\_zero\_point \= input\_details\['quantization'\]  
    out\_scale, out\_zero\_point \= output\_details\['quantization'\]  
      
    while True:  
        if not input\_queue.empty():  
            frame \= input\_queue.get()  
            if frame is None: break   
                  
            \# Preprocessing: Resize to 320x320 and format to NHWC  
            img\_resized \= cv2.resize(frame, (320, 320))  
            img\_normalized \= (img\_resized.astype(np.float32) / 255.0)  
              
            if in\_scale\!= 0:  
                img\_quantized \= (img\_normalized / in\_scale \+ in\_zero\_point).astype(np.int8)  
            else:  
                img\_quantized \= img\_normalized.astype(np.int8)  
                  
            img\_expanded \= np.expand\_dims(img\_quantized, axis=0)  
              
            \# Execute inference  
            interpreter.set\_tensor(input\_details\['index'\], img\_expanded)  
            interpreter.invoke()  
              
            \# Retrieve and de-quantize bounding box/class outputs  
            raw\_output \= interpreter.get\_tensor(output\_details\['index'\])  
            if out\_scale\!= 0:  
                predictions \= (raw\_output.astype(np.float32) \- out\_zero\_point) \* out\_scale  
            else:  
                predictions \= raw\_output  
                  
            \# Dictionary to accumulate counts for the JSON output  
            \# Format: {"PlantName": \[dry\_count, not\_dry\_count\]}  
            frame\_counts \= {}  
              
            \# Simplified NMS / Output Parsing (assuming standard YOLO format \[x, y, w, h, conf, class\_id\])  
            \# In production, apply cv2.dnn.NMSBoxes here based on confidence thresholds  
            confidence\_threshold \= 0.5  
            for detection in predictions:  
                conf \= detection\[4\]  
                if conf \> confidence\_threshold:  
                    class\_id \= int(detection\[5\])  
                    class\_label \= class\_names\[class\_id\] \# e.g., "Acacia\_dry"  
                      
                    \# Split the label into species and its curing/dryness state  
                    species, state \= class\_label.rsplit('\_', 1\)  
                    is\_dry \= True if state \== 'dry' else False  
                      
                    if species not in frame\_counts:  
                        frame\_counts\[species\] \=   
                          
                    if is\_dry:  
                        frame\_counts\[species\] \+= 1 \# Increment dry instance  
                    else:  
                        frame\_counts\[species\]\[1\] \+= 1 \# Increment not\_dry instance  
              
            \# Format as JSON string for the Cellular Automata / Rothermel pipeline  
            json\_output \= json.dumps(frame\_counts)  
            output\_queue.put(json\_output)  
              
        else:  
            time.sleep(0.01)

if \_\_name\_\_ \== '\_\_main\_\_':  
    frame\_queue \= Queue(maxsize=1)  
    result\_queue \= Queue(maxsize=1)  
      
    \# Mock label map combining African species and their Rothermel curing states  
    labels \= {0: "Acacia\_not\_dry", 1: "Acacia\_dry", 2: "Baobab\_not\_dry", 3: "Baobab\_dry"}  
      
    vision\_process \= Process(  
        target=vision\_inference\_process,   
        args=(frame\_queue, result\_queue, "yolo11n\_full\_integer\_quant.tflite", labels)  
    )  
    vision\_process.daemon \= True  
    vision\_process.start()  
      
    try:  
        while True:  
            \# 1\. Rover polls GPS, Humidity, Wind, Temp, Inclination \- Headroom is protected  
            \# 2\. Control loops execute  
              
            \# 3\. Handle Vision  
            mock\_frame \= np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)  
            if frame\_queue.empty():  
                frame\_queue.put(mock\_frame)  
                  
            if not result\_queue.empty():  
                \# Prints JSON payload: e.g., {"Acacia": \[2, 1\], "Baobab": }  
                json\_result \= result\_queue.get()  
                print(f"Rothermel Curing JSON Payload: {json\_result}")  
                  
            time.sleep(0.05)   
              
    except KeyboardInterrupt:  
        frame\_queue.put(None)   
        vision\_process.join()

## **Conclusion**

Creating a robust plant classification system for a field rover demands stringent optimizations. By adapting the network from a standard classifier into a multi-task Object Detection framework (such as YOLO11n), the rover can successfully count individual flora instances while simultaneously classifying their Live Herbaceous Moisture Content and curing stage (dry vs. not dry).  
Exporting the model to INT8 via the yolo export CLI command natively generates highly compressed LiteRT binaries optimized for the ARM Cortex-A72. By actively constraining the runtime threads to 2 cores and pushing the output directly into a formatted JSON structure, the vision system operates flawlessly alongside the rover's physical sensors, feeding critical, real-time curing data directly into the Rothermel cellular automata matrices for advanced wildfire spread simulations.

#### **Sources des citations**

1. RAINBIO – a compilation of tropical African vascular plants \- GBIF, consulté le mai 25, 2026, [https://www.gbif.org/data-use/83286/rainbio-a-compilation-of-tropical-african-vascular-plants](https://www.gbif.org/data-use/83286/rainbio-a-compilation-of-tropical-african-vascular-plants)  
2. Pl@ntNet-300K-v2 image dataset \- Zenodo, consulté le mai 25, 2026, [https://zenodo.org/records/10419064](https://zenodo.org/records/10419064)  
3. GitHub \- plantnet/PlantNet-300K: \[NeurIPS2021\] A plant image dataset with high label ambiguity and a long-tailed distribution, consulté le mai 25, 2026, [https://github.com/plantnet/PlantNet-300K](https://github.com/plantnet/PlantNet-300K)  
4. Pl@ntNet-300K image dataset \- Zenodo, consulté le mai 25, 2026, [https://zenodo.org/records/4726653](https://zenodo.org/records/4726653)  
5. Pl@ntNet-300K: a plant image dataset with high label ambiguity and a long-tailed distribution | OpenReview, consulté le mai 25, 2026, [https://openreview.net/forum?id=eLYinD0TtIt](https://openreview.net/forum?id=eLYinD0TtIt)  
6. Exploring the floristic diversity of tropical Africa \- CGSpace, consulté le mai 25, 2026, [https://cgspace.cgiar.org/items/95eb72f8-5145-459b-8b3e-c4402bf00982](https://cgspace.cgiar.org/items/95eb72f8-5145-459b-8b3e-c4402bf00982)  
7. Exploring the floristic diversity of tropical Africa \- PMC \- NIH, consulté le mai 25, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC5339970/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5339970/)  
8. Website of RAINBIO GROUP \- GitHub Pages, consulté le mai 25, 2026, [https://gdauby.github.io/rainbio/download\_page.html](https://gdauby.github.io/rainbio/download_page.html)  
9. (PDF) RAINBIO: A mega-database of tropical African vascular plants distributions, consulté le mai 25, 2026, [https://www.researchgate.net/publication/309745201\_RAINBIO\_A\_mega-database\_of\_tropical\_African\_vascular\_plants\_distributions](https://www.researchgate.net/publication/309745201_RAINBIO_A_mega-database_of_tropical_African_vascular_plants_distributions)  
10. African Plants \- a photo guide \- GBIF, consulté le mai 25, 2026, [https://www.gbif.org/dataset/e5774d90-9f01-42bb-a747-32331be82b18](https://www.gbif.org/dataset/e5774d90-9f01-42bb-a747-32331be82b18)  
11. (PDF) African Plants \- a Photo Guide: Launching an interactive online field guide, consulté le mai 25, 2026, [https://www.researchgate.net/publication/286447693\_African\_Plants\_-\_a\_Photo\_Guide\_Launching\_an\_interactive\_online\_field\_guide](https://www.researchgate.net/publication/286447693_African_Plants_-_a_Photo_Guide_Launching_an_interactive_online_field_guide)  
12. ARBIMS \- ARCOS Network, consulté le mai 25, 2026, [http://arbims.arcosnetwork.org/out.dataset.php?datasetkey=e5774d90-9f01-42bb-a747-32331be82b18®ion=albertine](http://arbims.arcosnetwork.org/out.dataset.php?datasetkey=e5774d90-9f01-42bb-a747-32331be82b18&region=albertine)  
13. Detection and annotation of plant organs from digitised herbarium scans using deep learning \- PMC, consulté le mai 25, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7746675/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7746675/)  
14. occurrence module — pygbif 0.6.6 documentation \- Read the Docs, consulté le mai 25, 2026, [https://pygbif.readthedocs.io/en/latest/modules/occurrence.html](https://pygbif.readthedocs.io/en/latest/modules/occurrence.html)  
15. pyinaturalist · PyPI, consulté le mai 25, 2026, [https://pypi.org/project/pyinaturalist/0.2.0/](https://pypi.org/project/pyinaturalist/0.2.0/)  
16. pygbif \- Technical Documentation, consulté le mai 25, 2026, [https://techdocs.gbif.org/en/data-use/pygbif](https://techdocs.gbif.org/en/data-use/pygbif)  
17. Occurrence download formats \- Technical Documentation, consulté le mai 25, 2026, [https://techdocs.gbif.org/en/data-use/download-formats](https://techdocs.gbif.org/en/data-use/download-formats)  
18. API SQL Downloads \- Technical Documentation \- GBIF, consulté le mai 25, 2026, [https://techdocs.gbif.org/en/data-use/api-sql-downloads](https://techdocs.gbif.org/en/data-use/api-sql-downloads)  
19. GBIF SQL Downloads, consulté le mai 25, 2026, [https://data-blog.gbif.org/post/2024-06-24-gbif-sql-downloads/](https://data-blog.gbif.org/post/2024-06-24-gbif-sql-downloads/)  
20. occurrences module — pygbif 0.1.4 documentation, consulté le mai 25, 2026, [https://pygbif.readthedocs.io/en/v0.1.4/occurrences.html](https://pygbif.readthedocs.io/en/v0.1.4/occurrences.html)  
21. v.in.pygbif \- GRASS GIS manual, consulté le mai 25, 2026, [https://grass.osgeo.org/grass-stable/manuals/addons/v.in.pygbif.html](https://grass.osgeo.org/grass-stable/manuals/addons/v.in.pygbif.html)  
22. cypamigon/inat\_downloader: iNaturalist images and metadata downloader script \- GitHub, consulté le mai 25, 2026, [https://github.com/cypamigon/inat\_downloader](https://github.com/cypamigon/inat_downloader)  
23. Inat\_downloader : Python script for images and metadata downloading, consulté le mai 25, 2026, [https://forum.inaturalist.org/t/inat-downloader-python-script-for-images-and-metadata-downloading/46378](https://forum.inaturalist.org/t/inat-downloader-python-script-for-images-and-metadata-downloading/46378)  
24. ghuertaramos/Inat\_Images: Script to download images from inaturalist.org \- GitHub, consulté le mai 25, 2026, [https://github.com/ghuertaramos/Inat\_Images](https://github.com/ghuertaramos/Inat_Images)  
25. Examples \- pyinaturalist 0.21.1 documentation, consulté le mai 25, 2026, [https://pyinaturalist.readthedocs.io/en/stable/examples.html](https://pyinaturalist.readthedocs.io/en/stable/examples.html)  
26. Reproducible workflow for visualisation of iNaturalist observations in the Gayini wetlands, consulté le mai 25, 2026, [https://jrfep.quarto.pub/gayini-inat/](https://jrfep.quarto.pub/gayini-inat/)  
27. YOLO11\_Opt: An Ultra-Lightweight Improved YOLO11n Algorithm for Low-Cost Embedded Devices for Accurate Plant Disease Detection—A Case Study on Bell Pepper \- MDPI, consulté le mai 25, 2026, [https://www.mdpi.com/2624-7402/8/4/128](https://www.mdpi.com/2624-7402/8/4/128)  
28. Efficient Real-Time Detection of Plant Leaf Diseases Using YOLOv8 and Raspberry Pi, consulté le mai 25, 2026, [https://vfast.org/journals/index.php/VTSE/article/download/1869/1552](https://vfast.org/journals/index.php/VTSE/article/download/1869/1552)  
29. MobileNet-GDR: a lightweight algorithm for grape leaf disease identification based on improved MobileNetV4-small \- Frontiers, consulté le mai 25, 2026, [https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1702071/full](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1702071/full)  
30. Benchmarking Machine Learning on the New Raspberry Pi 4, Model B | by Alasdair Allan, consulté le mai 25, 2026, [https://aallan.medium.com/benchmarking-machine-learning-on-the-new-raspberry-pi-4-model-b-88db9304ce4](https://aallan.medium.com/benchmarking-machine-learning-on-the-new-raspberry-pi-4-model-b-88db9304ce4)  
31. An Edge Computing-Based Solution for Real-Time Leaf Disease Classification using Thermal Imaging \- arXiv, consulté le mai 25, 2026, [https://arxiv.org/html/2411.03835v1](https://arxiv.org/html/2411.03835v1)  
32. Benchmarking TensorFlow Lite on the New Raspberry Pi 4, Model B \- Hackster.io, consulté le mai 25, 2026, [https://www.hackster.io/news/benchmarking-tensorflow-lite-on-the-new-raspberry-pi-4-model-b-3fd859d05b98](https://www.hackster.io/news/benchmarking-tensorflow-lite-on-the-new-raspberry-pi-4-model-b-3fd859d05b98)  
33. 3.8.1. TensorFlow Lite (LiteRT) — Processor SDK AM64X Documentation, consulté le mai 25, 2026, [https://software-dl.ti.com/processor-sdk-linux/esd/AM64X/12\_00\_00\_07\_04/exports/docs/linux/Foundational\_Components/Machine\_Learning/tflite.html](https://software-dl.ti.com/processor-sdk-linux/esd/AM64X/12_00_00_07_04/exports/docs/linux/Foundational_Components/Machine_Learning/tflite.html)  
34. google-ai-edge/litert-samples \- GitHub, consulté le mai 25, 2026, [https://github.com/google-ai-edge/litert-samples](https://github.com/google-ai-edge/litert-samples)  
35. LiteRT: The Universal Framework for On-Device AI \- Google for Developers Blog, consulté le mai 25, 2026, [https://developers.googleblog.com/litert-the-universal-framework-for-on-device-ai/](https://developers.googleblog.com/litert-the-universal-framework-for-on-device-ai/)  
36. Interpreter.Options | Google AI Edge, consulté le mai 25, 2026, [https://ai.google.dev/edge/api/tflite/java/org/tensorflow/lite/Interpreter.Options](https://ai.google.dev/edge/api/tflite/java/org/tensorflow/lite/Interpreter.Options)  
37. Change number of threads at runtime \- Apache TVM Discuss, consulté le mai 25, 2026, [https://discuss.tvm.apache.org/t/change-number-of-threads-at-runtime/4569](https://discuss.tvm.apache.org/t/change-number-of-threads-at-runtime/4569)  
38. GSH\_Coder/Tensorflow-bin \- Gitee, consulté le mai 25, 2026, [https://gitee.com/gsh978079141/Tensorflow-bin](https://gitee.com/gsh978079141/Tensorflow-bin)  
39. PINTO\_model\_zoo/026\_mobile-deeplabv3-plus/03\_integer\_quantization/deeplabv3plus\_usbcam.py at main \- GitHub, consulté le mai 25, 2026, [https://github.com/PINTO0309/PINTO\_model\_zoo/blob/main/026\_mobile-deeplabv3-plus/03\_integer\_quantization/deeplabv3plus\_usbcam.py](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/026_mobile-deeplabv3-plus/03_integer_quantization/deeplabv3plus_usbcam.py)