import numpy as np
import cv2
import time
import json
from multiprocessing import Process, Queue
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    print("tflite_runtime missing. Run: pip install tflite-runtime")
    exit(1)

def vision_inference_process(input_queue, output_queue, model_path, class_names):
    """
    Isolated process for running multi-task plant and dryness detection.
    Outputs a JSON string containing plant types and their dry/not_dry counts.
    """
    interpreter = Interpreter(model_path=model_path)
    
    # CRITICAL: Restrict CPU utilization to leave headroom for rover sensors.
    # Limiting to 2 threads ensures 2 cores are entirely free on the RPi 4.
    try:
        interpreter.set_num_threads(2)
        print("LiteRT Interpreter locked to 2 threads for CPU headroom.")
    except Exception as e:
        print(f"Warning: Thread configuration failed: {e}")
        
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    # Get expected input shape
    input_shape = input_details['shape']
    img_h, img_w = input_shape[1], input_shape[2]
    input_dtype = input_details['dtype']
    
    # Precompute quantization parameters
    in_quant = input_details.get('quantization_parameters', {})
    in_scale = in_quant.get('scales', [0])[0] if in_quant else 0
    in_zero_point = in_quant.get('zero_points', [0])[0] if in_quant else 0
    
    out_quant = output_details.get('quantization_parameters', {})
    out_scale = out_quant.get('scales', [0])[0] if out_quant else 0
    out_zero_point = out_quant.get('zero_points', [0])[0] if out_quant else 0
    
    while True:
        if not input_queue.empty():
            frame = input_queue.get()
            if frame is None: break 
                
            # Preprocessing: Resize and format to NHWC
            img_resized = cv2.resize(frame, (img_w, img_h))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_normalized = img_rgb.astype(np.float32) / 255.0
            
            # Quantize input based on model dtype
            if input_dtype == np.int8:
                if in_scale != 0:
                    img_input = (img_normalized / in_scale + in_zero_point).astype(np.int8)
                else:
                    img_input = img_normalized.astype(np.int8)
            elif input_dtype == np.uint8:
                if in_scale != 0:
                    img_input = (img_normalized / in_scale + in_zero_point).astype(np.uint8)
                else:
                    img_input = (img_normalized * 255).astype(np.uint8)
            else:
                img_input = img_normalized
                
            img_expanded = np.expand_dims(img_input, axis=0)
            
            # Execute inference
            interpreter.set_tensor(input_details['index'], img_expanded)
            interpreter.invoke()
            
            # Retrieve and de-quantize outputs
            raw_output = interpreter.get_tensor(output_details['index'])
            if out_scale != 0:
                predictions = (raw_output.astype(np.float32) - out_zero_point) * out_scale
            else:
                predictions = raw_output.astype(np.float32)
                
            # Dictionary to accumulate counts for the JSON output
            # Format: {"PlantName": [dry_count, not_dry_count]}
            frame_counts = {}
            
            # Parse YOLO detection output [x, y, w, h, conf, class_id]
            confidence_threshold = 0.5
            for detection in predictions[0]:  # First batch item
                conf = detection[4]
                if conf > confidence_threshold:
                    class_id = int(detection[5])
                    if class_id not in class_names:
                        continue
                    class_label = class_names[class_id]
                    
                    # Split the label into species and its curing/dryness state
                    if class_label.endswith("_not_dry"):
                        species = class_label[:-8]
                        is_dry = False
                    elif class_label.endswith("_dry"):
                        species = class_label[:-4]
                        is_dry = True
                    else:
                        continue
                    
                    if species not in frame_counts:
                        frame_counts[species] = [0, 0]
                        
                    if is_dry:
                        frame_counts[species][0] += 1  # Increment dry instance
                    else:
                        frame_counts[species][1] += 1  # Increment not_dry instance
            
            # Format as JSON string for the Cellular Automata / Rothermel pipeline
            json_output = json.dumps(frame_counts)
            output_queue.put(json_output)
            
        else:
            time.sleep(0.01)

if __name__ == '__main__':
    frame_queue = Queue(maxsize=1)
    result_queue = Queue(maxsize=1)
    
    # Label map matching data.yaml class ordering
    labels = {
        0: "adansonia_not_dry", 1: "adansonia_dry",
        2: "acacia_not_dry", 3: "acacia_dry",
        4: "vachellia_not_dry", 5: "vachellia_dry",
        6: "senegalia_not_dry", 7: "senegalia_dry",
        8: "combretum_not_dry", 9: "combretum_dry",
        10: "brachystegia_not_dry", 11: "brachystegia_dry",
        12: "colophospermum_not_dry", 13: "colophospermum_dry",
        14: "ficus_not_dry", 15: "ficus_dry",
        16: "khaya_not_dry", 17: "khaya_dry",
        18: "macaranga_not_dry", 19: "macaranga_dry",
        20: "euphorbia_not_dry", 21: "euphorbia_dry",
        22: "aloe_not_dry", 23: "aloe_dry",
        24: "protea_not_dry", 25: "protea_dry",
        26: "erica_not_dry", 27: "erica_dry",
        28: "themeda_not_dry", 29: "themeda_dry",
        30: "andropogon_not_dry", 31: "andropogon_dry",
        32: "tamarix_not_dry", 33: "tamarix_dry"
    }
    
    vision_process = Process(
        target=vision_inference_process, 
        args=(frame_queue, result_queue, "yolo11n_int8.tflite", labels)
    )
    vision_process.daemon = True
    vision_process.start()
    
    try:
        while True:
            # 1. Rover polls GPS, Humidity, Wind, Temp, Inclination - Headroom is protected
            # 2. Control loops execute
            
            # 3. Handle Vision
            mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            if frame_queue.empty():
                frame_queue.put(mock_frame)
                
            if not result_queue.empty():
                # Prints JSON payload: e.g., {"acacia": [2, 1], "baobab": [0, 3]}
                json_result = result_queue.get()
                print(f"Rothermel Curing JSON Payload: {json_result}")
                
            time.sleep(0.05) 
            
    except KeyboardInterrupt:
        frame_queue.put(None) 
        vision_process.join()
