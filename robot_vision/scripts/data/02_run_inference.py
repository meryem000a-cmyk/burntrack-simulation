import os
import glob
import numpy as np
import cv2
import yaml
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    print("tflite_runtime missing. Run: pip install tflite-runtime")
    exit(1)

# Load class names from yaml
with open("flora_data.yaml", "r") as f:
    data_yaml = yaml.safe_load(f)
class_names = data_yaml.get("names", {})

model_path = "runs/detect/train/weights/best_saved_model/best_full_integer_quant.tflite"

if not os.path.exists(model_path):
    # Try alternate location
    model_path = "runs/detect/train/weights/best_full_integer_quant.tflite"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Did the quantization step succeed?")
        exit(1)

print(f"Loading quantized model: {model_path}")
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

in_scale, in_zero_point = input_details[0].get('quantization', (0.0, 0))
out_scale, out_zero_point = output_details[0].get('quantization', (0.0, 0))

test_images = glob.glob("datasets/yolo_flora/images/val/*.jpg")
if not test_images:
    print("No test images found!")
    exit(1)

# Sort or take first 10
test_images = test_images[:10]
print(f"\n--- Running Inference on {len(test_images)} UNSEEN Test Images ---\n")

for img_path in test_images:
    img = cv2.imread(img_path)
    if img is None:
        continue
        
    img_resized = cv2.resize(img, (320, 320))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = (img_rgb.astype(np.float32) / 255.0)
    
    if in_scale != 0.0:
        img_quantized = (img_normalized / in_scale + in_zero_point).astype(np.int8)
    else:
        img_quantized = img_normalized.astype(np.int8)
        
    img_expanded = np.expand_dims(img_quantized, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], img_expanded)
    interpreter.invoke()
    
    raw_output = interpreter.get_tensor(output_details[0]['index'])
    raw_output = np.transpose(raw_output, (0, 2, 1))
    
    if out_scale != 0.0:
        predictions = (raw_output.astype(np.float32) - out_zero_point) * out_scale
    else:
        predictions = raw_output
        
    preds = predictions[0]
    
    class_confs = preds[:, 4:]
    max_confs = np.max(class_confs, axis=1)
    best_idx = np.argmax(max_confs)
    
    best_conf = max_confs[best_idx]
    best_class = np.argmax(class_confs[best_idx])
    
    predicted_label = class_names.get(int(best_class), "Unknown")
    
    print(f"Image: {os.path.basename(img_path)}")
    if best_conf > 0.1:
        print(f"  Prediction: {predicted_label} (Confidence: {best_conf:.2f})")
    else:
        print(f"  Prediction: None (Confidence too low: {best_conf:.2f})")
    print("-" * 40)
