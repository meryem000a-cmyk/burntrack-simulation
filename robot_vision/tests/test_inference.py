import unittest
import sys
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image

# Add scripts directory to path to import infer
sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
import infer

class TestVisionInference(unittest.TestCase):
    def setUp(self):
        # Create a temp image file for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.image_path = Path(self.temp_dir.name) / "test_plant.jpg"
        
        # Create a mock 224x224 RGB image and save it
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(self.image_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_model_files_exist(self):
        """Verify that the production ONNX models are present in the models directory."""
        project_dir = Path(__file__).resolve().parents[1]
        yolo_path = project_dir / "models" / "yolo_fp16.onnx"
        cnn_path = project_dir / "models" / "cnn_fp32.onnx"
        
        self.assertTrue(yolo_path.exists(), f"YOLO ONNX model not found at {yolo_path}")
        self.assertTrue(cnn_path.exists(), f"CNN ONNX model not found at {cnn_path}")

    def test_inference_runs(self):
        """Verify that prediction runs on a mock image and returns a valid output structure."""
        try:
            genus, dryness = infer.predict(str(self.image_path))
            
            # Check outputs are strings
            self.assertIsInstance(genus, str)
            self.assertIsInstance(dryness, str)
            
            # Check dryness value is one of the expected classifications
            self.assertIn(dryness, ["dry", "alive"])
            
            # Check genus is in YOLO_CLASSES (or adansonia after mapping)
            expected_classes = infer.YOLO_CLASSES + ["adansonia"]
            self.assertIn(genus, expected_classes)
            
            print(f"\n✅ Test inference succeeded! Predicted genus: {genus}, dryness: {dryness}")
            
        except Exception as e:
            self.fail(f"Inference failed with exception: {e}")

if __name__ == "__main__":
    unittest.main()
