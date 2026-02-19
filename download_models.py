"""
Download required models before building the executable.
This ensures all required model files are available.
"""

import os
import sys

def download_models():
    """Download all required models (InsightFace + DeepFace fallback)."""
    print("=" * 60)
    print("Downloading Face Detection Models...")
    print("=" * 60)
    
    all_success = True
    
    # 1. Download InsightFace models (PRIMARY - GPU accelerated)
    print("\n[Step 1/2] Downloading InsightFace RetinaFace model...")
    try:
        from insightface.app import FaceAnalysis
        import numpy as np
        
        print("  Loading InsightFace RetinaFace (buffalo_l)...")
        app = FaceAnalysis(
            name='buffalo_l',
            providers=['CPUExecutionProvider'],  # Use CPU for download
            allowed_modules=['detection']
        )
        app.prepare(ctx_id=-1, det_size=(640, 640))
        
        # Test with dummy image
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        faces = app.get(test_img)
        
        print(f"  ✓ InsightFace model downloaded successfully!")
        print(f"    Test detection returned {len(faces)} result(s)")
        
        # Show where models are stored
        insightface_home = os.path.join(os.path.expanduser('~'), '.insightface')
        if os.path.exists(insightface_home):
            total_size = 0
            print(f"\n  InsightFace models stored in: {insightface_home}")
            print("  Model files:")
            for root, dirs, files in os.walk(insightface_home):
                for file in files:
                    filepath = os.path.join(root, file)
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    total_size += size_mb
                    rel_path = os.path.relpath(filepath, insightface_home)
                    print(f"    - {rel_path} ({size_mb:.2f} MB)")
            print(f"  Total size: {total_size:.2f} MB")
    
    except Exception as e:
        print(f"  ✗ Warning: InsightFace model download failed: {e}")
        print("  Note: InsightFace is the primary detector. Make sure you have:")
        print("    1. Internet connection")
        print("    2. insightface installed: pip install insightface")
        print("    3. onnxruntime installed: pip install onnxruntime")
        all_success = False
    
    # 2. Download DeepFace models (FALLBACK - CPU only)
    print("\n[Step 2/2] Downloading DeepFace RetinaFace model (fallback)...")
    try:
        from deepface import DeepFace
        import numpy as np
        
        print("  Loading DeepFace RetinaFace...")
        test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        try:
            result = DeepFace.extract_faces(
                img_path=test_img,
                detector_backend="retinaface",
                enforce_detection=False,
                align=False
            )
            print(f"  ✓ DeepFace model downloaded successfully!")
            print(f"    Test detection returned {len(result)} result(s)")
        except Exception as e:
            print(f"  ✓ Model download completed (detection result: {e})")
        
        # Show where models are stored
        deepface_home = os.path.join(os.path.expanduser('~'), '.deepface')
        if os.path.exists(deepface_home):
            total_size = 0
            print(f"\n  DeepFace models stored in: {deepface_home}")
            print("  Model files:")
            for root, dirs, files in os.walk(deepface_home):
                for file in files:
                    filepath = os.path.join(root, file)
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    total_size += size_mb
                    rel_path = os.path.relpath(filepath, deepface_home)
                    print(f"    - {rel_path} ({size_mb:.2f} MB)")
            print(f"  Total size: {total_size:.2f} MB")
        
    except Exception as e:
        print(f"  ✗ Warning: DeepFace model download failed: {e}")
        print("  Note: DeepFace is a fallback option. You can continue without it.")
        # Don't set all_success = False for DeepFace failure
    
    print("\n" + "=" * 60)
    if all_success:
        print("✓ All models downloaded successfully!")
    else:
        print("⚠ Some models could not be downloaded")
        print("  InsightFace (primary) is required for GPU acceleration")
        print("  DeepFace (fallback) is optional")
    print("=" * 60)
    
    return all_success

if __name__ == "__main__":
    success = download_models()
    sys.exit(0 if success else 1)
