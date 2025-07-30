#!/usr/bin/env python3
"""
Debug tool for PLY file upload issues
"""
import os
import sys
import traceback
from plyfile import PlyData
import numpy as np

def check_ply_file(file_path):
    """Check if a PLY file can be loaded correctly"""
    print(f"🔍 Checking PLY file: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ File does not exist: {file_path}")
        return False
    
    print(f"📊 File size: {os.path.getsize(file_path)} bytes")
    
    try:
        # Try to read the PLY file
        ply = PlyData.read(file_path)
        print(f"✅ PLY file loaded successfully")
        
        # Check vertex data
        if 'vertex' not in ply:
            print("❌ No vertex data found in PLY file")
            return False
        
        v = ply['vertex'].data
        print(f"📈 Number of vertices: {len(v)}")
        
        # Check required coordinates
        required_coords = ['x', 'y', 'z']
        available_props = list(v.dtype.names)
        print(f"📋 Available properties: {available_props}")
        
        missing_coords = [coord for coord in required_coords if coord not in available_props]
        if missing_coords:
            print(f"❌ Missing required coordinates: {missing_coords}")
            return False
        
        # Extract coordinates
        xyz = np.vstack([v['x'], v['y'], v['z']]).T.astype(np.float32)
        print(f"✅ Coordinates extracted: shape {xyz.shape}")
        print(f"📐 X range: [{xyz[:, 0].min():.3f}, {xyz[:, 0].max():.3f}]")
        print(f"📐 Y range: [{xyz[:, 1].min():.3f}, {xyz[:, 1].max():.3f}]")
        print(f"📐 Z range: [{xyz[:, 2].min():.3f}, {xyz[:, 2].max():.3f}]")
        
        # Check optional properties
        if all(key in available_props for key in ['nx', 'ny', 'nz']):
            print("✅ Normals found")
        else:
            print("⚠️  No normals found (this is okay)")
        
        if all(key in available_props for key in ['scale_0', 'scale_1', 'scale_2']):
            print("✅ Scales found")
        else:
            print("⚠️  No scales found (this is okay)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading PLY file: {str(e)}")
        print("📋 Traceback:")
        traceback.print_exc()
        return False

def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔧 Checking dependencies...")
    
    required_modules = [
        'dash',
        'dash_bootstrap_components', 
        'plotly',
        'plyfile',
        'numpy',
        'torch',
        'pandas'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} - NOT FOUND")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n⚠️  Missing modules: {missing_modules}")
        print("Please install them with: pip install " + " ".join(missing_modules))
        return False
    
    print("✅ All dependencies are available")
    return True

def main():
    print("🚀 PLY File Upload Diagnostic Tool\n")
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Please fix dependency issues first")
        return
    
    print("\n" + "="*50)
    
    # Check if a PLY file path is provided
    if len(sys.argv) > 1:
        ply_file = sys.argv[1]
        if check_ply_file(ply_file):
            print(f"\n✅ PLY file {ply_file} should work with the upload system")
        else:
            print(f"\n❌ PLY file {ply_file} has issues that may prevent upload")
    else:
        print("\n💡 Usage: python debug_upload.py <path_to_ply_file>")
        print("   This will check if your PLY file is compatible")
        
        # Check for common PLY files in the project
        common_paths = [
            "uploads/init.ply",
            "../uploads/init.ply", 
            "../../uploads/init.ply",
            "test.ply",
            "example.ply"
        ]
        
        print("\n🔍 Looking for PLY files in common locations...")
        found_files = []
        for path in common_paths:
            if os.path.exists(path):
                found_files.append(path)
                print(f"📁 Found: {path}")
        
        if found_files:
            print(f"\n💡 You can test with: python debug_upload.py {found_files[0]}")
        else:
            print("\n📝 No PLY files found in common locations")
    
    print("\n🔧 Troubleshooting tips:")
    print("1. Make sure your PLY file has x, y, z coordinates")
    print("2. Check that the file is not corrupted")
    print("3. Ensure the file size is reasonable (< 100MB for web upload)")
    print("4. Try with a simple PLY file first")
    print("5. Check the browser console for JavaScript errors")
    print("6. Make sure the Dash app is running on the correct port")

if __name__ == "__main__":
    main() 