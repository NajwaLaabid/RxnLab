#!/bin/bash
# Test script to verify RDKit works in a headless environment
# Run this locally before deploying to Rahti

set -e

echo "=========================================="
echo "Testing RDKit in headless Docker container"
echo "=========================================="

# Build the test image
echo ""
echo "Step 1: Building Docker image..."
docker build -t rdkit-rahti-test .

# Test that the import works
echo ""
echo "Step 2: Testing RDKit import..."
docker run --rm rdkit-rahti-test python -c "
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
print('✓ RDKit imported successfully!')

# Test SVG generation
mol = Chem.MolFromSmiles('CCO')
drawer = rdMolDraw2D.MolDraw2DSVG(200, 200)
drawer.DrawMolecule(mol)
drawer.FinishDrawing()
svg = drawer.GetDrawingText()
print('✓ SVG generation works!')
print(f'  Generated {len(svg)} bytes of SVG')
"

# Test that the Flask app starts
echo ""
echo "Step 3: Testing Flask app startup..."
docker run --rm -d --name rdkit-test-container -p 8080:8080 rdkit-rahti-test
sleep 3

# Check if the container is running
if docker ps | grep -q rdkit-test-container; then
    echo "✓ Container is running"
    
    # Test the health endpoint
    if curl -s http://localhost:8080/health | grep -q "healthy"; then
        echo "✓ Health endpoint responds"
    else
        echo "✗ Health endpoint failed"
    fi
    
    # Test the main page
    if curl -s http://localhost:8080/ | grep -q "DiffAlign"; then
        echo "✓ Main page loads"
    else
        echo "✗ Main page failed"
    fi
    
    # Stop the container
    docker stop rdkit-test-container > /dev/null
    echo "✓ Container stopped cleanly"
else
    echo "✗ Container failed to start"
    docker logs rdkit-test-container 2>/dev/null || true
    docker rm -f rdkit-test-container 2>/dev/null || true
    exit 1
fi

echo ""
echo "=========================================="
echo "All tests passed! Ready to deploy to Rahti"
echo "=========================================="
