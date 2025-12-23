# Best Practices Guide for Projective Drawings CV Project

## Code Style

### 1. Constants
Always define constants at the module level for magic numbers and configuration values:

```python
# Good
BACKGROUND_BLUR_KERNEL_SIZE = 101
INK_DETECTION_THRESHOLD = 10

def process_image(image):
    blurred = cv2.GaussianBlur(image, (BACKGROUND_BLUR_KERNEL_SIZE, BACKGROUND_BLUR_KERNEL_SIZE), 0)
    mask = inverted > INK_DETECTION_THRESHOLD
```

```python
# Bad
def process_image(image):
    blurred = cv2.GaussianBlur(image, (101, 101), 0)
    mask = inverted > 10
```

### 2. Type Hints
Use type hints for all function parameters and return values:

```python
# Good
def extract_features(image: np.ndarray, threshold: Optional[float] = None) -> Dict[str, Any]:
    pass

# Bad
def extract_features(image, threshold=None):
    pass
```

### 3. Docstrings
Write comprehensive docstrings for all public methods:

```python
def extract_square_features(image: np.ndarray, square_id: int) -> Dict[str, float]:
    """
    Extract psychological features from a Wartegg square drawing.
    
    Args:
        image: Binary image of the square (white strokes on black background)
        square_id: Square identifier (1-8)
        
    Returns:
        Dictionary containing extracted features specific to the square
        
    Raises:
        ValueError: If image is None, empty, or square_id is invalid
        
    Example:
        >>> image = load_square_image("square_1.png")
        >>> features = extract_square_features(image, 1)
        >>> print(features['centroid_x'])
        75.5
    """
```

## Input Validation

### Always Validate Inputs
```python
def process_data(data: np.ndarray) -> np.ndarray:
    if data is None or data.size == 0:
        raise ValueError("Data cannot be None or empty")
    
    if len(data.shape) != 2:
        raise ValueError(f"Expected 2D array, got {len(data.shape)}D")
    
    # Process data
    return result
```

### Check File Existence
```python
def load_image(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    
    return image
```

## Error Handling

### Use Specific Exceptions
```python
# Good
try:
    image = load_image(path)
except FileNotFoundError as e:
    logger.error(f"Image file missing: {e}")
    raise
except ValueError as e:
    logger.error(f"Invalid image format: {e}")
    raise

# Bad
try:
    image = load_image(path)
except Exception as e:
    print("Error:", e)
```

### Provide Context in Errors
```python
# Good
raise ValueError(f"Square ID must be 1-8, got {square_id}")

# Bad
raise ValueError("Invalid square ID")
```

## Safe Mathematics

### Avoid Division by Zero
```python
# Good
EPSILON = 1e-5
aspect_ratio = width / (height + EPSILON)

# Even Better
if height == 0:
    aspect_ratio = 0.0
else:
    aspect_ratio = width / height
```

### Handle Edge Cases
```python
def calculate_density(pixels: int, total: int) -> float:
    """Calculate pixel density safely."""
    if total <= 0:
        return 0.0
    return pixels / total
```

## Configuration Management

### Use Config System
```python
# Good
from src.config import config

threshold = config.get('preprocessing.blur_threshold', DEFAULT_BLUR_THRESHOLD)

# Bad
threshold = 100.0  # hardcoded
```

### Provide Defaults
```python
def __init__(self, threshold: Optional[float] = None):
    self.threshold = threshold or config.get('default.threshold', DEFAULT_THRESHOLD)
```

## Image Processing Best Practices

### Check Image Properties
```python
def preprocess_image(image: np.ndarray) -> np.ndarray:
    # Validate input
    if image is None or image.size == 0:
        raise ValueError("Invalid image")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    return gray
```

### Handle Different Image Formats
```python
def binarize(image: np.ndarray) -> np.ndarray:
    # Ensure grayscale
    gray = to_grayscale(image)
    
    # Auto-detect if inversion needed
    mean_value = np.mean(gray)
    if mean_value > 127:
        gray = cv2.bitwise_not(gray)
    
    return gray
```

### Normalize Outputs
```python
def extract_pressure(image: np.ndarray) -> float:
    """Returns pressure in range [0, 1]."""
    raw_pressure = calculate_raw_pressure(image)
    return float(np.clip(raw_pressure / 255.0, 0.0, 1.0))
```

## Logging Best Practices

### Use Structured Logging
```python
from loguru import logger

# Good
logger.info(f"Processing image: {image_path}")
logger.debug(f"Found {len(tokens)} tokens in square {square_id}")
logger.error(f"Failed to process {image_path}: {error}")

# Bad
print("Processing...")
print(f"Error: {error}")
```

### Log at Appropriate Levels
- `DEBUG`: Detailed diagnostic information
- `INFO`: General informational messages
- `WARNING`: Warning messages for potentially harmful situations
- `ERROR`: Error messages for failures

## Testing Guidelines

### Write Clear Test Names
```python
def test_extract_square_1_features_with_centered_dot():
    """Test that centered dot produces zero displacement."""
    pass

def test_extract_square_1_features_with_corner_dot():
    """Test that corner dot produces maximum displacement."""
    pass
```

### Test Edge Cases
```python
def test_empty_image():
    """Test handling of empty image."""
    image = np.zeros((100, 100), dtype=np.uint8)
    result = extract_features(image)
    assert result['count'] == 0

def test_single_pixel():
    """Test handling of single pixel."""
    image = np.zeros((100, 100), dtype=np.uint8)
    image[50, 50] = 255
    result = extract_features(image)
    assert result['centroid_x'] == 50
```

### Use Fixtures
```python
import pytest

@pytest.fixture
def sample_square_image():
    """Create a sample square image for testing."""
    image = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(image, (50, 50), 10, 255, -1)
    return image

def test_with_fixture(sample_square_image):
    result = process_image(sample_square_image)
    assert result is not None
```

## Performance Considerations

### Avoid Repeated Calculations
```python
# Good
height, width = image.shape
center_x = width / 2
center_y = height / 2

# Bad
for point in points:
    dx = point[0] - image.shape[1] / 2  # Recalculating every time
```

### Use Vectorized Operations
```python
# Good
distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)

# Bad
distances = []
for x, y in zip(x_coords, y_coords):
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    distances.append(dist)
```

### Cache Expensive Computations
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def compute_expensive_feature(image_hash: int) -> float:
    """Compute and cache expensive features."""
    pass
```

## Documentation

### Keep README Updated
- Update installation instructions
- Add usage examples
- Document configuration options
- Include troubleshooting tips

### Document Complex Algorithms
```python
def find_paper_contour(image: np.ndarray) -> np.ndarray:
    """
    Find the paper contour using edge detection and contour approximation.
    
    Algorithm:
    1. Convert to grayscale
    2. Apply Gaussian blur to reduce noise
    3. Use Canny edge detection
    4. Find contours and sort by area
    5. Approximate contours to polygons
    6. Return the first 4-sided contour (assumed to be paper)
    
    This approach is robust to lighting variations and small distortions.
    """
```

### Add Inline Comments for Complex Logic
```python
# Check if dots are connected: if we have ~8 dots (stimulus)
# but only <=4 contours, they must be connected
if square_id == 7:
    is_connected = (num_contours <= MAX_CONTOURS_FOR_CONNECTION and 
                   pixel_count > MIN_DRAWN_PIXELS)
```

## Version Control

### Write Clear Commit Messages
```
Good:
- "Add input validation to extract_features()"
- "Fix division by zero in aspect ratio calculation"
- "Improve docstrings for FeatureExtractor class"

Bad:
- "fix bug"
- "update code"
- "changes"
```

### Keep Commits Focused
Each commit should address a single concern:
- One bug fix per commit
- One feature per commit
- Separate refactoring from functional changes

## Code Review Checklist

Before submitting code:
- [ ] All functions have type hints
- [ ] All public methods have docstrings
- [ ] Input validation is present
- [ ] Constants are used instead of magic numbers
- [ ] Error handling is appropriate
- [ ] Tests are added/updated
- [ ] Documentation is updated
- [ ] Code follows project style
- [ ] No unnecessary print statements
- [ ] Imports are organized

## Summary

Following these best practices will ensure:
- **Maintainability**: Code is easy to understand and modify
- **Reliability**: Proper error handling prevents failures
- **Performance**: Efficient algorithms and caching
- **Collaboration**: Clear documentation helps team members
- **Quality**: Consistent standards across the codebase
