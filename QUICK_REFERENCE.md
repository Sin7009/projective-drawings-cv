# Quick Reference: Common Code Improvements

This quick reference shows the most impactful improvements made to the codebase with before/after examples.

## 1. Replace Magic Numbers with Constants

### ❌ Before
```python
bg_blur = cv2.GaussianBlur(gray, (101, 101), 0)
ink_mask = inverted_norm > 10
```

### ✅ After
```python
BACKGROUND_BLUR_KERNEL_SIZE = 101
INK_DETECTION_THRESHOLD = 10

bg_blur = cv2.GaussianBlur(gray, (BACKGROUND_BLUR_KERNEL_SIZE, BACKGROUND_BLUR_KERNEL_SIZE), 0)
ink_mask = inverted_norm > INK_DETECTION_THRESHOLD
```

**Why?** Constants make code self-documenting and easier to tune.

---

## 2. Add Input Validation

### ❌ Before
```python
def process_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray
```

### ✅ After
```python
def process_image(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Image cannot be None or empty")
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return gray
```

**Why?** Prevents runtime errors and provides clear feedback.

---

## 3. Improve Docstrings

### ❌ Before
```python
def extract_features(image: np.ndarray) -> Dict[str, float]:
    """Extract features from image."""
    pass
```

### ✅ After
```python
def extract_features(image: np.ndarray) -> Dict[str, float]:
    """
    Extract psychological features from a Wartegg square drawing.
    
    Args:
        image: Binary image of the square (white strokes on black background)
        
    Returns:
        Dictionary containing extracted features with keys:
        - 'centroid_x': X coordinate of drawing centroid
        - 'centroid_y': Y coordinate of drawing centroid
        - 'displacement_distance': Distance from image center
        
    Raises:
        ValueError: If image is None or empty
        
    Example:
        >>> img = cv2.imread("square_1.png", cv2.IMREAD_GRAYSCALE)
        >>> features = extract_features(img)
        >>> print(features['displacement_distance'])
        35.5
    """
    pass
```

**Why?** Helps developers understand usage without reading implementation.

---

## 4. Use Type Hints Properly

### ❌ Before
```python
def process_data(data, threshold=None):
    pass
```

### ✅ After
```python
from typing import Optional, Dict, Any

def process_data(data: np.ndarray, threshold: Optional[float] = None) -> Dict[str, Any]:
    pass
```

**Why?** Enables IDE autocomplete and catches type errors early.

---

## 5. Better Error Messages

### ❌ Before
```python
if threshold < 0:
    raise ValueError("Invalid threshold")
```

### ✅ After
```python
if threshold < 0:
    raise ValueError(f"Threshold must be non-negative, got {threshold}")
```

**Why?** Specific error messages speed up debugging.

---

## 6. Safe Division

### ❌ Before
```python
aspect_ratio = width / (height + 1e-5)
```

### ✅ After
```python
MIN_DIMENSION_EPSILON = 1e-5

aspect_ratio = width / (height + MIN_DIMENSION_EPSILON)
```

**Why?** Named constant explains the purpose of the epsilon.

---

## 7. Configuration Management

### ❌ Before
```python
def __init__(self):
    self.threshold = 100.0
```

### ✅ After
```python
from src.config import config

DEFAULT_THRESHOLD = 100.0

def __init__(self, threshold: Optional[float] = None):
    self.threshold = threshold or config.get('default.threshold', DEFAULT_THRESHOLD)
```

**Why?** Centralized configuration makes system more flexible.

---

## 8. Proper Exception Handling

### ❌ Before
```python
try:
    image = cv2.imread(path)
except Exception as e:
    print(f"Error: {e}")
```

### ✅ After
```python
try:
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
except FileNotFoundError as e:
    logger.error(f"Image file not found: {path}")
    raise
except ValueError as e:
    logger.error(f"Invalid image format: {e}")
    raise
```

**Why?** Specific exceptions allow targeted error handling.

---

## 9. Function Initialization

### ❌ Before
```python
def __init__(self, width=None):
    self.width = width or 2000
```

### ✅ After
```python
DEFAULT_WIDTH = 2000

def __init__(self, width: Optional[int] = None):
    """
    Initialize with optional width parameter.
    
    Args:
        width: Target width in pixels (uses default if None)
    """
    self.width = width or config.get('preprocessing.width', DEFAULT_WIDTH)
```

**Why?** Clear defaults and configuration hierarchy.

---

## 10. Vectorized Operations

### ❌ Before
```python
distances = []
for x, y in zip(x_coords, y_coords):
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    distances.append(dist)
```

### ✅ After
```python
distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
```

**Why?** Numpy vectorization is much faster and cleaner.

---

## Quick Checklist for New Code

Before committing code, verify:
- [ ] No magic numbers (use constants)
- [ ] All functions have type hints
- [ ] All public methods have docstrings
- [ ] Input validation is present
- [ ] Specific exception types are used
- [ ] Error messages provide context
- [ ] Constants use descriptive names
- [ ] Configuration values are configurable
- [ ] No unnecessary print statements
- [ ] Imports are organized

---

## Common Patterns

### Pattern: Safe Dictionary Access
```python
value = config.get('key.subkey', DEFAULT_VALUE)
```

### Pattern: Validate Then Process
```python
def process(data: np.ndarray) -> np.ndarray:
    # 1. Validate
    if data is None or data.size == 0:
        raise ValueError("Invalid data")
    
    # 2. Process
    result = transform(data)
    
    # 3. Return
    return result
```

### Pattern: Early Return for Edge Cases
```python
def calculate_features(image: np.ndarray) -> Dict[str, float]:
    if np.count_nonzero(image) == 0:
        return {"count": 0, "mean": 0.0}
    
    # Normal processing
    return compute_features(image)
```

---

## Anti-Patterns to Avoid

### ❌ Bare Except
```python
try:
    process()
except:  # Catches everything, including KeyboardInterrupt
    pass
```

### ❌ Mutable Default Arguments
```python
def process(items=[]):  # Same list shared across calls!
    items.append(1)
```

### ❌ String Concatenation in Loops
```python
result = ""
for item in items:
    result += str(item)  # Creates new string each time
```

### ✅ Use join instead
```python
result = "".join(str(item) for item in items)
```

---

For more details, see:
- [BEST_PRACTICES.md](BEST_PRACTICES.md) - Complete coding guidelines
- [IMPROVEMENTS.md](IMPROVEMENTS.md) - Detailed change summary
- [CODE_REVIEW_SUMMARY.md](CODE_REVIEW_SUMMARY.md) - Executive summary
