"""
Optional lightweight vision module using YOLO or OpenCV.
Can complement DOM extraction for visual element detection.
"""
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class VisualElement:
    """Detected visual element."""
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    label: str
    center: Tuple[int, int]


class LightweightVisionDetector:
    """
    Lightweight vision detector for UI elements.
    Uses OpenCV for basic detection, optionally YOLO for advanced.
    """
    
    def __init__(self, use_yolo: bool = False, model_path: Optional[str] = None):
        """
        Initialize vision detector.
        
        Args:
            use_yolo: Whether to use YOLO (requires ultralytics)
            model_path: Path to custom YOLO model
        """
        self.use_yolo = use_yolo
        
        if use_yolo:
            try:
                from ultralytics import YOLO
                # Use nano model for CPU efficiency
                self.model = YOLO(model_path or 'yolov8n.pt')
                print("✅ YOLO model loaded")
            except ImportError:
                print("⚠️  ultralytics not installed. Install with: pip install ultralytics")
                self.use_yolo = False
    
    def detect_buttons(self, image: np.ndarray) -> List[VisualElement]:
        """
        Detect button-like elements using OpenCV.
        
        Args:
            image: Image as numpy array (BGR)
            
        Returns:
            List of detected visual elements
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        elements = []
        for contour in contours:
            # Filter by area (buttons are usually medium-sized)
            area = cv2.contourArea(contour)
            if 500 < area < 50000:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio (buttons are usually rectangular)
                aspect_ratio = w / h if h > 0 else 0
                if 0.3 < aspect_ratio < 10:
                    elements.append(VisualElement(
                        bbox=(x, y, w, h),
                        confidence=0.7,  # Fixed confidence for OpenCV
                        label="button",
                        center=(x + w // 2, y + h // 2)
                    ))
        
        return elements
    
    def detect_text_fields(self, image: np.ndarray) -> List[VisualElement]:
        """
        Detect text input fields using OpenCV.
        
        Args:
            image: Image as numpy array (BGR)
            
        Returns:
            List of detected text fields
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Text fields often have horizontal lines
        edges = cv2.Canny(gray, 30, 100)
        
        # Dilate to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 2))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        elements = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 200 < area < 100000:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Text fields are usually wide and short
                aspect_ratio = w / h if h > 0 else 0
                if aspect_ratio > 2:
                    elements.append(VisualElement(
                        bbox=(x, y, w, h),
                        confidence=0.6,
                        label="text_field",
                        center=(x + w // 2, y + h // 2)
                    ))
        
        return elements
    
    def detect_with_yolo(self, image: np.ndarray) -> List[VisualElement]:
        """
        Detect UI elements using YOLO.
        
        Args:
            image: Image as numpy array (BGR)
            
        Returns:
            List of detected elements
        """
        if not self.use_yolo:
            return []
        
        results = self.model(image, verbose=False)
        
        elements = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                label = self.model.names[class_id]
                
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)
                
                elements.append(VisualElement(
                    bbox=(x, y, w, h),
                    confidence=confidence,
                    label=label,
                    center=(x + w // 2, y + h // 2)
                ))
        
        return elements
    
    def detect_all(self, image: np.ndarray) -> Dict[str, List[VisualElement]]:
        """
        Detect all types of UI elements.
        
        Args:
            image: Image as numpy array (BGR)
            
        Returns:
            Dictionary mapping element types to detected elements
        """
        if self.use_yolo:
            elements = self.detect_with_yolo(image)
            # Group by label
            grouped = {}
            for elem in elements:
                if elem.label not in grouped:
                    grouped[elem.label] = []
                grouped[elem.label].append(elem)
            return grouped
        else:
            return {
                "buttons": self.detect_buttons(image),
                "text_fields": self.detect_text_fields(image)
            }
    
    def annotate_image(self, image: np.ndarray, elements: List[VisualElement]) -> np.ndarray:
        """
        Draw bounding boxes on image.
        
        Args:
            image: Image as numpy array (BGR)
            elements: List of visual elements
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        for elem in elements:
            x, y, w, h = elem.bbox
            
            # Color based on label
            if elem.label == "button":
                color = (0, 255, 0)  # Green
            elif elem.label == "text_field":
                color = (255, 0, 0)  # Blue
            else:
                color = (0, 0, 255)  # Red
            
            # Draw rectangle
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            # Add label
            label_text = f"{elem.label} ({elem.confidence:.2f})"
            cv2.putText(
                annotated, label_text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        
        return annotated
    
    def merge_with_dom(
        self, 
        visual_elements: List[VisualElement],
        dom_elements: List[Dict]
    ) -> List[Dict]:
        """
        Merge visual detections with DOM elements.
        
        Args:
            visual_elements: Detected visual elements
            dom_elements: DOM-extracted elements
            
        Returns:
            Merged element list with visual context
        """
        merged = []
        
        for dom_elem in dom_elements:
            elem_dict = dom_elem.copy()
            
            # Try to match with visual detection
            if 'bbox' in dom_elem and dom_elem['bbox']:
                dom_bbox = dom_elem['bbox']
                dom_center = (
                    dom_bbox['x'] + dom_bbox['width'] / 2,
                    dom_bbox['y'] + dom_bbox['height'] / 2
                )
                
                # Find nearest visual element
                min_dist = float('inf')
                nearest_visual = None
                
                for visual_elem in visual_elements:
                    dist = np.sqrt(
                        (visual_elem.center[0] - dom_center[0])**2 +
                        (visual_elem.center[1] - dom_center[1])**2
                    )
                    if dist < min_dist:
                        min_dist = dist
                        nearest_visual = visual_elem
                
                # If close enough, add visual context
                if nearest_visual and min_dist < 50:
                    elem_dict['visual_confidence'] = nearest_visual.confidence
                    elem_dict['visual_label'] = nearest_visual.label
            
            merged.append(elem_dict)
        
        return merged


def image_from_base64(b64_string: str) -> np.ndarray:
    """Convert base64 string to image."""
    import base64
    img_data = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def image_to_base64(image: np.ndarray) -> str:
    """Convert image to base64 string."""
    import base64
    _, buffer = cv2.imencode('.png', image)
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    return img_b64
