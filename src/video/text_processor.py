import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import pytesseract
import random

from .utils.logging_utils import configure_logger, timed, log_exceptions

# Configure logger
logger = configure_logger("TextProcessor")

@dataclass
class TextRegion:
    x: int
    y: int
    width: int
    height: int
    text: str
    confidence: float

class TextProcessor:
    def __init__(self, min_confidence: float = 0.6):
        logger.info("=== TextProcessor Initialization ===")
        self.min_confidence = min_confidence
        logger.info(f"Minimum confidence threshold: {min_confidence}")
        logger.info("=== TextProcessor Initialization Complete ===")
        
    def _validate_frame(self, frame: np.ndarray) -> bool:
        """Validate if a frame is a valid numpy array with correct dimensions."""
        logger.debug("=== TextProcessor Frame Validation ===")
        logger.debug(f"Frame type: {type(frame)}")
        logger.debug(f"Frame attributes: {dir(frame)}")
        
        if not isinstance(frame, np.ndarray):
            logger.error("Frame is not a numpy array")
            return False
            
        if len(frame.shape) < 2:
            logger.error(f"Frame has insufficient dimensions: {frame.shape}")
            return False
            
        if frame.shape[0] <= 0 or frame.shape[1] <= 0:
            logger.error(f"Frame has invalid dimensions: {frame.shape}")
            return False
            
        logger.debug(f"Frame shape: {frame.shape}")
        logger.debug("=== TextProcessor Frame Validation Complete ===")
        return True
        
    @timed
    @log_exceptions
    def detect_text_regions(self, frame: np.ndarray) -> List[TextRegion]:
        """Detect text regions in a frame using basic image processing."""
        logger.info("=== Detecting Text Regions ===")
        if not self._validate_frame(frame):
            logger.error("Frame validation failed")
            return []
            
        try:
            # Convert to grayscale
            logger.debug("Converting frame to grayscale")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            logger.debug("Applying adaptive thresholding")
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Find contours
            logger.debug("Finding contours")
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            logger.info(f"Found {len(contours)} contours")
            
            text_regions = []
            for i, contour in enumerate(contours):
                # Filter contours by area and aspect ratio
                x, y, w, h = cv2.boundingRect(contour)
                logger.debug(f"\nProcessing contour {i+1}:")
                logger.debug(f"Position: ({x}, {y})")
                logger.debug(f"Size: {w}x{h}")
                
                if w < 20 or h < 20:  # Skip too small regions
                    logger.debug("Skipping: Region too small")
                    continue
                    
                # Extract text from region
                logger.debug("Extracting text from region")
                text = self._extract_text(frame, TextRegion(
                    x=x, y=y, width=w, height=h,
                    text="",
                    confidence=0.8  # Default confidence for now
                ))
                if text:
                    logger.info(f"Found text: {text}")
                    text_regions.append(TextRegion(
                        x=x, y=y, width=w, height=h,
                        text=text,
                        confidence=0.8  # Default confidence for now
                    ))
                else:
                    logger.debug("No text found in region")
            
            logger.info(f"Total text regions found: {len(text_regions)}")
            logger.info("=== Text Region Detection Complete ===")
            return text_regions
        except Exception as e:
            logger.error(f"Error in text detection: {str(e)}")
            return []
    
    def _extract_text(self, frame: np.ndarray, region: TextRegion) -> str:
        """Extract text from a region using Tesseract OCR."""
        logger.info("\n=== Extracting Text ===")
        if not self._validate_frame(frame):
            logger.error("Frame validation failed")
            return ""
            
        try:
            # Extract the region of interest
            logger.debug(f"Extracting region: ({region.x}, {region.y}) to ({region.x + region.width}, {region.y + region.height})")
            roi = frame[region.y:region.y + region.height,
                       region.x:region.x + region.width]
            
            # Preprocess the ROI for better OCR
            logger.debug("Preprocessing ROI")
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Perform OCR
            logger.debug("Performing OCR")
            text = pytesseract.image_to_string(binary, config='--psm 6')
            text = text.strip()
            logger.info(f"Extracted text: {text}")
            logger.info("=== Text Extraction Complete ===")
            return text
        except Exception as e:
            logger.error(f"ERROR in OCR: {e}")
            return ""
    
    @timed
    @log_exceptions
    def create_mask(self, frame: np.ndarray, text_regions: List[TextRegion], 
                   padding: int = 5) -> np.ndarray:
        """Create a mask for text regions with padding."""
        logger.info("=== Creating Mask ===")
        if not self._validate_frame(frame):
            logger.error("Frame validation failed")
            return np.zeros((0, 0), dtype=np.uint8)
            
        try:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            logger.debug(f"Created mask with shape: {mask.shape}")
            
            for i, region in enumerate(text_regions):
                logger.debug(f"\nProcessing region {i+1}:")
                # Add padding to region
                x1 = max(0, region.x - padding)
                y1 = max(0, region.y - padding)
                x2 = min(frame.shape[1], region.x + region.width + padding)
                y2 = min(frame.shape[0], region.y + region.height + padding)
                logger.debug(f"Padded region: ({x1}, {y1}) to ({x2}, {y2})")
                
                # Fill region in mask
                mask[y1:y2, x1:x2] = 255
            
            logger.info("=== Mask Creation Complete ===")
            return mask
        except Exception as e:
            logger.error(f"Error creating mask: {str(e)}")
            return np.zeros(frame.shape[:2], dtype=np.uint8)
    
    @timed
    @log_exceptions
    def inpaint_text(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Remove text using inpainting."""
        logger.info("=== Inpainting Text ===")
        if not self._validate_frame(frame):
            logger.error("Frame validation failed")
            return frame
            
        try:
            # Apply inpainting
            logger.debug("Applying inpainting")
            inpainted = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
            
            # Apply bilateral filter to smooth edges
            logger.debug("Applying bilateral filter")
            inpainted = cv2.bilateralFilter(inpainted, 9, 75, 75)
            
            logger.info("=== Inpainting Complete ===")
            return inpainted
        except Exception as e:
            logger.error(f"Error in inpainting: {str(e)}")
            return frame
    
    def get_safe_positions(self, frame: np.ndarray, text_regions: List[TextRegion]) -> List[Tuple[int, int]]:
        """Generate safe positions for text relocation."""
        logger.info("\n=== Finding Safe Positions ===")
        if not self._validate_frame(frame):
            logger.error("Frame validation failed")
            return []
            
        try:
            safe_positions = []
            logger.debug(f"Frame dimensions: {frame.shape}")
            
            # Create a grid of potential positions
            grid_size = 10
            logger.debug(f"Using grid size: {grid_size}")
            
            for y in range(0, frame.shape[0], grid_size):
                for x in range(0, frame.shape[1], grid_size):
                    # Check if position overlaps with existing text
                    is_safe = True
                    for region in text_regions:
                        if (x >= region.x and x <= region.x + region.width and
                            y >= region.y and y <= region.y + region.height):
                            is_safe = False
                            break
                    
                    if is_safe:
                        safe_positions.append((x, y))
            
            # Return random safe positions
            if not safe_positions:
                logger.info("No safe positions found")
                return []
                
            num_positions = min(len(safe_positions), len(text_regions))
            logger.debug(f"Found {len(safe_positions)} potential positions")
            logger.debug(f"Selecting {num_positions} positions")
            selected_positions = random.sample(safe_positions, num_positions)
            logger.info("=== Safe Position Selection Complete ===")
            return selected_positions
        except Exception as e:
            logger.error(f"ERROR in get_safe_positions: {e}")
            return []
    
    @timed
    @log_exceptions
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[TextRegion]]:
        """Process a single frame: detect text, create mask, and inpaint."""
        logger.info("=== Processing Frame ===")
        if not self._validate_frame(frame):
            logger.error("Frame validation failed")
            return frame, []
            
        try:
            # Detect text regions
            logger.debug("Detecting text regions")
            text_regions = self.detect_text_regions(frame)
            
            if not text_regions:
                logger.info("No text regions found")
                return frame, []
                
            # Create mask for text regions
            logger.debug("Creating mask")
            mask = self.create_mask(frame, text_regions)
            
            # Remove text using inpainting
            logger.debug("Removing text")
            inpainted = self.inpaint_text(frame, mask)
            
            logger.info("=== Frame Processing Complete ===")
            return inpainted, text_regions
        except Exception as e:
            logger.error(f"Error in process_frame: {str(e)}")
            return frame, [] 