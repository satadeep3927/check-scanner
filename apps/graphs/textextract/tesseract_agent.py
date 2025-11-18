"""
TesseractAgent: Traditional OCR-based extraction using Tesseract (for comparison/demo).
"""

from pathlib import Path
from typing import Literal, TypedDict

import cv2
import numpy as np
import pytesseract
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from PIL import Image
from pydantic import BaseModel, Field

from apps.common.copilot import get_access_token_from_copilot
from apps.common.settings import get_settings


# Define the structured output schema for extracted data
class ExtractedField(BaseModel):
    """A single extracted field with label and value."""

    label: str = Field(
        description="Field label/name (e.g., 'payor', 'invoice_number', 'tax_amount')"
    )
    value: str = Field(description="Field value")


class ExtractedData(BaseModel):
    """Dynamically extracted information from the document."""

    extracted_fields: list[ExtractedField] = Field(
        description="List of all extracted fields with their labels and values"
    )

    items: list[dict[str, str]] | None = Field(
        default=None,
        description="List of itemized data if present (e.g., invoice line items with quantity, description, rate, total)",
    )


# Define the state for the agent graph
class AgentState(TypedDict):
    """State passed between nodes in the agent graph."""

    input_path: str
    input_type: Literal["pdf", "image"]
    raw_text: str
    structured_data: dict | None
    error: str | None


class TesseractAgent:
    """Traditional OCR-based agent using Tesseract (for demo/comparison)."""

    def __init__(self):
        """Initialize the TesseractAgent with LLM and graph."""
        settings = get_settings()

        # Initialize the LLM for structuring
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=get_access_token_from_copilot(),
            base_url=settings.llm_api_base_url,
            default_headers=settings.copilot_extra_headers,
        )

        # Create the LangGraph workflow
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph workflow for text extraction and structuring."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("extract_text", self._extract_text_node)
        workflow.add_node("structure_data", self._structure_data_node)

        # Define edges
        workflow.set_entry_point("extract_text")
        workflow.add_edge("extract_text", "structure_data")
        workflow.add_edge("structure_data", END)

        return workflow.compile()

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)

        return denoised

    def _extract_text_from_image(self, image_path: str) -> str:
        """Extract text from an image using Tesseract OCR."""
        try:
            # Load image
            image = cv2.imread(image_path)

            if image is None:
                raise Exception(f"Failed to load image: {image_path}")

            # Preprocess image
            processed = self._preprocess_image(image)

            # Configure Tesseract
            custom_config = r"--oem 3 --psm 6"

            # Extract text
            text = pytesseract.image_to_string(processed, config=custom_config)

            if not text.strip():
                # Try without preprocessing
                pil_image = Image.open(image_path)
                text = pytesseract.image_to_string(pil_image, config=custom_config)

            return text

        except Exception as e:
            raise Exception(
                f"Failed to extract text from image using Tesseract: {str(e)}"
            )

    def _extract_text_node(self, state: AgentState) -> AgentState:
        """Node to extract raw text from the input document using Tesseract."""
        try:
            input_path = state["input_path"]
            input_type = state["input_type"]

            if input_type == "image":
                raw_text = self._extract_text_from_image(input_path)
            else:
                raise ValueError(
                    f"Tesseract agent only supports images, got: {input_type}"
                )

            state["raw_text"] = raw_text
            state["error"] = None

        except Exception as e:
            state["raw_text"] = ""
            state["error"] = str(e)

        return state

    def _structure_data_node(self, state: AgentState) -> AgentState:
        """Node to structure the raw text using LLM."""
        try:
            if state["error"]:
                return state

            raw_text = state["raw_text"]

            if not raw_text.strip():
                state["error"] = "No text extracted from document"
                return state

            # Use LLM with structured output
            structured_llm = self.llm.with_structured_output(ExtractedData)

            prompt = f"""You are an expert at extracting ALL possible information from financial documents (checks, invoices, receipts, etc.).
Given the following raw text extracted via OCR, extract as much information as possible.

Raw Text:
{raw_text}

INSTRUCTIONS:
1. Extract ALL visible information you can find

2. Put all extracted data into the 'extracted_fields' list using these COMMON LABELS when applicable:
   - payor (who pays/wrote the check/billed to)
   - payee (who receives payment)
   - amount (total numerical amount)
   - amount_in_words (amount written in words)
   - date (transaction/document date)
   - due_date (payment due date if present)
   - notes_memo (any notes, memo, or project description)
   - bank_name (bank name if this is a check)
   - check_number (check number if applicable)
   - invoice_number (invoice number if applicable)
   - routing_number (bank routing number)
   - account_number (account number)

3. For ANY OTHER information you find, add it to 'extracted_fields' list with your own descriptive label 
   (e.g., "tax_amount", "subtotal", "payor_address", "payee_address", "phone_number", etc.)

4. IMPORTANT: If there's a TABLE or ITEMIZED LIST (like invoice line items), you MUST extract it into the 'items' field.
   The 'items' field should be a LIST OF DICTIONARIES where each dictionary represents one row/item.
   
   Example for invoice items:
   items = [
       {{"item_description": "Floor Tiles", "quantity": "3", "rate": "$52", "total": "$156"}},
       {{"item_description": "Glue", "quantity": "1", "rate": "$75", "total": "$75"}}
   ]
   
   Use clear keys like: item_description, item_name, quantity, qty, rate, price, unit_price, total, amount, etc.

5. Only include fields that have actual values. Skip fields with no data.

Extract EVERYTHING you can see!
"""

            extracted_data = structured_llm.invoke(prompt)

            if isinstance(extracted_data, ExtractedData):
                data_dict = extracted_data.model_dump()
                # Convert list of ExtractedField objects to dictionary
                result = {
                    field["label"]: field["value"]
                    for field in data_dict.get("extracted_fields", [])
                }
                if data_dict.get("items"):
                    result["items"] = data_dict["items"]
                state["structured_data"] = result
            else:
                state["structured_data"] = extracted_data
            state["error"] = None

        except Exception as e:
            state["error"] = f"Failed to structure data: {str(e)}"
            state["structured_data"] = None

        return state

    def process(
        self, input_path: str, input_type: Literal["pdf", "image"] | None = None
    ) -> dict:
        """
        Process an image file to extract structured data using Tesseract OCR.

        Args:
            input_path: Path to the image file
            input_type: Type of input (only "image" supported). If None, will be inferred.

        Returns:
            Dictionary containing the structured data or error information
        """
        # Infer input type if not provided
        if input_type is None:
            path = Path(input_path)
            ext = path.suffix.lower()
            if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]:
                input_type = "image"
            else:
                return {"error": f"Tesseract agent only supports images. Got: {ext}"}

        # Initialize state
        initial_state = AgentState(
            input_path=input_path,
            input_type=input_type,
            raw_text="",
            structured_data=None,
            error=None,
        )

        # Run the graph
        result = self.graph.invoke(initial_state)

        # Return structured result
        if result["error"]:
            return {"error": result["error"]}

        if result["structured_data"]:
            return {
                "success": True,
                "data": result["structured_data"],
                "raw_text": result["raw_text"],
            }

        return {"error": "Failed to process document"}
