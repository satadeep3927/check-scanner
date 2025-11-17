"""
TextExtractAgent: AI Agent for extracting and structuring check data from PDFs and images.
"""

import base64
import io
from pathlib import Path
from typing import Literal, TypedDict

import fitz  # PyMuPDF
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from PIL import Image
from pydantic import BaseModel, Field

from apps.common.copilot import get_access_token_from_copilot
from apps.common.settings import get_settings


# Define the structured output schema for extracted data
class ExtractedField(BaseModel):
    """A single extracted field with label and value."""
    label: str = Field(description="Field label/name (e.g., 'payor', 'invoice_number', 'tax_amount')")
    value: str = Field(description="Field value")

class ExtractedData(BaseModel):
    """Dynamically extracted information from the document."""
    
    # Common fields with suggested labels
    extracted_fields: list[ExtractedField] = Field(
        description="List of all extracted fields with their labels and values"
    )
    
    # Additional structured data like item tables
    items: list[dict[str, str]] | None = Field(
        default=None,
        description="List of itemized data if present (e.g., invoice line items with quantity, description, rate, total)"
    )


# Define the state for the agent graph
class AgentState(TypedDict):
    """State passed between nodes in the agent graph."""

    input_path: str
    input_type: Literal["pdf", "image"]
    raw_text: str
    structured_data: dict | None
    error: str | None


class TextExtractAgent:
    """AI Agent for extracting and structuring check data from PDFs and images."""

    def __init__(self):
        """Initialize the TextExtractAgent with LLM and graph."""
        settings = get_settings()

        # Initialize the LLM with vision capabilities
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

        # Create the graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("extract_text", self._extract_text_node)
        workflow.add_node("structure_data", self._structure_data_node)

        # Define edges
        workflow.set_entry_point("extract_text")
        workflow.add_edge("extract_text", "structure_data")
        workflow.add_edge("structure_data", END)

        return workflow.compile()

    def _encode_image_to_base64(self, image: Image.Image, max_size_kb: int = 400) -> str:
        """Encode PIL Image to base64 string for vision API, ensuring size is under max_size_kb."""
        # Start with high quality
        quality = 95
        format_type = "JPEG"
        
        while quality > 10:
            buffer = io.BytesIO()
            # Convert to RGB if needed for JPEG
            if image.mode in ("RGBA", "P"):
                rgb_image = image.convert("RGB")
            else:
                rgb_image = image
            
            rgb_image.save(buffer, format=format_type, quality=quality, optimize=True)
            size_kb = len(buffer.getvalue()) / 1024
            
            if size_kb <= max_size_kb:
                return base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            # Reduce quality for next iteration
            quality -= 10
        
        # If still too large, resize the image
        buffer = io.BytesIO()
        scale = (max_size_kb * 1024 / len(buffer.getvalue())) ** 0.5
        new_size = (int(image.width * scale * 0.8), int(image.height * scale * 0.8))
        resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        if resized_image.mode in ("RGBA", "P"):
            resized_image = resized_image.convert("RGB")
        
        resized_image.save(buffer, format=format_type, quality=85, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _extract_text_from_image(self, image_path: str) -> str:
        """Extract text from an image using vision LLM."""
        try:
            image = Image.open(image_path)
            
            # Convert image to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Encode image to base64
            image_b64 = self._encode_image_to_base64(image)
            
            # Use vision LLM to extract text
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": """Extract all visible text from this check image. 
                        Include everything you can see: bank name, check number, date, payor name, 
                        payee name, amount (numerical), amount in words, memo/notes, routing numbers, 
                        account numbers, and any other text visible on the check.
                        
                        Provide the text in a structured format, clearly labeling each field. Don't include anything extra."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                    }
                ]
            )
            
            response = self.llm.invoke([message])
            return str(response.content)
            
        except Exception as e:
            raise Exception(f"Failed to extract text from image: {str(e)}")

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a scanned PDF using vision LLM."""
        try:
            doc = fitz.open(pdf_path)
            text = ""

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Convert page to high-resolution image
                zoom = 2  # 2x zoom for good quality
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)

                # Convert to PIL Image
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

                # Encode image to base64

                # Use vision LLM to extract text
                message = HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": """Extract all visible text from this check image. 
                            Include everything you can see: bank name, check number, date, payor name, 
                            payee name, amount (numerical), amount in words, memo/notes, routing numbers, 
                            account numbers, and any other text visible on the check.
                            
                            Provide the text in a structured format, clearly labeling each field."""
                        },
                        {
                            "type": "image_url",
                    }
                ]
            )
            
            response = self.llm.invoke([message])
            text += str(response.content) + "\n"

            doc.close()
            return text
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")

    def _extract_text_node(self, state: AgentState) -> AgentState:
        """Node to extract raw text from the input document."""
        try:
            input_path = state["input_path"]
            input_type = state["input_type"]

            if input_type == "pdf":
                raw_text = self._extract_text_from_pdf(input_path)
            elif input_type == "image":
                raw_text = self._extract_text_from_image(input_path)
            else:
                raise ValueError(f"Unsupported input type: {input_type}")

            state["raw_text"] = raw_text
            state["error"] = None

        except Exception as e:
            state["raw_text"] = ""
            state["error"] = str(e)

        return state

    def _structure_data_node(self, state: AgentState) -> AgentState:
        """Node to structure the raw text into CheckData using LLM."""
        try:
            if state["error"]:
                # If there was an error in text extraction, skip structuring
                return state

            raw_text = state["raw_text"]

            if not raw_text.strip():
                state["error"] = "No text extracted from document"
                return state

            # Use LLM with structured output to extract all information dynamically
            structured_llm = self.llm.with_structured_output(ExtractedData)

            prompt = f"""You are an expert at extracting ALL possible information from financial documents (checks, invoices, receipts, etc.).
Given the following raw text, extract as much information as possible.

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
            # Convert ExtractedData Pydantic model to dict for TypedDict compatibility
            if isinstance(extracted_data, ExtractedData):
                data_dict = extracted_data.model_dump()
                # Convert list of ExtractedField objects to dictionary
                result = {field["label"]: field["value"] for field in data_dict.get("extracted_fields", [])}
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
        Process a PDF or image file to extract structured check data.

        Args:
            input_path: Path to the PDF or image file
            input_type: Type of input ("pdf" or "image"). If None, will be inferred from file extension.

        Returns:
            Dictionary containing the structured check data or error information
        """
        # Infer input type if not provided
        if input_type is None:
            path = Path(input_path)
            ext = path.suffix.lower()
            if ext == ".pdf":
                input_type = "pdf"
            elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]:
                input_type = "image"
            else:
                return {"error": f"Unsupported file type: {ext}"}

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
