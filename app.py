"""
Streamlit app for Check Scanner using TextExtractAgent.
"""

import json
import tempfile
from pathlib import Path

import streamlit as st

from apps.graphs.textextract.agent import TextExtractAgent
from apps.graphs.textextract.tesseract_agent import TesseractAgent

# Page configuration
st.set_page_config(
    page_title="Check Scanner",
    page_icon="📄",
    layout="wide",
)


def main():
    """Main Streamlit application."""

    st.title("Check Scanner")
    st.markdown(
        """
        Upload a scanned check or invoice image to extract structured information using AI.
        Compare different extraction approaches.
        """
    )

    # Approach selector
    col_selector1, col_selector2 = st.columns([2, 3])

    with col_selector1:
        approach = st.selectbox(
            "Select Extraction Approach",
            options=["Vision LLM (GPT-4.1)", "Tesseract OCR + LLM"],
            help="Choose between Vision LLM (recommended, 85-90% accuracy) or Tesseract OCR (0-6% accuracy, for demo only)",
        )

    with col_selector2:
        if approach == "Tesseract OCR + LLM":
            st.warning(
                "⚠️ Tesseract approach has very low accuracy (0-6%) and is shown for comparison only."
            )
        else:
            st.info(
                "✓ Vision LLM provides best accuracy (85-90%) for financial documents."
            )

    # Initialize agent based on selected approach
    if approach == "Vision LLM (GPT-4.1)":
        agent = TextExtractAgent()
        approach_name = "Vision LLM"
    else:
        agent = TesseractAgent()
        approach_name = "Tesseract OCR"

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Check or Invoice Image",
        type=["jpg", "jpeg", "png"],
        help="Upload a scanned check or invoice image (JPEG or PNG)",
    )

    if uploaded_file is not None:
        # Create two columns for layout
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Uploaded Check")

            # Display the uploaded file
            file_type = uploaded_file.type
            if file_type.startswith("image"):
                st.image(uploaded_file, width="stretch")
            else:
                st.info("PDF uploaded. Processing...")

        with col2:
            st.subheader("Extracted Information")
            st.caption(f"Using: {approach_name}")

            # Process button
            if st.button("Extract Check Data", type="primary", width="stretch"):
                with st.spinner(
                    f"Processing with {approach_name}... This may take a moment."
                ):
                    try:
                        # Save uploaded file to temporary location
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=Path(uploaded_file.name).suffix
                        ) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name

                        # Extract check data
                        result = agent.process(tmp_path)

                        # Clean up temp file
                        Path(tmp_path).unlink()

                        # Display results
                        if result.get("success"):
                            data = result["data"]

                            st.success("Data extracted successfully!")

                            # Display structured data in a nice format
                            st.markdown("### Extracted Information")

                            # Separate items from other fields
                            items = data.pop("items", None)

                            # Skip fields that shouldn't be displayed
                            skip_fields = ["visible_text_other"]

                            # Display all fields uniformly
                            st.markdown("### Extracted Fields")

                            for key, value in data.items():
                                if (
                                    key.lower() not in skip_fields
                                    and value
                                    and value != "Not Found"
                                ):
                                    label = key.replace("_", " ").title()
                                    st.text_input(
                                        label, value, disabled=True, key=f"field_{key}"
                                    )

                            # Display items table if present
                            if items and isinstance(items, list) and len(items) > 0:
                                st.markdown("### Itemized List")
                                st.dataframe(items, width="stretch")

                            # Show raw OCR text in expander
                            with st.expander("View Raw OCR Text"):
                                st.text_area(
                                    "Raw Text",
                                    result.get("raw_text", ""),
                                    height=200,
                                    disabled=True,
                                )

                            # Download JSON
                            st.download_button(
                                label="Download as JSON",
                                data=json.dumps(data, indent=2),
                                file_name="check_data.json",
                                mime="application/json",
                                width="stretch",
                            )

                        else:
                            st.error(f"Error: {result.get('error', 'Unknown error')}")

                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.markdown(
            """
            This app extracts information from scanned checks and invoices.
            
            ### Extraction Approaches
            
            **1. Vision LLM (Recommended)**
            - GPT-4.1 with vision capabilities
            - Accuracy: 85-90%
            - Direct image understanding
            - Best for all document types
            
            **2. Tesseract OCR (Demo Only)**
            - Traditional OCR approach
            - Accuracy: 0-6%
            - Poor with handwritten text
            - Shown for comparison
            
            ### Dynamic Extraction
            The app intelligently extracts ALL visible information including:
            - Common fields: Payor, Payee, Amount, Date, Check/Invoice Numbers
            - Bank details: Bank Name, Routing Number, Account Number
            - Additional data: Tax, Subtotal, Addresses, Phone Numbers
            - Itemized lists: Line items with quantities, rates, totals
            
            The AI creates appropriate labels for any data it finds!
            
            ### Tips for Best Results
            - Use high-quality scans (300 DPI or higher)
            - Ensure good lighting and contrast
            - Avoid blurry or skewed images
            """
        )

        st.header("Comparison")
        st.markdown(
            """
            | Approach | Accuracy | Cost |
            |----------|----------|------|
            | Vision LLM | 85-90% | $10/mo |
            | Tesseract | 0-6% | $0 |
            
            See RND_REPORT.md for detailed analysis.
            """
        )


if __name__ == "__main__":
    main()
