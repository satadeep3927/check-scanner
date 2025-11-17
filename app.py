"""
Streamlit app for Check Scanner using TextExtractAgent.
"""

import streamlit as st
from pathlib import Path
import tempfile
import json

from apps.graphs.textextract.agent import TextExtractAgent


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
        The app uses Vision AI to read the document and extract structured data.
        """
    )
    
    # Initialize agent
    agent = TextExtractAgent()
    
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
            
            # Process button
            if st.button("Extract Check Data", type="primary", width="stretch"):
                with st.spinner("Processing check... This may take a moment."):
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
                                if key.lower() not in skip_fields and value and value != "Not Found":
                                    label = key.replace("_", " ").title()
                                    st.text_input(label, value, disabled=True, key=f"field_{key}")
                            
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
            This app extracts information from scanned checks using:
            
            - **Vision AI**: For reading text from check images
            - **AI Models**: For extracting and structuring the data
            
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
        
        st.header("Settings")
        
        # Optional: Add settings for OCR or processing
        with st.expander("Advanced Settings"):
            st.info("Advanced OCR settings coming soon!")


if __name__ == "__main__":
    main()
