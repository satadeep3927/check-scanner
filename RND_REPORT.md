 <img src="https://lh3.googleusercontent.com/d/1jj0iO4jG9sXnTpUvsasR1d8OHPLSuaim" alt="Logo" width="200" style="margin-right: 20px; margin-bottom: 20px; display: block;"/>

# Research & Development Report: Check/Invoice Data Extraction

**Project:** Check Scanner - AI-powered Document Data Extraction  
**Date:** November 17, 2025  
**Author:** Research Team  
**Repository:** satadeep3927/check-scanner

<br clear="left"/>

---

## Executive Summary

This document presents a comprehensive comparison of four different approaches for extracting structured data from scanned checks and invoices. Our research evaluated accuracy, cost, and implementation complexity across OCR-based, cloud AI, and local AI solutions.

**Key Finding:** Cloud-based Vision LLMs (Google Gemini 2.5 Flash and GPT-4.1 via Copilot) significantly outperform traditional OCR approaches, achieving 85-90%+ accuracy for financial document extraction.

---

## 1. Tesseract OCR with NLP-based Extractor

### Overview
Traditional approach using Tesseract OCR for text extraction followed by NLP/LLM for structuring the data.

### Implementation Details
- **OCR Engine:** Tesseract 5.x
- **Preprocessing:** Image enhancement, grayscale conversion, adaptive thresholding
- **Text Extraction:** Direct OCR output
- **Structuring:** LLM post-processing of extracted text

### Results
- **Accuracy:** 0-6%
- **Cost:** $0 (Open source)

### Key Issues Identified
- Failed to detect large, clear logos on checks
- Poor performance on handwritten text
- Unreliable with scanned/low-quality images
- Required extensive preprocessing with minimal improvement
- Inconsistent text extraction even with optimal configurations

### Verdict
**Not Recommended** - Extremely low accuracy makes this approach unsuitable for production use in financial document processing.

---

## 2. Google Gemini 2.5 Flash (Vision LLM)

### Overview
Cloud-based vision language model with native image understanding capabilities.

### Implementation Details
- **Model:** Gemini 2.5 Flash
- **Context Window:** 1,048,576 tokens
- **Approach:** Direct image-to-structured-data extraction
- **Release Date:** June 17, 2025

### Results
- **Accuracy:** >90%
- **Pricing Structure:**
  - Input tokens: $0.30 per 1M tokens
  - Output tokens: $2.50 per 1M tokens
  - Image inputs: $1.238 per 1K images

### Cost Analysis for 100 Invoices/Month

#### Per-Invoice Breakdown
- **Input:** 1 image per invoice
- **Estimated tokens per invoice:**
  - Image encoding: ~1,000 tokens equivalent
  - Prompt: ~500 tokens
  - Total input: ~1,500 tokens
- **Estimated output:** ~500 tokens (structured JSON)

#### Monthly Cost Calculation
```
Image cost:     100 images × ($1.238 / 1000) = $0.124
Input tokens:   100 × 1,500 × ($0.30 / 1M) = $0.045
Output tokens:  100 × 500 × ($2.50 / 1M) = $0.125

Total monthly cost: $0.294 (~$0.30)
Annual cost: $3.53
```

#### Scaling Estimates
| Volume | Monthly Cost | Annual Cost |
|--------|--------------|-------------|
| 100 invoices | $0.30 | $3.53 |
| 500 invoices | $1.49 | $17.63 |
| 1,000 invoices | $2.94 | $35.26 |
| 5,000 invoices | $14.69 | $176.28 |
| 10,000 invoices | $29.40 | $352.56 |

### Advantages
- Highest accuracy (>90%)  
- Extremely cost-effective at scale  
- Large context window for complex documents  
- Fast inference times  
- No infrastructure management required  

### Disadvantages
- Requires internet connectivity  
- Data sent to external service (privacy considerations)  
- Potential API rate limits  

### Verdict
**Highly Recommended** - Best balance of accuracy and cost for production deployments. Ideal for most use cases.

---

## 3. GPT-4.1 via GitHub Copilot

### Overview
Access to GPT-4.1 Vision capabilities through GitHub Copilot subscription.

### Implementation Details
- **Model:** GPT-4.1 (Vision-capable)
- **Access:** GitHub Copilot API
- **Authentication:** Token-based via Copilot subscription
- **Approach:** Direct image analysis with structured output

### Results
- **Accuracy:** 85-90%
- **Cost:** $10/month (fixed via Copilot subscription)

### Cost Analysis
```
Fixed monthly cost: $10.00
Unlimited requests within rate limits
Effective cost per invoice: $10 / N invoices
```

#### Volume-Based Effective Cost
| Volume/Month | Cost per Invoice |
|--------------|------------------|
| 100 | $0.100 |
| 500 | $0.020 |
| 1,000 | $0.010 |
| 5,000 | $0.002 |
| 10,000 | $0.001 |

### Advantages
- Fixed predictable pricing  
- Cost-effective at high volumes  
- Strong accuracy (85-90%)  
- Easy integration with development workflow  
- No per-request billing concerns  

### Disadvantages
- Potential rate limits (not publicly documented)  
- Requires GitHub Copilot subscription  
- Slightly lower accuracy than Gemini 2.5 Flash  
- Rate limit behavior under high load unclear  

### Verdict
**Recommended for High Volume** - Excellent choice for applications processing >500 invoices/month where the fixed cost becomes highly economical.

---

## 4. Local Open-Source Models (Gemma3/LLaVA + Qwen3)

### Overview
Fully local solution using open-source vision and language models.

### Implementation Details
- **Vision Model:** Gemma3 or LLaVA for text extraction
- **Language Model:** Qwen3 for text structuring/formatting
- **Deployment:** Local GPU-based inference
- **Framework:** Ollama/vLLM for model serving

### Results
- **Accuracy:** 75-80%
- **Cost:** $0 (after initial hardware investment)

### Infrastructure Requirements
- **GPU:** NVIDIA GPU with 12-16GB VRAM minimum
  - Recommended: RTX 4090 (24GB), A5000 (24GB), or better
  - Budget option: RTX 3060 (12GB) with performance trade-offs
- **RAM:** 32GB+ system RAM
- **Storage:** 50-100GB for model weights
- **Software:** CUDA drivers, Docker, model serving framework

### Hardware Cost Estimates
| Configuration | Initial Cost | Performance |
|---------------|--------------|-------------|
| RTX 3060 (12GB) | $300-400 | 2-4 seconds/invoice |
| RTX 4070 Ti (16GB) | $700-800 | 1-2 seconds/invoice |
| RTX 4090 (24GB) | $1,600-2,000 | <1 second/invoice |
| Professional (A5000) | $2,500-3,000 | <1 second/invoice |

### Total Cost of Ownership (3 Years)
```
Hardware: $1,600 (RTX 4090)
Electricity: ~$100/year × 3 = $300
Maintenance/Upgrades: $200
Total: ~$2,100

Break-even vs Gemini (10k/month): ~6 months
Break-even vs GPT Copilot: 17.5 years (fixed $10/month)
```

### Advantages
- Zero recurring API costs  
- Complete data privacy (no external data transmission)  
- No rate limits  
- Full control over infrastructure  
- No internet dependency for inference  
- Potential for model fine-tuning  

### Disadvantages
- Lower accuracy (75-80%) vs cloud solutions  
- High upfront hardware investment  
- Requires GPU infrastructure  
- Ongoing maintenance and monitoring  
- Slower inference without high-end GPU  
- Model management and updates needed  

### Verdict
**Recommended for Specific Use Cases:**
- High-security/compliance requirements (banking, healthcare)
- Very high volume (>50,000 invoices/month)
- Air-gapped or offline environments
- Organizations with existing GPU infrastructure

---

## Comparative Analysis

### Accuracy Comparison
```
Gemini 2.5 Flash:   ████████████████████ 90%+
GPT-4.1 Copilot:    ██████████████████   85-90%
Local Models:       ███████████████      75-80%
Tesseract + NLP:    █                    0-6%
```

### Cost Comparison (1,000 invoices/month)
```
Gemini 2.5 Flash:   $2.94/month   ($35.26/year)
GPT-4.1 Copilot:    $10/month     ($120/year)
Local Models:       $0/month*     ($2,100 initial + electricity)
Tesseract + NLP:    $0/month      ($0 total)

*After hardware investment
```

### Implementation Complexity
```
Tesseract + NLP:    Medium   (OCR setup + preprocessing)
Gemini 2.5 Flash:   Low      (API integration only)
GPT-4.1 Copilot:    Low      (API integration only)
Local Models:       High     (GPU setup, model deployment, monitoring)
```

---

## Recommendations by Use Case

### 1. Small Business / Startup (< 500 invoices/month)
**Recommended:** Google Gemini 2.5 Flash
- Lowest total cost at this scale (~$1.50/month)
- Highest accuracy (>90%)
- Minimal implementation effort
- Pay-as-you-grow model

### 2. Medium Business (500-5,000 invoices/month)
**Recommended:** GPT-4.1 via GitHub Copilot
- Fixed $10/month cost becomes economical
- Strong accuracy (85-90%)
- Predictable billing
- Break-even point: ~500 invoices/month vs Gemini

### 3. Enterprise / High Volume (> 5,000 invoices/month)
**Option A:** GPT-4.1 via Copilot (if rate limits acceptable)
- Most cost-effective for extremely high volumes
- Consider rate limit testing

**Option B:** Local Models (if privacy/compliance critical)
- Upfront investment justified at this scale
- Complete data control
- Unlimited processing capacity

### 4. Regulated Industries (Banking, Healthcare, Government)
**Recommended:** Local Open-Source Models
- Data never leaves premises
- Full audit trail and control
- Compliance with data sovereignty requirements
- 75-80% accuracy may be acceptable with human review workflow

---

## Technical Implementation Notes

### Current Implementation
The check-scanner project currently uses **GPT-4.1 via GitHub Copilot** with the following architecture:

```
1. Image Upload → Streamlit UI
2. Image Processing → Compression to <400KB (API requirement)
3. Vision LLM → Direct image-to-structured data extraction
4. Dynamic Schema → LLM generates field labels
5. Display → Uniform text input fields + items table
```

### Key Features
- Dynamic field extraction (no fixed schema)
- Itemized list support (invoice line items)
- Support for JPEG/PNG images
- Image compression for API compliance
- Structured output using Pydantic models

### Migration Considerations
To switch between approaches:

**To Gemini 2.5 Flash:**
- Update LLM provider in `settings.py`
- Adjust authentication to Google Cloud credentials
- Minimal code changes required

**To Local Models:**
- Set up GPU infrastructure
- Deploy Ollama/vLLM with Gemma3/LLaVA + Qwen3
- Update agent to use local endpoints
- Adjust timeout configurations for slower inference

---

## Conclusion

For the check-scanner project, **Google Gemini 2.5 Flash** offers the best overall solution for most deployment scenarios, providing:
- Highest accuracy (>90%)
- Extremely low cost (<$3/month for 1,000 invoices)
- Minimal infrastructure requirements
- Fast development and deployment

**GPT-4.1 via Copilot** remains an excellent choice for:
- Developers with existing Copilot subscriptions
- High-volume applications (>500/month)
- Predictable monthly budgeting

**Local models** should be reserved for:
- Strict data privacy requirements
- Air-gapped environments
- Very high volumes (>50k/month) where hardware investment is justified

**Tesseract OCR** is not viable for this use case and should be avoided.

---

## Appendix: Testing Methodology

### Test Dataset
- 50 scanned checks (mix of handwritten and printed)
- 50 invoices (various formats and layouts)
- Image quality: Mix of high-quality scans and mobile photos
- Document types: Bank checks, contractor invoices, retail receipts

### Accuracy Measurement
Accuracy calculated as:
```
Accuracy = (Correctly Extracted Fields / Total Fields) × 100
```

Fields considered:
- Payor/Payer name
- Payee name
- Amount (numerical)
- Date
- Check/Invoice number
- Bank name (checks only)
- Line items (invoices only)
- Additional fields (addresses, tax, subtotals)

### Performance Criteria
- **Correct:** Exact match or semantically equivalent
- **Partial:** Minor formatting differences but data preserved
- **Incorrect:** Wrong data or missing critical information

---

**Document Version:** 1.0  
**Last Updated:** November 17, 2025  
**Next Review:** January 2026
