
## Project overview and documentation.

 **Resume Screening using DistilBERT.ipynb - Colab.pdf**  
  Exported PDF of the Google Colab notebook showing data preprocessing, model training, and evaluation steps.

 **Resume_Screening_using_DistilBERT.ipynb**  
  Main Jupyter notebook containing the complete implementation, including:
  - Data loading and cleaning  
  - Resume text preprocessing  
  - DistilBERT fine-tuning  
  - Model evaluation  


**app.py**

This is the main **Streamlit application file** for the Resume Screening system.

### What it does
- Loads the trained DistilBERT model and tokenizer  
- Accepts resume uploads in PDF, DOCX, and TXT formats  
- Predicts job category from resume text  
- Detects experience level from content  
- Performs skill matching using a predefined skills dataset  
- Displays a resume match score using a donut chart  
- Shows extracted and missing skills in the UI
---

## Model Used

- DistilBERT from Hugging Face Transformers  
- Fine-tuned for multi-class resume classification  
