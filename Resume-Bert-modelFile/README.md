## Model Files (`resume_bert_model/`)

This folder contains the **fine-tuned DistilBERT model** used for resume classification.  

- `model.safetensors` → The trained model weights.  
- `config.json` → Model configuration (architecture, hidden sizes, etc.).  
- `tokenizer_config.json` → Configuration for the tokenizer.  
- `tokenizer.json` → Tokenizer vocabulary and rules in JSON format.  
- `vocab.txt` → Vocabulary file mapping tokens to IDs.  
- `special_tokens_map.json` → Defines special tokens (e.g., [CLS], [SEP], [PAD]).  
- `training_args.bin` → Saved training arguments used during fine-tuning.  

These files are required to **load the model and tokenizer** for inference in the app.

`label_encoder.pkl` → Stores the **LabelEncoder object** used to convert category names (like "Data Science") into numeric labels for training and back during prediction.  
  This ensures that the model’s numeric predictions can be translated into readable job categories.
