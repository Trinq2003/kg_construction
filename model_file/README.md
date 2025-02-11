This directory contains used pretrain model files, categorized into `embedding` and `llm`.
----
### Command to download models:
1. `SciPhi/Triplex`
```bash
huggingface-cli download SciPhi/Triplex --local-dir model_file/llm/sciphi_triplex
```

2. `google/gemma-2-2b-jpn-it`
```bash
huggingface-cli download google/gemma-2-2b-jpn-it --local-dir model_file/llm/google_gemma2
```

3. `intfloat/multilingual-e5-large-instruct`
```bash
huggingface-cli download intfloat/multilingual-e5-large-instruct --local-dir model_file/embedding/intfloat_ml_e5
```