# 🌿 Bean Leaf Disease Detector

End-to-end deep learning project: fine-tuning a pretrained Vision Transformer to detect bean leaf diseases, hosted on Hugging Face and deployed with Streamlit.

## Problem
Bean crop diseases (angular leaf spot, bean rust) reduce yields. Early detection from a simple leaf photo helps farmers act quickly.

## Dataset
[AI-Lab-Makerere/beans](https://huggingface.co/datasets/AI-Lab-Makerere/beans): ~1,300 leaf images, 3 classes (healthy, angular_leaf_spot, bean_rust), pre-split into train/validation/test.

## Model
- Base: [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k)
- Fine-tuned model: https://huggingface.co/YOUR_HF_USERNAME/bean-leaf-vit

## Project structure
```
├── app.py            # Streamlit application
├── requirements.txt
└── README.md
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Live app
https://YOUR-APP-NAME.streamlit.app

## Results
Test accuracy: XX% (use the value from your Colab Step 7)
