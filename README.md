# semicon-wafer-inspect
# WaferVision: Semiconductor Wafer Defect Detection

## 📌 Project Overview
WaferVision is an advanced machine learning-powered semiconductor wafer defect detection system. By leveraging computer vision and predictive analytics, this project enhances quality control in semiconductor manufacturing, reducing manual inspection efforts and improving defect identification.

## 🔑 Key Highlights
✅ **XGBoost Classifier** trained with **92% accuracy**, ensuring precise wafer defect prediction and optimized classification performance.
✅ **Image processing techniques (scikit-image)** for feature extraction and transformation:
   - **Radon Transform** for defect pattern analysis (`skimage.transform.radon`)
   - **Probabilistic Hough Transform** for line detection in wafer images (`skimage.transform.probabilistic_hough_line`)
   - **Region-based analysis** for defect segmentation (`skimage.measure`)
✅ **Statistical analysis (scipy.stats)** and **interpolation techniques (scipy.interpolate)** to refine defect detection and improve model performance.
✅ **Streamlit Web Application** for an intuitive and interactive user experience, allowing real-time predictions.
✅ **Automated Data Preprocessing Pipelines** to enhance model accuracy, scalability, and efficiency.
✅ **Scalable Architecture** designed for industrial applications.

## 🔧 Tech Stack
- **Machine Learning**: XGBoost (92% accuracy), Scikit-learn
- **Computer Vision**: Scikit-image (radon, probabilistic_hough_line, measure), OpenCV
- **Statistical & Mathematical Tools**: Scipy (interpolate, stats)
- **Web Development**: Streamlit
- **Data Processing**: Pandas, NumPy

## 📂 Project Structure
```
semicon-wafer-inspect/
│── deployy.py                  # Deployment script
│── requirements.txt            # Dependencies
│── saved_model.json            # Model metadata
│── saved_model_xgb.sav         # Trained XGBoost model
│── savemodle3.sav              # Additional saved model
│── wm-811k-wafermap.ipynb      # Notebook for dataset processing & analysis
│── xgb_model.json              # Trained XGBoost model configuration
│── README.md                   # Project documentation
```

## 🚀 Installation & Usage

### Step 1: Clone the Repository
```bash
git clone https://github.com/<your-username>/semicon-wafer-inspect.git
cd semicon-wafer-inspect
```
Step 2: (Optional) Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

Step 3: Install Dependencies
Ensure you have Python installed (>=3.8), then run:

```bash

pip install -r requirements.txt
```
Running the Model
To deploy the wafer defect detection system:

```bash
 
python deployy.py
```
To explore the dataset and analysis:

```bash
 
jupyter notebook wm-811k-wafermap.ipynb
```
## 🛠 Model Training & Experiment Tracking
- The XGBoost model was trained on the **WM-811K wafer dataset**, achieving 92% accuracy.
- Preprocessing includes **Radon Transform**, **Hough Transform**, and **region-based segmentation** for feature extraction.
- Statistical techniques were applied to refine classification performance.

## 📊 Sample Data for Testing

A sample wafer dataset is provided in sample_data.txt(Edge-Ring) for testing the model, or you can get it from WM-811k Wafermap dataset.
![1738683624411](https://github.com/user-attachments/assets/b05f328a-3272-4519-a234-00dc17c35c60)

![1738682139616](https://github.com/user-attachments/assets/026b66b7-e3d9-4d1c-b672-b0e8830c67ae)

![image](https://github.com/user-attachments/assets/8354dfb9-747c-4f67-b6f3-e44ee415a81f)




## 🤝 Collaboration
Open to discussions, collaborations, and improvements. Let’s innovate together!

📧 **Contact**: Reach out for contributions, feedback, or collaboration opportunities.

