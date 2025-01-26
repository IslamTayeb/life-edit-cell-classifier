# Life Edit Cell Classifier
This project, developed by the Duke Impact Investing Group (DIIG) Data Division for Life Edit, is a machine learning classifier designed to distinguish between edited and unedited single cells using DNA expression data.  The classifier leverages natural language processing (NLP) techniques to analyze gene descriptions and then employs machine learning models to perform the classification task. A Streamlit dashboard provides an interactive interface for data exploration and model visualization.

## Features
* **Gene Description Processing:**  Utilizes NLP techniques (TF-IDF vectorization, Sentence Transformers) for feature extraction from gene descriptions.
* **Unsupervised Clustering:**  Employs K-Means clustering to group genes based on their descriptions.
* **Supervised Classification:**  Trains a Random Forest classifier (with optional hyperparameter tuning using RandomizedSearchCV) to predict editing mechanisms based on gene expression data.
* **Data Filtering:**  Applies filtering techniques to select genes exhibiting significant differences in expression between edited and unedited cells.
* **Elbow Method for Optimal Clusters:** Uses the elbow method to determine the optimal number of clusters for K-Means clustering.
* **NCBI Data Parsing:** Integrates with the NCBI Entrez API to retrieve gene information for improved annotations.
* **Streamlit Dashboard:** Offers an interactive interface to explore data, visualize results, and interact with the model.
* **Forest Classifier:** The resulting classifier achieves accuracy scores ranging from 0.8 to 1.0 using 5-fold stratified k-fold cross-validation, depending on the selected filter criteria and classifier used (Random Forest, LightGBM, and KNN are evaluated).
* **Model Persistence:**  Saves and loads the trained classifier model.
* **Visualization:**  Provides various plots and visualizations using Matplotlib, Seaborn, and Plotly.

## Usage
1. **Data Preparation:** Prepare your single-cell DNA expression data in a TSV file.  Ensure your data includes gene IDs and expression values for each cell.
2. **Gene Annotation:**  Include gene annotation data, preferably with descriptions.
3. **Run the Notebooks:** Execute the Jupyter Notebooks (`code/Elbow_mz/elbowClassifier/*`, `code/NLP_mz/*`, `code/analysis + prediction/*`, `code/interpretability/*`) sequentially to process data, perform analyses, train the model, and generate visualizations.
4. **Streamlit App:** Run the Streamlit app (`code/interpretability/streamlit_app_imt.py`) for interactive exploration of the data and model.

## Installation
This project requires Python 3.  Install the necessary packages using pip:

```bash
pip install -r requirements.txt
```

A `requirements.txt` file should be created listing all dependencies.  If not already present, create it using:

```bash
pip freeze > requirements.txt
```

## Technologies Used
* **Python:** The primary programming language for the entire project.
* **Pandas:** Used for data manipulation and analysis.
* **Scikit-learn:**  Provides machine learning algorithms (StandardScaler, KMeans, RandomForestClassifier, RandomizedSearchCV).
* **Sentence Transformers:**  Generates embeddings from text descriptions.
* **Matplotlib & Seaborn:**  Used for creating static visualizations.
* **Plotly:** Creates interactive plots used in the Streamlit dashboard.
* **Biopython:** Used for interaction with NCBI Entrez API.
* **LightGBM:**  Gradient Boosting Classifier algorithm (Optional).
* **Streamlit:**  Creates an interactive web application.
* **NLTK:** Used for Natural Language Processing tasks.
* **Joblib:** Used for saving and loading the trained model.
* **Ace-Tools:** Used to display the dataframe in the notebook.

## Statistical Analysis
* **StandardScaler:** Used for feature scaling of gene expression data.
* **K-Means Clustering:**  Applied for unsupervised clustering of genes based on processed descriptions.
* **Random Forest Classifier:** Used for classification of cells as edited or unedited.
* **RandomizedSearchCV:** Used for hyperparameter tuning of the Random Forest Classifier.
* **T-tests:** Used to compare gene expression between edited and unedited groups.
* **PCA (Principal Component Analysis):** Dimensionality reduction for visualization and feature extraction.
* **Elbow Method:** Used to determine the optimal number of clusters for KMeans.
* **TF-IDF Vectorization:** Used to convert gene descriptions into numerical representations.

## Predictive vs. Interpretable Models
This project pursued two primary objectives: developing a highly predictive model for cell classification and creating a separate, readily interpretable model.  The Random Forest classifier, along with LightGBM and KNN (evaluated but not the primary focus), represents the predictive aspect. This hands-off approach prioritizes classification accuracy, leveraging the power of ensemble methods like Random Forest to achieve high performance (accuracy scores ranging from 0.8 to 1.0 using 5-fold stratified k-fold cross-validation).  The interpretability goal, conversely, is served by several components.  The Streamlit dashboard, visualizations generated using Matplotlib, Seaborn, and Plotly, and the K-Means clustering visualizations, allow bioinformaticians and scientists to directly examine the data and model's inner workings.  The detailed exploration of clusters, including the analysis found in the `code/Elbow_mz/elbowClassifier` and `code/interpretability` directories, and the functional labeling of clusters, significantly enhance interpretability.  In essence, the project balances the need for a robust, accurate classifier with the crucial requirement of providing biological insight into the classification process.  The use of Sentence Transformers for generating embeddings, though used for prediction, also contributes to interpretability by allowing for investigation of textual features related to gene descriptions.

## Configuration
No specific configuration files are used, but parameters such as the number of estimators for the Random Forest and the maximum depth of the decision trees are adjustable within the code.  The NCBI Entrez API key and email should be set in the relevant notebook.

## Testing
No formal testing framework is implemented; however, the notebooks contain multiple print statements and visualization blocks that aid in informal testing.

*README.md was made with [Etchr](https://etchr.dev)*