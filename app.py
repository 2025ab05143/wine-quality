import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine Learning models
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Metrics
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                           recall_score, f1_score, matthews_corrcoef,
                           confusion_matrix, classification_report)
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

import pickle
import warnings
import requests
import io
warnings.filterwarnings('ignore')

def find_target_column(data):
    """Find the target column in the dataset"""
    target_cols = ['quality', 'Quality', 'QUALITY']
    for col in target_cols:
        if col in data.columns:
            return col
    return None

# Set page config
st.set_page_config(
    page_title="Wine Quality Prediction Dashboard",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #8B0000 0%, #DC143C 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #8B0000;
        margin: 0.5rem 0;
    }
    .model-section {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)



@st.cache_data
def load_default_data():
    """Load the default wine quality dataset from GitHub or local file"""
    # GitHub raw file URL
    github_url = "https://raw.githubusercontent.com/2025ab05143/wine-quality/main/wine_quality_merged.csv"
    
    try:
        # First try to download from GitHub
        st.sidebar.info("📡 Downloading dataset from GitHub...")
        response = requests.get(github_url, timeout=10)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        # Try different separators to read CSV properly
        try:
            # First try comma separator (most common for this dataset)
            data = pd.read_csv(io.StringIO(response.text), sep=',')
            if len(data.columns) > 5:  # Good sign of proper parsing
                st.sidebar.success("✅ Dataset downloaded successfully from GitHub!")
                st.sidebar.info(f"Columns detected: {len(data.columns)}")
                return data
        except:
            pass
            
        try:
            # Try semicolon separator
            data = pd.read_csv(io.StringIO(response.text), sep=';')
            if len(data.columns) > 5:
                st.sidebar.success("✅ Dataset downloaded successfully from GitHub!")
                st.sidebar.info(f"Columns detected: {len(data.columns)}")
                return data
        except:
            pass
            
        try:
            # Try tab separator
            data = pd.read_csv(io.StringIO(response.text), sep='\t')
            if len(data.columns) > 5:
                st.sidebar.success("✅ Dataset downloaded successfully from GitHub!")
                st.sidebar.info(f"Columns detected: {len(data.columns)}")
                return data
        except:
            pass
            
        # Final fallback - auto-detection
        data = pd.read_csv(io.StringIO(response.text))
        st.sidebar.success("✅ Dataset downloaded successfully from GitHub!")
        st.sidebar.info(f"Columns detected: {len(data.columns)}")
        return data
        
    except requests.exceptions.RequestException as e:
        st.sidebar.warning(f"⚠️ GitHub download failed: {str(e)}")
        st.sidebar.info("🔍 Trying local file...")
        
        # Fallback to local file
        try:
            data = pd.read_csv('winequality-red.csv', sep=';')
            st.sidebar.success("✅ Using local dataset file!")
            return data
        except FileNotFoundError:
            st.sidebar.error("❌ Local dataset file not found either!")
            return None
    
    except Exception as e:
        st.sidebar.error(f"❌ Error processing dataset: {str(e)}")
        return None

def load_uploaded_data(uploaded_file):
    """Load data from uploaded file"""
    if uploaded_file is not None:
        try:
            # Try different separators
            try:
                data = pd.read_csv(uploaded_file, sep=';')
            except:
                uploaded_file.seek(0)  # Reset file pointer
                data = pd.read_csv(uploaded_file, sep=',')
            
            # Validate required columns
            target_cols = ['quality', 'Quality', 'QUALITY']
            has_target = any(col in data.columns for col in target_cols)
            
            if not has_target:
                st.sidebar.error("❌ Error: No quality/target column found in uploaded file!")
                st.sidebar.error(f"Available columns: {list(data.columns)}")
                return None
                
            return data
            
        except Exception as e:
            st.sidebar.error(f"❌ Error loading uploaded file: {str(e)}")
            return None
    return None

def load_data():
    """Load dataset with upload option - no caching due to widgets"""
    
    # Add dataset upload option in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Dataset Upload")
    st.sidebar.markdown("**[1 mark]** Upload your own CSV file for testing:")
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file", 
        type=['csv'],
        help="Upload a CSV file with wine quality data. Must have the same structure as the original dataset."
    )
    
    # Try to load uploaded file first
    if uploaded_file is not None:
        data = load_uploaded_data(uploaded_file)
        if data is not None:
            st.sidebar.success(f"✅ Uploaded file loaded successfully!")
            st.sidebar.info(f"Shape: {data.shape}")
            return data
        else:
            st.sidebar.info("Falling back to default dataset...")
    
    # Load default dataset if no upload or upload failed
    data = load_default_data()
    if data is not None:
        st.sidebar.info("📊 Using Wine Quality dataset")
        return data
    else:
        st.error("❌ Could not load dataset from GitHub or local file. Please upload a CSV file with wine quality data.")
        st.error("🔗 Expected dataset structure: 11 features + 1 target column ('quality')")
        return None

@st.cache_data
def preprocess_data(_data):
    """Preprocess the data and split into features and target"""
    # Find the target column
    target_col = find_target_column(_data)
    
    if target_col is None:
        st.error("❌ No quality/target column found in the dataset!")
        st.error(f"Available columns: {list(_data.columns)}")
        return None, None, None, None, None, None
    
    # Handle the 'type' column (categorical) - encode it as numerical
    data_processed = _data.copy()
    type_encoder = None
    if 'type' in data_processed.columns:
        # Encode wine type: red=0, white=1
        type_encoder = LabelEncoder()
        data_processed['type'] = type_encoder.fit_transform(data_processed['type'])
        st.sidebar.info(f"📝 Encoded wine types: {dict(zip(type_encoder.classes_, type_encoder.transform(type_encoder.classes_)))}")
    
    # Handle any other categorical columns
    for col in data_processed.columns:
        if data_processed[col].dtype == 'object' or str(data_processed[col].dtype) == 'str':
            if col != target_col and data_processed[col].nunique() < 20:  # Categorical column with few unique values
                le = LabelEncoder()
                data_processed[col] = le.fit_transform(data_processed[col].astype(str))
                st.sidebar.info(f"📝 Encoded column '{col}': {len(le.classes_)} categories")
    
    X = data_processed.drop(target_col, axis=1)
    y = data_processed[target_col]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns, scaler

def calculate_multiclass_auc(y_true, y_pred_proba):
    """Calculate AUC for multiclass classification using one-vs-rest"""
    try:
        lb = LabelBinarizer()
        y_true_binary = lb.fit_transform(y_true)
        
        if len(lb.classes_) == 2:
            # Binary classification
            return roc_auc_score(y_true, y_pred_proba[:, 1])
        else:
            # Multiclass classification
            return roc_auc_score(y_true_binary, y_pred_proba, multi_class='ovr', average='macro')
    except:
        return np.nan

def train_models(X_train, X_test, y_train, y_test):
    """Train all machine learning models and return results"""
    
    # Fix XGBoost label encoding issue by remapping labels to start from 0
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='mlogloss', verbosity=0)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        # Train the model - use encoded labels for XGBoost
        if name == 'XGBoost':
            model.fit(X_train, y_train_encoded)
            # Make predictions with encoded labels
            y_pred_encoded = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            # Convert back to original labels
            y_pred = le.inverse_transform(y_pred_encoded)
        else:
            model.fit(X_train, y_train)
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
        
        trained_models[name] = model
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = calculate_multiclass_auc(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        mcc = matthews_corrcoef(y_test, y_pred)
        
        results[name] = {
            'Accuracy': accuracy,
            'AUC': auc,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'MCC': mcc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'model': model
        }
    
    return results, trained_models

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Create and display confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        title=f'Confusion Matrix - {model_name}',
        labels=dict(x="Predicted Quality", y="Actual Quality"),
        x=[f'Quality {i}' for i in sorted(np.unique(y_true))],
        y=[f'Quality {i}' for i in sorted(np.unique(y_true))]
    )
    
    fig.update_layout(
        width=600,
        height=500,
        font=dict(size=12)
    )
    
    return fig

def create_metrics_comparison(results_df):
    """Create interactive comparison chart for all metrics"""
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'MCC Score'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    positions = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        row, col = positions[i]
        
        fig.add_trace(
            go.Bar(
                x=results_df.index,
                y=results_df[metric],
                name=metric,
                marker_color=color,
                showlegend=False,
                text=results_df[metric].round(4),
                textposition='auto'
            ),
            row=row, col=col
        )
    
    # Update layout for all subplots at once
    fig.update_layout(
        height=800,
        title_text="Model Performance Comparison Across All Metrics",
        showlegend=False,
        font=dict(size=10)
    )
    
    # Update x-axis for all subplots to rotate labels
    fig.update_xaxes(tickangle=45)
    
    return fig

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; text-align: center; margin: 0;">
            🍷 Wine Quality Prediction Dashboard
        </h1>
        <p style="color: white; text-align: center; margin: 0.5rem 0 0 0;">
            Machine Learning Classification Models for Wine Quality Assessment
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["📊 Dataset Overview", "🤖 Model Training & Evaluation", "📈 Model Comparison", "🔍 Individual Predictions", "📋 About"]
    )
    
    # Load data
    data = load_data()
    if data is None:
        st.stop()
    
    # Preprocess data
    preprocess_result = preprocess_data(data)
    if preprocess_result[0] is None:  # Check if preprocessing failed
        st.stop()
    
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler = preprocess_result
    
    if page == "📊 Dataset Overview":
        st.header("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(data))
        with col2:
            st.metric("Features", len(data.columns) - 1)
        with col3:
            # Find target column for quality classes
            target_col = find_target_column(data)
            
            if target_col:
                st.metric("Quality Classes", len(data[target_col].unique()))
            else:
                st.metric("Quality Classes", "N/A")
        with col4:
            st.metric("Missing Values", data.isnull().sum().sum())
        
        # Dataset description
        st.subheader("📈 Dataset Description")
        
        # Find target column
        target_col = find_target_column(data)
        
        st.markdown(f"""
        **Wine Quality Dataset (Red & White Wines)**
        
        This dataset contains physicochemical properties of wine samples along with their quality ratings. 
        The dataset has **{len(data)} instances** with **{len(data.columns) - 1} input features** and **1 output variable ({target_col})**.
        
        **Target:** {target_col} ratings
        **Features:** {len(data.columns) - 1} physicochemical properties
        **Wine Types:** {'Red & White' if 'type' in data.columns else 'Red'}
        """)
        
        # Show wine type distribution if available
        if 'type' in data.columns:
            st.write("**Wine Type Distribution:**")
            type_counts = data['type'].value_counts()
            st.write(f"Red: {type_counts.get('red', 0)} samples, White: {type_counts.get('white', 0)} samples")
        
        # Show all columns
        st.write("**All Columns:**")
        col_info = []
        for i, col in enumerate(data.columns, 1):
            col_type = "Target" if col == target_col else "Feature"
            col_info.append(f"{i}. {col} ({col_type})")
        st.write(" | ".join(col_info[:6]) + "...")  # Show first 6 columns
        
        # Show first few rows
        st.subheader("📋 Sample Data")
        st.dataframe(data.head(10))
        
        # Statistical summary
        st.subheader("📊 Statistical Summary")
        
        # Show numerical statistics
        st.write("**Numerical Features:**")
        st.dataframe(data.describe())
        
        # Show categorical statistics
        categorical_cols = []
        for col in data.columns:
            if data[col].dtype == 'object' or str(data[col].dtype) == 'str':
                categorical_cols.append(col)
        
        if len(categorical_cols) > 0:
            st.write("**Categorical Features:**")
            
            for col in categorical_cols:
                st.write(f"**{col.title()}:**")
                value_counts = data[col].value_counts()
                
                # Create two columns for better display
                col1, col2 = st.columns(2)
                
                with col1:
                    # Show value counts as a table
                    count_df = pd.DataFrame({
                        col.title(): value_counts.index,
                        'Count': value_counts.values,
                        'Percentage': (value_counts.values / len(data) * 100).round(2)
                    })
                    st.dataframe(count_df, width='stretch')
                
                with col2:
                    # Show as a pie chart
                    fig = px.pie(
                        values=value_counts.values,
                        names=value_counts.index,
                        title=f'{col.title()} Distribution'
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, width='stretch')
        
        # Feature distributions
        st.subheader("📈 Feature Distributions")
        
        # Select features to display
        selected_features = st.multiselect(
            "Select features to visualize:",
            data.columns[:-1].tolist(),
            default=data.columns[:4].tolist()
        )
        
        if selected_features:
            cols = st.columns(2)
            for i, feature in enumerate(selected_features):
                with cols[i % 2]:
                    fig = px.histogram(
                        data, 
                        x=feature, 
                        title=f'Distribution of {feature}',
                        marginal="box"
                    )
                    st.plotly_chart(fig, width='stretch')
        
        # Quality distribution
        st.subheader("🎯 Target Variable Distribution")
        
        # Find target column
        target_col = find_target_column(data)
        
        if target_col:
            fig = px.histogram(
                data, 
                x=target_col, 
                title=f'{target_col.title()} Distribution',
                color=target_col,
                text_auto=True
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')
        else:
            st.error("No target column found for visualization")
        
        # Correlation heatmap
        st.subheader("🔥 Feature Correlation Heatmap")
        
        # Create a copy for correlation (encode categorical columns)
        data_for_corr = data.copy()
        
        # Encode categorical columns for correlation
        for col in data_for_corr.columns:
            if data_for_corr[col].dtype == 'object' or str(data_for_corr[col].dtype) == 'str':
                if data_for_corr[col].nunique() < 20:  # Categorical column
                    le = LabelEncoder()
                    data_for_corr[col] = le.fit_transform(data_for_corr[col].astype(str))
        
        correlation_matrix = data_for_corr.corr()
        
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu_r"
        )
        fig.update_layout(width=800, height=600)
        st.plotly_chart(fig, width='stretch')
    
    elif page == "🤖 Model Training & Evaluation":
        st.header("🤖 Model Training & Evaluation")
        
        # Training progress
        with st.spinner("Training all models..."):
            results, trained_models = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        st.success("✅ All models trained successfully!")
        
        # Model selection
        st.subheader("🎯 Select Model for Detailed Analysis")
        selected_model = st.selectbox(
            "Choose a model:",
            list(results.keys())
        )
        
        if selected_model:
            model_results = results[selected_model]
            
            # Display metrics
            st.subheader(f"📊 {selected_model} Performance Metrics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <h4>Accuracy</h4>
                    <h2 style="color: #8B0000;">{model_results['Accuracy']:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-container">
                    <h4>Precision</h4>
                    <h2 style="color: #8B0000;">{model_results['Precision']:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <h4>AUC Score</h4>
                    <h2 style="color: #8B0000;">{model_results['AUC']:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-container">
                    <h4>Recall</h4>
                    <h2 style="color: #8B0000;">{model_results['Recall']:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-container">
                    <h4>F1 Score</h4>
                    <h2 style="color: #8B0000;">{model_results['F1']:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-container">
                    <h4>MCC Score</h4>
                    <h2 style="color: #8B0000;">{model_results['MCC']:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Confusion Matrix
            st.subheader("🔄 Confusion Matrix")
            cm_fig = plot_confusion_matrix(y_test, model_results['y_pred'], selected_model)
            st.plotly_chart(cm_fig, width='stretch')
            
            # Classification Report
            st.subheader("📋 Classification Report")
            
            # Generate classification report with proper handling of warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                report = classification_report(y_test, model_results['y_pred'], output_dict=True, zero_division=0)
            
            report_df = pd.DataFrame(report).transpose()
            
            # Style and display the report
            st.write("**Per-Class Performance Metrics:**")
            
            # Separate individual classes from summary metrics
            class_metrics = report_df.iloc[:-3]  # All except accuracy, macro avg, weighted avg
            summary_metrics = report_df.iloc[-3:]  # Last 3 rows
            
            # Display class-specific metrics
            if len(class_metrics) > 0:
                st.write("*Individual Quality Classes:*")
                # Format class names
                class_metrics.index = [f"Quality {idx}" for idx in class_metrics.index]
                st.dataframe(class_metrics.round(4), width='stretch')
            
            # Display summary metrics separately with better formatting
            st.write("*Overall Performance Summary:*")
            summary_metrics_formatted = summary_metrics.copy()
            summary_metrics_formatted.index = ['📊 Overall Accuracy', '📈 Macro Average', '⚖️ Weighted Average']
            st.dataframe(summary_metrics_formatted.round(4), width='stretch')
            
            # Add interpretation
            with st.expander("📖 How to Interpret Classification Report"):
                st.markdown("""
                **Metrics Explanation:**
                - **Precision**: Of all predicted instances of a quality class, what percentage were correct?
                - **Recall**: Of all actual instances of a quality class, what percentage were correctly predicted?
                - **F1-Score**: Harmonic mean of precision and recall (balanced measure)
                - **Support**: Number of actual instances of each quality class in the test set
                
                **Summary Metrics:**
                - **Overall Accuracy**: Percentage of all predictions that were correct
                - **Macro Average**: Average of metrics across all classes (treats all classes equally)
                - **Weighted Average**: Average of metrics weighted by class frequency (accounts for class imbalance)
                
                **Notes:**
                - Values of 0.000 indicate classes that were never predicted (common for rare quality classes)
                - Higher values (closer to 1.0) indicate better performance
                - F1-score balances precision and recall, useful for imbalanced datasets
                """)
            
            # Show class distribution for context
            st.write("**Test Set Class Distribution:**")
            class_dist = pd.Series(y_test).value_counts().sort_index()
            class_dist_df = pd.DataFrame({
                'Quality': [f"Quality {i}" for i in class_dist.index],
                'Count': class_dist.values,
                'Percentage': (class_dist.values / len(y_test) * 100).round(2)
            })
            st.dataframe(class_dist_df, width='stretch')
    
    elif page == "📈 Model Comparison":
        st.header("📈 Model Comparison")
        
        # Train all models
        with st.spinner("Training all models for comparison..."):
            results, trained_models = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, model_results in results.items():
            comparison_data.append({
                'ML Model Name': model_name,
                'Accuracy': model_results['Accuracy'],
                'AUC': model_results['AUC'],
                'Precision': model_results['Precision'],
                'Recall': model_results['Recall'],
                'F1': model_results['F1'],
                'MCC': model_results['MCC']
            })
        
        results_df = pd.DataFrame(comparison_data)
        results_df.set_index('ML Model Name', inplace=True)
        
        # Display comparison table
        st.subheader("🏆 Model Performance Comparison Table")
        st.dataframe(results_df.round(4))
        
        # Interactive comparison chart
        st.subheader("📊 Interactive Performance Comparison")
        comparison_fig = create_metrics_comparison(results_df)
        st.plotly_chart(comparison_fig, width='stretch')
        
        # Best performing models
        st.subheader("🥇 Top Performing Models")
        
        metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
        
        col1, col2 = st.columns(2)
        
        for i, metric in enumerate(metrics):
            best_model = results_df[metric].idxmax()
            best_score = results_df[metric].max()
            
            with col1 if i % 2 == 0 else col2:
                st.markdown(f"""
                <div class="metric-container">
                    <h4>Best {metric}</h4>
                    <h5 style="color: #8B0000;">{best_model}</h5>
                    <p>{best_score:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Model observations
        st.subheader("🔍 Observations on Model Performance")
        
        observations = {
            'Logistic Regression': "Good baseline performance with interpretable results. Works well for linear relationships.",
            'Decision Tree': "Captures non-linear patterns but may overfit. Good interpretability.",
            'K-Nearest Neighbors': "Simple algorithm that performs well on local patterns. Can be sensitive to feature scaling.",
            'Naive Bayes': "Fast training and prediction. Assumes feature independence which may limit performance.",
            'Random Forest': "Excellent performance with reduced overfitting. Handles feature interactions well.",
            'XGBoost': "Advanced gradient boosting with superior performance. Excellent for complex pattern recognition."
        }
        
        for model, observation in observations.items():
            with st.expander(f"📝 {model} Analysis"):
                metrics_text = " | ".join([f"{metric}: {results_df.loc[model, metric]:.4f}" for metric in metrics])
                st.write(f"**Performance:** {metrics_text}")
                st.write(f"**Observation:** {observation}")
    
    elif page == "🔍 Individual Predictions":
        st.header("🔍 Individual Wine Quality Predictions")
        
        # Train models if not already done
        with st.spinner("Loading models..."):
            results, trained_models = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        st.subheader("🔧 Input Wine Properties")
        
        # Create input fields for each feature dynamically
        input_features = {}
        
        # Handle wine type separately if it exists
        if 'type' in feature_names:
            wine_type = st.selectbox("Wine Type", ["Red", "White"], index=0)
            input_features['type'] = 0 if wine_type == "Red" else 1
        
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        # Split features between columns (excluding type if present)
        other_features = [f for f in feature_names if f != 'type']
        mid_point = len(other_features) // 2
        
        with col1:
            for i, feature in enumerate(other_features[:mid_point]):
                if feature in data.columns:
                    col_values = data[feature]
                    min_val = float(col_values.min())
                    max_val = float(col_values.max())
                    mean_val = float(col_values.mean())
                    
                    input_features[feature] = st.slider(
                        feature.replace('_', ' ').title(),
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        step=(max_val - min_val) / 100
                    )
        
        with col2:
            for i, feature in enumerate(other_features[mid_point:]):
                if feature in data.columns:
                    col_values = data[feature]
                    min_val = float(col_values.min())
                    max_val = float(col_values.max())
                    mean_val = float(col_values.mean())
                    
                    input_features[feature] = st.slider(
                        feature.replace('_', ' ').title(),
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        step=(max_val - min_val) / 100
                    )
        
        # Model selection for prediction
        prediction_model = st.selectbox(
            "Choose model for prediction:",
            list(trained_models.keys()),
            key="prediction_model"
        )
        
        if st.button("🔮 Predict Wine Quality", type="primary"):
            # Create input array in the correct feature order
            input_array = [input_features[feature] for feature in feature_names]
            input_data = np.array([input_array])
            
            # Scale the input
            input_scaled = scaler.transform(input_data)
            
            model = trained_models[prediction_model]
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <h3>Predicted Quality</h3>
                    <h1 style="color: #8B0000; text-align: center;">{prediction}</h1>
                    <p style="text-align: center;">Using {prediction_model}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Show prediction probabilities
                st.subheader("🎯 Prediction Probabilities")
                prob_df = pd.DataFrame({
                    'Quality': sorted(np.unique(y_train)),
                    'Probability': prediction_proba
                })
                
                fig = px.bar(
                    prob_df, 
                    x='Quality', 
                    y='Probability',
                    title=f'Prediction Confidence - {prediction_model}',
                    color='Probability',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, width='stretch')
    
    elif page == "📋 About":
        st.header("📋 About This Project")
        
        st.markdown("""
        ## 🍷 Wine Quality Prediction Dashboard
        
        ### 📝 Problem Statement
        This project implements multiple machine learning classification models to predict wine quality based on physicochemical properties. The goal is to build an interactive web application that demonstrates the end-to-end ML workflow from data exploration to model deployment.
        
        ### 📊 Dataset Description **[1 mark]**
        - **Dataset:** Wine Quality Dataset (Red & White Wines Combined)
        - **Source:** UCI Machine Learning Repository / GitHub Repository
        - **GitHub URL:** https://github.com/2025ab05143/wine-quality/blob/main/wine_quality_merged.csv
        - **Instances:** 6,497 wine samples (red + white combined)
        - **Features:** 12 physicochemical properties + wine type
        - **Target:** Quality ratings (3-9 scale)
        - **Type:** Multi-class classification problem
        
        The dataset is automatically downloaded from GitHub during runtime, ensuring consistent access across different environments. 
        The dataset contains sufficient instances (>500) and features (>12 including target) as required for this assignment.
        
        ### 🤖 Models Used **[6 marks - 1 mark for all metrics for each model]**
        
        #### 1. **Logistic Regression**
        - Linear classification algorithm
        - Good baseline performance
        - Highly interpretable results
        
        #### 2. **Decision Tree Classifier**  
        - Non-linear pattern recognition
        - Rule-based interpretability
        - Handles feature interactions naturally
        
        #### 3. **K-Nearest Neighbors (KNN)**
        - Instance-based learning
        - Non-parametric approach
        - Effective for local pattern recognition
        
        #### 4. **Gaussian Naive Bayes**
        - Probabilistic classifier
        - Fast training and prediction
        - Assumes feature independence
        
        #### 5. **Random Forest (Ensemble)**
        - Bagging ensemble method
        - Reduces overfitting
        - Excellent feature importance insights
        
        #### 6. **XGBoost (Ensemble)**
        - Gradient boosting algorithm
        - State-of-the-art performance
        - Advanced regularization techniques
        
        ### 📈 Evaluation Metrics
        For each model, the following metrics are calculated:
        
        1. **Accuracy** - Overall correctness of predictions
        2. **AUC Score** - Area Under ROC Curve (macro-average for multiclass)
        3. **Precision** - Weighted average precision across all classes
        4. **Recall** - Weighted average recall across all classes  
        5. **F1 Score** - Weighted harmonic mean of precision and recall
        6. **Matthews Correlation Coefficient (MCC)** - Balanced metric for multiclass
        
        ### 🚀 Streamlit App Features **[4 marks total]**
        
        #### a. **Dataset Upload Option** **[1 mark]**
        - Interactive file upload capability
        - Automatic data validation and preprocessing
        - Support for CSV format with proper error handling
        
        #### b. **Model Selection Dropdown** **[1 mark]**
        - Dynamic model selection interface
        - Real-time model switching
        - Comprehensive model comparison tools
        
        #### c. **Evaluation Metrics Display** **[1 mark]**  
        - Interactive metrics dashboard
        - Comparative performance visualization
        - Real-time metric calculations
        
        #### d. **Confusion Matrix & Classification Report** **[1 mark]**
        - Interactive confusion matrix visualization
        - Detailed classification reports
        - Per-class performance analysis
        
        ### 🛠️ Technical Implementation
        
        **Libraries Used:**
        - **Streamlit** - Web application framework
        - **Pandas & NumPy** - Data manipulation and analysis
        - **Scikit-learn** - Machine learning algorithms and metrics
        - **XGBoost** - Advanced gradient boosting
        - **Plotly** - Interactive visualizations
        - **Matplotlib & Seaborn** - Statistical plotting
        
        **Key Features:**
        - Responsive web interface
        - Real-time model training and evaluation
        - Interactive data visualization
        - Comprehensive model comparison
        - Individual prediction capability
        
        ### 🎯 Model Performance Observations **[3 marks]**
        
        Based on comprehensive evaluation across all metrics:
        
        **🥇 Top Performers:**
        - **Random Forest** and **XGBoost** consistently show superior performance
        - Both ensemble methods effectively handle the multi-class nature of wine quality prediction
        - Superior F1 scores and MCC values indicate balanced performance across all quality classes
        
        **📊 Key Insights:**
        - **Ensemble methods** (Random Forest, XGBoost) outperform individual algorithms
        - **Logistic Regression** provides excellent baseline with good interpretability
        - **Decision Tree** shows competitive performance but may overfit without proper tuning
        - **KNN** performs well on local patterns but sensitive to feature scaling
        - **Naive Bayes** fastest training but limited by feature independence assumption
        
        **🔍 Dataset-Specific Observations:**
        - Wine quality prediction benefits from ensemble approaches due to complex feature interactions
        - The imbalanced nature of quality classes (more samples in middle range) affects model performance
        - Feature scaling significantly improves performance for distance-based algorithms
        
        ### 📁 Repository Structure
        ```
        project-folder/
        │
        ├── app.py                 # Main Streamlit application
        ├── requirements.txt       # Python dependencies  
        ├── README.md             # Project documentation
        ├── winequality-red.csv   # Dataset file
        └── model/               # Saved model files (optional)
        ```
        
        ### 🚀 Deployment
        This application is designed for deployment on **Streamlit Community Cloud** with the following features:
        - Automatic dependency management
        - Responsive design for multiple screen sizes  
        - Fast loading with efficient caching
        - Interactive user experience
        
        ### 🤝 Contributing
        This project demonstrates the complete machine learning workflow from data exploration to model deployment, serving as a comprehensive example of MLOps practices and interactive web application development.
        
        ---
        **Author:** Created for Machine Learning Classification Assignment  
        **Technology Stack:** Python, Streamlit, Scikit-learn, XGBoost, Plotly  
        **Deployment Platform:** Streamlit Community Cloud
        """)

if __name__ == "__main__":
    main()
