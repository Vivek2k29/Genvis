import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import json
import os
import uuid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="GenVis - Data Visualization Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.success-box {
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
}
.info-box {
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: #d1ecf1;
    border: 1px solid #bee5eb;
    color: #0c5460;
}
</style>
""", unsafe_allow_html=True)

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize OpenAI - USING STREAMLIT SECRETS
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    api_key_available = True
except:
    st.error("⚠️ OpenAI API key not found. Please add it to Streamlit secrets.")
    st.info("Go to ☰ → Settings → Secrets to add your OPENAI_API_KEY")
    api_key_available = False

# ✅ Fixed SimpleRAGSystem class
class SimpleRAGSystem:
    def __init__(self):
        self.chart_templates = []
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None

    def add_chart_template(self, description, chart_type, code_snippet, data_requirements):
        template = {
            'id': str(uuid.uuid4()),
            'description': description,
            'chart_type': chart_type,
            'code_snippet': code_snippet,
            'data_requirements': data_requirements
        }
        self.chart_templates.append(template)
        self._update_tfidf_index()

    def _update_tfidf_index(self):
        if len(self.chart_templates) == 0:
            return
        descriptions = [template['description'] for template in self.chart_templates]
        self.tfidf_matrix = self.vectorizer.fit_transform(descriptions)

    def search_similar_charts(self, query, k=3):
        if self.tfidf_matrix is None or len(self.chart_templates) == 0:
            return []
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-k:][::-1]
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:
                results.append({
                    'template': self.chart_templates[idx],
                    'similarity_score': float(similarities[idx])
                })
        return results

# Initialize RAG system
@st.cache_resource
def initialize_rag_system():
    rag_system = SimpleRAGSystem()

    templates = [
        {
            'description': 'bar chart comparing categories with values',
            'chart_type': 'bar',
            'code_snippet': '''
plt.figure(figsize=(10, 6))
plt.bar(data['{x_column}'], data['{y_column}'])
plt.title('{title}')
plt.xlabel('{x_label}')
plt.ylabel('{y_label}')
plt.xticks(rotation=45)
plt.tight_layout()
''',
            'data_requirements': 'Requires one categorical column and one numerical column'
        },
        {
            'description': 'line chart showing trends over time',
            'chart_type': 'line',
            'code_snippet': '''
plt.figure(figsize=(12, 6))
plt.plot(data['{x_column}'], data['{y_column}'], marker='o')
plt.title('{title}')
plt.xlabel('{x_label}')
plt.ylabel('{y_label}')
plt.grid(True, alpha=0.3)
plt.tight_layout()
''',
            'data_requirements': 'Requires date/time column and numerical column'
        },
        {
            'description': 'scatter plot showing correlation between two variables',
            'chart_type': 'scatter',
            'code_snippet': '''
plt.figure(figsize=(10, 6))
plt.scatter(data['{x_column}'], data['{y_column}'], alpha=0.6)
plt.title('{title}')
plt.xlabel('{x_label}')
plt.ylabel('{y_label}')
plt.grid(True, alpha=0.3)
plt.tight_layout()
''',
            'data_requirements': 'Requires two numerical columns'
        },
        {
            'description': 'histogram showing distribution of data',
            'chart_type': 'histogram',
            'code_snippet': '''
plt.figure(figsize=(10, 6))
plt.hist(data['{column}'], bins=20, alpha=0.7, edgecolor='black')
plt.title('{title}')
plt.xlabel('{x_label}')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.tight_layout()
''',
            'data_requirements': 'Requires one numerical column'
        },
        {
            'description': 'pie chart showing proportions',
            'chart_type': 'pie',
            'code_snippet': '''
plt.figure(figsize=(8, 8))
plt.pie(data['{values}'], labels=data['{labels}'], autopct='%1.1f%%', startangle=90)
plt.title('{title}')
plt.axis('equal')
plt.tight_layout()
''',
            'data_requirements': 'Requires one categorical column and one numerical column'
        },
        {
            'description': 'box plot showing distribution statistics',
            'chart_type': 'box',
            'code_snippet': '''
plt.figure(figsize=(10, 6))
plt.boxplot(data['{column}'])
plt.title('{title}')
plt.ylabel('{y_label}')
plt.grid(True, alpha=0.3)
plt.tight_layout()
''',
            'data_requirements': 'Requires one numerical column'
        }
    ]

    for template in templates:
        rag_system.add_chart_template(
            template['description'],
            template['chart_type'],
            template['code_snippet'],
            template['data_requirements']
        )

    return rag_system

def analyze_dataframe(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    date_columns = []
    for col in categorical_columns:
        try:
            pd.to_datetime(df[col])
            date_columns.append(col)
        except:
            pass
    return {
        'numeric_columns': numeric_columns,
        'categorical_columns': categorical_columns,
        'date_columns': date_columns,
        'shape': df.shape,
        'sample_data': df.head(5).to_dict('records')
    }

def generate_chart_code(user_query, df, similar_templates):
    column_info = analyze_dataframe(df)
    if api_key_available:
        context = f"""
        User Query: {user_query}
        DataFrame Information:
        - Numeric Columns: {column_info['numeric_columns']}
        - Categorical Columns: {column_info['categorical_columns']}
        - Date Columns: {column_info['date_columns']}
        - Shape: {column_info['shape']}
        Generate Python matplotlib code to create the requested visualization.
        Use the dataframe 'data' which is already loaded.
        Return ONLY the Python code without any explanation.
        Make sure the chart is well-formatted with proper labels and title.
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a data visualization expert. Generate clean, efficient matplotlib code based on the user's request and available data."},
                    {"role": "user", "content": context}
                ],
                max_tokens=500,
                temperature=0.1
            )
            code = response.choices[0].message.content.strip()
            if code.startswith('```python'):
                code = code[9:]
            if code.endswith('```'):
                code = code[:-3]
            return code.strip()
        except Exception as e:
            st.warning(f"OpenAI API error: {str(e)}. Using fallback method.")
    return generate_fallback_code(user_query, df, similar_templates)

def generate_fallback_code(user_query, df, similar_templates):
    if similar_templates:
        template = similar_templates[0]['template']
        code_snippet = template['code_snippet']
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        if template['chart_type'] in ['bar', 'pie']:
            x_col = categorical_cols[0] if len(categorical_cols) > 0 else df.columns[0]
            y_col = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[1] if len(df.columns) > 1 else df.columns[0]
        elif template['chart_type'] in ['scatter', 'line']:
            x_col = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]
            y_col = numeric_cols[1] if len(numeric_cols) > 1 else df.columns[-1]
        else:
            x_col = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]
            y_col = x_col
        code = code_snippet.format(
            x_column=x_col,
            y_column=y_col,
            column=x_col,
            values=y_col,
            labels=x_col,
            title=user_query,
            x_label=x_col,
            y_label=y_col
        )
        return code
    else:
        return '''
plt.figure(figsize=(10, 6))
if len(data.columns) >= 2:
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1])
    plt.title('Data Visualization')
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
else:
    plt.text(0.5, 0.5, 'Not enough columns for visualization', ha='center', va='center')
plt.tight_layout()
'''

def execute_chart_code(code, df):
    try:
        local_vars = {'data': df, 'plt': plt, 'pd': pd, 'np': np, 'sns': sns}
        exec(code, globals(), local_vars)
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return image_base64
    except Exception as e:
        raise Exception(f"Error executing chart code: {str(e)}")

def main():
    st.markdown('<h1 class="main-header">📊 GenVis</h1>', unsafe_allow_html=True)
    st.markdown("### Transform your data into beautiful visualizations with simple English descriptions")
    rag_system = initialize_rag_system()

    if 'uploaded_df' not in st.session_state:
        st.session_state.uploaded_df = None
    if 'data_analysis' not in st.session_state:
        st.session_state.data_analysis = None
    if 'generated_chart' not in st.session_state:
        st.session_state.generated_chart = None

    with st.sidebar:
        st.markdown("## 📁 Data Upload")
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.uploaded_df = df
                st.session_state.data_analysis = analyze_dataframe(df)
                st.markdown("### 📊 Data Overview")
                analysis = st.session_state.data_analysis
                st.write(f"**Shape:** {analysis['shape']}")
                st.write(f"**Numeric Columns:** {', '.join(analysis['numeric_columns'])}")
                st.write(f"**Categorical Columns:** {', '.join(analysis['categorical_columns'])}")
                with st.expander("View Sample Data"):
                    st.dataframe(df.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("## 💬 Describe Your Visualization")
        if st.session_state.uploaded_df is not None:
            st.markdown('<div class="success-box">✅ File uploaded successfully! Now describe what you want to visualize.</div>', unsafe_allow_html=True)
            st.markdown("### 💡 Example Queries:")
            example_queries = [
                "Show me a bar chart of sales by region",
                "Create a line chart showing revenue trend over time",
                "Make a scatter plot comparing price and quantity",
                "Generate a histogram of customer ages",
                "Create a pie chart of product categories"
            ]
            for i, query in enumerate(example_queries):
                if st.button(f"Example {i+1}: {query}", key=f"example_{i}"):
                    st.session_state.user_query = query
            user_query = st.text_area(
                "Describe the chart you want to create:",
                value=st.session_state.get('user_query', ''),
                placeholder="e.g., 'Create a bar chart showing sales by category'",
                height=100
            )
            if st.button("🚀 Generate Visualization", type="primary", use_container_width=True):
                if user_query and st.session_state.uploaded_df is not None:
                    with st.spinner("Generating your visualization..."):
                        try:
                            similar_templates = rag_system.search_similar_charts(user_query)
                            chart_code = generate_chart_code(user_query, st.session_state.uploaded_df, similar_templates)
                            chart_image = execute_chart_code(chart_code, st.session_state.uploaded_df)
                            st.session_state.generated_chart = {
                                'image': chart_image,
                                'code': chart_code,
                                'similar_templates': len(similar_templates),
                                'query': user_query
                            }
                            st.success("✅ Visualization generated successfully!")
                        except Exception as e:
                            st.error(f"❌ Chart generation failed: {str(e)}")
                else:
                    st.warning("Please enter a description for your visualization")
        else:
            st.markdown('<div class="info-box">📝 Please upload a CSV file to get started.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("## 📈 Generated Visualization")
        if st.session_state.generated_chart:
            chart_data = st.session_state.generated_chart
            st.image(f"data:image/png;base64,{chart_data['image']}", use_column_width=True, caption=f"Chart for: '{chart_data['query']}'")
            st.markdown("### ℹ️ Generation Info")
            st.write(f"Similar templates found: {chart_data['similar_templates']}")
            with st.expander("View Generated Code"):
                st.code(chart_data['code'], language='python')
            st.download_button(
                label="📥 Download Chart",
                data=base64.b64decode(chart_data['image']),
                file_name="genvis_chart.png",
                mime="image/png",
                use_container_width=True
            )
        else:
            st.markdown('<div class="info-box">👆 Describe your visualization and click generate to see the chart here.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "### 🔧 How GenVis works:\n"
        "1. **Upload** your CSV file\n"
        "2. **Describe** what visualization you want in plain English\n"
        "3. **Generate** and view your chart using AI + RAG\n"
        "4. **Download** or modify your query\n\n"
        "*Powered by RAG Architecture and AI*"
    )

if __name__ == "__main__":
    main()
