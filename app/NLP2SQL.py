import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import io
import json
import re
import logging
from typing import Dict, List, Optional, Union, TypedDict
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from streamlit_extras.colored_header import colored_header
import streamlit_nested_layout
import numpy as np
from streamlit_extras.dataframe_explorer import dataframe_explorer
import src.database.DB_Config as DB_Config
from src.prompts.Base_Prompt import SYSTEM_MESSAGE
from src.api.LLM_Config import get_completion_from_messages
import hashlib
from datetime import datetime
from time import time
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

SUPPORTED_CHART_TYPES = {
    "Bar Chart": "A chart that presents categorical data with rectangular bars.",
    "Line Chart": "A chart that displays information as a series of data points called 'markers' connected by straight line segments.",
    "Scatter Plot": "A plot that displays values for typically two variables for a set of data.",
    "Area Chart": "A chart that displays quantitative data visually, using the area below the line.",
    "Histogram": "A graphical representation of the distribution of numerical data.",
    "Pie Chart": "A chart that shows proportions of a whole using slices.",
    "Box Plot": "A chart that shows the distribution of data based on quartiles."
}

# Page Configuration
st.set_page_config(
    page_icon="🗃️",
    page_title="Transforming Questions into Queries",
    layout="wide"
)

load_dotenv()

@st.cache_resource
def load_system_message(schemas: dict) -> str:
    """Load and format the system message with JSON-serialized schemas."""
    return SYSTEM_MESSAGE.format(schemas=json.dumps(schemas, indent=2))

# Add input validation to prevent SQL injection and other security vulnerabilities

def validate_sql_query(query: str) -> bool:
    """
    Ensure the SQL query is valid and safe (select queries only).

    Parameters:
    - query (str): The SQL query to validate.

    Returns:
    - bool: True if the query is valid and safe, False otherwise.
    """
    if not isinstance(query, str):
        return False

    disallowed_keywords = r'\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|EXEC)\b'

    if re.search(disallowed_keywords, query, re.IGNORECASE):
        return False

    if not query.strip().lower().startswith(('select', 'with')):
        return False

    if query.count('(') != query.count(')'):
        return False

    return True

def get_data(query: str, db_name: str, db_type: str, host: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None) -> pd.DataFrame:
    """Run the specified query and return the resulting DataFrame."""
    if not validate_sql_query(query):
        logger.error("Invalid or unsafe SQL query.")
        return pd.DataFrame()

    return DB_Config.query_database(query, db_name, db_type, host, user, password)

def save_temp_file(uploaded_file) -> str:
    """Saves an uploaded file to a temporary location."""
    temp_file_path = "temp_database.db"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())
    return temp_file_path

# Step 1: Define Type Classes
class Path(TypedDict):
    description: str
    tables: List[str]
    columns: List[List[str]]
    score: int

class TableColumn(TypedDict):
    table: str
    columns: List[str]
    reason: str

class DecisionLog(TypedDict):
    query_input_details: List[str]
    preprocessing_steps: List[str]
    path_identification: List[Path]
    ambiguity_detection: List[str]
    resolution_criteria: List[str]
    chosen_path_explanation: List[TableColumn]
    generated_sql_query: str
    alternative_paths: List[str]
    execution_feedback: List[str]
    final_summary: str
    visualization_suggestion: Optional[str]

DECISION_LOG_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "The generated SQL query"},
        "error": {"type": ["string", "null"], "description": "Error message if query generation failed"},
        "decision_log": {
            "type": "object",
            "properties": {
                "query_input_details": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Details about the input query"
                },
                "preprocessing_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Steps taken to preprocess the query"
                },
                "path_identification": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "tables": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "columns": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            },
                            "score": {"type": "integer"}
                        },
                        "required": ["description", "tables", "columns", "score"]
                    }
                },
                "ambiguity_detection": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "resolution_criteria": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "chosen_path_explanation": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "table": {"type": "string"},
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "reason": {"type": "string"}
                        },
                        "required": ["table", "columns", "reason"]
                    }
                },
                "generated_sql_query": {"type": "string"},
                "alternative_paths": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "execution_feedback": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "final_summary": {"type": "string"},
                "visualization_suggestion": {"type": ["string", "null"]}
            },
            "required": [
                "query_input_details",
                "preprocessing_steps",
                "path_identification",
                "ambiguity_detection",
                "resolution_criteria",
                "chosen_path_explanation",
                "generated_sql_query",
                "alternative_paths",
                "execution_feedback",
                "final_summary"
            ]
        }
    },
    "required": ["query", "decision_log"]
}

# Step 3: Implement the Modified generate_sql_query Function
def generate_sql_query(user_message: str, schemas: dict, max_attempts: int = 1) -> dict:
    """Generate a SQL query using LLM responses and validate output structure."""
    formatted_system_message = f"""
    {load_system_message(schemas)}

    IMPORTANT: Your response must be valid JSON matching this schema:
    {json.dumps(DECISION_LOG_SCHEMA, indent=2)}

    Ensure all responses strictly follow this format. Include a final_summary and visualization_suggestion in the decision_log.
    """

    for attempt in range(max_attempts):
        try:
            response = get_completion_from_messages(formatted_system_message, user_message)
            # Strip any triple-backtick fences
            response = re.sub(r'^```json\s*', '', response.strip())
            response = re.sub(r'```$', '', response.strip())
            json_response = json.loads(response)

            if not validate_response_structure(json_response):
                logger.warning(f"Invalid response structure. Attempt: {attempt + 1}")
                continue

            return {
                "query": json_response.get('query'),
                "error": json_response.get('error'),
                "decision_log": json_response['decision_log'],
                "visualization_recommendation": json_response['decision_log'].get('visualization_suggestion')
            }

        except json.JSONDecodeError as e:
            logger.exception(f"Invalid JSON response: {response}, Error: {e}")
            continue
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            continue

    return {
        "error": "Failed to generate a valid SQL query after multiple attempts.",
        "decision_log": {
            "execution_feedback": ["Failed to generate a valid response after multiple attempts."],
            "final_summary": "Query generation failed."
        }
    }

# Step 4: Implement Response Validation
def validate_response_structure(response: dict) -> bool:
    """Check if the LLM response follows the expected JSON schema."""
    try:
        if not all(key in response for key in ["query", "decision_log"]):
            return False

        decision_log = response["decision_log"]
        required_sections = [
            "query_input_details",
            "preprocessing_steps",
            "path_identification",
            "ambiguity_detection",
            "resolution_criteria",
            "chosen_path_explanation",
            "generated_sql_query",
            "alternative_paths",
            "execution_feedback",
            "final_summary"
        ]

        if not all(key in decision_log for key in required_sections):
            return False

        for path in decision_log["path_identification"]:
            if not all(key in path for key in ["description", "tables", "columns", "score"]):
                return False

        for explanation in decision_log["chosen_path_explanation"]:
            if not all(key in explanation for key in ["table", "columns", "reason"]):
                return False

        return True

    except Exception as e:
        logger.error(f"Validation error: {e}")
        return False

def build_markdown_decision_log(decision_log: Dict) -> str:
    """Convert the decision log into a markdown-formatted string."""
    markdown_log = []

    # Query Input Details
    if query_details := decision_log.get("query_input_details"):
        markdown_log.extend([
            "### Query Input Analysis",
            "\n".join(f"- {detail}" for detail in query_details),
            ""
        ])

    # Preprocessing Steps
    if preprocessing := decision_log.get("preprocessing_steps"):
        markdown_log.extend([
            "### Preprocessing Steps",
            "\n".join(f"- {step}" for step in preprocessing),
            ""
        ])

    # Path Identification
    if paths := decision_log.get("path_identification"):
        markdown_log.extend([
            "### Path Identification",
            "\n".join([
                f"**Path {i+1}** (Score: {path['score']})\n"
                f"- Description: {path['description']}\n"
                f"- Tables: {', '.join(path['tables'])}\n"
                f"- Columns: {', '.join([', '.join(cols) for cols in path['columns']])}"
                for i, path in enumerate(paths)
            ]),
            ""
        ])

    # Ambiguity Detection
    if ambiguities := decision_log.get("ambiguity_detection"):
        markdown_log.extend([
            "### Ambiguity Analysis",
            "\n".join(f"- {ambiguity}" for ambiguity in ambiguities),
            ""
        ])

    # Resolution Criteria
    if criteria := decision_log.get("resolution_criteria"):
        markdown_log.extend([
            "### Resolution Criteria",
            "\n".join(f"- {criterion}" for criterion in criteria),
            ""
        ])

    # Chosen Path Explanation
    if chosen_path := decision_log.get("chosen_path_explanation"):
        markdown_log.extend([
            "### Selected Tables and Columns",
            "\n".join([
                f"**{table['table']}**\n"
                f"- Columns: {', '.join(table['columns'])}\n"
                f"- Reason: {table['reason']}"
                for table in chosen_path
            ]),
            ""
        ])

    # Generated SQL Query
    if sql_query := decision_log.get("generated_sql_query"):
        markdown_log.extend([
            "### Generated SQL Query",
            f"```sql\n{sql_query}\n```",
            ""
        ])

    # Alternative Paths
    if alternatives := decision_log.get("alternative_paths"):
        markdown_log.extend([
            "### Alternative Approaches",
            "\n".join(f"- {alt}" for alt in alternatives),
            ""
        ])

    # Execution Feedback
    if feedback := decision_log.get("execution_feedback"):
        markdown_log.extend([
            "### Execution Feedback",
            "\n".join(f"- {item}" for item in feedback),
            ""
        ])

    # Final Summary
    if summary := decision_log.get("final_summary"):
        markdown_log.extend([
            "### Summary",
            summary,
            ""
        ])

    # Visualization Suggestion
    if viz_suggestion := decision_log.get("visualization_suggestion"):
        markdown_log.extend([
            "### Visualization Recommendation",
            f"Suggested visualization type: `{viz_suggestion}`",
            ""
        ])

    # Join with proper line breaks and clean up any extra spaces
    return "\n".join(line.rstrip() for line in markdown_log)

def create_chart(df: pd.DataFrame, chart_type: str, x_col: str, y_col: str) -> Optional[any]:
    """Construct a Plotly chart without color column and trendline options."""
    try:
        if chart_type == "Bar Chart":
            fig = px.bar(df, x=x_col, y=y_col, title=f"Bar Chart of {y_col} by {x_col}")
        elif chart_type == "Line Chart":
            fig = px.line(df, x=x_col, y=y_col, title=f"Line Chart of {y_col} by {x_col}", markers=True)
        elif chart_type == "Scatter Plot":
            fig = px.scatter(df, x=x_col, y=y_col, title=f"Scatter Plot of {y_col} vs {x_col}", hover_data=[x_col, y_col])
        elif chart_type == "Area Chart":
            fig = px.area(df, x=x_col, y=y_col, title=f"Area Chart of {y_col} by {x_col}")
        elif chart_type == "Histogram":
            fig = px.histogram(df, x=x_col, nbins=30, title=f"Histogram of {x_col}")
        elif chart_type == "Pie Chart":
            fig = px.pie(df, names=x_col, values=y_col, title=f"Pie Chart of {x_col}")
        elif chart_type == "Box Plot":
            fig = px.box(df, x=x_col, y=y_col, title=f"Box Plot of {y_col} grouped by {x_col}")
        else:
            st.warning("Chart type not recognized.")
            return None

        fig.update_layout(autosize=True, margin=dict(l=20, r=20, t=40, b=20))
        return fig

    except Exception as e:
        st.error(f"Error generating the chart: {e}")
        logger.error(f"Error generating chart: {e}")
        return None

def display_summary_statistics(df: pd.DataFrame) -> None:
    """Show enhanced numeric and categorical summaries with advanced statistics."""
    if df.empty:
        st.warning("The DataFrame is empty. Unable to display summary statistics.")
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

    tab1, tab2, tab3 = st.tabs(["Numeric Summary", "Categorical Analysis", "Advanced Stats"])

    # Numeric Summary Tab
    with tab1:
        st.markdown("### Numeric Summary Statistics")
        stats_df = df[numeric_cols].describe().T
        stats_df['median'] = df[numeric_cols].median()
        stats_df['mode'] = df[numeric_cols].mode().iloc[0]
        stats_df['variance'] = df[numeric_cols].var()
        stats_df['iqr'] = stats_df['75%'] - stats_df['25%']
        stats_df['skew'] = df[numeric_cols].skew()
        stats_df['kurtosis'] = df[numeric_cols].kurt()
        stats_df['coef_var'] = df[numeric_cols].std() / df[numeric_cols].mean()
        st.dataframe(stats_df.style.format("{:.2f}").highlight_max(axis=0, color="lightgreen"))

        for col in numeric_cols:
            st.markdown(f"**Distribution of {col}**")
            hist_fig = px.histogram(df, x=col, nbins=30, title=f"Histogram of {col}")
            st.plotly_chart(hist_fig, use_container_width=True)
            box_fig = px.box(df, y=col, title=f"Box Plot of {col}")
            st.plotly_chart(box_fig, use_container_width=True)

    # Categorical Analysis Tab
    with tab2:
        st.markdown("### Categorical Data Insights")
        for col in non_numeric_cols:
            st.markdown(f"**Frequency of {col}**")
            freq_table = df[col].value_counts().reset_index()
            freq_table.columns = ['Category', 'Count']
            freq_table['Percentage'] = (freq_table['Count'] / len(df) * 100).round(2)
            st.table(freq_table.style.format({"Percentage": "{:.2f}%"}))
            if freq_table.shape[0] <= 10:
                pie_fig = px.pie(freq_table, names='Category', values='Count', title=f"Pie Chart for {col}")
                st.plotly_chart(pie_fig, use_container_width=True)
            else:
                bar_fig = px.bar(freq_table, x='Category', y='Count', title=f"Bar Chart for {col}")
                st.plotly_chart(bar_fig, use_container_width=True)

    # Advanced Stats Tab
    with tab3:
        st.markdown("### Advanced Statistics")
        st.markdown("**Missing Data Analysis**")
        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ['Column', 'Missing Values']
        missing_df['Percentage'] = (missing_df['Missing Values'] / len(df) * 100).round(2)
        st.table(missing_df.style.format({"Percentage": "{:.2f}%"}))

        st.markdown("**Correlation Matrix**")
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            heat_fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix")
            st.plotly_chart(heat_fig, use_container_width=True)
        else:
            st.info("Not enough numeric columns for correlation analysis.")

        st.markdown("**Combined Distribution Overview**")
        if numeric_cols.size:
            melted_df = df.melt(value_vars=numeric_cols, var_name="Variable", value_name="Value")
            dist_fig = px.histogram(melted_df, x="Value", color="Variable", nbins=30, title="Combined Distribution")
            st.plotly_chart(dist_fig, use_container_width=True)

def handle_query_response(response: dict, db_name: str, db_type: str, host: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None) -> None:
    """Process LLM-generated SQL query, display results, and handle visualizations."""
    try:
        query = response.get('query', '')
        error = response.get('error', '')
        decision_log = response.get('decision_log', {})
        visualization_recommendation = response.get('visualization_recommendation', None)

        if error:
            detailed_error = generate_detailed_error_message(error)
            st.error(f"Error generating SQL query: {detailed_error}")
            return

        if not query:
            st.warning("No query generated. Please refine your message.")
            return

        st.success("SQL Query generated successfully!")
        colored_header("SQL Query and Summary", color_name="blue-70", description="")
        st.code(query, language="sql")

        if decision_log:
            with st.expander("Decision Log", expanded=False):
                # Replace the old build_markdown_decision_log usage with our new function
                display_decision_log_widgets(decision_log)

        start_time = time()
        sql_results = get_data(query, db_name, db_type, host, user, password)
        execution_time = time() - start_time

        if sql_results.empty:
            no_result_reason = "The query executed successfully but did not match any records in the database."
            if 'no valid SQL query generated' in decision_log.get("execution_feedback",[]):
                no_result_reason = "The query was not generated due to insufficient or ambiguous input."
            elif 'SQL query validation failed' in decision_log.get("execution_feedback",[]):
                no_result_reason = "The query failed validation checks and was not executed."
            st.warning(f"The query returned no results because: {no_result_reason}")
            return

        if sql_results.columns.duplicated().any():
            st.error("The query returned a DataFrame with duplicate column names. Please modify your query to avoid this.")
            return

        for col in sql_results.select_dtypes(include=['object']):
            try:
                sql_results[col] = pd.to_datetime(sql_results[col], format='%Y-%m-%d %H:%M:%S')
            except (ValueError, TypeError):
                pass

        colored_header("Query Results and Filter", color_name="blue-70", description="")
        filtered_results = dataframe_explorer(sql_results, case=False)
        st.dataframe(filtered_results, use_container_width=True, height=600)

        colored_header("Summary Statistics", color_name="blue-70", description="")
        display_summary_statistics(filtered_results)

        performance_metrics = analyze_query_performance(
            query,
            execution_time,
            len(sql_results)
        )

        with st.expander("🔍 Query Performance Analysis", expanded=True):
            st.markdown("### Performance Metrics")

            # Display execution metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Execution Time", f"{performance_metrics['execution_time']:.3f}s")
            with col2:
                st.metric("Rows Returned", performance_metrics['rows_returned'])
            with col3:
                st.metric("Rows/Second", f"{performance_metrics['rows_per_second']:.0f}")
            with col4:
                st.metric("Performance", performance_metrics['performance_class'])

            # Display optimization suggestions
            if performance_metrics['suggestions']:
                st.markdown("### Optimization Suggestions")
                for suggestion in performance_metrics['suggestions']:
                    if suggestion['type'] == 'error':
                        st.error(suggestion['message'])
                    elif suggestion['type'] == 'warning':
                        st.warning(suggestion['message'])
                    else:
                        st.info(suggestion['message'])

        # Add a colored header to separate the Visualization section from Summary Statistics
        colored_header("Visualization Section", color_name="blue-70", description="")

        if len(filtered_results.columns) >= 2:
            with st.sidebar.expander("📊 Visualization Options", expanded=True):
                numerical_cols = filtered_results.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = filtered_results.select_dtypes(include=['object', 'category']).columns.tolist()

                # Suggest default X and Y columns
                suggested_x = numerical_cols[0] if numerical_cols else filtered_results.columns[0]
                suggested_y = numerical_cols[1] if len(numerical_cols) > 1 else (filtered_results.columns[1] if len(filtered_results.columns) > 1 else filtered_results.columns[0])

                x_options = [f"{col} ⭐" if col == suggested_x else col for col in filtered_results.columns]
                y_options = [f"{col} ⭐" if col == suggested_y else col for col in filtered_results.columns]

                x_col = st.selectbox("Select X-axis Column", options=x_options, index=x_options.index(f"{suggested_x} ⭐") if f"{suggested_x} ⭐" in x_options else 0, key="x_axis")
                y_col = st.selectbox("Select Y-axis Column", options=y_options, index=y_options.index(f"{suggested_y} ⭐") if f"{suggested_y} ⭐" in y_options else 0, key="y_axis")
                x_col_clean = x_col.replace(" ⭐", "")
                y_col_clean = y_col.replace(" ⭐", "")

                chart_type_options = ["None", "Bar Chart", "Line Chart", "Scatter Plot", "Area Chart", "Histogram", "Pie Chart", "Box Plot"]
                suggested_chart_type = visualization_recommendation if visualization_recommendation in chart_type_options else ("Bar Chart" if numerical_cols else "None")
                chart_type_display = [f"{chart} ⭐" if chart == suggested_chart_type else chart for chart in chart_type_options]

                try:
                    default_chart_index = chart_type_display.index(f"{suggested_chart_type} ⭐")
                except ValueError:
                    default_chart_index = 0

                chart_type = st.selectbox(
                    "Select Chart Type",
                    options=chart_type_display,
                    index=default_chart_index,
                    help=f"Recommended Chart Type: {suggested_chart_type}",
                    key="chart_type"
                )
                chart_type_clean = chart_type.replace(" ⭐", "")

            if chart_type_clean != "None" and x_col_clean and y_col_clean:
                chart = create_chart(filtered_results, chart_type_clean, x_col_clean, y_col_clean)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)

        export_format = st.selectbox("Select Export Format", options=["CSV", "Excel", "JSON"], key="export_format")
        export_results(filtered_results, export_format)

        if "query_history" not in st.session_state:
            st.session_state.query_history = []
            st.session_state.query_timestamps = []

        st.session_state.query_history.append(query)
        st.session_state.query_timestamps.append(pd.Timestamp.now())

    except Exception as e:
        detailed_error = generate_detailed_error_message(str(e))
        st.error(f"An unexpected error occurred: {detailed_error}")
        logger.exception(f"Unexpected error: {e}")

def export_results(sql_results: pd.DataFrame, export_format: str) -> None:
    """Allow the user to download query results in CSV, Excel, or JSON format."""
    if export_format == "CSV":
        st.download_button(
            label="📥 Download Results as CSV",
            data=sql_results.to_csv(index=False),
            file_name='query_results.csv',
            mime='text/csv'
        )
    elif export_format == "Excel":
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            sql_results.to_excel(writer, index=False, sheet_name='Sheet1')
        excel_buffer.seek(0)
        st.download_button(
            label="📥 Download Results as Excel",
            data=excel_buffer,
            file_name='query_results.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    elif export_format == "JSON":
        st.download_button(
            label="📥 Download Results as JSON",
            data=sql_results.to_json(orient='records'),
            file_name='query_results.json',
            mime='application/json'
        )
    else:
        st.error("⚠️ Selected export format is not supported.")

def analyze_dataframe_for_visualization(df: pd.DataFrame) -> list:
    """Propose suitable chart types based on numeric and categorical column analysis."""
    suggestions = set()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    logger.debug(f"Numerical Columns: {numerical_cols}")
    logger.debug(f"Categorical Columns: {categorical_cols}")

    if len(numerical_cols) == 1:
        suggestions.update(["Histogram", "Box Plot"])
    if len(categorical_cols) == 1:
        suggestions.update(["Bar Chart", "Pie Chart"])

    if len(numerical_cols) >= 2:
        suggestions.update(["Scatter Plot", "Line Chart"])
    elif len(numerical_cols) == 1 and len(categorical_cols) == 1:
        suggestions.update(["Bar Chart"])

    if len(numerical_cols) > 2:
        suggestions.add("Scatter Plot")

    time_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    if time_cols:
        suggestions.add("Line Chart")

    ordered_suggestions = [chart for chart in SUPPORTED_CHART_TYPES.keys() if chart in suggestions]
    logger.debug(f"Ordered Suggestions: {ordered_suggestions}")
    return ordered_suggestions

def generate_detailed_error_message(error_message: str) -> str:
    """Use the LLM to produce a user-friendly explanation of any encountered error."""
    prompt = f"Provide a detailed and user-friendly explanation for the following error message:\n\n{error_message}"
    detailed_error = get_completion_from_messages(SYSTEM_MESSAGE, prompt)
    return detailed_error.strip() if detailed_error else error_message

def display_decision_log_widgets(decision_log: Dict) -> None:
    """
    Display the complete decision log with enhanced visual organization and styling.
    Each section of the decision log is displayed in its own tab with appropriate formatting
    and visual hierarchy.
    """
    # Create custom CSS for better tab styling
    st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Create tabs with professional labels
    tabs = st.tabs([
        "Input Analysis",
        "Paths",
        "Ambiguities",
        "Resolution",
        "Selected Path",
        "SQL Query",
        "Alternatives",
        "Feedback",
        "Summary"
    ])

    # Input Details Tab
    with tabs[0]:
        st.markdown("### Query Input Details")
        for detail in decision_log.get("query_input_details", []):
            st.info(detail)

        if preprocessing_steps := decision_log.get("preprocessing_steps"):
            st.markdown("### Preprocessing Steps")
            for step in preprocessing_steps:
                st.markdown(f"```\n{step}\n```")

    # Paths Tab
    with tabs[1]:
        st.markdown("### Path Identification")
        for i, path in enumerate(decision_log.get("path_identification", []), 1):
            with st.expander(f"Path {i} (Score: {path['score']})", expanded=i == 1):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Description**:")
                    st.markdown(f"_{path['description']}_")
                with col2:
                    st.metric("Score", path['score'])

                st.divider()

                col3, col4 = st.columns(2)
                with col3:
                    st.markdown("**Tables**")
                    for table in path['tables']:
                        st.markdown(f"- `{table}`")
                with col4:
                    st.markdown("**Columns**")
                    for cols in path['columns']:
                        st.markdown(f"- `{', '.join(cols)}`")

    # Ambiguities Tab
    with tabs[2]:
        st.markdown("### Ambiguity Analysis")
        if ambiguities := decision_log.get("ambiguity_detection"):
            for ambiguity in ambiguities:
                st.warning(ambiguity)
        else:
            st.success("No ambiguities detected")

    # Resolution Tab
    with tabs[3]:
        st.markdown("### Resolution Criteria")
        if criteria := decision_log.get("resolution_criteria"):
            for i, criterion in enumerate(criteria, 1):
                st.markdown(f"**{i}.** {criterion}")
                st.divider()
        else:
            st.info("No resolution criteria specified")

    # Chosen Path Tab
    with tabs[4]:
        st.markdown("### Selected Tables and Columns")
        if chosen_path := decision_log.get("chosen_path_explanation"):
            for item in chosen_path:
                with st.expander(f"{item['table']}", expanded=True):
                    st.markdown("#### Selected Columns:")
                    cols = st.columns(min(3, len(item['columns'])))
                    for i, col in enumerate(item['columns']):
                        with cols[i % len(cols)]:
                            st.code(col)

                    st.markdown("#### Selection Rationale:")
                    st.info(item['reason'])
        else:
            st.warning("No path selection details available")

    # SQL Query Tab
    with tabs[5]:
        st.markdown("### Generated SQL Query")
        if sql_query := decision_log.get("generated_sql_query"):
            st.code(sql_query, language="sql")
            if st.button("Copy Query"):
                st.write("Query copied to clipboard!")
                st.session_state['clipboard'] = sql_query
        else:
            st.error("No SQL query generated")

    # Alternatives Tab
    with tabs[6]:
        st.markdown("### Alternative Approaches")
        if alternatives := decision_log.get("alternative_paths"):
            for i, alt in enumerate(alternatives, 1):
                with st.expander(f"Alternative {i}", expanded=False):
                    st.markdown(alt)
        else:
            st.info("No alternative approaches identified")

    # Feedback Tab
    with tabs[7]:
        st.markdown("### Execution Feedback")
        if feedback := decision_log.get("execution_feedback"):
            for item in feedback:
                if "error" in item.lower():
                    st.error(item)
                elif "warning" in item.lower():
                    st.warning(item)
                else:
                    st.success(item)
        else:
            st.info("No execution feedback available")

    # Summary Tab
    with tabs[8]:
        st.markdown("### Analysis Summary")
        if summary := decision_log.get("final_summary"):
            st.markdown(f"**Key Findings:**")
            st.markdown(f"_{summary}_")

            if viz_suggestion := decision_log.get("visualization_suggestion"):
                st.divider()
                st.markdown("### Visualization Recommendation")
                st.success(f"Suggested visualization type: **{viz_suggestion}**")
                st.markdown("_This chart type was selected based on the data structure and analysis goals._")

def analyze_query_performance(query: str, execution_time: float, row_count: int) -> dict:
    """Analyze query performance and suggest optimizations."""
    performance_metrics = {
        "execution_time": execution_time,
        "rows_returned": row_count,
        "rows_per_second": row_count / execution_time if execution_time > 0 else 0,
        "suggestions": []
    }

    # Basic query analysis
    query_lower = query.lower()

    # Check for SELECT *
    if "select *" in query_lower:
        performance_metrics["suggestions"].append({
            "type": "warning",
            "message": "Consider specifying required columns instead of SELECT * to improve performance"
        })

    # Check for missing WHERE clause
    if "where" not in query_lower:
        performance_metrics["suggestions"].append({
            "type": "info",
            "message": "Query has no WHERE clause - consider adding filters if large data set"
        })

    # Check for potential cartesian products
    if query_lower.count("join") > query_lower.count("on"):
        performance_metrics["suggestions"].append({
            "type": "error",
            "message": "Possible cartesian product detected - ensure proper JOIN conditions"
        })

    # Performance classification
    if execution_time < 0.1:
        performance_metrics["performance_class"] = "Excellent"
    elif execution_time < 0.5:
        performance_metrics["performance_class"] = "Good"
    elif execution_time < 2.0:
        performance_metrics["performance_class"] = "Fair"
    else:
        performance_metrics["performance_class"] = "Poor"
        performance_metrics["suggestions"].append({
            "type": "warning",
            "message": f"Query execution time ({execution_time:.2f}s) is high"
        })

    return performance_metrics

# Database Setup
db_type = st.sidebar.selectbox("Select Database Type 🗄️", options=["SQLite", "PostgreSQL"])

if db_type == "SQLite":
    uploaded_file = st.sidebar.file_uploader("Upload SQLite Database 📂", type=["db", "sqlite", "sql"])

    if uploaded_file:
        db_file = save_temp_file(uploaded_file)
        schemas = DB_Config.get_all_schemas(db_file, db_type='sqlite')
        table_names = list(schemas.keys())

        if not schemas:
            st.error("Could not load any schemas please check the database file")

        if table_names:
            options = ["Select All"] + table_names
            selected_tables = st.sidebar.multiselect("Select Tables 📋", options=options, key="sqlite_tables")
            if "Select All" in selected_tables:
                if len(selected_tables) < len(options):
                    selected_tables = table_names
                else:
                    selected_tables = options
            selected_tables = [table for table in selected_tables if table != "Select All"]
            colored_header(f"🔍 Selected Tables: {', '.join(selected_tables)}", color_name="blue-70", description="")
            if len(selected_tables) > 3:

                    with st.expander("View All Table Schemas 📖", expanded=False):
                        for table in selected_tables:
                            with st.expander(f"Schema: {table}", expanded=False):
                                st.json(schemas[table])
            else:
                for table in selected_tables:
                    with st.expander(f"View Schema: {table} 📖", expanded=False):
                        st.json(schemas[table])

            user_message = st.text_input(placeholder="Type your SQL query here...", key="user_message", label="Your Query 💬", label_visibility="hidden")
            if user_message:
                selected_schemas = {table: schemas[table] for table in selected_tables}
                logger.debug(f"Schemas being passed to `generate_sql_query`: {selected_schemas}")
                with st.spinner('🧠 Generating SQL query...'):
                    response = generate_sql_query(user_message, selected_schemas)
                handle_query_response(response, db_file, db_type='sqlite')

        else:
            st.info("📭 No tables found in the database.")
    else:
        st.info("📥 Please upload a database file to start.")

elif db_type == "PostgreSQL":
    with st.sidebar.expander("🔐 PostgreSQL Connection Details", expanded=True):
        postgres_host = st.text_input("Host 🏠", placeholder="PostgreSQL Host")
        postgres_db = st.text_input("DB Name 🗄️", placeholder="Database Name")
        postgres_user = st.text_input("Username 👤", placeholder="Username")
        postgres_password = st.text_input("Password 🔑", type="password", placeholder="Password")

    if all([postgres_host, postgres_db, postgres_user, postgres_password]):
        schemas = DB_Config.get_all_schemas(postgres_db, db_type='postgresql', host=postgres_host, user=postgres_user, password=postgres_password)
        table_names = list(schemas.keys())

        if table_names:
            options = ["Select All"] + table_names
            selected_tables = st.sidebar.multiselect("Select Tables 📋", options=options, key="postgresql_tables")
            if "Select All" in selected_tables:
                if len(selected_tables) < len(options):
                    selected_tables = table_names
                else:
                    selected_tables = options
            selected_tables = [table for table in selected_tables if table != "Select All"]
            colored_header("🔍 Selected Tables:", color_name="blue-70", description="")
            for table in selected_tables:
                with st.expander(f"View Schema: {table} 📖", expanded=False):
                    st.json(schemas[table])

            user_message = st.text_input(placeholder="Type your SQL query here...", key="user_message_pg", label="Your Query 💬", label_visibility="hidden")
            if user_message:
                with st.spinner('🧠 Generating SQL query...'):
                    selected_schemas = {table: schemas[table] for table in selected_tables}
                    logger.debug(f"Schemas being passed to `generate_sql_query`: {selected_schemas}")
                    response = generate_sql_query(user_message, selected_schemas)
                handle_query_response(response, postgres_db, db_type='postgresql', host=postgres_host, user=postgres_user, password=postgres_password)
        else:
            st.info("📭 No tables found in the database.")
    else:
        st.info("🔒 Please fill in all PostgreSQL connection details to start.")

# Query history
with st.sidebar.expander(" Query History", expanded=False):
    if st.session_state.get("query_history"):
        st.write("### 📝 Saved Queries")

        search_query = st.text_input("Search Queries 🔍", key="search_query")
        query_history_df = pd.DataFrame({
            "Query": st.session_state.query_history,
            "Timestamp": pd.to_datetime(st.session_state.query_timestamps)
        })

        if search_query:
            query_history_df = query_history_df[query_history_df['Query'].str.contains(search_query, case=False, na=False)]

        queries_per_page = 5
        total_queries = len(query_history_df)
        num_pages = max((total_queries // queries_per_page) + (total_queries % queries_per_page > 0), 1)
        current_page = st.number_input("Page 📄", min_value=1, max_value=num_pages, value=1)

        start_index = (current_page - 1) * queries_per_page
        end_index = start_index + queries_per_page
        page_queries = query_history_df.iloc[start_index:end_index]

        for i, row in page_queries.iterrows():
            with st.expander(f"🗂️ Query {i + 1}: {row['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"):
                st.write("**SQL Query:**")
                st.code(row['Query'], language="sql")

                if st.button(f"🔄 Re-run Query {i + 1}", key=f"rerun_query_{i}"):
                    user_message = row['Query']
                    with st.spinner('🔄 Re-running the saved SQL query...'):
                        selected_schemas = {table: schemas[table] for table in selected_tables}
                        response = generate_sql_query(user_message, selected_schemas)
                        handle_query_response(
                            response,
                            db_file if db_type == "SQLite" else postgres_db,
                            db_type,
                            host=postgres_host if db_type == "PostgreSQL" else None,
                            user=postgres_user if db_type == "PostgreSQL" else None,
                            password=postgres_password if db_type == "PostgreSQL" else None
                        )

        st.write(f"Page {current_page} of {num_pages}")

    else:
        st.info("📭 No query history available.")
