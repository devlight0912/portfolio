# NLP2SQL

NLP2SQL is a project that aims to generate SQL queries from natural language text. The system converts natural language inputs into semantically correct SQL queries, enabling dynamic database handling, data visualization, and interactive querying.

## 🖼️ Diagram

<p align="center">
    <img src="NLP2SQL.png" alt="NLP2SQL Diagram" width="600px">
</p>

## 🧐 Features

Here are some of the project's key features:

- **Natural Language to SQL**: Convert text queries into SQL commands.
- **Multi-Database Support**: Supports both SQLite and PostgreSQL.
- **Interactive Data Explorer**: Utilize data explorer for filtering and analyzing query results.
- **Dynamic Schema Representation**: Fetch and display database schemas dynamically.
- **Custom Visualizations**: Generate charts and graphs (Bar, Line, Scatter, Area, Histogram).
- **Export Options**: Export query results in CSV or Parquet formats.
- **Query History**: Save and revisit previous queries for future use.

## 🛠️ Installation Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/lohitkolluri/NLP2SQL.git
   cd NLP2SQL
   ```

2. **Create and Activate a Virtual Environment:**

   - On macOS/Linux:

     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

   - On Windows:

     ```bash
     python -m venv venv
     .\venv\Scripts\activate
     ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables:**

   Create a `.env` file in the root directory and add the necessary environment variables for Azure OpenAI API:

   ```env
   OPENAI_ENDPOINT=https://your-azure-openai-endpoint
   OPENAI_API_KEY=your-azure-openai-api-key
   ```

5. **Running the App:**

   ```bash
   streamlit run NLP2SQL.py
   ```

6. **Access the App:**

   Open your web browser and go to:

   ```
   http://localhost:8501
   ```

## 💻 Built With

Technologies used in the project:

- **Streamlit**: For building the interactive web application.
- **Altair**: For creating data visualizations.
- **Pandas**: For data manipulation.
- **SQLite & PostgreSQL**: For database support.
- **Azure OpenAI**: For converting natural language to SQL queries.
- **Streamlit Extras**: Enhancing UI components like data explorer and chart containers.
- **GPT-3.5 Turbo**: Language model used for query generation.

## 🚀 Additional Features

- **Custom CSS Styling**: Improved the interface aesthetics with custom CSS.
- **Schema Explorer**: View database schema before query execution.
- **Safe SQL Execution**: Validation to prevent execution of harmful SQL commands like `DROP`, `DELETE`, or `ALTER`.
- **Database File Upload**: Upload SQLite databases directly into the app for querying.
- **Collapsible Query History**: Re-run or review previous queries via a collapsible section.
