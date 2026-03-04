import os
import pandas as pd
import json
import difflib
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import io

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="Radiance Sales Analytics")

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_DATA_PATH = os.path.join(DATA_DIR, "sales_dataset.csv")

# Create dirs if they don't exist
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "css"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "js"), exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Global in-memory storage for the current dataset
# In a real app, this would be tied to a user session or DB
app.state.current_df = None

def load_default_data():
    if os.path.exists(DEFAULT_DATA_PATH):
        try:
            app.state.current_df = pd.read_csv(DEFAULT_DATA_PATH)
        except Exception as e:
            print(f"Error loading default data: {e}")

load_default_data()

# --- Page Routes ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    return templates.TemplateResponse("analytics.html", {"request": request})

# --- API Endpoints ---

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        return JSONResponse(status_code=400, content={"error": "Only CSV files are allowed."})
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        app.state.current_df = df
        return {"message": "File uploaded successfully", "filename": file.filename, "rows": len(df)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/load-sample")
async def load_sample():
    try:
        if os.path.exists(DEFAULT_DATA_PATH):
            app.state.current_df = pd.read_csv(DEFAULT_DATA_PATH)
            return {"message": "Sample data loaded successfully", "filename": "sales_dataset.csv", "rows": len(app.state.current_df)}
        else:
            return JSONResponse(status_code=404, content={"error": "Sample data file not found."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/analytics-data")
async def get_analytics_data():
    df = app.state.current_df
    if df is None or df.empty:
        return JSONResponse(status_code=404, content={"error": "No data available."})
    
    try:
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        metrics = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
        }
        
        # Calculate sums and averages for key numeric columns
        for col in numeric_cols:
            metrics[f"total_{col.lower()}"] = float(df[col].sum())
            metrics[f"avg_{col.lower()}"] = float(df[col].mean())
            metrics[f"max_{col.lower()}"] = float(df[col].max())
            metrics[f"min_{col.lower()}"] = float(df[col].min())

        # Generate chart data
        charts = []
        
        # 1. Bar Chart Example: First categorical column vs First numeric column
        if categorical_cols and numeric_cols:
            cat_col = categorical_cols[0]
            # Try to find a 'Sales' or similar column, else use first numeric
            num_col = next((c for c in numeric_cols if 'sale' in c.lower() or 'revenue' in c.lower()), numeric_cols[0])
            
            # Group by and aggregate
            grouped = df.groupby(cat_col)[num_col].sum().reset_index()
            # Sort for better visualization
            grouped = grouped.sort_values(by=num_col, ascending=False).head(10)
            
            charts.append({
                "id": "chart_1",
                "type": "bar",
                "title": f"Total {num_col} by {cat_col}",
                "labels": grouped[cat_col].tolist(),
                "values": grouped[num_col].tolist()
            })

            # 2. Pie Chart Example: Another categorical or same one if limited
            pie_col = categorical_cols[1] if len(categorical_cols) > 1 else categorical_cols[0]
            pie_grouped = df.groupby(pie_col).size().reset_index(name='count')
            pie_grouped = pie_grouped.sort_values(by='count', ascending=False)
            if len(pie_grouped) > 10:
                top_10 = pie_grouped.head(10)
                other_sum = pie_grouped['count'][10:].sum()
                other_df = pd.DataFrame({pie_col: ['Other'], 'count': [other_sum]})
                pie_grouped = pd.concat([top_10, other_df], ignore_index=True)

            charts.append({
                "id": "chart_2",
                "type": "pie",
                "title": f"Distribution by {pie_col}",
                "labels": pie_grouped[pie_col].tolist(),
                "values": [int(x) for x in pie_grouped['count'].tolist()]
            })

            # 3. Line Chart Example: Look for a Date column
            date_col = next((c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()), None)
            if date_col:
                # Group by date
                time_grouped = df.groupby(date_col)[num_col].sum().reset_index()
                # Sort by date
                time_grouped = time_grouped.sort_values(by=date_col)
                # Limit to latest 100 points
                if len(time_grouped) > 100:
                    time_grouped = time_grouped.tail(100)
                charts.append({
                    "id": "chart_3",
                    "type": "line",
                    "title": f"{num_col} Trend over Time",
                    "labels": time_grouped[date_col].tolist(),
                    "values": time_grouped[num_col].tolist()
                })

        return {"metrics": metrics, "charts": charts}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/ask")
async def ask_question(question: str = Form(...)):
    df = app.state.current_df
    if df is None or df.empty:
        return {"answer": "I don't have any data loaded to answer questions about."}
        
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"answer": "Gemini API Key is not configured. Please set GEMINI_API_KEY environment variable before running the server."}
        
    try:
        # Construct dataset metadata context
        columns_info = {col: str(df[col].dtype) for col in df.columns}
        
        prompt = f"""
        You are an AI data assistant. The user is asking a question about a pandas DataFrame.
        DataFrame Columns and data types: {columns_info}
        User Question: "{question}"
        
        Analyze the question and the dataframe schema. Your goal is to map the user's intent to a specific column and a mathematical operation. 
        Supported operations: sum, mean, max, min, count, unique_count. 
        If the question is about rows in the dataset, return column as "ROW_COUNT" and operation as "count".
        
        Return ONLY a JSON object (no markdown, no backticks, no markdown codeblocks) in the exact format:
        {{"column": "Exact_Column_Name", "operation": "operation_name"}}
        
        If you cannot determine a column or operation, return:
        {{"error": "Could not understand the question related to this dataset."}}
        """
        
        # Use standard flash for fast responses and stability
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        
        res_text = response.text.strip()
        if res_text.startswith("```json"):
            res_text = res_text[7:]
        if res_text.startswith("```"):
            res_text = res_text[3:]
        if res_text.endswith("```"):
            res_text = res_text[:-3]
        res_text = res_text.strip()
            
        action = json.loads(res_text)
        
        if "error" in action:
            return {"answer": action["error"]}
            
        col = action.get("column")
        op = action.get("operation")
        
        if col == "ROW_COUNT":
            return {"answer": f"There are {len(df):,} rows in the dataset."}
            
        if col not in df.columns:
            return {"answer": f"I understood you wanted to check column '{col}', but it's not in the dataset."}
            
        val = None
        if op == "sum":
            val = df[col].sum()
        elif op == "mean":
            val = df[col].mean()
        elif op == "max":
            val = df[col].max()
        elif op == "min":
            val = df[col].min()
        elif op == "count":
            val = df[col].count()
        elif op == "unique_count":
            val = df[col].nunique()
            return {"answer": f"The number of unique values in {col} is {val:,.0f}."}
        else:
            return {"answer": f"Unsupported operation '{op}' on column '{col}'."}
            
        if isinstance(val, (float, int)):
            if op in ["count", "unique_count"]:
                return {"answer": f"The {op} of {col} is {val:,.0f}."}
            return {"answer": f"The {op} of {col} is {val:,.2f}."}
        else:
            return {"answer": f"The {op} of {col} is {val}."}
            
    except Exception as e:
        error_msg = str(e)
        if "404 models" in error_msg:
            return {"answer": "Sorry, your Gemini API Key is invalid or not authorized for the Gemini API. Please check your Google AI Studio dashboard."}
        if "429 You exceeded" in error_msg:
            return {"answer": "You have exceeded your Gemini API quote/rate limits. Please check your Google AI Studio billing details."}
        return {"answer": f"Sorry, I encountered an error communicating with the AI: {error_msg}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
