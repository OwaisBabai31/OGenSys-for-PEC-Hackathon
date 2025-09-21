# OGenSys - Executive Analytics for Admissions
# Hugging Face Spaces Deployment Version

import warnings, sys, os, datetime, tempfile, time
warnings.filterwarnings("ignore")

import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from typing import List, Tuple, Dict, Any
import requests

# --------------------------
# Initialize API Clients
# --------------------------
groq_client = None
gemini_client = None
GROQ_API_KEY = None
GEMINI_API_KEY = None

def get_hf_secret(secret_name):
    """Get secrets from Hugging Face Spaces environment variables"""
    try:
        secret = os.environ.get(secret_name)
        if secret and secret.strip():
            print(f"✅ Found secret '{secret_name}' from environment")
            return secret.strip()
        else:
            print(f"❌ Secret '{secret_name}' not found in environment")
            return None
    except Exception as e:
        print(f"❌ Error getting secret '{secret_name}': {e}")
        return None

def initialize_apis():
    """Initialize both Groq and Gemini APIs for Hugging Face deployment"""
    global groq_client, gemini_client, GROQ_API_KEY, GEMINI_API_KEY

    print("🔄 Initializing AI APIs for Hugging Face...")

    # Initialize Groq
    GROQ_API_KEY = get_hf_secret('GROQ_API_KEY')
    if GROQ_API_KEY:
        try:
            from groq import Groq
            groq_client = Groq(api_key=GROQ_API_KEY)
            # Test with a simple call
            test_response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": "Test"}],
                model="llama-3.1-8b-instant",
                max_tokens=5
            )
            print("✅ Groq API initialized and tested successfully")
        except Exception as e:
            print(f"❌ Groq API initialization failed: {e}")
            groq_client = None
            # Try direct HTTP approach
            try:
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "messages": [{"role": "user", "content": "Test"}],
                        "model": "llama-3.1-8b-instant",
                        "max_tokens": 5
                    }
                )
                if response.status_code == 200:
                    print("✅ Groq API accessible via direct HTTP")
                    groq_client = "DIRECT_HTTP"
                else:
                    print(f"❌ Direct HTTP also failed: {response.status_code}")
            except Exception as http_error:
                print(f"❌ Direct HTTP failed: {http_error}")

    # Initialize Gemini as backup
    GEMINI_API_KEY = get_hf_secret('GEMINI_API_KEY')
    if GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_client = genai.GenerativeModel('gemini-1.5-flash')
            # Test Gemini
            test_response = gemini_client.generate_content("Test")
            print("✅ Gemini API initialized and tested successfully")
        except Exception as e:
            print(f"❌ Gemini API initialization failed: {e}")
            gemini_client = None

    # Final status
    if groq_client:
        print("🎯 Primary AI: Groq")
    elif gemini_client:
        print("🎯 Primary AI: Gemini (backup)")
    else:
        print("⚠️ No AI APIs available - limited functionality")

def call_ai_api(messages, use_gemini=False):
    """Universal AI API caller with fallback and better error handling"""
    global groq_client, gemini_client, GROQ_API_KEY

    try:
        # Try Groq first (unless specifically requesting Gemini)
        if not use_gemini and groq_client:
            try:
                # Handle both official client and direct HTTP
                if groq_client == "DIRECT_HTTP":
                    response = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {GROQ_API_KEY}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "messages": messages,
                            "model": "llama-3.1-8b-instant",
                            "temperature": 0.3,
                            "max_tokens": 1024
                        }
                    )
                    if response.status_code == 200:
                        return response.json()["choices"][0]["message"]["content"]
                    else:
                        raise Exception(f"HTTP {response.status_code}: {response.text}")
                else:
                    # Use official Groq client
                    response = groq_client.chat.completions.create(
                        messages=messages,
                        model="llama-3.1-8b-instant",
                        temperature=0.3,
                        max_tokens=1024
                    )
                    return response.choices[0].message.content

            except Exception as groq_error:
                print(f"Groq failed: {groq_error}, trying Gemini...")
                use_gemini = True

        # Use Gemini as fallback or primary
        if use_gemini and gemini_client:
            # Convert messages to Gemini format
            prompt = "\n\n".join([f"**{msg['role'].title()}:** {msg['content']}" for msg in messages])
            response = gemini_client.generate_content(prompt)
            return response.text

        # If no APIs work
        if not groq_client and not gemini_client:
            return """❌ **No AI APIs Available**

To fix this issue, please ensure your API keys are properly configured in Hugging Face Spaces secrets:

1. **Add GROQ_API_KEY** to your Space secrets
2. **Add GEMINI_API_KEY** to your Space secrets (optional backup)
3. **Restart your Space**

The system will work with either API key configured."""

        return "❌ AI analysis temporarily unavailable. Please check API configuration."

    except Exception as e:
        return f"❌ AI API error: {str(e)}"

# Initialize APIs on startup
initialize_apis()

# --------------------------
# Globals - FIXED to store consolidated data separately
# --------------------------
global_df: pd.DataFrame = None
consolidated_df: pd.DataFrame = None  # NEW: Store consolidated data separately
duplicate_pairs: List[Tuple[int,int]] = []
degree_dup_mask = None
implicated_pairs: List[Tuple[int,int,str]] = []
highlight_indices: set = set()
last_pre_swap_path: str = None
last_swap_audit: List[dict] = []
cleaned_file_path: str = None
swapped_file_path: str = None
choices_file_path: str = None
consolidated_file_path: str = None

CHECK_COL_INDICES = [7,8,9,12,13,14,15,16,17]
PREF_NUM_COL = 10
PREF_NAME_COL = 11
COL_H_IDX = 7

FIRST_DEGREE_ALLOWED = {"Matric","O-Level","O Level","OLevel","O'Level","Olevel"}
SECOND_DEGREE_ALLOWED = {"Intermediate","A-Level","A Level","DAE","Alevel","ALevel"}

# --------------------------
# Merit List Generation Functions - FIXED
# --------------------------
def process_test_marks_with_consolidated_display(test_file, consolidated_data):
    """Process test marks and display consolidated data with test scores first"""
    if test_file is None or consolidated_data is None:
        return None, None, "Please upload test marks file and ensure consolidated data is available."

    try:
        # Read test marks file
        test_df = pd.read_excel(test_file.name).fillna("")

        # Validate test marks file structure
        if len(test_df.columns) < 2:
            return None, None, "Test marks file must have at least 2 columns: Reference No and Test Marks"

        # Rename columns for consistency
        test_df.columns = ['Reference No', 'Test Marks'] + list(test_df.columns[2:])
        test_df['Reference No'] = test_df['Reference No'].astype(str).str.strip()

        # Create a copy of consolidated data with test scores
        display_df = consolidated_data.copy()
        
        # Add test score column
        test_scores = []
        for idx, row in display_df.iterrows():
            ref_no = str(row['Reference No']).strip()
            test_match = test_df[test_df['Reference No'] == ref_no]
            test_score = "NA" if test_match.empty else test_match.iloc[0]['Test Marks']
            test_scores.append(test_score)
        
        display_df['Test Score'] = test_scores
        
        # Move Test Score column to the end for better visibility
        cols = [col for col in display_df.columns if col != 'Test Score'] + ['Test Score']
        display_df = display_df[cols]

        return display_df, test_df, f"Successfully processed {len(display_df)} candidates with test marks."

    except Exception as e:
        return None, None, f"Error processing test marks: {str(e)}"

def generate_merit_list_data(consolidated_data, test_marks_df):
    """Generate merit list data from consolidated data and test marks"""
    merit_data = []
    serial_no = 1

    for idx, row in consolidated_data.iterrows():
        ref_no = str(row['Reference No']).strip()

        # Find matching test score
        test_match = test_marks_df[test_marks_df['Reference No'] == ref_no]
        test_score = "NA" if test_match.empty else test_match.iloc[0]['Test Marks']

        # Calculate test percentage
        test_percent = "NA"
        if test_score != "NA" and pd.notna(test_score):
            try:
                test_percent = (float(test_score) / 50) * 100
            except:
                test_percent = "NA"

        # Extract grade percentages from consolidated data
        tenth_percent = "NA"
        twelfth_percent = "NA"

        # Find 10th grade percentage (look for columns containing "10th" and "Percentage")
        tenth_cols = [col for col in consolidated_data.columns if '10th' in col and 'Percentage' in col]
        if tenth_cols:
            tenth_val = row[tenth_cols[0]]
            if pd.notna(tenth_val) and str(tenth_val).strip() not in ["", "nan"]:
                try:
                    tenth_percent = float(str(tenth_val).replace('%', ''))
                except:
                    tenth_percent = "NA"

        # Find 12th grade percentage
        twelfth_cols = [col for col in consolidated_data.columns if '12th' in col and 'Percentage' in col]
        if twelfth_cols:
            twelfth_val = row[twelfth_cols[0]]
            if pd.notna(twelfth_val) and str(twelfth_val).strip() not in ["", "nan"]:
                try:
                    twelfth_percent = float(str(twelfth_val).replace('%', ''))
                except:
                    twelfth_percent = "NA"

        # Calculate total score using the formula
        total_score = "NA"
        if all(x != "NA" for x in [tenth_percent, twelfth_percent, test_percent]):
            try:
                total_score = (0.1 * tenth_percent) + (0.4 * twelfth_percent) + (0.5 * test_percent)
                total_score = round(total_score, 2)
            except:
                total_score = "NA"

        # Extract program choices for dropdown
        choice_cols = [col for col in consolidated_data.columns if 'Choice' in col and col != 'Count of Preferences']
        choices = []
        for choice_col in choice_cols:
            choice_val = str(row[choice_col]).strip()
            if choice_val and choice_val not in ["nan", "", "NA"]:
                choices.append(choice_val)

        # Extract other required fields from consolidated data
        applicant_name = row.get('Applicant Name', 'N/A')
        
        # Get first choice as Program Applied
        program_applied = choices[0] if choices else "N/A"

        # Look for Mode of Test in consolidated data
        mode_test = "N/A"
        mode_cols = [col for col in consolidated_data.columns if 'mode' in col.lower() or ('test' in col.lower() and 'score' not in col.lower())]
        if mode_cols:
            mode_test = str(row[mode_cols[0]]) if pd.notna(row[mode_cols[0]]) else "N/A"

        # Look for 12th Grade details in consolidated data
        result_status = "N/A"
        grade_group = "N/A" 
        grade_board = "N/A"

        result_cols = [col for col in consolidated_data.columns if '12th' in col and ('result' in col.lower() or 'status' in col.lower())]
        if result_cols:
            result_status = str(row[result_cols[0]]) if pd.notna(row[result_cols[0]]) else "N/A"

        group_cols = [col for col in consolidated_data.columns if '12th' in col and 'group' in col.lower()]
        if group_cols:
            grade_group = str(row[group_cols[0]]) if pd.notna(row[group_cols[0]]) else "N/A"

        board_cols = [col for col in consolidated_data.columns if '12th' in col and 'board' in col.lower()]
        if board_cols:
            grade_board = str(row[board_cols[0]]) if pd.notna(row[board_cols[0]]) else "N/A"

        merit_data.append({
            'Serial Number': serial_no,
            'Reference No': ref_no,
            'Applicant Name': applicant_name,
            'Program Applied': program_applied,
            'Mode of Test': mode_test,
            'Test Score': test_score,
            'Test %': test_percent,
            '10th Grade/Equivalent %': tenth_percent,
            '12th Grade/Equivalent %': twelfth_percent,
            'Total Score': total_score,
            'Admission Decision': "",  # Empty by default
            '12th Grade Result Status': result_status,
            '12th Grade Group': grade_group,
            '12th Grade Board': grade_board,
            'Available Choices': choices  # Store for dropdown
        })
        serial_no += 1

    return pd.DataFrame(merit_data)

def create_merit_table_html(merit_df):
    """Create interactive merit list table with filtering and color coding"""
    if merit_df is None or merit_df.empty:
        return "<p>No data to display</p>"

    # Create HTML table with color coding for test scores
    html_rows = []

    # Header row
    header = "<tr>"
    for col in merit_df.columns:
        if col != 'Available Choices':  # Don't show this helper column
            header += f'<th style="background-color: #f8f9fa; position: sticky; top: 0; z-index: 10; padding: 12px; border: 1px solid #dee2e6; font-weight: bold; text-align: center;">{col}</th>'
    header += "</tr>"

    # Data rows
    for idx, row in merit_df.iterrows():
        html_row = "<tr>"
        for col in merit_df.columns:
            if col == 'Available Choices':
                continue

            cell_value = row[col]
            cell_style = "padding: 10px; border: 1px solid #dee2e6; text-align: center;"

            # Color coding for test scores
            if col == 'Test Score' and cell_value != "NA":
                try:
                    score = float(cell_value)
                    if score < 25:
                        cell_style += " background-color: #ffcccc; color: #d32f2f; font-weight: bold;"  # Red
                    else:
                        cell_style += " background-color: #c8e6c9; color: #2e7d32; font-weight: bold;"  # Green
                except:
                    pass

            # Create dropdown for Admission Decision
            if col == 'Admission Decision':
                choices = row['Available Choices'] if isinstance(row['Available Choices'], list) else []
                options = choices + ['Admission Rejected']
                dropdown_options = "".join([f'<option value="{opt}">{opt}</option>' for opt in options])
                cell_content = f'<select onchange="updateAdmissionDecision(\'{row["Reference No"]}\', this.value)" style="width: 100%; padding: 6px; border: 1px solid #ccc; border-radius: 4px;"><option value="">Select...</option>{dropdown_options}</select>'
            else:
                cell_content = str(cell_value)

            html_row += f'<td style="{cell_style}">{cell_content}</td>'
        html_row += "</tr>"
        html_rows.append(html_row)

    # Complete table HTML with enhanced styling
    table_html = f"""
    <div style="overflow: auto; max-height: 70vh; max-width: 100%; border: 1px solid #ddd; border-radius: 8px;">
        <style>
            .merit-table {{
                border-collapse: collapse;
                width: 100%;
                font-size: 13px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }}
            .merit-table th {{
                position: sticky;
                top: 0;
                background-color: #f8f9fa;
                z-index: 10;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .merit-table tr:hover {{
                background-color: #f5f5f5;
            }}
        </style>
        <table class="merit-table">
            <thead>{header}</thead>
            <tbody>{"".join(html_rows)}</tbody>
        </table>
    </div>

    <script>
        function updateAdmissionDecision(refNo, decision) {{
            console.log('Updated admission for ' + refNo + ' to ' + decision);
            // Trigger seat count update
            updateSeatCounts();
        }}
        
        function updateSeatCounts() {{
            // This would update seat counts in real-time
            // Implementation would depend on your framework
        }}
    </script>
    """

    return table_html

def get_program_statistics(merit_df, consolidated_data):
    """Generate program-wise statistics using consolidated data for total count"""
    if merit_df is None or merit_df.empty or consolidated_data is None:
        return "No data available"

    # Get total candidate count from consolidated data (correct count)
    total_candidates = len(consolidated_data)
    
    # Get unique programs from all choices in consolidated data
    programs = {}
    choice_cols = [col for col in consolidated_data.columns if 'Choice' in col and col != 'Count of Preferences']
    
    for _, row in consolidated_data.iterrows():
        for choice_col in choice_cols:
            program = str(row[choice_col]).strip()
            if program and program not in ["nan", "", "NA"]:
                programs[program] = programs.get(program, 0) + 1

    # Count filled seats by program from merit list
    filled_seats = {}
    for program in programs.keys():
        program_candidates = merit_df[merit_df['Program Applied'] == program]
        filled_count = len(program_candidates[
            (program_candidates['Admission Decision'] != '') &
            (program_candidates['Admission Decision'] != 'Admission Rejected')
        ])
        filled_seats[program] = filled_count

    # Create statistics display with corrected total
    stats_html = f"""
    <div style='background-color: #e3f2fd; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
        <h3 style='margin: 0; color: #1565c0;'>📊 Overall Statistics</h3>
        <p style='margin: 10px 0 0 0; font-size: 18px; font-weight: bold; color: #333;'>
            Total Candidates: <span style='color: #2e7d32;'>{total_candidates}</span>
        </p>
    </div>
    
    <div style='display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 20px;'>
    """

    # Sort programs by popularity (most applications first)
    sorted_programs = sorted(programs.items(), key=lambda x: x[1], reverse=True)
    
    for program, total_applications in sorted_programs:
        filled = filled_seats.get(program, 0)
        stats_html += f"""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #007bff; min-width: 250px; flex: 1;'>
            <div style='font-weight: bold; color: #333; margin-bottom: 5px; font-size: 16px;'>{program}</div>
            <div style='color: #666; font-size: 14px; margin-bottom: 3px;'>Applications: <span style='font-weight: bold;'>{total_applications}</span></div>
            <div style='color: #666; font-size: 14px;'>Seats Filled: <span style='color: #28a745; font-weight: bold; font-size: 16px;'>{filled}</span></div>
        </div>
        """

    stats_html += "</div>"
    return stats_html

def generate_merit_list_from_files(test_marks_file):
    """Main function to generate merit list from consolidated data and test marks"""
    global consolidated_df

    if consolidated_df is None:
        return "Please process data in the Data Processing tab first.", "", ""

    if test_marks_file is None:
        return "Please upload test marks file to generate merit list.", "", ""

    try:
        # Process test marks with consolidated data
        display_with_scores, test_marks_df, message = process_test_marks_with_consolidated_display(
            test_marks_file, consolidated_df
        )
        
        if display_with_scores is None:
            return "", "", message

        # Generate merit list data
        merit_df = generate_merit_list_data(consolidated_df, test_marks_df)
        
        if merit_df is not None and not merit_df.empty:
            # Create table HTML
            table_html = create_merit_table_html(merit_df)
            
            # Create statistics HTML using consolidated data for correct counts
            stats_html = get_program_statistics(merit_df, consolidated_df)
            
            # Also display the consolidated data with test scores
            consolidated_display = df_to_display_html(display_with_scores)
            
            final_message = f"""
            <div style='background-color: #d4edda; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                <h4 style='color: #155724; margin: 0 0 10px 0;'>✅ {message}</h4>
                <p style='margin: 0; color: #155724;'>Below is your consolidated data with test scores, followed by the merit list table.</p>
            </div>
            
            <h3>📋 Consolidated Data with Test Scores</h3>
            {consolidated_display}
            
            <br><h3>🏆 Merit List</h3>
            """
            
            return final_message + table_html, stats_html, ""
        else:
            return "", "", "Error generating merit list data."

    except Exception as e:
        return "", "", f"Error processing files: {str(e)}"

# --------------------------
# AI Analysis Functions (unchanged)
# --------------------------
def analyze_data_with_ai(df, user_query):
    """Enhanced AI analysis with detailed data insights"""
    try:
        # Prepare comprehensive data summary
        data_summary = prepare_data_summary(df)

        # Create enhanced prompt
        prompt = f"""
        You are an expert university admissions analyst. Analyze this dataset and provide detailed insights.

        Dataset Overview:
        - Total Candidates: {data_summary['total_candidates']}
        - Total Applications: {data_summary['total_applications']}
        - Available Columns: {', '.join(data_summary['columns'])}

        Key Statistics:
        - Most Popular Programs: {data_summary.get('popular_programs', [])}
        - Average Choices per Candidate: {data_summary.get('avg_choices', 0):.2f}
        - Completion Rate: {data_summary.get('completion_rate', 0):.1%}

        User Question: {user_query}

        Provide a detailed analysis with:
        1. Direct answer to the user's question
        2. Supporting statistics and trends
        3. Actionable insights for admissions officers
        4. Any data quality observations
        """

        messages = [
            {"role": "system", "content": "You are an expert admissions data analyst. Provide clear, detailed, and actionable insights."},
            {"role": "user", "content": prompt}
        ]

        response = call_ai_api(messages)
        return f"🤖 **AI Analysis:**\n\n{response}"

    except Exception as e:
        return f"❌ Error in AI analysis: {str(e)}"

def generate_statistical_insights(df):
    """Generate comprehensive statistical insights"""
    try:
        insights = []

        # Basic statistics
        stats = prepare_data_summary(df)

        # Program popularity analysis
        if stats.get('popular_programs'):
            insights.append("📊 **Program Popularity:**")
            for i, (program, count) in enumerate(stats['popular_programs'][:5], 1):
                insights.append(f"{i}. {program}: {count} applications")

        # Choice patterns
        insights.append(f"\n🎯 **Application Patterns:**")
        insights.append(f"- Average choices per candidate: {stats.get('avg_choices', 0):.2f}")
        insights.append(f"- Total candidates: {stats['total_candidates']}")
        insights.append(f"- Application completion rate: {stats.get('completion_rate', 0):.1%}")

        # Data quality insights
        quality_issues = analyze_data_quality(df)
        if quality_issues:
            insights.append(f"\n⚠️ **Data Quality Observations:**")
            insights.extend(quality_issues)

        return "\n".join(insights)

    except Exception as e:
        return f"❌ Error generating insights: {str(e)}"

def prepare_data_summary(df):
    """Prepare comprehensive data summary for AI analysis"""
    summary = {
        "total_candidates": 0,
        "total_applications": len(df),
        "columns": list(df.columns),
        "popular_programs": [],
        "avg_choices": 0,
        "completion_rate": 0
    }

    try:
        # Calculate candidate count
        if "Reference No" in df.columns:
            df_clean = df.copy().fillna("")
            df_clean["CandidateID"] = df_clean["Reference No"].replace("", pd.NA).ffill()
            summary["total_candidates"] = df_clean["CandidateID"].nunique()

            # Calculate choice statistics
            if PREF_NAME_COL < len(df.columns):
                choice_counts = []
                program_popularity = {}
                complete_candidates = 0

                for cand, group in df_clean.groupby("CandidateID"):
                    candidate_choices = 0
                    has_complete_info = True

                    for _, row in group.iterrows():
                        pref_name = str(row.iloc[PREF_NAME_COL]).strip()
                        if pref_name not in ("", "nan"):
                            candidate_choices += 1
                            program_popularity[pref_name] = program_popularity.get(pref_name, 0) + 1

                    choice_counts.append(candidate_choices)
                    if candidate_choices > 0:
                        complete_candidates += 1

                summary["avg_choices"] = sum(choice_counts) / len(choice_counts) if choice_counts else 0
                summary["completion_rate"] = complete_candidates / summary["total_candidates"] if summary["total_candidates"] > 0 else 0
                summary["popular_programs"] = sorted(program_popularity.items(), key=lambda x: x[1], reverse=True)[:10]

    except Exception as e:
        print(f"Error in data summary: {e}")

    return summary

def analyze_data_quality(df):
    """Analyze data quality issues"""
    issues = []

    try:
        # Check for missing critical data
        if "Reference No" in df.columns:
            missing_ref = df["Reference No"].isna().sum()
            if missing_ref > 0:
                issues.append(f"- {missing_ref} rows with missing Reference Numbers")

        # Check for duplicate entries
        if len(duplicate_pairs) > 0:
            issues.append(f"- {len(duplicate_pairs)} duplicate degree entries detected")

        # Check mobile number formatting
        mobile_cols = [col for col in df.columns if 'mobile' in col.lower() or 'cell' in col.lower()]
        for col in mobile_cols:
            invalid_mobile = df[col].apply(lambda x: len(str(x)) not in [11, 10] if str(x).isdigit() else True).sum()
            if invalid_mobile > 0:
                issues.append(f"- {invalid_mobile} potentially invalid mobile numbers in {col}")

    except Exception as e:
        issues.append(f"- Error checking data quality: {e}")

    return issues

def create_visualization(df, chart_type, query=""):
    """Create statistical visualizations based on user request"""
    try:
        if df is None or df.empty:
            return "❌ No data available for visualization"

        # Prepare data for visualization
        df_clean = df.copy().fillna("")
        df_clean["CandidateID"] = df_clean["Reference No"].replace("", pd.NA).ffill()

        if chart_type
