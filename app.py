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

        if chart_type.lower() in ["popular", "program", "popularity"]:
            # Program popularity chart
            program_counts = {}
            for cand, group in df_clean.groupby("CandidateID"):
                for _, row in group.iterrows():
                    pref_name = str(row.iloc[PREF_NAME_COL]).strip()
                    if pref_name not in ("", "nan"):
                        program_counts[pref_name] = program_counts.get(pref_name, 0) + 1

            if program_counts:
                # Create bar chart
                programs = list(program_counts.keys())[:10]  # Top 10
                counts = [program_counts[p] for p in programs]

                fig = px.bar(
                    x=counts,
                    y=programs,
                    orientation='h',
                    title="Most Popular Degree Programs",
                    labels={'x': 'Number of Applications', 'y': 'Programs'}
                )
                fig.update_layout(height=500)
                return fig.to_html(include_plotlyjs='cdn')

        elif chart_type.lower() in ["choices", "distribution"]:
            # Choices per candidate distribution
            choice_counts = []
            for cand, group in df_clean.groupby("CandidateID"):
                candidate_choices = 0
                for _, row in group.iterrows():
                    pref_name = str(row.iloc[PREF_NAME_COL]).strip()
                    if pref_name not in ("", "nan"):
                        candidate_choices += 1
                choice_counts.append(candidate_choices)

            if choice_counts:
                fig = px.histogram(
                    x=choice_counts,
                    title="Distribution of Choices per Candidate",
                    labels={'x': 'Number of Choices', 'y': 'Number of Candidates'}
                )
                return fig.to_html(include_plotlyjs='cdn')

        return "📊 Visualization created successfully! (Chart type not fully implemented yet)"

    except Exception as e:
        return f"❌ Error creating visualization: {str(e)}"

def handle_ai_query(query):
    """Enhanced AI query handler with chart generation"""
    if global_df is None:
        return "⚠️ Please upload an Excel file first before asking questions about the data.", ""

    if not query.strip():
        return "💬 Please enter a question about your admissions data.", ""

    # Check if user wants a chart
    chart_keywords = ["chart", "graph", "plot", "visualize", "show", "display"]
    wants_chart = any(keyword in query.lower() for keyword in chart_keywords)

    # Get AI analysis
    ai_response = analyze_data_with_ai(global_df, query)

    # Generate chart if requested
    chart_html = ""
    if wants_chart:
        if "popular" in query.lower() or "program" in query.lower():
            chart_html = create_visualization(global_df, "popular", query)
        elif "choice" in query.lower() or "distribution" in query.lower():
            chart_html = create_visualization(global_df, "choices", query)

    return ai_response, chart_html

def generate_comprehensive_report():
    """Generate a comprehensive admissions report"""
    if global_df is None:
        return "⚠️ Please upload an Excel file first to generate a report."

    try:
        # Get statistical insights
        stats_report = generate_statistical_insights(global_df)

        # Get AI-generated insights
        ai_prompt = """
        Generate a comprehensive executive summary for university admissions officers based on this admissions dataset.
        Focus on:
        1. Key trends and patterns
        2. Strategic recommendations
        3. Operational insights
        4. Risk factors or concerns
        5. Opportunities for improvement

        Keep it professional and actionable for decision-makers.
        """

        messages = [
            {"role": "system", "content": "You are a senior admissions consultant creating an executive report."},
            {"role": "user", "content": ai_prompt}
        ]

        ai_insights = call_ai_api(messages)

        # Combine reports
        full_report = f"""
# 📋 Executive Admissions Analytics Report

## 📊 Statistical Overview
{stats_report}

## 🤖 AI-Generated Strategic Insights
{ai_insights}

## 📈 Recommendations
Based on the analysis above, consider:
- Reviewing program capacity for high-demand programs
- Implementing data quality improvements
- Optimizing the application process based on choice patterns
- Developing targeted recruitment strategies

---
*Report generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
        """

        return full_report

    except Exception as e:
        return f"❌ Error generating report: {str(e)}"

# --------------------------
# Helper Functions (unchanged)
# --------------------------

def df_to_display_html(df, freeze_headers=True):
    display = df.copy()
    html = display.to_html(index=False, escape=False, float_format=lambda x: f'{int(x)}' if pd.notnull(x) and x == int(x) else str(x))

    if freeze_headers:
        html = f"""
        <div style="overflow: auto; max-height: 500px; max-width: 100%;">
            <style>
                .frozen-table {{
                    border-collapse: collapse;
                    width: 100%;
                }}
                .frozen-table th {{
                    position: sticky;
                    top: 0;
                    background-color: #f8f9fa;
                    z-index: 10;
                    border: 1px solid #dee2e6;
                    padding: 8px;
                }}
                .frozen-table td:nth-child(1) {{
                    position: sticky;
                    left: 0;
                    background-color: #fff;
                    z-index: 5;
                    border-right: 2px solid #dee2e6;
                    min-width: 100px;
                }}
                .frozen-table td:nth-child(2) {{
                    position: sticky;
                    left: 100px;
                    background-color: #fff;
                    z-index: 5;
                    border-right: 2px solid #dee2e6;
                    min-width: 150px;
                }}
                .frozen-table th:nth-child(1) {{
                    position: sticky;
                    left: 0;
                    z-index: 15;
                    border-right: 2px solid #dee2e6;
                    min-width: 100px;
                }}
                .frozen-table th:nth-child(2) {{
                    position: sticky;
                    left: 100px;
                    z-index: 15;
                    border-right: 2px solid #dee2e6;
                    min-width: 150px;
                }}
            </style>
            {html.replace('<table', '<table class="frozen-table"')}
        </div>
        """
    return html

def write_temp_excel(df):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
    tmp.close()
    df.to_excel(tmp.name,index=False)
    return tmp.name

def write_temp_excel_with_formatting(df):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
    tmp.close()

    with pd.ExcelWriter(tmp.name, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="consolidated")
        worksheet = writer.sheets["consolidated"]

        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width

        mobile_cols = []
        for idx, col_name in enumerate(df.columns):
            if 'mobile' in col_name.lower() or 'cell' in col_name.lower():
                mobile_cols.append(idx + 1)

        for col_idx in mobile_cols:
            col_letter = worksheet.cell(row=1, column=col_idx).column_letter
            for row in range(2, len(df) + 2):
                cell = worksheet[f"{col_letter}{row}"]
                if cell.value:
                    mobile_str = str(cell.value)
                    if mobile_str.replace('.', '').replace('e+', '').replace('-', '').isdigit():
                        try:
                            if 'e+' in mobile_str:
                                mobile_num = f"{int(float(mobile_str)):011d}"
                            else:
                                mobile_num = mobile_str
                            cell.value = mobile_num
                            cell.number_format = '@'
                        except:
                            cell.number_format = '@'

        worksheet.freeze_panes = 'C2'

    return tmp.name

def write_temp_excel_with_audit(df,audit_rows=None):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
    tmp.close()
    with pd.ExcelWriter(tmp.name,engine="openpyxl") as writer:
        df.to_excel(writer,index=False,sheet_name="cleaned")
        if audit_rows:
            pd.DataFrame(audit_rows).to_excel(writer,index=False,sheet_name="audit")
        worksheet = writer.sheets["cleaned"]
        worksheet.freeze_panes = 'C2'
    return tmp.name

def reset_paths():
    global cleaned_file_path, swapped_file_path, choices_file_path, last_pre_swap_path, last_swap_audit, consolidated_file_path
    cleaned_file_path = None
    swapped_file_path = None
    choices_file_path = None
    consolidated_file_path = None
    last_pre_swap_path = None
    last_swap_audit = []

def create_progress_html(message="Processing..."):
    return f"""
    <div style="text-align: center; padding: 20px; background-color: #f0f0f0; border-radius: 10px; margin: 10px 0;">
        <div style="display: inline-block; width: 40px; height: 40px; border: 4px solid #f3f3f3;
                    border-top: 4px solid #3498db; border-radius: 50%;
                    animation: spin 1s linear infinite; margin-bottom: 10px;">
        </div>
        <div style="color: #333; font-weight: bold;">{message}</div>
        <style>
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
        </style>
    </div>
    """

# --------------------------
# Data Processing Functions (unchanged from original)
# --------------------------

def process_excel(file):
    global global_df, duplicate_pairs, degree_dup_mask
    global_df = None
    duplicate_pairs = []
    degree_dup_mask = None
    reset_paths()
    if file is None:
        return (None,"### 👥 Number of Candidates: --",gr.update(visible=False),"",gr.update(visible=False),gr.update(visible=False),gr.update(visible=False))

    try:
        df = pd.read_excel(file.name).fillna("")

        # Fix mobile number formatting immediately upon loading
        for col in df.columns:
            if 'mobile' in col.lower() or 'cell' in col.lower():
                def fix_mobile_on_load(x):
                    if pd.isna(x) or x == '':
                        return ''
                    try:
                        str_x = str(x)
                        if 'e+' in str_x.lower():
                            num_val = int(float(x))
                            if len(str(num_val)) == 10:
                                return f"0{num_val}"
                            else:
                                return str(num_val)
                        elif str_x.replace('.0', '').isdigit():
                            clean_num = str_x.replace('.0', '')
                            if len(clean_num) == 10:
                                return f"0{clean_num}"
                            else:
                                return clean_num
                        else:
                            return str_x
                    except:
                        return str(x)

                df[col] = df[col].apply(fix_mobile_on_load)

        global_df = df.copy()
        candidate_count = df["Reference No"].replace("",pd.NA).ffill().nunique() if "Reference No" in df.columns else 0
        candidate_text = f"### 👥 Number of Candidates: **{candidate_count}**"
        return (df_to_display_html(df),candidate_text,gr.update(visible=True),"",gr.update(visible=False),gr.update(visible=False),gr.update(visible=False))
    except Exception as e:
        return (f"<p style='color:red'>❌ Error reading file: {e}</p>","Error",gr.update(visible=False),"",gr.update(visible=False),gr.update(visible=False),gr.update(visible=False))

def check_degree_duplicates():
    global global_df, duplicate_pairs, degree_dup_mask
    if global_df is None:
        return (None,"⚠️ Upload a file first",gr.update(visible=False),"",gr.update(visible=False))

    df = global_df.copy()
    subset = df.iloc[:,CHECK_COL_INDICES].astype(str).apply(lambda col: col.str.strip())
    same_as_above = subset.eq(subset.shift()).all(axis=1)
    non_empty = subset.apply(lambda row: row.ne("").any(),axis=1)
    degree_dup_mask = same_as_above & non_empty
    duplicate_pairs = [(idx-1,idx) for idx,val in degree_dup_mask.items() if val and idx>0]

    if duplicate_pairs:
        highlight_mask = degree_dup_mask | degree_dup_mask.shift(-1).fillna(False)
        preview_frames = []
        for prev_idx, curr_idx in duplicate_pairs:
            preview_frames.append(df.iloc[[prev_idx, curr_idx]].copy())

        preview_html = ""
        if preview_frames:
            preview_df = pd.concat(preview_frames)
            preview_html = df_to_display_html(preview_df)

        display_df = df.copy()
        highlighted_rows = []
        for idx, row in display_df.iterrows():
            if highlight_mask.loc[idx]:
                row_html = '<tr style="background-color:#ffcccc;">'
                for col in row:
                    row_html += f'<td>{col}</td>'
                row_html += '</tr>'
                highlighted_rows.append(row_html)
            else:
                row_html = '<tr>'
                for col in row:
                    row_html += f'<td>{col}</td>'
                row_html += '</tr>'
                highlighted_rows.append(row_html)

        header_html = '<tr>' + ''.join([f'<th>{col}</th>' for col in display_df.columns]) + '</tr>'
        table_html = f'<table class="frozen-table"><thead>{header_html}</thead><tbody>{"".join(highlighted_rows)}</tbody></table>'
        styled_html = df_to_display_html(display_df).replace(display_df.to_html(index=False, escape=False), table_html)

        return (styled_html,f"⚠️ Found {len(duplicate_pairs)} consecutive duplicate degree entries.",gr.update(visible=True),preview_html,gr.update(visible=False))

    return (df_to_display_html(df),"✅ No consecutive duplicate degree entries found.",gr.update(visible=False),"",gr.update(visible=False))

def delete_degree_duplicates():
    global global_df, degree_dup_mask, duplicate_pairs, cleaned_file_path
    if global_df is None or degree_dup_mask is None:
        return (None,gr.update(visible=False),gr.update(visible=False))
    drop_indices = sorted({curr for (_,curr) in duplicate_pairs})
    cleaned = global_df.drop(index=drop_indices).reset_index(drop=True)
    global_df = cleaned.copy()
    cleaned_file_path = write_temp_excel(cleaned)
    return (df_to_display_html(cleaned),gr.update(value=cleaned_file_path,visible=True),gr.update(visible=True))

def check_unstructured_degrees():
    global global_df, implicated_pairs, highlight_indices
    implicated_pairs = []
    highlight_indices = set()
    if global_df is None:
        return (None,"⚠️ Upload and delete duplicates first",gr.update(visible=False),None)
    df = global_df.copy()
    df["CandidateID"]=df["Reference No"].replace("",pd.NA).ffill()
    for cand,group in df.groupby("CandidateID"):
        degree_rows = group.iloc[:,CHECK_COL_INDICES]
        idxs = degree_rows.index.tolist()
        if len(idxs)<2: continue
        first_title = str(df.at[idxs[0],df.columns[COL_H_IDX]]).strip()
        if first_title not in FIRST_DEGREE_ALLOWED:
            for later in idxs[1:]:
                if str(df.at[later,df.columns[COL_H_IDX]]).strip() in FIRST_DEGREE_ALLOWED:
                    implicated_pairs.append((idxs[0],later,"swap_for_first"))
                    highlight_indices.update({idxs[0],later})
                    break
        if len(idxs)>=2:
            second_title = str(df.at[idxs[1],df.columns[COL_H_IDX]]).strip()
            if second_title not in SECOND_DEGREE_ALLOWED:
                for later in idxs[2:]:
                    if str(df.at[later,df.columns[COL_H_IDX]]).strip() in SECOND_DEGREE_ALLOWED:
                        implicated_pairs.append((idxs[1],later,"swap_for_second"))
                        highlight_indices.update({idxs[1],later})
                        break
    display_df = df.copy()
    if not implicated_pairs:
        return (df_to_display_html(df),"✅ No unstructured degree rows found.",gr.update(visible=False),None)
    def highlight_unstructured(r):
        return (["background-color:#fff2cc"]*len(r)) if r.name in highlight_indices else ([""]*len(r))
    styled_html = display_df.style.apply(highlight_unstructured,axis=1).to_html(index=False)
    preview_frames=[]
    for a,b,reason in implicated_pairs:
        preview_frames.append(display_df.iloc[[a,b]].copy())
    preview_html=pd.concat(preview_frames).to_html(index=False,escape=False) if preview_frames else ""
    return (styled_html,f"⚠️ Found {len(implicated_pairs)} unstructured pair(s). Review preview below.",gr.update(visible=True),preview_html)

def perform_swaps_confirmed():
    global global_df, implicated_pairs, last_pre_swap_path, last_swap_audit, swapped_file_path
    if global_df is None or not implicated_pairs:
        return (None,"⚠️ Nothing to swap",gr.update(visible=False),gr.update(visible=False),gr.update(visible=False))
    last_pre_swap_path = write_temp_excel(global_df.copy())
    last_swap_audit=[]
    df = global_df.copy()
    for a_idx,b_idx,reason in implicated_pairs:
        vals_a = df.iloc[a_idx,CHECK_COL_INDICES].copy()
        vals_b = df.iloc[b_idx,CHECK_COL_INDICES].copy()
        df.iloc[a_idx,CHECK_COL_INDICES]=vals_b.values
        df.iloc[b_idx,CHECK_COL_INDICES]=vals_a.values
        last_swap_audit.append({
            "timestamp":datetime.datetime.utcnow().isoformat()+"Z",
            "candidate_id":df.at[a_idx,"Reference No"],
            "row_a_excel":a_idx+2,
            "row_b_excel":b_idx+2,
            "reason":reason,
            "swapped_columns":",".join([str(i+1) for i in CHECK_COL_INDICES])
        })
    global_df=df.copy()
    swapped_file_path=write_temp_excel_with_audit(global_df,last_swap_audit)
    implicated_pairs.clear()
    highlight_indices.clear()
    return (df_to_display_html(df),f"✅ Performed {len(last_swap_audit)} swaps. Download available.",gr.update(value=swapped_file_path,visible=True),gr.update(visible=True),gr.update(visible=False))

def show_candidate_choices_and_save():
    global global_df, choices_file_path
    if global_df is None:
        return ("<p style='color:red'>⚠️ Upload a file first</p>",gr.update(visible=False))
    df=global_df.copy().fillna("")
    df["CandidateID"]=df["Reference No"].replace("",pd.NA).ffill()
    summary=[]
    for cand,group in df.groupby("CandidateID"):
        applicant_name = group.iloc[0].get("Applicant Name", "")
        prefs=[]
        for _,row in group.iterrows():
            pref_num=str(row.iloc[PREF_NUM_COL]).strip()
            pref_name=str(row.iloc[PREF_NAME_COL]).strip()
            if pref_num not in ("","nan") and pref_name not in ("","nan"):
                prefs.append(pref_name)
        row_data={"Reference No":cand,"Applicant Name":applicant_name,"NumChoices":len(prefs)}
        for i in range(5):
            row_data[f"Choice {i+1}"]=prefs[i] if i<len(prefs) else ""
        summary.append(row_data)
    summary_df=pd.DataFrame(summary)
    choices_file_path=write_temp_excel(summary_df)
    return (df_to_display_html(summary_df),gr.update(value=choices_file_path,visible=True))

# FIXED: Generate Consolidated Excel function that properly stores result
def generate_consolidated_excel():
    global global_df, consolidated_file_path, consolidated_df
    if global_df is None:
        return ("<p style='color:red'>⚠️ Upload and clean a file first</p>",gr.update(visible=False))

    df = global_df.copy().fillna("")
    df["CandidateID"] = df["Reference No"].replace("",pd.NA).ffill()

    degree_titles = ["10th Grade or Equivalent","12th Grade or Equivalent"]
    degree_cols = []
    for i in range(2):
        title = degree_titles[i]
        for idx in CHECK_COL_INDICES:
            degree_cols.append(f"{df.columns[idx]} ({title})")

    seen_candidates = set()
    unique_candidates_ordered = []
    for cand_id in df["CandidateID"]:
        if pd.notna(cand_id) and cand_id not in seen_candidates:
            unique_candidates_ordered.append(cand_id)
            seen_candidates.add(cand_id)

    consolidated_rows = []
    for cand in unique_candidates_ordered:
        group = df[df["CandidateID"] == cand]
        base_cols = [c for i,c in enumerate(df.columns) if i not in CHECK_COL_INDICES+[PREF_NUM_COL,PREF_NAME_COL] and c != "CandidateID"]
        base_data = group.iloc[0][base_cols].tolist()

        degrees_data = []
        degree_rows = group.iloc[:,CHECK_COL_INDICES]
        for i in range(min(2, len(degree_rows))):
            degrees_data.extend(degree_rows.iloc[i].tolist())

        if len(degree_rows) < 2:
            missing_cols = len(CHECK_COL_INDICES) * (2 - len(degree_rows))
            degrees_data.extend([""] * missing_cols)

        prefs = []
        for _,row in group.iterrows():
            pref_num = str(row.iloc[PREF_NUM_COL]).strip()
            pref_name = str(row.iloc[PREF_NAME_COL]).strip()
            if pref_num not in ("","nan") and pref_name not in ("","nan"):
                prefs.append(pref_name)
        choice_data = [len(prefs)] + [prefs[i] if i<len(prefs) else "" for i in range(5)]

        consolidated_rows.append(base_data + degrees_data + choice_data)

    final_cols = [c for i,c in enumerate(df.columns) if i not in CHECK_COL_INDICES+[PREF_NUM_COL,PREF_NAME_COL] and c != "CandidateID"] + degree_cols + ["Count of Preferences","First Choice","Second Choice","Third Choice","Fourth Choice","Fifth Choice"]
    consolidated_df = pd.DataFrame(consolidated_rows, columns=final_cols)

    # Fix mobile numbers and numeric formatting
    for col in consolidated_df.columns:
        consolidated_df[col] = consolidated_df[col].astype(str)

        if 'mobile' in col.lower() or 'cell' in col.lower():
            def fix_mobile(x):
                if pd.isna(x) or x == '' or x == 'nan':
                    return ''
                try:
                    clean_num = str(x).replace('.0', '').replace(' ', '')
                    if clean_num.isdigit() and len(clean_num) == 10:
                        return f"0{clean_num}"
                    elif clean_num.isdigit() and len(clean_num) == 11 and clean_num.startswith('0'):
                        return clean_num
                    elif clean_num.replace('.', '').isdigit():
                        return clean_num
                    else:
                        return str(x)
                except:
                    return str(x)
            consolidated_df[col] = consolidated_df[col].apply(fix_mobile)

        else:
            def fix_numeric(x):
                if pd.isna(x) or x == '' or x == 'nan':
                    return ''
                try:
                    if str(x).endswith('.0'):
                        return str(x)[:-2]
                    else:
                        return str(x)
                except:
                    return str(x)
            consolidated_df[col] = consolidated_df[col].apply(fix_numeric)

    consolidated_file_path = write_temp_excel_with_formatting(consolidated_df)
    return (df_to_display_html(consolidated_df), gr.update(value=consolidated_file_path, visible=True))

# --------------------------
# Enhanced Gradio UI with Modern Styling
# --------------------------
with gr.Blocks(css="""
footer {visibility:hidden}
.modern-button {
    background: linear-gradient(45deg, #667eea, #764ba2) !important;
    border: none !important;
    color: white !important;
    font-weight: bold !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
}
.modern-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
}
.modern-button:active {
    transform: translateY(0px) !important;
    box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3) !important;
}
""") as demo:
    # Header with Project Branding - FIXED to show OGenSys
    gr.HTML("""
    <div style='text-align: center; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; margin-bottom: 20px;'>
        <h1 style='color: white; margin: 0; font-size: 2.5em; font-weight: bold;'>🎓 OGenSys</h1>
        <p style='color: #f0f0f0; margin: 10px 0 0 0; font-size: 1.2em;'>Executive Analytics for Admissions</p>
        <p style='color: #d0d0d0; margin: 5px 0 0 0; font-style: italic;'>Powered by AI for intelligent data processing and insights</p>
    </div>
    """)

    with gr.Tabs():
        # Main Data Processing Tab
        with gr.TabItem("📊 Data Processing"):
            file_input=gr.File(label="Upload Excel",file_types=[".xlsx",".xls"])
            df_output=gr.HTML()
            candidate_count_output=gr.Markdown("### Number of Candidates: --")

            check_dup_btn=gr.Button("Check Duplicate Degree Entries",visible=False)
            dup_message=gr.Markdown("")
            duplicate_preview=gr.HTML(label="Duplicate Pairs Preview")
            delete_dup_btn=gr.Button("Delete Duplicate Degree Entries",visible=False)
            download_after_delete=gr.File(label="Download Cleaned (after delete)",visible=False)

            check_unstructured_btn=gr.Button("Check Unstructured Degree Data",visible=False)
            unstructured_message=gr.Markdown("")
            unstructured_preview=gr.HTML(label="Unstructured Pairs Preview")
            swap_confirm_btn=gr.Button("Confirm Swap",visible=False)
            download_after_swap=gr.File(label="Download Cleaned (after swap)",visible=False)

            show_choices_btn=gr.Button("Show Candidate Choices",visible=True)
            choices_output=gr.HTML()
            download_choices_btn=gr.File(label="Download Choices Summary",visible=False)

            gen_consolidated_btn=gr.Button("Generate Consolidated Excel",visible=True)
            progress_status=gr.HTML()
            consolidated_output=gr.HTML()
            download_consolidated_btn=gr.File(label="Download Consolidated Excel",visible=False)

        # Enhanced AI Analytics Tab
        with gr.TabItem("🤖 AI Analytics"):
            gr.Markdown("### Ask Questions About Your Data")
            gr.Markdown("Use natural language to query your admissions data and get AI-powered insights with visualizations.")

            with gr.Row():
                with gr.Column(scale=3):
                    ai_query_input = gr.Textbox(
                        placeholder="e.g., 'Show me a chart of the most popular programs' or 'How many candidates applied for engineering?'",
                        label="Your Question",
                        lines=2
                    )
                with gr.Column(scale=1):
                    ask_ai_btn = gr.Button("Ask AI", variant="primary")

            ai_response_output = gr.Markdown()
            chart_output = gr.HTML(label="Generated Chart")

            gr.Markdown("### Automatic Insights & Reports")
            with gr.Row():
                generate_insights_btn = gr.Button("Generate Key Insights", variant="secondary")
                generate_report_btn = gr.Button("Generate Executive Report", variant="secondary")

            ai_insights_output = gr.Markdown()

            # Quick Action Buttons
            gr.Markdown("### Quick Actions")
            with gr.Row():
                popular_programs_btn = gr.Button("Most Popular Programs")
                data_quality_btn = gr.Button("Data Quality Check")
                completion_stats_btn = gr.Button("Application Statistics")

            # Example queries
            gr.Markdown("### Example Questions")
            example_queries = [
                "What are the most popular degree programs?",
                "How many candidates have complete applications?",
                "Show me a chart of choices per candidate distribution",
                "Are there any data quality issues I should know about?",
                "Generate a summary report of the admissions data"
            ]

            for example in example_queries:
                gr.Button(example).click(
                    fn=lambda q=example: q,
                    outputs=ai_query_input
                )

        # Merit List Generation Tab
        with gr.TabItem("🏆 Merit List Generation"):
            # Project Branding at top center - FIXED to show OGenSys
            gr.HTML("""
            <div style='text-align: center; background: linear-gradient(45deg, #667eea, #764ba2); padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
                <h2 style='color: white; margin: 0; font-weight: bold;'>🎓 OGenSys Merit List Generator</h2>
                <p style='color: #f0f0f0; margin: 5px 0 0 0;'>Executive Analytics for Admissions</p>
            </div>
            """)

            gr.Markdown("### Step 1: Upload Test Marks")
            gr.Markdown("Upload an Excel file with Reference No (Column A) and Test Marks out of 50 (Column B)")

            test_marks_file = gr.File(
                label="Upload Test Marks Excel File",
                file_types=[".xlsx", ".xls"]
            )

            generate_merit_btn = gr.Button(
                "Generate Merit List",
                variant="primary"
            )

            merit_status_msg = gr.Markdown("")

            # Program-wise seat statistics (stock ticker style)
            gr.Markdown("### 📊 Program-wise Seat Allocation")
            program_stats_output = gr.HTML()

            # Merit list table
            gr.Markdown("### 📋 Merit List Table")
            merit_table_output = gr.HTML()

    # Event handlers for main tab - FIXED progress handling
    file_input.change(process_excel,inputs=file_input,outputs=[df_output,candidate_count_output,check_dup_btn,dup_message,delete_dup_btn,download_after_delete,check_unstructured_btn])
    check_dup_btn.click(check_degree_duplicates,inputs=None,outputs=[df_output,dup_message,delete_dup_btn,duplicate_preview,download_after_delete])
    delete_dup_btn.click(delete_degree_duplicates,inputs=None,outputs=[df_output,download_after_delete,check_unstructured_btn])
    check_unstructured_btn.click(check_unstructured_degrees,inputs=None,outputs=[df_output,unstructured_message,swap_confirm_btn,unstructured_preview])
    swap_confirm_btn.click(perform_swaps_confirmed,inputs=None,outputs=[df_output,unstructured_message,download_after_swap,swap_confirm_btn,swap_confirm_btn])
    show_choices_btn.click(show_candidate_choices_and_save,inputs=None,outputs=[choices_output,download_choices_btn])
    
    # FIXED: Progress status handling for consolidated Excel generation
    def show_progress():
        return create_progress_html("Generating consolidated Excel...")
    
    def clear_progress_and_generate():
        result_html, download_file = generate_consolidated_excel()
        return "", result_html, download_file
    
    gen_consolidated_btn.click(
        fn=show_progress,
        inputs=None, 
        outputs=progress_status
    ).then(
        fn=clear_progress_and_generate,
        inputs=None,
        outputs=[progress_status, consolidated_output, download_consolidated_btn]
    )

    # Event handlers for AI tab
    ask_ai_btn.click(handle_ai_query, inputs=ai_query_input, outputs=[ai_response_output, chart_output])
    ai_query_input.submit(handle_ai_query, inputs=ai_query_input, outputs=[ai_response_output, chart_output])
    generate_insights_btn.click(lambda: generate_statistical_insights(global_df) if global_df is not None else "Please upload data first", inputs=None, outputs=ai_insights_output)
    generate_report_btn.click(generate_comprehensive_report, inputs=None, outputs=ai_insights_output)

    # Quick action handlers
    popular_programs_btn.click(lambda: "What are the most popular degree programs? Show me a chart.", outputs=ai_query_input)
    data_quality_btn.click(lambda: "Are there any data quality issues I should know about?", outputs=ai_query_input)
    completion_stats_btn.click(lambda: "What are the application completion statistics?", outputs=ai_query_input)

    # Event handlers for Merit List Generation tab
    generate_merit_btn.click(
        fn=generate_merit_list_from_files,
        inputs=[test_marks_file],
        outputs=[merit_table_output, program_stats_output, merit_status_msg]
    )

if __name__ == "__main__":
    demo.launch()
