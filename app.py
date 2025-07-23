import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import os
from datetime import datetime

# --- App Configuration ---
st.set_page_config(
    page_title="AI Powered Expense Tracker",
    page_icon="https://i.ibb.co/ycXsxct7/logo.jpg",
    layout="wide"
)

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        
        :root {
            --primary: #1F3B73;
            --secondary: #4B79A1;
            --accent: #FF6B6B;
            --light: #F8F9FA;
            --dark: #212529;
            --success: #28A745;
            --warning: #FFC107;
            --danger: #DC3545;
        }
        
        * {
            font-family: 'Poppins', sans-serif;
        }
        
        [data-testid="stAppViewContainer"] {
            background-color: #f5f7fa;
        }
        
        .main-header {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1.5rem;
        }
        
        .main-header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            color: white !important;  /* Force white title */
        
        }
        
        .main-header p {
            font-size: 1.1rem;
            color: rgba(255, 255, 255, 0.9);
            margin-bottom: 0;
        }
        
        .card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            margin-bottom: 1.5rem;
            border: none;
        }
        
        .card-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--primary);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        [data-testid="stTabs"] {
            margin-bottom: 1.5rem;
        }
        
        [data-testid="stTabs"] [role="tab"] {
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            margin-right: 0.5rem;
            font-weight: 500;
            background: transparent;
            color: var(--dark);
            border: 1px solid transparent;
            transition: all 0.2s ease;
        }
        
        [data-testid="stTabs"] [role="tab"]:hover {
            background: rgba(31, 59, 115, 0.1);
            color: var(--primary);
        }
        
        [data-testid="stTabs"] [aria-selected="true"] {
            background: var(--primary);
            color: white;
            border-color: var(--primary);
        }
        
        .sidebar .sidebar-content {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        
        .sidebar-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--primary);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .budget-status {
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            font-weight: 500;
        }
        
        .budget-status.within {
            background: rgba(40, 167, 69, 0.1);
            color: var(--success);
        }
        
        .budget-status.exceeded {
            background: rgba(220, 53, 69, 0.1);
            color: var(--danger);
        }
        
        .transaction-item {
            padding: 0.75rem;
            border-radius: 8px;
            background: white;
            margin-bottom: 0.75rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: black !important;  /* Force black text */
        }
        
        .transaction-item strong {
            color: black !important;  /* Force black for description */
        }
        
        .transaction-item small {
            color: #555 !important;  /* Slightly lighter for date/category */
        }
        

        
        .transaction-item.income {
            border-left: 4px solid var(--success);
        }
        
        .transaction-item.expense {
            border-left: 4px solid var(--danger);
        }
        
        .btn-primary {
            background: var(--primary);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        
        .btn-primary:hover {
            background: #18315c;
            color: white;
        }
        
        .btn-outline {
            background: transparent;
            color: var(--primary);
            border: 1px solid var(--primary);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        
        .btn-outline:hover {
            background: rgba(31, 59, 115, 0.1);
            color: var(--primary);
        }
        
        .stDataFrame {
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        
        .stAlert {
            border-radius: 8px;
        }
        
        .stProgress > div > div {
            background-color: var(--primary);
        }
        
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: var(--primary);
        }
        
        .stMarkdown hr {
            margin: 1.5rem 0;
            border-top: 1px solid rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# --- Logo and Header ---
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown(
        """
        <div style='text-align: center;'>
            <img src='https://i.ibb.co/ycXsxct7/logo.jpg' width='80' style='border-radius: 50%;'/>
        </div>
        """,
        unsafe_allow_html=True
    )
with col2:

  # --- Logo and Header ---
  st.markdown("""
    <div class="main-header">
        <div style="text-align: left;">
            <h1>AI Powered Finance Tracker</h1>
            <p style="margin-bottom: 0.5rem;">Smart financial management for individuals and businesses</p>
            <p style="font-size: 0.95rem; opacity: 0.9; margin-bottom: 0;">
                Track expenses, analyze spending patterns, and get AI-powered insights to optimize your finances
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

DATA_FILE = "finance_history.csv"

# --- Tabbed Layout ---
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Dashboard", "📊 Analytics", "🤖 AI Insights", "📂 Transactions"])

# --- Categorization Function ---
def categorize(description):
    categories = {
        'Groceries': ['grocery', 'supermarket'],
        'Utilities': ['electricity', 'water', 'internet', 'mobile'],
        'Entertainment': ['movie', 'concert', 'streaming'],
        'Dining': ['restaurant', 'coffee', 'fast food'],
        'Transport': ['gas', 'fuel', 'taxi'],
        'Shopping': ['shopping', 'clothing', 'bookstore'],
        'Health': ['pharmacy', 'gym'],
        'Income': ['salary', 'bonus', 'freelance']
    }
    description = description.lower()
    for category, keywords in categories.items():
        if any(keyword in description for keyword in keywords):
            return category
    return 'Other'

# --- Data Upload / Initialization ---
if os.path.exists(DATA_FILE):
    try:
        df = pd.read_csv(DATA_FILE, parse_dates=['Date'])
    except:
        df = pd.DataFrame(columns=['Date', 'Description', 'Amount', 'Category'])
else:
    df = pd.DataFrame(columns=['Date', 'Description', 'Amount', 'Category'])

# --- Clean Data ---
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
df = df.dropna(subset=['Amount'])

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; margin-bottom: 1.5rem;'>
            <h3 style='color: var(--primary); margin-bottom: 0;'>Quick Actions</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Add Transaction Form
    with st.expander("➕ Add Transaction", expanded=True):
        with st.form("transaction_form"):
            description = st.text_input("Description", "")
            amount = st.number_input("Amount (Rs.)", min_value=0.0, step=1.0)
            transaction_type = st.radio("Type", ["Expense", "Income"], horizontal=True)
            if transaction_type == "Expense":
                amount = -abs(amount)
            submitted = st.form_submit_button("Add Transaction", use_container_width=True)
            
        if submitted and description:
            new_entry = {
                "Date": datetime.now(),
                "Description": description,
                "Amount": amount,
                "Category": categorize(description)
            }
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
            df.to_csv(DATA_FILE, index=False)
            st.success(f"Added: {description} - Rs.{abs(amount):.2f}")

    # Budget Settings
    with st.expander("💵 Budget Settings"):
        budget_limit = st.number_input("Monthly Budget (Rs.)", min_value=0.0, value=0.0, step=100.0)
    
    # Data Management
    with st.expander("⚙ Data Management"):
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file is not None:
            try:
                new_df = pd.read_csv(uploaded_file, parse_dates=['Date'])
                df = pd.concat([df, new_df], ignore_index=True)
                df.to_csv(DATA_FILE, index=False)
                st.success("Data imported successfully!")
            except Exception as e:
                st.error(f"Error reading file: {e}")
        
        if st.button("🔄 Reset All Data", use_container_width=True):
            df = pd.DataFrame(columns=['Date', 'Description', 'Amount', 'Category'])
            if os.path.exists(DATA_FILE):
                os.remove(DATA_FILE)
            st.success("Data reset successfully!")

# Calculate totals
total_income = df[df['Amount'] > 0]['Amount'].sum()
total_expenses = abs(df[df['Amount'] < 0]['Amount'].sum())
net_balance = total_income - total_expenses

# --- DASHBOARD Tab ---
with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="card">
                <div class="card-title">💰 Total Income</div>
                <h2 style='color: var(--success);'>Rs.{:,.2f}</h2>
            </div>
        """.format(total_income), unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="card">
                <div class="card-title">💸 Total Expenses</div>
                <h2 style='color: var(--danger);'>Rs.{:,.2f}</h2>
            </div>
        """.format(total_expenses), unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="card">
                <div class="card-title">⚖ Net Balance</div>
                <h2 style='color: {};'>Rs.{:,.2f}</h2>
            </div>
        """.format("#28A745" if net_balance >= 0 else "#DC3545", net_balance), unsafe_allow_html=True)
    
    # Budget Status
    if budget_limit > 0:
        budget_percentage = (total_expenses / budget_limit) * 100
        status_class = "within" if total_expenses <= budget_limit else "exceeded"
        status_text = "Within Budget" if total_expenses <= budget_limit else "Budget Exceeded"
        
        st.markdown(f"""
            <div class="card">
                <div class="card-title">📊 Budget Status</div>
                <div>Monthly Budget: Rs.{budget_limit:,.2f}</div>
                <div>Spent: Rs.{total_expenses:,.2f} ({budget_percentage:.1f}%)</div>
                <div class="budget-status {status_class}">
                    {status_text}
                </div>
                <div class="stProgress">
                    <div data-testid="stProgress" style="width: 100%;">
                        <div role="progressbar" style="width: {min(budget_percentage, 100)}%; background-color: {'var(--success)' if total_expenses <= budget_limit else 'var(--danger)'}; height: 8px; border-radius: 4px;"></div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Recent Transactions
    st.markdown("""
        <div class="card">
            <div class="card-title">🔄 Recent Transactions</div>
    """, unsafe_allow_html=True)
    
    if not df.empty:
        recent_transactions = df.sort_values('Date', ascending=False).head(5)
        for _, row in recent_transactions.iterrows():
            amount = row['Amount']
            is_income = amount > 0
            amount_class = "income" if is_income else "expense"
            amount_text = "+Rs.{:,.2f}" if is_income else "-Rs.{:,.2f}"
            
            st.markdown(f"""
                <div class="transaction-item {amount_class}">
                    <div>
                        <strong>{row['Description']}</strong><br>
                        <small>{row['Date'].strftime('%b %d, %Y')} • {row['Category']}</small>
                    </div>
                    <div style='font-weight: 600; color: {"var(--success)" if is_income else "var(--danger)"}'>
                        {amount_text.format(abs(amount))}
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No transactions recorded yet.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# --- ANALYTICS Tab ---
with tab2:
    st.markdown("""
        <div class="card">
            <div class="card-title">📊 Spending Analytics</div>
    """, unsafe_allow_html=True)
    
    if not df.empty:
        # Filter expenses only
        expenses_df = df[df['Amount'] < 0].copy()
        expenses_df['Amount'] = expenses_df['Amount'].abs()
        
        if not expenses_df.empty:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("#### Spending by Category")
                category_totals = expenses_df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
                
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                category_totals.plot(kind='pie', autopct='%1.1f%%', startangle=140, ax=ax1, 
                                   colors=plt.cm.Pastel1.colors, wedgeprops={'edgecolor': 'white'})
                ax1.set_ylabel('')
                ax1.set_title('')
                plt.tight_layout()
                st.pyplot(fig1)
            
            with col2:
                st.markdown("#### Top Categories")
                st.dataframe(
                    category_totals.reset_index().rename(columns={'Amount': 'Total (Rs.)'}),
                    use_container_width=True,
                    hide_index=True
                )
                
                st.markdown("#### Monthly Trend")
                expenses_df['Month'] = expenses_df['Date'].dt.to_period('M')
                monthly_totals = expenses_df.groupby('Month')['Amount'].sum().reset_index()
                monthly_totals['Month'] = monthly_totals['Month'].astype(str)
                
                fig2, ax2 = plt.subplots(figsize=(8, 3))
                ax2.plot(monthly_totals['Month'], monthly_totals['Amount'], marker='o', color=st.get_option("theme.primaryColor"))
                ax2.set_title('Monthly Spending')
                ax2.set_ylabel('Amount (Rs.)')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig2)
        else:
            st.info("No expense data to analyze.")
    else:
        st.info("No transactions recorded yet.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# --- AI INSIGHTS Tab ---
with tab3:
    st.markdown("""
        <div class="card">
            <div class="card-title">🤖 AI-Powered Insights</div>
    """, unsafe_allow_html=True)
    
    if len(df) >= 2:
        # Prepare data for prediction
        df = df.sort_values('Date')
        df['Day'] = np.arange(len(df))
        
        # Filter expenses only for prediction
        expenses_df = df[df['Amount'] < 0].copy()
        expenses_df['Amount'] = expenses_df['Amount'].abs()
        
        if len(expenses_df) >= 2:
            X = expenses_df[['Day']]
            y = expenses_df['Amount']
            model = LinearRegression().fit(X, y)
            
            future_days = np.arange(len(df), len(df) + 7).reshape(-1, 1)
            predicted_expenses = model.predict(future_days)
            future_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=7)
            
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            ax3.plot(df['Date'], df['Amount'].abs(), label='Actual Expenses', marker='o', color=st.get_option("theme.primaryColor"))
            ax3.plot(future_dates, predicted_expenses, linestyle='--', color='#FF6B6B', label='Predicted')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Amount (Rs.)')
            ax3.set_title('Expense Trend & Prediction')
            ax3.legend()
            ax3.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig3)
            
            avg_daily = expenses_df['Amount'].mean()
            st.markdown(f"""
                <div style='background: rgba(31, 59, 115, 0.05); padding: 1rem; border-radius: 8px; margin-top: 1rem;'>
                    <h4 style='color: var(--primary); margin-bottom: 0.5rem;'>💡 Insights</h4>
                    <p>• Average daily expense: <strong>Rs.{avg_daily:.2f}</strong></p>
                    <p>• Predicted expenses for next 7 days: <strong>Rs.{predicted_expenses.mean():.2f}/day</strong></p>
                    <p>• Total predicted for next week: <strong>Rs.{predicted_expenses.sum():.2f}</strong></p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Not enough expense data for prediction. Add at least 2 expense transactions.")
    else:
        st.info("Not enough data for trend prediction. Add at least 2 transactions.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# --- TRANSACTIONS Tab ---
with tab4:
    st.markdown("""
        <div class="card">
            <div class="card-title">📂 Transaction History</div>
    """, unsafe_allow_html=True)
    
    if not df.empty:
        # Format display dataframe
        display_df = df.copy()
        display_df['Amount'] = display_df['Amount'].apply(lambda x: f"+Rs.{x:,.2f}" if x > 0 else f"-Rs.{abs(x):,.2f}")
        display_df['Date'] = display_df['Date'].dt.strftime('%b %d, %Y')
        
        st.dataframe(
            display_df[['Date', 'Description', 'Amount', 'Category']],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Date": "Date",
                "Description": "Description",
                "Amount": st.column_config.NumberColumn(
                    "Amount",
                    format="Rs.%.2f"
                ),
                "Category": "Category"
            }
        )
        
        st.download_button(
            label="📥 Download as CSV",
            data=df.to_csv(index=False),
            file_name="finance_history.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("No transactions recorded yet.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
    <div style='text-align: center; margin-top: 3rem; color: #6c757d; font-size: 0.9rem;'>
        <hr style='margin-bottom: 1rem;'>
        <p>AI Powered Finance Tracker • v1.0</p>
    </div>
""", unsafe_allow_html=True)