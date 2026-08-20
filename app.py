import asyncio
import nest_asyncio
nest_asyncio.apply()

if not asyncio.get_event_loop().is_running():
    asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
import pandas as pd
from datetime import datetime
import re
from typing import Dict, Any, List
import logging

# Import your RAG system
try:
    from loan_rag_enhanced import LoanRAGSystem
except ImportError:
    st.error("Please ensure loan_rag_enhanced.py is in the same directory")
    st.stop()

# Configure page
st.set_page_config(
    page_title="Loan Finder & Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
    }
    .loan-card {
        border: 2px solid #235789;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }
    .user-message {
        background-color: #007bff; /* Changed to a lighter blue */
        color: white;
        margin-left: 2rem;
    }
    .assistant-message {
        background-color: #f0f2f6; /* Light gray for assistant */
        color: #333;
        margin-right: 2rem;
    }
    .stButton > button {
        width: 100%;
        height: 2.5rem;
        font-weight: bold;
        background-color: #28a745; /* Green for action buttons */
        color: white;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: #218838;
    }
    .metric-card {
        background: #235789;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #fff;
        text-align: center;
        color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .metric-card h4 {
        color: white;
    }
    .chart-container {
        margin-top: 2rem;
        padding: 1.5rem;
        background-color: #e0f2f7; /* Light blue background for charts */
        border-radius: 10px;
        border: 1px solid #a7d9eb;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = None
    if 'loan_options' not in st.session_state:
        st.session_state.loan_options = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'form_submitted' not in st.session_state:
        st.session_state.form_submitted = False
    if 'show_chat' not in st.session_state:
        st.session_state.show_chat = False

def create_user_profile_query(user_data: Dict) -> str:
    """Create a natural language query for finding suitable loan options."""
    
    age = user_data.get('age', 0)
    income = user_data.get('monthly_income', 0)
    employment = user_data.get('employment_type', '')
    loan_purpose = user_data.get('loan_purpose', '')
    loan_amount = user_data.get('loan_amount', 0)
    credit_score = user_data.get('credit_score', 'Unknown')
    existing_loans = user_data.get('existing_loans', 'No')
    work_experience = user_data.get('work_experience', 0)
    
    query = f"""
    Act as an expert financial loan advisor. Your task is to analyze the user's profile and generate a professional, easy-to-read summary of the most suitable loan options.

**User Profile:**
- **Age:** {age} years
- **Employment:** {employment}
- **Monthly Income:** ₹{income:,}
- **Work Experience:** {work_experience} years
- **Credit Score:** {credit_score}
- **Existing Loans:** {existing_loans}

**Loan Requirements:**
- **Purpose:** {loan_purpose}
- **Required Amount:** ₹{loan_amount:,}

**Output Instructions:**
1.  For each recommended loan, strictly follow the format in the template below.
2.  Use only bold markdown for labels (e.g., **Interest Rate:**). Do not use any other markdown like italics or backticks.
3.  Use simple hyphens (-) for bullet points.
4.  Do not include any conversational phrases, greetings, or concluding summaries. The output should begin directly with the first loan option.
5.  Ensure the values for 'Interest Rate' and 'Max Loan Amount' are clearly stated for easy data extraction.

**Template for Each Loan Option:**

### [Bank Name] - [Loan Type]

- **Interest Rate:** [Provide the rate or range, e.g., 8.75% - 9.50% p.a.]
- **Max Loan Amount:** [Provide the maximum possible amount, e.g., Up to ₹75,00,000]
- **Tenure:** [Provide the loan term, e.g., 5 to 30 years]
- **Key Eligibility Criteria:**
  - [Criterion 1, e.g., Minimum Age: 21 years]
  - [Criterion 2, e.g., CIBIL Score: 750+ recommended]
  - [Criterion 3, e.g., For Salaried: Min. 2 years total work experience]
- **Required Documents:**
  - [Document 1, e.g., Proof of Identity (PAN Card, Aadhaar)]
  - [Document 2, e.g., Proof of Address (Utility Bill, Passport)]
  - [Document 3, e.g., Income Proof (Last 3 months' salary slips)]
  - [Document 4, e.g., Bank Statement (Last 6 months)]
---"""
    
    return query.strip()

def render_user_form():
    """Render the user input form."""
    st.markdown("### 👤 Your Profile Information")
    
    with st.form("user_profile_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Personal Details")
            age = st.number_input("Age", min_value=18, max_value=80, value=30)
            employment_type = st.selectbox("Employment Type", 
                ["Salaried", "Self-Employed Business", "Self-Employed Professional", "Freelancer"])
            monthly_income = st.number_input("Monthly Income (₹)", min_value=10000, value=50000, step=5000)
            work_experience = st.number_input("Work Experience (years)", min_value=0, max_value=50, value=3)
        
        with col2:
            st.markdown("#### Loan Requirements")
            loan_purpose = st.selectbox("Loan Purpose", [
                "Home Purchase", "Home Construction", "Home Renovation",
                "Car Purchase", "Personal Need", "Business Expansion",
                "Education", "Medical Emergency", "Debt Consolidation"
            ])
            loan_amount = st.number_input("Required Loan Amount (₹)", 
                min_value=50000, max_value=10000000, value=1000000, step=50000)
            credit_score = st.selectbox("Credit Score Range", [
                "Don't Know", "Below 600", "600-650", "650-700", "700-750", "750-800", "Above 800"
            ])
            existing_loans = st.selectbox("Existing Loans", ["No", "Yes"])
        
        submitted = st.form_submit_button("🔍 Find My Loan Options", use_container_width=True)
        
        if submitted:
            user_data = {
                'age': age,
                'employment_type': employment_type,
                'monthly_income': monthly_income,
                'work_experience': work_experience,
                'loan_purpose': loan_purpose,
                'loan_amount': loan_amount,
                'credit_score': credit_score,
                'existing_loans': existing_loans
            }
            
            # Store user profile in session state
            st.session_state.user_profile = user_data
            st.session_state.form_submitted = True
            st.session_state.show_chat = False # Reset chat view
            st.rerun()

def display_loan_options(response_text: str, source_docs: List, graph_data: List[Dict[str, Any]]):
    """Display loan options in an attractive format."""
    st.markdown("### 🎯 Available Loan Options for You")
    
    # Display main AI response with enhanced formatting
    st.markdown("#### 💡 AI Recommendation")
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #e6f3ff 0%, #cceeff 100%); 
                color: #333; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;
                border: 1px solid #a7d9eb;">
        {response_text}
    </div>
    """, unsafe_allow_html=True)

    # Display graphs if data is available
    if graph_data:
        st.markdown("#### 📊 Loan Comparison")
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        df_graph = pd.DataFrame(graph_data)
        
        # Filter for Interest Rates
        interest_rate_data = df_graph[df_graph['Metric'].str.contains('Interest Rate')]
        if not interest_rate_data.empty:
            st.subheader("Interest Rate Comparison")
            # For interest rates, let's show both min and max if available
            interest_pivot = interest_rate_data.pivot_table(index='Bank/Product', columns='Metric', values='Value')
            if 'Interest Rate (Min %)' in interest_pivot.columns and 'Interest Rate (Max %)' in interest_pivot.columns:
                 st.bar_chart(interest_pivot[['Interest Rate (Min %)', 'Interest Rate (Max %)']])
            elif 'Interest Rate (Min %)' in interest_pivot.columns:
                st.bar_chart(interest_pivot['Interest Rate (Min %)'])
            elif 'Interest Rate (Max %)' in interest_pivot.columns:
                 st.bar_chart(interest_pivot['Interest Rate (Max %)'])


        # Filter for Max Loan Amounts
        loan_amount_data = df_graph[df_graph['Metric'] == 'Max Loan Amount (₹)']
        if not loan_amount_data.empty:
            st.subheader("Maximum Loan Amount Comparison")
            st.bar_chart(loan_amount_data.set_index('Bank/Product')['Value'])
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display sources used
    if source_docs:
        with st.expander(f"📚 View Data Sources ({len(source_docs)} documents used)", expanded=False):
            for i, doc in enumerate(source_docs, 1):
                source = doc.metadata.get('source', doc.metadata.get('title', f'Document {i}'))
                content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                
                st.markdown(f"**Source {i}: {source}**")
                st.text_area(f"Content {i}", content, height=100, key=f"source_{i}")
                st.markdown("---")
    
    # Quick actions
    st.markdown("#### 🚀 Next Steps")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💬 Ask Questions", use_container_width=True):
            st.session_state.show_chat = True
            st.rerun()
    
    with col2:
        if st.button("🔄 Modify Profile", use_container_width=True):
            st.session_state.form_submitted = False
            st.session_state.user_profile = None
            st.session_state.loan_options = None
            st.session_state.chat_history = []
            st.session_state.show_chat = False
            st.rerun()
    
    with col3:
        if st.button("📄 Get Application Guide", use_container_width=True):
            add_chat_message("Please guide me through the loan application process step by step.", is_user=True)
            st.session_state.show_chat = True # Ensure chat is visible when asking this
            st.rerun()


def render_chatbot():
    """Render the chatbot interface."""
    st.markdown("### 💬 Loan Assistant Chatbot")
    st.markdown("Ask me anything about loans, eligibility, documents, or application process!")
    
    # Display chat history
    chat_container = st.container(height=300, border=True) # Fixed height for chat history
    
    with chat_container:
        for message in st.session_state.chat_history:
            if message['is_user']:
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 You:</strong><br>
                    {message['content']}
                    <small style="opacity: 0.8; float: right; margin-top: 5px;">{message['timestamp']}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 Assistant:</strong><br>
                    {message['content']}
                    <small style="opacity: 0.6; float: right; margin-top: 5px;">{message['timestamp']}</small>
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    with st.container():
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_question = st.text_input("Type your question here...", 
                key="chat_input", placeholder="e.g., What documents do I need for home loan?")
        
        with col2:
            if st.button("Send", use_container_width=True, key="send_button"):
                if user_question.strip():
                    process_chat_message(user_question)
                    st.rerun()
    
    # Quick question buttons
    st.markdown("#### ❓ Common Questions")
    quick_questions = [
        "What documents do I need?",
        "How to improve my loan eligibility?",
        "What is the application process?",
        "How long does approval take?",
        "Can I prepay my loan?",
        "What are the charges and fees?"
    ]
    
    cols = st.columns(3)
    for i, question in enumerate(quick_questions):
        with cols[i % 3]:
            if st.button(question, key=f"quick_{i}", use_container_width=True):
                process_chat_message(question)
                st.rerun()

def add_chat_message(content: str, is_user: bool):
    """Add a message to chat history."""
    message = {
        'content': content,
        'is_user': is_user,
        'timestamp': datetime.now().strftime("%H:%M")
    }
    st.session_state.chat_history.append(message)

def process_chat_message(user_question: str):
    """Process user's chat message and get AI response."""
    # Add user message to chat
    add_chat_message(user_question, is_user=True)
    
    # Create context-aware query
    profile_context = ""
    if st.session_state.user_profile:
        profile_context = f"""
        Context about user for the current question:
        - Age: {st.session_state.user_profile['age']}
        - Employment: {st.session_state.user_profile['employment_type']}
        - Income: ₹{st.session_state.user_profile['monthly_income']:,}
        - Loan Purpose: {st.session_state.user_profile['loan_purpose']}
        - Loan Amount: ₹{st.session_state.user_profile['loan_amount']:,}
        
        User Question: {user_question}
        
        Please provide a specific answer based on the user's profile and the context provided, 
        in a clear, bullet-point format if possible.
        """
    else:
        profile_context = user_question
    
    # Get AI response
    try:
        with st.spinner("Getting answer..."):
            response = st.session_state.rag_system.query(profile_context)
            
            if response["success"]:
                add_chat_message(response["result"], is_user=False)
            else:
                add_chat_message("Sorry, I couldn't process your question. Please try again.", is_user=False)
    except Exception as e:
        add_chat_message(f"Error: {str(e)}", is_user=False)

def display_user_summary():
    """Display user profile summary."""
    if st.session_state.user_profile:
        profile = st.session_state.user_profile
        
        st.markdown("### 👤 Your Profile Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>💼 Employment</h4>
                <p>{profile['employment_type']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>💰 Income</h4>
                <p>₹{profile['monthly_income']:,}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🎯 Purpose</h4>
                <p>{profile['loan_purpose']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h4>💳 Amount</h4>
                <p>₹{profile['loan_amount']:,}</p>
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🏦 Loan Finder & Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Find your perfect loan match and get instant answers to your questions</p>', unsafe_allow_html=True)
    
    # Initialize RAG system
    if st.session_state.rag_system is None:
        with st.spinner("🤖 Initializing AI Loan Assistant..."):
            try:
                st.session_state.rag_system = LoanRAGSystem()
                st.success("✅ AI Assistant is ready!")
            except Exception as e:
                st.error(f"❌ Failed to initialize AI system: {str(e)}")
                st.error("Please check your .env file and vectorstore directory.")
                st.stop()
    
    # Main application flow
    if not st.session_state.form_submitted or st.session_state.user_profile is None:
        # Step 1: User fills the form
        render_user_form()
        
    else:
        # Step 2: Display user profile and loan options
        display_user_summary()
        
        # Get loan options if not already fetched
        if st.session_state.loan_options is None:
            with st.spinner("🔍 Analyzing your profile and finding suitable loan options..."):
                try:
                    # Create query from user profile
                    profile_query = create_user_profile_query(st.session_state.user_profile)
                    
                    # Get AI response
                    response = st.session_state.rag_system.query(profile_query)
                    
                    if response["success"]:
                        st.session_state.loan_options = {
                            'result': response["result"],
                            'source_documents': response["source_documents"],
                            'graph_data': response.get("graph_data", []) # Get graph data
                        }
                    else:
                        st.error(f"Error getting loan options: {response['result']}")
                        return
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    return
        
        # Display loan options
        if st.session_state.loan_options:
            display_loan_options(
                st.session_state.loan_options['result'],
                st.session_state.loan_options['source_documents'],
                st.session_state.loan_options.get('graph_data', [])
            )
        
        # Step 3: Chatbot interface
        if st.session_state.get('show_chat', True):
            st.markdown("---")
            render_chatbot()
    
    # Sidebar with additional information
    with st.sidebar:
        st.markdown("### ℹ️ How It Works")
        st.info("""
        1. **Enter Your Details**: Fill in your profile information.
        2. **Get Loan Options**: AI analyzes your profile against our loan database.
        3. **Ask Questions**: Chat with our AI assistant for more details.
        4. **Apply**: Get guidance on next steps.
        """)
        
        st.markdown("### 🎯 What You'll Get")
        st.success("""
        ✅ Personalized loan recommendations  
        ✅ Interest rate ranges  
        ✅ Eligibility criteria  
        ✅ Required documents  
        ✅ 24/7 AI assistance  
        ✅ Application guidance  
        ✅ Visual comparisons of loan metrics
        """)
        
        if st.session_state.user_profile:
            st.markdown("### 🔧 Actions")
            if st.button("🗑️ Clear Profile", use_container_width=True):
                st.session_state.user_profile = None
                st.session_state.loan_options = None
                st.session_state.form_submitted = False
                st.session_state.chat_history = []
                st.session_state.show_chat = False
                st.rerun()
            
            if st.button("💾 Save Profile", use_container_width=True):
                profile_data = st.session_state.user_profile
                profile_text = f"""
Saved Profile - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Age: {profile_data['age']}
Employment: {profile_data['employment_type']}
Monthly Income: ₹{profile_data['monthly_income']:,}
Work Experience: {profile_data['work_experience']} years
Loan Purpose: {profile_data['loan_purpose']}
Loan Amount: ₹{profile_data['loan_amount']:,}
Credit Score: {profile_data['credit_score']}
Existing Loans: {profile_data['existing_loans']}
"""
                st.download_button(
                    label="Download Profile",
                    data=profile_text,
                    file_name=f"loan_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()

    