import os
import logging
import re
import json
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoanRAGSystem:
    """A RAG system for loan recommendations using Google Gemini and FAISS."""
    
    def __init__(self, vectorstore_path: str = "vectorstore/db_faiss"):
        load_dotenv()
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.vectorstore_path = vectorstore_path
        
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        self._initialize_models()
        self._load_vectorstore()
        self._setup_chain()
    
    def _initialize_models(self):
        """Initialize embedding and language models."""
        try:
            self.embedding_model = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-2",
                google_api_key=self.google_api_key
            )
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-3.5-flash", 
                google_api_key=self.google_api_key, 
                temperature=0.2
            )
            logger.info("Models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    def _load_vectorstore(self):
        """Load the FAISS vectorstore."""
        try:
            self.db = FAISS.load_local(
                self.vectorstore_path, 
                self.embedding_model, 
                allow_dangerous_deserialization=True
            )
            logger.info("Vectorstore loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load vectorstore: {e}")
            raise
    
    def _get_custom_prompt(self) -> PromptTemplate:
        """Create the custom prompt template for loan recommendations with structured output."""
        template = """
        You are a financial loan recommendation expert with extensive knowledge of various loan products.
        
        INSTRUCTIONS:
        - Use ONLY the information provided in the context below.
        - Recommend the most suitable loan type(s) based on the user's specific needs.
        - Present the information in clear, concise, and structured bullet points or sub-sections.
        - If multiple loans are suitable, present them clearly, indicating their differences and benefits.
        - Be specific about eligibility requirements and documentation.
        - If information is insufficient, clearly state what additional details are needed.
        - If you can extract comparative numerical data (e.g., interest rates, loan amounts for different banks/products), please present it clearly, perhaps in a simple list or table-like format within the text, so it can potentially be parsed for a graph. Indicate the bank/product name for each data point.

        Context (Available Loan Products): 
        {context}
        
        User Query: {question}
        
        RESPONSE FORMAT:
        Please structure your response using the following markdown format:
        
        **🎯 Recommended Loan Options:**
        *   **Loan Type 1 Name:** [Brief overview of this loan type]
            *   **Key Features:**
                *   Interest Rate Range: [e.g., 8.0% - 10.5% (Bank A), 8.2% - 10.7% (Bank B)]
                *   Maximum Loan Amount: [e.g., Up to ₹50 lakhs (Bank A), Up to ₹75 lakhs (Bank B)]
                *   Tenure Options: [e.g., Up to 30 years]
            *   **Eligibility Criteria:**
                *   Age: [e.g., 21-65 years]
                *   Monthly Income: [e.g., ₹25,000 minimum]
                *   Credit Score: [e.g., 700+ preferred]
            *   **Required Documents:**
                *   [List essential documents]
            *   **Why it's suitable:** [Brief explanation]

        *   **Loan Type 2 Name (if applicable):** [Brief overview]
            *   **Key Features:** ...
            *   **Eligibility Criteria:** ...
            *   **Required Documents:** ...
            *   **Why it's suitable:** ...

        **📝 General Eligibility & Documentation (if not specific to a loan type):**
        *   [General requirements]

        **💡 Additional Considerations:**
        *   [Tips, next steps, or missing information if any]

        If the context doesn't contain sufficient information to answer the query, respond with:
        "I don't have enough information in my knowledge base to provide a specific recommendation for this query. Please provide more details about [specific missing information]."
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def _get_json_prompt(self) -> PromptTemplate:
        """Create the custom prompt template for loan recommendations with structured JSON output."""
        template = """
        You are a financial loan recommendation expert with extensive knowledge of various loan products.
        
        INSTRUCTIONS:
        - Use ONLY the information provided in the context below.
        - Recommend the most suitable loan type(s) based on the user's specific needs.
        - Respond ONLY with a valid JSON object matching the schema below. Do not include any other text or markdown outside the JSON block.

        Context (Available Loan Products): 
        {context}
        
        User Query: {question}
        
        RESPONSE SCHEMA:
        Your response must be a single JSON object with the exact fields shown below:
        {{
            "eligibility": "Eligible" or "Partially Eligible" or "Not Eligible",
            "approvalChance": "e.g., 85% or High",
            "loanType": "e.g., Home Loan, Personal Loan",
            "monthlyEMI": "e.g., ₹12,500 estimated",
            "processingTime": "e.g., 3-5 working days",
            "riskLevel": "Low" or "Medium" or "High",
            "cibilImpact": "Brief explanation of how the CIBIL score affects this recommendation",
            "maritalBenefit": "Brief explanation of any marital status benefits or impact",
            "recommendedBanks": ["Bank A", "Bank B", ...],
            "interestRates": ["Rate A (e.g. 8.5% p.a.)", "Rate B (e.g. 8.7% p.a.)", ...],
            "repaymentOptions": ["Option 1", "Option 2", ...],
            "explanation": "Detailed professional explanation and recommendations"
        }}
        """
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _setup_chain(self):
        """Setup the QA chains using Runnable API."""
        try:
            # Create retriever
            retriever = self.db.as_retriever(
                search_type="similarity",
                search_kwargs={'k': 5, 'fetch_k': 10}
            )
            
            # Create prompts
            markdown_prompt = self._get_custom_prompt()
            json_prompt = self._get_json_prompt()
            
            # Create Runnable chain: retriever -> format docs -> prompt -> LLM
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            # Use modern Runnable API
            self.qa_chain_markdown = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | markdown_prompt
                | self.llm
                | StrOutputParser()
            )

            self.qa_chain_json = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | json_prompt
                | self.llm
                | StrOutputParser()
            )
            
            # Keep self.qa_chain as alias for backward compatibility
            self.qa_chain = self.qa_chain_markdown
            logger.info("QA chains setup successfully")
        except Exception as e:
            logger.error(f"Failed to setup QA chains: {e}")
            raise

    def _extract_graph_data(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Extracts numerical data for graphing from the LLM's response.
        This is a heuristic approach and might need refinement based on actual LLM output.
        It looks for patterns like 'Bank X: Y - Z%' or 'Bank P: Up to ₹Amount'.
        """
        graph_data = []

        # Try parsing response_text as JSON first
        try:
            json_match = re.search(r"\{[\s\S]*\}", response_text)
            if json_match:
                data = json.loads(json_match.group(0))
                banks = data.get('recommendedBanks', [])
                rates = data.get('interestRates', [])
                for i, bank in enumerate(banks):
                    rate_str = rates[i] if i < len(rates) else ""
                    rate_match = re.search(r"([\d.]+)", rate_str)
                    if rate_match:
                        rate_val = float(rate_match.group(1))
                        graph_data.append({
                            'Bank/Product': bank,
                            'Metric': 'Interest Rate (%)',
                            'Value': rate_val
                        })
                if graph_data:
                    return graph_data
        except Exception as e:
            logger.warning(f"Failed to extract graph data from JSON: {e}")
        
        # Regex to find interest rates for different banks/products
        # Example: "8.0% - 10.5% (Bank A)" or "Bank B: 8.2% - 10.7%"
        # Looking for "Bank Name: Interest Rate Range" or "Interest Rate Range (Bank Name)"
        interest_rate_pattern = re.compile(
            r"(?:(\w[\w\s&.]*?):\s*)?([\d.]+\%)(?:\s*-\s*([\d.]+\%))?(?:\s*\(([\w\s&.]*?)\))?", 
            re.IGNORECASE
        )
        
        # Regex to find loan amounts for different banks/products
        # Example: "Up to ₹50 lakhs (Bank A)" or "Bank C: Max ₹75 lakhs"
        loan_amount_pattern = re.compile(
            r"(?:(\w[\w\s&.]*?):\s*(?:Up to|Max)\s*)?₹([\d,]+)\s*(?:lakhs|crores)?(?:\s*\(([\w\s&.]*?)\))?", 
            re.IGNORECASE
        )

        lines = response_text.split('\n')
        current_loan_type = "Overall" # Default for general recommendations

        for line in lines:
            if "**Loan Type" in line:
                match = re.search(r"\*\*Loan Type \d+ Name:\*\* (.+)", line)
                if match:
                    current_loan_type = match.group(1).strip()
                continue
            
            # Extract interest rates
            for match in interest_rate_pattern.finditer(line):
                bank_name = match.group(1) or match.group(4)
                if not bank_name: # Try to infer from current loan type if no specific bank
                    bank_name = current_loan_type.replace("Loan Type", "Product") 
                
                if bank_name:
                    try:
                        rate_min = float(match.group(2).replace('%', '').strip())
                        rate_max = float(match.group(3).replace('%', '').strip()) if match.group(3) else rate_min
                        graph_data.append({
                            'Bank/Product': bank_name,
                            'Metric': 'Interest Rate (Min %)',
                            'Value': rate_min
                        })
                        if rate_max != rate_min:
                             graph_data.append({
                                'Bank/Product': bank_name,
                                'Metric': 'Interest Rate (Max %)',
                                'Value': rate_max
                            })
                    except (ValueError, TypeError):
                        pass # Ignore if conversion fails

            # Extract loan amounts
            for match in loan_amount_pattern.finditer(line):
                bank_name = match.group(1) or match.group(3)
                if not bank_name:
                    bank_name = current_loan_type.replace("Loan Type", "Product")
                
                if bank_name:
                    try:
                        amount_str = match.group(2).replace(',', '')
                        amount = float(amount_str)
                        if "lakhs" in line.lower():
                            amount *= 100000
                        elif "crores" in line.lower():
                            amount *= 10000000
                        graph_data.append({
                            'Bank/Product': bank_name,
                            'Metric': 'Max Loan Amount (₹)',
                            'Value': amount
                        })
                    except (ValueError, TypeError):
                        pass # Ignore if conversion fails
        
        # Filter out generic entries if more specific ones exist
        if any(d['Bank/Product'] != "Overall" for d in graph_data):
            graph_data = [d for d in graph_data if d['Bank/Product'] != "Overall"]
            
        return graph_data


    def query(self, user_query: str) -> Dict[str, Any]:
        """Process a user query and return recommendations with sources and graph data."""
        if not user_query or not user_query.strip():
            return {
                "result": "Please provide a valid loan-related question.",
                "source_documents": [],
                "success": False,
                "graph_data": []
            }
        
        try:
            logger.info(f"Processing query: {user_query[:50]}...")
            
            # Invoke the chain with just the query string
            if "json" in user_query.lower() or "structured fields" in user_query.lower():
                result_text = self.qa_chain_json.invoke(user_query)
            else:
                result_text = self.qa_chain_markdown.invoke(user_query)
            
            # Get source documents separately
            retriever = self.db.as_retriever(search_kwargs={'k': 5})
            source_docs = retriever.invoke(user_query)
            
            # Extract graph data from the LLM's structured response
            graph_data = self._extract_graph_data(result_text)

            # Enhance response with metadata
            enhanced_response = {
                "result": result_text,
                "source_documents": [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in source_docs],
                "success": True,
                "query": user_query,
                "num_sources": len(source_docs),
                "graph_data": graph_data
            }
            
            return enhanced_response
            
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            logger.error(f"Error processing query: {error_msg}")
            return {
                "result": f"Sorry, I encountered an error while processing your query: {error_msg}",
                "source_documents": [],
                "success": False,
                "graph_data": []
            }
    
    def get_similar_documents(self, query: str, k: int = 3) -> list:
        """Get similar documents without running through the full chain."""
        try:
            return self.db.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def interactive_chat(self):
        """Start an interactive chat session."""
        print("🏦 Loan Recommendation System")
        print("=" * 50)
        print("Ask me about loans, eligibility, interest rates, or documents required.")
        print("Type 'quit', 'exit', or 'bye' to end the session.\n")
        
        while True:
            try:
                user_query = input("\n💬 Your question: ").strip()
                
                if user_query.lower() in ['quit', 'exit', 'bye', '']:
                    print("\n👋 Thank you for using the Loan Recommendation System!")
                    break
                
                print("\n🔍 Searching for relevant information...")
                response = self.query(user_query)
                
                if response["success"]:
                    print(f"\n✅ **Answer:**\n{response['result']}")
                    
                    if response["graph_data"]:
                        print("\n📊 **Extracted Data for Graphing:**")
                        for item in response["graph_data"]:
                            print(f"  - {item['Bank/Product']}: {item['Metric']} = {item['Value']}")

                    if response["source_documents"]:
                        print(f"\n📚 **Sources Used ({response['num_sources']} documents):**")
                        for i, doc in enumerate(response["source_documents"], 1):
                            title = doc.metadata.get('source', doc.metadata.get('title', 'Unknown Source'))
                            content_preview = doc.page_content[:100].replace('\n', ' ')
                            print(f"{i}. {title}")
                            print(f"   Preview: {content_preview}...")
                            print()
                else:
                    print(f"\n❌ **Error:** {response['result']}")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Unexpected error in interactive chat: {e}")
                print(f"\n❌ An unexpected error occurred: {e}")


def main():
    """Main function to run the loan RAG system."""
    try:
        print("🚀 Initializing Loan RAG System...")
        loan_system = LoanRAGSystem()
        
        loan_system.interactive_chat()
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        print(f"❌ System initialization failed: {e}")
        print("\nPlease check:")
        print("1. GOOGLE_API_KEY is set in your .env file")
        print("2. vectorstore/db_faiss directory exists and contains valid data")
        print("3. All required packages are installed")


if __name__ == "__main__":
    main()