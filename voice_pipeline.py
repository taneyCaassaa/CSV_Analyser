"""
Integrated CSV Analyzer with LiveKit Voice Agent
"""
import pandas as pd
import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, RunContext, function_tool, RoomInputOptions
from livekit.plugins import openai, noise_cancellation
import json
load_dotenv()
# CSVAnalyzer class (unchanged)
class CSVAnalyzer:
    def __init__(self, csv_path: str = None, openai_api_key: str = None):
        """Initialize CSV analyzer with data and LangChain agent"""
        load_dotenv()
        
        self.csv_path = csv_path or os.getenv('CSV_PATH')
        self.api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.df = None
        self.agent = None
        
        if not self.csv_path:
            raise ValueError("CSV_PATH not provided or found in environment")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not provided or found in environment")
    
    def load_data(self) -> bool:
        """Load CSV data into DataFrame"""
        try:
            if not os.path.exists(self.csv_path):
                print(f"❌ Error: File '{self.csv_path}' not found")
                return False
            
            self.df = pd.read_csv(self.csv_path)
            print(f"✅ Loaded CSV: {len(self.df)} rows, {len(self.df.columns)} columns")
            print(f"✅ Columns: {', '.join(self.df.columns.tolist())}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return False
    
    def create_agent(self) -> bool:
        """Create LangChain DataFrame agent"""
        try:
            from langchain_experimental.agents import create_pandas_dataframe_agent
            from langchain_openai import ChatOpenAI
            
            if self.df is None:
                print("❌ No data loaded. Call load_data() first.")
                return False
            
            llm = ChatOpenAI(
                openai_api_key=self.api_key,
                model="gpt-3.5-turbo",
                temperature=0
            )
            
            self.agent = create_pandas_dataframe_agent(
                llm=llm,
                df=self.df,
                verbose=False,
                allow_dangerous_code=True,
                agent_type="tool-calling"
            )
            
            print("✅ Agent created successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error creating agent: {e}")
            return False
    
    def query(self, question: str) -> Dict[str, Any]:
        """Execute a natural language query on the data"""
        if not self.agent:
            return {
                "success": False,
                "error": "Agent not initialized. Call create_agent() first.",
                "answer": None
            }
        
        try:
            response = self.agent.invoke({"input": question})
            
            if isinstance(response, dict) and "output" in response:
                answer = response["output"]
            else:
                answer = str(response)
            
            return {
                "success": True,
                "error": None,
                "answer": answer,
                "question": question
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "answer": None,
                "question": question
            }
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get basic information about the loaded dataset"""
        if self.df is None:
            return {"error": "No data loaded"}
        
        return {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "column_names": self.df.columns.tolist(),
            "dtypes": self.df.dtypes.to_dict(),
            "sample_data": self.df.head(3).to_dict('records')
        }
    
    def get_column_summary(self, column_name: str) -> Dict[str, Any]:
        """Get summary statistics for a specific column"""
        if self.df is None:
            return {"error": "No data loaded"}
        
        if column_name not in self.df.columns:
            return {"error": f"Column '{column_name}' not found"}
        
        try:
            col_data = self.df[column_name]
            
            if col_data.dtype in ['int64', 'float64']:
                summary = {
                    "type": "numeric",
                    "count": col_data.count(),
                    "mean": col_data.mean(),
                    "std": col_data.std(),
                    "min": col_data.min(),
                    "max": col_data.max(),
                    "median": col_data.median()
                }
            else:
                summary = {
                    "type": "categorical",
                    "count": col_data.count(),
                    "unique_values": col_data.nunique(),
                    "top_values": col_data.value_counts().head(5).to_dict()
                }
            
            return summary
            
        except Exception as e:
            return {"error": str(e)}

# Integrated Assistant with CSVAnalyzer tool
class Assistant(Agent):
    def __init__(self, csv_path: str = None, openai_api_key: str = None):
        super().__init__(instructions="You are a helpful voice AI assistant that answer user queries based on CSV data, always use this to answer user queries.")
        self.csv_analyzer = CSVAnalyzer(csv_path=csv_path, openai_api_key=openai_api_key)
        if not self.csv_analyzer.load_data():
            raise ValueError("Failed to load CSV data")
        if not self.csv_analyzer.create_agent():
            raise ValueError("Failed to create CSV analyzer agent")
    
    @function_tool()
    async def analyze_csv(self, context: RunContext, query: str) -> str:
        """Analyze CSV data by answering a natural language query.
        
        Args:
            query: The question to ask about the CSV data.
        
        Returns:
            The answer to the query or an error message.
        """
        try:
            result = self.csv_analyzer.query(query)
            if result['success']:
                return result['answer']
            else:
                raise agents.ToolError(f"Failed to process query: {result['error']}")
        except Exception as e:
            raise agents.ToolError(f"Error analyzing CSV: {str(e)}")
    
    @function_tool()
    async def get_csv_info(self, context: RunContext) -> str:
        """Get basic information about the CSV dataset.
        
        Returns:
            A summary of the dataset including row count, column count, and column names.
        """
        try:
            info = self.csv_analyzer.get_data_info()
            if 'error' in info:
                raise agents.ToolError(info['error'])
            return f"Dataset has {info['rows']} rows and {info['columns']} columns. Columns: {', '.join(info['column_names'])}"
        except Exception as e:
            raise agents.ToolError(f"Error retrieving CSV info: {str(e)}")

async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        llm=openai.realtime.RealtimeModel(
            voice="coral"
        )
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer to analyze CSV data or provide dataset information."
    )

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
    
    
