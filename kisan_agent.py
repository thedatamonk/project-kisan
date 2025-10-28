
import os
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from openai import OpenAI

from mods.tools.agro_scheme_analyser import AgroSchemeAnalyserTool
from mods.tools.agro_market_analyser import AgroMarketAnalyserTool
from mods.tools.agro_disease_analyser import AgroDiseaseAnalyserTool

@dataclass
class AgentThought:
    """Represents agent's thought process for debugging"""
    timestamp: str
    step: str
    reasoning: str
    action: str
    details: Optional[Dict] = None

class ProjectKisanAgent:
    """
    Main orchestration agent for Project Kisan.
    Coordinates multiple tools to help farmers with crop diseases, market prices, and government schemes.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        data_gov_api_key: Optional[str] = None,
        debug_mode: bool = True
    ):
        """
        Initialize the Project Kisan Agent
        
        Args:
            openai_api_key: OpenAI API key for LLM and embeddings
            data_gov_api_key: data.gov.in API key for market prices
            debug_mode: If True, shows agent's thought process
        """
        self.client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-5"
        self.debug_mode = debug_mode

        # Initialize tools
        self.agro_market_price_tools = AgroMarketAnalyserTool()
        self.agro_disease_diagnosis_tools = AgroDiseaseAnalyserTool()
        self.agro_schemes_analyser_tools = AgroSchemeAnalyserTool()
        
        # Tool registry
        self.tool_registry = {}
        self._register_tool_class(AgroMarketAnalyserTool, self.agro_market_price_tools)
        self._register_tool_class(AgroDiseaseAnalyserTool, self.agro_disease_diagnosis_tools)
        self._register_tool_class(AgroSchemeAnalyserTool, self.agro_schemes_analyser_tools)

        # Conversation history
        self.conversation_history: List[Dict] = []
        
        # Thought process tracking
        self.thought_log: List[AgentThought] = []
        
        # System prompt
        self.system_prompt = self._create_system_prompt()
        
        # Initialize conversation
        self._initialize_conversation()
        
        # Tool definitions for OpenAI function calling
        self.tool_definitions = self._create_tool_definitions()

    def _register_tool_class(self, tool_class, instance):
        """
        Register all decorated methods from a tool class.
        
        Args:
            tool_class: The tool class (e.g., CompanyTools)
            instance: Instance of the tool class
        """
        for method_name in tool_class.get_method_names():
            self.tool_registry[method_name] = (instance, method_name)

    def _create_system_prompt(self) -> str:
        """Create the system prompt that defines agent behavior"""
        return """You are Kisan Mitra (Farmer's Friend), a helpful and friendly agricultural advisor for Indian farmers.

YOUR IDENTITY:
- You understand the challenges faced by small-scale farmers in India
- You speak in simple, easy-to-understand language
- You are patient, encouraging, and supportive
- You provide practical, actionable advice

YOUR CAPABILITIES:
You have access to three powerful tools:

1. **get_commodity_price**: Get current market prices for crops in different mandis
   - Use when farmers ask about prices, rates, or market values
   - Requires: commodity name and location
   - Example: "What's the tomato price in Bangalore?"

2. **diagnose_crop_disease**: Analyze plant images to identify diseases and recommend treatments
   - Use when farmers describe plant problems or upload images
   - Requires: image (if available) and description of symptoms
   - Example: "My tomato plant has yellow spots"

3. **search_government_schemes**: Find relevant agricultural subsidies and government support
   - Use when farmers ask about financial help, subsidies, loans, or schemes
   - Requires: description of what they need help with
   - Example: "Subsidy for buying drip irrigation?"

TOOL USAGE GUIDELINES:
✅ Ask for missing information before calling tools (like commodity name or location)
✅ Use multiple tools together when the query needs it (automatic chaining)
✅ Explain what you're doing and why (transparency)
✅ If a query doesn't need tools, answer directly from your knowledge

RESPONSE STYLE:
🌾 Start with a friendly greeting if it's the first message
💬 Use conversational language (like talking to a friend)
📝 Break down complex information into simple points
✨ End with helpful suggestions or follow-up questions
🚫 Avoid technical jargon - explain like you're talking to someone without formal education
🎯 Always be specific and actionable

EXAMPLES OF GOOD RESPONSES:
❌ Bad: "The current price of Solanum lycopersicum in the wholesale market is ₹2500/quintal"
✅ Good: "Right now, tomatoes are selling for about ₹25 per kg in Bangalore mandi. That's a decent price this week!"

❌ Bad: "Symptoms indicate fungal pathogen infection. Apply fungicide immediately."
✅ Good: "It looks like your tomato plant has a fungal disease called early blight. Don't worry, this is common and treatable! You can spray neem oil (available at any pesticide shop for around ₹50-100) mixed with water once a week."

WHEN CHAINING TOOLS:
When you need to use multiple tools, ALWAYS explain your plan first:
"Let me help you with that! I'll do two things:
1. First, diagnose what's wrong with your plant
2. Then, check if there are any government insurance schemes that can help

Give me a moment..."

Remember: You're here to make farming easier and more profitable. Be their trusted advisor! 🌾"""

    def _create_tool_definitions(self) -> List[Dict]:
        """
        Collect tool definitions from all registered tool classes.
        
        Returns:
            List of OpenAI function definitions
        """
        all_definitions = []
        
        # Collect all tool definitions
        all_definitions.extend(AgroMarketAnalyserTool.get_tool_definitions())
        all_definitions.extend(AgroDiseaseAnalyserTool.get_tool_definitions())
        all_definitions.extend(AgroSchemeAnalyserTool.get_tool_definitions())
        
        return all_definitions

    def _initialize_conversation(self):
        """Initialize conversation with system prompt"""
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]

    def _log_thought(self, step: str, reasoning: str, action: str, details: Optional[Dict] = None):
        """Log agent's thought process for debugging"""
        thought = AgentThought(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            step=step,
            reasoning=reasoning,
            action=action,
            details=details
        )
        self.thought_log.append(thought)
        
        if self.debug_mode:
            print(f"\n🧠 [{thought.timestamp}] {thought.step}")
            print(f"   💭 Reasoning: {thought.reasoning}")
            print(f"   ⚡ Action: {thought.action}")
            if thought.details:
                print(f"   📋 Details: {json.dumps(thought.details, indent=6)}")

    def _execute_tool(self, tool_name: str, arguments: Dict) -> Any:
        """
        Execute a specific tool and return results
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments as dictionary
            
        Returns:
            Tool execution results
        """
        self._log_thought(
            step=f"Executing Tool: {tool_name}",
            reasoning=f"User's query requires {tool_name}",
            action=f"Calling {tool_name} with parameters",
            details=arguments
        )

        # Get tool instance and method name
        tool_instance, method_name = self.tool_registry[tool_name]

        # Call the tool method directly
        try:
            tool_method = getattr(tool_instance, method_name)
            tool_result = tool_method(**arguments)
            self._log_thought(
                step=f"Tool {tool_name} Execution Complete",
                reasoning="Tool executed successfully",
                action="Returning tool results",
                details={"result": tool_result}
            )
            return json.dumps(tool_result)
        except AttributeError:
            error_msg = f"Method '{method_name}' not found on tool"
            raise Exception(error_msg)
        except TypeError as e:
            error_msg = f"Invalid parameters for {tool_name}: {str(e)}"
            raise Exception(error_msg)
    
    def chat(self, user_message: str, image_path: Optional[str] = None) -> dict[str, Any]:
        """
        Main chat interface for text-based interaction
        
        Args:
            user_message: User's text message
            image_path: Optional path to uploaded image
            
        Returns:
            Agent's response
        """


        self._log_thought(
            step="New User Message",
            reasoning="Processing incoming user query",
            action="Adding to conversation history",
            details={"message": user_message, "has_image": image_path is not None}
        )

        # Prepare user message
        if image_path:
            # TODO: Handle image uploads
            # For image messages, we'd need to encode it
            # For now, just note that an image was provided
            user_content = f"{user_message}\n[Image uploaded: {image_path}]"
        else:
            user_content = user_message
        
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_content
        })
        
        # Call OpenAI with tools
        self._log_thought(
            step="Calling LLM",
            reasoning=f"Requesting {self.model.capitalize()} to analyze query and decide on tool usage",
            action="Sending request to OpenAI API",
            details={"model": self.model, "tools_available": len(self.tool_definitions)}
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation_history,
            tools=self.tool_definitions,
            tool_choice="auto"  # Let GPT decide
        )

        assistant_message = response.choices[0].message
        
        # Check if tools were called
        if assistant_message.tool_calls:
            self._log_thought(
                step="Tool Usage Decision",
                reasoning=f"GPT-4 determined that {len(assistant_message.tool_calls)} tool(s) are needed",
                action="Preparing to execute tools",
                details={
                    "num_tools": len(assistant_message.tool_calls),
                    "tools": [tc.function.name for tc in assistant_message.tool_calls]
                }
            )
            
            # Show what we're doing (transparency for farmer)
            if self.debug_mode and len(assistant_message.tool_calls) > 1:
                print("\n🔗 TOOL CHAINING DETECTED")
                print(f"   I need to use {len(assistant_message.tool_calls)} tools:")
                for i, tc in enumerate(assistant_message.tool_calls, 1):
                    print(f"   {i}. {tc.function.name}")
                print()
            
            # Add assistant's tool calls to conversation
            self.conversation_history.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in assistant_message.tool_calls
                ]
            })
            
            # Execute all tool calls
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                # Execute tool
                tool_result = self._execute_tool(tool_name, tool_args)
                
                # Add tool result to conversation
                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })
            
            # Get final response after tool execution
            self._log_thought(
                step="Generating Final Response",
                reasoning="All tools executed successfully, synthesizing final answer",
                action="Calling LLM to generate farmer-friendly response",
                details=None
            )
            
            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history
            )
            
            final_message = final_response.choices[0].message.content
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": final_message
            })
            
            return {"role": "KisanAgent", "content": final_message}
        
        else:
            # No tools needed - direct response
            self._log_thought(
                step="Direct Response",
                reasoning="No tools needed - answering from general knowledge",
                action="Returning response directly",
                details=None
            )
            
            response_content = assistant_message.content
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": response_content
            })
            
            return response_content
        
    def reset_conversation(self):
        """Reset conversation history"""
        self._log_thought(
            step="Conversation Reset",
            reasoning="User requested or session ended",
            action="Clearing conversation history",
            details=None
        )
        self._initialize_conversation()
        self.thought_log = []
        print("🔄 Conversation reset. Starting fresh!\n")
    
    def get_conversation_history(self) -> List[Dict]:
        """Get the conversation history"""
        return self.conversation_history
    
    def get_thought_log(self) -> List[AgentThought]:
        """Get the agent's thought process log"""
        return self.thought_log
    
    def print_thought_summary(self):
        """Print a summary of the agent's thought process"""
        print("\n" + "="*60)
        print("🧠 AGENT THOUGHT PROCESS SUMMARY")
        print("="*60)
        
        for i, thought in enumerate(self.thought_log, 1):
            print(f"\n{i}. [{thought.timestamp}] {thought.step}")
            print(f"   Reasoning: {thought.reasoning}")
            print(f"   Action: {thought.action}")
        
        print("\n" + "="*60 + "\n")

# Example usage and demonstration
def example_simple_query():
    """Example 1: Simple single-tool query"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Simple Price Query (Single Tool)")
    print("="*60 + "\n")
    
    agent = ProjectKisanAgent(debug_mode=True)
    
    response = agent.chat("What's the price of tomatoes in Karnataka?")
    print("\n📱 AGENT RESPONSE:")
    print(response)
    print()


def example_multi_tool_chaining():
    """Example 2: Multi-tool automatic chaining"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Multi-Tool Query (Automatic Chaining)")
    print("="*60 + "\n")
    
    agent = ProjectKisanAgent(debug_mode=True)
    
    response = agent.chat(
        "My tomato plant has yellow spots. Can you diagnose it and also tell me if there's any insurance scheme available?"
    )
    print("\n📱 AGENT RESPONSE:")
    print(response)
    print()


def example_clarification_needed():
    """Example 3: Query needing clarification"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Clarification Needed")
    print("="*60 + "\n")
    
    agent = ProjectKisanAgent(debug_mode=True)
    
    # First query - missing information
    response1 = agent.chat("What's the price?")
    print("\n📱 AGENT RESPONSE:")
    print(response1)
    
    # Follow-up with complete information
    response2 = agent.chat("Tomato in Bangalore")
    print("\n📱 AGENT RESPONSE:")
    print(response2)
    print()


def example_no_tools_needed():
    """Example 4: Direct response without tools"""
    print("\n" + "="*60)
    print("EXAMPLE 4: No Tools Needed (Direct Response)")
    print("="*60 + "\n")
    
    agent = ProjectKisanAgent(debug_mode=True)
    
    response = agent.chat("Hello! Can you help me with farming?")
    print("\n📱 AGENT RESPONSE:")
    print(response)
    print()


def example_conversation_flow():
    """Example 5: Multi-turn conversation"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Multi-Turn Conversation")
    print("="*60 + "\n")
    
    agent = ProjectKisanAgent(debug_mode=True)
    
    # Turn 1
    print("👨‍🌾 Farmer: Hello!")
    response1 = agent.chat("Hello!")
    print(f"🤖 Agent: {response1}\n")
    
    # Turn 2
    print("👨‍🌾 Farmer: I want to start organic farming. What help is available?")
    response2 = agent.chat("I want to start organic farming. What help is available?")
    print(f"🤖 Agent: {response2}\n")
    
    # Turn 3
    print("👨‍🌾 Farmer: What about the prices for organic vegetables?")
    response3 = agent.chat("What about the prices for organic vegetables in Maharashtra?")
    print(f"🤖 Agent: {response3}\n")
    
    # Show thought summary
    agent.print_thought_summary()


if __name__ == "__main__":
    print("\n🌾 PROJECT KISAN - AGENT ORCHESTRATOR")
    print("=" * 60)
    print("Agent with automatic tool chaining and thought process visibility")
    print("=" * 60 + "\n")
    
    print("⚠️  NOTE: Set OPENAI_API_KEY environment variable")
    print("⚠️  Tool implementations are mocked for demonstration")
    print("⚠️  Uncomment tool initializations to use real implementations\n")
    
    # Run examples
    # example_simple_query()
    example_multi_tool_chaining()
    # example_clarification_needed()
    # example_no_tools_needed()
    # example_conversation_flow()


