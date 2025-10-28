import base64
import inspect
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from mods.tools.tool_schema import tool_schema
from mods.tools.tool_types import ToolResponse

load_dotenv()

class AgroDiseaseAnalyserTool:
    """
    Tool for diagnosing crop diseases from images using GPT-4 Vision.
    Provides treatment recommendations with focus on locally available remedies.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the diagnosis tool
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o"  # GPT-4 with vision capabilities
        
    def encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 string
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded string of the image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def validate_image(self, image_path: str) -> bool:
        """
        Validate if the image is readable and in correct format
        
        Args:
            image_path: Path to image file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            img = Image.open(image_path)
            # Check if it's a reasonable image format
            if img.format not in ['JPEG', 'PNG', 'JPG', 'WEBP']:
                print(f"⚠️ Warning: Image format {img.format} might not be optimal")
            return True
        except Exception as e:
            print(f"❌ Image validation failed: {e}")
            return False
    
    def diagnose(
        self,
        image_path: str,
        additional_context: Optional[str] = None,
        language: str = "english"
    ) -> Dict:
        """
        Diagnose crop disease from an image
        
        Args:
            image_path: Path to the crop/plant image
            additional_context: Optional context (e.g., "leaves turning yellow", "noticed 3 days ago")
            language: Response language (default: english)
            
        Returns:
            Dictionary containing diagnosis results with:
                - crop_type: Identified crop
                - disease_name: Disease/pest identification
                - confidence: Confidence level
                - symptoms: Observed symptoms
                - causes: Likely causes
                - treatments: List of treatment recommendations
                - preventive_measures: Prevention tips
                - severity: Disease severity level
        """
        # Validate image
        if not self.validate_image(image_path):
            return {"error": "Invalid image file"}
        
        # Encode image
        base64_image = self.encode_image(image_path)
        
        # Construct the prompt
        system_prompt = """You are an expert agricultural pathologist specializing in crop diseases in India. 
Your role is to analyze plant images and provide accurate, actionable diagnosis and treatment recommendations.

CRITICAL INSTRUCTIONS:
1. Always identify the crop type first
2. Provide clear disease/pest identification
3. Focus on LOCALLY AVAILABLE and AFFORDABLE treatments in India
4. Recommend organic/natural remedies when possible
5. Provide both immediate and long-term solutions
6. Be specific about treatment quantities and application methods
7. Mention severity level (Mild, Moderate, Severe, Critical)

Response Format (JSON):
{
  "crop_type": "name of the crop",
  "disease_name": "specific disease or pest name",
  "confidence": "high/medium/low",
  "symptoms": ["symptom 1", "symptom 2"],
  "causes": ["cause 1", "cause 2"],
  "treatments": [
    {
      "name": "treatment name",
      "type": "organic/chemical/cultural",
      "ingredients": ["ingredient 1", "ingredient 2"],
      "application": "detailed application method",
      "frequency": "how often to apply",
      "cost_estimate": "approximate cost in INR",
      "availability": "where to find in rural areas"
    }
  ],
  "preventive_measures": ["measure 1", "measure 2"],
  "severity": "Mild/Moderate/Severe/Critical",
  "additional_notes": "any other important information"
}
"""
        
        user_prompt = f"""Analyze this crop/plant image and provide a comprehensive diagnosis.

{f'Additional Context: {additional_context}' if additional_context else ''}

Provide your response in {language}.
Focus on treatments available in Indian rural markets (local pesticide shops, organic materials, home remedies).
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"  # High detail for better diagnosis
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.3  # Lower temperature for more consistent diagnosis
            )
            
            # Extract the response
            diagnosis_text = response.choices[0].message.content
            
            # Try to parse as JSON, otherwise return raw text
            import json
            try:
                diagnosis_result = json.loads(diagnosis_text)
            except json.JSONDecodeError:
                # If not JSON, structure the text response
                diagnosis_result = {
                    "raw_diagnosis": diagnosis_text,
                    "note": "Response was not in structured format"
                }
            
            return diagnosis_result
            
        except Exception as e:
            return {
                "error": str(e),
                "status": "failed"
            }

    @tool_schema(
        description="Get a quick, concise diagnosis summary from a crop image",
        image_path_description="Path to the crop image file",
        additional_context_description="Additional context for diagnosis (optional)"
    )
    def get_quick_diagnosis(self, image_path: str, additional_context: Optional[str] = None) -> ToolResponse:
        """
        Get a quick, concise diagnosis (text summary)
        
        Args:
            image_path: Path to crop image
            additional_context: Additional context for diagnosis (optional)

        Returns:
            String with quick diagnosis summary
        """
        result = self.diagnose(image_path, additional_context=additional_context)
        
        if "error" in result:
            return f"❌ Error: {result['error']}"
        
        if "raw_diagnosis" in result:
            return result["raw_diagnosis"]
        
        # Format structured response into readable text
        summary = f"""
🌾 CROP: {result.get('crop_type', 'Unknown')}
🔍 DISEASE: {result.get('disease_name', 'Not identified')}
📊 CONFIDENCE: {result.get('confidence', 'Unknown')}
⚠️ SEVERITY: {result.get('severity', 'Unknown')}

SYMPTOMS:
{self._format_list(result.get('symptoms', []))}

RECOMMENDED TREATMENTS:
"""
        
        treatments = result.get('treatments', [])
        for i, treatment in enumerate(treatments[:3], 1):  # Show top 3 treatments
            summary += f"\n{i}. {treatment.get('name', 'Unknown treatment')}"
            summary += f"\n   Type: {treatment.get('type', 'N/A')}"
            summary += f"\n   Application: {treatment.get('application', 'N/A')}"
            summary += f"\n   Cost: ₹{treatment.get('cost_estimate', 'N/A')}"
        
        return {"diagnosis": [{
            "image_path": image_path,
            "summary": summary
        }]}
    
    def _format_list(self, items: List[str]) -> str:
        """Helper to format list items"""
        return "\n".join([f"• {item}" for item in items])
    
    # @tool_schema(
    #     description="Diagnose multiple crop images in batch",
    #     image_paths_description="List of paths to crop image files"
    # )
    def batch_diagnose(self, image_paths: List[str]) -> ToolResponse:
        """
        Diagnose multiple images in batch
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of diagnosis results
        """
        results = []
        for i, image_path in enumerate(image_paths, 1):
            print(f"Processing image {i}/{len(image_paths)}: {image_path}")
            result = self.diagnose(image_path)
            results.append({
                "image_path": image_path,
                "summary": result
            })
        return {"diagnosis": results}
    
    @classmethod
    def get_tool_definitions(cls) -> List[Dict]:
        """
        Extract all decorated methods and return their OpenAI function schemas.
        
        Returns:
            List of tool definitions in OpenAI function calling format
        """
        definitions = []
        
        # Iterate through class methods
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            # Check if method has our schema decorator
            if hasattr(method, '__tool_schema__'):
                schema = method.__tool_schema__
                definitions.append({
                    "type": "function",
                    "function": schema
                })
        
        return definitions
    
    @classmethod
    def get_method_names(cls) -> List[str]:
        """
        Get list of all decorated method names.
        
        Returns:
            List of method names that have tool_schema decorator
        """
        method_names = []
        
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if hasattr(method, '__tool_schema__'):
                method_names.append(name)
        
        return method_names


if __name__ == "__main__":
    print("\n🌾 Crop Disease Diagnosis Tool\n")
    
    # Note: These examples require actual image files and OpenAI API key
    print("⚠️ Note: Set OPENAI_API_KEY environment variable before running")
    print("⚠️ Replace image paths with actual crop images\n")
    

    tool = AgroDiseaseAnalyserTool()
    # test_image = "potato blight.jpg"
    # test_image = "wheat leaf rust nice_1.jpg"
    # test_image = "mango-scab-mango-1557912424.jpg"
    test_image = "./assets/crop_diseases/brinjal-leaf-spot.jpg"
    # summary = tool.get_quick_diagnosis(test_image, additional_context="Is this wheat leaf rust or another disease?")
    summary = tool.get_quick_diagnosis(test_image)

    print (summary)
    print (summary)
