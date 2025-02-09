from .helper import*
from typing import Dict, Any, List


class DataTransformationPipeline:
    def __init__(self, api_key: str):
        self.model = initialize_gemini(api_key)
        if self.model is None:
            print("API not valid")
        # Strategy prompt: generate a structured transformation plan.
        self.strategy_prompt = (
            "You are a senior data scientist and data transformation expert. Analyze the dataset context "
            "and column details to generate a comprehensive transformation strategy for a Pandas DataFrame 'df'. "
            "The strategy should include:\n"
            "1. Datatype handling: ensure each column is converted to the appropriate data type if needed.\n"
            "2. Categorical encoding: apply an appropriate method (e.g., one-hot, ordinal, target encoding) for categorical columns, "
            "except for those specified in the skip list.\n"
            "3. Scaling/Normalization: apply a normalization or scaling method (e.g., StandardScaler, MinMaxScaler, log transformation) "
            "to numerical columns, except for those specified in the skip list.\n\n"
            "Return the strategy in JSON format with keys 'datatype_handling', 'categorical_encoding', and 'scaling_normalisation'. "
            "For each column, include the recommended action and a brief explanation."
        )
        # Code prompt: generate production-ready Python code based on the strategy.
        self.code_prompt = (
            "Generate production-ready Python code to implement the data transformation strategy on a DataFrame named 'df'.\n"
            "Perform datatype conversions as needed, apply categorical encoding for appropriate columns (skip those in the provided skip list), "
            "and apply scaling/normalization on numerical columns (skip those provided).\n"
            "For columns with applied transformations, create new columns with a '_transformed' suffix while preserving the original columns.\n"
            "Include all necessary imports (e.g., pandas, scikit-learn) and detailed inline comments. Output only the final Python code without any extra commentary."
        )
        # Validator prompt: instruct the LLM to verify the code meets the requirements.
        self.validation_prompt = (
            "Review the following Python code for data transformation. Check for:\n"
            "1. Datatype conversions are performed where needed.\n"
            "2. Categorical encoding is applied only to columns not in the skip list.\n"
            "3. Scaling/Normalization is applied only to numerical columns not in the skip list.\n"
            "4. New columns are correctly named (with a '_transformed' suffix) and the original data is preserved.\n"
            "5. All necessary imports and comments are present.\n\n"
            "If the code is production-ready, reply with 'production-ready' or 'no errors'. Otherwise, provide specific suggestions for improvement."
        )

    def generate_transformation_code(
        self,
        dataset,
        target: str = "",
        skip_encoding: List[str] = None,
        skip_normalisation: List[str] = None,
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        if self.model is None:
            return {"error": "API not valid"}
        if skip_encoding is None:
            skip_encoding = []
        if skip_normalisation is None:
            skip_normalisation = []
        dataset_description = gen_des(dataset)
        # Build the strategy prompt with context details
        combined_strategy_prompt = "\n".join([
            self.strategy_prompt,
            f"Dataset Context: {dataset_description}",
            f"Target Variable: {target}" if target else "",
            f"Skip categorical encoding for: {skip_encoding}" if skip_encoding else "Apply encoding for all applicable categorical columns.",
            f"Skip normalization for: {skip_normalisation}" if skip_normalisation else "Apply normalization for all applicable numerical columns."
        ])
        strategy_response = self.model.generate_content(combined_strategy_prompt)
        if not strategy_response.text:
            raise ValueError("Failed to generate transformation strategy")
        strategy_text = strategy_response.text.strip()

        # Build the code prompt using the generated strategy
        combined_code_prompt = "\n".join([
            self.code_prompt,
            f"Strategy: {strategy_text}"
        ])
        code_response = self.model.generate_content(combined_code_prompt)
        if not code_response.text:
            raise ValueError("Failed to generate transformation code")
        current_code = code_response.text.strip()

        # Iteratively refine the code until it is production-ready
        iteration = 0
        while iteration < max_iterations:
            combined_val_prompt = "\n".join([self.validation_prompt, current_code])
            validation_response = self.model.generate_content(combined_val_prompt)
            if not validation_response.text:
                raise ValueError("Failed to validate generated transformation code")
            feedback = validation_response.text.lower()
            if "production-ready" in feedback or "no errors" in feedback:
                break
            else:
                refinement_prompt = "\n".join([
                    f"Feedback: {validation_response.text}",
                    "Please refine the Python code accordingly. Output only the refined Python code.",
                    f"Current Code:\n{current_code}"
                ])
                refinement_response = self.model.generate_content(refinement_prompt)
                if not refinement_response.text:
                    raise ValueError("Failed to refine transformation code after feedback.")
                current_code = refinement_response.text.strip()
                iteration += 1

        return {"code": current_code}
    
    