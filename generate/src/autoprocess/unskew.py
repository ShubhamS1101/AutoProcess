from .helper import*



class SkewCorrectionPipeline:
    def __init__(self, api_key: str, model_name="gemini-1.5-flash"):
        self.model_name = model_name
        self.model = initialize_gemini(api_key, model_name)
        if not self.model:
            raise ValueError("Invalid API key or model initialization failed")
        
         # Strategy prompt: Recommend optimal skew correction method.
        self.strategy_prompt = (
            "You are a senior data scientist specializing in statistical distributions. Analyze the following dataset context and "
            "column statistics to recommend an optimal skew correction method. Consider:\n"
            "- Data range and presence of zeros/negatives\n"
            "- Current skew direction and magnitude\n"
            "- Domain-specific requirements\n"
            "- Compatibility with downstream ML tasks\n\n"
            "Available methods:\n"
            "1. log_transform\n"
            "2. box_cox\n"
            "3. yeo_johnson\n"
            "4. quantile_transform\n"
            "5. custom_scaling"
        )

        # Code prompt: Generate production-ready code.
        self.code_prompt = (
            "Generate production-ready Python code implementing a skew correction method as described. Requirements:\n"
            "- Input is a clean DataFrame named 'df' (nulls and dtypes already handled)\n"
            "- For the target column '{col}', apply the recommended transformation and create a new column '{col}_unskewed'\n"
            "- Handle edge cases (e.g. zeros/negatives) if needed\n"
            "- Preserve the original data\n"
            "- Include all necessary imports and clear comments explaining your choices"
            "-don't use sample data just consider df as name of dataset and perform strategy"
        )

        # Validation prompt: Check the generated code for correctness.
        self.validation_prompt = (
            "Validate the following Python code for skew correction. Check for:\n"
            "1. Proper handling of data boundaries (e.g., zeros/negatives)\n"
            "2. Correct use of function parameters\n"
            "3. Appropriate naming for the new column\n"
            "4. Inclusion of safety checks and comments\n"
            "5. Compatibility with sklearn pipelines\n\n"
            "If the code is production-ready, respond with 'production-ready' or 'no errors'. "
            "Otherwise, provide specific feedback for necessary improvements."
        )

    def generate_skew_correction(self, dataset, column_name, max_iterations=3):
        
        dataset_description = gen_des(dataset)
        # Step 1: Generate transformation strategy.
        complete_strategy_prompt = "\n".join([
            self.strategy_prompt,
            f"Dataset Context: {dataset_description}",
            f"Column: {column_name}"
        ])
        strategy_response = self.model.generate_content(complete_strategy_prompt)
        if not strategy_response.text:
            raise ValueError("Failed to generate transformation strategy")
        
        # Step 2: Generate initial skew correction code using the strategy.
        complete_code_prompt = "\n".join([
            self.code_prompt.format(col=column_name),
            f"Strategy: {strategy_response.text}"
        ])
        code_response = self.model.generate_content(complete_code_prompt)
        if not code_response.text:
            raise ValueError("Failed to generate skew correction code")
        
        current_code = code_response.text
        iteration = 0

        # Iteratively refine the code until validation indicates production-readiness.
        while iteration < max_iterations:
            complete_validation_prompt = "\n".join([self.validation_prompt, current_code])
            validation_response = self.model.generate_content(complete_validation_prompt)
            if not validation_response.text:
                raise ValueError("Failed to validate the generated code")
            feedback_text = validation_response.text.lower()
            if "production-ready" in feedback_text or "no errors" in feedback_text:
                break
            else:
                updated_code_prompt = "\n".join([
                    f"Feedback: {validation_response.text}",
                    "Refine the previously generated Python code accordingly. Output only the refined code.",
                    f"Current Code:\n{current_code}"
                ])
                refinement_response = self.model.generate_content(updated_code_prompt)
                if not refinement_response.text:
                    raise ValueError("Failed to refine skew correction code after feedback")
                current_code = refinement_response.text
                iteration += 1

        return current_code

