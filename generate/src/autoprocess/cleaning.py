from .helper import*



class DataCleaningPipeline:
    def __init__(self, api_key: str):
        self.model = initialize_gemini(api_key)
        if self.model is None:
            print("API not valid")
        # Strategy prompt: Generate cleaning strategy for missing values, outliers, and duplicates.
        self.strategy_prompt = (
            "You are a senior data scientist specializing in data cleaning. Analyze the dataset context to "
            "recommend a comprehensive cleaning strategy. Your strategy should include:\n"
            "- Missing value handling: Recommend imputation or removal methods based on feature distribution.\n"
            "- Outlier handling: Recommend robust methods (e.g., IQR filtering, winsorization) for detecting and treating outliers.\n"
            "- Duplicate handling: Suggest criteria for identifying and removing duplicate rows.\n\n"
            "If any of these tasks are not required, exclude them from the strategy.\n\n"
            "Return the strategy in the following JSON format:\n"
            "{\n"
            '  "missing_values": { "method": "impute|remove", "parameters": { ... }, "reason": "..." },\n'
            '  "outlier_handling": { "method": "IQR|winsorization|custom", "parameters": { ... }, "reason": "..." },\n'
            '  "duplicate_handling": { "action": "drop", "parameters": { ... }, "reason": "..." },\n'
            '  "recommendations": ["Any additional advice"]\n'
            "}"
        )

        # Code prompt: Generate Python code based on the cleaning strategy.
        self.code_prompt = (
            "Generate production-ready Python code that implements the following data cleaning strategy on a Pandas DataFrame named 'df'.\n"
            "- Handle missing values as recommended (imputation/removal).\n"
            "- Detect and handle outliers using the suggested method (e.g., IQR filtering, winsorization).\n"
            "- Remove duplicate rows according to best practices.\n\n"
            "Requirements:\n"
            "- Preserve original data by adding '_cleaned' suffix to transformed columns if applicable.\n"
            "- Include necessary imports and comments explaining each step.\n\n"
            "If any task is excluded from the strategy, ensure it is not included in the code.\n"
            "Output only the final Python code without any additional commentary."
        )

        # Validation prompt: Validate the generated code.
        self.validation_prompt = (
            "Validate the following Python code for data cleaning. Check for:\n"
            "- Correct handling of missing values, outliers, and duplicates (if applicable).\n"
            "- Proper use of function parameters and safety checks.\n"
            "- Clear column naming (e.g., '_cleaned' suffix) for transformed columns.\n"
            "- Necessary imports and compatibility with sklearn pipelines if needed.\n\n"
            "If the code is production-ready, respond with 'production-ready' or 'no errors'. Otherwise, provide specific feedback for improvement."
        )

    def data_clean(self, dataset, target: str = "", outlier=True, missing=True, duplicate=True) -> dict:
        # If the API key was invalid, self.model will be None.
        dataset_description = gen_des(dataset)
        if not self.model:
            return {"error": "API not valid"}

        # --- Strategy Generation ---
        combined_strategy_prompt = "\n".join([
            self.strategy_prompt,
            f"Dataset Context: {dataset_description}",
            f"Target Variable: {target}" if target else "",
            f"Tasks to include: {'Missing Values' if missing else ''}, {'Outliers' if outlier else ''}, {'Duplicates' if duplicate else ''}"
        ])
        strategy_response = self.model.generate_content(combined_strategy_prompt)
        if not strategy_response.text:
            raise ValueError("Failed to generate a cleaning strategy.")
        strategy_text = strategy_response.text.strip()

        # --- Code Generation ---
        combined_code_prompt = "\n".join([
            self.code_prompt,
            f"Strategy: {strategy_text}"
        ])
        code_response = self.model.generate_content(combined_code_prompt)
        if not code_response.text:
            raise ValueError("Failed to generate cleaning code.")
        current_code = code_response.text.strip()
        if "import" in current_code:
            current_code = current_code[current_code.index("import"):]

        # --- Iterative Validation Loop (silently refine until production-ready) ---
        iteration = 0
        max_iterations = 3
        while iteration < max_iterations:
            combined_val_prompt = "\n".join([self.validation_prompt, current_code])
            validation_response = self.model.generate_content(combined_val_prompt)
            if not validation_response.text:
                raise ValueError("Failed to validate the generated cleaning code.")
            feedback = validation_response.text.lower()

            # Exit loop if the code is production-ready
            if "production-ready" in feedback or "no errors" in feedback:
                break
            else:
                refinement_prompt = "\n".join([
                    f"Feedback: {validation_response.text}",
                    "Refine the Python code accordingly. Output only the refined Python code without any additional commentary.",
                    f"Current Code:\n{current_code}"
                ])
                refinement_response = self.model.generate_content(refinement_prompt)
                if not refinement_response.text:
                    raise ValueError("Failed to refine the cleaning code after feedback.")
                current_code = refinement_response.text.strip()
                if "import" in current_code:
                    current_code = current_code[current_code.index("import"):]
                iteration += 1

        return {"code": current_code}
    
    

    