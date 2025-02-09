from .helper import*

class FeatureEngineeringPipeline:
    def __init__(self, api_key: str):
        self.model = initialize_gemini(api_key)
        if self.model is None:
            raise ValueError("Invalid API key")
        
        # Strategy prompt template
        self.strategy_prompt = (
            "You are an expert feature engineering strategist. Analyze this dataset context and target variable "
            "to create a comprehensive feature engineering plan. Focus exclusively on:\n"
            "- Interaction/polynomial features for numerical columns\n"
            "- Domain-specific feature creation\n"
            "- Advanced datetime feature extraction\n"
            "- Context-aware categorical encoding\n"
            "- Irrelevant feature removal (if requested)\n\n"
            "Provide the strategy in this JSON format:\n"
            "{\n"
            '"feature_creation": [{"method": "...", "columns": "...", "reason": "..."}],\n'
            '"feature_transformation": [{"method": "...", "columns": "...", "reason": "..."}],\n'
            '"columns_to_drop": ["..."]\n'
            "}"
        )

        # Code generation prompt template
        self.code_prompt = (
            "Generate production-ready Python code implementing this feature engineering strategy on DataFrame 'df'. "
            "Requirements:\n"
            "- Preserve original columns\n"
            "- Add '_engineered' suffix to new features\n"
            "- Include necessary imports\n"
            "- Detailed comments explaining each step\n"
            "- No data cleaning/preprocessing\n\n"
            "Strategy:\n{strategy}\n\n"
            "Output only the code without explanations."
        )

        # Validation prompt template
        self.validation_prompt = (
            "Validate this feature engineering code. Check:\n"
            "1. Faithful strategy implementation\n"
            "2. Proper column naming\n"
            "3. Absence of data cleaning steps\n"
            "4. Presence of safety checks\n"
            "5. Production-readiness\n\n"
            "Code:\n{code}\n\n"
            "Respond 'VALID' or list specific improvements."
        )

    def generate_features(self, dataset, target: str, drop_columns: bool = True, max_iterations: int = 3) -> dict:
        
        dataset_description = gen_des(dataset)
        if not self.model:
            return {"error": "API not valid"}
        
        try:
            # Generate strategy
            strategy_response = self.model.generate_content([
                self.strategy_prompt,
                f"Dataset Context: {dataset_description}",
                f"Target Variable: {target}",
                "Remove irrelevant columns: Yes" if drop_columns else ""
            ])
            
            if not strategy_response.text:
                return {"error": "Failed to generate strategy"}
            
            strategy = strategy_response.text.strip()

            # Generate initial code
            code_response = self.model.generate_content(
                self.code_prompt.format(strategy=strategy)
            )
            
            if not code_response.text:
                return {"error": "Failed to generate initial code"}
            
            current_code = code_response.text
            if "import" in current_code:
                current_code = current_code[current_code.index("import"):]

            # Refinement loop
            iteration = 0
            while iteration < max_iterations:
                # Validate code
                validation_response = self.model.generate_content(
                    self.validation_prompt.format(code=current_code)
                )
                
                if not validation_response.text:
                    break
                
                feedback = validation_response.text.lower()
                if "valid" in feedback:
                    break

                # Refine code
                refinement_response = self.model.generate_content([
                    f"Refinement Instructions: {validation_response.text}",
                    "Current Code:\n" + current_code,
                    "Output only the improved code without comments:"
                ])
                
                if refinement_response.text:
                    current_code = refinement_response.text.strip()
                
                iteration += 1

            return {"code": current_code}

        except Exception as e:
            return {"error": f"Pipeline failed: {str(e)}"}
