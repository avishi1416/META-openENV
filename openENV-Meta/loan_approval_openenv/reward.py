def calculate_reward(state, action, task_level="easy"):
    """
    Calculate reward between 0.0 and 1.0 based on state, action, and task difficulty.
    Actions: "approve", "reject", "review"
    """
    income = state["income"]
    credit_score = state["credit_score"]
    loan_amount = state["loan_amount"]
    employment_years = state["employment_years"]
    
    if action not in ["approve", "reject", "review"]:
        return 0.0
        
    if task_level == "easy":
        # Task 1 - Easy: simple rule-based grading
        if credit_score > 650:
            correct_action = "approve"
        elif credit_score < 500:
            correct_action = "reject"
        else:
            # For simplicity in easy mode, return 0.5 for mid-range if correct action not explicitly defined
            correct_action = "review"
            
        if action == correct_action:
            return 1.0
        else:
            return 0.0
            
    elif task_level == "medium":
        # Task 2 - Medium: Multiple features
        debt_to_income = loan_amount / (income + 1)
        
        is_safe = credit_score >= 700 and debt_to_income < 0.3
        is_high_risk = credit_score < 600 or debt_to_income > 0.5
        
        if action == "approve":
            if is_safe:
                return 0.9 # Safe approval
            elif is_high_risk:
                return 0.0 # High risk approval
            else:
                return 0.5 # Moderate risk approval
        elif action == "reject":
            if is_high_risk:
                return 1.0 # Correct rejection
            elif is_safe:
                return 0.0 # Bad rejection
            else:
                return 0.5 # Moderate risk rejection
        elif action == "review":
            if not is_safe and not is_high_risk:
                return 1.0 # Correct review
            else:
                return 0.3 # Partial progress
                
    elif task_level == "hard":
        # Task 3 - Hard: Fraud and risk detection
        is_fraud_risk = employment_years <= 1 and loan_amount > 40000 and credit_score < 600
        debt_to_income = loan_amount / (income + 1)
        
        is_high_risk = credit_score < 600 or debt_to_income > 0.5
        is_safe = credit_score >= 700 and debt_to_income < 0.3
        
        if is_fraud_risk:
            if action == "reject":
                return 1.0 # Fraud prevented
            elif action == "review":
                return 0.5 # Partial, caught something suspicious
            else:
                return 0.0 # Fraud approved (Default)
                
        if action == "approve":
            if is_safe:
                return 1.0
            elif is_high_risk:
                return 0.0 # Default
            else:
                return 0.2 # Risky approval
        elif action == "reject":
            if is_high_risk:
                return 1.0
            elif is_safe:
                return 0.0
            else:
                return 0.3 # Partial progress
        elif action == "review":
            if not is_safe and not is_high_risk and not is_fraud_risk:
                return 1.0
            elif is_safe:
                return 0.0
            else:
                return 0.5 # Partial progress
    
    return 0.0
