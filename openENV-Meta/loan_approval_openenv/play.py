import os
from environment import LoanApprovalEnv

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def play_in_terminal():
    # 'hard' mode includes all features (credit score, income, loan amount, employment years)
    env = LoanApprovalEnv(task_level="hard")
    
    while True:
        clear_terminal()
        state = env.state()
        print("=== AI Loan Approval Environment ===")
        print("           (Terminal Mode)          ")
        print("-" * 40)
        print("Applicant Details:")
        print(f"  Income:           ${state['income']:,}")
        print(f"  Credit Score:     {state['credit_score']}")
        print(f"  Loan Amount:      ${state['loan_amount']:,}")
        print(f"  Employment Years: {state['employment_years']}")
        print("-" * 40)
        
        print("\nAvailable Actions:")
        print("1. Approve")
        print("2. Reject")
        print("3. Review")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1/2/3/4): ").strip()
        
        if choice == '1':
            action = "approve"
        elif choice == '2':
            action = "reject"
        elif choice == '3':
            action = "review"
        elif choice == '4':
            print("\nExiting. Thank you for playing!")
            break
        else:
            print("\nInvalid choice. Press Enter to try again.")
            input()
            continue
            
        next_state, reward, done = env.step(action)
        
        print("\n" + "=" * 40)
        print(f"Action Taken: {action.upper()}")
        print(f"Reward Received: {reward:.2f}")
        print("=" * 40)
        
        if reward == 1.0:
            print("Excellent decision! Maximum reward achieved.")
        elif reward == 0.0:
            print("Bad decision! This resulted in a default or high risk.")
        else:
            print("Okay decision. Partial reward received.")
            
        input("\nPress Enter to evaluate the next applicant...")

if __name__ == "__main__":
    play_in_terminal()
