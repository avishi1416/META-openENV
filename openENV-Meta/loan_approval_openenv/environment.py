import random
from reward import calculate_reward

class LoanApprovalEnv:
    """
    OpenEnv Reinforcement Learning Environment for Loan Approval and Risk Optimization.
    """
    def __init__(self, task_level="easy"):
        """
        Initialize the environment.
        :param task_level: "easy", "medium", or "hard"
        """
        self.task_level = task_level
        self.current_state = None
        self.actions = ["approve", "reject", "review"]
        self.reset()
        
    def reset(self):
        """
        Generate random realistic customer data.
        """
        self.current_state = {
            "income": random.randint(20000, 100000),
            "credit_score": random.randint(300, 850),
            "loan_amount": random.randint(5000, 50000),
            "employment_years": random.randint(0, 10)
        }
        return self.state()
        
    def state(self):
        """
        Return current state.
        """
        return self.current_state
        
    def step(self, action):
        """
        Process the agent's action and return next_state, reward, done.
        """
        if action not in self.actions:
            raise ValueError(f"Invalid action: {action}. Must be one of {self.actions}")
            
        reward = calculate_reward(self.current_state, action, self.task_level)
        
        # In this task, every decision is terminal for the current customer (episode length = 1)
        done = True
        
        # Automatically determine the next state to mirror standard environments transitioning
        next_state = self.reset()
        
        return next_state, reward, done
