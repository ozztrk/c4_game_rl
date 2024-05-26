from agent import Agent

def evaluate_agent(agent, num_episodes=100):
    wins = 0
    draws = 0
    losses = 0
    
    for _ in range(num_episodes):
        agent.env.reset()
        state = agent.env.board_state.copy()
        done = False
        
        while not done:
            available_actions = agent.env.get_available_actions()
            
            # Agent's turn
            action = agent.select_action(state, available_actions, training=False)
            state, reward = agent.env.make_move(action, 'p1')
            if reward == 1:
                wins += 1
                done = True
            elif reward == -1:
                losses += 1
                done = True
            elif len(available_actions) == 0:
                draws += 1
                done = True
                
            if not done:
                # Opponent's turn (greedy agent)
                opponent_action = greedy_agent(agent.env)
                state, reward = agent.env.make_move(opponent_action, 'p2')
                if reward == 1:
                    losses += 1
                    done = True
                elif reward == -1:
                    wins += 1
                    done = True
                elif len(agent.env.get_available_actions()) == 0:
                    draws += 1
                    done = True
    
    print(f"Wins: {wins}, Draws: {draws}, Losses: {losses}")
    return wins, draws, losses

def greedy_agent(env):
    available_actions = env.get_available_actions()
    best_action = None
    best_reward = float('-inf')
    
    for action in available_actions:
        _, reward = env.make_move(action, 'p2')
        if reward > best_reward:
            best_reward = reward
            best_action = action
    
    return best_action


def main():
    # Train the agent
    agent = Agent()
    agent.train()
    agent.demo()
    # Save the trained model
    agent.save_model('policy_net.pth')

    # Evaluate the agent
    wins, draws, losses = evaluate_agent(agent, num_episodes=100)

    print(f"Wins: {wins}, Draws: {draws}, Losses: {losses}")

if __name__ == '__main__':
    main()
