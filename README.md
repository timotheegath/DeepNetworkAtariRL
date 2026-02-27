# LTworkinprogress

**🎮 Project Overview**
- Explains this is a 2017 implementation of Deep Q-Network (DQN) for training an AI agent to play Doom & Atari games
- This was to assist research work at the Department of Bioengineering at Imperial College London.
- Uses reinforcement learning to learn optimal action policies.

**🏗️ Architecture**
- **DQN Neural Network**: Multi-head architecture with:
  - Shared convolutional layers for processing visual input (8-channel 64×64 images)
  - Two specialized action branches (attack & low-health scenarios)
  - Classifier head to dynamically switch between strategies
  - Separate linear layer for game variables (health, ammo, etc.)

**📚 Key Features**
- Dueling multi-head network for scenario-specific Q-learning
- Experience replay memory (100k capacity)
- Epsilon-greedy exploration strategy
- RMSprop optimizer for training
- Batch size of 64 with gamma discount of 0.9
- Real-time performance visualization

**🛠️ Dependencies**
- PyTorch
- ViZDoom (Doom game environment)
- Matplotlib for visualization

**🚀 Usage**
- Instructions for initializing the game environment
- Training loop details (700 episodes)
- Performance testing and Q-value visualization
