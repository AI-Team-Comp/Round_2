import numpy as np
import random
from knu_rl_env.grid_survivor import GridSurvivorAgent, make_grid_survivor, evaluate


class GridSurvivorRLAgent(GridSurvivorAgent):
    global_agent_pos = (0, 0)
    global_goal_positions = []
    global_now_pos = 'A'

    def __init__(self, state_space_size=1000, action_space_size=3, alpha=0.3, gamma=0.95, epsilon=0.5, epsilon_decay=0.995, epsilon_min=0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma 
        self.epsilon = epsilon  
        self.epsilon_decay = epsilon_decay  
        self.epsilon_min = epsilon_min 

        self.q_table = np.zeros((state_space_size, action_space_size))

    def act(self, state, info=None):
        """에이전트의 행동을 선택"""

        # print(self.global_goal_positions)
        # 목표 위치 계산
        target = self.find_closest_goal()
        # print(target)

        # print(target)

        if target is not None:
            # 목표를 기준으로 방향 결정
            target_row, target_col = target
            agent_row, agent_col = self.global_agent_pos

            # print(target)
            # print(self.global_agent_pos)

            if target_row < agent_row and global_now_pos == 'AL':
                GridSurvivorAgent.ACTION_FORWARD
            # elif target_row < agent_row:
            #     GridSurvivorAgent.ACTION_LEFT
            elif target_row > agent_row and global_now_pos == 'AR':
                GridSurvivorAgent.ACTION_FORWARD
            # elif target_row > agent_row:
            #     GridSurvivorAgent.ACTION_RIGHT
            elif target_col < agent_col and global_now_pos == 'AU':
                GridSurvivorAgent.ACTION_FORWARD
            # elif target_col < agent_col:
            #     GridSurvivorAgent
            elif target_col > agent_col and global_now_pos == 'AD':
                GridSurvivorAgent.ACTION_RIGHT

        # 목표가 없거나 이동할 필요가 없으면 랜덤 행동 선택 (탐험)
        if random.random() < self.epsilon:
            return random.choice([
                GridSurvivorAgent.ACTION_LEFT,
                GridSurvivorAgent.ACTION_RIGHT,
                GridSurvivorAgent.ACTION_FORWARD
            ])
        else:
            # Q-테이블 기반 행동 선택
            state_idx = self.encode_state(state)
            return np.argmax(self.q_table[state_idx])
        
    def find_closest_goal(self):
        """가장 가까운 목표 위치를 계산"""
        if not self.global_goal_positions:
            return None  # 목표 위치가 없을 경우 None 반환
        closest_goal = min(self.global_goal_positions, key=lambda g: np.linalg.norm(np.array(g) - np.array(self.global_agent_pos)))
        return closest_goal

    def update_q_table(self, state, action, reward, next_state, done):
        state_idx = self.encode_state(state)
        next_state_idx = self.encode_state(next_state)

        # Q-러닝 업데이트 공식
        best_next_action = np.argmax(self.q_table[next_state_idx])
        td_target = reward + (self.gamma * self.q_table[next_state_idx, best_next_action] * (1 - done))
        td_error = td_target - self.q_table[state_idx, action]
        self.q_table[state_idx, action] += self.alpha * td_error

    def compute_reward(self, info, action):
        cell = info.get('cell', 'E')  # 현재 위치의 셀 상태
        self.global_agent_pos = info.get('agent_position')  # 에이전트 위치
        self.global_goal_positions = info.get('goal_positions')  # 'B' 위치 리스트
        dangers = info.get('dangers', [])  # 위험 요소 리스트
        
        # print(cell)
        # print(agent_pos)
        # print(self.global_goal_positions)
        reward = -0.1  # 기본 이동 페널티

        # 꿀벌을 먹으면 큰 보상
        if cell == 'B':
            reward += 500  # 꿀벌을 구출한 보상
            return reward

        # 꿀벌과의 거리 기반 보상
        if self.global_goal_positions:
            # 에이전트와 가장 가까운 'B'까지의 거리 계산
            closest_goal = min(self.global_goal_positions, key=lambda g: np.linalg.norm(np.array(g) - np.array(self.global_agent_pos)))
            distance_to_goal = np.linalg.norm(np.array(self.global_agent_pos) - np.array(closest_goal))
            # print(distance_to_goal)
            if(distance_to_goal >= 1.0 and distance_to_goal < 2.0): reward += 30
            elif(distance_to_goal >= 2.0 and distance_to_goal < 3.0): reward += 20
            elif(distance_to_goal >= 3.0 and distance_to_goal < 4.0): reward += 10
            # reward += 10 / (distance_to_goal + 1e-6)  # 거리 기반 보상

        # 위험물과의 거리 기반 페널티
        for danger in dangers:
            danger_distance = np.linalg.norm(np.array(self.global_agent_pos) - np.array(danger))
            if danger_distance < 3:  # 위험 요소가 가까울수록 페널티 증가
                reward -= 10 / (danger_distance + 1e-6)

        return reward

    def encode_state(self, state):
        return hash(str(state)) % len(self.q_table)


def train():
    env = make_grid_survivor(show_screen=False)
    agent = GridSurvivorRLAgent()

    for episode in range(200):
        state, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            # 에이전트 행동 선택
            action = agent.act(state, info)

            # 환경과 상호작용
            next_state, reward, done, _, info = env.step(action)

            grid = next_state['grid'] 

            # 'B'의 위치 찾기
            goal_positions = np.argwhere(grid == 'B')  # 'B'가 있는 위치 검색
            goal_positions = goal_positions.tolist()  # numpy 배열을 리스트로 변환


            # AL, AR, AU, AD 값 찾기
            targets = ['AL', 'AR', 'AU', 'AD']
            row = 0
            col = 0
            for target in targets:
                # 특정 값을 가진 위치 찾기
                positions = np.argwhere(grid == target)
                if len(positions) > 0:
                    for pos in positions:
                        row, col = pos
                        global global_now_pos 
                        global_now_pos = grid[row][col]

            # 보상 계산
            reward = agent.compute_reward({
                'cell': info.get('cell', 'b'),
                'agent_position': info.get('agent_position', pos),  # 에이전트 위치
                'goal_positions': goal_positions,  # 리스트 형태로 전달
                'dangers': info.get('dangers', [])
            }, action)

            # Q-테이블 업데이트
            agent.update_q_table(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state

        

        print(f"Episode {episode}: Total Reward = {total_reward}")
        print(agent.q_table)

    return agent


if __name__ == '__main__':
    agent = train()
    evaluate(agent)
