import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 2
        num_adversaries = 1
        num_landmarks = 2
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            if i < num_adversaries:
                agent.adversary = True
            else:
                agent.adversary = False
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.1, 0.1, 0.1])
            landmark.index = i
        # set goal landmark
        goal = np.random.choice(world.landmarks)
        goal.color = np.array([1, 0.5, 0.25])
        for i, agent in enumerate(world.agents):
            if agent.adversary:
                agent.color = np.array([0.75, 0.25, 0.25])
            else:
                agent.goal_a = goal
                agent.color = np.array([0.25, 0.75, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        # the distance to the goal
        agent_dist = [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in world.agents if not a.adversary]
        pos_rew = min(agent_dist)
        return -pos_rew

    def adversary_reward(self, agent, world):
        # keep the nearest good agents away from the goal
        agent_dist = [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in world.agents if not a.adversary]
        pos_rew = min(agent_dist)
        return pos_rew
               
    def observation(self, agent, world):
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos)
        agent_states = []
        for agent in world.agents:
            agent_states.append(agent.state.p_pos)
            agent_states.append(agent.state.p_vel)
        if not agent.adversary:
            return np.concatenate(agent_states + entity_pos + [agent.goal_a.state.p_pos])
        else:
            return np.concatenate(agent_states + entity_pos)

    def benchmark_data(self, agent, world):
        for agent in world.agents:
            if agent.adversary:
                continue
            dist = np.sqrt(np.sum(np.square(agent.state.p_pos-agent.goal_a.state.p_pos)))
            if dist < 0.1:
                occupied_landmarks = 1
            else:
                occupied_landmarks = 0
        return [occupied_landmarks]
