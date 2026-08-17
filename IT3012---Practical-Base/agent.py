# agent.py
import heapq
import random
from collections import deque


class SimpleReflexAgent:
    """A memoryless agent that reacts only to the current percept."""

    def sense_and_act(self, percept: dict) -> str:
        if percept['food_here']:
            return 'suck'
        if percept['wall_ahead']:
            return random.choice(['Left', 'Right', 'Up', 'Down'])
        return 'Up'


class ModelBasedAgent:
    """A reflex agent with a small internal model of visited cells and facing direction."""

    _LEFT_TURN = {
        (0, 1): (-1, 0),
        (-1, 0): (0, -1),
        (0, -1): (1, 0),
        (1, 0): (0, 1),
    }
    _RIGHT_TURN = {value: key for key, value in _LEFT_TURN.items()}

    def __init__(self):
        self.visited_cells = set()
        self.last_action = None
        self.current_pos_estimate = (0, 0)
        self.facing_direction = (0, 1)

    def _cell_to_left(self, pos):
        x, y = pos
        dx, dy = self._LEFT_TURN[self.facing_direction]
        return x + dx, y + dy

    def _advance(self, pos):
        x, y = pos
        dx, dy = self.facing_direction
        return x + dx, y + dy

    def sense_and_act(self, percept: dict) -> str:
        self.visited_cells.add(self.current_pos_estimate)

        if percept['food_here']:
            action = 'suck'
        elif percept['wall_ahead']:
            left_cell = self._cell_to_left(self.current_pos_estimate)
            if self.last_action == 'turn_left' or left_cell in self.visited_cells:
                action = 'turn_right'
            else:
                action = 'turn_left'
        else:
            action = 'move_forward'
            self.current_pos_estimate = self._advance(self.current_pos_estimate)

        if action == 'turn_left':
            self.facing_direction = self._LEFT_TURN[self.facing_direction]
        elif action == 'turn_right':
            self.facing_direction = self._RIGHT_TURN[self.facing_direction]

        self.last_action = action
        return action


class GreedyGridAgent(SimpleReflexAgent):
    """Compatibility agent used by the simulator."""