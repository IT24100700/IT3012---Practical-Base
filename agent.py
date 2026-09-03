# agent.py
from collections import deque
import heapq
import math
import random


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


class SearchAgent:
  """Goal-based planning agent using graph-search strategies."""

  def __init__(self):
    self.plan = []
    self.active_algo = 'AStar'  # Lab 04 default algorithm එක AStar ලෙස යොදා ඇත

  def _neighbors(self, position, walls, grid_size):
    width, height = grid_size
    wall_set = set(walls)
    directions = [
        ((0, 1), 'Up'),
        ((1, 0), 'Right'),
        ((0, -1), 'Down'),
        ((-1, 0), 'Left'),
    ]
    for (dx, dy), action in directions:
      next_pos = (position[0] + dx, position[1] + dy)
      if not (0 <= next_pos[0] < width and 0 <= next_pos[1] < height):
        continue
      if next_pos in wall_set:
        continue
      yield next_pos, action

  # ==========================================
  # Step 1.1: Heuristic Functions
  # ==========================================
  def manhattan_distance(self, pos, goal):
    """Calculates Manhattan distance: |x1 - x2| + |y1 - y2|"""
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

  def euclidean_distance(self, pos, goal):
    """Calculates Euclidean distance: sqrt((x1 - x2)^2 + (y1 - y2)^2)"""
    return math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)

  # ==========================================
  # Step 1.2: A* Search Implementation
  # ==========================================
  def astar_search(
      self, start_pos, goal_pos, walls, grid_size, heuristic_type='manhattan'
  ):
    """A* Search evaluating nodes using f(n) = g(n) + h(n).

    Tuple format: (f_cost, g_cost, current_pos, path_taken)
    """
    width, height = grid_size
    wall_set = set(walls)

    # Initial heuristic calculation for start_pos
    if heuristic_type == 'manhattan':
      h_start = self.manhattan_distance(start_pos, goal_pos)
    else:
      h_start = self.euclidean_distance(start_pos, goal_pos)

    # Initial state: g(n) = 0, f(n) = g(n) + h(n)
    pq = []
    heapq.heappush(pq, (h_start, 0, start_pos, []))
    reached_states = {}

    while pq:
      f_cost, g_cost, current_pos, path_taken = heapq.heappop(pq)

      if current_pos == goal_pos:
        return path_taken

      if current_pos in reached_states and reached_states[current_pos] <= g_cost:
        continue
      reached_states[current_pos] = g_cost

      for next_pos, action in self._neighbors(
          current_pos, wall_set, (width, height)
      ):
        g_new = g_cost + 1

        if next_pos in reached_states and reached_states[next_pos] <= g_new:
          continue

        if heuristic_type == 'manhattan':
          h_new = self.manhattan_distance(next_pos, goal_pos)
        else:
          h_new = self.euclidean_distance(next_pos, goal_pos)

        f_new = g_new + h_new
        heapq.heappush(pq, (f_new, g_new, next_pos, path_taken + [action]))

    return []

  def bfs_search(self, start_pos, goal_pos, walls, grid_size):
    width, height = grid_size
    wall_set = set(walls)
    queue = deque([(start_pos, [])])
    visited = {start_pos}

    while queue:
      position, path = queue.popleft()
      if position == goal_pos:
        return path

      for next_pos, action in self._neighbors(
          position, wall_set, (width, height)
      ):
        if next_pos in visited:
          continue
        visited.add(next_pos)
        queue.append((next_pos, path + [action]))

    return None

  def dfs_search(self, start_pos, goal_pos, walls, grid_size):
    width, height = grid_size
    wall_set = set(walls)
    stack = [(start_pos, [])]
    visited = {start_pos}

    while stack:
      position, path = stack.pop()
      if position == goal_pos:
        return path

      next_nodes = []
      for next_pos, action in self._neighbors(
          position, wall_set, (width, height)
      ):
        if next_pos in visited:
          continue
        visited.add(next_pos)
        next_nodes.append((next_pos, path + [action]))

      for child in reversed(next_nodes):
        stack.append(child)

    return None

  def ucs_search(self, start_pos, goal_pos, walls, grid_size):
    width, height = grid_size
    wall_set = set(walls)
    frontier = [(0, start_pos, [])]
    reached = {start_pos: 0}

    while frontier:
      cost, position, path = heapq.heappop(frontier)

      if position == goal_pos:
        return path

      for next_pos, action in self._neighbors(
          position, wall_set, (width, height)
      ):
        new_cost = cost + 1
        if next_pos in reached and new_cost >= reached[next_pos]:
          continue
        reached[next_pos] = new_cost
        heapq.heappush(frontier, (new_cost, next_pos, path + [action]))

    return None

  # ==========================================
  # Step 1.3: Integrating A* into sense_and_act
  # ==========================================
  def sense_and_act(self, percept: dict) -> str:
    if not self.plan:
      all_food = percept.get('all_food', [])
      if not all_food:
        return 'Up'

      start_pos = tuple(percept.get('agent_pos', (0, 0)))
      # Closest food selection
      goal_pos = min(
          all_food, key=lambda food: self.manhattan_distance(start_pos, food)
      )
      walls = percept.get('walls', [])
      grid_size = percept.get('grid_size', (10, 10))

      if self.active_algo == 'AStar':
        self.plan = (
            self.astar_search(
                start_pos,
                goal_pos,
                walls,
                grid_size,
                heuristic_type='manhattan',
            )
            or []
        )
      elif self.active_algo == 'DFS':
        self.plan = (
            self.dfs_search(start_pos, goal_pos, walls, grid_size) or []
        )
      elif self.active_algo == 'UCS':
        self.plan = (
            self.ucs_search(start_pos, goal_pos, walls, grid_size) or []
        )
      else:
        self.plan = (
            self.bfs_search(start_pos, goal_pos, walls, grid_size) or []
        )

    if not self.plan:
      return 'Up'

    return self.plan.pop(0)


if __name__ == '__main__':
  # Step 1.1 - 5: Testing Checkpoint for Heuristics
  test_agent = SearchAgent()
  m_dist = test_agent.manhattan_distance((0, 0), (3, 4))
  e_dist = test_agent.euclidean_distance((0, 0), (3, 4))
  print(f'Testing Checkpoint Manhattan (Expected: 7): {m_dist}')
  print(f'Testing Checkpoint Euclidean (Expected: 5.0): {e_dist}')