import cv2
import numpy as np
import random
from collections import deque
from PIL import Image


class SnakeGameEnv:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.grid_area = grid_size * grid_size
        self.grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
        self.clock_wise = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self.edges = [0, grid_size-1]
        self.MAX_CELL_VISIT = 2
        self.MOVE_PENALTY = 0
        self.LOOPING_PENALTY = -5
        self.COLLISION_PENALTY = -10
        self.FOOD_REWARD = 10
        self.reset()

    def generate_food(self):
        return random.choice([(x, y) for x in range(self.grid_size) for y in range(self.grid_size) if (x, y) not in self.snake])
    
    def _is_collision(self, cell):
        if (
            cell in self.snake
            or cell[0] < 0
            or cell[0] >= self.grid_size
            or cell[1] < 0
            or cell[1] >= self.grid_size
        ):
            return True
        return False

    def _update_grid(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        # Snake body cells as 1
        for r, c in self.snake:
            self.grid[r, c] = 1

        # Food cell as 2
        self.grid[self.food[0], self.food[1]] = 2

    def _body_within_radius(self, radius=5):
        head = self.snake[0]
        start_row, start_col = min(max(head[0] - radius // 2, 0), 5) , min(max(head[1] - radius // 2, 0), 5)
        end_row, end_col = start_row + radius, start_col + radius 
        
        count = 0
        for r in range(start_row, end_row):
            for c in range(start_col, end_col):
                if self.grid[r, c] == 1:
                    count += 1

        return count

    def _visit_limit_reached(self, direction):
        dr, dc = direction
        head_r, head_c = self.snake[0]
        r, c = head_r + dr, head_c + dc

        if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
            return self.visit_count[r, c] == self.MAX_CELL_VISIT
        return 0
    
    def _is_surrounded_by_body(self, direction, start=None):
        start_row, start_col = self.snake[0][0] + direction[0], self.snake[0][1] + direction[1]
        if start:
            start_row, start_col = start
        if not 0 <= start_row < self.grid_size or not 0 <= start_col < self.grid_size or self.grid[start_row, start_col] == 1: 
            return False

        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        visited = [[False] * self.grid_size for _ in range(self.grid_size)]        
        queue = deque([(start_row, start_col)])
        
        while queue:
            row, col = queue.popleft()
            visited[row][col] = True
            
            # If we reach boundary then it is not close loop
            if row == 0 or row == self.grid_size - 1 or col == 0 or col == self.grid_size - 1:
                return False 
            
            for dr, dc in moves:
                r, c = row + dr, col + dc
                if 0 <= r < self.grid_size and 0 <= c < self.grid_size and not visited[r][c] and self.grid[r, c] == 0:
                    queue.append((r, c))
        
        return True
    
    def _open_area(self, direction, start=None):
        start_row, start_col = self.snake[0][0] + direction[0], self.snake[0][1] + direction[1]
        if start:
            start_row, start_col = start       

        if not 0 <= start_row < self.grid_size or not 0 <= start_col < self.grid_size or self.grid[start_row, start_col] == 1: 
            return -1
        visited = set()
        queue = deque([(start_row, start_col)])

        area = 0
        while queue:
            r, c = queue.popleft()

            if (r, c) not in visited and 0 <= r < self.grid_size and 0 <= c < self.grid_size and self.grid[r, c] != 1:
                visited.add((r, c))
                area += 1

                # Check adjacent cells
                directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                for dr, dc in directions:
                    new_r, new_c = r + dr, c + dc
                    queue.append((new_r, new_c))

        return area

    def get_state(self):
        head = self.snake[0]
        coord_l = (head[0], head[1]-1)
        coord_r = (head[0], head[1]+1)
        coord_u = (head[0]-1, head[1])
        coord_d = (head[0]+1, head[1])

        dir_l = self.direction == (0, -1)
        dir_r = self.direction == (0, 1)
        dir_u = self.direction == (-1, 0)
        dir_d = self.direction == (1, 0)

        cur_dir = self.direction
        idx = self.clock_wise.index(cur_dir)
        cur_dir_l = self.clock_wise[(idx - 1) % 4]
        cur_dir_r = self.clock_wise[(idx + 1) % 4]
        
        self.area_l = self.area_r = 0
        if head[0] in self.edges or head[1] in self.edges:
            self.area_l = self._open_area(cur_dir_l)
            self.area_r = self._open_area(cur_dir_r)
            if self.area_l == -1 or self.area_r == -1: self.area_l = self.area_r = -1
            
        self.min_entry_area = 0.6 * (self.grid_area - self.snake_len)

        state = [
            # Danger Straight
            (dir_r and self._is_collision(coord_r)) or
            (dir_l and self._is_collision(coord_l)) or
            (dir_u and self._is_collision(coord_u)) or
            (dir_d and self._is_collision(coord_d)),

            # Danger Right
            (dir_u and self._is_collision(coord_r)) or
            (dir_r and self._is_collision(coord_d)) or
            (dir_d and self._is_collision(coord_l)) or
            (dir_l and self._is_collision(coord_u)),

            # Danger Left
            (dir_u and self._is_collision(coord_l)) or
            (dir_r and self._is_collision(coord_u)) or
            (dir_d and self._is_collision(coord_r)) or
            (dir_l and self._is_collision(coord_d)),
            
            # Check if next move head will be surrounded by snake body
            self._is_surrounded_by_body(cur_dir) and self._open_area(cur_dir) < self.min_entry_area,
            self._is_surrounded_by_body(cur_dir_l) and self._open_area(cur_dir_l) < self.min_entry_area,
            self._is_surrounded_by_body(cur_dir_r) and self._open_area(cur_dir_r) < self.min_entry_area,
            
            # Check which enclosed area is smaller
            self.area_r < self.area_l,    # Right is smaller
            self.area_l < self.area_r,    # Left is smaller

            # Check the cell visit limit
            self._visit_limit_reached(cur_dir),   # Straight
            self._visit_limit_reached(cur_dir_r),   # Right
            self._visit_limit_reached(cur_dir_l),   # Left

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food Location
            self.food[1] > head[1],     # food is right
            self.food[1] < head[1],     # food is left
            self.food[0] < head[0],     # food is up
            self.food[0] > head[0],     # food is down
            self.food[0] == head[0],    # food is in the same column
            self.food[1] == head[1]     # food is in the same row
        ]

        return np.array(state, dtype=int)
    
    def step(self, action):
        idx = self.clock_wise.index(self.direction)
        
        if action == 0:
            self.direction = self.clock_wise[idx]
        elif action == 1:
            next_idx = (idx + 1) % 4
            self.direction = self.clock_wise[next_idx]
        elif action == 2:
            next_idx = (idx - 1) % 4
            self.direction = self.clock_wise[next_idx]

        new_head = (self.snake[0][0]+self.direction[0], self.snake[0][1]+self.direction[1])

        # Check for the collisions
        if self._is_collision(new_head):
            self.done = True
            return self.COLLISION_PENALTY, self.done
        
        # Check for any cell visited extra to avoid loop movement
        self.visit_count[new_head] += 1
        if self.visit_count[new_head] > self.MAX_CELL_VISIT:
            self.done = True
            reward = self.LOOPING_PENALTY
            return reward, self.done
        
        # Check for traps formed by body and wall
        head = self.snake[0]
        if (head[0] in self.edges or head[1] in self.edges) and self.area_l != self.area_r:
            open_area = self._open_area(self.direction, new_head)
            if open_area < max(self.area_l, self.area_r):
                self.done = True
                reward = self.LOOPING_PENALTY
                return reward, self.done
        
        # Check for body loop traps 
        if self._is_surrounded_by_body(self.direction, new_head) and self._open_area(self.direction, new_head) < self.min_entry_area:
            self.done = True
            reward = self.LOOPING_PENALTY
            return reward, self.done

        # Move the snake
        self.snake.insert(0, new_head)
        
        reward = 0

        # Check if the snake ate the food
        if new_head == self.food:
            self.food = self.generate_food()
            self.score += 1
            self.snake_len += 1
            reward = self.FOOD_REWARD
            self.visit_count = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        else:
            self.snake.pop()
            # reward = self.MOVE_PENALTY
        
        self._update_grid()

        return reward, self.done
    
    def reset(self):
        mid = self.grid_size // 2
        self.snake = [(mid, mid)]
        self.direction = (0, 1)
        self.snake_len = len(self.snake)
        self.food = self.generate_food()
        self._update_grid()
        self.visit_count = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.score = 0
        self.done = False
        return self.get_state()
    
    def render(self):
        img = self.get_image()
        # Resize the RGB image to the desired dimensions
        img_resized = cv2.resize(img, (300, 300), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("SnakeDQN", np.array(img_resized))
        if cv2.waitKey(10) & 0xFF == ord('q'):
            return
    
    def get_image(self):
        # Define color mapping
        colors = {
            0: (255, 218, 51), # Snake Head
            1: (255, 0, 0),   # Snake Body(Green)
            2: (0, 255, 0)    # Food (Red)
        }

        # Create an RGB image
        rgb_frame = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)

        for x, y in self.snake:     # Fill the Snake cells
            rgb_frame[x, y, :] = colors[1]
        rgb_frame[self.food[0], self.food[1], :] = colors[2]    # Fill food cell

        rgb_frame[self.snake[0][0], self.snake[0][1]] = colors[0]

        img = Image.fromarray(rgb_frame, 'RGB')
        return np.array(img)