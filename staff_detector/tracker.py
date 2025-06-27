import math
from collections import deque

class PersonTracker:
    def __init__(self, dist_threshold=50, tag_hit_required=2, max_missed=3):
        self.memory = deque(maxlen=1000)  # (cx, cy, tag_hits, last_seen_frame)
        self.dist_threshold = dist_threshold
        self.tag_hit_required = tag_hit_required
        self.max_missed = max_missed

    def is_same(self, cx1, cy1, cx2, cy2):
        return math.hypot(cx1 - cx2, cy1 - cy2) < self.dist_threshold

    def update(self, cx, cy, tag_found, frame_count):
        # This is where your provided code snippet lives
        # ... (rest of the update method logic as described previously)

        for i, (px, py, hits, last_seen) in enumerate(self.memory):
            if self.is_same(cx, cy, px, py):
                missed = frame_count - last_seen
                if missed > self.max_missed:
                    # Person re-appeared after a long absence, treat as new
                    self.memory[i] = (cx, cy, 1 if tag_found else 0, frame_count)
                else:
                    # Same person, update tracking
                    new_hits = hits + 1 if tag_found else 0
                    self.memory[i] = (cx, cy, new_hits, frame_count)

                is_staff = self.memory[i][2] >= self.tag_hit_required
                return True, is_staff, i
        
        # New person
        if tag_found:
            self.memory.append((cx, cy, 1, frame_count))
        else:
            # If no tag found for a new person, still add them to track their presence
            self.memory.append((cx, cy, 0, frame_count)) 
            
        return False, False, len(self.memory) - 1