import torch
import random

class Memory:

    def __init__(self, memory_size: int):

        self.memory_size = memory_size
        self.tasks_memory = {}      
        self.num_tasks = 0    
    
    def _rebalance(self):
        task_quota = self.memory_size // self.num_tasks

        for task_id in self.tasks_memory:
            while(len(self.tasks_memory) > task_quota): 
                idx = random.randrange(len(self.tasks_memory[task_id]))
                self.tasks_memory[task_id].pop(idx)
        
    def add_samples(self, task_id, samples): 
        if task_id not in self.tasks_memory.keys():
            self.tasks_memory[task_id] = []
            self.num_tasks += 1
            self.rebalance()

        task_quota = self.memory_size // self.num_tasks

        for sample in samples:
            if len(self.tasks_memory[task_id]) < task_quota:
                self.tasks_memory[task_id].append(sample)
            else:
                j = random.randint(0, self.n_seen[task_id] - 1)
                if j < task_quota:
                    self.tasks_memory[task_id][j] = sample

    def get_samples(self, n_replay_samples: int):
        samples_per_task = n_replay_samples // self.num_tasks

        samples = []

        for task_id, memory_samples in self.tasks_memory: 
            n_samples = min(samples_per_task, len(memory_samples))
            samples_idx = torch.randperm(len(memory_samples))[:n_samples]
            samples.append(self.tasks_memory[task_id][samples_idx])

        return samples    
