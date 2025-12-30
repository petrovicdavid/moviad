class ReplayMemory:

    def __init__(self, memory_size: int):

        """
        This class manage the replay memory for continual learning strategies
        """

        self.memory_size = memory_size
        self.memory
