from .base import Action
from .base import BPE

# purpose: instantiate a standard bpe model that greedily accepts merges of highest co-frequency
# arguments: (see __init__ from base.BPE)
# prereqs: (see base.BPE)
# use methods: (see base.BPE)
# use attributes: (see base.BPE)
class GreedyBPE(BPE):
    # purpose: return a list of actions, ranked according to the current count value for each action's pair of tokens
    # arguments:
    # - batch_size: int, indicating the number of potentially-optimizing actions to rank per test batch (merge and split, each)
    # output: list of Action objects
    def get_actions(self, batch_size, _):
        return [Action(pair, type='merge', count=cnt) for pair, cnt in self._digraph.most_common(batch_size)]

    # purpose: rank a list of actions according to the system's current count value for each action's pair of tokens
    # arguments:
    # - actions: list of Action objects
    # output: list of Action objects, ordered by decreasing sorting value, such as count 
    def rank_actions(self, actions):
        return sorted(actions, reverse=True, key=lambda a: a.count)

    # purpose: halt the given training process when the largest co-frequency pair has count 1
    # arguments: NA
    # output: boolean, indicating whether or not a stopping criterion has been reached
    def do_break_early(self):
        return self.get_actions(1, 1)[0].count == 1
