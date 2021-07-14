from .base import Action
from .base import BPE


class GreedyBPE(BPE):
    def get_actions(self, batch_size):
        return [Action(pair, type='merge', count=cnt) for pair, cnt in self._digraph.most_common(batch_size)]

    def rank_actions(self, actions):
        return sorted(actions, reverse=True, key=lambda a: a.count)

    def do_break_early(self):
        return self.get_actions(1)[0].count == 1
