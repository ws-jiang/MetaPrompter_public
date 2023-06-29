from method.verbalizer.rep_verbalizer import RepVerbalizer


class HardVerbalizer(RepVerbalizer):
    def __init__(self, config):
        super(HardVerbalizer, self).__init__(config=config)
        self.lambdax = 0.0