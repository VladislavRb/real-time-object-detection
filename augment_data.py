import random as random


class AugmentData:
    def __init__(self,
                 x_shift,
                 y_shift,
                 zoom,
                 hue_shift,
                 saturation_shift):
        self.x_shift = x_shift
        self.y_shift = y_shift
        self.zoom = zoom
        self.hue_shift = hue_shift
        self.saturation_shift = saturation_shift

    @staticmethod
    def create_random():
        return AugmentData(x_shift=random.randint(-2, 2),
                           y_shift=random.randint(-2, 2),
                           zoom=1 - 0.1 * random.random(),
                           hue_shift=0.2 * random.random() - 0.1,
                           saturation_shift=0.2 * random.random() + 0.9)
