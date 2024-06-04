class _Constants:
    def __init__(self, s, b, c, image_resolution):
        # picture is divided on s x s grid cells
        self.s: int = s

        # amount of bounding boxes
        self.b: int = b

        # amount of classes
        self.c: int = c

        # image width in pixels
        self.image_resolution: int = image_resolution

        # amount of prediction values for a single cell
        self.cell_predictions_amount = 5 * self.b + self.c

        self.LCOORD = 5
        self.LNOOBJ = 0.5
        self.BATCH_SIZE = 16
        self.EPOCHS = 175
        self.AUGMENT_PROB = 0.35

        self.CONFIDENCE_THRESHOLD = 0.7
        self.IOU_THRESHOLD = 0.7

        self.TT_SCORE = 5
        self.TT_SCORE_CTOBB_RATIO = 0.5
        self.FF_SCORE = 0.5

        self.VOC_TRAIN_CARDINALITY = 2501
        self.VOC_VALIDATION_CARDINALITY = 2510
        self.VOC_TEST_CARDINALITY = 4952

        self.LEARNING_RATE_0 = 10
        self.LEARNING_RATE_1 = 85
        self.LEARNING_RATE_2 = 115
        self.LEARNING_RATE_3 = 145

        self.EPSILON = 1E-6

        self.GROUND_TRUTH_METRICS = 'ground truth accuracy'


constants = _Constants(7, 2, 20, 448)
