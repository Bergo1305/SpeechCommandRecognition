from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax, RMSprop, Nadam, SGD


class Optimizers(object):

    @staticmethod
    def adam(*args, **kwargs):
        return Adam(*args, **kwargs)

    @staticmethod
    def adadelta(*args, **kwargs):
        return Adadelta(*args, **kwargs)

    @staticmethod
    def adagrad(*args, **kwargs):
        return Adagrad(*args, **kwargs)

    @staticmethod
    def adamax(*args, **kwargs):
        return Adamax(*args, **kwargs)

    @staticmethod
    def rmsprop(*args, **kwargs):
        return RMSprop(*args, **kwargs)

    @staticmethod
    def nadam(*args, **kwargs):
        return Nadam(*args, **kwargs)

    @staticmethod
    def sgd(*args, **kwargs):
        return SGD(*args, **kwargs)




