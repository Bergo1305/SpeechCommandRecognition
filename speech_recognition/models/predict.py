import argparse
from speech_recognition.utils.evaluator import Evaluator

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Predict wav files using csv or raw input.')

    parser.add_argument('--wav_files_path',
                        type=str,
                        help='Full path to csv or wav directory',
                        default='/home/berin/PycharmProjects/voice-recognition/dataset/test/test.csv'
                        )

    parser.add_argument('--model_path',
                        type=str,
                        help='Model path for prediction',
                        default='/home/berin/PycharmProjects/voice-recognition/best_model.h5'
                        )

    parser.add_argument('--num_classes',
                        type=str,
                        help='Number of classes used in training',
                        default=35
                        )

    parser.add_argument('--testing_type',
                        type=str,
                        help='For now, it can be csv or raw. If raw is selected path of directory should be passed',
                        default='csv'
                        )

    args = parser.parse_args()

    wav_files_path = args.wav_files_path
    model_path = args.model_path
    num_classes = args.num_classes
    test_type = args.testing_type

    evaluator = Evaluator(wav_files_path, num_classes, model_path, test_type)

    evaluator.predict()

    evaluator.plot()
