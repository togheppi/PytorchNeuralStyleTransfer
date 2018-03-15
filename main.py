import argparse
from stylize import NeuralStyle


def main(option):
    model = NeuralStyle(option)
    model.stylize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--image_dir', type=str, default='Images/')
    parser.add_argument('--model_dir', type=str, default='Models/')
    parser.add_argument('--result_dir', type=str, default='Result/')
    parser.add_argument('--content_img', type=str, default='photo_mountain.jpg')
    parser.add_argument('--style_img', type=str, default='artwork_mountain.jpg')
    parser.add_argument('--lr_img_size', type=int, default=512)
    parser.add_argument('--max_iter', type=int, default=500)
    parser.add_argument('--show_iter', type=int, default=50)

    option = parser.parse_args()
    print(option)
    main(option)
