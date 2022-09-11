import argparse
import torch
from torchvision import transforms
import json
from PIL import Image
from create_model import create_model


def get_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('image_path', help='Path to the image.')
    arg_parser.add_argument('checkpoint', help='Path to the checkpoint of the model.')
    arg_parser.add_argument('--top_k', default=5, type=int, help='Top K most likely classes.')
    arg_parser.add_argument('--category_names', default='cat_to_name.json',
                            help='Path to the mapping of categories to real names.')
    arg_parser.add_argument('--gpu', action='store_true', help='If typed, gpu will be used for computations.')

    return arg_parser.parse_args()


def load_checkpoint(input_args: argparse.Namespace):
    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'

    checkpoint = torch.load(input_args.checkpoint, map_location=map_location)
    model = create_model(checkpoint['arch'], checkpoint['hidden_units'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model


def process_image(input_args: argparse.Namespace) -> Image:
    im = Image.open(input_args.image_path)
    preprocess = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    return preprocess(im)


def predict(model, processed_image: Image, input_args: argparse.Namespace):
    device = torch.device('cuda' if input_args.gpu and torch.cuda.is_available() else 'cpu')
    model.to(device)

    with torch.no_grad():
        processed_input = processed_image.unsqueeze(0).float()
        processed_input = processed_input.to(device)
        logits = model.forward(processed_input)
        output = torch.exp(logits)
        top_index, top_classes = output.topk(input_args.top_k, dim=1)
        class_to_idx_inverted = {model.class_to_idx[k]: k for k in model.class_to_idx}
        top_classes = [class_to_idx_inverted[i] for i in top_classes.cpu().numpy()[0]]

    return top_index.tolist()[0], top_classes


def print_results(prob_list: list, top_classes, input_args: argparse.Namespace) -> None:
    cat_to_name = json.load(open(input_args.category_names))
    converted_classes = [cat_to_name[i] for i in top_classes]
    for prob, class_name in zip(prob_list, converted_classes):
        print(f'The flower name is: {class_name} with {prob:.2f} confidence.')


def main() -> None:
    input_args = get_args()
    model = load_checkpoint(input_args)
    processed_image = process_image(input_args)
    prob_list, top_classes = predict(model, processed_image, input_args)
    print_results(prob_list, top_classes, input_args)


if __name__ == '__main__':
    main()
