import argparse
import torch
from torch import optim, nn
from torchvision import datasets, transforms
from create_model import create_model


def get_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('data_dir', help='Name of the base directory for datasets.')
    arg_parser.add_argument('--save_dir', default='checkpoint.pth', help='Path to which the model will be saved.')
    arg_parser.add_argument('--arch', default='vgg13', help='Architecture of the CNN.')
    arg_parser.add_argument('--learning_rate', default=0.001, type=float,
                            help='The scaling factor for optimizing the network.')
    arg_parser.add_argument('--hidden_units', default=1024, type=int, help='Number of nodes in the hidden layer.')
    arg_parser.add_argument('--epochs', default=5, type=int, help='Amount of epochs')
    arg_parser.add_argument('--gpu', action='store_true', help='If typed, gpu will be used for computations.')

    return arg_parser.parse_args()


def get_dataloaders(data_dir: str):
    train_dir = data_dir + '/train'
    val_dir = data_dir + '/valid'

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean, std)])

    val_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean, std)])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)
    class_to_idx = train_dataset.class_to_idx

    return train_dataloader, val_dataloader, class_to_idx


def save_checkpoint(model, input_args: argparse.Namespace):
    torch.save(model.checkpoint, input_args.save_dir)


def train(model, train_dataloader, val_dataloader, class_to_idx, input_args: argparse.Namespace):
    optimizer = optim.Adam(model.classifier.parameters(), lr=input_args.learning_rate)
    criterion = nn.NLLLoss()

    device = torch.device('cuda' if input_args.gpu and torch.cuda.is_available() else 'cpu')

    model.to(device)

    epochs = input_args.epochs
    steps = 0
    print_divisor = 10
    running_loss = 0
    for e in range(epochs):
        for images, labels in train_dataloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            logit = model.forward(images)
            loss = criterion(logit, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if steps % print_divisor == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in val_dataloader:
                        images, labels = images.to(device), labels.to(device)
                        logit = model.forward(images)
                        batch_loss = criterion(logit, labels)
                        validation_loss += batch_loss.item()
                        output = torch.exp(logit)
                        top_p, top_class = output.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {e + 1}/{epochs}\n"
                      f"Train loss: {running_loss / print_divisor:.3f}.. "
                      f"Validation loss: {validation_loss / len(val_dataloader):.3f}.. "
                      f"Validation accuracy: {accuracy / len(val_dataloader):.3f}")
                running_loss = 0
                model.train()
    model.checkpoint = {
        'arch': input_args.arch,
        'hidden_units': input_args.hidden_units,
        'state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'epochs': epochs
    }

    return model


def main():
    input_args = get_args()
    model = create_model(input_args.arch, input_args.hidden_units)
    train_dl, val_dl, class_to_idx = get_dataloaders(input_args.data_dir)
    model = train(model, train_dl, val_dl, class_to_idx, input_args)
    save_checkpoint(model, input_args)


if __name__ == '__main__':
    main()
