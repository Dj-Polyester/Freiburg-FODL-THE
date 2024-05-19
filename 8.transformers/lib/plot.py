import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch
import torch.nn as nn
import torchvision
from lib.counting import CountingModel
from lib.dataset import CountingDataset
from lib.utils import create_dataloader, calculate_jacobian

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Running on device: {device}")

with open(os.path.join("data", "class_names.json")) as json_file:
    class_names = json.load(json_file)


def get_subtitle(prediction) -> str:
    """
    Given the prediction tensor, calculates probabilities for each class and forms a string which lists
    the 3 classes which are classified as most likely by the neural network
    :param prediction: A tensor of shape (1, 1000)
    :return: The string to be used as subtitle in figures
    """
    p = torch.softmax(prediction, dim=-1)
    p_values, indices = torch.topk(p, 3)
    subtitle = "Predicted classes: \n"
    for probability, index in zip(
        p_values.detach().numpy()[0], indices.detach().numpy()[0]
    ):
        subtitle += f"{class_names[str(index)]}, p={probability:.2f} \n"
    return subtitle


def show_attention_on_input_image(net) -> None:
    torch.device("cpu")
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    dataloader = create_dataloader(1)
    net.eval()  # we fix the batch-statistics to get better results

    for i, (data, target) in enumerate(dataloader):
        if i in [
            13,
            217,
            331,
            366,
        ]:  # these are the samples we use for demonstration purposes,
            # feel free to experiment with others
            data, target = data.to(device), target.to(device)
            gradient, output = calculate_jacobian(data, target, net, criterion)

            # calculate the L2-norm of the gradient w.r.t. input along the channel axis and then
            # normalize by linearly rescaling values between 0 and 1
            # the normalized gradient will have shape (Height, Width) and values ranging from 0 (min) to 1 (max)
            # START TODO #############
            # Get rid of dimension which has only 1 element
            normalized_gradient = torch.squeeze(gradient)
            # Vector norm is l2 norm by default, dim=0 indicates the channel axis after squeeze
            normalized_gradient = torch.linalg.vector_norm(normalized_gradient, dim=0)
            # Normalize between 0,1
            normalized_gradient = (normalized_gradient - normalized_gradient.min()) / (
                normalized_gradient.max() - normalized_gradient.min()
            )
            # END TODO #############

            fig, ax = plt.subplots(1, 2)
            title = get_subtitle(output)
            fig.suptitle(f"Classification of a/an {class_names[str(i)]}\n" + title)
            img = torchvision.utils.make_grid(data[:, :, :, :])
            img = img / 2 + 0.5
            npimg = img.detach().numpy()
            ax[0].imshow(np.transpose(npimg, (1, 2, 0)))
            ax[0].set_title("original image")
            grad = ax[1].imshow(normalized_gradient)
            ax[1].set_title("normalized gradients")
            plt.colorbar(grad, ax=ax[1], shrink=0.5)
            plt.tight_layout()
            save_folder = "results"
            save_path = os.path.join(save_folder, f"res18_img_and_grad{i:04d}.jpg")
            os.makedirs(save_folder, exist_ok=True)
            plt.savefig(save_path)
            plt.show()
            # fig.clear()


def show_adversarial_example(net) -> None:
    torch.device("cpu")
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    dataloader = create_dataloader(1)
    net.eval()  # we fix the batch-statistics to get better results

    for i, (data, target) in enumerate(dataloader):
        if i in [
            2,
            30,
            145,
        ]:  # these are the samples we use for demonstration purposes,
            # feel free to experiment with others
            try:
                orig_data, target = data.to(device), target.to(device)
                gradient, output = calculate_jacobian(orig_data, target, net, criterion)
                # START TODO #############
                # augment the original data by adding the gradient w.r.t. the input to get an adversarial example
                # scale the gradient upwards to increase its contribution
                scale = 100
                adversarial_example = orig_data + scale * gradient

                # END TODO #############
                orig_title = get_subtitle(output)
                img = torchvision.utils.make_grid(orig_data[:, :, :, :])
                adv_img = torchvision.utils.make_grid(adversarial_example[:, :, :, :])
                adv_output = net(adversarial_example)
                adv_title = get_subtitle(adv_output)
                fig, ax = plt.subplots(1, 2)
                fig.suptitle(f"Original class: {class_names[str(i)]}")
                img = img / 2 + 0.5
                npimg = img.detach().numpy()
                adv_img = adv_img / 2 + 0.5
                np_adv_img = adv_img.detach().numpy()
                ax[0].imshow(np.transpose(npimg, (1, 2, 0)))
                ax[0].set_title("Original image \n " + orig_title)
                ax[1].imshow(np.transpose(np_adv_img, (1, 2, 0)))
                ax[1].set_title("Adversarial example \n " + adv_title)
                plt.tight_layout()
                save_folder = "results"
                save_path = os.path.join(
                    save_folder, f"adversarial18_img_and_grad{i:04d}.jpg"
                )
                os.makedirs(save_folder, exist_ok=True)
                plt.savefig(save_path)
                plt.show()
            except RuntimeError:
                print(
                    "runtime error \n pytorch keeps track of which tensors require gradients and will"
                    "need much more computing power if you input tensors into a neural network that already have"
                    "gradient information"
                )
                pass


def plot_attention():
    # Setup
    torch.manual_seed(0)
    torch.device("cpu")
    np.random.seed(0)
    sequence_length = 10
    vocab_size = 3
    hidden = 8
    # Instantiate a Counting Model
    model = CountingModel(vocab_size, sequence_length, hidden)
    # Get a testing example
    x, y = CountingDataset(vocab_size, sequence_length, 1)[:]
    # Load model parameters and predict
    model.load_state_dict(torch.load(os.path.join("results", "model.pth")))
    model.eval()
    predictions, attention = model(x)
    predictions = predictions.detach().numpy()
    predictions = predictions.argmax(axis=2)
    # Output of attention with shape (1, vocab_size, sequence_length)
    attention = attention.detach().numpy()
    np.set_printoptions(precision=3)

    print(f"Input: {x.argmax(axis=2)}\n")
    print(f"Prediction: {predictions}\n")

    for i, output_step in enumerate(attention[0]):
        steps = np.where(output_step == np.max(output_step))[0]
        print(f"Output step {i} attended mainly to Input steps: {steps}")
        print(f"Attention: {output_step}\n")

    fig, ax = plt.subplots()
    plot_kwargs = {
        "xticklabels": np.arange(sequence_length),
        "yticklabels": ["Output-0", "Output-1", "Output-2"],
        "cmap": "plasma",
        "cbar": True,
        "cbar_kws": {"orientation": "horizontal"},
    }

    # Plot attention heatmap
    # START TODO #############
    # Hint: the requested plot can be generated using seaborn's heatmap method
    seaborn.heatmap(attention.squeeze(), ax=ax, **plot_kwargs)
    # END TODO #############

    ax.set_aspect("equal")
    for tick in ax.get_yticklabels():
        tick.set_rotation(0)
    plt.show()
