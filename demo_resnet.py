import torch
import torchvision

from tqdm import tqdm

from imagenet_bg import ImageNetBG

IMAGENET_BG_PATH = "./ImageNet-Background/"
BATCH_SIZE = 1024

model = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V1)
model.eval()
model.cuda()

preprocess = torchvision.models.ResNet152_Weights.IMAGENET1K_V1.transforms()
dataset = ImageNetBG(root_dir=IMAGENET_BG_PATH, labels_file="./imagenet_class_index.json", transform=preprocess)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

correct_all, total_all = 0, 0
correct_good, total_good = 0, 0
correct_good_abstract, total_good_abstract = 0, 0
correct_good_white, total_good_white = 0, 0

for i_batch, (inputs, targets, infos) in enumerate(tqdm(data_loader)):
    inputs = inputs.cuda()
    targets = targets.cuda()

    info_quantity, info_type, info_background, info_background_real_type = infos
    
    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

    total_all += targets.size(0)
    correct_all += (preds == targets).sum().item()

    is_good = [((q == "good") or (q == "v_good")) for q in info_quantity]
    is_good = torch.tensor(is_good, device=targets.device)
    total_good += targets[is_good].size(0)
    correct_good += (preds[is_good] == targets[is_good]).sum().item()

    is_good_abstract = [((q == "good") or (q == "v_good")) and (t == "abstract") for q, t in zip(info_quantity, info_type)]
    is_good_abstract = torch.tensor(is_good_abstract, device=targets.device)
    total_good_abstract += targets[is_good_abstract].size(0)
    correct_good_abstract += (preds[is_good_abstract] == targets[is_good_abstract]).sum().item()

    is_good_white = [((q == "good") or (q == "v_good")) and (b == "white") for q, b in zip(info_quantity, info_background)]
    is_good_white = torch.tensor(is_good_white, device=targets.device)
    total_good_white += targets[is_good_white].size(0)
    correct_good_white += (preds[is_good_white] == targets[is_good_white]).sum().item()

acc_all = correct_all / total_all
acc_good = correct_good / total_good if correct_good > 0 else 0
acc_good_abstract = correct_good_abstract / total_good_abstract if total_good_abstract > 0 else 0
acc_good_white = correct_good_white / total_good_white if total_good_white > 0 else 0

print(f"Accuracy (all): {acc_all:.2f} on {total_all} images")
print(f"Accuracy (good/v_good): {acc_good:.2f} on {correct_good} images")
print(f"Accuracy (good/v_good on abstract backgrounds): {acc_good_abstract:.2f} on {total_good_abstract} images")
print(f"Accuracy (good/v_good on only white bacground): {acc_good_white:.2f} on {total_good_white} images")
