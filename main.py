import os
import time
from pathlib import Path
import numpy as np
import torch.optim as optim
import torch.utils.data as data
from PIL import Image
from torch.optim import lr_scheduler
from torchvision import transforms as trans

from evaluation import evaluate, hflip_batch, gen_plot
from model.model import *
from parser import args
from prepare_data import get_train_dataset, get_val_data, get_dataset
from pruning.FilterPrunner import FilterPrunner
from pruning.prune_MFN import prune_MFN


def load_data():
    data_path = Path(args.data_path)
    # dataset_train, _ = get_train_dataset(data_path / 'imgs')
    dataset_train = get_dataset(args.tfrecord_path, args.index_path)
    dataloader = data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=2)

    print('Training Data Loaded!')

    return dataloader


def train(model, dataloader, epochs=args.epoch_size, is_prune=False):
    margin = Arcface(embedding_size=args.embedding_size, classnum=args.class_size, s=32., m=0.5).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer_ft = optim.SGD([
        {'params': model.parameters(), 'weight_decay': 5e-4},
        {'params': margin.parameters(), 'weight_decay': 5e-4}], lr=0.01, momentum=0.9, nesterov=True)

    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[1], gamma=0.3)

    total_iters = 0
    for epoch in range(epochs):
        # train model
        exp_lr_scheduler.step()
        model.train()
        for det in dataloader:
            img, label = det["image_raw"].to(device), det["label"].numpy()
            label = np.reshape(label, [-1]).astype(np.int64)
            label = torch.from_numpy(label).to(device)
            # img, label = det[0].to(device), det[1].to(device)
            optimizer_ft.zero_grad()

            with torch.set_grad_enabled(True):
                raw_logits = model(img)
                output = margin(raw_logits, label)
                loss = criterion(output, label)
                loss.backward()
                optimizer_ft.step()

                total_iters += 1
                # print train information
                if total_iters % args.summary_interval == 0 and total_iters != 0:
                    # current training accuracy
                    _, preds = torch.max(output.data, 1)
                    total = label.size(0)
                    correct = (np.array(preds.cpu()) == np.array(label.data.cpu())).sum()

                    print("Epoch {}/{}, Iters: {:0>6d}, loss: {:.4f}, train_accuracy: {:.4f}"
                          .format(epoch+1, epochs, total_iters, loss.item(), correct / total))

                # Print validation
                if total_iters % args.evaluate_interval == 0 and total_iters != 0:
                    validation(model)

                # Save checkpoints
                if total_iters % args.save_interval == 0 and total_iters != 0:
                    if is_prune:
                        save_path = Path(args.pruned_ckpt_path)
                    else:
                        save_path = Path(args.ckpt_path)
                    torch.save(model.state_dict(),
                               save_path / ('model_epoch:{}_step:{}.pth'.format(epoch, total_iters)))
                    torch.save(margin.state_dict(),
                               save_path / ('head_epoch:{}_step:{}.pth'.format(epoch, total_iters)))
                    torch.save(criterion.state_dict(),
                               save_path / ('optimizer_epoch:{}_step:{}.pth'.format(epoch, total_iters)))


def prune(model, dataloader):
    # Load Model
    save_dir = args.output_path
    model.load_state_dict(torch.load(args.saved_model_path, map_location=lambda storage, loc: storage))

    print("Check the initial model accuracy...")
    since = time.time()
    validation(model)
    print("initial test :: time cost is {:.2f} s".format(time.time() - since))

    # Ordered module list for easier indexing and pruning
    modules = rearrange(model)

    # Make sure all the layers are trainable
    for param in model.parameters():
        param.requires_grad = True

    # Get number of iterations to prune
    number_of_filters = total_num_filters(modules)
    print("total model conv2D filters are: ", number_of_filters)
    num_filters_to_prune_per_iteration = args.filter_size
    iterations = math.ceil(
        (float(number_of_filters) * args.filter_percentage) / (num_filters_to_prune_per_iteration + 1e-6))
    print("Number of iterations to prune {} % filters:".format(args.filter_percentage * 100), iterations)

    # Start pruning
    for it in range(iterations):
        print("iter{}. Ranking filters ..".format(it))
        prunner = FilterPrunner(modules, use_cuda=True)
        filters_to_prune = getFiltersToPrune(model, prunner)

        savePrunedFilters(args.pruned_filters_path, filters_to_prune)
        layers_prunned = [(k, len(filters_to_prune[k])) for k in
                          sorted(filters_to_prune.keys())]  # k: layer index, number of filters
        print("iter{}. Layers that will be prunned".format(it), layers_prunned)

        print("iter{}. Prunning filters.. ".format(it))
        for layer_index, filter_index in filters_to_prune.items():
            model, modules = prune_MFN(model, layer_index, *filter_index, use_cuda=True)

        model = model.to(device)

        print(
            "iter{}. {:.2f}% Filters remaining".format(it, 100 * float(total_num_filters(modules)) / number_of_filters))

        print("iter{}. without retrain...".format(it))
        validation(model)

        print("iter{}. Fine tuning to recover from prunning iteration.. ".format(it))
        torch.cuda.empty_cache()
        train(model, dataloader, epochs=2, is_prune=True)

        print("iter{}. after retrain...".format(it))
        since = time.time()
        validation(model)
        print("iter{}. test time cost is {:.2f} s".format(it, time.time() - since))

        torch.save(model.state_dict(), os.path.join(save_dir, 'MFN_weights_pruned_{}'.format(it)))
        torch.save(model, os.path.join(save_dir, 'MFN_prunned_{}'.format(it)))

    print("Finished prunning")


def getFiltersToPrune(model, prunner):
    model.train()
    margin = Arcface(embedding_size=args.embedding_size, classnum=args.class_size, s=32., m=0.5).to(device)
    margin.load_state_dict(torch.load(args.saved_margin_path, map_location=lambda storage, loc: storage))

    criterion = torch.nn.CrossEntropyLoss().to(device)
    criterion.load_state_dict(torch.load(args.saved_optimizer_path, map_location=lambda storage, loc: storage))

    prunner.reset()

    for i_batch, det in enumerate(dataloader):
        img, label = det["image_raw"].to(device), det["label"].numpy()
        label = np.reshape(label, [-1]).astype(np.int64)
        label = torch.from_numpy(label).to(device)

        # zero the parameter gradients
        model.zero_grad()

        with torch.set_grad_enabled(True):
            raw_logits = prunner.forward(img)
            output = margin(raw_logits, label)
            loss = criterion(output, label)
            loss.backward()

        if i_batch == 10000:  # only use 1/10 train data
            break

    prunner.normalize_ranks_per_layer()
    filters_to_prune = prunner.get_prunning_plan(args.filter_size)

    return filters_to_prune


def loadModel(model):
    filters_to_prune = loadPrunedFilters(args.pruned_filters_path)

    for layer_index, filter_index in filters_to_prune.items():
        model, modules = prune_MFN(model, layer_index, *filter_index, use_cuda=True)

    model.load_state_dict(torch.load(args.saved_pruned_model_path, map_location=lambda storage, loc: storage))


def loadPrunedFilters(file_path):
    f = open(file_path, "r")
    filters_to_prune = {}

    while True:
        line = f.readline()
        if line != "":
            line_split = line.split(":")
            layer = int(line_split[0])
            if layer not in filters_to_prune:
                filters_to_prune[layer] = []

            filters = line_split[1].split(",")[:-1]
            for filter in filters:
                filters_to_prune[layer].append(int(filter))

        if not line:
            break

    f.close()
    return filters_to_prune


def savePrunedFilters(file_path, filters_to_prune):
    f = open(file_path, "a")
    for k in filters_to_prune.keys():
        filters = filters_to_prune[k]
        f.write(str(k) + ":")
        for filter in filters:
            f.write(str(filter) + ",")
        f.write("\n")

    f.close()


def rearrange(model):
    index = 0
    modules = {}
    for names, module in list(model._modules.items()):
        if isinstance(module, Depth_Wise):
            for _, module_sub in list(module._modules.items()):
                modules[index] = module_sub
                index += 1

        elif isinstance(module, Residual):
            for i in range(len(module.model)):
                for _, model_sub_sub in list(module.model[i]._modules.items()):
                    modules[index] = model_sub_sub
                    index += 1
        else:
            modules[index] = module
            index += 1

    return modules


def total_num_filters(modules):
    filters = 0
    for name, module in modules.items():
        if isinstance(module, Conv_block) or isinstance(module, Linear_block):
            filters = filters + module.conv.out_channels

    return filters


def validation(model):
    agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame = get_val_data(Path(args.data_path))

    accuracy, best_threshold, roc_curve_tensor = _evaluate(model, lfw, lfw_issame, 10, True)
    print('lfw accuracy: {}, threshold: {}'.format(accuracy, best_threshold))

    # accuracy, best_threshold, roc_curve_tensor = _evaluate(model, cfp_fp, cfp_fp_issame, 10, True)
    # print('cfp_fp accuracy: {}, threshold: {}'.format(accuracy, best_threshold))
    #
    # accuracy, best_threshold, roc_curve_tensor = _evaluate(model, agedb_30, agedb_30_issame, 10, True)
    # print('agedb_30 accuracy: {}, threshold: {}'.format(accuracy, best_threshold))


def _evaluate(model, carray, issame, nrof_folds=10, tta=True):
    model.eval()
    idx = 0
    embeddings = np.zeros([len(carray), args.embedding_size])

    with torch.no_grad():
        while idx + args.batch_size <= len(carray):
            batch = torch.tensor(carray[idx:idx + args.batch_size])
            if tta:
                flipped = hflip_batch(batch)
                emb_batch = model(batch.to(device)) + model(flipped.to(device))
                embeddings[idx:idx + args.batch_size] = l2_norm(emb_batch).cpu()
            else:
                embeddings[idx:idx + args.batch_size] = model(batch.to(device)).cpu()
            idx += args.batch_size

        if idx < len(carray):
            batch = torch.tensor(carray[idx:])
            if tta:
                flipped = hflip_batch(batch)
                emb_batch = model(batch.to(device)) + model(flipped.to(device))
                embeddings[idx:] = l2_norm(emb_batch).cpu()
            else:
                embeddings[idx:] = model(batch.to(device)).cpu()

    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = trans.ToTensor()(roc_curve)
    return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataloader = load_data()
    model = MobileFaceNet(args.embedding_size).to(device)

    if args.options == "train":
        train(model, dataloader)
    elif args.options == "prune":
        prune(model, dataloader)
    elif args.options == "load":
        loadModel(model)
