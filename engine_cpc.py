import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging


def train_model_cpc(train_loader, model, optimizer, criterion,
                    criterion_cpc, device, lr_scheduler, train_writer, epoch, cpc_warm_up=0):
    """
    Note: train_loss and train_acc is accurate only if set drop_last=False in loader

    :param train_loader: y: one_hot float tensor
    :param model:
    :param optimizer:
    :param criterion: set reduction='sum'
    :param device:
    :return:
    """
    model.train()
    if cpc_warm_up > 0:
        model.adjust_cpc_weight(current_epoch=epoch, warm_up=cpc_warm_up)
    train_loss = 0
    cls_loss = 0
    cpc_loss = 0
    correct = 0
    correct_cpc = 0
    samples = 0
    cpc_samples = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        if batch_idx == 0 and model.cpc_weight > 0:
            for metric in model.cpc_metrics:
                metric.restart()
        # target: (B,)
        data, target = data.to(device), target.to(device)
        # (B, C, T)
        logits, cpc_logits, cpc_label = model(data)

        # list into tensor
        if model.cpc_weight > 0:
            cpc_logits = torch.cat(cpc_logits, dim=0)
            cpc_label = torch.cat(cpc_label, dim=0)

        # (B, T)
        target_repeat = target.unsqueeze(1).repeat(1, logits.size(-1))

        _cls_loss = criterion(logits, target_repeat)
        if model.cpc_weight > 0:
            _cpc_loss = criterion_cpc(cpc_logits, cpc_label)
            loss = _cls_loss + _cpc_loss * model.cpc_weight
        else:
            loss = _cls_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        train_loss += loss.item()
        if model.cpc_weight > 0:
            cpc_loss += _cpc_loss.item()
        cls_loss += _cls_loss.item()
        with torch.no_grad():
            # (B, C)
            probs = F.softmax(logits, dim=1).mean(-1)
            pred = probs.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            samples += target_repeat.numel()

            if model.cpc_weight > 0:

                pred_cpc = cpc_logits.argmax(-1)
                correct_cpc += pred_cpc.eq(cpc_label).sum().item()
                cpc_samples += cpc_label.size(0)

    train_loss /= (samples + cpc_samples)
    train_acc = correct / len(train_loader.dataset)
    cls_loss /= samples
    train_writer.add_scalar("cls_loss", cls_loss, epoch)

    cpc_result_log = {}

    if model.cpc_weight > 0:
        cpc_loss /= cpc_samples
        cpc_acc = correct_cpc / cpc_samples

        train_writer.add_scalar("cpc_loss", cpc_loss, epoch)
        train_writer.add_scalar("cpc_acc", cpc_acc, epoch)
        train_writer.add_scalar("cpc_samples", cpc_samples, epoch)

        for metric in model.cpc_metrics:
            _cpc_log = metric.write(writer=train_writer, epoch=epoch, is_train=True)
            cpc_result_log = {**cpc_result_log, **_cpc_log}

    return {'loss': train_loss, 'acc': train_acc}, cpc_result_log


def eval_model_cpc(test_loader, model, criterion, criterion_cpc, device, valid_writer, epoch, cpc_warm_up=0):
    if cpc_warm_up > 0:
        model.adjust_cpc_weight(current_epoch=epoch, warm_up=cpc_warm_up)
    model.eval()
    test_loss = 0
    cls_loss = 0
    cpc_loss = 0
    correct = 0
    correct_cpc = 0
    samples = 0
    cpc_samples = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
            if batch_idx == 0 and model.cpc_weight > 0:
                for metric in model.cpc_metrics:
                    metric.restart()
            # target: (B,)
            data, target = data.to(device), target.to(device)
            # (B, C, T)
            logits, cpc_logits, cpc_label = model(data)

            # list into tensor
            if model.cpc_weight > 0:
                cpc_logits = torch.cat(cpc_logits, dim=0)
                cpc_label = torch.cat(cpc_label, dim=0)

            # (B, T)
            target_repeat = target.unsqueeze(1).repeat(1, logits.size(-1))

            _cls_loss = criterion(logits, target_repeat)
            if model.cpc_weight > 0:
                _cpc_loss = criterion_cpc(cpc_logits, cpc_label)
                loss = _cls_loss + _cpc_loss * model.cpc_weight
            else:
                loss = _cls_loss

            test_loss += loss.item()
            if model.cpc_weight > 0:
                cpc_loss += _cpc_loss.item()
            cls_loss += _cls_loss.item()

            # get the index of the max log-probability
            # (B, C)
            probs = F.softmax(logits, dim=1).mean(-1)
            pred = probs.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            samples += target_repeat.numel()

            if model.cpc_weight > 0:
                pred_cpc = cpc_logits.argmax(-1)
                correct_cpc += pred_cpc.eq(cpc_label).sum().item()
                cpc_samples += cpc_label.size(0)

    test_loss /= (samples + cpc_samples)
    test_acc = correct / len(test_loader.dataset)
    cls_loss /= samples
    valid_writer.add_scalar("cls_loss", cls_loss, epoch)

    cpc_result_log = {}

    if model.cpc_weight > 0:
        cpc_loss /= cpc_samples
        cpc_acc = correct_cpc / cpc_samples

        valid_writer.add_scalar("cpc_loss", cpc_loss, epoch)
        valid_writer.add_scalar("cpc_acc", cpc_acc, epoch)
        valid_writer.add_scalar("cpc_samples", cpc_samples, epoch)

        for metric in model.cpc_metrics:
            _cpc_log = metric.write(writer=valid_writer, epoch=epoch, is_train=False)
            cpc_result_log = {**cpc_result_log, **_cpc_log}

    return {'loss': test_loss, 'acc': test_acc}, cpc_result_log
