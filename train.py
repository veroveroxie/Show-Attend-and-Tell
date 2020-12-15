import argparse, json
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.translate.bleu_score import corpus_bleu
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

from dataset import ImageCaptionDataset
from decoder import Decoder
from encoder import Encoder
from utils import AverageMeter, accuracy, calculate_caption_lengths
import os
from utils import Saver
import matplotlib.pyplot as plt

os.environ['TORCH_HOME']='/user_data/shaoanxi/models'

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class Trainer(nn.Module):
    def __init__(self, encoder, decoder, optimizer, scheduler):
        super(Trainer, self).__init__()
        self.model_dict = {'encoder':encoder, 'decoder':decoder, 'opt':optimizer, 'sche':scheduler}


def main(args):
    tf = '_tf' if args.tf else ''
    run_dir = os.path.join('results/', args.config+tf+'_%.4f' % args.lambda_ient+'_%.4f'%args.lambda_tent)
    word_dict = json.load(open(args.data + '/word_dict.json', 'r'))
    vocabulary_size = len(word_dict)
    encoder = Encoder(args.network, args.config)
    decoder = Decoder(vocabulary_size, encoder.dim, args.tf)
    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_size)
    encoder.cuda()
    decoder.cuda()
    cross_entropy_loss = nn.CrossEntropyLoss().cuda()
    saver = Saver(Trainer(encoder, decoder, optimizer, scheduler), run_dir)
    writer = SummaryWriter(saver.log_dir)
    train_loader = torch.utils.data.DataLoader(
        ImageCaptionDataset(data_transforms, args.data),
        batch_size=args.batch_size, shuffle=True, num_workers=1)

    val_loader = torch.utils.data.DataLoader(
        ImageCaptionDataset(data_transforms, args.data, split_type='val'),
        batch_size=args.batch_size, shuffle=True, num_workers=1)

    test_loader = torch.utils.data.DataLoader(
        ImageCaptionDataset(data_transforms, args.data, split_type='test'),
        batch_size=args.batch_size, shuffle=False, num_workers=1)

    print('Starting training with {}'.format(args))
    for epoch in range(saver.start_epoch, args.epochs + 1):
        train(epoch, encoder, decoder, optimizer, cross_entropy_loss,
              train_loader, word_dict, args.alpha_c, args.log_interval, writer, saver, val_loader, args)
        saver.save_model(epoch)
        validate(epoch, encoder, decoder, cross_entropy_loss, val_loader,
                 word_dict, args.alpha_c, args.log_interval, writer, saver)
        test(epoch, encoder, decoder, cross_entropy_loss, test_loader,
                 word_dict, args.alpha_c, args.log_interval, writer, saver)
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))
    writer.close()


def att_entropy(vector):
    # B, T, M (64, 50, 196)
    # we want to minimize this
    entropy = 0
    B,T,M = vector.size()
    for i in range(len(vector)):
        x = vector[i]
        for j in range(len(x)):
            probs = x[j] # 196
            entropy += torch.sum(-probs * torch.log(probs+1e-9))
    return entropy/(B*T)

def time_entropy(vector):
    # we want to maximize this
    B, T, M = vector.size()
    vector = torch.sum(vector, dim=1) # sum with t
    softmax_vector = torch.softmax(vector, dim=1)
    entropy = torch.sum(-softmax_vector*torch.log(softmax_vector+1e-9))
    return entropy/(B)

def train(epoch, encoder, decoder, optimizer, cross_entropy_loss, data_loader, word_dict, alpha_c, log_interval, writer, saver, val_loader, args):
    encoder.eval()
    decoder.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for batch_idx, (imgs, captions) in enumerate(data_loader):
        imgs, captions = Variable(imgs).cuda(), Variable(captions).cuda()
        img_features = encoder(imgs)
        optimizer.zero_grad()
        preds, alphas = decoder(img_features, captions)



        targets = captions[:, 1:]

        targets = pack_padded_sequence(targets, [len(tar) - 1 for tar in targets], batch_first=True)[0]
        preds = pack_padded_sequence(preds, [len(pred) - 1 for pred in preds], batch_first=True)[0]

        att_regularization = alpha_c * ((1 - alphas.sum(1))**2).mean()
        entropy_reg = att_entropy(alphas)
        time_reg = -time_entropy(alphas)  # maximize this to hope each image can be treated equally

        loss = cross_entropy_loss(preds, targets)
        loss += att_regularization
        loss += args.lambda_ient * entropy_reg
        loss += args.lambda_tent * time_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 10)
        optimizer.step()

        total_caption_length = calculate_caption_lengths(word_dict, captions)
        acc1 = accuracy(preds, targets, 1)
        acc5 = accuracy(preds, targets, 5)
        losses.update(loss.item(), total_caption_length)
        top1.update(acc1, total_caption_length)
        top5.update(acc5, total_caption_length)

        if batch_idx % log_interval == 0:
            #print('(Time: {}) Train Batch: [{0}/{1}]\t'
            #      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #      'Top 1 Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
            #      'Top 5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(saver.used_time,
            #          batch_idx, len(data_loader), loss=losses, top1=top1, top5=top5))
            print('(Epoch: %03d, Iter: %06d, Time: %s), Loss: %.2f, I_ent: %.2f, T_ent: %.2f, top1: %.2f, top5: %.2f'%(
                epoch, batch_idx, saver.used_time, losses.avg, entropy_reg, time_reg, top1.val, top5.val,
            ))
    writer.add_scalar('train_loss', losses.avg, epoch)
    writer.add_scalar('train_top1_acc', top1.avg, epoch)
    writer.add_scalar('train_top5_acc', top5.avg, epoch)


def validate(epoch, encoder, decoder, cross_entropy_loss, data_loader, word_dict, alpha_c, log_interval, writer, saver):
    encoder.eval()
    decoder.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # used for calculating bleu scores
    references = []
    hypotheses = []
    with torch.no_grad():
        for batch_idx, (imgs, captions, all_captions) in enumerate(data_loader):
            imgs, captions = Variable(imgs).cuda(), Variable(captions).cuda()
            img_features = encoder(imgs)
            preds, alphas = decoder(img_features, captions)
            targets = captions[:, 1:]

            targets = pack_padded_sequence(targets, [len(tar) - 1 for tar in targets], batch_first=True)[0]
            packed_preds = pack_padded_sequence(preds, [len(pred) - 1 for pred in preds], batch_first=True)[0]

            att_regularization = alpha_c * ((1 - alphas.sum(1))**2).mean()

            loss = cross_entropy_loss(packed_preds, targets)
            loss += att_regularization

            total_caption_length = calculate_caption_lengths(word_dict, captions)
            acc1 = accuracy(packed_preds, targets, 1)
            acc5 = accuracy(packed_preds, targets, 5)
            losses.update(loss.item(), total_caption_length)
            top1.update(acc1, total_caption_length)
            top5.update(acc5, total_caption_length)

            for cap_set in all_captions.tolist():
                caps = []
                for caption in cap_set:
                    cap = [word_idx for word_idx in caption
                                    if word_idx != word_dict['<start>'] and word_idx != word_dict['<pad>']]
                    caps.append(cap)
                references.append(caps)

            word_idxs = torch.max(preds, dim=2)[1]
            for idxs in word_idxs.tolist():
                hypotheses.append([idx for idx in idxs
                                       if idx != word_dict['<start>'] and idx != word_dict['<pad>']])

            if batch_idx % log_interval == 0:
                info = ('Validation Batch: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top 1 Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Top 5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(
                          batch_idx, len(data_loader), loss=losses, top1=top1, top5=top5))
                print(info)
        writer.add_scalar('val_loss', losses.avg, epoch)
        writer.add_scalar('val_top1_acc', top1.avg, epoch)
        writer.add_scalar('val_top5_acc', top5.avg, epoch)

        bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
        bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
        bleu_3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
        bleu_4 = corpus_bleu(references, hypotheses)

        writer.add_scalar('val_bleu1', bleu_1, epoch)
        writer.add_scalar('val_bleu2', bleu_2, epoch)
        writer.add_scalar('val_bleu3', bleu_3, epoch)
        writer.add_scalar('val_bleu4', bleu_4, epoch)
        saver.save_print_msg('Validation Epoch: {}\t'
              'BLEU-1 ({})\t'
              'BLEU-2 ({})\t'
              'BLEU-3 ({})\t'
              'BLEU-4 ({})\t'.format(epoch, bleu_1, bleu_2, bleu_3, bleu_4))

def test(epoch, encoder, decoder, cross_entropy_loss, data_loader, word_dict, alpha_c, log_interval, writer, saver):
    encoder.eval()
    decoder.eval()

    # used for calculating bleu scores
    references = []
    hypotheses = []
    with torch.no_grad():
        for batch_idx, (imgs, captions, all_captions) in enumerate(data_loader):
            imgs, captions = Variable(imgs).cuda(), Variable(captions).cuda()
            img_features = encoder(imgs)
            preds, alphas = decoder(img_features, captions)
            targets = captions[:, 1:]

            targets = pack_padded_sequence(targets, [len(tar) - 1 for tar in targets], batch_first=True)[0]
            packed_preds = pack_padded_sequence(preds, [len(pred) - 1 for pred in preds], batch_first=True)[0]

            total_caption_length = calculate_caption_lengths(word_dict, captions)

            for cap_set in all_captions.tolist():
                caps = []
                for caption in cap_set:
                    cap = [word_idx for word_idx in caption
                           if word_idx != word_dict['<start>'] and word_idx != word_dict['<pad>']]
                    caps.append(cap)
                references.append(caps)

            word_idxs = torch.max(preds, dim=2)[1]
            for idxs in word_idxs.tolist():
                hypotheses.append([idx for idx in idxs
                                   if idx != word_dict['<start>'] and idx != word_dict['<pad>']])

            if batch_idx % log_interval == 0:
                print('[%d/%d]'%(batch_idx, len(data_loader)))

        bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
        bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
        bleu_3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
        bleu_4 = corpus_bleu(references, hypotheses)

        writer.add_scalar('test_bleu1', bleu_1, epoch)
        writer.add_scalar('test_bleu2', bleu_2, epoch)
        writer.add_scalar('test_bleu3', bleu_3, epoch)
        writer.add_scalar('test_bleu4', bleu_4, epoch)
        saver.save_print_msg('Test Epoch: {}\t'
                             'BLEU-1 ({})\t'
                             'BLEU-2 ({})\t'
                             'BLEU-3 ({})\t'
                             'BLEU-4 ({})\t'.format(epoch, bleu_1, bleu_2, bleu_3, bleu_4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show, Attend and Tell')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='batch size for training (default: 64)')
    parser.add_argument('--config', type=str, default='New_ITent',
                        help='configuration of the model')
    parser.add_argument('--epochs', type=int, default=10, metavar='E',
                        help='number of epochs to train for (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate of the decoder (default: 1e-4)')
    parser.add_argument('--step-size', type=int, default=5,
                        help='step size for learning rate annealing (default: 5)')
    parser.add_argument('--alpha-c', type=float, default=1, metavar='A',
                        help='regularization constant (default: 1)')
    parser.add_argument('--lambda_ient', type=float, default=0.01,
                        help='regularization constant (default: 1)')
    parser.add_argument('--lambda_tent', type=float, default=0.00,
                        help='regularization constant (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='L',
                        help='number of batches to wait before logging training stats (default: 100)')
    parser.add_argument('--data', type=str, default='../datasets/MSCOCO',
                        help='path to data images (default: data/coco)')
    parser.add_argument('--network', choices=['vgg19', 'resnet152', 'densenet161'], default='vgg19',
                        help='Network to use in the encoder (default: vgg19)')
    parser.add_argument('--model', type=str, help='path to model')
    parser.add_argument('--tf', action='store_true', default=True,
                        help='Use teacher forcing when training LSTM (default: False)')

    main(parser.parse_args())
