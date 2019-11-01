from __future__ import print_function
from __future__ import division
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
import copy
import numpy as np

##############################################################################
############################ Helper functions ################################


def batch_helper(x):
    b_size = int(len(x))
    anchor = x[list(range(0, b_size, 3))]
    pos = x[list(range(1, b_size, 3))]
    neg = x[list(range(2, b_size, 3))]
    return anchor, pos, neg


def make_data_array(data):
    out = []
    for l in range(len(data)//7):
        hold = {'img':[],'year':[],'meta':[]}
        for i in range(7):
            img, year, meta = data[l*7+i]
            hold['img'].append(img)
            hold['year'].append(year)
            hold['meta'].append(meta)
        out.append(hold)
    return out
#############################################################################


class Model(nn.Module):
    def __init__(self, latent_space=128, input_shape=224, dropout = 0):
        super(Model, self).__init__()

        self.input_shape = input_shape
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.latent_space = latent_space

        self.model = models.resnet18(pretrained=True)
        self.best_acc = 0

        # For pretrained model turn off all gradients
        for param in self.model.parameters():
            param.requires_grad = False

        # Setup output FC layer for tuning
        num_ftrs = self.model.fc.in_features

        # Allow the option for dropout
        if dropout > 0:
            self.model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(num_ftrs, self.latent_space))
        else:
            self.model.fc = nn.Linear(num_ftrs, self.latent_space)

        self.model = self.model.to(self.device)

        # Setup Optimizer
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
        self.optimizer = optim.SGD(params_to_update, lr=0.0001, momentum=0.9)

    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def train_model(self, dataloaders_train, dataloaders_val, margin=0.0, num_epochs=25):
        since = time.time()

        val_acc_history = []
        training_loss_history = []

        epoch_score = []

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = self.best_acc

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                    dataloaders = dataloaders_train
                else:
                    self.model.eval()  # Set model to evaluate mode
                    dataloaders = dataloaders_val

                running_loss = 0.0
                running_corrects = 0.0
                batches = 0.0
                score_hold = {'anchor': [], 'pos': [], 'neg': []}

                # Iterate over data.
                for x, y, meta in dataloaders:
                    x = batch_helper(x)
                    anchor = x[0].to(self.device)
                    pos = x[1].to(self.device)
                    neg = x[2].to(self.device)

                    # forward
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        anchor_output = self.model(anchor)
                        pos_output = self.model(pos)
                        neg_output = self.model(neg)

                        # trying to fix NAN, bootleg fix that does the job
                        anchor_output[anchor_output != anchor_output] = 0
                        pos_output[pos_output != pos_output] = 0
                        neg_output[neg_output != neg_output] = 0

                        pos_term = torch.sqrt((anchor_output - pos_output).pow(2)).sum(1)
                        neg_term = torch.sqrt((anchor_output - neg_output).pow(2)).sum(1)

                        loss = torch.mean(F.relu(pos_term - neg_term + margin))

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item()
                    acc = torch.mean(torch.stack([pos_term < neg_term]).double())
                    running_corrects += acc
                    batches += 1.0

                epoch_loss = running_loss / batches
                epoch_acc = running_corrects.double() / batches

                epoch_score.append(score_hold)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)
                else:
                    training_loss_history.append(epoch_loss)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        self.best_acc = best_acc
        return val_acc_history, training_loss_history, epoch_score

    def solo_run(self, dataloaders):
        score_hold = {'anchor': [], 'pos': [], 'neg': []}

        self.model.eval()  # Set model to evaluate mode

        for x, y, meta in dataloaders:
            anchor = x[0].view(-1, 3, self.input_shape, self.input_shape).to(self.device)
            pos = x[1].view(-1, 3, self.input_shape, self.input_shape).to(self.device)
            neg = x[2].view(-1, 3, self.input_shape, self.input_shape).to(self.device)

            with torch.set_grad_enabled(False):
                anchor_output = self.model(anchor)
                pos_output = self.model(pos)
                neg_output = self.model(neg)

                score_hold['anchor'].append(anchor_output.detach().cpu().numpy())
                score_hold['pos'].append(pos_output.detach().cpu().numpy())
                score_hold['neg'].append(neg_output.detach().cpu().numpy())

        return score_hold

# Uses loader data as input and produces a matrix of latents over all data
    def get_latents(self, data):
        data_array = make_data_array(data)
        self.model.eval()
        out = []
        meta = []
        for location in data_array:
            hold = []
            for year in location['img']:
                data_input = year.view(-1, 3, self.input_shape, self.input_shape).to(self.device)
                output = self.model(data_input)
                output[output != output] = 0
                hold.append(output.detach().cpu().numpy().reshape(-1))
            out.append(np.array(hold))
            meta.append(location['meta'][0])
        return np.array(out), np.array(meta)
