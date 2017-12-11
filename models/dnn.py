from collections import OrderedDict
import math
import os
import shutil
import sys
sys.path.append("./")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from models.ocr_model import OCRModel

from skimage import io
import matplotlib

class OCRCNN(OCRModel):
    def __init__(self, image_dir="images/", debug=False, kernel_sizes=[], unit_counts=[], strides=[], maxpool_sizes=[]):
        super(OCRCNN, self).__init__(image_dir=image_dir, debug=debug)
        self.on_gpu = torch.cuda.is_available()

        assert(len(kernel_sizes) == len(unit_counts) == len(strides) == len(maxpool_sizes))
        
        # Fix random seed for debugging
        torch.manual_seed(1)

        # Build CNN
        self.layers = OrderedDict()
        current_height = self.char_image_size[0]
        current_width = self.char_image_size[1]
        current_channels = current_width*current_height
        
        for idx in range(len(kernel_sizes)):
            out_channels = unit_counts[idx]
            self.layers["lin_%d" % idx] = nn.Linear(current_channels, out_channels)
            current_channels = out_channels
            # Formula from from http://pytorch.org/docs/master/nn.html#conv2d
            # Assumes padding, dilation = 0

            '''
            maxpool_size = maxpool_sizes[idx]
            self.layers["maxpool_%d" % idx] = nn.MaxPool2d(maxpool_size)
            # Formula from from http://pytorch.org/docs/master/nn.html#maxpool2d
            # Assumes padding, dilation = 0
            current_height = int(math.floor((current_height - maxpool_size) / float(maxpool_size))) + 1
            current_width = int(math.floor((current_width - maxpool_size) / float(maxpool_size))) + 1
            '''

            self.layers["bn_%d" % idx] = nn.BatchNorm2d(current_channels)
            self.layers["relu_%d" % idx] = nn.ReLU()
        self.lin_input_size = current_channels * current_height * current_width
        # self.layers["lin_final"] = nn.Linear(self.lin_input_size, self.kjv.unique_chars())
        self.layers["lin_final"] = nn.Linear(current_channels, self.kjv.unique_chars())
        self.classifier = nn.Sequential(self.layers)
        if self.on_gpu:
            self.classifier.cuda()
        print(self.classifier)

    def forward_classifier(self, feats):
        output = feats
        output = output.view((-1, (self.char_width+3)* self.char_height))
        for i, (layer_name, layer) in enumerate(self.layers.items()):
            '''
            if layer_name == "lin_final":
                output = output.view((-1, self.lin_input_size))
            '''
            output = layer(output)
        return output

    def train(self):
        training_feats, training_labels = self.training_data() 
        training_feats = torch.FloatTensor(training_feats)
        training_labels = torch.LongTensor(list(map(int, training_labels)))
        
        val_feats, val_labels = self.val_data() 
        val_feats = torch.FloatTensor(val_feats)
        val_labels = torch.LongTensor(list(map(int, val_labels)))

        print("Training convnet...")
        batch_size = 256
        max_epochs = 5
        learning_rate = 0.01
        optimizer = optim.SGD(self.classifier.parameters(), lr=learning_rate)

        # Regularize via patience-based early stopping
        best_val_loss = float('inf')
        save_best_only = True   # Set to false to always save model, regardless of improvement
        max_patience = 3
        min_improvement = 0.000100
        epochs_since_improvement = 0

        for epoch in range(max_epochs):
            # Training run
            self.classifier.train()
            train_loss = 0.0

            num_batches = int(math.ceil(training_feats.shape[0] / float(batch_size)))
            print_interval = max(1, int(num_batches / 100.0))
            for batch_idx in range(num_batches):
                feats = training_feats[batch_idx * batch_size:(batch_idx + 1) * batch_size, :]
                feats = feats.view(batch_size, 1, self.char_width+3, self.char_height)
                labels = training_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                feats = Variable(feats)
                labels = Variable(labels)
                if self.on_gpu:
                    feats = feats.cuda()
                    labels = labels.cuda()
                """
                for i in range(5):
                    flattened_img = feats.cpu().data.numpy()[i,:]
                    img = flattened_img.reshape((self.char_image_size[0], self.char_image_size[1]))
                    io.imshow(img)
                    label = labels.cpu().data.numpy()[i]
                    print(label)
                    matplotlib.pyplot.show()
                """
                optimizer.zero_grad()
                predictions = self.forward_classifier(feats)
                loss = nn.CrossEntropyLoss()(predictions, labels)
                loss.backward()
                train_loss += loss.data[0]
                optimizer.step()

                if batch_idx % print_interval == 0:
                    # Print update in place
                    sys.stdout.write("\rTrain %d%% complete" % int(batch_idx / float(training_feats.shape[0] / float(batch_size) / 100.0)))
                    sys.stdout.flush()

            # Write new line to clear line
            sys.stdout.write("\rTrain 100% complete!\n")
            sys.stdout.flush()

            train_loss /= float(training_feats.shape[0] / float(batch_size))
            print("===> Epoch %d: training loss %.6f" % (epoch + 1, train_loss))

            # Validation run
            self.classifier.eval()
            val_loss = 0.0

            num_batches = int(math.ceil(val_feats.shape[0] / float(batch_size)))
            print_interval = max(1, int(num_batches / 100.0))
            for batch_idx in range(int(math.ceil(val_feats.shape[0] / float(batch_size)))):
                feats = Variable(val_feats[batch_idx * batch_size:(batch_idx + 1) * batch_size, :],
                                 volatile=True)     # Set to volatile so history isn't saved
                feats = feats.view(batch_size, 1, self.char_width+3, self.char_height)
                labels = Variable(val_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size],
                                  volatile=True)    # Set to volatile so history isn't saved
                if self.on_gpu:
                    feats = feats.cuda()
                    labels = labels.cuda()

                predictions = self.forward_classifier(feats)
                loss = nn.CrossEntropyLoss()(predictions, labels)
                val_loss += loss.data[0]

                if batch_idx % print_interval == 0:
                    # Print update in place
                    sys.stdout.write("\rVal %d%% complete" % int(batch_idx / float(val_feats.shape[0] / float(batch_size) / 100.0)))
                    sys.stdout.flush()

            # Write new line to clear line
            sys.stdout.write("\rVal 100% complete!\n")
            sys.stdout.flush()

            val_loss /= float(val_feats.shape[0] / float(batch_size))
            print("===> Epoch %d: validation loss %.6f" % (epoch + 1, val_loss))

            # Check early stopping criterion
            is_best = (val_loss + min_improvement <= best_val_loss)
            if is_best:
                best_val_loss = val_loss
                epochs_since_improvement = 0
                print("New best validation set loss: %.6f" % best_val_loss)
            else:
                epochs_since_improvement += 1
                print("No minimum improvement in %d epochs (best val set loss: %.6f)" % (epochs_since_improvement, best_val_loss))
                if epochs_since_improvement >= max_patience:
                    print("STOPPING EARLY")
                    break

            if not (save_best_only and not is_best):
                # Save a checkpoint for our model!
                state_obj = {
                        "epoch": epoch,
                        "state_dict": self.classifier.state_dict(),
                        "best_val_loss": best_val_loss,
                        "val_loss": val_loss,
                        "optimizer": optimizer.state_dict()
                }
                self.save_checkpoint(state_obj, is_best, "saved_models")
                print("Saved checkpoint for model")
            else:
                print("Not saving checkpoint; no improvement made")
        
        print("Trained convnet.")

    def eval(self):
        eval_feats, eval_labels = self.all_data() 
        eval_feats = torch.FloatTensor(eval_feats)
        eval_labels = torch.LongTensor(list(map(int, eval_labels)))

        print("Evaluating data points with convnet...")

        # Use one-hot to save some memory
        predictions = np.empty(len(eval_labels), dtype=int)

        batch_size = 128
        self.classifier.eval()

        num_batches = int(math.ceil(eval_feats.shape[0] / float(batch_size)))
        print_interval = max(1, int(num_batches / 100.0))
        for batch_idx in range(num_batches):
            feats = eval_feats[batch_idx * batch_size:(batch_idx + 1) * batch_size, :]
            feats = feats.view(batch_size, 1,self.char_width+3, self.char_height)
            feats = Variable(feats, volatile=True)      # Set to volatile so history isn't saved
            if self.on_gpu:
                    feats = feats.cuda()

            batch_probs = self.forward_classifier(feats)
            #print(batch_probs.cpu().data.numpy())
            batch_preds = np.argmax(batch_probs.cpu().data.numpy(), axis=1).reshape((-1))
            predictions[batch_idx * batch_size:(batch_idx + 1) * batch_size] = batch_preds

            if batch_idx % print_interval == 0:
                # Print update in place
                sys.stdout.write("\rTrain %d%% complete" % int(batch_idx / float(eval_feats.shape[0] / float(batch_size) / 100.0)))
                sys.stdout.flush()

        # Write new line to clear line
        sys.stdout.write("\rEval 100% complete!\n")
        sys.stdout.flush()

        print("Evaluated data points with classifier.")

        return predictions

    def save_checkpoint(self, state_obj, is_best, model_dir):
        filepath = os.path.join(model_dir, "ckpt_cnn_%d.pth.tar" % state_obj["epoch"])
        torch.save(state_obj, filepath)
        if is_best:
            shutil.copyfile(filepath, os.path.join(model_dir, "best_cnn.pth.tar"))
