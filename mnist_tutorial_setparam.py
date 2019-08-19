import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

import models
from cleverhans.attacks import FastGradientMethod, ProjectedGradientDescent, MomentumIterativeMethod
from cleverhans.model import CallableModelWrapper
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf

# from cleverhans.dataset import CIFAR10

ATTACK_TYPES=['fgsm']
ATTACK_TYPE = ATTACK_TYPES[0]
FLAGS = flags.FLAGS

model_path = '/workspace/FYP/adversarial-defence/checkpoint/011_Res18_rand10/011_Res18_rand10_epoch199.ckpt'
BATCH_SIZE = 128
LEARNING_RATE = .001

def load_pytorch_model(model_path):
  # Load PyTorch model
  torch_model = models.ResNet_maker(n_layers=18, preact=False, BN=True)
  torch_model.load_state_dict(torch.load(model_path),strict=True)
  torch_model.eval()
  if torch.cuda.is_available():
    torch_model = torch_model.cuda()
  return torch_model 

def load_data():
  # Load CIFAR10 dataset 
  cifar10_path = '/workspace/FYP/adversarial-defence/data/cifar10/downloads'
  transform_test = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])
  testset = datasets.CIFAR10(root=cifar10_path, train=False, download=False,
                             transform=transform_test)
  test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
  return test_loader

def evaluate_pytorch(test_loader, torch_model):
  # Evaluate on clean data 
  total = 0
  correct = 0
  for xs, ys in test_loader:
    xs, ys = Variable(xs), Variable(ys)
    if torch.cuda.is_available():
      xs, ys = xs.cuda(), ys.cuda()

    preds = torch_model(xs)
    preds_np = preds.data.cpu().detach().numpy()
    correct += (np.argmax(preds_np, axis=1) == ys.cpu().detach().numpy()).sum()
    total += len(xs)

  acc = float(correct) / total
  print('Accuracy: %.2f%%' % (acc * 100))
  
def pytorch_to_tf_model(torch_model):
  # Convert pytorch model to a tf_model and wrap it in cleverhans
  tf_model_fn = convert_pytorch_model_to_tf(torch_model)
  cleverhans_model = CallableModelWrapper(tf_model_fn, output_layer='logits')
  return tf_model_fn,cleverhans_model


def generate_adv_example(sess, cleverhans_model, attack_type, attack_param, x_op):
  ''' Return adversarial example '''
  if attack_type == 'fgsm':
    attack_op = FastGradientMethod(cleverhans_model, sess=sess)
  elif attack_type == 'pgd':
    attack_op = ProjectedGradientDescent(cleverhans_model, sess=sess)
  adv_x_op = attack_op.generate(x_op, **attack_param)
  return adv_x_op
def evaluate_tf(sess,test_loader, tf_model_fn, attack_params, adv_x_op, x_op):
  adv_preds_op = tf_model_fn(adv_x_op) 
  # Run an evaluation of our model against fgsm
  total = 0
  correct = 0
  for xs, ys in test_loader:
    adv_preds = sess.run(adv_preds_op, feed_dict={x_op: xs})
    correct += (np.argmax(adv_preds, axis=1) == ys.cpu().detach().numpy()).sum()
    total += len(xs)
  acc = float(correct) / total
  print('{:.3f} Adv accuracy: {:.3f}'.format(attack_params['eps'], acc * 100))


def attack(attack_type, sess, test_loader, tf_model_fn, cleverhans_model, x_op):
#  PGD  
#  pgd_param_list = [{'eps': 0.01,'eps_iter':0.001}, {'eps': 0.03,'eps_iter':0.003}, {'eps': 0.05,'eps_iter':0.05}, {'eps': 0.07,'eps_iter':0.007}, {'eps': 0.09,'eps_iter':0.009}, {'eps': 0.2,'eps_iter':0.02}, {'eps': 0.3,'eps_iter':0.03}]
# FGSM 
  fgsm_param_list = [{'eps': 0.01,}, {'eps': 0.03,}, {'eps': 0.05,}, {'eps': 0.07,}, {'eps': 0.09,}, {'eps': 0.2,}, {'eps': 0.3,}]
  for attack_params in fgsm_param_list:
    adv_x_op = generate_adv_example(sess,cleverhans_model, 'fgsm',attack_params, x_op)
    evaluate_tf(sess, test_loader, tf_model_fn, attack_params, adv_x_op, x_op)
## MomentumIterativeMethod

def main(_=None):
  from cleverhans_tutorials import check_installation
  check_installation(__file__)
  test_loader = load_data()
  torch_model = load_pytorch_model(model_path)
  evaluate_pytorch(test_loader, torch_model)
  
  # We use tf for evaluation on adversarial data
  sess = tf.Session()
  x_op = tf.placeholder(tf.float32, shape=(None, 3, 32, 32,))
  tf_model_fn, cleverhans_model = pytorch_to_tf_model(torch_model)
  attack(ATTACK_TYPE, sess, test_loader, tf_model_fn, cleverhans_model, x_op)

if __name__ == '__main__':
  flags.DEFINE_integer('batch_size', BATCH_SIZE,
                       'Size of training batches')
  flags.DEFINE_float('learning_rate', LEARNING_RATE,
                     'Learning rate for training')

  tf.app.run()
