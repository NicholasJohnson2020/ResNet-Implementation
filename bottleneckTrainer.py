import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ResNet import *
import os, sys, time
import argparse
from collections import OrderedDict
import numpy as np

# To run:
# python bottleneckTrainer.py --dataset_type cifar10 --network_type resnet --num_layers 8 8 8 --resume False --update_rule momentum --momentum_params 0.1 0.9 --l2_reg 1e-4 --num_epochs 100 --train_batchsize 64 --epoch_decay 50 75 --exp_dir /data/milatmp1/jojason/resnet_test/

# Argparser
parser = argparse.ArgumentParser(description='Generic trainer argparser.')
# Dataset specific arguments:
parser.add_argument('--dataset_type', type=str,
					help='The type of dataset. Choices are: ["cifar10",  "cifar100"]')
# Network specific arguments:
parser.add_argument('--network_type', type=str,
					help='The type of network. Choices are: ["resnet", "preact_resnet"]')
# Generic network arguments:
parser.add_argument('--random_seed', default=0, type=int,
					help='Sets the random seed, necessary for reproducibility. Default is set to 0.')
# ResNet specific arguments:
parser.add_argument('--num_layers', type=int, nargs='+',
					help='The depth of each of the layers.')
# Checkpoint parameters:
parser.add_argument('--resume', type=str,
					help='Indicates if we are loading from a checkpoint or not.')
# Training specific arguments:
parser.add_argument('--update_rule', type=str,
					help='The update rule to use.')
parser.add_argument('--sgd_params', nargs='+', type=float,
					help='The tuple of SGD params: (lr).')
parser.add_argument('--momentum_params', nargs='+', type=float,
					help='The tuple of Momentum params: (lr, mu).')
parser.add_argument('--adadelta_params', nargs='+', type=float,
					help='The tuple of Adadelta params: (lr, rho, eps).')
parser.add_argument('--adam_params', nargs='+', type=float,
					help='The tuple of Adam params: (lr, beta1, beta2, eps).')
parser.add_argument('--l2_reg', default=0.0, type=float,
					help='The L2 decay parameter, default is set to 0.0.')
parser.add_argument('--num_epochs', type=int,
					help='Number of training epochs.')
parser.add_argument('--train_batchsize', default=64, type=int,
					help='Size of the training batch, default is 64.')
parser.add_argument('--epoch_decay', nargs='+', type=int,
					help='Which epochs to decay the LR.')
parser.add_argument('--epoch_decay_factor', default=0.1, type=float,
					help='The LR decay factor, default is set to 0.1.')
parser.add_argument('--test_batchsize', default=100, type=int,
					help='Size of the test batch, default is 100.')
# Save arguments:
parser.add_argument('--exp_dir', type=str,
					help='The root experiment directory.')
args=parser.parse_args()
print(args)

# Set the random seed:
torch.manual_seed(args.random_seed)
np.random.seed(seed=args.random_seed)

# String to Bool:
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

# Count parameters function:
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Load the data:
print('Loading the data...')

if args.dataset_type == 'cifar10':

	C,H,W = 3,32,32

	# CIFAR-10 training set image transforms:
	transform_train = transforms.Compose([
	transforms.RandomCrop(H, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	# Fetch the CIFAR-10 data:
	trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
	train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batchsize, shuffle=True, num_workers=1)

	testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
	test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batchsize, shuffle=False, num_workers=1)

	val_loader = test_loader

elif args.dataset_type == 'cifar100':

	# CIFAR-100 training set image transforms:
	transform_train = transforms.Compose([
	transforms.RandomCrop(H, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	# Fetch the CIFAR-100 data:
	trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
	train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batchsize, shuffle=True, num_workers=1)

	testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
	test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batchsize, shuffle=False, num_workers=1)

	val_loader = test_loader

# Add the dataset to the model dir:
model_dir = args.exp_dir + '%s/' % (args.dataset_type)

# Initialize a network:
print("Initializing a %s..." % (args.network_type))
if args.network_type == 'resnet':
	model = ResNet(block=Bottleneck, layers=args.num_layers)
	print("Initialized a ResNet with layers [%d, %d, %d] and param count %d" % (args.num_layers[0], args.num_layers[1], args.num_layers[2], count_parameters(model)))

	# Initialize a model directory:
	model_dir = model_dir +'resnet_layers_%d_%d_%d_random_seed_%d/' % (args.num_layers[0], args.num_layers[1], args.num_layers[2], args.random_seed)
else:
    print('Invalid model, valid choices are: ["resnet", "preact_resnet"]')
    exit()

# Define a loss criterion:
criterion = nn.CrossEntropyLoss()

if not str2bool(args.resume):
	train_log = OrderedDict()

	# Loss and accuracy arrays:
	train_log['train_loss_epoch_arr'] = np.zeros(args.num_epochs,)
	train_log['val_loss_epoch_arr'] = np.zeros(args.num_epochs,)
	train_log['val_acc_epoch_arr'] = np.zeros(args.num_epochs,)
	train_log['test_loss'] = 0.0
	train_log['test_acc'] = 0.0

	start_epoch = 0

# Define the number of train, val batches:
num_train_batches = 50000 // args.train_batchsize
num_val_batches = 10000 // args.test_batchsize
num_test_batches = num_val_batches


# Define an optimizer:
print("Defining an %s optimizer..." % (args.update_rule))
if args.update_rule == 'sgd':
	lr = args.sgd_params[0]
	model_dir = model_dir+'%s_lr_%.6f_batchsize_%d_l2_reg_%.6f/' % (args.update_rule, lr, args.train_batchsize, args.l2_reg)
	
	# Collect all the trainable parameters:
	params = list(model.parameters())
	optimizer = optim.SGD(params, lr=lr, momentum=0.0, weight_decay=args.l2_reg)
	if args.epoch_decay is not None:
		lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.epoch_decay, gamma=args.epoch_decay_factor)

	if str2bool(args.resume) and os.path.isfile(model_dir+'checkpoint.pth.tar'):
		# Load the checkpoint
		print('Loading checkpoint from path %s...' % (model_dir+'checkpoint.pth.tar'))
		checkpoint = torch.load(model_dir+'checkpoint.pth.tar')
		start_epoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		train_log = checkpoint['train_log']

elif args.update_rule == 'momentum':
	lr, mu = args.momentum_params
	model_dir = model_dir+'%s_lr_%.6f_mu_%.6f_batchsize_%d_l2_reg_%.6f/' % ( args.update_rule, lr, mu, args.train_batchsize, args.l2_reg)

	# Collect all the trainable parameters:
	params = list(model.parameters())

	optimizer = optim.SGD(params, lr=lr, momentum=mu, weight_decay=args.l2_reg)
	if args.epoch_decay is not None:
		lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.epoch_decay, gamma=args.epoch_decay_factor)

	if str2bool(args.resume) and os.path.isfile(model_dir+'checkpoint.pth.tar'):
		# Load the checkpoint
		print('Loading checkpoint from path %s...' % (model_dir+'checkpoint.pth.tar'))
		checkpoint = torch.load(model_dir+'checkpoint.pth.tar')
		start_epoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		train_log = checkpoint['train_log']

elif args.update_rule == 'adadelta':
	lr, rho, eps = args.adadelta_params
	model_dir = model_dir+'%s_lr_%.6f_rho_%.6f_batchsize_%d_l2_reg_%.6f/' % (args.update_rule, lr, rho, args.train_batchsize, args.l2_reg)

	# Collect all the trainable parameters:
	params = list(model.parameters())

	optimizer = optim.Adadelta(params, lr=lr, rho=rho, eps=eps, weight_decay=args.l2_reg)

	if str2bool(args.resume) and os.path.isfile(model_dir+'checkpoint.pth.tar'):
		# Load the checkpoint
		print('Loading checkpoint from path %s...' % (model_dir+'checkpoint.pth.tar'))
		checkpoint = torch.load(model_dir+'checkpoint.pth.tar')
		start_epoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		train_log = checkpoint['train_log']

elif args.update_rule == 'adam':
	lr, beta1, beta2, eps = args.adam_params
	model_dir = model_dir+'%s_lr_%.6f_beta1_%.6f_beta2_%.6f_batchsize_%d_l2_reg_%.6f/' % (args.update_rule, lr, beta1, beta2, args.train_batchsize, l2_reg)

	# Collect all the trainable parameters:
	params = list(model.parameters())
	optimizer = optim.Adam(params, lr=lr, betas=(beta1,beta2), eps=eps, weight_decay=l2_reg)

	if str2bool(args.resume) and os.path.isfile(model_dir+'checkpoint.pth.tar'):
		# Load the checkpoint
		print('Loading checkpoint from path %s...' % (model_dir+'checkpoint.pth.tar'))
		checkpoint = torch.load(model_dir+'checkpoint.pth.tar')
		start_epoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		train_log = checkpoint['train_log']

# Create the experimental directory:
if not os.path.exists(model_dir):
	os.makedirs(model_dir)

# Check if this experiment has been run already:
if os.path.isfile(model_dir+'final_weights.pth.tar'):
	# Kill the job, it's already been run before:
	print("This job has already been run, killing it..")
	exit()

# CUDA:
use_cuda = torch.cuda.is_available()
if use_cuda:
	print('Using CUDA...')
	device = torch.device('cuda')
	criterion = criterion.cuda()
	model.to(device)
	cudnn.benchmark = True

best_val_acc = 0.0

# Save the initial configuration:
if str2bool(args.resume) is False:
	torch.save({'epoch'			:	0,
				'state_dict'	:	model.state_dict(),
				'optimizer'		:	optimizer.state_dict(),
				'train_log'		:	train_log,
				'exp_config'	:	args
				 },
				os.path.join(model_dir,'initial_config.pth.tar')
				)

# Train it!
for epoch in range(start_epoch, args.num_epochs):

	# Enter training mode:
	model.train()
	running_loss = 0.0
	start_time = time.time()

	# Decay the LR:
	if args.epoch_decay is not None:
		lr_scheduler.step()

	for inputs, labels in train_loader:
		inputs, labels = inputs.to(device), labels.to(device)
		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
	running_loss /= float(num_train_batches)
	train_log['train_loss_epoch_arr'][epoch] = running_loss

	# Evaluate on the validation set:
	model.eval()
	val_loss, val_acc = 0.0, 0.0
	for inputs, labels in val_loader:
		inputs, labels = inputs.to(device), labels.to(device)
		outputs = model(inputs)
		loss = criterion(outputs, labels)
		val_loss += loss.item()
		_, val_predictions = outputs.max(1)
		val_acc += val_predictions.eq(labels).sum().item()
	val_loss /= float(num_val_batches)
	val_acc /= float(10000)
	train_log['val_loss_epoch_arr'][epoch] = val_loss
	train_log['val_acc_epoch_arr'][epoch] = val_acc

	if val_acc > best_val_acc:
		# Save the best validation weights:
		torch.save({'epoch'			:	epoch+1,
					'state_dict'	:	model.state_dict(),
					'optimizer'		:	optimizer.state_dict(),
					'train_log'		:	train_log,
					'exp_config'	:	args
					 },
					os.path.join(model_dir,'best_checkpoint.pth.tar')
					)
		best_val_acc = val_acc


	# Checkpoint:
	torch.save({'epoch'			:	epoch+1,
				'state_dict'	:	model.state_dict(),
				'optimizer'		:	optimizer.state_dict(),
				'train_log'		:	train_log,
				'exp_config'	:	args
				 },
				os.path.join(model_dir,'checkpoint.pth.tar')
					)
	# Print epoch status update:
	print("Completed epoch %d in %.4f seconds with train loss %.6f, val loss %.6f and val acc %.2f%%..." % (epoch+1, time.time()-start_time, running_loss, val_loss, 100*val_acc))

# Evaluate on the test set:
model.eval()
test_loss, test_acc = 0.0, 0.0
for inputs, labels in test_loader:
	inputs, labels = inputs.to(device), labels.to(device)
	outputs = model(inputs)
	loss = criterion(outputs, labels)
	test_loss += loss.item()
	_, test_predictions = outputs.max(1)
	test_acc += test_predictions.eq(labels).sum().item()
test_loss /= float(num_val_batches)
test_acc /= float(10000)
train_log['test_loss'] = test_loss/float(num_test_batches)
train_log['test_acc'] = test_acc

# Checkpoint:
torch.save({'epoch'			:	epoch+1,
			'state_dict'	:	model.state_dict(),
			'optimizer'		:	optimizer.state_dict(),
			'train_log'		:	train_log,
			'exp_config'	:	args
			 },
			os.path.join(model_dir,'final_weights.pth.tar')
			)
print("Final test loss %.6f, test acc %.2f%%..." % (train_log['test_loss'], 100*train_log['test_acc']))
