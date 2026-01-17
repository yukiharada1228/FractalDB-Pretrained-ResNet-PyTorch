# -*- coding: utf-8 -*-
"""
Created on Sat Aug 05 23:55:12 2018
@author: Kazushige Okayasu, Hirokatsu Kataoka
"""
import datetime
import time
import random
import os

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms

from args import conf
from model_select import model_select
from train_val import train, val
from loadDB import DBLoader

args = conf()

def worker_init_fn(worker_id):
	random.seed(worker_id+args.seed)

if __name__== "__main__":
	print (args)

	# Device selection (CUDA, MPS, or CPU)
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	use_mps = not args.no_cuda and not use_cuda and torch.backends.mps.is_available()

	if use_cuda:
		device = torch.device("cuda")
	elif use_mps:
		device = torch.device("mps")
	else:
		device = torch.device("cpu")

	print(f"Using device: {device}")

	#to deterministic
	if use_cuda:
		cudnn.deterministic = True
	random.seed(args.seed)
	torch.manual_seed(args.seed)
	
	# Training settings
	train_transform = transforms.Compose([
						transforms.Resize(args.img_size, interpolation=2),
						transforms.RandomCrop(args.crop_size),
						transforms.ToTensor(),
						transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
	train_fractal = DBLoader(args.path2db,'train',train_transform)
	train_loader = torch.utils.data.DataLoader(dataset=train_fractal, batch_size=args.batch_size,
											shuffle=True, num_workers=args.num_workers,
											pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)
	
	# Validation settings
	test_transform = transforms.Compose([
						transforms.Resize(args.img_size, interpolation=2),
						transforms.CenterCrop(args.crop_size),
						transforms.ToTensor(),
						transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
	test_fractal = DBLoader(args.path2db,'val',test_transform)
	test_loader = torch.utils.data.DataLoader(dataset=test_fractal, batch_size=args.test_batch_size,
											shuffle=False, num_workers=args.num_workers*2,
											pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)

	# Model & optimizer
	model = model_select(args).to(device)
	params = list(model.parameters())
	optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
	'''
	if args.multigpu:
		#model = nn.DataParallel(model)
		optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
	else:
		optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
	'''
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60], gamma=0.1)
	starttime = time.time()
	iteration = 0
	
	# optionally resume from a checkpoint
	if args.resume:
		assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
		print("=> loading checkpoint '{}'".format(args.resume))
		checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
		args.start_epoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		scheduler.load_state_dict(checkpoint['scheduler'])
		print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
	if args.multigpu:
		model = nn.DataParallel(model)
	iteration = (args.start_epoch-1)*len(train_loader)

	# TensorBoard writer
	log_dir = os.path.join('runs', f'pt_{args.dataset}_ft_{args.ft_dataset}_{args.usenet}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
	writer = SummaryWriter(log_dir)
	print(f"TensorBoard logging to: {log_dir}")

	# Training & Validation
	for epoch in range(1, args.epochs + 1):
		print("\nepoch {}".format(epoch))
		train_loss, train_acc = train(args, model, device, train_loader, optimizer, epoch, iteration)

		# Log training metrics to TensorBoard
		writer.add_scalar('Loss/train', train_loss, epoch)
		writer.add_scalar('Accuracy/train', train_acc, epoch)
		writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

		scheduler.step()
		iteration += len(train_loader)
		validation_loss, validation_accuracy = val(args, model, device, test_loader, iteration)

		# Log validation metrics to TensorBoard
		writer.add_scalar('Loss/val', validation_loss, epoch)
		writer.add_scalar('Accuracy/val', validation_accuracy, epoch)
		if epoch % args.save_interval == 0:
			saved_weight = os.path.join(args.path2weight, "pt_"+args.dataset+"_ft_"+args.ft_dataset+"_"+args.usenet+"_epoch"+ str(epoch) +".pth")
			if args.multigpu:
				torch.save(model.module.cpu().state_dict(), saved_weight)
				model_state = model.module.cpu().state_dict()
			else:
				torch.save(model.cpu().state_dict(), saved_weight)
				model_state = model.cpu().state_dict()

			# Save checkpoint
			checkpoint = "{}/{}_{}_checkpoint.pth.tar".format(args.path2weight, args.dataset, args.usenet)
			torch.save({'epoch': epoch + 1,
						'state_dict': model_state,
						'optimizer' : optimizer.state_dict(),
						'scheduler' : scheduler.state_dict(),}, checkpoint)

			model = model.to(device)

	# Close TensorBoard writer
	writer.close()

	# Processing time
	endtime = time.time()
	interval = endtime - starttime
	print ("elapsed time = %dh %dm %ds" % (int(interval/3600),int((interval%3600)/60),int((interval%3600)%60)))
