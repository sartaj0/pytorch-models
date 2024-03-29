import torch

def load_split_train_test(datadir, batch_size, valid_size = .2):
	train_transforms = transforms.Compose([transforms.Resize((224, 244)), transforms.ToTensor(),])
	test_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),])

	train_data = datasets.ImageFolder(datadir, transform=train_transforms)
	test_data = datasets.ImageFolder(datadir, transform=test_transforms)

	num_train = len(train_data)
	indices = list(range(num_train))
	split = int(np.floor(valid_size * num_train))
	np.random.shuffle(indices)

	train_idx, test_idx = indices[split:], indices[:split]
	train_sampler = SubsetRandomSampler(train_idx)
	test_sampler = SubsetRandomSampler(test_idx)

	trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
	testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

	return trainloader, testloader