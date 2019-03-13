import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils import model_zoo

transform = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=40, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=40, shuffle=False, num_workers=2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#self define the Conv2d Layer
class BasicConv2d(nn.Module):

	def __init__(self, in_channels, out_channels, **kwargs):
		super(BasicConv2d, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
		self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		return F.relu(x, inplace=True)

#define Inception for creating inception layers
class Inception(nn.Module): 

	def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
		super(Inception, self).__init__()

		self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

		self.branch2 = nn.Sequential(
			BasicConv2d(in_channels, ch3x3red, kernel_size=1),
			BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
		)

		self.branch3 = nn.Sequential(
			BasicConv2d(in_channels, ch5x5red, kernel_size=1),
			BasicConv2d(ch5x5red, ch5x5, kernel_size=3, padding=1)
		)

		self.branch4 = nn.Sequential(
			nn.MaxPool2d(kernel_size=3, stride=1, padding=1,ceil_mode=True),
			BasicConv2d(in_channels, pool_proj, kernel_size=1)
		)

	def forward(self, x):
		branch1 = self.branch1(x)
		branch2 = self.branch2(x)
		branch3 = self.branch3(x)
		branch4 = self.branch4(x)

		outputs = [branch1, branch2, branch3, branch4]
		return torch.cat(outputs, 1)

convs = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.BatchNorm2d(3),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.BatchNorm2d(64),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.BatchNorm2d(64),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.BatchNorm2d(128),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.BatchNorm2d(128),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.Dropout(0.3),

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.BatchNorm2d(256),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.BatchNorm2d(256),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.BatchNorm2d(256),
    nn.ReLU(),  # relu3-3 
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.Dropout(0.4),
  	# nn.ReflectionPad2d((1, 1, 1, 1)),
   #  nn.Conv2d(256, 512, (3, 3)),
   #  nn.BatchNorm2d(512),
   #  nn.ReLU(),  # relu4-1
   #  nn.ReflectionPad2d((1, 1, 1, 1)),
   #  nn.Conv2d(512, 512, (3, 3)),
   #  nn.BatchNorm2d(512),
   #  nn.ReLU(),  # relu4-2
   #  nn.Dropout(0.5),
   #  nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
)


class NetModel(nn.Module):
	def __init__(self):
		super(NetModel, self).__init__()

		self.convs = convs
		# self.inception = models.inception_v3(pretrained=False)
		self.inception3a = Inception(256, 128, 128, 192, 32, 96, 64)
		self.inception4a = Inception(480, 192, 96, 208, 16, 48 ,64)
		self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
		# self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
		self.pool = nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)
		self.fc1 = nn.Linear(512 * 2 * 2, 4096)
		self.fc2 = nn.Linear(4096, 512)
		self.fc3 = nn.Linear(512, 10)

	def forward(self, x):

		x = self.convs(x)
		# 256 x 4 x 4
		x = self.inception3a(x)
		# 480 x 4 x 4
		x = self.inception4a(x)
		# 512 x 4 x 4 
		x = self.inception4b(x)
		# 512 x 4 x 4
		# x = self.inception4c(x)
		# 512 x 4 x 4
		x = self.pool(x)

		x = x.view(-1, 512 * 2 * 2)
		x = self.fc1(x)
		x = self.fc2(x)
		x = self.fc3(x)
		return x

net = NetModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)

def adjust_learning_rate(optimizer, iteration_count,lr=0.001,lr_decay=5e-6):
    lr = lr / (1.0 + lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class Trainer:
	def __init__(self):
		self.loss_t = []
		self.steps_t = []
		self.acc_t = []
		self.global_step = 0
		self.epoch = 0

	def train(self, max_epoch):
		for self.epoch in range(max_epoch):
			running_loss = 0.0
			for i, data in enumerate(trainloader, 0):
				self.global_step += 1
				adjust_learning_rate(optimizer, self.global_step)
				net.cuda()
				inputs, labels = data
				inputs, labels = inputs.to(device), labels.to(device)
				optimizer.zero_grad()

				outputs = net(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()

				running_loss += loss.item()

				if i%200 == 199:
					self.loss_t.append(running_loss)
					self.steps_t.append(self.global_step)
					correct = 0
					total = 0
					with torch.no_grad():
						for data in testloader:
							images, labels = data
							images, labels = images.to(device), labels.to(device)
							outputs = net(images)
							_, predicted = torch.max(outputs.data, 1)
							total += labels.size(0)
							correct += (predicted == labels).sum().item()
							accu = 100 * correct / total
					print('[%d, %5d] loss: %.3f accuracy: %d %%' % (self.epoch + 1, self.global_step, running_loss / 200, (accu)))
					print('lr',optimizer.param_groups[0]['lr'])
					self.acc_t.append(accu)
					running_loss = 0.0

		print('Finished Training')

trainer = Trainer()
trainer.train(100)
torch.save(net.state_dict(), './models/inception10.pth')

plt.plot(trainer.steps_t, trainer.loss_t, 'r--')
plt.xlabel('Number of Iterations')
plt.ylabel("Loss")
plt.savefig('./diagrams/check_i10_1.png')

plt.clf()
plt.plot(trainer.steps_t, trainer.acc_t, 'r--')
plt.xlabel('Number of Iterations')
plt.ylabel("Accuracy")
plt.savefig('./diagrams/check_i10_2.png')
