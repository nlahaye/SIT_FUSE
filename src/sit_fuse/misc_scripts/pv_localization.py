import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from osgeo import osr, gdal
from skimage.util import view_as_windows
from tqdm import tqdm


fnames = [
"/data/nlahaye/remoteSensing/PV_Mapping/large_urban_gambia_clipped_4_s1_s2_ls8.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/large_urban_gambia_clipped_5_s1_s2_ls8.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/urban_small_road_clipped_s1_s2_ls8.tif",
]

lab_fnames = [
"/data/nlahaye/remoteSensing/PV_Mapping/pv_polygons_10.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/pv_polygons_11.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/pv_polygons_16.tif",
]

# Add a new classification head
class ClassificationHead(torch.nn.Module):
    def __init__(self, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc = torch.nn.Linear(2048, num_classes)

    def forward(self, x):
        x = x.view(-1, 2048)
        x = self.fc(x)
        return x


class CustomImageDataset(Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.img_labels = labels
        self.images = data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx])
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label




# Load the pre-trained ResNet model
model = torchvision.models.resnet152(pretrained=True)


data = None
targets = None

for i in range(len(fnames)):
    in_dat = gdal.Open(fnames[i]).ReadAsArray()
    in_dat = in_dat[8:11,:,:]
    print(in_dat.shape)

    in_dat = np.squeeze(view_as_windows(in_dat, (3,224,224), step = 40))
    in_dat = in_dat.reshape((in_dat.shape[0]*in_dat.shape[1], in_dat.shape[2], in_dat.shape[3], in_dat.shape[4]))


    lab_dat = gdal.Open(lab_fnames[i]).ReadAsArray()
    print(lab_dat.shape)
    lab_dat = np.squeeze(view_as_windows(lab_dat, (224, 224), step = 40))
    lab_dat = lab_dat.reshape((lab_dat.shape[0]*lab_dat.shape[1], 1, lab_dat.shape[2], lab_dat.shape[3]))


    delete_inds = []
    for tle in range(in_dat.shape[0]):
        if np.max(in_dat[tle]) <= -9999:
                delete_inds.append(tle)
        elif abs(np.std(in_dat[tle])) < 0.0000001:
                delete_inds.append(tle)

    if len(delete_inds) > 0:
        np.delete(in_dat, delete_inds, axis=0)
        np.delete(lab_dat, delete_inds, axis=0)



    if data is None:
        data = in_dat
        targets = lab_dat
    else:
        data = np.append(data, in_dat, axis=0)
        targets = np.append(targets, lab_dat, axis=0)

final_labels = []

print(targets.shape, data.shape)

for i in range(targets.shape[0]):
    if len(np.where(targets[i] > 0)[0]) > 0:
        final_labels.append(1)
    else:
        final_labels.append(0)




preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    #transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print(data.shape, len(final_labels))
dataset = CustomImageDataset(data, final_labels, preprocess)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
 

# Freeze the weights of the pre-trained model
for param in model.parameters():
    param.requires_grad = False

# Create a new classification head
classification_head = ClassificationHead(num_classes=2)

# Add the new classification head to the pre-trained model
model.fc = classification_head

model = model.cuda()

# Train the model
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for x, y in tqdm(train_loader):
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        x = x.detach().cpu()
        y = y.detach().cpu()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

torch.save(model.state_dict(), "PV_LOCALIZER.pt")






