import torch
from torch import nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from sentence_transformers import SentenceTransformer

# Setting up dataloader for the DAQUAR dataset
class DAQUARDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_filename = self.data.iloc[idx, 2]  # third column contains the image filenames
        if not image_filename.endswith('.png'):
            image_filename += '.png'  # Append the .png extension if missing
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        question = self.data.iloc[idx, 0]  # Get the question
        answer = self.data.iloc[idx, 1]  # Get the answer (label)

        return image, question, answer  # Return the image, question, and answer

# Define the transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dir = 'problem_2'
train_csv = 'new_data_train.csv'
val_csv = 'new_data_val.csv'
test_csv = 'new_data_test.csv'
image_dir = 'images'

# Create dataset instances
train_dataset = DAQUARDataset(csv_file=train_csv, image_dir=image_dir, transform=transform)
val_dataset = DAQUARDataset(csv_file=val_csv, image_dir=image_dir, transform=transform)
test_dataset = DAQUARDataset(csv_file=test_csv, image_dir=image_dir, transform=transform)

# Create dataloader instances
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load the ResNet-50 model
resnet = models.resnet50(pretrained=True)
# Remove the classification head
resnet = nn.Sequential(*list(resnet.children())[:-1])

def get_image_embeddings(images):
    with torch.no_grad():
        embeddings = resnet(images)
    return embeddings.view(embeddings.size(0), -1)

# Load BERT model

sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
if torch.cuda.is_available():
    sentence_model = sentence_model.cuda()

def get_sentence_embeddings(sentences):
    embeddings = sentence_model.encode(sentences)
    return torch.tensor(embeddings)

# Dimension reduction to have the same size for both embeddings
class ReduceDimension(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ReduceDimension, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.linear(x))

image_dim_reduction = ReduceDimension(input_dim=2048, output_dim=512)
sentence_dim_reduction = ReduceDimension(input_dim=384, output_dim=512)

# Concatenate the embeddings
def concat_embeddings(image_embeddings, sentence_embeddings):
    return torch.cat((image_embeddings, sentence_embeddings), dim=1)

# Define the classifier
class Classifier(nn.Module):
    def __init__(self, input_dim, class_num):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(input_dim, class_num)
    
    def forward(self, x):
        return self.linear(x)

# Define the model
def train_model(dataloader, classifier, criterion, optimizer):
    if torch.cuda.is_available():
        classifier.cuda()
    correct = 0
    total = 0
    for images, questions, labels in dataloader:
        images, labels = images.cuda(), labels.cuda()
        image_embeddings = get_image_embeddings(images)
        sentence_embeddings = get_sentence_embeddings(questions).cuda()
        image_embeddings = image_dim_reduction(image_embeddings)
        sentence_embeddings = sentence_dim_reduction(sentence_embeddings)
        embeddings = concat_embeddings(image_embeddings, sentence_embeddings)

        outputs = classifier(embeddings)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print("Accuracy on train set", (correct / total) * 100)

    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, questions, labels in val_loader:
            images, labels = images.cuda(), labels.cuda()
            image_embeddings = get_image_embeddings(images)
            sentence_embeddings = get_sentence_embeddings(questions).cuda()
            image_embeddings = image_dim_reduction(image_embeddings)
            sentence_embeddings = sentence_dim_reduction(sentence_embeddings)
            embeddings = concat_embeddings(image_embeddings, sentence_embeddings)

            outputs = classifier(embeddings)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_accuracy = (correct / total) * 100        
       

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(classifier.state_dict(), dir + 'best_model.pth')
            print(f"Best model saved with validation accuracy: {best_val_accuracy:.2f}%")



# Evaluate the model
test_predictions = []
test_labels = []
def evaluate_model(dataloader, classifier):
    if torch.cuda.is_available():
        classifier.cuda()
    
    classifier.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, questions, labels in dataloader:
            images, labels = images.cuda(), labels.cuda()
            image_embeddings = get_image_embeddings(images)
            sentence_embeddings = get_sentence_embeddings(questions).cuda()
            image_embeddings = image_dim_reduction(image_embeddings)
            sentence_embeddings = sentence_dim_reduction(sentence_embeddings)
            embeddings = concat_embeddings(image_embeddings, sentence_embeddings)

            outputs = classifier(embeddings)
            _, predicted = torch.max(outputs.data, 1)
            test_predictions.extend(predicted.cpu().numpy())  # Store predictions
            test_labels.extend(labels.cpu().numpy())  # Store true labels
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print("Accuracy on test set", (correct / total) * 100)
    
    results_df = pd.DataFrame({
        'True Label': test_labels,
        'Predicted Label': test_predictions
    })

    results_df.to_csv('test_predictions.csv', index=False)

if __name__ == '__main__':
    classifier = Classifier(input_dim=1024, class_num=30)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    train_model(train_loader, val_loader, classifier, criterion, optimizer)
    evaluate_model(val_loader, classifier)
    evaluate_model(test_loader, classifier)