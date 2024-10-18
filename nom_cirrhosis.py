import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from pathlib import Path

df = pd.read_csv("./cirrhosis.csv")

y_df = df["Status"]
df.drop("Status", axis=1, inplace=True)

dummies_sex = pd.get_dummies(df["Sex"], dtype=float, prefix="Sex")
dummies_dru = pd.get_dummies(df["Drug"], dtype=float, prefix="Drug")
dummies_asc = pd.get_dummies(df["Ascites"], dtype=float, prefix="Ascites")
dummies_hep = pd.get_dummies(df["Hepatomegaly"], dtype=float, prefix="Hepatomegaly")
dummies_spi = pd.get_dummies(df["Spiders"], dtype=float, prefix="Spiders")
dummies_ede = pd.get_dummies(df["Edema"], dtype=float, prefix="Edema")
dummies_sta = pd.get_dummies(df["Stage"], dtype=float, prefix="Edema")

df = pd.concat([df, dummies_sex, dummies_dru, dummies_asc, dummies_hep, dummies_spi, dummies_ede, dummies_sta], axis=1)
df.drop(["Sex", "Drug", "Ascites", "Hepatomegaly", "Spiders", "Edema", "Stage"], axis=1, inplace=True)

for i in df.columns:
    df[i] = pd.to_numeric(df[i], errors="coerce")
    med = df[i].median()
    df[i] = df[i].fillna(med)
    df[i] = df[i] / df[i].max()

device = "cuda" if torch.cuda.is_available() else "cpu"
device

y_df

X = torch.from_numpy(np.array(df)).float().to(device)
y = torch.from_numpy(np.array(y_df)).float().to(device)

class NNModel(nn.Module):
    def __init__(self, hid1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(X.shape[1], hid1),
            nn.LeakyReLU(0.3),
            nn.Linear(hid1, 3),
            nn.LeakyReLU(0.3),
        )
        
    def forward(self, X):
        return self.layers(X)
    
def modelTest(hid1, lr, momentum, epochs):
    batch_size = 32
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model1 = NNModel(hid1).to(device)
    model2 = NNModel(hid1).to(device)
    model3 = NNModel(hid1).to(device)
    model4 = NNModel(hid1).to(device)
    model5 = NNModel(hid1).to(device)
    
    lossf = nn.CrossEntropyLoss()
    optim1 = torch.optim.SGD(params=model1.parameters(), lr=lr, momentum=momentum)
    optim2 = torch.optim.SGD(params=model2.parameters(), lr=lr, momentum=momentum)
    optim3 = torch.optim.SGD(params=model3.parameters(), lr=lr, momentum=momentum)
    optim4 = torch.optim.SGD(params=model4.parameters(), lr=lr, momentum=momentum)
    optim5 = torch.optim.SGD(params=model5.parameters(), lr=lr, momentum=momentum)
    
    softmax = nn.Softmax(dim=1).to(device)
    def pred(y_logits):
        return torch.argmax(softmax(y_logits), dim=1).float()
    
    def accuracy(y_pred, y_true=y_test):
        return torch.sum(torch.eq(y_pred, y_true) / len(y_true))
    
    for i in range(epochs):
        perm = torch.randperm(X_train.size()[0]).to(device)

        for j in range(0, X_train.size()[0], batch_size):
            indices = perm[j:j+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]

            model1.train()
            y_logits1 = model1(batch_x)
            loss1 = lossf(y_logits1, batch_y.long())
            optim1.zero_grad()
            loss1.backward()
            optim1.step()
            model1.eval()

            model2.train()
            y_logits2 = model2(batch_x)
            loss2 = lossf(y_logits2, batch_y.long())
            optim2.zero_grad()
            loss2.backward()
            optim2.step()
            model2.eval()

            model3.train()
            y_logits3 = model3(batch_x)
            loss3 = lossf(y_logits3, batch_y.long())
            optim3.zero_grad()
            loss3.backward()
            optim3.step()
            model3.eval()

            model4.train()
            y_logits4 = model4(batch_x)
            loss4 = lossf(y_logits4, batch_y.long())
            optim4.zero_grad()
            loss4.backward()
            optim4.step()
            model4.eval()

            model5.train()
            y_logits5 = model5(batch_x)
            loss5 = lossf(y_logits5, batch_y.long())
            optim5.zero_grad()
            loss5.backward()
            optim5.step()
            model5.eval()
            
#             # Check models are training
#             if i % 25 == 0 and j == 0:
#                 print(i, loss1.data.item(), accuracy(pred(y_logits1), batch_y).item(), accuracy(pred(model1(X_test))).item())
#                 print(i, loss2.data.item(), accuracy(pred(y_logits2), batch_y).item(), accuracy(pred(model2(X_test))).item())
#                 print(i, loss3.data.item(), accuracy(pred(y_logits3), batch_y).item(), accuracy(pred(model3(X_test))).item())
#                 print(i, loss4.data.item(), accuracy(pred(y_logits4), batch_y).item(), accuracy(pred(model4(X_test))).item())
#                 print(i, loss5.data.item(), accuracy(pred(y_logits5), batch_y).item(), accuracy(pred(model5(X_test))).item())
            
    def combine5_pred(X=X_test):
        with torch.inference_mode():
            y_pred1 = pred(model1(X))
            y_pred2 = pred(model2(X))
            y_pred3 = pred(model3(X))
            y_pred4 = pred(model4(X))
            y_pred5 = pred(model5(X))

        combined = torch.mode(torch.concat([y_pred1.unsqueeze(dim=-1), y_pred2.unsqueeze(dim=-1), y_pred3.unsqueeze(dim=-1), y_pred4.unsqueeze(dim=-1), y_pred5.unsqueeze(dim=-1)], dim=1).to("cpu"), 1)
        return combined.values.to(device)
    
    X_p = df.copy()
    X_d = df.copy()

    X_p['Drug_1'] = 0
    X_p['Drug_2'] = 1
    X_d['Drug_1'] = 1
    X_d['Drug_2'] = 0
    
    y_harmful = torch.full([len(X_p)], 2).to(device)
    X_p = torch.from_numpy(np.array(X_p)).float().to(device)
    placebo_risk = accuracy(combine5_pred(X_p), y_harmful)
    X_d = torch.from_numpy(np.array(X_d)).float().to(device)
    drug_risk = accuracy(combine5_pred(X_d), y_harmful)
    
    return [accuracy(combine5_pred()).item(), placebo_risk.item(), drug_risk.item()]

out = []

hid1 = 1024
lr = 0.01
momentum = 0.1
epochs = 200

for i in range(200):
    modelout = modelTest(hid1=hid1, lr=lr, momentum=momentum, epochs=epochs)
    out.append(modelout)
    print(i, modelout)

output_df = pd.DataFrame(np.array(out))
filepath = Path(f"./nom_{hid1}_{lr}_{momentum}_{epochs}.csv")
filepath.parent.mkdir(parents=True, exist_ok=True)
output_df.to_csv(filepath, header=False, index=False)

import winsound
winsound.Beep(440, 1000)